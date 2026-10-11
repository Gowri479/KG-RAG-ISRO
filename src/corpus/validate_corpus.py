"""Validate registry, artifacts, ontology, graph types, and fact provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus"
REPORT_PATH = CORPUS / "validation_report.json"
MARKDOWN_PATH = ROOT / "reports" / "phase1_validation_report.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_corpus() -> dict[str, Any]:
    registry = json.loads((CORPUS / "document_registry.json").read_text(encoding="utf-8"))
    ontology = json.loads((CORPUS / "ontology.json").read_text(encoding="utf-8"))
    relation_schema = json.loads((CORPUS / "relations.json").read_text(encoding="utf-8"))
    chunks = read_jsonl(CORPUS / "chunks.jsonl")
    entities = read_jsonl(CORPUS / "entities.jsonl")
    triples = read_jsonl(CORPUS / "triples.jsonl")

    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    documents = registry["documents"]
    documents_by_id = {document["document_id"]: document for document in documents}
    registry_ids_unique = len(documents_by_id) == len(documents)
    record("unique_registry_document_ids", registry_ids_unique, f"{len(documents)} registered documents")
    registry_required = (
        "document_id", "title", "mission", "document_type", "organization", "source_url",
        "source_domain", "authority_tier", "retrieval_date", "file_type", "local_path",
        "checksum_sha256", "language", "status",
    )
    registry_missing = [
        document.get("document_id", "<unknown>")
        for document in documents
        if any(not document.get(field) for field in registry_required)
    ]
    record("registry_required_metadata", not registry_missing, f"Missing required metadata: {registry_missing}")

    source_urls_valid = all(
        urlparse(document["source_url"]).scheme == "https"
        and urlparse(document["source_url"]).hostname == document["source_domain"]
        for document in documents
    )
    record("official_source_urls_well_formed", source_urls_valid, "Each URL is HTTPS and its host matches source_domain.")
    tier1_only = all(document["authority_tier"] == "TIER_1" for document in documents)
    record("registered_documents_are_tier1", tier1_only, "Only official ISRO/ISSDC primary sources are in this corpus.")

    hashes: dict[str, list[str]] = {}
    missing_files: list[str] = []
    checksum_mismatches: list[str] = []
    for document in documents:
        file_path = ROOT / document["local_path"]
        if not file_path.is_file():
            missing_files.append(document["document_id"])
            continue
        actual_hash = sha256(file_path)
        if actual_hash != document["checksum_sha256"]:
            checksum_mismatches.append(document["document_id"])
        hashes.setdefault(actual_hash, []).append(document["document_id"])
    duplicate_hashes = {digest: ids for digest, ids in hashes.items() if len(ids) > 1}
    record("registered_source_files_exist", not missing_files, f"Missing source files: {missing_files}")
    record("registered_source_hashes_match", not checksum_mismatches, f"Checksum mismatches: {checksum_mismatches}")
    record("no_duplicate_document_hashes", not duplicate_hashes, f"Duplicate hashes: {duplicate_hashes}")
    registered_paths = {str(Path(document["local_path"])) for document in documents}
    raw_root = CORPUS / "raw"
    unregistered_raw_files = [
        str(path.relative_to(ROOT))
        for path in raw_root.rglob("*")
        if path.is_file() and str(path.relative_to(ROOT)) not in registered_paths
    ]
    record("every_raw_file_is_registered", not unregistered_raw_files, f"Unregistered raw files: {unregistered_raw_files}")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    record("unique_chunk_ids", len(chunk_ids) == len(set(chunk_ids)), f"{len(chunk_ids)} chunks")
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    chunk_orphans = [chunk["chunk_id"] for chunk in chunks if chunk["document_id"] not in documents_by_id]
    record("chunks_map_to_registered_documents", not chunk_orphans, f"Unregistered chunk references: {chunk_orphans}")
    bad_chunk_urls = [
        chunk["chunk_id"]
        for chunk in chunks
        if chunk["document_id"] in documents_by_id
        and chunk["source_url"] != documents_by_id[chunk["document_id"]]["source_url"]
    ]
    record("chunk_source_urls_match_registry", not bad_chunk_urls, f"Mismatched chunk URLs: {bad_chunk_urls}")
    chunks_missing_content = [
        chunk["chunk_id"]
        for chunk in chunks
        if not chunk.get("text") or not chunk.get("section") or not chunk.get("authority_tier")
    ]
    record("chunks_have_text_section_and_tier", not chunks_missing_content, f"Incomplete chunks: {chunks_missing_content}")

    valid_types = {entity["name"] for entity in ontology["entity_types"]}
    entity_ids = [entity["entity_id"] for entity in entities]
    record("unique_entity_ids", len(entity_ids) == len(set(entity_ids)), f"{len(entities)} entities")
    entities_missing_type = [entity["entity_id"] for entity in entities if entity.get("type") not in valid_types]
    record("entities_use_ontology_types", not entities_missing_type, f"Invalid entity types: {entities_missing_type}")
    relation_definitions = {relation["name"]: relation for relation in relation_schema["relations"]}
    entity_by_id = {entity["entity_id"]: entity for entity in entities}

    triple_ids = [triple["triple_id"] for triple in triples]
    record("unique_triple_ids", len(triple_ids) == len(set(triple_ids)), f"{len(triples)} triples")
    graph_errors: list[str] = []
    seen_triples: set[tuple[str, str, str]] = set()
    for triple in triples:
        relation = relation_definitions.get(triple["relation"])
        if relation is None:
            graph_errors.append(f"{triple['triple_id']}: uncontrolled relation {triple['relation']}")
            continue
        subject = entity_by_id.get(triple["subject_id"])
        object_ = entity_by_id.get(triple["object_id"])
        if not subject or not object_:
            graph_errors.append(f"{triple['triple_id']}: missing endpoint")
            continue
        if subject["type"] != triple["subject_type"] or object_["type"] != triple["object_type"]:
            graph_errors.append(f"{triple['triple_id']}: endpoint type mismatch")
        if subject["type"] not in relation["subject_types"] or object_["type"] not in relation["object_types"]:
            graph_errors.append(f"{triple['triple_id']}: endpoint type disallowed for {triple['relation']}")
        key = (triple["subject_id"], triple["relation"], triple["object_id"])
        if key in seen_triples:
            graph_errors.append(f"{triple['triple_id']}: duplicate triple {key}")
        seen_triples.add(key)
    record("triples_are_typed_controlled_and_unique", not graph_errors, "; ".join(graph_errors))
    entity_endpoints = {
        endpoint for triple in triples for endpoint in (triple["subject_id"], triple["object_id"])
    }
    orphan_entities = sorted(set(entity_ids) - entity_endpoints)
    record("no_orphan_entities", not orphan_entities, f"Orphan entities: {orphan_entities}")

    provenance_errors: list[str] = []
    entity_provenance_errors: list[str] = []
    for entity in entities:
        entity_evidence = entity.get("evidence", [])
        if not entity_evidence:
            entity_provenance_errors.append(f"{entity['entity_id']}: no evidence")
        for evidence in entity_evidence:
            chunk = chunks_by_id.get(evidence.get("source_chunk_id"))
            if (
                chunk is None
                or evidence.get("document_id") != chunk["document_id"]
                or evidence.get("source_url") != chunk["source_url"]
                or evidence.get("source_text") != chunk["text"]
                or evidence.get("source_section") != chunk["section"]
                or evidence.get("source_page") != chunk.get("page")
                or evidence.get("source_type") != "official_primary"
                or evidence.get("authority_tier") != "TIER_1"
            ):
                entity_provenance_errors.append(f"{entity['entity_id']}: invalid chunk provenance")
                continue
            excerpt = evidence.get("supporting_excerpt", "")
            excerpt_field = "text" if evidence.get("supporting_excerpt_location") == "chunk_text" else "section"
            if not excerpt or excerpt not in chunk[excerpt_field]:
                entity_provenance_errors.append(f"{entity['entity_id']}: excerpt not found in source")
    record("every_entity_has_chunk_provenance", not entity_provenance_errors, "; ".join(entity_provenance_errors))
    for triple in triples:
        evidence_items = triple.get("provenance", [])
        if not evidence_items:
            provenance_errors.append(f"{triple['triple_id']}: no provenance")
            continue
        if triple.get("evidence_type") != "direct":
            provenance_errors.append(f"{triple['triple_id']}: evidence is not direct")
        for evidence in evidence_items:
            chunk = chunks_by_id.get(evidence.get("source_chunk_id"))
            if chunk is None:
                provenance_errors.append(f"{triple['triple_id']}: unknown chunk {evidence.get('source_chunk_id')}")
                continue
            if evidence.get("document_id") != chunk["document_id"]:
                provenance_errors.append(f"{triple['triple_id']}: chunk/document mismatch")
            if evidence.get("source_url") != chunk["source_url"]:
                provenance_errors.append(f"{triple['triple_id']}: evidence URL mismatch")
            if evidence.get("source_text") != chunk["text"]:
                provenance_errors.append(f"{triple['triple_id']}: source text differs from chunk")
            if evidence.get("source_section") != chunk["section"] or evidence.get("source_page") != chunk.get("page"):
                provenance_errors.append(f"{triple['triple_id']}: section/page differs from chunk")
            if evidence.get("source_type") != "official_primary" or evidence.get("authority_tier") != "TIER_1":
                provenance_errors.append(f"{triple['triple_id']}: source authority metadata is invalid")
            excerpt = evidence.get("supporting_excerpt", "")
            excerpt_field = "text" if evidence.get("supporting_excerpt_location") == "chunk_text" else "section"
            if not excerpt or excerpt not in chunk[excerpt_field]:
                provenance_errors.append(f"{triple['triple_id']}: supporting excerpt not present in source {excerpt_field}")
            document = documents_by_id.get(evidence.get("document_id"))
            if document is None or evidence.get("source_url") != document["source_url"]:
                provenance_errors.append(f"{triple['triple_id']}: evidence references unknown or mismatched document")
        primary = evidence_items[0]
        if (
            triple.get("source_document") != primary.get("document_id")
            or triple.get("source_url") != primary.get("source_url")
            or triple.get("source_chunk_id") != primary.get("source_chunk_id")
            or triple.get("source_text") != primary.get("source_text")
        ):
            provenance_errors.append(f"{triple['triple_id']}: primary provenance is inconsistent")
    record("every_triple_has_exact_chunk_provenance", not provenance_errors, "; ".join(provenance_errors))

    provenance_linked = sum(bool(triple.get("provenance")) for triple in triples)
    report = {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "check_count": len(checks),
        "passed_check_count": sum(check["passed"] for check in checks),
        "failed_check_count": sum(not check["passed"] for check in checks),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "entity_count": len(entities),
        "triple_count": len(triples),
        "provenance_linked_triple_count": provenance_linked,
        "source_url_network_reachability_checked": False,
        "checks": checks,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# Phase 1 structural validation\n\n"
        f"- Status: **{report['status'].upper()}**\n"
        f"- Checks passed: {report['passed_check_count']}/{report['check_count']}\n"
        f"- Registered documents: {len(documents)}\n"
        f"- Chunks: {len(chunks)}\n"
        f"- Entities: {len(entities)}\n"
        f"- Triples with provenance: {provenance_linked}/{len(triples)}\n\n"
        "This is structural validation; live HTTP reachability of source URLs was not tested by this script.\n\n"
        "| Check | Result | Detail |\n|---|---|---|\n"
        + "\n".join(
            f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} | {item['detail'].replace('|', '&#124;')} |"
            for item in checks
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = validate_corpus()
    print(
        f"Corpus validation {report['status']}: "
        f"{report['passed_check_count']}/{report['check_count']} checks passed."
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
