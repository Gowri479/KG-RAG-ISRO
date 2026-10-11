"""Build typed entities and evidence-linked triples from curated annotations."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus"
ANNOTATIONS_PATH = CORPUS / "fact_annotations.jsonl"
REGISTRY_PATH = CORPUS / "document_registry.json"
ONTOLOGY_PATH = CORPUS / "ontology.json"
RELATIONS_PATH = CORPUS / "relations.json"
CHUNKS_PATH = CORPUS / "chunks.jsonl"
ENTITIES_PATH = CORPUS / "entities.jsonl"
TRIPLES_PATH = CORPUS / "triples.jsonl"
LOGGER = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required corpus file does not exist: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_evidence(
    reference: dict[str, Any],
    chunks_by_document: dict[str, list[dict[str, Any]]],
    documents_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    document_id = reference["document_id"]
    if document_id not in documents_by_id:
        raise ValueError(f"Evidence references unknown document {document_id!r}")
    candidates = chunks_by_document.get(document_id, [])
    if reference.get("chunk_id"):
        candidates = [chunk for chunk in candidates if chunk["chunk_id"] == reference["chunk_id"]]
    if reference.get("page") is not None:
        candidates = [chunk for chunk in candidates if chunk.get("page") == reference["page"]]
    locator = reference["match"]
    candidates = [
        chunk for chunk in candidates
        if re.search(re.escape(locator), chunk["text"], re.I)
        or re.search(re.escape(locator), chunk["section"], re.I)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Evidence locator for {document_id} must identify exactly one chunk; "
            f"found {len(candidates)} for {locator!r}"
        )

    chunk = candidates[0]
    match = re.search(re.escape(locator), chunk["text"], re.I)
    evidence_text = chunk["text"]
    if match is None:
        match = re.search(re.escape(locator), chunk["section"], re.I)
        evidence_text = chunk["section"]
    if match is None:
        raise AssertionError("Evidence match disappeared after candidate selection")
    document = documents_by_id[document_id]
    if document["source_url"] != chunk["source_url"]:
        raise ValueError(f"Chunk URL does not match registry for {document_id}")
    return {
        "document_id": document_id,
        "title": document["title"],
        "source_url": document["source_url"],
        "source_domain": document["source_domain"],
        "authority_tier": document["authority_tier"],
        "source_type": "official_primary",
        "source_section": chunk["section"],
        "source_chunk_id": chunk["chunk_id"],
        "source_page": chunk.get("page"),
        "source_text": chunk["text"],
        "supporting_excerpt": match.group(0),
        "supporting_excerpt_location": "chunk_text" if evidence_text == chunk["text"] else "section_heading",
    }


def build_kg_ready() -> dict[str, int]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    relation_schema = json.loads(RELATIONS_PATH.read_text(encoding="utf-8"))
    annotations = _read_jsonl(ANNOTATIONS_PATH)
    chunks = _read_jsonl(CHUNKS_PATH)

    documents_by_id = {document["document_id"]: document for document in registry["documents"]}
    chunks_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[chunk["document_id"]].append(chunk)
    entity_types = {entity["name"] for entity in ontology["entity_types"]}
    relations = {relation["name"]: relation for relation in relation_schema["relations"]}
    entities: dict[str, dict[str, Any]] = {}
    triples: list[dict[str, Any]] = []
    triple_keys: set[tuple[str, str, str]] = set()
    annotation_ids: set[str] = set()

    def add_entity(entity: dict[str, Any], mission: str, evidence: list[dict[str, Any]]) -> None:
        entity_id = entity["id"]
        entity_type = entity["type"]
        if entity_type not in entity_types:
            raise ValueError(f"Entity {entity_id} uses uncontrolled type {entity_type!r}")
        stored = entities.get(entity_id)
        if stored and (stored["name"] != entity["name"] or stored["type"] != entity_type):
            raise ValueError(f"Conflicting definitions for entity ID {entity_id}")
        if stored is None:
            stored = {
                "entity_id": entity_id,
                "name": entity["name"],
                "type": entity_type,
                "mission": mission,
                "aliases": sorted(set(entity.get("aliases", []))),
                "properties": entity.get("properties", {}),
                "evidence": [],
            }
            entities[entity_id] = stored
        elif stored["mission"] != mission and mission != "cross-mission":
            stored["mission"] = "cross-mission"
        for item in evidence:
            key = (item["document_id"], item["source_chunk_id"])
            if not any((entry["document_id"], entry["source_chunk_id"]) == key for entry in stored["evidence"]):
                stored["evidence"].append({
                    "document_id": item["document_id"],
                    "source_url": item["source_url"],
                    "authority_tier": item["authority_tier"],
                    "source_type": item["source_type"],
                    "source_section": item["source_section"],
                    "source_chunk_id": item["source_chunk_id"],
                    "source_page": item["source_page"],
                    "source_text": item["source_text"],
                    "supporting_excerpt": item["supporting_excerpt"],
                    "supporting_excerpt_location": item["supporting_excerpt_location"],
                })

    for annotation in annotations:
        annotation_id = annotation["annotation_id"]
        if annotation_id in annotation_ids:
            raise ValueError(f"Duplicate annotation_id {annotation_id}")
        annotation_ids.add(annotation_id)
        relation_name = annotation["relation"]
        relation = relations.get(relation_name)
        if relation is None:
            raise ValueError(f"Relation {relation_name!r} is not in the controlled vocabulary")

        subject = annotation["subject"]
        object_ = annotation["object"]
        if subject["type"] not in relation["subject_types"]:
            raise ValueError(f"{relation_name} does not accept subject type {subject['type']}")
        if object_["type"] not in relation["object_types"]:
            raise ValueError(f"{relation_name} does not accept object type {object_['type']}")
        evidence_type = annotation.get("evidence_type", "direct")
        if relation["evidence_mode"] == "direct_only" and evidence_type != "direct":
            raise ValueError(f"{relation_name} is direct-only but {annotation_id} is marked {evidence_type}")

        evidence = [
            _resolve_evidence(item, chunks_by_document, documents_by_id)
            for item in annotation["evidence"]
        ]
        if not evidence:
            raise ValueError(f"Annotation {annotation_id} has no provenance")

        triple_key = (subject["id"], relation_name, object_["id"])
        if triple_key in triple_keys:
            raise ValueError(f"Duplicate curated triple {triple_key}")
        triple_keys.add(triple_key)
        mission = annotation["mission"]
        add_entity(subject, mission, evidence)
        add_entity(object_, mission, evidence)
        primary = evidence[0]
        triples.append({
            "triple_id": annotation_id,
            "mission": mission,
            "subject_id": subject["id"],
            "subject": subject["name"],
            "subject_type": subject["type"],
            "relation": relation_name,
            "object_id": object_["id"],
            "object": object_["name"],
            "object_type": object_["type"],
            "evidence_type": evidence_type,
            "extraction_method": "human_curated_from_primary_source",
            "annotation_note": annotation.get("annotation_note", ""),
            "source_document": primary["document_id"],
            "source_url": primary["source_url"],
            "source_type": primary["source_type"],
            "source_section": primary["source_section"],
            "source_chunk_id": primary["source_chunk_id"],
            "source_page": primary["source_page"],
            "source_text": primary["source_text"],
            "supporting_excerpt": primary["supporting_excerpt"],
            "supporting_excerpt_location": primary["supporting_excerpt_location"],
            "provenance": evidence,
        })

    referenced_ids = {item for triple in triples for item in (triple["subject_id"], triple["object_id"])}
    if set(entities) != referenced_ids:
        raise ValueError("Entity output contains an orphan or a triple endpoint is missing.")

    ENTITIES_PATH.write_text(
        "".join(json.dumps(entity, ensure_ascii=False) + "\n" for entity in sorted(entities.values(), key=lambda x: x["entity_id"])),
        encoding="utf-8",
    )
    TRIPLES_PATH.write_text(
        "".join(json.dumps(triple, ensure_ascii=False) + "\n" for triple in triples),
        encoding="utf-8",
    )
    result = {"entity_count": len(entities), "triple_count": len(triples), "annotation_count": len(annotations)}
    LOGGER.info("Built %d typed entities and %d provenance-linked triples.", len(entities), len(triples))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    build_kg_ready()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
