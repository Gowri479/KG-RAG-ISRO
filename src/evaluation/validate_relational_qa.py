"""Validate and freeze the provenance-linked relational QA benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus"
BENCHMARK_DIR = ROOT / "data" / "relational_benchmark"
BENCHMARK_PATH = BENCHMARK_DIR / "relational_qa_v1.json"
PROVENANCE_PATH = BENCHMARK_DIR / "relational_qa_provenance.jsonl"
FREEZE_PATH = BENCHMARK_DIR / "relational_qa_v1.freeze.json"
CATEGORIES = {"DIRECT_CONTROL", "SINGLE_RELATION", "TWO_HOP_RELATION", "MULTI_RELATION"}
KG_REQUIRED = {"YES", "NO"}
REQUIRED_FIELDS = {
    "question_id", "question", "category", "kg_required", "relation_type", "mission",
    "match_group_id", "reference_answer", "acceptable_answers", "supporting_triples",
    "supporting_paths", "supporting_chunks", "source_document", "source_url",
    "reasoning_requirement", "answerability",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.casefold()).strip()


def validate(
    benchmark_path: Path = BENCHMARK_PATH,
    provenance_path: Path = PROVENANCE_PATH,
) -> dict[str, Any]:
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    questions = payload.get("questions", [])
    registry = json.loads((CORPUS / "document_registry.json").read_text(encoding="utf-8"))
    documents = {item["document_id"]: item for item in registry["documents"]}
    relation_data = json.loads((CORPUS / "relations.json").read_text(encoding="utf-8"))
    valid_relations = {item["name"] for item in relation_data["relations"]}
    triples = _read_jsonl(CORPUS / "triples.jsonl")
    triple_by_id = {item["triple_id"]: item for item in triples}
    chunks = {item["chunk_id"]: item for item in _read_jsonl(CORPUS / "chunks.jsonl")}
    entities = {item["entity_id"]: item for item in _read_jsonl(CORPUS / "entities.jsonl")}
    valid_missions = {
        entity["name"] for entity in entities.values()
        if entity["type"] == "Mission"
    }
    errors: list[str] = []

    if payload.get("total_questions") != len(questions):
        errors.append("total_questions does not match questions array length")
    question_ids = [item.get("question_id") for item in questions]
    if len(question_ids) != len(set(question_ids)):
        errors.append("question IDs are not unique")
    duplicate_questions: dict[str, list[str]] = defaultdict(list)
    for item in questions:
        duplicate_questions[_normalized(item.get("question", ""))].append(
            item.get("question_id", "<missing>")
        )
    for ids in duplicate_questions.values():
        if len(ids) > 1:
            errors.append(f"duplicate question text: {ids}")

    provenance_rows = _read_jsonl(provenance_path)
    provenance_by_id = {item.get("question_id"): item for item in provenance_rows}
    if len(provenance_rows) != len(questions) or set(provenance_by_id) != set(question_ids):
        errors.append("provenance JSONL IDs do not exactly match benchmark questions")

    match_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    relation_counts: Counter[str] = Counter()
    mission_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    required_counts: Counter[str] = Counter()
    path_pattern_counts: Counter[str] = Counter()
    used_triples_by_question: dict[str, set[str]] = {}

    for question in questions:
        qid = question.get("question_id", "<missing>")
        missing = REQUIRED_FIELDS - set(question)
        if missing:
            errors.append(f"{qid}: missing fields {sorted(missing)}")
            continue
        if not question["question"].strip():
            errors.append(f"{qid}: empty question")
        elif _normalized(question["reference_answer"]) in _normalized(question["question"]):
            errors.append(f"{qid}: reference answer is copied verbatim into the question")
        if question["category"] not in CATEGORIES:
            errors.append(f"{qid}: invalid category {question['category']!r}")
        if question["kg_required"] not in KG_REQUIRED:
            errors.append(f"{qid}: invalid kg_required {question['kg_required']!r}")
        if question["mission"] not in valid_missions:
            errors.append(f"{qid}: unknown mission {question['mission']!r}")
        if question["answerability"] != "ANSWERABLE":
            errors.append(f"{qid}: primary benchmark item is not ANSWERABLE")
        if not question["reference_answer"].strip():
            errors.append(f"{qid}: empty reference answer")
        if not isinstance(question["acceptable_answers"], list) or not question["acceptable_answers"]:
            errors.append(f"{qid}: acceptable_answers must be a non-empty list")
        elif any(not isinstance(answer, str) or not answer.strip() for answer in question["acceptable_answers"]):
            errors.append(f"{qid}: acceptable_answers contains an empty/non-string item")
        if _normalized(question["reference_answer"]) not in {
            _normalized(answer) for answer in question["acceptable_answers"]
        }:
            errors.append(f"{qid}: reference answer is not in acceptable_answers")
        if not isinstance(question["relation_type"], list) or not question["relation_type"]:
            errors.append(f"{qid}: relation_type must be a non-empty list")
        elif not set(question["relation_type"]) <= valid_relations:
            errors.append(f"{qid}: relation_type outside controlled vocabulary")
        if not question["reasoning_requirement"].strip():
            errors.append(f"{qid}: missing reasoning requirement")
        if question.get("system_result_contamination") is not False:
            errors.append(f"{qid}: missing performance-blind construction declaration")

        triple_ids = [item.get("triple_id") for item in question["supporting_triples"]]
        if len(triple_ids) != len(set(triple_ids)):
            errors.append(f"{qid}: duplicate supporting triple")
        triple_ids_set = set(triple_ids)
        used_triples_by_question[qid] = triple_ids_set
        found_relations = set()
        for citation in question["supporting_triples"]:
            triple = triple_by_id.get(citation.get("triple_id"))
            if triple is None:
                errors.append(f"{qid}: unknown supporting triple {citation.get('triple_id')}")
                continue
            exact_fields = (
                ("subject_id", "subject_id"), ("subject", "subject"),
                ("relation", "relation"), ("object_id", "object_id"),
                ("object", "object"),
            )
            if any(citation.get(field) != triple.get(source_field) for field, source_field in exact_fields):
                errors.append(f"{qid}: cited triple endpoints differ for {citation['triple_id']}")
            if triple["relation"] not in valid_relations:
                errors.append(f"{qid}: uncontrolled relation on {triple['triple_id']}")
            found_relations.add(triple["relation"])
            if not triple.get("provenance"):
                errors.append(f"{qid}: triple {triple['triple_id']} has no provenance")
        if found_relations != set(question["relation_type"]):
            errors.append(f"{qid}: relation_type does not match cited triples")

        path_edges = []
        for path in question["supporting_paths"]:
            if not isinstance(path, list) or not path:
                errors.append(f"{qid}: empty or invalid supporting path")
                continue
            path_triples = [triple_by_id.get(triple_id) for triple_id in path]
            if any(item is None for item in path_triples):
                errors.append(f"{qid}: path references an unknown triple")
                continue
            if not set(path) <= triple_ids_set:
                errors.append(f"{qid}: supporting path contains a triple absent from supporting_triples")
            for left, right in zip(path_triples, path_triples[1:]):
                if left["object_id"] != right["subject_id"]:
                    errors.append(
                        f"{qid}: path is disconnected between {left['triple_id']} and {right['triple_id']}"
                    )
            for triple in path_triples:
                if not triple.get("provenance"):
                    errors.append(f"{qid}: path edge {triple['triple_id']} lacks provenance")
            path_edges.extend(path)
            if len(path) >= 2:
                pattern = " -> ".join(
                    [path_triples[0]["subject_type"]]
                    + [
                        label
                        for triple in path_triples
                        for label in (triple["relation"], triple["object_type"])
                    ]
                )
                path_pattern_counts[pattern] += 1
                if question["kg_required"] == "YES":
                    path_chunk_sets = []
                    for triple in path_triples:
                        path_chunk_sets.append({
                            item["source_chunk_id"]
                            for item in triple.get("provenance", [])
                        })
                    if len(set.intersection(*path_chunk_sets)) > 0:
                        errors.append(
                            f"{qid}: KG_REQUIRED=YES path has a single chunk containing every edge"
                        )
        if question["kg_required"] == "YES" and not any(len(path) >= 2 for path in question["supporting_paths"]):
            errors.append(f"{qid}: KG_REQUIRED=YES lacks a multi-edge supporting path")
        if question["category"] == "TWO_HOP_RELATION" and not any(
            len(path) == 2 for path in question["supporting_paths"]
        ):
            errors.append(f"{qid}: TWO_HOP_RELATION must cite a two-edge path")

        object_values = {
            triple_by_id[triple_id]["object"]
            for triple_id in triple_ids_set
            if triple_id in triple_by_id
        }
        if not any(
            _normalized(answer) in {_normalized(value) for value in object_values}
            for answer in question.get("acceptable_answers", [])
        ):
            errors.append(f"{qid}: no acceptable answer matches a supported triple object")
        answer_entities = question.get("answer_entities", [])
        supported_object_ids = {
            triple_by_id[triple_id]["object_id"]
            for triple_id in triple_ids_set
            if triple_id in triple_by_id
        }
        if not answer_entities or not set(answer_entities) <= supported_object_ids:
            errors.append(f"{qid}: answer_entities are missing or unsupported by cited facts")

        cited_chunk_keys: set[tuple[str, str]] = set()
        for citation in question["supporting_triples"]:
            triple = triple_by_id.get(citation.get("triple_id"))
            if triple is None:
                continue
            for source in triple.get("provenance", []):
                key = (source["document_id"], source["source_chunk_id"])
                cited_chunk_keys.add(key)
                chunk = chunks.get(source["source_chunk_id"])
                if chunk is None:
                    errors.append(f"{qid}: source chunk does not exist: {source['source_chunk_id']}")
                elif chunk["document_id"] != source["document_id"]:
                    errors.append(f"{qid}: source chunk/document mismatch for {source['source_chunk_id']}")
                elif _normalized(source["supporting_excerpt"]) not in _normalized(chunk["text"]):
                    errors.append(
                        f"{qid}: supporting excerpt is absent from chunk {source['source_chunk_id']}"
                    )
                if source["document_id"] not in documents:
                    errors.append(f"{qid}: unknown source document {source['document_id']}")
                elif documents[source["document_id"]]["source_url"] != source["source_url"]:
                    errors.append(f"{qid}: source URL differs from document registry")
        declared_chunks = {
            (item.get("document_id"), item.get("chunk_id"))
            for item in question["supporting_chunks"]
        }
        if cited_chunk_keys != declared_chunks:
            errors.append(f"{qid}: supporting_chunks do not exactly match triple provenance chunks")
        for item in question["supporting_chunks"]:
            doc_id, chunk_id = item.get("document_id"), item.get("chunk_id")
            if doc_id not in documents:
                errors.append(f"{qid}: unknown supporting source document {doc_id}")
                continue
            if documents[doc_id]["source_url"] != item.get("source_url"):
                errors.append(f"{qid}: supporting chunk URL differs from registry for {doc_id}")
            chunk = chunks.get(chunk_id)
            if chunk is None or chunk.get("document_id") != doc_id:
                errors.append(f"{qid}: missing/mismatched supporting chunk {chunk_id}")
            elif _normalized(item.get("supporting_excerpt", "")) not in _normalized(chunk["text"]):
                errors.append(f"{qid}: cited excerpt is absent from supporting chunk {chunk_id}")

        docs_for_edges = {
            item["document_id"]
            for triple_id in triple_ids_set if triple_id in triple_by_id
            for item in triple_by_id[triple_id].get("provenance", [])
        }
        if set(question.get("source_documents", [])) != docs_for_edges:
            errors.append(f"{qid}: source_documents does not match triple provenance")
        expected_urls = {
            documents[doc_id]["source_url"] for doc_id in docs_for_edges if doc_id in documents
        }
        if set(question.get("source_urls", [])) != expected_urls:
            errors.append(f"{qid}: source_urls do not match triple provenance")
        if question["source_document"] not in documents:
            errors.append(f"{qid}: unknown primary source document {question['source_document']}")
        elif documents[question["source_document"]]["source_url"] != question["source_url"]:
            errors.append(f"{qid}: primary source URL differs from document registry")

        if question.get("match_group_id"):
            match_groups[question["match_group_id"]].append(question)
        if question["category"] in {"DIRECT_CONTROL", "SINGLE_RELATION"} and question["kg_required"] != "NO":
            errors.append(f"{qid}: direct/control item must have KG_REQUIRED=NO")
        if question["category"] in {"TWO_HOP_RELATION", "MULTI_RELATION"} and question["kg_required"] != "YES":
            errors.append(f"{qid}: relational item must have KG_REQUIRED=YES")
        relation_counts.update(question["relation_type"])
        mission_counts[question["mission"]] += 1
        category_counts[question["category"]] += 1
        required_counts[question["kg_required"]] += 1

        provenance_item = provenance_by_id.get(qid)
        if provenance_item is None:
            errors.append(f"{qid}: missing provenance JSONL row")
        else:
            if set(item.get("triple_id") for item in provenance_item.get("supporting_triples", [])) != triple_ids_set:
                errors.append(f"{qid}: benchmark/provenance JSONL triple IDs differ")
            for field in ("supporting_triples", "supporting_paths", "supporting_chunks", "source_documents", "source_urls"):
                if provenance_item.get(field) != question.get(field):
                    errors.append(f"{qid}: benchmark/provenance JSONL {field} differ")

    for group_id, members in match_groups.items():
        if len(members) != 2:
            errors.append(f"{group_id}: matched group must contain exactly two questions")
            continue
        if {item["kg_required"] for item in members} != {"YES", "NO"}:
            errors.append(f"{group_id}: matched pair must contain one YES and one NO")
        if len({item["category"] for item in members}) != 2:
            errors.append(f"{group_id}: matched pair must use two different categories")

    return {
        "status": "passed" if not errors else "failed",
        "question_count": len(questions),
        "unique_question_count": len(set(question_ids)),
        "provenance_row_count": len(provenance_rows),
        "error_count": len(errors),
        "errors": errors,
        "mission_distribution": dict(sorted(mission_counts.items())),
        "relation_distribution": dict(sorted(relation_counts.items())),
        "kg_required_distribution": dict(sorted(required_counts.items())),
        "category_distribution": dict(sorted(category_counts.items())),
        "two_hop_or_longer_path_pattern_counts": dict(sorted(path_pattern_counts.items())),
    }


def freeze(report: dict[str, Any], benchmark_path: Path = BENCHMARK_PATH) -> dict[str, Any]:
    if report["status"] != "passed":
        raise ValueError("Cannot freeze a benchmark that failed validation")
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    freeze_record = {
        "benchmark_id": payload["benchmark_id"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_system_evaluation": True,
        "benchmark_sha256": _hash(benchmark_path),
        "provenance_sha256": _hash(PROVENANCE_PATH),
        "benchmark_file": str(benchmark_path.relative_to(ROOT)).replace("\\", "/"),
        "provenance_file": str(PROVENANCE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "question_count": report["question_count"],
        "mission_distribution": report["mission_distribution"],
        "relation_distribution": report["relation_distribution"],
        "kg_required_distribution": report["kg_required_distribution"],
        "category_distribution": report["category_distribution"],
        "two_hop_or_longer_path_pattern_counts": report["two_hop_or_longer_path_pattern_counts"],
        "validation_status": report["status"],
        "system_outputs_consulted_during_construction": False,
    }
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(json.dumps(freeze_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return freeze_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="Write the SHA-256 freeze record after successful validation.")
    args = parser.parse_args()
    report = validate()
    print(
        f"Relational QA validation {report['status']}: "
        f"{report['question_count']} questions, "
        f"{report['unique_question_count']} unique IDs, "
        f"{report['error_count']} errors."
    )
    if report["errors"]:
        for error in report["errors"]:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"KG_REQUIRED: {report['kg_required_distribution']}")
    print(f"Categories: {report['category_distribution']}")
    if args.freeze:
        freeze_record = freeze(report)
        print(f"Frozen SHA-256: {freeze_record['benchmark_sha256']}")


if __name__ == "__main__":
    main()
