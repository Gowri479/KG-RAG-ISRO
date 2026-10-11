"""Validate the question-level annotations against the frozen benchmark and KG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.analyze_kg_relevance import (
    ANNOTATION_PATH,
    KG_RELEVANCE,
    KG_REQUIRED,
    QUESTION_TYPES,
    _load_relations,
)
from src.evaluation.analysis_utils import BENCHMARK_PATH, TEST_IDS_PATH, load_json

ROOT = Path(__file__).resolve().parents[2]
TRIPLES_PATH = ROOT / "data" / "corpus" / "triples.jsonl"
REQUIRED_FIELDS = {
    "question_id",
    "question",
    "mission",
    "tier",
    "question_type",
    "kg_relevance",
    "kg_required",
    "kg_relation_types",
    "reason",
    "supporting_evidence",
}
EVIDENCE_FIELDS = {
    "triple_id",
    "subject",
    "relation",
    "object",
    "source_document",
    "chunk_id",
    "supporting_excerpt",
}


def validate() -> dict[str, Any]:
    benchmark = load_json(BENCHMARK_PATH)
    test_ids = set(load_json(TEST_IDS_PATH))
    benchmark_by_id = {item["id"]: item for item in benchmark}
    relation_names = _load_relations()
    triples = [
        json.loads(line)
        for line in TRIPLES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    triples_by_id = {triple["triple_id"]: triple for triple in triples}
    lines = [
        json.loads(line)
        for line in ANNOTATION_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors = []
    ids = [item.get("question_id") for item in lines]
    if len(lines) != 180:
        errors.append(f"Expected 180 annotation records, found {len(lines)}")
    if len(set(ids)) != len(ids):
        errors.append("Question IDs are not unique")
    if set(ids) != test_ids:
        errors.append("Annotation question IDs do not exactly match the frozen test split")
    for record in lines:
        qid = record.get("question_id", "<missing>")
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            errors.append(f"{qid}: missing required fields {sorted(missing)}")
            continue
        source_question = benchmark_by_id.get(qid)
        if source_question is None:
            errors.append(f"{qid}: not found in canonical benchmark")
        elif (
            record["question"] != source_question["question"]
            or record["tier"] != source_question["tier"]
        ):
            errors.append(f"{qid}: question text or tier differs from frozen benchmark")
        if record["question_type"] not in QUESTION_TYPES:
            errors.append(f"{qid}: invalid question_type {record['question_type']!r}")
        if record["kg_relevance"] not in KG_RELEVANCE:
            errors.append(f"{qid}: invalid kg_relevance {record['kg_relevance']!r}")
        if record["kg_required"] not in KG_REQUIRED:
            errors.append(f"{qid}: invalid kg_required {record['kg_required']!r}")
        if not isinstance(record["kg_relation_types"], list):
            errors.append(f"{qid}: kg_relation_types must be a list")
        elif not set(record["kg_relation_types"]) <= relation_names:
            errors.append(f"{qid}: relation names are outside the controlled vocabulary")
        if not isinstance(record["reason"], str) or not record["reason"].strip():
            errors.append(f"{qid}: missing annotation rationale")
        if not isinstance(record["supporting_evidence"], list):
            errors.append(f"{qid}: supporting_evidence must be a list")
            continue
        if record["kg_required"] == "YES" and (
            not record["kg_relation_types"] or not record["supporting_evidence"]
        ):
            errors.append(f"{qid}: KG_REQUIRED=YES needs relation and evidence")
        for evidence in record["supporting_evidence"]:
            if not EVIDENCE_FIELDS <= set(evidence):
                errors.append(f"{qid}: evidence missing fields {sorted(EVIDENCE_FIELDS - set(evidence))}")
                continue
            triple = triples_by_id.get(evidence["triple_id"])
            if triple is None:
                errors.append(f"{qid}: evidence cites unknown triple {evidence['triple_id']}")
                continue
            if (
                evidence["subject"] != triple["subject"]
                or evidence["relation"] != triple["relation"]
                or evidence["object"] != triple["object"]
            ):
                errors.append(f"{qid}: evidence endpoints/relation differ from the cited triple")
            if evidence["relation"] not in relation_names:
                errors.append(f"{qid}: cited relation is not controlled")
            provenance = triple.get("provenance", [])
            if not any(
                item["document_id"] == evidence["source_document"]
                and item["source_chunk_id"] == evidence["chunk_id"]
                and item["supporting_excerpt"] == evidence["supporting_excerpt"]
                for item in provenance
            ):
                errors.append(f"{qid}: evidence does not match triple provenance")
    return {
        "status": "passed" if not errors else "failed",
        "annotation_count": len(lines),
        "unique_question_count": len(set(ids)),
        "frozen_test_id_count": len(test_ids),
        "error_count": len(errors),
        "errors": errors,
    }


def main() -> None:
    report = validate()
    print(
        f"KG relevance validation {report['status']}: "
        f"{report['annotation_count']} records, "
        f"{report['unique_question_count']} unique IDs, "
        f"{report['error_count']} errors."
    )
    if report["errors"]:
        for error in report["errors"]:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
