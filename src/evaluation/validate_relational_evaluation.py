"""Validate Phase 4 evaluation outputs without changing frozen inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "relational_benchmark" / "relational_qa_v1.json"
FREEZE = ROOT / "data" / "relational_benchmark" / "relational_qa_v1.freeze.json"
CHECKPOINT = ROOT / "data" / "results" / "relational_qa_v1" / "per_question_results.jsonl"
SYSTEMS = {"bm25_llm", "vanilla_rag", "kg_rag"}


def validate() -> dict:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf8"))
    questions = {item["question_id"]: item for item in benchmark["questions"]}
    expected_hash = json.loads(FREEZE.read_text(encoding="utf8"))["benchmark_sha256"]
    actual_hash = hashlib.sha256(BENCHMARK.read_bytes()).hexdigest()
    errors = []
    rows = [
        json.loads(line)
        for line in CHECKPOINT.read_text(encoding="utf8").splitlines()
        if line.strip()
    ] if CHECKPOINT.exists() else []
    keys = [(row.get("question_id"), row.get("system")) for row in rows]
    if actual_hash != expected_hash:
        errors.append("benchmark SHA-256 no longer matches freeze record")
    if len(questions) != 62:
        errors.append(f"expected 62 benchmark questions, found {len(questions)}")
    if len(rows) != 186:
        errors.append(f"expected 186 evaluations, found {len(rows)}")
    if len(keys) != len(set(keys)):
        errors.append("duplicate question/system evaluation rows")
    if set(key[0] for key in keys) != set(questions):
        errors.append("evaluation question IDs do not match benchmark IDs")
    if set(key[1] for key in keys) != SYSTEMS:
        errors.append("evaluation system names are incomplete or invalid")
    for row in rows:
        qid = row.get("question_id")
        question = questions.get(qid)
        if question is None:
            errors.append(f"unknown question ID {qid}")
            continue
        if row.get("question") != question["question"]:
            errors.append(f"{qid}: question text changed")
        for field in ("category", "kg_required", "relation_type", "mission", "match_group_id", "reference_answer"):
            if row.get(field) != question[field]:
                errors.append(f"{qid}: preserved field changed: {field}")
        if not isinstance(row.get("generated_answer"), str):
            errors.append(f"{qid}: generated_answer is not a string")
        if not isinstance(row.get("retrieval_trace"), dict):
            errors.append(f"{qid}: retrieval trace missing")
    return {
        "status": "passed" if not errors else "failed",
        "benchmark_sha256": actual_hash,
        "question_count": len(questions),
        "evaluation_count": len(rows),
        "error_count": len(errors),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = validate()
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
