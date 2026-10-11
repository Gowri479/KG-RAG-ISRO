"""Build a deterministic blinded human-evaluation template for Phase 4."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "relational_benchmark" / "relational_qa_v1.json"
RESULTS = ROOT / "data" / "results" / "relational_qa_v1"
CHECKPOINT = RESULTS / "per_question_results.jsonl"
OUTPUT = RESULTS / "human_evaluation_blinded.csv"
MANIFEST = RESULTS / "human_evaluation_blinded_manifest.json"
SYSTEMS = ["bm25_llm", "vanilla_rag", "kg_rag"]


def build() -> None:
    questions = {
        item["question_id"]: item
        for item in json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
    }
    rows = [
        json.loads(line)
        for line in CHECKPOINT.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    by_question: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_question.setdefault(row["question_id"], {})[row["system"]] = row
    rng = random.Random(20261007)
    fieldnames = [
        "question_id", "question", "category", "kg_required", "reference_answer",
        "system_a", "answer_a", "evidence_a", "system_b", "answer_b", "evidence_b",
        "system_c", "answer_c", "evidence_c",
        "correctness_system_a", "correctness_system_b", "correctness_system_c",
        "completeness_system_a", "completeness_system_b", "completeness_system_c",
        "groundedness_system_a", "groundedness_system_b", "groundedness_system_c",
        "relevance_system_a", "relevance_system_b", "relevance_system_c",
        "unsupported_claim_system_a", "unsupported_claim_system_b", "unsupported_claim_system_c",
        "notes",
    ]
    output_rows = []
    for qid in sorted(questions):
        question = questions[qid]
        order = SYSTEMS[:]
        rng.shuffle(order)
        evaluation = by_question[qid]
        anonymized = {"system_a": "A", "system_b": "B", "system_c": "C"}
        out = {
            "question_id": qid,
            "question": question["question"],
            "category": question["category"],
            "kg_required": question["kg_required"],
            "reference_answer": question["reference_answer"],
            "notes": "Human scores intentionally blank; evaluation is pending.",
        }
        for index, system in enumerate(order):
            label = f"system_{chr(ord('a') + index)}"
            row = evaluation[system]
            evidence = row["retrieval_trace"].get("final_fused_evidence", "")
            out[f"{label}"] = anonymized[label]
            out[f"answer_{chr(ord('a') + index)}"] = row["generated_answer"]
            out[f"evidence_{chr(ord('a') + index)}"] = evidence
            label = chr(ord("a") + index)
            out[f"correctness_system_{label}"] = ""
            out[f"completeness_system_{label}"] = ""
            out[f"groundedness_system_{label}"] = ""
            out[f"relevance_system_{label}"] = ""
            out[f"unsupported_claim_system_{label}"] = ""
        output_rows.append(out)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    manifest = {
        "status": "PENDING",
        "question_count": len(output_rows),
        "systems_per_question": 3,
        "random_seed": 20261007,
        "system_mapping_is_hidden_from_annotator": True,
        "scoring_fields": ["correct_system_[a-c]", "grounded_system_[a-c]", "complete_system_[a-c]"],
        "source_checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "instructions": "Score each anonymized answer for correctness, evidence grounding, and completeness. Do not infer scores from lexical metrics.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"csv": str(OUTPUT), "questions": len(output_rows), "status": "PENDING"}, indent=2))


if __name__ == "__main__":
    build()
