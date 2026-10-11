"""Build a blinded human package from only the 13 held-out frozen questions."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "relation_aware_benchmark" / "relation_aware_qa_v2.json"
FIXED = ROOT / "data" / "results" / "generation_fix" / "generation_results.jsonl"
FROZEN = ROOT / "data" / "results" / "relation_aware" / "generation_results.jsonl"
OUT = ROOT / "data" / "annotations" / "generation_fix_heldout_human_eval.csv"
GUIDELINES = ROOT / "data" / "annotations" / "generation_fix_heldout_human_eval_guidelines.md"
MAPPING = ROOT / "data" / "results" / "generation_fix" / "heldout_human_eval_mapping.json"
MANIFEST = ROOT / "data" / "results" / "generation_fix" / "heldout_human_eval_manifest.json"
SEED = 20261008


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text: str) -> str:
    return " ".join((text or "").casefold().split())


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def main() -> None:
    questions = [
        q for q in json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
        if q["kg_required"] == "YES" and q["split"] == "evaluation"
    ]
    fixed = {(r["question_id"], r["condition"]): r for r in load(FIXED)}
    frozen = {(r["question_id"], r["system"]): r for r in load(FROZEN)}
    rng = random.Random(SEED)
    systems = ("vanilla_dense_rag", "A_CURRENT", "C_TWO_STAGE")
    rows = []
    mapping = {}
    for question in questions:
        qid = question["question_id"]
        answers = {
            "vanilla_dense_rag": frozen[(qid, "vanilla_dense_rag")]["answer"],
            "A_CURRENT": fixed[(qid, "A_CURRENT")]["answer"],
            "C_TWO_STAGE": fixed[(qid, "C_TWO_STAGE")]["answer"],
        }
        labels = ["A", "B", "C"]
        rng.shuffle(labels)
        label_to_system = dict(zip(labels, systems))
        mapping[qid] = {
            "label_to_system": label_to_system,
            "label_hashes": {
                label: hashlib.sha256(norm(answers[system]).encode()).hexdigest()
                for label, system in label_to_system.items()
            },
        }
        rows.append({
            "question_id": qid,
            "question": question["question"],
            "candidate A": answers[label_to_system["A"]],
            "candidate B": answers[label_to_system["B"]],
            "candidate C": answers[label_to_system["C"]],
            "correctness score": "",
            "completeness score": "",
            "groundedness score": "",
            "relevance score": "",
            "unsupported-claim flag": "",
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with OUT.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    csv_text = OUT.read_text(encoding="utf8")
    exact_hash_matches = 0
    for row in rows:
        for label in ("A", "B", "C"):
            system = mapping[row["question_id"]]["label_to_system"][label]
            expected_answer = (
                frozen[(row["question_id"], "vanilla_dense_rag")]["answer"]
                if system == "vanilla_dense_rag"
                else fixed[(row["question_id"], system)]["answer"]
            )
            expected_hash = hashlib.sha256(norm(expected_answer).encode()).hexdigest()
            if expected_hash == mapping[row["question_id"]]["label_hashes"][label]:
                exact_hash_matches += 1
    validation = {
        "question_count": len(rows),
        "candidate_count": len(rows) * 3,
        "unique_question_ids": len({r["question_id"] for r in rows}),
        "duplicate_question_ids": len(rows) - len({r["question_id"] for r in rows}),
        "blank_score_fields": all(not r[field] for r in rows for field in fields[5:]),
        "system_name_leakage": any(system in csv_text for system in systems),
        "candidate_hashes_present": len(mapping) == 13 and all(len(v["label_hashes"]) == 3 for v in mapping.values()),
        "exact_candidate_hash_matches": exact_hash_matches,
        "expected_candidate_hash_matches": len(rows) * 3,
    }
    MAPPING.write_text(json.dumps({"seed": SEED, "question_ids": [q["question_id"] for q in questions], "systems": list(systems), "mapping": mapping}, indent=2) + "\n", encoding="utf8")
    MANIFEST.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_sha256": sha256(BENCHMARK),
        "generation_output_sha256": sha256(FIXED),
        "prior_generation_output_sha256": sha256(FROZEN),
        "held_out_question_ids": [q["question_id"] for q in questions],
        "development_question_ids": [q["question_id"] for q in json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"] if q["kg_required"] == "YES" and q["split"] == "development"],
        "system_count": 3,
        "systems": list(systems),
        "randomization_seed": SEED,
        "rubric_version": "1-5 correctness/completeness/groundedness/relevance; 0-1 unsupported claim",
        "exact_source_files": [str(BENCHMARK.relative_to(ROOT)), str(FIXED.relative_to(ROOT)), str(FROZEN.relative_to(ROOT))],
        "validation": validation,
    }, indent=2) + "\n", encoding="utf8")
    GUIDELINES.write_text(
        "# Held-out generation-fix blinded human evaluation\n\n"
        "This package contains only the 13 held-out KG-required questions. Score candidates independently without inferring system identity.\n\n"
        "- Correctness: 1 substantially incorrect; 5 fully correct.\n"
        "- Completeness: 1 misses essentially all required information; 5 complete.\n"
        "- Groundedness: 1 largely unsupported; 5 fully grounded in supplied evidence.\n"
        "- Relevance: 1 largely irrelevant; 5 directly relevant.\n"
        "- Unsupported claim: 0 no materially unsupported claim; 1 at least one materially unsupported claim.\n\n"
        "Do not consult automated metrics, system mappings, or prior rankings. Leave every score field complete.\n",
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
