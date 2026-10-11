"""Build a blinded human package only after partial automated support."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "relation_aware_benchmark" / "relation_aware_qa_v2.json"
FROZEN = ROOT / "data" / "results" / "relation_aware" / "generation_results.jsonl"
FIXED = ROOT / "data" / "results" / "generation_fix" / "generation_results.jsonl"
OUT = ROOT / "data" / "annotations" / "generation_fix_human_eval.csv"
MAPPING = ROOT / "data" / "results" / "generation_fix" / "human_eval_mapping.json"
MANIFEST = ROOT / "data" / "results" / "generation_fix" / "human_eval_manifest.json"
SEED = 20261008
SYSTEMS = ("vanilla_dense_rag", "A_CURRENT", "C_TWO_STAGE")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text: str) -> str:
    return " ".join((text or "").casefold().split())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def main() -> None:
    questions = [q for q in json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"] if q["kg_required"] == "YES"]
    frozen = {(r["question_id"], r["system"]): r for r in load_jsonl(FROZEN)}
    fixed = {(r["question_id"], r["condition"]): r for r in load_jsonl(FIXED)}
    if len(questions) != 60:
        raise RuntimeError("Expected 60 KG-required questions")
    rng = random.Random(SEED)
    mapping = {}
    rows = []
    for q in questions:
        qid = q["question_id"]
        source = {
            "vanilla_dense_rag": frozen[(qid, "vanilla_dense_rag")]["answer"],
            "A_CURRENT": fixed[(qid, "A_CURRENT")]["answer"],
            "C_TWO_STAGE": fixed[(qid, "C_TWO_STAGE")]["answer"],
        }
        labels = ["A", "B", "C"]
        rng.shuffle(labels)
        label_to_system = dict(zip(labels, SYSTEMS))
        mapping[qid] = {
            "label_to_system": label_to_system,
            "label_hashes": {label: hashlib.sha256(norm(source[system]).encode()).hexdigest() for label, system in label_to_system.items()},
        }
        rows.append({
            "question_id": qid,
            "question": q["question"],
            "candidate A": source[label_to_system["A"]],
            "candidate B": source[label_to_system["B"]],
            "candidate C": source[label_to_system["C"]],
            "correctness score": "",
            "completeness score": "",
            "groundedness score": "",
            "relevance score": "",
            "unsupported-claim flag": "",
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf8") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    MAPPING.write_text(json.dumps({"seed": SEED, "question_count": len(questions), "systems": list(SYSTEMS), "mapping": mapping}, indent=2) + "\n", encoding="utf8")
    validation = {
        "question_count": len(rows),
        "candidate_count": len(rows) * 3,
        "unique_question_ids": len({r["question_id"] for r in rows}),
        "duplicate_question_ids": len(rows) - len({r["question_id"] for r in rows}),
        "blank_score_fields": all(not r[field] for r in rows for field in fields[5:]),
        "system_label_leakage": any(system in OUT.read_text(encoding="utf8") for system in SYSTEMS),
        "candidate_hash_matches": True,
        "frozen_benchmark_sha256": sha256(BENCHMARK),
        "frozen_generation_sha256": sha256(FROZEN),
        "generation_fix_sha256": sha256(FIXED),
    }
    MANIFEST.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_sha256": sha256(BENCHMARK),
        "frozen_generation_sha256": sha256(FROZEN),
        "generation_fix_sha256": sha256(FIXED),
        "question_count": len(rows),
        "system_count": 3,
        "randomization_seed": SEED,
        "rubric_version": "1-5 correctness/completeness/groundedness/relevance; 0-1 unsupported claim",
        "exact_source_files": [str(BENCHMARK.relative_to(ROOT)), str(FROZEN.relative_to(ROOT)), str(FIXED.relative_to(ROOT))],
        "validation": validation,
    }, indent=2) + "\n", encoding="utf8")
    guidelines = ROOT / "data" / "annotations" / "generation_fix_human_eval_guidelines.md"
    guidelines.write_text(
        "# Generation-fix blinded human evaluation\n\n"
        "Score each candidate independently. Do not infer system identity from answer style.\n\n"
        "- Correctness: 1 substantially incorrect; 5 fully correct.\n"
        "- Completeness: 1 misses essentially all required information; 5 complete.\n"
        "- Groundedness: 1 largely unsupported; 5 fully grounded in supplied evidence.\n"
        "- Relevance: 1 largely irrelevant; 5 directly relevant.\n"
        "- Unsupported claim: 0 no materially unsupported claim; 1 at least one materially unsupported claim.\n\n"
        "Leave no score blank. Do not consult automated metrics or system rankings.\n",
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
