"""Run isolated A/B/C generation architectures over frozen relation-aware traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.evaluate_relational_qa import metrics
from src.generator.kg_grounded_generator import facts_from_paths, generate_condition

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "relation_aware_benchmark" / "relation_aware_qa_v2.json"
FROZEN_GENERATIONS = ROOT / "data" / "results" / "relation_aware" / "generation_results.jsonl"
OUTPUT = ROOT / "data" / "results" / "generation_fix"
RESULTS = OUTPUT / "generation_results.jsonl"
OPTIONS = {"temperature": 0.1, "num_predict": 150, "num_ctx": 2048}
CONDITIONS = ("A_CURRENT", "B_STRUCTURED", "C_TWO_STAGE")
TOKEN_BUDGET = 1500


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_questions() -> list[dict]:
    questions = json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
    return [q for q in questions if q["kg_required"] == "YES"]


def load_frozen_traces() -> dict[str, dict]:
    traces: dict[str, dict] = {}
    for line in FROZEN_GENERATIONS.read_text(encoding="utf8").splitlines():
        row = json.loads(line)
        if row["system"] == "relation_aware_kg_rag" and row["kg_required"] == "YES":
            if row["question_id"] in traces:
                raise RuntimeError(f"Duplicate frozen relation-aware trace: {row['question_id']}")
            traces[row["question_id"]] = row
    return traces


def read_checkpoint() -> dict[tuple[str, str], dict]:
    if not RESULTS.exists():
        return {}
    rows = {}
    for line in RESULTS.read_text(encoding="utf8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = (row["question_id"], row["condition"])
            if key in rows:
                raise RuntimeError(f"Duplicate generation-fix row: {key}")
            rows[key] = row
    return rows


def main(limit: int | None = None) -> None:
    from src.generator.ollama_api import UNKNOWN

    OUTPUT.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    traces = load_frozen_traces()
    if len(questions) != 60 or len(traces) != 60:
        raise RuntimeError(f"Expected 60 questions and frozen traces, got {len(questions)} and {len(traces)}")
    rows = read_checkpoint()
    target = questions[:limit] if limit else questions
    for question in target:
        frozen = traces[question["question_id"]]
        facts = facts_from_paths(frozen["selected_evidence"])
        dense_text = frozen["context"].split("[DENSE EVIDENCE]", 1)[1].strip() if "[DENSE EVIDENCE]" in frozen["context"] else ""
        hop_depth = int(frozen.get("hop_depth", 1))
        for condition in CONDITIONS:
            key = (question["question_id"], condition)
            if key in rows:
                continue
            started = time.perf_counter()
            answer, trace, telemetry = generate_condition(
                condition=condition,
                question=question["question"],
                current_context=frozen["context"],
                facts=facts,
                options=OPTIONS,
                hop_depth=hop_depth,
                dense_text=dense_text,
            )
            answer = answer or UNKNOWN
            row = {
                "question_id": question["question_id"],
                "question": question["question"],
                "reference_answer": question["reference_answer"],
                "category": question["category"],
                "kg_required": question["kg_required"],
                "split": question["split"],
                "condition": condition,
                "answer": answer,
                "metrics": metrics(answer, question["reference_answer"]),
                "retrieval_source": "frozen relation_aware_kg_rag row; no retrieval rerun",
                "retrieved_kg_paths": frozen["selected_evidence"],
                "provenance": frozen["provenance_ids"],
                "context_statistics": {
                    "kg_path_count": trace.kg_path_count,
                    "kg_fact_count": trace.kg_fact_count,
                    "provenance_count": trace.provenance_count,
                    "kg_token_count": trace.kg_token_count,
                    "dense_token_count": trace.dense_token_count,
                    "bm25_token_count": trace.bm25_token_count,
                    "total_token_count": trace.total_token_count,
                    "token_budget": TOKEN_BUDGET,
                },
                "conflict_classification": trace.conflict_classification,
                "generation_configuration": OPTIONS,
                "trace": {
                    **trace.__dict__,
                    "used_fact_path_ids": list(trace.used_fact_path_ids),
                    "ignored_fact_path_ids": list(trace.ignored_fact_path_ids),
                },
                "generation_telemetry": telemetry,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            with RESULTS.open("a", encoding="utf8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            rows[key] = row
            print(f"checkpointed {len(rows)}/{len(target) * len(CONDITIONS)} {key}", flush=True)

    manifest = {
        "benchmark_sha256": sha256(BENCHMARK),
        "frozen_generation_sha256": sha256(FROZEN_GENERATIONS),
        "generation_results_sha256": sha256(RESULTS),
        "question_count": len(target),
        "expected_rows": len(target) * len(CONDITIONS),
        "conditions": list(CONDITIONS),
        "model": "mistral:7b-instruct-q4_K_M",
        "options": OPTIONS,
        "token_budget": TOKEN_BUDGET,
        "retrieval_policy": "exactly reused frozen relation_aware_kg_rag evidence; no retrieval rerun",
        "development_policy": "All configuration decisions pre-specified; development labels retained for audit only.",
        "seed": 20261008,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"complete": len(rows) == len(target) * len(CONDITIONS), "rows": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    main(parser.parse_args().limit)
