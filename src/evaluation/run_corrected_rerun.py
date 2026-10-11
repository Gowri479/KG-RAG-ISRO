"""Pre-registered corrected-context rerun (reports/preregistration_corrected_rerun.md).

Generates V (vanilla dense RAG), A (A_CURRENT) and C (C_TWO_STAGE) once over the
frozen v2 benchmark and frozen retrieval, with every prompt fitted to the
context window. Refuses to generate unless the pre-registration is approved
and committed.

    python -m src.evaluation.run_corrected_rerun --dry-run   # build prompts only
    python -m src.evaluation.run_corrected_rerun             # generate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.evaluate_relational_qa import metrics
from src.generator import kg_grounded_generator
from src.generator import ollama_api
from src.generator.kg_grounded_generator import extract_dense_text, facts_from_paths, generate_condition
from src.generator.prompt import SYSTEM_PROMPT, build_user_prompt
from src.generator.token_budget import EVIDENCE, fit_evidence, trim_to_tokens

ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = ROOT / "reports" / "preregistration_corrected_rerun.md"
BENCHMARK = ROOT / "data" / "relation_aware_benchmark" / "relation_aware_qa_v2.json"
FROZEN_GENERATIONS = ROOT / "data" / "results" / "relation_aware" / "generation_results.jsonl"
BENCHMARK_SHA256 = "6c7f3600995c0091de73d20d2f02953cb414023df281f7178069f9f81b432b64"
FROZEN_SHA256 = "e13e4c82a3638df15832887621a066b57afea62da5ce8d28394cb18cd608635c"
OUTPUT = ROOT / "data" / "results" / "corrected_rerun"
RESULTS = OUTPUT / "generation_results.jsonl"
SEED = 20261008
OPTIONS = {"temperature": 0.1, "num_predict": 150, "num_ctx": 2048, "seed": SEED}
SYSTEMS = ("V_VANILLA", "A_CURRENT", "C_TWO_STAGE")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()


def check_inputs() -> None:
    if sha256(BENCHMARK) != BENCHMARK_SHA256:
        raise RuntimeError("Benchmark hash differs from the pre-registration")
    if sha256(FROZEN_GENERATIONS) != FROZEN_SHA256:
        raise RuntimeError("Frozen retrieval hash differs from the pre-registration")


def check_preregistration() -> str:
    """Return the commit of the approved pre-registration, or refuse to run."""
    if "DRAFT, awaiting approval" in PREREGISTRATION.read_text(encoding="utf8"):
        raise RuntimeError("Pre-registration is still a draft; approve and commit it first")
    if git("status", "--porcelain", "--", str(PREREGISTRATION.relative_to(ROOT))):
        raise RuntimeError("Pre-registration has uncommitted changes")
    if git("status", "--porcelain", "--untracked-files=no", "--", "src"):
        raise RuntimeError("Source code has uncommitted changes; commit before generating")
    return git("log", "-1", "--format=%H", "--", str(PREREGISTRATION.relative_to(ROOT)))


def load_inputs() -> tuple[list[dict], dict[tuple[str, str], dict]]:
    questions = json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
    frozen: dict[tuple[str, str], dict] = {}
    for line in FROZEN_GENERATIONS.read_text(encoding="utf8").splitlines():
        row = json.loads(line)
        if row["system"] in ("vanilla_dense_rag", "relation_aware_kg_rag"):
            key = (row["question_id"], row["system"])
            if key in frozen:
                raise RuntimeError(f"Duplicate frozen row: {key}")
            frozen[key] = row
    missing = [q["question_id"] for q in questions for s in ("vanilla_dense_rag", "relation_aware_kg_rag") if (q["question_id"], s) not in frozen]
    if missing:
        raise RuntimeError(f"Frozen contexts missing for {missing}")
    return questions, frozen


def generate_system(system: str, question: dict, frozen: dict[tuple[str, str], dict]) -> tuple[str, dict, dict]:
    qid = question["question_id"]
    if system == "V_VANILLA":
        # Same 1,500-token evidence cap as A_CURRENT, applied the same way.
        budget = kg_grounded_generator._evidence_budget(SYSTEM_PROMPT, question["question"], EVIDENCE, OPTIONS)
        context = trim_to_tokens(frozen[(qid, "vanilla_dense_rag")]["context"], budget)
        answer, telemetry = ollama_api.generate_with_metrics(question["question"], context, options=OPTIONS)
        return answer, telemetry, {}
    trace_row = frozen[(qid, "relation_aware_kg_rag")]
    answer, trace, telemetry = generate_condition(
        condition=system,
        question=question["question"],
        current_context=trace_row["context"],
        facts=facts_from_paths(trace_row["selected_evidence"]),
        options=OPTIONS,
        hop_depth=int(trace_row.get("hop_depth", 1)),
        dense_text=extract_dense_text(trace_row["context"]),
    )
    trace_dict = {**trace.__dict__, "used_fact_path_ids": list(trace.used_fact_path_ids), "ignored_fact_path_ids": list(trace.ignored_fact_path_ids)}
    return answer, telemetry, trace_dict


def model_calls(telemetry: dict) -> list[dict]:
    return [telemetry["plan"], telemetry["final"]] if "final" in telemetry else [telemetry]


def integrity(telemetry: dict) -> dict:
    calls = model_calls(telemetry)
    return {
        "calls": len(calls),
        "all_status_ok": all(c.get("status") == "ok" for c in calls),
        "all_prompt_counts_match": all(c.get("prompt_tokens") == c.get("prompt_tokens_expected") for c in calls),
        "any_ollama_truncated": any(c.get("ollama_truncated") for c in calls),
        "any_context_trimmed": any(c.get("context_trimmed") for c in calls),
    }


def dry_run() -> None:
    """Build every prompt without calling the model and summarise evidence fitting."""

    def stub(question, context, options=None, system_prompt=None):
        _, report = fit_evidence(system_prompt or SYSTEM_PROMPT, build_user_prompt(EVIDENCE, question), context, options)
        return "", {**report, "status": "ok", "prompt_tokens": report["prompt_tokens_expected"]}

    ollama_api_generate = ollama_api.generate_with_metrics
    ollama_api.generate_with_metrics = stub
    kg_grounded_generator.generate_with_metrics = stub
    try:
        questions, frozen = load_inputs()
        summary: dict[str, list[dict]] = {s: [] for s in SYSTEMS}
        for question in questions:
            for system in SYSTEMS:
                _, telemetry, _ = generate_system(system, question, frozen)
                summary[system].extend(model_calls(telemetry))
    finally:
        ollama_api.generate_with_metrics = ollama_api_generate
        kg_grounded_generator.generate_with_metrics = ollama_api_generate
    for system, calls in summary.items():
        print(json.dumps({
            "system": system,
            "model_calls": len(calls),
            "max_prompt_tokens": max(c["prompt_tokens_expected"] for c in calls),
            "median_context_tokens_full": statistics.median(c["context_tokens_full"] for c in calls),
            "median_context_tokens_used": statistics.median(c["context_tokens_used"] for c in calls),
            "calls_trimmed_by_window_guard": sum(c["context_trimmed"] for c in calls),
        }))


def read_checkpoint() -> dict[tuple[str, str], dict]:
    if not RESULTS.exists():
        return {}
    rows = {}
    for line in RESULTS.read_text(encoding="utf8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = (row["question_id"], row["system"])
            if key in rows:
                raise RuntimeError(f"Duplicate rerun row: {key}")
            rows[key] = row
    return rows


def main() -> None:
    check_inputs()
    prereg_commit = check_preregistration()
    questions, frozen = load_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = read_checkpoint()
    expected = len(questions) * len(SYSTEMS)
    for question in questions:
        for system in SYSTEMS:
            key = (question["question_id"], system)
            if key in rows:
                continue
            started = time.perf_counter()
            answer, telemetry, trace = generate_system(system, question, frozen)
            check = integrity(telemetry)
            if not check["all_status_ok"]:
                # Pre-registered: one retry for transport failures only.
                print(f"retrying transport failure {key}", flush=True)
                answer, telemetry, trace = generate_system(system, question, frozen)
                check = {**integrity(telemetry), "retried": True}
            row = {
                "question_id": question["question_id"],
                "question": question["question"],
                "reference_answer": question["reference_answer"],
                "category": question["category"],
                "kg_required": question["kg_required"],
                "split": question["split"],
                "system": system,
                "answer": answer,
                "metrics": metrics(answer, question["reference_answer"]),
                "integrity": check,
                "trace": trace,
                "generation_configuration": OPTIONS,
                "generation_telemetry": telemetry,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            with RESULTS.open("a", encoding="utf8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows[key] = row
            print(f"checkpointed {len(rows)}/{expected} {key} integrity_ok={check['all_prompt_counts_match'] and not check['any_ollama_truncated']}", flush=True)

    integrity_failures = [k for k, r in rows.items() if not (r["integrity"]["all_status_ok"] and r["integrity"]["all_prompt_counts_match"] and not r["integrity"]["any_ollama_truncated"])]
    manifest = {
        "preregistration": str(PREREGISTRATION.relative_to(ROOT)),
        "preregistration_commit": prereg_commit,
        "preregistration_sha256": sha256(PREREGISTRATION),
        "code_commit": git("rev-parse", "HEAD"),
        "benchmark_sha256": sha256(BENCHMARK),
        "frozen_retrieval_sha256": sha256(FROZEN_GENERATIONS),
        "generation_results_sha256": sha256(RESULTS),
        "systems": list(SYSTEMS),
        "expected_rows": expected,
        "rows": len(rows),
        "integrity_failures": [list(k) for k in integrity_failures],
        "model": ollama_api.MODEL_NAME,
        "options": OPTIONS,
        "python": sys.version,
        "platform": platform.platform(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"complete": len(rows) == expected, "rows": len(rows), "integrity_failures": len(integrity_failures)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Build and fit every prompt without calling the model")
    dry_run() if parser.parse_args().dry_run else main()
