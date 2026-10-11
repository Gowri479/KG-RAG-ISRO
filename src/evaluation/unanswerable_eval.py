"""Evaluate grounded abstention and unanswerable QA robustness.

Compares BM25 + LLM, Vanilla RAG, and KG-RAG across 20 stratified unanswerable questions
(4 questions in each of 5 failure categories):
1. Completely absent fact
2. Related entity but absent relationship
3. Similar entity with misleading evidence
4. Temporal fact absent from corpus
5. Plausible but unsupported question

Computes:
- Strict IDK Rate (containing 'I don't know')
- Grounded Abstention Rate (strict IDK or explicit non-evidence declaration)
- Unsupported Answer Rate (asserting unsubstantiated factual claims)
- Category breakdown across all 5 failure modes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.bm25_llm import retrieve_context as bm25_retrieve
from src.baselines.vanilla_rag import retrieve_context as vanilla_retrieve
from src.evaluation.analysis_utils import is_idk
from src.generator.ollama_api import generate_with_metrics
from src.retriever.hybrid import retrieve as hybrid_retrieve

BENCHMARK_PATH = ROOT / "data" / "benchmark" / "unanswerable_qa.json"
CHECKPOINT_PATH = ROOT / "data" / "results" / "unanswerable_eval_checkpoint.jsonl"
RESULTS_JSON = ROOT / "data" / "results" / "unanswerable_eval_results.json"
REPORT_MD = ROOT / "data" / "results" / "unanswerable_eval_report.md"

ABSTENTION_PHRASES = [
    "i don't know",
    "i do not know",
    "don't know",
    "does not provide",
    "does not mention",
    "does not contain",
    "not mentioned",
    "not provided",
    "no information",
    "cannot be answered",
    "not available in the context",
    "not found in the context",
    "is not in the provided text",
]


def is_grounded_abstention(text: str) -> bool:
    t = (text or "").lower().strip()
    if not t:
        return True
    return any(p in t for p in ABSTENTION_PHRASES)


def _cap_words(text: str, max_words: int = 350) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def load_checkpoints() -> dict[tuple[str, str], dict]:
    completed = {}
    if CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    completed[(item["question_id"], item["system"])] = item
    return completed


def append_checkpoint(item: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item) + "\n")


def run_evaluation(per_category: int = 4) -> None:
    all_questions = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    cats = sorted(list({q["category"] for q in all_questions}))

    selected_questions = []
    for c in cats:
        c_qs = [q for q in all_questions if q["category"] == c][:per_category]
        selected_questions.extend(c_qs)

    checkpoints = load_checkpoints()

    systems = {
        "bm25_llm": lambda q: _cap_words(bm25_retrieve(q, top_k=3), 350),
        "vanilla_rag": lambda q: _cap_words(vanilla_retrieve(q, top_k=3), 350),
        "kg_rag": lambda q: hybrid_retrieve(q, passage_limit=3, max_tokens=600),
    }

    all_results = list(checkpoints.values())
    total_runs = len(selected_questions) * len(systems)

    print(f"Starting unanswerable evaluation: {len(selected_questions)} questions ({per_category}/cat) x 3 systems = {total_runs} runs.", flush=True)
    print(f"Existing valid checkpoints: {len(checkpoints)}", flush=True)

    for idx, q_item in enumerate(selected_questions, start=1):
        qid = q_item["id"]
        q_text = q_item["question"]
        cat = q_item["category"]

        for sys_name, ret_fn in systems.items():
            if (qid, sys_name) in checkpoints:
                continue

            # Retrieve context
            context = ret_fn(q_text)

            # Generate
            t0 = time.perf_counter()
            ans, metrics = generate_with_metrics(q_text, context)
            latency = (time.perf_counter() - t0) * 1000

            strict_idk = is_idk(ans)
            abstain = is_grounded_abstention(ans)
            unsupported = not abstain

            record = {
                "question_id": qid,
                "question": q_text,
                "category": cat,
                "unanswerable_reason": q_item["unanswerable_reason"],
                "system": sys_name,
                "retrieved_context_length": len(context),
                "generated_answer": ans,
                "strict_idk": strict_idk,
                "grounded_abstention": abstain,
                "unsupported_answer": unsupported,
                "latency_ms": round(latency, 2),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

            append_checkpoint(record)
            checkpoints[(qid, sys_name)] = record
            all_results.append(record)
            print(f"[{len(checkpoints)}/{total_runs}] {qid} | {sys_name} | Abstain: {abstain} | Strict IDK: {strict_idk} ({latency:.1f}ms)", flush=True)

    # Filter to only the selected questions
    sel_ids = {q["id"] for q in selected_questions}
    eval_results = [r for r in all_results if r["question_id"] in sel_ids]

    summary = {}
    for sys_name in systems:
        sys_records = [r for r in eval_results if r["system"] == sys_name]
        n_total = len(sys_records)
        n_abstain = sum(1 for r in sys_records if r.get("grounded_abstention", r.get("correct_abstention")))
        n_strict_idk = sum(1 for r in sys_records if r.get("strict_idk", False))
        n_unsupported = sum(1 for r in sys_records if r.get("unsupported_answer", not r.get("grounded_abstention")))

        cat_breakdown = {}
        for c in cats:
            c_records = [r for r in sys_records if r["category"] == c]
            c_abstain = sum(1 for r in c_records if r.get("grounded_abstention", r.get("correct_abstention")))
            cat_breakdown[c] = {
                "n_questions": len(c_records),
                "grounded_abstention_rate": round(c_abstain / len(c_records), 4) if c_records else 0.0,
                "unsupported_answer_rate": round((len(c_records) - c_abstain) / len(c_records), 4) if c_records else 0.0,
            }

        summary[sys_name] = {
            "n_questions": n_total,
            "grounded_abstention_rate": round(n_abstain / n_total, 4) if n_total else 0.0,
            "strict_idk_rate": round(n_strict_idk / n_total, 4) if n_total else 0.0,
            "unsupported_answer_rate": round(n_unsupported / n_total, 4) if n_total else 0.0,
            "mean_latency_ms": round(mean(r["latency_ms"] for r in sys_records), 2) if n_total else 0.0,
            "category_breakdown": cat_breakdown,
        }

    out_payload = {
        "experiment": "unanswerable_abstention_evaluation",
        "benchmark_file": str(BENCHMARK_PATH.relative_to(ROOT)),
        "n_questions_evaluated": len(selected_questions),
        "total_benchmark_questions": len(all_questions),
        "systems": summary,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_JSON.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    # Generate Markdown Report
    lines = [
        "# Unanswerable QA & Grounded Abstention Evaluation Report",
        "",
        f"**Benchmark Dataset:** 40 curated unanswerable questions ({len(selected_questions)} evaluated: {per_category}/category across 5 failure categories).",
        "**Target Systems:** BM25 + LLM, Vanilla RAG, KG-RAG (Mistral-7B-Instruct).",
        "",
        "## Summary Metrics",
        "",
        "| System | Grounded Abstention Rate | Strict IDK Rate | Unsupported Answer Rate | Mean Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for sys_name, data in summary.items():
        lines.append(
            f"| **{sys_name}** | {data['grounded_abstention_rate']*100:.2f}% | "
            f"{data['strict_idk_rate']*100:.2f}% | {data['unsupported_answer_rate']*100:.2f}% | {data['mean_latency_ms']:.1f} |"
        )

    lines.extend([
        "",
        "## Category-Wise Grounded Abstention Rates",
        "",
        "| Failure Category | BM25 + LLM | Vanilla RAG | KG-RAG |",
        "| :--- | :---: | :---: | :---: |",
    ])
    for c in cats:
        b_rate = summary["bm25_llm"]["category_breakdown"][c]["grounded_abstention_rate"] * 100
        v_rate = summary["vanilla_rag"]["category_breakdown"][c]["grounded_abstention_rate"] * 100
        k_rate = summary["kg_rag"]["category_breakdown"][c]["grounded_abstention_rate"] * 100
        lines.append(f"| {c} | {b_rate:.1f}% | {v_rate:.1f}% | {k_rate:.1f}% |")

    lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nEvaluation complete! Saved results to {RESULTS_JSON} and report to {REPORT_MD}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=4, help="Questions per unanswerable category")
    args = parser.parse_args()
    run_evaluation(per_category=args.per_category)
