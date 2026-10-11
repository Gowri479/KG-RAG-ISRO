"""Evaluate the 36-question Aditya-L1 benchmark across BM25+LLM, Vanilla RAG, and KG-RAG."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.baselines.bm25_llm import _load_bm25, retrieve_context as bm25_context
from src.baselines.vanilla_rag import retrieve_context as vanilla_context
from src.evaluation.evaluate import (
    exact_match,
    is_idk,
    reference_token_coverage,
    rouge_l,
)
from src.generator.ollama_api import generate
from src.retriever import hybrid, kg_retriever
from src.retriever.embedding_model import get_embedding_model
from src.retriever.faiss_retriever import (
    _load_chunks as _load_faiss_chunks,
    _load_index as _load_faiss_index,
)
from src.retriever.hybrid import retrieve as kgrag_context

BENCHMARK_PATH = ROOT / "data" / "benchmark" / "aditya_l1_optional_qa.json"
DEFAULT_CHECKPOINT = ROOT / "data" / "results" / "aditya_l1_benchmark_checkpoint.jsonl"
DEFAULT_ANSWERS = ROOT / "data" / "results" / "answers_aditya_l1.json"
DEFAULT_RESULTS = ROOT / "data" / "results" / "aditya_l1_benchmark_results.json"
DEFAULT_REPORT = ROOT / "data" / "results" / "aditya_l1_benchmark_report.md"

OPTIONS = {"num_predict": 150, "temperature": 0.1, "num_ctx": 2048}
UNKNOWN = "I don't know."

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and", "or", "in", "on", "at",
    "for", "from", "by", "with", "what", "which", "when", "where", "how", "did", "does", "do", "has",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2 and t not in STOPWORDS}


def _assign_reasoning_type(category: str) -> str:
    cat = (category or "").lower()
    if "mission facts" in cat:
        return "Type A: Direct fact lookup"
    if "multi-hop" in cat:
        return "Type C: Multiple related entities / multi-hop"
    if "temporal" in cat or "trajectory/time" in cat:
        return "Type D: Temporal relationship"
    if "payload" in cat or "launch" in cat:
        return "Type B: One explicit relationship"
    return "Type A: Direct fact lookup"


def _prewarm() -> None:
    _load_bm25()
    _load_faiss_chunks()
    _load_faiss_index()
    get_embedding_model()
    kg_retriever._load_graph()
    hybrid._get_nlp()


def _limit_context(context: str, max_tokens: int = 1500) -> str:
    return " ".join((context or "").split()[:max_tokens])


def _run_question(item: dict) -> dict:
    started = time.perf_counter()
    question = item["question"]

    # Extract KG metadata for provenance and analysis
    keywords = hybrid._query_keywords(question)
    entities = [
        ent for ent in hybrid._extract_entities(question)
        if any(kw in ent.lower() for kw in keywords)
    ]
    kg_ctx = kg_retriever.get_kg_context(entities)
    G = kg_retriever._load_graph()
    kg_nodes_matched = [e for e in entities if e in G]

    bm_ctx = bm25_context(question, top_k=3)
    van_ctx = vanilla_context(question, top_k=3)
    kg_ctx_full = kgrag_context(question, passage_limit=3, max_tokens=1500)

    contexts = {
        "bm25_llm": bm_ctx,
        "vanilla_rag": van_ctx,
        "kg_rag": kg_ctx_full,
    }

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            sys_name: executor.submit(generate, question, _limit_context(ctx), options=OPTIONS)
            for sys_name, ctx in contexts.items()
        }
        answers = {sys_name: (fut.result() or UNKNOWN) for sys_name, fut in futures.items()}

    ref = item.get("answer", "")
    metrics_per_sys = {}
    for sys_name, ans in answers.items():
        if is_idk(ans):
            metrics_per_sys[sys_name] = {
                "rouge_l": 0.0,
                "reference_token_coverage": 0.0,
                "exact_match": 0.0,
                "idk": 1.0,
            }
        else:
            metrics_per_sys[sys_name] = {
                "rouge_l": round(rouge_l(ans, ref), 4),
                "reference_token_coverage": round(reference_token_coverage(ans, ref), 4),
                "exact_match": exact_match(ans, ref),
                "idk": 0.0,
            }

    return {
        "id": item["id"],
        "question": question,
        "reference_answer": ref,
        "category": item.get("category", "unclassified"),
        "reasoning_type": _assign_reasoning_type(item.get("category", "")),
        "source_document_id": item.get("source_document_id"),
        "source_url": item.get("source_url"),
        "page": item.get("page"),
        "bm25_answer": answers["bm25_llm"],
        "vanilla_rag_answer": answers["vanilla_rag"],
        "kgrag_answer": answers["kg_rag"],
        "metrics": metrics_per_sys,
        "contexts": {
            "bm25": _limit_context(bm_ctx, 300),
            "vanilla_rag": _limit_context(van_ctx, 300),
            "kg_rag": _limit_context(kg_ctx_full, 300),
            "kg_triples": kg_ctx,
        },
        "kg_analysis": {
            "query_entities_extracted": entities,
            "graph_nodes_matched": kg_nodes_matched,
            "kg_context_non_empty": bool(kg_ctx.strip()),
        },
        "checkpoint_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_wall_time_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def _read_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["id"]] = row
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _bootstrap_ci(diffs: list[float], n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    if not diffs:
        return 0.0, 0.0
    import random
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        sample = [rng.choice(diffs) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low = means[int(0.025 * n_boot)]
    high = means[int(0.975 * n_boot)]
    return round(low, 4), round(high, 4)


def _wilcoxon_signed_rank(diffs: list[float]) -> dict:
    non_zero = [d for d in diffs if abs(d) > 1e-9]
    if len(non_zero) < 5:
        return {
            "n_non_zero": len(non_zero),
            "reliable": False,
            "note": "Too few non-zero paired differences for asymptotic Wilcoxon test",
            "p_value": None,
        }
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(non_zero)
        return {
            "n_non_zero": len(non_zero),
            "reliable": True,
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
        }
    except Exception as exc:
        return {
            "n_non_zero": len(non_zero),
            "reliable": False,
            "note": str(exc),
            "p_value": None,
        }


def _compute_aggregate_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}

    systems = ("bm25_llm", "vanilla_rag", "kg_rag")
    agg = {}
    for sys_name in systems:
        rouge_scores = [r["metrics"][sys_name]["rouge_l"] for r in rows]
        cov_scores = [r["metrics"][sys_name]["reference_token_coverage"] for r in rows]
        em_scores = [r["metrics"][sys_name]["exact_match"] for r in rows]
        idk_scores = [r["metrics"][sys_name]["idk"] for r in rows]

        agg[sys_name] = {
            "n": n,
            "rouge_l": round(sum(rouge_scores) / n, 4),
            "reference_token_coverage": round(sum(cov_scores) / n, 4),
            "exact_match": round(sum(em_scores) / n, 4),
            "idk_rate": round(sum(idk_scores) / n, 4),
        }
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 36-question Aditya-L1 benchmark.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--answers-output", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    checkpoint_rows = _read_jsonl(args.checkpoint)

    print(f"Loaded {len(benchmark)} Aditya-L1 benchmark questions.")
    print(f"Existing checkpoint rows: {len(checkpoint_rows)} / {len(benchmark)}")

    _prewarm()

    pending = [q for q in benchmark if q["id"] not in checkpoint_rows]
    for idx, item in enumerate(pending, start=len(checkpoint_rows) + 1):
        row = _run_question(item)
        _append_jsonl(args.checkpoint, row)
        checkpoint_rows[row["id"]] = row
        print(f"[{idx}/{len(benchmark)}] Checkpointed {row['id']} ({row['category']}) in {row['question_wall_time_ms']} ms", flush=True)

    ordered_rows = [checkpoint_rows[q["id"]] for q in benchmark]

    # Save answers_aditya_l1.json
    raw_answers = []
    for r in ordered_rows:
        for sys_name, ans in [("bm25_llm", r["bm25_answer"]), ("vanilla_rag", r["vanilla_rag_answer"]), ("kg_rag", r["kgrag_answer"])]:
            raw_answers.append({
                "question_id": r["id"],
                "question": r["question"],
                "reference_answer": r["reference_answer"],
                "category": r["category"],
                "reasoning_type": r["reasoning_type"],
                "source_document_id": r["source_document_id"],
                "source_url": r["source_url"],
                "page": r["page"],
                "system": sys_name,
                "generated_answer": ans,
                "retrieved_context_snippet": r["contexts"][sys_name.replace("_llm", "") if sys_name != "vanilla_rag" else "vanilla_rag"],
                "metrics": r["metrics"][sys_name],
            })
    args.answers_output.parent.mkdir(parents=True, exist_ok=True)
    args.answers_output.write_text(json.dumps(raw_answers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Aggregate overall metrics
    overall_metrics = _compute_aggregate_metrics(ordered_rows)

    # Per-category metrics
    categories = sorted(list({r["category"] for r in ordered_rows}))
    per_category = {}
    for cat in categories:
        cat_rows = [r for r in ordered_rows if r["category"] == cat]
        per_category[cat] = _compute_aggregate_metrics(cat_rows)

    # Per-reasoning-type metrics
    reasoning_types = sorted(list({r["reasoning_type"] for r in ordered_rows}))
    per_reasoning = {}
    for rtype in reasoning_types:
        rtype_rows = [r for r in ordered_rows if r["reasoning_type"] == rtype]
        per_reasoning[rtype] = _compute_aggregate_metrics(rtype_rows)

    # Statistical tests: paired differences
    stat_tests = {}
    for metric in ("rouge_l", "reference_token_coverage", "exact_match", "idk"):
        kg_vals = [r["metrics"]["kg_rag"][metric] for r in ordered_rows]
        bm_vals = [r["metrics"]["bm25_llm"][metric] for r in ordered_rows]
        van_vals = [r["metrics"]["vanilla_rag"][metric] for r in ordered_rows]

        diff_kg_vs_bm = [k - b for k, b in zip(kg_vals, bm_vals)]
        diff_kg_vs_van = [k - v for k, v in zip(kg_vals, van_vals)]

        ci_bm = _bootstrap_ci(diff_kg_vs_bm)
        ci_van = _bootstrap_ci(diff_kg_vs_van)
        w_bm = _wilcoxon_signed_rank(diff_kg_vs_bm)
        w_van = _wilcoxon_signed_rank(diff_kg_vs_van)

        stat_tests[metric] = {
            "kg_vs_bm25": {
                "mean_paired_difference": round(sum(diff_kg_vs_bm) / len(diff_kg_vs_bm), 4),
                "bootstrap_95_ci": ci_bm,
                "wilcoxon": w_bm,
            },
            "kg_vs_vanilla_rag": {
                "mean_paired_difference": round(sum(diff_kg_vs_van) / len(diff_kg_vs_van), 4),
                "bootstrap_95_ci": ci_van,
                "wilcoxon": w_van,
            },
        }

    # KG path and retrieval analysis
    kg_entity_hits = sum(1 for r in ordered_rows if r["kg_analysis"]["graph_nodes_matched"])
    kg_ctx_hits = sum(1 for r in ordered_rows if r["kg_analysis"]["kg_context_non_empty"])

    final_results = {
        "benchmark_metadata": {
            "benchmark_file": str(BENCHMARK_PATH.relative_to(ROOT)),
            "benchmark_size": len(benchmark),
            "valid_questions": len(benchmark),
            "excluded_questions": 0,
            "categories": categories,
            "reasoning_types": reasoning_types,
        },
        "overall_metrics": overall_metrics,
        "per_category_metrics": per_category,
        "per_reasoning_type_metrics": per_reasoning,
        "paired_statistical_tests": stat_tests,
        "kg_path_analysis": {
            "total_questions": len(benchmark),
            "questions_with_graph_node_match": kg_entity_hits,
            "questions_with_non_empty_kg_context": kg_ctx_hits,
            "kg_coverage_fraction": round(kg_ctx_hits / len(benchmark), 4),
        },
        "limitations": [
            "Sample size is N=36; exact match and non-zero differences have small sample support.",
            "Metrics are lexical (ROUGE-L, token coverage, EM, IDK) rather than human-adjudicated correctness.",
            "All systems use local 4-bit quantized Mistral-7b under identical prompt and temperature constraints.",
        ],
    }

    args.results_output.write_text(json.dumps(final_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Generate Markdown Report
    report_lines = [
        "# Aditya-L1 Benchmark Evaluation Report: Knowledge Graph Benefit Analysis",
        "",
        f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "**Dataset**: 36-question targeted Aditya-L1 benchmark (`data/benchmark/aditya_l1_optional_qa.json`)",
        "**Backends**: Local Ollama (`mistral:7b-instruct-q4_K_M`), deterministic temp=0.1, max_tokens=1500 context",
        "",
        "## 1. Overall System Results (N=36)",
        "",
        "| System | ROUGE-L | Ref Token Coverage | Exact Match | IDK Rate |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for sname in ("bm25_llm", "vanilla_rag", "kg_rag"):
        m = overall_metrics[sname]
        report_lines.append(f"| **{sname}** | {m['rouge_l']:.4f} | {m['reference_token_coverage']:.4f} | {m['exact_match']:.4f} | {m['idk_rate']:.4f} |")

    report_lines.extend([
        "",
        "## 2. Targeted Reasoning Type Breakdown",
        "",
        "| Reasoning Type | N | BM25 ROUGE-L | Vanilla ROUGE-L | KG-RAG ROUGE-L | KG vs Van Diff |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for rtype in reasoning_types:
        cnt = len([r for r in ordered_rows if r["reasoning_type"] == rtype])
        m_bm = per_reasoning[rtype]["bm25_llm"]["rouge_l"]
        m_van = per_reasoning[rtype]["vanilla_rag"]["rouge_l"]
        m_kg = per_reasoning[rtype]["kg_rag"]["rouge_l"]
        diff = round(m_kg - m_van, 4)
        report_lines.append(f"| {rtype} | {cnt} | {m_bm:.4f} | {m_van:.4f} | {m_kg:.4f} | {diff:+.4f} |")

    report_lines.extend([
        "",
        "## 3. Paired Statistical Tests (KG-RAG vs Baselines)",
        "",
        "- **ROUGE-L**:",
        f"  - KG-RAG vs Vanilla RAG: Mean Diff = {stat_tests['rouge_l']['kg_vs_vanilla_rag']['mean_paired_difference']:+.4f}, 95% Bootstrap CI = {stat_tests['rouge_l']['kg_vs_vanilla_rag']['bootstrap_95_ci']}",
        f"  - KG-RAG vs BM25: Mean Diff = {stat_tests['rouge_l']['kg_vs_bm25']['mean_paired_difference']:+.4f}, 95% Bootstrap CI = {stat_tests['rouge_l']['kg_vs_bm25']['bootstrap_95_ci']}",
        "- **Reference Token Coverage**:",
        f"  - KG-RAG vs Vanilla RAG: Mean Diff = {stat_tests['reference_token_coverage']['kg_vs_vanilla_rag']['mean_paired_difference']:+.4f}, 95% Bootstrap CI = {stat_tests['reference_token_coverage']['kg_vs_vanilla_rag']['bootstrap_95_ci']}",
        f"  - KG-RAG vs BM25: Mean Diff = {stat_tests['reference_token_coverage']['kg_vs_bm25']['mean_paired_difference']:+.4f}, 95% Bootstrap CI = {stat_tests['reference_token_coverage']['kg_vs_bm25']['bootstrap_95_ci']}",
        "",
        "## 4. KG Path & Evidence Retrieval Analysis",
        "",
        f"- Total Aditya-L1 Questions: {len(benchmark)}",
        f"- Questions matching KG entity nodes: {kg_entity_hits} ({kg_entity_hits/len(benchmark)*100:.1f}%)",
        f"- Questions retrieving non-empty serialized KG triples: {kg_ctx_hits} ({kg_ctx_hits/len(benchmark)*100:.1f}%)",
        "",
        "## 5. Limitations & Scientific Caveats",
        "",
        "1. **Sample Size**: N=36 provides an indicative benchmark for the newly added domain, but statistical power for fine-grained significance testing is limited.",
        "2. **Lexical Proxy**: Evaluation relies on lexical overlap metrics (ROUGE-L and reference token coverage), which can penalize semantically valid answers phrased differently.",
        "3. **Local LLM Ceiling**: When the KG supplies accurate triples, generation failures (hallucination or generic abstention by 7B quantized model) occasionally bottleneck end-to-end performance.",
    ])

    args.report_output.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nFinal evaluation saved to {args.results_output}")
    print(f"Answers saved to {args.answers_output}")
    print(f"Report saved to {args.report_output}")


if __name__ == "__main__":
    main()
