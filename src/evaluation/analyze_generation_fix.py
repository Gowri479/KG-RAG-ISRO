"""Analyze the isolated generation-fix experiment without modifying prior results."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "results" / "generation_fix"
RESULTS = OUTPUT / "generation_results.jsonl"
BENCHMARK = ROOT / "data" / "relation_aware_benchmark" / "relation_aware_qa_v2.json"
FROZEN = ROOT / "data" / "results" / "relation_aware" / "generation_results.jsonl"
CONDITIONS = ("A_CURRENT", "B_STRUCTURED", "C_TWO_STAGE")
SEED = 20261008


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows() -> dict[tuple[str, str], dict]:
    rows = {}
    for line in RESULTS.read_text(encoding="utf8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = (row["question_id"], row["condition"])
            if key in rows:
                raise RuntimeError(f"Duplicate row {key}")
            rows[key] = row
    return rows


def paired(rows: dict, condition: str, baseline: str, field: str, question_ids: list[str]) -> list[float]:
    return [
        float(rows[(qid, condition)]["metrics"][field]) - float(rows[(qid, baseline)]["metrics"][field])
        for qid in question_ids
    ]


def bootstrap(values: list[float], seed: int = SEED, resamples: int = 10000) -> list[float]:
    if not values:
        return [None, None]
    rng = np.random.default_rng(seed)
    sample = rng.choice(np.asarray(values, dtype=float), size=(resamples, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))]


def wilcoxon(values: list[float]) -> dict:
    nonzero = [v for v in values if v != 0]
    if not nonzero:
        return {"statistic": 0.0, "p_value": 1.0}
    try:
        from scipy.stats import wilcoxon
        result = wilcoxon(values, zero_method="wilcox", alternative="two-sided", method="auto")
        return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
    except Exception as exc:
        return {"statistic": None, "p_value": None, "error": type(exc).__name__}


def comparison(values: list[float]) -> dict:
    wins = sum(v > 0 for v in values)
    losses = sum(v < 0 for v in values)
    ties = len(values) - wins - losses
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean_difference": float(mean(values)) if values else None,
        "median_difference": float(median(values)) if values else None,
        "bootstrap_ci95": bootstrap(values),
        "wilcoxon": wilcoxon(values),
        "paired_dz": float(mean(values) / sd) if sd else None,
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def holm(p_values: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted(((key, value) for key, value in p_values.items() if value is not None), key=lambda item: item[1])
    adjusted = {}
    previous = 0.0
    total = len(valid)
    for index, (key, value) in enumerate(valid):
        adjusted_value = min(1.0, max(previous, (total - index) * value))
        adjusted[key] = adjusted_value
        previous = adjusted_value
    return {key: adjusted.get(key) for key in p_values}


def failure_classification(a: dict, b_or_c: dict) -> str:
    trace = b_or_c["trace"]
    if not a["trace"]["used_fact_path_ids"] and trace["kg_fact_count"]:
        return "Evidence dilution"
    if trace["conflict_classification"] == "DIRECT_CONTRADICTION":
        return "Contradiction handling failure"
    if trace["unsupported_claim_proxy"]:
        return "Unsupported generation"
    if trace["hop_depth"] >= 2 and not trace["plan"] and b_or_c["condition"] == "C_TWO_STAGE":
        return "Answer-planning failure"
    if trace["provenance_count"] == 0:
        return "Provenance loss"
    return "Other"


def main() -> None:
    rows = load_rows()
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
    required = [q for q in benchmark if q["kg_required"] == "YES"]
    ids = [q["question_id"] for q in required]
    expected = {(qid, condition) for qid in ids for condition in CONDITIONS}
    if set(rows) != expected:
        raise RuntimeError(f"Expected exactly 180 A/B/C rows, found {len(rows)}")

    category_results = {}
    statistics = {}
    p_values = {}
    for category in sorted({q["category"] for q in required}):
        category_ids = [q["question_id"] for q in required if q["category"] == category]
        category_results[category] = {"n": len(category_ids), "conditions": {}}
        for condition in CONDITIONS:
            values = [rows[(qid, condition)]["metrics"] for qid in category_ids]
            category_results[category]["conditions"][condition] = {
                metric: float(mean(item[metric] for item in values))
                for metric in ("rouge_l", "reference_token_coverage", "exact_match", "idk")
            }
        for condition in CONDITIONS[1:]:
            for metric in ("rouge_l", "reference_token_coverage"):
                key = f"{category}:{condition}:{metric}"
                result = comparison(paired(rows, condition, "A_CURRENT", metric, category_ids))
                statistics[key] = result
                p_values[key] = result["wilcoxon"]["p_value"]
    adjusted = holm(p_values)
    for key, value in adjusted.items():
        statistics[key]["holm_adjusted_p_value"] = value

    overall = {}
    for condition in CONDITIONS:
        values = [rows[(qid, condition)]["metrics"] for qid in ids]
        overall[condition] = {
            metric: float(mean(item[metric] for item in values))
            for metric in ("rouge_l", "reference_token_coverage", "exact_match", "idk")
        }
    overall_comparisons = {}
    overall_p = {}
    for condition in CONDITIONS[1:]:
        for metric in ("rouge_l", "reference_token_coverage"):
            key = f"{condition}:{metric}"
            result = comparison(paired(rows, condition, "A_CURRENT", metric, ids))
            overall_comparisons[key] = result
            overall_p[key] = result["wilcoxon"]["p_value"]
    overall_holm = holm(overall_p)
    for key, value in overall_holm.items():
        overall_comparisons[key]["holm_adjusted_p_value"] = value

    failures = []
    failure_counts = Counter()
    for qid in ids:
        for condition in CONDITIONS[1:]:
            a = rows[(qid, "A_CURRENT")]
            current = rows[(qid, condition)]
            if current["metrics"]["rouge_l"] < a["metrics"]["rouge_l"] or current["metrics"]["reference_token_coverage"] < a["metrics"]["reference_token_coverage"]:
                kind = failure_classification(a, current)
                failure_counts[kind] += 1
                failures.append({
                    "question_id": qid,
                    "category": current["category"],
                    "condition": condition,
                    "failure_category": kind,
                    "rouge_l_difference": current["metrics"]["rouge_l"] - a["metrics"]["rouge_l"],
                    "coverage_difference": current["metrics"]["reference_token_coverage"] - a["metrics"]["reference_token_coverage"],
                    "conflict_classification": current["conflict_classification"],
                    "unsupported_claim_proxy": current["trace"]["unsupported_claim_proxy"],
                })

    decision = "NO SUPPORT"
    c_coverage = overall_comparisons["C_TWO_STAGE:reference_token_coverage"]
    relational_categories = ("SINGLE_RELATION", "TWO_HOP_RELATION", "MULTI_RELATION")
    c_relational_coverage = [
        statistics.get(f"{category}:C_TWO_STAGE:reference_token_coverage", {})
        for category in relational_categories
    ]
    strong_support = (
        c_coverage.get("mean_difference", 0) > 0
        and c_coverage.get("bootstrap_ci95", [0, 0])[0] > 0
        and all(item.get("mean_difference", 0) > 0 for item in c_relational_coverage)
        and all(item.get("bootstrap_ci95", [0, 0])[0] > 0 for item in c_relational_coverage)
        and overall["C_TWO_STAGE"]["idk"] <= overall["A_CURRENT"]["idk"]
    )
    if strong_support:
        decision = "STRONG SUPPORT"
    elif any(item["mean_difference"] > 0 for item in overall_comparisons.values()):
        decision = "PARTIAL SUPPORT"

    analysis = {
        "integrity": {
            "benchmark_sha256": sha256(BENCHMARK),
            "frozen_generation_sha256": sha256(FROZEN),
            "question_count": len(ids),
            "row_count": len(rows),
            "conditions": list(CONDITIONS),
            "retrieval_frozen": True,
        },
        "overall": overall,
        "overall_comparisons": overall_comparisons,
        "category_results": category_results,
        "statistics": statistics,
        "failure_counts": dict(failure_counts),
        "failure_count": len(failures),
        "decision": decision,
        "human_evaluation_triggered": decision in {"STRONG SUPPORT", "PARTIAL SUPPORT"},
        "decision_rule": "Strong support requires positive paired coverage in every relational category, positive bootstrap intervals, and no increased abstention; mixed automated improvement is partial support; otherwise no support.",
        "limitations": ["Automated lexical metrics do not establish factual quality.", "Failure categories are trace-based diagnostics, not causal labels."],
    }
    analysis_path = OUTPUT / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    integrity = {
        "benchmark_sha256": sha256(BENCHMARK),
        "frozen_generation_sha256": sha256(FROZEN),
        "generation_results_sha256": sha256(RESULTS),
        "analysis_sha256": sha256(analysis_path),
        "question_count": len(ids),
        "row_count": len(rows),
        "unique_question_condition_pairs": len(set(rows)),
        "conditions": list(CONDITIONS),
        "retrieval_rerun": False,
    }
    (OUTPUT / "integrity_manifest.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf8")
    (OUTPUT / "statistical_results.json").write_text(json.dumps({"overall": overall_comparisons, "category": statistics}, indent=2) + "\n", encoding="utf8")
    with (OUTPUT / "failure_analysis.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(failures[0]) if failures else ["question_id"])
        writer.writeheader()
        writer.writerows(failures)
    with (OUTPUT / "question_level_results.csv").open("w", newline="", encoding="utf8") as handle:
        fields = ["question_id", "category", "condition", "answer", "rouge_l", "reference_token_coverage", "exact_match", "idk", "conflict_classification"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for qid in ids:
            for condition in CONDITIONS:
                row = rows[(qid, condition)]
                writer.writerow({"question_id": qid, "category": row["category"], "condition": condition, "answer": row["answer"], **row["metrics"], "conflict_classification": row["conflict_classification"]})

    report = [
        "# Generation-fix analysis",
        "",
        "## Scope and integrity",
        "",
        f"- 60 KG-required questions and 180 A/B/C rows were verified.",
        "- Retrieval was reused exactly from the frozen Relation-Aware KG-RAG traces; no retrieval experiment was rerun.",
        f"- Frozen benchmark SHA-256: `{analysis['integrity']['benchmark_sha256']}`",
        f"- Frozen prior-generation SHA-256: `{analysis['integrity']['frozen_generation_sha256']}`",
        "",
        "## Overall automated results",
        "",
        "| Condition | ROUGE-L | Coverage | Exact match | IDK |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        item = overall[condition]
        report.append(f"| {condition} | {item['rouge_l']:.3f} | {item['reference_token_coverage']:.3f} | {item['exact_match']:.3f} | {item['idk']:.3f} |")
    report += ["", "## Category results", "", "| Category | N | A ROUGE-L | B ROUGE-L | C ROUGE-L | A coverage | B coverage | C coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for category, item in category_results.items():
        c = item["conditions"]
        report.append(f"| {category} | {item['n']} | {c['A_CURRENT']['rouge_l']:.3f} | {c['B_STRUCTURED']['rouge_l']:.3f} | {c['C_TWO_STAGE']['rouge_l']:.3f} | {c['A_CURRENT']['reference_token_coverage']:.3f} | {c['B_STRUCTURED']['reference_token_coverage']:.3f} | {c['C_TWO_STAGE']['reference_token_coverage']:.3f} |")
    report += ["", "## Failure analysis", "", f"- Failures where B or C lost on at least one lexical metric: {len(failures)}", f"- Categories: `{dict(failure_counts)}`", "", "## Decision", "", f"**{decision}**", "", "Human evaluation was " + ("triggered for consideration because a meaningful automated improvement was observed." if analysis["human_evaluation_triggered"] else "not triggered because the decision rule found no meaningful automated improvement requiring human follow-up."), "", "Automated metrics remain lexical diagnostics and do not establish factual quality."]
    (OUTPUT / "analysis.md").write_text("\n".join(report) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
