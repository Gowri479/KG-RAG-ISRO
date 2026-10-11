"""Recompute frozen generation-fix results by the benchmark's stored split."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "relation_aware_benchmark" / "relation_aware_qa_v2.json"
FIXED = ROOT / "data" / "results" / "generation_fix" / "generation_results.jsonl"
FROZEN = ROOT / "data" / "results" / "relation_aware" / "generation_results.jsonl"
OUT = ROOT / "data" / "results" / "generation_fix"
CONDITIONS = ("A_CURRENT", "B_STRUCTURED", "C_TWO_STAGE")
SEED = 20261008


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(rows: dict, condition: str, metric: str, ids: list[str]) -> list[float]:
    return [
        float(rows[(qid, condition)]["metrics"][metric]) - float(rows[(qid, "A_CURRENT")]["metrics"][metric])
        for qid in ids
    ]


def bootstrap(values: list[float], resamples: int = 10000) -> list[float | None]:
    if not values:
        return [None, None]
    rng = np.random.default_rng(SEED)
    sample = rng.choice(np.asarray(values), size=(resamples, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))]


def signed_rank(values: list[float]) -> dict:
    if not any(values):
        return {"statistic": 0.0, "p_value": 1.0}
    try:
        from scipy.stats import wilcoxon
        result = wilcoxon(values, zero_method="wilcox", alternative="two-sided", method="auto")
        return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
    except Exception as exc:
        return {"statistic": None, "p_value": None, "error": type(exc).__name__}


def paired_stats(values: list[float]) -> dict:
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": float(mean(values)) if values else None,
        "median": float(median(values)) if values else None,
        "bootstrap_ci95": bootstrap(values),
        "wilcoxon_signed_rank": signed_rank(values),
        "paired_dz": float(mean(values) / sd) if sd else None,
        "wins": sum(value > 0 for value in values),
        "losses": sum(value < 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "differences": values,
    }


def summarize(rows: dict, ids: list[str]) -> dict:
    result = {}
    for condition in CONDITIONS:
        selected = [rows[(qid, condition)] for qid in ids]
        result[condition] = {
            metric: float(mean(float(row["metrics"][metric]) for row in selected))
            for metric in ("rouge_l", "reference_token_coverage", "exact_match", "idk")
        }
    return result


def main() -> None:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
    required = [question for question in benchmark if question["kg_required"] == "YES"]
    dev_ids = [question["question_id"] for question in required if question["split"] == "development"]
    eval_ids = [question["question_id"] for question in required if question["split"] == "evaluation"]
    rows = {}
    for line in FIXED.read_text(encoding="utf8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = (row["question_id"], row["condition"])
            if key in rows:
                raise RuntimeError(f"Duplicate generation-fix row: {key}")
            rows[key] = row
    expected = {(qid, condition) for qid in dev_ids + eval_ids for condition in CONDITIONS}
    if set(rows) != expected:
        raise RuntimeError("Generation-fix rows do not exactly cover 60 questions x 3 conditions")

    split_results = {
        "development": {"question_ids": dev_ids, "question_count": len(dev_ids), "systems": summarize(rows, dev_ids)},
        "held_out_evaluation": {"question_ids": eval_ids, "question_count": len(eval_ids), "systems": summarize(rows, eval_ids)},
    }
    evaluation_stats = {}
    per_question = []
    for condition in ("B_STRUCTURED", "C_TWO_STAGE"):
        for metric in ("rouge_l", "reference_token_coverage"):
            evaluation_stats[f"{condition}_minus_A_CURRENT_{metric}"] = paired_stats(paired(rows, condition, metric, eval_ids))
    for qid in eval_ids:
        row = {"question_id": qid, "category": next(q["category"] for q in required if q["question_id"] == qid)}
        for condition in CONDITIONS:
            item = rows[(qid, condition)]
            row[f"{condition}_rouge_l"] = item["metrics"]["rouge_l"]
            row[f"{condition}_coverage"] = item["metrics"]["reference_token_coverage"]
        row["C_minus_A_rouge_l"] = row["C_TWO_STAGE_rouge_l"] - row["A_CURRENT_rouge_l"]
        row["C_minus_A_coverage"] = row["C_TWO_STAGE_coverage"] - row["A_CURRENT_coverage"]
        per_question.append(row)

    analysis = {
        "integrity": {
            "benchmark_sha256": sha256(BENCHMARK),
            "generation_fix_sha256": sha256(FIXED),
            "prior_relation_aware_generation_sha256": sha256(FROZEN),
            "retrieval_rerun": False,
            "generation_rerun": False,
        },
        "split_definition": {
            "development_count": len(dev_ids),
            "evaluation_count": len(eval_ids),
            "development_ids": dev_ids,
            "evaluation_ids": eval_ids,
        },
        "results": split_results,
        "held_out_A_vs_C": {
            "rouge_l": evaluation_stats["C_TWO_STAGE_minus_A_CURRENT_rouge_l"],
            "coverage": evaluation_stats["C_TWO_STAGE_minus_A_CURRENT_reference_token_coverage"],
        },
        "per_question_evaluation": per_question,
        "all_60_context": {
            "prior_aggregate_C_minus_A": {
                "rouge_l": 0.06896035,
                "coverage": 0.14730158333333335,
            },
            "note": "The all-60 aggregate is retained only for comparison; the 13-question held-out evaluation is primary.",
        },
    }
    (OUT / "split_analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    with (OUT / "split_question_level_results.csv").open("w", newline="", encoding="utf8") as handle:
        fields = ["split", "question_id", "category", "condition", "rouge_l", "coverage", "exact_match", "idk"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, ids in (("development", dev_ids), ("held_out_evaluation", eval_ids)):
            for qid in ids:
                for condition in CONDITIONS:
                    row = rows[(qid, condition)]
                    writer.writerow({
                        "split": split,
                        "question_id": qid,
                        "category": row["category"],
                        "condition": condition,
                        "rouge_l": row["metrics"]["rouge_l"],
                        "coverage": row["metrics"]["reference_token_coverage"],
                        "exact_match": row["metrics"]["exact_match"],
                        "idk": row["metrics"]["idk"],
                    })
    report = [
        "# Split-aware generation-fix analysis",
        "",
        "The existing 180 generations were reused without regeneration. Condition C is unchanged.",
        "",
        "## Primary held-out evaluation",
        "",
        f"- Questions: {len(eval_ids)}",
        f"- IDs: `{', '.join(eval_ids)}`",
        "",
        "| Condition | ROUGE-L | Coverage | Exact match | IDK |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, values in split_results["held_out_evaluation"]["systems"].items():
        report.append(f"| {condition} | {values['rouge_l']:.3f} | {values['reference_token_coverage']:.3f} | {values['exact_match']:.3f} | {values['idk']:.3f} |")
    report += [
        "",
        "### C_TWO_STAGE minus A_CURRENT on held-out evaluation",
        "",
        f"- ROUGE-L mean difference: `{evaluation_stats['C_TWO_STAGE_minus_A_CURRENT_rouge_l']['mean']:.6f}`",
        f"- ROUGE-L median difference: `{evaluation_stats['C_TWO_STAGE_minus_A_CURRENT_rouge_l']['median']:.6f}`",
        f"- ROUGE-L bootstrap 95% CI: `{evaluation_stats['C_TWO_STAGE_minus_A_CURRENT_rouge_l']['bootstrap_ci95']}`",
        f"- ROUGE-L Wilcoxon p-value: `{evaluation_stats['C_TWO_STAGE_minus_A_CURRENT_rouge_l']['wilcoxon_signed_rank']['p_value']}`",
        f"- Coverage mean difference: `{evaluation_stats['C_TWO_STAGE_minus_A_CURRENT_reference_token_coverage']['mean']:.6f}`",
        f"- Coverage median difference: `{evaluation_stats['C_TWO_STAGE_minus_A_CURRENT_reference_token_coverage']['median']:.6f}`",
        f"- Coverage bootstrap 95% CI: `{evaluation_stats['C_TWO_STAGE_minus_A_CURRENT_reference_token_coverage']['bootstrap_ci95']}`",
        f"- Coverage Wilcoxon p-value: `{evaluation_stats['C_TWO_STAGE_minus_A_CURRENT_reference_token_coverage']['wilcoxon_signed_rank']['p_value']}`",
        f"- Coverage wins/losses/ties: `{evaluation_stats['C_TWO_STAGE_minus_A_CURRENT_reference_token_coverage']['wins']}/{evaluation_stats['C_TWO_STAGE_minus_A_CURRENT_reference_token_coverage']['losses']}/{evaluation_stats['C_TWO_STAGE_minus_A_CURRENT_reference_token_coverage']['ties']}`",
        "",
        "The held-out result is decisive. The all-60 improvement does not by itself establish generalization.",
        "",
        "## Development results",
        "",
        f"- Questions: {len(dev_ids)}",
        "",
        "| Condition | ROUGE-L | Coverage | Exact match | IDK |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, values in split_results["development"]["systems"].items():
        report.append(f"| {condition} | {values['rouge_l']:.3f} | {values['reference_token_coverage']:.3f} | {values['exact_match']:.3f} | {values['idk']:.3f} |")
    report += ["", "## Conclusion", "", "The 13-question held-out evaluation, not the 60-question aggregate, determines whether C_TWO_STAGE beats A_CURRENT. Automated metrics do not establish factual quality; human evaluation is required for that claim."]
    (OUT / "split_analysis.md").write_text("\n".join(report) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
