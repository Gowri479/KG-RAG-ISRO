"""Analyze the small prompt-only bottleneck diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "results" / "relation_aware"
INPUT = RESULTS / "prompt_diagnostic_results.jsonl"
OUTPUT = RESULTS / "prompt_diagnostic_results.json"
CSV_OUTPUT = RESULTS / "prompt_diagnostic_results.csv"
VARIANTS = ("P1_current", "P2_kg_priority", "P3_structured_evidence")
SEED = 20261008


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * p
    low, high = int(position), int(position + 1)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def paired(left, right):
    differences = [a - b for a, b in zip(left, right)]
    rng = random.Random(SEED)
    samples = [
        mean(differences[rng.randrange(len(differences))] for _ in differences)
        for _ in range(10000)
    ] if differences else []
    try:
        from scipy.stats import wilcoxon
        nonzero = [x for x in differences if x]
        result = wilcoxon(differences) if len(nonzero) >= 2 else None
        test = {
            "statistic": float(result.statistic) if result else None,
            "p_value": float(result.pvalue) if result else None,
            "n_nonzero": len(nonzero),
        }
    except Exception as exc:
        test = {"statistic": None, "p_value": None, "error": str(exc)}
    return {
        "n": len(differences),
        "mean_difference": mean(differences) if differences else None,
        "median_difference": median(differences) if differences else None,
        "bootstrap_95_ci": [percentile(samples, 0.025), percentile(samples, 0.975)],
        "wilcoxon": test,
        "wins_losses_ties": [
            sum(x > 0 for x in differences),
            sum(x < 0 for x in differences),
            sum(x == 0 for x in differences),
        ],
        "differences": differences,
    }


def main():
    rows = [
        json.loads(line)
        for line in INPUT.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    keys = [(row["question_id"], row["prompt_variant"]) for row in rows]
    expected = {(qid, variant) for qid in {row["question_id"] for row in rows} for variant in VARIANTS}
    if len(rows) != 54 or len(set(keys)) != 54 or set(keys) != expected:
        raise RuntimeError("Prompt diagnostic must contain exactly 54 unique question/variant rows")
    indexed = {(row["question_id"], row["prompt_variant"]): row for row in rows}
    question_ids = sorted({row["question_id"] for row in rows})
    summaries = {}
    for variant in VARIANTS:
        selected = [indexed[(qid, variant)] for qid in question_ids]
        summaries[variant] = {
            "n": len(selected),
            "rouge_l_mean": mean(row["metrics"]["rouge_l"] for row in selected),
            "rouge_l_median": median(row["metrics"]["rouge_l"] for row in selected),
            "coverage_mean": mean(row["metrics"]["reference_token_coverage"] for row in selected),
            "coverage_median": median(row["metrics"]["reference_token_coverage"] for row in selected),
            "exact_match_rate": mean(float(row["metrics"]["exact_match"]) for row in selected),
            "idk_rate": mean(float(row["metrics"]["idk"]) for row in selected),
        }
    comparisons = {}
    for variant in VARIANTS[1:]:
        comparisons[f"{variant}_minus_P1_current"] = {
            metric: paired(
                [indexed[(qid, variant)]["metrics"][metric] for qid in question_ids],
                [indexed[(qid, "P1_current")]["metrics"][metric] for qid in question_ids],
            )
            for metric in ("rouge_l", "reference_token_coverage", "exact_match", "idk")
        }
    question_rows = []
    for qid in question_ids:
        base = indexed[(qid, "P1_current")]
        for variant in VARIANTS:
            row = indexed[(qid, variant)]
            question_rows.append({
                "question_id": qid,
                "prompt_variant": variant,
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "rouge_l": row["metrics"]["rouge_l"],
                "coverage": row["metrics"]["reference_token_coverage"],
                "exact_match": row["metrics"]["exact_match"],
                "idk": row["metrics"]["idk"],
                "delta_rouge_l_vs_P1": row["metrics"]["rouge_l"] - base["metrics"]["rouge_l"],
                "delta_coverage_vs_P1": row["metrics"]["reference_token_coverage"] - base["metrics"]["reference_token_coverage"],
                "answer": row["answer"],
            })
    result = {
        "integrity": {
            "rows": len(rows),
            "questions": len(question_ids),
            "variants": list(VARIANTS),
            "duplicates": len(keys) - len(set(keys)),
            "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        },
        "selection": "First 18 KG-required questions by question_id, selected before diagnostic outcomes.",
        "variant_summaries": summaries,
        "paired_comparisons_vs_P1": comparisons,
        "interpretation": "Small prompt-only diagnostic, not a benchmark. Automated lexical metrics do not establish factual quality.",
        "retrieval_frozen": True,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    with CSV_OUTPUT.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(question_rows[0]))
        writer.writeheader()
        writer.writerows(question_rows)
    bottleneck_path = RESULTS / "generation_bottleneck_analysis.json"
    bottleneck = json.loads(bottleneck_path.read_text(encoding="utf8"))
    bottleneck["prompt_diagnostic"] = result
    bottleneck["critical_comparison"]["prompt_diagnostic_observation"] = (
        "On this fixed 18-question diagnostic, P3 structured-evidence prompting had the highest "
        "ROUGE-L and coverage descriptively; P2 KG-priority prompting did not improve over P1. "
        "Intervals and p-values are exploratory and do not establish a general prompt effect."
    )
    bottleneck_path.write_text(json.dumps(bottleneck, indent=2) + "\n", encoding="utf8")
    report = RESULTS / "generation_bottleneck_analysis.md"
    with report.open("a", encoding="utf8") as handle:
        handle.write(
            "\n## Small prompt-only diagnostic\n\n"
            "- Fixed subset: first 18 KG-required questions by question ID.\n"
            "- Retrieval contexts: frozen Relation-Aware KG-RAG contexts.\n"
            "- P1 is the current prompt; P2 adds KG priority; P3 separates structured evidence instructions.\n\n"
            "| Variant | ROUGE-L mean | Coverage mean | IDK rate |\n"
            "|---|---:|---:|---:|\n"
        )
        for variant, values in summaries.items():
            handle.write(
                f"| {variant} | {values['rouge_l_mean']:.6f} | "
                f"{values['coverage_mean']:.6f} | {values['idk_rate']:.6f} |\n"
            )
        handle.write(
            "\nP3 was descriptively highest on this small diagnostic, while P2 was below P1. "
            "This is exploratory and is not a benchmark or evidence of factual-quality improvement. "
            "The complete paired statistics are in `prompt_diagnostic_results.json`.\n"
        )


if __name__ == "__main__":
    main()
