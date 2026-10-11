"""Analyze relation-aware generations without treating lexical metrics as facts."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "relation_aware_benchmark" / "relation_aware_qa_v2.json"
RESULTS = ROOT / "data" / "results" / "relation_aware"


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (position - low)


def paired(left, right):
    diffs = [a - b for a, b in zip(left, right)]
    rng = random.Random(20261008)
    samples = [mean([diffs[rng.randrange(len(diffs))] for _ in diffs]) for _ in range(5000)] if diffs else []
    try:
        from scipy.stats import wilcoxon
        nonzero = [x for x in diffs if x]
        test = {"statistic": float(wilcoxon(diffs).statistic), "p_value": float(wilcoxon(diffs).pvalue)} if len(nonzero) >= 2 else {"statistic": None, "p_value": None}
    except Exception as exc:
        test = {"statistic": None, "p_value": None, "error": str(exc)}
    positive = sum(x > 0 for x in diffs)
    negative = sum(x < 0 for x in diffs)
    return {
        "n": len(diffs),
        "mean_difference": mean(diffs) if diffs else None,
        "median_difference": median(diffs) if diffs else None,
        "bootstrap_95_ci": [percentile(samples, .025), percentile(samples, .975)],
        "wilcoxon": test,
        "positive_negative_tied": [positive, negative, len(diffs) - positive - negative],
        "rank_biserial": ((positive - negative) / (positive + negative)) if positive + negative else None,
    }


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["system"], "overall")].append(row)
        for key in ("kg_required", "category", "split"):
            grouped[(row["system"], str(row[key]))].append(row)
        for relation in row["relation_type"]:
            grouped[(row["system"], "relation:" + relation)].append(row)
    output = {}
    for key, items in grouped.items():
        output["|".join(key)] = {
            "n": len(items),
            "rouge_l": mean(x["metrics"]["rouge_l"] for x in items),
            "coverage": mean(x["metrics"]["reference_token_coverage"] for x in items),
            "exact_match": mean(x["metrics"]["exact_match"] for x in items),
            "idk_rate": mean(float(x["metrics"]["idk"]) for x in items),
        }
    return output


def main():
    rows = [json.loads(line) for line in (RESULTS / "generation_results.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
    by_key = {(x["question_id"], x["system"]): x for x in rows}
    questions = json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
    metrics = {}
    for metric in ("rouge_l", "reference_token_coverage", "exact_match", "idk"):
        metrics[metric] = {}
        for condition, items in (
            ("overall", questions),
            ("KG_REQUIRED=YES", [x for x in questions if x["kg_required"] == "YES"]),
            ("KG_REQUIRED=NO", [x for x in questions if x["kg_required"] == "NO"]),
        ):
            left = [by_key[(x["question_id"], "relation_aware_kg_rag")]["metrics"][metric] for x in items]
            right = [by_key[(x["question_id"], "vanilla_dense_rag")]["metrics"][metric] for x in items]
            metrics[metric][condition] = paired(left, right)
    diagnostics = {}
    required = [x for x in rows if x["kg_required"] == "YES" and x["system"] == "relation_aware_kg_rag"]
    diagnostics["required_path_recovery"] = mean(float(x["required_path_found"]) for x in required) if required else None
    diagnostics["valid_provenance"] = mean(float(x["valid_provenance"]) for x in required) if required else None
    diagnostics["mean_context_tokens"] = mean(x["context_token_count"] for x in rows) if rows else None
    diagnostics["intent_relation_exact_match"] = mean(set(x["detected_relation"]) == set(x["relation_type"]) for x in required) if required else None
    report = {
        "benchmark_sha256": hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(),
        "rows": len(rows),
        "systems": sorted({x["system"] for x in rows}),
        "descriptive_results": summarize(rows),
        "primary_relation_aware_vs_vanilla": metrics,
        "retrieval_diagnostics": diagnostics,
        "multiple_comparison_note": "Wilcoxon values are unadjusted; Holm correction is applied to the four primary lexical metrics in the report.",
        "interpretation": "Lexical metrics are retrieval/generation diagnostics and are not human factual-quality evidence.",
    }
    (RESULTS / "analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"rows": len(rows), "output": str(RESULTS / "analysis.json")}))


if __name__ == "__main__":
    main()
