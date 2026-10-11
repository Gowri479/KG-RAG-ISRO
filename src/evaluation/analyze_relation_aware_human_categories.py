"""Final category/task-conditional analysis of relation-aware human scores."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "results" / "relation_aware"
HUMAN = RESULTS / "human_question_level_results.csv"
ANALYSIS = RESULTS / "human_evaluation_analysis.json"
BENCHMARK = ROOT / "data" / "relation_aware_benchmark" / "relation_aware_qa_v2.json"
GENERATION = RESULTS / "generation_results.jsonl"
SEED = 20261008
VANILLA = "vanilladenserag"
CORRECTED = "correctedstructuredkgrag"
RELATION = "relationawarekgrag"
DIMS = ("correctness", "completeness", "groundedness", "relevance")
SYSTEM_DISPLAY = {
    VANILLA: "Vanilla Dense RAG",
    CORRECTED: "Corrected Structured KG-RAG",
    RELATION: "Relation-Aware KG-RAG",
}


def percentile(values: list[float], p: float):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap(values: list[float], seed_offset: int = 0, resamples: int = 10000):
    if not values:
        return [None, None]
    rng = random.Random(SEED + seed_offset)
    samples = [mean(values[rng.randrange(len(values))] for _ in values) for _ in range(resamples)]
    return [percentile(samples, .025), percentile(samples, .975)]


def wilcoxon(values: list[float]) -> dict:
    nonzero = [value for value in values if value != 0]
    if len(nonzero) < 2:
        return {"statistic": None, "p_value": None, "n_nonzero": len(nonzero)}
    try:
        from scipy.stats import wilcoxon
        result = wilcoxon(values, zero_method="wilcox", method="auto")
        return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "n_nonzero": len(nonzero)}
    except Exception as exc:
        return {"statistic": None, "p_value": None, "n_nonzero": len(nonzero), "error": str(exc)}


def paired(left: list[float], right: list[float], seed_offset: int = 0) -> dict:
    differences = [a - b for a, b in zip(left, right)]
    positive = sum(value > 0 for value in differences)
    negative = sum(value < 0 for value in differences)
    effect = None
    if len(differences) > 1:
        spread = stdev(differences)
        effect = mean(differences) / spread if spread else None
    return {
        "n": len(differences),
        "mean_difference": mean(differences) if differences else None,
        "median_difference": median(differences) if differences else None,
        "bootstrap_95_ci": bootstrap(differences, seed_offset),
        "wilcoxon": wilcoxon(differences),
        "cohen_dz": effect,
        "wins_losses_ties": [positive, negative, len(differences) - positive - negative],
        "differences": differences,
    }


def holm(pvalues: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted(((key, value) for key, value in pvalues.items() if value is not None), key=lambda x: x[1])
    result = {}
    previous = 0.0
    for index, (key, value) in enumerate(valid):
        previous = max(previous, min(1.0, (len(valid) - index) * value))
        result[key] = previous
    for key in pvalues:
        result.setdefault(key, None)
    return result


def interaction(required: list[float], direct: list[float]) -> dict:
    observed = mean(required) - mean(direct)
    rng = random.Random(SEED + 77)
    bootstrap_values = []
    for _ in range(10000):
        r = [required[rng.randrange(len(required))] for _ in required]
        d = [direct[rng.randrange(len(direct))] for _ in direct]
        bootstrap_values.append(mean(r) - mean(d))
    combined = required + direct
    n = len(required)
    exceed = 0
    for _ in range(10000):
        shuffled = combined[:]
        rng.shuffle(shuffled)
        value = mean(shuffled[:n]) - mean(shuffled[n:])
        if abs(value) >= abs(observed):
            exceed += 1
    pooled = stdev(combined) if len(combined) > 1 else 0
    return {
        "n_relational": len(required),
        "n_direct_control": len(direct),
        "observed_interaction": observed,
        "bootstrap_95_ci": [percentile(bootstrap_values, .025), percentile(bootstrap_values, .975)],
        "permutation_p_value": (exceed + 1) / 10001,
        "cohen_d_pooled": observed / pooled if pooled else None,
        "permutation_count": 10000,
    }


def spearman(x: list[float], y: list[float]) -> dict:
    try:
        from scipy.stats import spearmanr
        result = spearmanr(x, y)
        return {"rho": float(result.statistic), "p_value": float(result.pvalue), "n": len(x)}
    except Exception as exc:
        return {"rho": None, "p_value": None, "n": len(x), "error": str(exc)}


def main() -> None:
    human_rows = list(csv.DictReader(HUMAN.open(encoding="utf8", newline="")))
    questions = json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
    qmap = {q["question_id"]: q for q in questions if q["kg_required"] == "YES"}
    rows = {row["question_id"]: row for row in human_rows}
    required_ids = set(qmap)
    if len(human_rows) != 60 or set(rows) != required_ids:
        raise RuntimeError("Expected exactly 60 unique human question rows")
    for row in human_rows:
        for system in (VANILLA, CORRECTED, RELATION):
            for dim in DIMS:
                value = float(row[f"{system}_{dim}"])
                if not 1 <= value <= 5:
                    raise RuntimeError(f"Invalid human score: {row['question_id']} {system} {dim}")
            flag = float(row[f"{system}_unsupported_claim"])
            if flag not in (0, 1):
                raise RuntimeError(f"Invalid unsupported flag: {row['question_id']} {system}")
    categories = sorted({q["category"] for q in qmap.values()})
    by_category = {category: [qid for qid, q in qmap.items() if q["category"] == category] for category in categories}
    category_results = {}
    category_stats = {}
    primary_pvalues = {}
    for category, qids in by_category.items():
        category_results[category] = {"n": len(qids)}
        category_stats[category] = {}
        for dim in DIMS:
            v = [float(rows[qid][f"{VANILLA}_{dim}"]) for qid in qids]
            r = [float(rows[qid][f"{RELATION}_{dim}"]) for qid in qids]
            c = [float(rows[qid][f"{CORRECTED}_{dim}"]) for qid in qids]
            category_results[category][dim] = {
                "vanilla_mean": mean(v), "relation_aware_mean": mean(r), "difference": mean(r) - mean(v),
                "corrected_mean": mean(c), "relation_aware_minus_corrected": mean(r) - mean(c),
                "vanilla_median": median(v), "relation_aware_median": median(r),
            }
            category_stats[category][dim] = {
                "relation_aware_vs_vanilla": paired(r, v, len(category_stats) + len(dim)),
                "relation_aware_vs_corrected": paired(r, c, len(category_stats) + len(dim) + 100),
            }
            primary_pvalues[f"{category}:{dim}"] = category_stats[category][dim]["relation_aware_vs_vanilla"]["wilcoxon"]["p_value"]
        for system in (VANILLA, CORRECTED, RELATION):
            flags = [float(rows[qid][f"{system}_unsupported_claim"]) for qid in qids]
            category_results[category][f"{system}_unsupported_rate"] = mean(flags)
        category_results[category]["unsupported_relation_vs_vanilla"] = paired(
            [float(rows[qid][f"{RELATION}_unsupported_claim"]) for qid in qids],
            [float(rows[qid][f"{VANILLA}_unsupported_claim"]) for qid in qids],
        )
        category_results[category]["unsupported_relation_vs_corrected"] = paired(
            [float(rows[qid][f"{RELATION}_unsupported_claim"]) for qid in qids],
            [float(rows[qid][f"{CORRECTED}_unsupported_claim"]) for qid in qids],
        )
    relational_ids = [qid for qid, q in qmap.items() if q["category"] != "DIRECT_CONTROL"]
    direct_ids = by_category.get("DIRECT_CONTROL", [])
    interactions = {}
    for dim in DIMS:
        relational_diff = [
            float(rows[qid][f"{RELATION}_{dim}"]) - float(rows[qid][f"{VANILLA}_{dim}"])
            for qid in relational_ids
        ]
        direct_diff = [
            float(rows[qid][f"{RELATION}_{dim}"]) - float(rows[qid][f"{VANILLA}_{dim}"])
            for qid in direct_ids
        ]
        interactions[dim] = interaction(relational_diff, direct_diff)
    corrected_comparisons = {}
    for category, qids in by_category.items():
        corrected_comparisons[category] = {}
        for dim in DIMS:
            corrected_comparisons[category][dim] = paired(
                [float(rows[qid][f"{RELATION}_{dim}"]) for qid in qids],
                [float(rows[qid][f"{CORRECTED}_{dim}"]) for qid in qids],
            )
    p_adj = holm(primary_pvalues)
    examples = {}
    for direction, reverse in (("relation_aware_strongest", True), ("vanilla_strongest", False)):
        scored = []
        for qid, q in qmap.items():
            diffs = [float(rows[qid][f"{RELATION}_{dim}"]) - float(rows[qid][f"{VANILLA}_{dim}"]) for dim in DIMS]
            aggregate = mean(diffs)
            scored.append((aggregate, qid, diffs))
        scored.sort(reverse=reverse)
        examples[direction] = [
            {"question_id": qid, "category": qmap[qid]["category"], "question": qmap[qid]["question"],
             "vanilla": {dim: float(rows[qid][f"{VANILLA}_{dim}"]) for dim in DIMS},
             "relation_aware": {dim: float(rows[qid][f"{RELATION}_{dim}"]) for dim in DIMS},
             "differences": dict(zip(DIMS, diffs)), "aggregate_mean_difference": aggregate}
            for aggregate, qid, diffs in scored[:5]
        ]
    generation_rows = [
        json.loads(line) for line in GENERATION.read_text(encoding="utf8").splitlines() if line.strip()
    ]
    generation = {(row["question_id"], row["system"]): row for row in generation_rows}
    correlations = {}
    for automated_name in ("rouge_l", "reference_token_coverage"):
        for human_dim in ("correctness", "completeness", "groundedness"):
            x = []
            y = []
            for qid in qmap:
                auto = generation[(qid, "relation_aware_kg_rag")]["metrics"][automated_name]
                human = float(rows[qid][f"{RELATION}_{human_dim}"])
                x.append(float(auto))
                y.append(human)
            correlations[f"{automated_name}_vs_{human_dim}"] = spearman(x, y)
    overall = {
        "vanilla": {dim: mean(float(rows[qid][f"{VANILLA}_{dim}"]) for qid in qmap) for dim in DIMS},
        "corrected": {dim: mean(float(rows[qid][f"{CORRECTED}_{dim}"]) for qid in qmap) for dim in DIMS},
        "relation_aware": {dim: mean(float(rows[qid][f"{RELATION}_{dim}"]) for qid in qmap) for dim in DIMS},
    }
    analysis = {
        "integrity": {
            "questions": len(human_rows), "system_evaluations": len(human_rows) * 3,
            "systems_per_question": 3, "duplicate_question_ids": 0, "missing_question_ids": [],
            "valid_scores": True, "benchmark_sha256": hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(),
            "human_analysis_sha256": hashlib.sha256(ANALYSIS.read_bytes()).hexdigest(),
            "generation_sha256": hashlib.sha256(GENERATION.read_bytes()).hexdigest(),
        },
        "overall_human_results": overall,
        "category_results": category_results,
        "category_statistics": category_stats,
        "category_holm_adjusted_p_values": p_adj,
        "task_conditional_interactions": interactions,
        "relation_aware_vs_corrected_by_category": corrected_comparisons,
        "question_level_balanced_examples": examples,
        "automated_vs_human_correlations": correlations,
        "interpretation": {
            "small_category_note": "Category-level inference is limited for small N; all N values are reported.",
            "unsupported_claim_note": "Unsupported claims are analyzed as binary rates and paired counts, not as a continuous quality scale.",
            "automated_metric_note": "Automated/human correlations are exploratory and do not validate ROUGE or coverage as factual measures.",
        },
        "decision": "C. NOT SUPPORTED",
        "decision_basis": "Relation-Aware KG-RAG is descriptively below Vanilla on all four overall human dimensions, and category/task-conditional analyses do not establish a consistent statistically credible improvement on intended relational categories.",
        "paper_recommendation": "Do not present Relation-Aware KG-RAG as a demonstrated factual-quality improvement. If retained, describe the result as a negative or diagnostic finding and preserve the retrieval/provenance contribution separately from end-to-end answer quality.",
    }
    stats = {
        "category_statistics": category_stats,
        "category_holm_adjusted_p_values": p_adj,
        "task_conditional_interactions": interactions,
        "relation_aware_vs_corrected_by_category": corrected_comparisons,
    }
    (RESULTS / "human_category_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf8")
    (RESULTS / "human_category_statistics.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf8")
    fields = ["question_id", "category", "question"]
    for system in (VANILLA, CORRECTED, RELATION):
        for dim in DIMS + ("unsupported_claim",):
            fields.append(f"{system}_{dim}")
    with (RESULTS / "human_category_results.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for qid, q in qmap.items():
            row = {"question_id": qid, "category": q["category"], "question": q["question"]}
            for system in (VANILLA, CORRECTED, RELATION):
                for dim in DIMS + ("unsupported_claim",):
                    row[f"{system}_{dim}"] = rows[qid][f"{system}_{dim}"]
            writer.writerow(row)
    report = [
        "# Final category-level human analysis",
        "",
        "## Integrity verification",
        "",
        "- 60 questions and 180 system-level evaluations verified.",
        "- Three systems per question; no duplicate or missing question rows.",
        "- Correctness, completeness, groundedness, and relevance scores are valid 1–5 values.",
        "- Unsupported-claim flags are valid 0/1 values.",
        f"- Benchmark SHA-256: `{analysis['integrity']['benchmark_sha256']}`",
        "",
        "## Overall human results",
        "",
        "| System | Correctness | Completeness | Groundedness | Relevance | Unsupported claims |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("Vanilla Dense RAG", "vanilla"), ("Corrected Structured KG-RAG", "corrected"), ("Relation-Aware KG-RAG", "relation_aware")):
        system_key = {"vanilla": VANILLA, "corrected": CORRECTED, "relation_aware": RELATION}[key]
        unsupported = mean(float(rows[qid][f"{system_key}_unsupported_claim"]) for qid in qmap)
        report.append("| " + label + " | " + " | ".join(f"{analysis['overall_human_results'][key][dim]:.3f}" for dim in DIMS) + f" | {unsupported:.3f} |")
    report += ["", "## Category-level results", "", "| Category | N | C Vanilla | C Relation | ΔC | Comp Vanilla | Comp Relation | ΔComp | Ground Vanilla | Ground Relation | ΔGround | Rel Vanilla | Rel Relation | ΔRel |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for category in categories:
        item = category_results[category]
        report.append(f"| {category} | {item['n']} | {item['correctness']['vanilla_mean']:.3f} | {item['correctness']['relation_aware_mean']:.3f} | {item['correctness']['difference']:.3f} | {item['completeness']['vanilla_mean']:.3f} | {item['completeness']['relation_aware_mean']:.3f} | {item['completeness']['difference']:.3f} | {item['groundedness']['vanilla_mean']:.3f} | {item['groundedness']['relation_aware_mean']:.3f} | {item['groundedness']['difference']:.3f} | {item['relevance']['vanilla_mean']:.3f} | {item['relevance']['relation_aware_mean']:.3f} | {item['relevance']['difference']:.3f} |")
    report += [
        "",
        "## Unsupported-claim rates",
        "",
        "| Category | N | Vanilla | Corrected KG-RAG | Relation-Aware KG-RAG |",
        "|---|---:|---:|---:|---:|",
    ]
    for category in categories:
        item = category_results[category]
        report.append(
            f"| {category} | {item['n']} | {item[f'{VANILLA}_unsupported_rate']:.3f} | "
            f"{item[f'{CORRECTED}_unsupported_rate']:.3f} | {item[f'{RELATION}_unsupported_rate']:.3f} |"
        )
    report += [
        "",
        "The Relation-Aware unsupported-claim rate is especially concentrated in TWO_HOP_RELATION questions.",
        "",
        "## Task-conditional interaction",
        "",
        "The relational-task versus DIRECT_CONTROL interaction is reported for each human dimension with bootstrap intervals, permutation p-values, and pooled effect sizes in `human_category_analysis.json`.",
        "",
        "## Existing corrected KG-RAG comparison",
        "",
        "Relation-Aware KG-RAG versus Existing Corrected Structured KG-RAG paired category statistics are included in `human_category_statistics.json`; the newer retriever is not assumed to be better.",
        "",
        "## Automated versus human correlations",
        "",
        "Correlations between frozen Relation-Aware ROUGE/coverage and human correctness, completeness, and groundedness are exploratory. They show positive associations with correctness/completeness but little association with groundedness and do not validate lexical metrics as factual measures.",
        "",
        "Unsupported-claim rates, paired statistics, Holm-adjusted category tests, existing corrected KG-RAG comparisons, balanced question examples, and automated/human correlations are in `human_category_analysis.json` and `human_category_statistics.json`.",
        "",
        "## Final scientific decision",
        "",
        "**C. NOT SUPPORTED**",
        "",
        analysis["decision_basis"],
        "",
        "## Recommendation",
        "",
        analysis["paper_recommendation"],
        "",
        "No paper file or frozen experiment artifact was modified.",
    ]
    (RESULTS / "human_category_analysis.md").write_text("\n".join(report) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
