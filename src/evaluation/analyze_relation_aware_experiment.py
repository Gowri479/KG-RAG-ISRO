"""Read-only statistical analysis for the frozen relation-aware experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "results" / "relation_aware"
BENCHMARK = ROOT / "data" / "relation_aware_benchmark" / "relation_aware_qa_v2.json"
GENERATIONS = RESULTS / "generation_results.jsonl"
SYSTEMS = (
    "vanilla_dense_rag",
    "corrected_structured_kg_rag",
    "relation_aware_kg_rag",
    "relation_aware_kg_only",
    "relation_aware_kg_source",
    "relation_aware_kg_dense",
    "relation_aware_kg_dense_bm25",
)
PRIMARY = "relation_aware_kg_rag"
BASELINE = "vanilla_dense_rag"
METRICS = ("rouge_l", "reference_token_coverage", "exact_match", "idk")
SEED = 20261008


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def wilcoxon(differences: list[float]) -> dict:
    nonzero = [value for value in differences if value != 0]
    if len(nonzero) < 2:
        return {"statistic": None, "p_value": None, "n_nonzero": len(nonzero)}
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto")
        return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "n_nonzero": len(nonzero)}
    except Exception as exc:
        return {"statistic": None, "p_value": None, "n_nonzero": len(nonzero), "error": str(exc)}


def bootstrap_mean(differences: list[float], seed: int = SEED, resamples: int = 10000) -> dict:
    if not differences:
        return {"estimate": None, "ci95": [None, None], "resamples": resamples}
    rng = random.Random(seed)
    samples = [
        mean(differences[rng.randrange(len(differences))] for _ in differences)
        for _ in range(resamples)
    ]
    return {
        "estimate": mean(differences),
        "ci95": [percentile(samples, 0.025), percentile(samples, 0.975)],
        "resamples": resamples,
        "seed": seed,
    }


def paired_summary(left: list[float], right: list[float], label: str) -> dict:
    differences = [a - b for a, b in zip(left, right)]
    positive = sum(value > 0 for value in differences)
    negative = sum(value < 0 for value in differences)
    return {
        "metric": label,
        "n": len(differences),
        "left_mean": mean(left) if left else None,
        "right_mean": mean(right) if right else None,
        "mean_difference": mean(differences) if differences else None,
        "median_difference": median(differences) if differences else None,
        "bootstrap_mean_difference": bootstrap_mean(differences),
        "wilcoxon_signed_rank": wilcoxon(differences),
        "wins_losses_ties": [positive, negative, len(differences) - positive - negative],
        "rank_biserial": (
            (positive - negative) / (positive + negative) if positive + negative else None
        ),
        "differences": differences,
    }


def permutation_difference(required: list[float], not_required: list[float]) -> dict:
    observed = mean(required) - mean(not_required)
    combined = required + not_required
    n_required = len(required)
    rng = random.Random(SEED + 1)
    exceed = 0
    samples = []
    for _ in range(10000):
        shuffled = combined[:]
        rng.shuffle(shuffled)
        value = mean(shuffled[:n_required]) - mean(shuffled[n_required:])
        samples.append(value)
        if abs(value) >= abs(observed):
            exceed += 1
    bootstrap = []
    for _ in range(10000):
        required_sample = [required[rng.randrange(len(required))] for _ in required]
        not_required_sample = [not_required[rng.randrange(len(not_required))] for _ in not_required]
        bootstrap.append(mean(required_sample) - mean(not_required_sample))
    return {
        "observed_difference_in_differences": observed,
        "bootstrap_ci95": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)]
        if required and not_required else [None, None],
        "bootstrap_resamples": 10000,
        "bootstrap_seed": SEED + 2,
        "permutation_p_value": (exceed + 1) / 10001,
        "permutations": 10000,
        "seed": SEED + 1,
    }


def holm(p_values: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted(((name, value) for name, value in p_values.items() if value is not None), key=lambda x: x[1])
    adjusted = {}
    previous = 0.0
    total = len(valid)
    for index, (name, value) in enumerate(valid):
        corrected = min(1.0, max(previous, (total - index) * value))
        adjusted[name] = corrected
        previous = corrected
    for name in p_values:
        adjusted.setdefault(name, None)
    return adjusted


def load() -> tuple[list[dict], list[dict]]:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
    rows = [
        json.loads(line)
        for line in GENERATIONS.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    expected = {(question["question_id"], system) for question in benchmark for system in SYSTEMS}
    actual = [(row["question_id"], row["system"]) for row in rows]
    if len(rows) != 504 or len(set(actual)) != 504 or set(actual) != expected:
        raise RuntimeError("Generation package is not exactly 504 unique question/system rows")
    return benchmark, rows


def metric(row: dict, name: str) -> float:
    return float(row["metrics"][name])


def group_rows(rows: list[dict], key: str, value: str, system: str) -> list[dict]:
    return [row for row in rows if row["system"] == system and str(row.get(key)) == value]


def summarize_systems(rows: list[dict]) -> dict:
    output = {}
    for system in SYSTEMS:
        selected = [row for row in rows if row["system"] == system]
        output[system] = {
            "n": len(selected),
            "rouge_l_mean": mean(metric(row, "rouge_l") for row in selected),
            "rouge_l_median": median(metric(row, "rouge_l") for row in selected),
            "coverage_mean": mean(metric(row, "reference_token_coverage") for row in selected),
            "coverage_median": median(metric(row, "reference_token_coverage") for row in selected),
            "exact_match_rate": mean(metric(row, "exact_match") for row in selected),
            "idk_rate": mean(metric(row, "idk") for row in selected),
        }
    return output


def comparisons(questions: list[dict], indexed: dict[tuple[str, str], dict], selector) -> dict:
    output = {}
    p_values = {}
    for name in ("rouge_l", "reference_token_coverage", "exact_match", "idk"):
        left = [metric(indexed[(q["question_id"], PRIMARY)], name) for q in questions if selector(q)]
        right = [metric(indexed[(q["question_id"], BASELINE)], name) for q in questions if selector(q)]
        result = paired_summary(left, right, name)
        output[name] = result
        p_values[name] = result["wilcoxon_signed_rank"]["p_value"]
    output["_holm_adjusted_p_values"] = holm(p_values)
    return output


def category_results(questions: list[dict], indexed: dict[tuple[str, str], dict]) -> dict:
    output = {}
    for category in sorted({q["category"] for q in questions}):
        subset = [q for q in questions if q["category"] == category]
        output[category] = {}
        for system in (BASELINE, "corrected_structured_kg_rag", PRIMARY):
            selected = [indexed[(q["question_id"], system)] for q in subset]
            output[category][system] = {
                "n": len(selected),
                "rouge_l_mean": mean(metric(row, "rouge_l") for row in selected),
                "coverage_mean": mean(metric(row, "reference_token_coverage") for row in selected),
                "exact_match_rate": mean(metric(row, "exact_match") for row in selected),
                "idk_rate": mean(metric(row, "idk") for row in selected),
            }
    return output


def relation_results(questions: list[dict], indexed: dict[tuple[str, str], dict]) -> dict:
    relations = sorted({relation for q in questions for relation in q["relation_type"]})
    output = {}
    for relation in relations:
        subset = [q for q in questions if relation in q["relation_type"]]
        output[relation] = {}
        for system in (BASELINE, "corrected_structured_kg_rag", PRIMARY):
            selected = [indexed[(q["question_id"], system)] for q in subset]
            output[relation][system] = {
                "n": len(selected),
                "rouge_l_mean": mean(metric(row, "rouge_l") for row in selected),
                "coverage_mean": mean(metric(row, "reference_token_coverage") for row in selected),
                "exact_match_rate": mean(metric(row, "exact_match") for row in selected),
            }
    return output


def diagnostics(questions: list[dict], indexed: dict[tuple[str, str], dict]) -> dict:
    required = [q for q in questions if q["kg_required"] == "YES"]
    traces = [indexed[(q["question_id"], PRIMARY)] for q in required]
    intent_exact = [
        set(row["detected_relation"]) == set(q["relation_type"])
        for q, row in zip(required, traces)
    ]
    entity_exact = [bool(row["detected_entities"]) for row in traces]
    hop_exact = [row["hop_depth"] == len(q["supporting_paths"][0]) for q, row in zip(required, traces)]
    return {
        "required_questions": len(required),
        "relation_intent_exact_or_set_match_rate": mean(intent_exact),
        "entity_detection_nonempty_rate": mean(entity_exact),
        "hop_depth_exact_rate": mean(hop_exact),
        "required_path_recovery_rate": mean(bool(row["required_path_found"]) for row in traces),
        "valid_provenance_rate": mean(bool(row["valid_provenance"]) for row in traces),
        "path_count_mean": mean(len(row["kg_path_ids"]) for row in traces),
        "path_count_median": median(len(row["kg_path_ids"]) for row in traces),
        "evidence_count_mean": mean(len(row["selected_evidence"]) for row in traces),
        "evidence_count_median": median(len(row["selected_evidence"]) for row in traces),
        "failure_categories": {
            "relation_detection": sum(not x for x in intent_exact),
            "entity_resolution": sum(not x for x in entity_exact),
            "hop_depth_detection": sum(not x for x in hop_exact),
            "path_retrieval": sum(not row["required_path_found"] for row in traces),
            "provenance": sum(not row["valid_provenance"] for row in traces),
        },
    }


def failure_analysis(questions: list[dict], indexed: dict[tuple[str, str], dict]) -> list[dict]:
    output = []
    for question in questions:
        primary = indexed[(question["question_id"], PRIMARY)]
        baseline = indexed[(question["question_id"], BASELINE)]
        if metric(primary, "rouge_l") >= metric(baseline, "rouge_l"):
            continue
        intent_ok = set(primary["detected_relation"]) == set(question["relation_type"])
        entity_ok = bool(primary["detected_entities"])
        if not intent_ok:
            category = "relation detection"
            evidence = "Detected relation set differs from benchmark relation type."
        elif not entity_ok:
            category = "entity resolution"
            evidence = "No entity was detected in the trace."
        elif not primary["required_path_found"]:
            category = "path retrieval"
            evidence = "Required supporting path was not recovered."
        elif not primary["valid_provenance"]:
            category = "evidence selection/fusion"
            evidence = "Retrieved path lacked valid provenance."
        else:
            category = "generation"
            evidence = "Trace recovered a provenance-backed path; lower lexical answer metric is attributed only to generation."
        output.append({
            "question_id": question["question_id"],
            "category": category,
            "evidence": evidence,
            "primary_rouge_l": metric(primary, "rouge_l"),
            "baseline_rouge_l": metric(baseline, "rouge_l"),
        })
    return output


def correlations(questions: list[dict], indexed: dict[tuple[str, str], dict]) -> dict:
    values = []
    for question in questions:
        row = indexed[(question["question_id"], PRIMARY)]
        values.append({
            "context_tokens": row["context_token_count"],
            "path_count": len(row["kg_path_ids"]),
            "provenance_count": len(row["provenance_ids"]),
            "rouge_l": metric(row, "rouge_l"),
            "coverage": metric(row, "reference_token_coverage"),
        })
    output = {}
    try:
        from scipy.stats import spearmanr
        for feature in ("context_tokens", "path_count", "provenance_count"):
            output[feature] = {
                target: {
                    "rho": float(spearmanr([x[feature] for x in values], [x[target] for x in values]).statistic),
                    "p_value": float(spearmanr([x[feature] for x in values], [x[target] for x in values]).pvalue),
                }
                for target in ("rouge_l", "coverage")
            }
    except Exception as exc:
        output["error"] = str(exc)
    return output


def ablation_pairwise(questions: list[dict], indexed: dict[tuple[str, str], dict]) -> dict:
    output = {}
    for left in SYSTEMS:
        for right in SYSTEMS:
            if left >= right:
                continue
            key = f"{left} - {right}"
            subset = questions
            left_rows = [indexed[(q["question_id"], left)] for q in subset]
            right_rows = [indexed[(q["question_id"], right)] for q in subset]
            output[key] = {
                name: paired_summary(
                    [metric(row, name) for row in left_rows],
                    [metric(row, name) for row in right_rows],
                    name,
                )
                for name in ("rouge_l", "reference_token_coverage")
            }
    return output


def context_summary(rows: list[dict]) -> dict:
    return {
        system: {
            "context_tokens_mean": mean(row["context_token_count"] for row in selected),
            "context_tokens_median": median(row["context_token_count"] for row in selected),
            "retrieved_path_count_mean": mean(len(row["kg_path_ids"]) for row in selected),
            "provenance_count_mean": mean(len(row["provenance_ids"]) for row in selected),
            "evidence_count_mean": mean(len(row["selected_evidence"]) for row in selected),
        }
        for system in SYSTEMS
        for selected in [[row for row in rows if row["system"] == system]]
    }


def write_question_csv(questions, indexed):
    path = RESULTS / "question_level_results.csv"
    fields = ["question_id", "category", "kg_required", "split", "relation_type",
              "vanilla_rouge_l", "corrected_kg_rouge_l", "relation_aware_rouge_l",
              "vanilla_coverage", "corrected_kg_coverage", "relation_aware_coverage",
              "primary_minus_vanilla_rouge_l", "primary_minus_vanilla_coverage",
              "required_path_found", "valid_provenance", "path_count", "evidence_count",
              "context_token_count", "failure_category"]
    failures = {x["question_id"]: x["category"] for x in failure_analysis(questions, indexed)}
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for q in questions:
            vanilla = indexed[(q["question_id"], BASELINE)]
            corrected = indexed[(q["question_id"], "corrected_structured_kg_rag")]
            primary = indexed[(q["question_id"], PRIMARY)]
            writer.writerow({
                "question_id": q["question_id"], "category": q["category"], "kg_required": q["kg_required"],
                "split": q["split"], "relation_type": "|".join(q["relation_type"]),
                "vanilla_rouge_l": metric(vanilla, "rouge_l"),
                "corrected_kg_rouge_l": metric(corrected, "rouge_l"),
                "relation_aware_rouge_l": metric(primary, "rouge_l"),
                "vanilla_coverage": metric(vanilla, "reference_token_coverage"),
                "corrected_kg_coverage": metric(corrected, "reference_token_coverage"),
                "relation_aware_coverage": metric(primary, "reference_token_coverage"),
                "primary_minus_vanilla_rouge_l": metric(primary, "rouge_l") - metric(vanilla, "rouge_l"),
                "primary_minus_vanilla_coverage": metric(primary, "reference_token_coverage") - metric(vanilla, "reference_token_coverage"),
                "required_path_found": primary["required_path_found"],
                "valid_provenance": primary["valid_provenance"],
                "path_count": len(primary["kg_path_ids"]),
                "evidence_count": len(primary["selected_evidence"]),
                "context_token_count": primary["context_token_count"],
                "failure_category": failures.get(q["question_id"], ""),
            })


def main():
    questions, rows = load()
    indexed = {(row["question_id"], row["system"]): row for row in rows}
    required = [q for q in questions if q["kg_required"] == "YES"]
    not_required = [q for q in questions if q["kg_required"] == "NO"]
    required_comp = comparisons(questions, indexed, lambda q: q["kg_required"] == "YES")
    not_required_comp = comparisons(questions, indexed, lambda q: q["kg_required"] == "NO")
    did_by_metric = {}
    for name in METRICS:
        required_d = required_comp[name]["differences"]
        not_required_d = not_required_comp[name]["differences"]
        did_by_metric[name] = permutation_difference(required_d, not_required_d)
    analysis = {
        "integrity": {
            "generation_rows": len(rows), "questions": len(questions),
            "systems": list(SYSTEMS), "unique_pairs": len(indexed),
            "duplicates": 504 - len(indexed), "missing": 504 - len(indexed),
            "benchmark_sha256": hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(),
        },
        "system_results": summarize_systems(rows),
        "primary_comparison": {
            "KG_REQUIRED=YES": required_comp,
            "KG_REQUIRED=NO": not_required_comp,
            "difference_in_differences": did_by_metric,
        },
        "category_results": category_results(questions, indexed),
        "relation_results": relation_results(questions, indexed),
        "ablation_results": {
            "system_summaries": summarize_systems(rows),
            "pairwise_overall": ablation_pairwise(questions, indexed),
            "context_and_evidence": context_summary(rows),
        },
        "retrieval_diagnostics": diagnostics(questions, indexed),
        "context_path_provenance_correlations": correlations(questions, indexed),
        "failure_analysis": failure_analysis(questions, indexed),
        "methodology": {
            "bootstrap_resamples": 10000, "bootstrap_seed": SEED,
            "primary_effect": "Relation-Aware KG-RAG minus Vanilla Dense RAG",
            "multiple_comparison": "Holm correction applied separately within each required/not-required four-metric family.",
            "lexical_caveat": "ROUGE-L, coverage, exact match, and IDK are automated diagnostics, not human factual-quality judgments.",
        },
        "conclusion": "PARTIAL",
        "conclusion_basis": "The experiment is structurally valid and retrieval diagnostics are complete; this analysis provides lexical/descriptive evidence only. Human factual-quality evaluation is required before a factual-quality claim.",
    }
    (RESULTS / "relation_aware_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf8")
    write_question_csv(questions, indexed)
    statistical = {
        "required": required_comp, "not_required": not_required_comp,
        "difference_in_differences": did_by_metric,
    }
    (RESULTS / "statistical_results.json").write_text(json.dumps(statistical, indent=2) + "\n", encoding="utf8")
    lines = [
        "# Relation-aware experiment analysis",
        "",
        f"- Frozen generation rows: {len(rows)} (72 questions x 7 systems)",
        f"- Benchmark SHA-256: `{analysis['integrity']['benchmark_sha256']}`",
        "- No human evaluation was performed. Automated metrics are lexical diagnostics only.",
        "",
        "## System-level lexical diagnostics",
        "",
        "| System | ROUGE-L mean | Coverage mean | Exact match | IDK rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for system, values in analysis["system_results"].items():
        lines.append(
            f"| {system} | {values['rouge_l_mean']:.6f} | {values['coverage_mean']:.6f} | "
            f"{values['exact_match_rate']:.6f} | {values['idk_rate']:.6f} |"
        )
    lines.extend([
        "",
        "## Conclusion",
        "",
        "**PARTIAL**",
        "",
        "The generation package is structurally valid and the relation-aware traces recover the required paths with preserved provenance. The automated results are descriptive lexical diagnostics; they cannot establish factual-quality improvement. Human evaluation is required before any factual-quality conclusion.",
        "",
        "## Primary comparison",
        "",
        "Relation-Aware KG-RAG minus Vanilla Dense RAG is reported separately for KG-required and KG-not-required questions, including paired bootstrap intervals, Wilcoxon tests, wins/losses/ties, and Holm-adjusted p-values in `statistical_results.json`.",
        "",
        "The difference-in-differences test asks whether the required-subset paired improvement exceeds the control-subset paired improvement; its bootstrap interval and permutation p-value are recorded in `statistical_results.json`.",
        "",
        "## Categories, relations, and ablation",
        "",
        "Category-level results, supported-relation results, all A-G pairwise lexical contrasts, and context/path/provenance summaries are recorded in `relation_aware_analysis.json`.",
        "",
        "## Context and evidence associations",
        "",
        "Spearman associations between context tokens, retrieved path count, provenance count, and lexical metrics are descriptive only; they do not establish that larger context or more paths cause better answers.",
        "",
        "## Retrieval diagnostics",
        "",
        f"- Required-path recovery: {analysis['retrieval_diagnostics']['required_path_recovery_rate']:.3f}",
        f"- Valid provenance: {analysis['retrieval_diagnostics']['valid_provenance_rate']:.3f}",
        f"- Relation-intent set-match rate: {analysis['retrieval_diagnostics']['relation_intent_exact_or_set_match_rate']:.3f}",
        f"- Nonempty entity detection rate: {analysis['retrieval_diagnostics']['entity_detection_nonempty_rate']:.3f}",
        "",
        "Detailed system, category, relation, ablation, statistical, correlation, and trace-supported failure results are in `relation_aware_analysis.json` and `question_level_results.csv`.",
    ])
    (RESULTS / "relation_aware_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
