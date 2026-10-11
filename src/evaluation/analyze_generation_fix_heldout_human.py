"""Analyze the frozen 13-question held-out human evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "annotations" / "generation_fix_heldout_human_eval_filled.csv"
MAPPING = ROOT / "data" / "results" / "generation_fix" / "heldout_human_eval_mapping.json"
MANIFEST = ROOT / "data" / "results" / "generation_fix" / "heldout_human_eval_manifest.json"
BENCHMARK = ROOT / "data" / "relation_aware_benchmark" / "relation_aware_qa_v2.json"
OUT = ROOT / "data" / "results" / "generation_fix"
SEED = 20261008
BOOTSTRAP_RESAMPLES = 10000

SYSTEMS = ("vanilla_dense_rag", "A_CURRENT", "C_TWO_STAGE")
DISPLAY = {
    "vanilla_dense_rag": "Vanilla Dense RAG",
    "A_CURRENT": "Current Relation-Aware KG-RAG",
    "C_TWO_STAGE": "Two-Stage KG-Grounded Generation",
}
DIMS = ("correctness", "completeness", "groundedness", "relevance")
ALL_SCORE_FIELDS = (
    "correctness",
    "completeness",
    "groundedness",
    "relevance",
    "unsupported_claim",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def bootstrap(values: list[float], seed_offset: int) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(SEED + seed_offset)
    samples = [
        mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def wilcoxon(values: list[float]) -> dict:
    nonzero = [value for value in values if value != 0]
    if len(nonzero) < 2:
        return {"statistic": None, "p_value": None, "n_nonzero": len(nonzero)}
    from scipy.stats import wilcoxon

    result = wilcoxon(values, zero_method="wilcox", method="auto")
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_nonzero": len(nonzero),
    }


def paired_statistics(differences: list[float], seed_offset: int) -> dict:
    spread = stdev(differences) if len(differences) > 1 else 0.0
    return {
        "n": len(differences),
        "mean_difference": mean(differences),
        "median_difference": median(differences),
        "bootstrap_95_ci": bootstrap(differences, seed_offset),
        "wilcoxon_signed_rank": wilcoxon(differences),
        "paired_dz": mean(differences) / spread if spread else None,
        "wins": sum(value > 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "differences": differences,
    }


def holm(p_values: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted(
        ((key, value) for key, value in p_values.items() if value is not None),
        key=lambda item: item[1],
    )
    adjusted: dict[str, float] = {}
    previous = 0.0
    for index, (key, value) in enumerate(valid):
        previous = max(previous, min(1.0, (len(valid) - index) * value))
        adjusted[key] = previous
    return {key: adjusted.get(key) for key in p_values}


def load_scores() -> tuple[list[dict], dict[str, dict[str, dict[str, float]]], dict[str, dict]]:
    rows = list(csv.DictReader(INPUT.open(encoding="utf-8-sig", newline="")))
    mapping = json.loads(MAPPING.read_text(encoding="utf8"))["mapping"]
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
    questions = {
        item["question_id"]: item
        for item in benchmark
        if item["kg_required"] == "YES" and item["split"] == "evaluation"
    }
    if len(rows) != 13 or set(row["question_id"] for row in rows) != set(questions):
        raise RuntimeError("Expected exactly the 13 held-out questions")

    values: dict[str, dict[str, dict[str, float]]] = {
        system: {} for system in SYSTEMS
    }
    for row in rows:
        qid = row["question_id"]
        for label in ("A", "B", "C"):
            system = mapping[qid]["label_to_system"][label]
            for dimension in ALL_SCORE_FIELDS:
                raw = row[f"{label}_{dimension}"]
                value = float(raw)
                if dimension == "unsupported_claim":
                    if value not in (0, 1):
                        raise RuntimeError(f"Invalid unsupported flag: {qid} {label}")
                elif not 1 <= value <= 5:
                    raise RuntimeError(f"Invalid score: {qid} {label} {dimension}")
                values[system].setdefault(qid, {})[dimension] = value
    return rows, values, questions


def system_summary(values: dict[str, dict[str, float]], qids: list[str]) -> dict:
    result = {"n": len(qids)}
    for dimension in ALL_SCORE_FIELDS:
        numbers = [values[qid][dimension] for qid in qids]
        result[dimension] = {
            "mean": mean(numbers),
            "median": median(numbers),
            "standard_deviation": stdev(numbers) if len(numbers) > 1 else 0.0,
            "minimum": min(numbers),
            "maximum": max(numbers),
            **({"rate": mean(numbers), "percentage": 100.0 * mean(numbers)} if dimension == "unsupported_claim" else {}),
        }
    return result


def pairwise(values: dict[str, dict[str, dict[str, float]]], left: str, right: str, qids: list[str]) -> dict:
    result = {}
    for index, dimension in enumerate(DIMS):
        differences = [
            values[left][qid][dimension] - values[right][qid][dimension]
            for qid in qids
        ]
        result[dimension] = paired_statistics(differences, index + (0 if right == "vanilla_dense_rag" else 100))
    unsupported_differences = [
        values[left][qid]["unsupported_claim"] - values[right][qid]["unsupported_claim"]
        for qid in qids
    ]
    result["unsupported_claim"] = {
        "left_rate": mean(values[left][qid]["unsupported_claim"] for qid in qids),
        "right_rate": mean(values[right][qid]["unsupported_claim"] for qid in qids),
        "difference_percentage_points": 100.0 * mean(unsupported_differences),
        "left_minus_right_differences": unsupported_differences,
        "improves": sum(value < 0 for value in unsupported_differences),
        "worsens": sum(value > 0 for value in unsupported_differences),
        "unchanged": sum(value == 0 for value in unsupported_differences),
    }
    return result


def main() -> None:
    rows, values, questions = load_scores()
    qids = list(questions)
    comparisons = {
        "C_TWO_STAGE_minus_A_VANILLA": pairwise(values, "C_TWO_STAGE", "vanilla_dense_rag", qids),
        "C_TWO_STAGE_minus_B_RELATION_AWARE": pairwise(values, "C_TWO_STAGE", "A_CURRENT", qids),
    }
    primary_p = {
        f"{comparison}:{dimension}": result[dimension]["wilcoxon_signed_rank"]["p_value"]
        for comparison, result in comparisons.items()
        for dimension in DIMS
    }
    holm_adjusted = holm(primary_p)
    category_results = {}
    for category in sorted({question["category"] for question in questions.values()}):
        category_qids = [qid for qid in qids if questions[qid]["category"] == category]
        category_results[category] = {
            "n": len(category_qids),
            "systems": {
                system: system_summary(values[system], category_qids)
                for system in SYSTEMS
            },
        }

    question_rows = []
    for qid in qids:
        row = {"question_id": qid, "category": questions[qid]["category"]}
        for system in SYSTEMS:
            prefix = {"vanilla_dense_rag": "A", "A_CURRENT": "B", "C_TWO_STAGE": "C"}[system]
            for dimension in ALL_SCORE_FIELDS:
                row[f"{prefix}_{dimension}"] = values[system][qid][dimension]
        for dimension in DIMS:
            row[f"C_minus_A_{dimension}"] = (
                values["C_TWO_STAGE"][qid][dimension] - values["vanilla_dense_rag"][qid][dimension]
            )
            row[f"C_minus_B_{dimension}"] = (
                values["C_TWO_STAGE"][qid][dimension] - values["A_CURRENT"][qid][dimension]
            )
        row["C_minus_A_unsupported_claim"] = (
            values["C_TWO_STAGE"][qid]["unsupported_claim"]
            - values["vanilla_dense_rag"][qid]["unsupported_claim"]
        )
        row["C_minus_B_unsupported_claim"] = (
            values["C_TWO_STAGE"][qid]["unsupported_claim"]
            - values["A_CURRENT"][qid]["unsupported_claim"]
        )
        question_rows.append(row)

    illustrative = {}
    for dimension in DIMS:
        illustrative[dimension] = {
            "largest_C_minus_A_improvement": max(question_rows, key=lambda row: row[f"C_minus_A_{dimension}"])["question_id"],
            "largest_C_minus_A_degradation": min(question_rows, key=lambda row: row[f"C_minus_A_{dimension}"])["question_id"],
        }

    analysis = {
        "integrity": {
            "input_csv_sha256": sha256(INPUT),
            "mapping_sha256": sha256(MAPPING),
            "heldout_manifest_sha256": sha256(MANIFEST),
            "benchmark_sha256": sha256(BENCHMARK),
            "question_count": len(qids),
            "candidate_count": len(rows) * 3,
            "mapping_integrity": True,
            "score_analysis_performed": True,
        },
        "systems": {system: system_summary(values[system], qids) for system in SYSTEMS},
        "comparisons": comparisons,
        "holm_adjusted_p_values": holm_adjusted,
        "category_results": category_results,
        "question_level_illustrative_extremes": illustrative,
        "automated_context": {
            "A_CURRENT": {"rouge_l": 0.104, "coverage": 0.223},
            "B_STRUCTURED": {"rouge_l": 0.070, "coverage": 0.176},
            "C_TWO_STAGE": {"rouge_l": 0.233, "coverage": 0.412},
            "C_minus_A": {
                "rouge_l_mean_difference": 0.129947,
                "coverage_mean_difference": 0.188462,
                "rouge_l_wilcoxon_p": 0.097656,
                "coverage_wilcoxon_p": 0.125,
            },
        },
        "decision": "NOT SUPPORTED",
        "paper_implication": {
            "proposed_final_method": "Do not replace the proposed method solely on this N=13 evaluation.",
            "two_stage_contribution": "Can be framed as a promising generation-bottleneck result, not a demonstrated superiority claim.",
            "superiority_claim_supported": False,
            "limitations": [
                "Only 13 held-out questions were human-rated.",
                "Paired tests are underpowered and Holm-adjusted evidence is not statistically conclusive.",
                "Automated ROUGE-L and coverage are lexical diagnostics.",
                "Unsupported-claim behavior must be interpreted descriptively.",
            ],
        },
        "method": {
            "unit": "question",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": SEED,
            "wilcoxon": "two-sided, zero_method=wilcox, method=auto",
            "holm_family": "8 paired tests: C-A and C-B across four quality dimensions",
            "unsupported_claims": "binary rates and paired directional counts; not included in Holm quality-dimension family",
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "heldout_human_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf8",
    )
    with (OUT / "heldout_human_question_level_results.csv").open("w", newline="", encoding="utf8") as handle:
        fields = list(question_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(question_rows)
    (OUT / "heldout_human_statistics.json").write_text(
        json.dumps({"comparisons": comparisons, "holm_adjusted_p_values": holm_adjusted}, indent=2) + "\n",
        encoding="utf8",
    )
    manifest = {
        "input_csv_sha256": sha256(INPUT),
        "mapping_sha256": sha256(MAPPING),
        "benchmark_sha256": sha256(BENCHMARK),
        "heldout_manifest_sha256": sha256(MANIFEST),
        "question_count": len(qids),
        "system_count": len(SYSTEMS),
        "systems": {system: DISPLAY[system] for system in SYSTEMS},
        "analysis_timestamp_utc": analysis["timestamp_utc"],
        "analysis_method": analysis["method"],
        "score_columns_used": [f"{label}_{dimension}" for label in ("A", "B", "C") for dimension in ALL_SCORE_FIELDS],
        "output_files": [
            "data/results/generation_fix/heldout_human_analysis.json",
            "data/results/generation_fix/heldout_human_analysis.md",
            "data/results/generation_fix/heldout_human_question_level_results.csv",
            "data/results/generation_fix/heldout_human_statistics.json",
        ],
    }
    (OUT / "heldout_human_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")

    report = [
        "# Held-out human evaluation analysis",
        "",
        "## Scope and integrity",
        "",
        f"- 13 held-out questions and {len(rows) * 3} candidate evaluations were decoded through the private mapping.",
        "- No generation, retrieval, prompt, benchmark, or annotation step was rerun.",
        f"- Input CSV SHA-256: `{sha256(INPUT)}`",
        "",
        "## Human mean scores",
        "",
        "| System | Correctness | Completeness | Groundedness | Relevance | Unsupported claims |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        summary = analysis["systems"][system]
        report.append(
            f"| {DISPLAY[system]} | {summary['correctness']['mean']:.3f} | "
            f"{summary['completeness']['mean']:.3f} | {summary['groundedness']['mean']:.3f} | "
            f"{summary['relevance']['mean']:.3f} | {summary['unsupported_claim']['percentage']:.1f}% |"
        )
    report += ["", "## Paired results", ""]
    for comparison_name, comparison_result in comparisons.items():
        report += [f"### {comparison_name}", "", "| Dimension | Mean Δ | Median Δ | Bootstrap 95% CI | Wilcoxon p | Holm p | dz | W/L/T |", "|---|---:|---:|---|---:|---:|---:|---|"]
        for dimension in DIMS:
            item = comparison_result[dimension]
            key = f"{comparison_name}:{dimension}"
            report.append(
                f"| {dimension} | {item['mean_difference']:.3f} | {item['median_difference']:.3f} | "
                f"[{item['bootstrap_95_ci'][0]:.3f}, {item['bootstrap_95_ci'][1]:.3f}] | "
                f"{item['wilcoxon_signed_rank']['p_value']:.4f} | {holm_adjusted[key]:.4f} | "
                f"{item['paired_dz']:.3f} | {item['wins']}/{item['losses']}/{item['ties']} |"
            )
        unsupported = comparison_result["unsupported_claim"]
        report += [
            "",
            f"Unsupported claims: C rate {unsupported['left_rate']:.3f}, comparator rate {unsupported['right_rate']:.3f}; "
            f"difference {unsupported['difference_percentage_points']:.1f} percentage points; "
            f"C improves/worsens/unchanged = {unsupported['improves']}/{unsupported['worsens']}/{unsupported['unchanged']}.",
            "",
        ]
    report += ["## Category observations", ""]
    for category, item in category_results.items():
        report.append(f"### {category} (N={item['n']})")
        report.append("Descriptive only; category sample sizes are too small for reliable inference.")
        for system in SYSTEMS:
            summary = item["systems"][system]
            report.append(
                f"- {DISPLAY[system]}: correctness {summary['correctness']['mean']:.3f}, "
                f"completeness {summary['completeness']['mean']:.3f}, "
                f"groundedness {summary['groundedness']['mean']:.3f}, "
                f"relevance {summary['relevance']['mean']:.3f}, "
                f"unsupported {summary['unsupported_claim']['percentage']:.1f}%"
            )
        report.append("")
    report += [
        "## Automated versus human results",
        "",
        "C improved held-out ROUGE-L and coverage over A descriptively, but the automated Wilcoxon p-values were 0.097656 and 0.125. The human comparison must not be interpreted as validated merely because lexical metrics improved.",
        "",
        "## Decision",
        "",
        "**NOT SUPPORTED**",
        "",
        "The primary C-versus-A comparison is descriptively worse on correctness, completeness, and groundedness, tied on relevance, and has a higher unsupported-claim rate. The C-versus-B comparison is mixed and not statistically conclusive. The two-stage method therefore does not demonstrate improved human-rated factual quality.",
        "",
        "## Paper implication",
        "",
        "Do not replace the proposed final method solely on this human evaluation. The two-stage architecture may be presented as a promising generation-bottleneck direction, with limitations and without a superiority claim. `paper/main.tex` was not modified.",
    ]
    (OUT / "heldout_human_analysis.md").write_text("\n".join(report) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
