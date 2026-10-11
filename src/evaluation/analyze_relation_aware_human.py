"""Analyze the completed blinded relation-aware human evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(r"d:\Users\NANS\Downloads\relation_aware_human_eval_filled.csv")
ANNOTATIONS = ROOT / "data" / "annotations"
RESULTS = ROOT / "data" / "results" / "relation_aware"
PACKAGE = ANNOTATIONS / "relation_aware_human_eval.csv"
MAPPING = RESULTS / "relation_aware_human_eval_mapping.json"
BENCHMARK = ROOT / "data" / "relation_aware_benchmark" / "relation_aware_qa_v2.json"
SUBMITTED = ANNOTATIONS / "relation_aware_human_eval_filled.csv"
SYSTEMS = ("vanilla_dense_rag", "corrected_structured_kg_rag", "relation_aware_kg_rag")
DISPLAY = {
    "vanilla_dense_rag": "Vanilla Dense RAG",
    "corrected_structured_kg_rag": "Corrected Structured KG-RAG",
    "relation_aware_kg_rag": "Relation-Aware KG-RAG",
}
DIMENSIONS = ("correctness", "completeness", "groundedness", "relevance", "unsupported_claim")
SEED = 20261008


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(values: list[float], resamples: int = 10000) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(SEED)
    samples = [mean(values[rng.randrange(len(values))] for _ in values) for _ in range(resamples)]
    ordered = sorted(samples)
    return [ordered[int(.025 * (len(ordered) - 1))], ordered[int(.975 * (len(ordered) - 1))]]


def test(differences: list[float]) -> dict:
    nonzero = [x for x in differences if x]
    if len(nonzero) < 2:
        return {"statistic": None, "p_value": None, "n_nonzero": len(nonzero)}
    from scipy.stats import wilcoxon
    result = wilcoxon(differences, zero_method="wilcox", method="auto")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "n_nonzero": len(nonzero)}


def paired(left: list[float], right: list[float]) -> dict:
    differences = [a - b for a, b in zip(left, right)]
    positive = sum(x > 0 for x in differences)
    negative = sum(x < 0 for x in differences)
    return {
        "n": len(differences),
        "left_mean": mean(left),
        "right_mean": mean(right),
        "mean_difference": mean(differences),
        "median_difference": median(differences),
        "bootstrap_95_ci": bootstrap(differences),
        "wilcoxon": test(differences),
        "wins_losses_ties": [positive, negative, len(differences) - positive - negative],
        "differences": differences,
    }


def holm(values: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted(((key, value) for key, value in values.items() if value is not None), key=lambda x: x[1])
    result = {}
    previous = 0.0
    for index, (key, value) in enumerate(valid):
        previous = max(previous, min(1.0, (len(valid) - index) * value))
        result[key] = previous
    for key in values:
        result.setdefault(key, None)
    return result


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    shutil.copyfile(SOURCE, SUBMITTED)
    rows = list(csv.DictReader(SOURCE.open(encoding="utf8", newline="")))
    package_rows = {row["question_id"]: row for row in csv.DictReader(PACKAGE.open(encoding="utf8", newline=""))}
    mapping = json.loads(MAPPING.read_text(encoding="utf8"))["mapping"]
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
    required = [q for q in benchmark if q["kg_required"] == "YES"]
    label_by_system = {
        qid: {info["system"]: label for label, info in labels.items()}
        for qid, labels in mapping.items()
    }
    values = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        qid = row["question_id"]
        for system in SYSTEMS:
            label = label_by_system[qid][system]
            for dimension in DIMENSIONS:
                values[system][qid][dimension] = float(row[f"{label}_{dimension}"])
    summaries = {}
    for system in SYSTEMS:
        summaries[system] = {"system": DISPLAY[system], "n": len(required)}
        for dimension in DIMENSIONS:
            numbers = [values[system][q["question_id"]][dimension] for q in required]
            summaries[system][dimension] = {
                "mean": mean(numbers),
                "median": median(numbers),
                "n": len(numbers),
            }
    comparisons = {}
    pvalues = {}
    for system in SYSTEMS[1:]:
        comparisons[system] = {}
        for dimension in DIMENSIONS:
            result = paired(
                [values[system][q["question_id"]][dimension] for q in required],
                [values["vanilla_dense_rag"][q["question_id"]][dimension] for q in required],
            )
            comparisons[system][dimension] = result
            pvalues[f"{system}:{dimension}"] = result["wilcoxon"]["p_value"]
    question_rows = []
    for q in required:
        qid = q["question_id"]
        item = {"question_id": qid, "question": q["question"], "category": q["category"]}
        for system in SYSTEMS:
            prefix = system.replace("_", "")
            for dimension in DIMENSIONS:
                item[f"{prefix}_{dimension}"] = values[system][qid][dimension]
        question_rows.append(item)
    result = {
        "input": {
            "submitted_csv_sha256": sha256(SOURCE),
            "stored_submitted_csv_sha256": sha256(SUBMITTED),
            "benchmark_sha256": sha256(BENCHMARK),
            "mapping_sha256": sha256(MAPPING),
            "questions": len(required),
            "annotations": len(rows) * 3,
        },
        "systems": summaries,
        "primary_comparisons_vs_vanilla": comparisons,
        "holm_adjusted_p_values": holm(pvalues),
        "method": {
            "unit": "question",
            "bootstrap_resamples": 10000,
            "seed": SEED,
            "unsupported_claim_interpretation": "reported as a binary 0/1 rate; paired testing is descriptive and not treated as a continuous quality scale.",
        },
        "conclusion": "PENDING_CONSERVATIVE_INTERPRETATION",
        "note": "Human results are analyzed against the submitted scores. This artifact does not modify benchmark, generation, retrieval, corpus, paper, or prior experiment outputs.",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (RESULTS / "human_evaluation_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    with (RESULTS / "human_question_level_results.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(question_rows[0]))
        writer.writeheader()
        writer.writerows(question_rows)
    lines = [
        "# Relation-aware human evaluation analysis",
        "",
        f"- Questions: {len(required)}",
        f"- Candidate annotations: {len(rows) * 3}",
        f"- Submitted CSV SHA-256: `{sha256(SOURCE)}`",
        "",
        "## System means and medians",
        "",
        "| System | Correctness mean | Completeness mean | Groundedness mean | Relevance mean | Unsupported claim rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        s = summaries[system]
        lines.append(
            f"| {s['system']} | {s['correctness']['mean']:.3f} | {s['completeness']['mean']:.3f} | "
            f"{s['groundedness']['mean']:.3f} | {s['relevance']['mean']:.3f} | "
            f"{s['unsupported_claim']['mean']:.3f} |"
        )
    lines += [
        "",
        "## Pairwise tests",
        "",
        "Paired differences, bootstrap intervals, Wilcoxon tests, wins/losses/ties, and Holm-adjusted p-values are in `human_evaluation_analysis.json`.",
        "",
        "These results are from human annotation, but interpretation remains conservative until category-level and task-conditional review is complete.",
    ]
    (RESULTS / "human_evaluation_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
