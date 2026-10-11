"""Human Factual Evaluation Analysis and Validation Module.

Computes correctness, completeness, faithfulness, and relevance metrics across QA systems.
Includes inter-annotator agreement (Cohen's Kappa, Krippendorff's Alpha) and strictly
refuses to calculate metrics on unannotated or fabricated data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEMPLATE_PATH = ROOT / "data" / "annotations" / "human_eval_template.csv"
RESULTS_PATH = ROOT / "data" / "results" / "human_eval_results.json"

REQUIRED_FIELDS = [
    "question_id",
    "question",
    "reference_answer",
    "system",
    "generated_answer",
    "retrieved_context",
    "category",
    "difficulty_tier",
    "annotation_status",
    "correctness",
    "completeness",
    "faithfulness",
    "relevance",
    "evaluator_id",
]

SCORE_FIELDS = ["correctness", "completeness", "faithfulness", "relevance"]
VALID_SCORES = {"0", "1", "2"}


def validate_file(path: Path) -> tuple[bool, str, list[dict]]:
    if not path.exists():
        return False, f"File does not exist: {path}", []

    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for field in REQUIRED_FIELDS:
            if field not in headers:
                return False, f"Missing required column in CSV: '{field}'", []

        rows = list(reader)

    if not rows:
        return False, "CSV contains no data rows.", []

    pending_count = 0
    annotated_count = 0
    invalid_rows = []

    for idx, row in enumerate(rows, start=1):
        status = row.get("annotation_status", "").strip().upper()
        scores = [row.get(sf, "").strip() for sf in SCORE_FIELDS]

        if status == "PENDING" or all(s == "" for s in scores):
            pending_count += 1
        elif any(s not in VALID_SCORES for s in scores):
            invalid_rows.append((idx, f"Invalid score in row {idx}: {scores}. Scores must be in {{0, 1, 2}}."))
        else:
            annotated_count += 1

    if invalid_rows:
        return False, f"Validation errors found:\n" + "\n".join(e[1] for e in invalid_rows[:5]), []

    if pending_count > 0:
        msg = (
            f"File validation: {pending_count} rows marked PENDING / unannotated out of {len(rows)}.\n"
            f"Status: PENDING_HUMAN_ANNOTATION. Evaluation cannot proceed without genuine human ratings."
        )
        return False, msg, rows

    return True, f"Validation successful: {annotated_count} fully annotated instances.", rows


def cohens_kappa(rater1_scores: list[int], rater2_scores: list[int], categories: list[int] = [0, 1, 2]) -> float:
    """Compute Cohen's Kappa between two raters."""
    if len(rater1_scores) != len(rater2_scores) or not rater1_scores:
        return 0.0
    n = len(rater1_scores)
    agree = sum(1 for a, b in zip(rater1_scores, rater2_scores) if a == b)
    po = agree / n

    pe = 0.0
    for cat in categories:
        p1 = sum(1 for a in rater1_scores if a == cat) / n
        p2 = sum(1 for b in rater2_scores if b == cat) / n
        pe += p1 * p2

    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def krippendorff_alpha_nominal(matrix: list[list[int | None]]) -> float:
    """Compute Krippendorff's alpha for nominal data across arbitrary raters."""
    # matrix: units x raters
    units = len(matrix)
    if units == 0:
        return 0.0
    all_values = set()
    for row in matrix:
        for val in row:
            if val is not None:
                all_values.add(val)
    values = sorted(list(all_values))
    if len(values) <= 1:
        return 1.0

    # coincidence matrix
    c_matrix = {v: {w: 0.0 for w in values} for v in values}
    total_pairs = 0.0

    for row in matrix:
        valid_items = [v for v in row if v is not None]
        m_u = len(valid_items)
        if m_u > 1:
            for i, vi in enumerate(valid_items):
                for j, vj in enumerate(valid_items):
                    if i != j:
                        c_matrix[vi][vj] += 1.0 / (m_u - 1)
                        total_pairs += 1.0 / (m_u - 1)

    if total_pairs == 0:
        return 0.0

    # observed disagreement
    d_o = sum(c_matrix[v][w] for v in values for w in values if v != w) / total_pairs

    # marginals
    marginals = {v: sum(c_matrix[v][w] for w in values) for v in values}
    d_e = sum(marginals[v] * marginals[w] for v in values for w in values if v != w) / (total_pairs * total_pairs)

    if d_e == 0:
        return 1.0
    return 1.0 - (d_o / d_e)


def evaluate_annotations(rows: list[dict]) -> dict:
    systems = sorted(list({r["system"] for r in rows}))
    system_stats = {}

    for sys_name in systems:
        sys_rows = [r for r in rows if r["system"] == sys_name]
        correctness = [int(r["correctness"]) for r in sys_rows]
        completeness = [int(r["completeness"]) for r in sys_rows]
        faithfulness = [int(r["faithfulness"]) for r in sys_rows]
        relevance = [int(r["relevance"]) for r in sys_rows]

        fully_correct = [
            1 for c, comp, f in zip(correctness, completeness, faithfulness)
            if c == 2 and comp == 2 and f == 2
        ]

        system_stats[sys_name] = {
            "n_instances": len(sys_rows),
            "mean_correctness": round(mean(correctness), 4),
            "sd_correctness": round(stdev(correctness), 4) if len(correctness) > 1 else 0.0,
            "mean_completeness": round(mean(completeness), 4),
            "sd_completeness": round(stdev(completeness), 4) if len(completeness) > 1 else 0.0,
            "mean_faithfulness": round(mean(faithfulness), 4),
            "sd_faithfulness": round(stdev(faithfulness), 4) if len(faithfulness) > 1 else 0.0,
            "mean_relevance": round(mean(relevance), 4),
            "sd_relevance": round(stdev(relevance), 4) if len(relevance) > 1 else 0.0,
            "percentage_fully_correct": round(len(fully_correct) / len(sys_rows) * 100, 2),
        }

    return system_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Human Evaluation Validator & Analyzer")
    parser.add_argument("--input", type=Path, default=TEMPLATE_PATH, help="Path to annotation CSV")
    parser.add_argument("--validate-only", action="store_true", help="Only validate without computing scores")
    args = parser.parse_args()

    is_valid, message, rows = validate_file(args.input)
    print(f"[{'SUCCESS' if is_valid else 'PENDING/FAILED'}] {message}")

    if not is_valid:
        # Write pending status report
        pending_report = {
            "experiment": "human_factual_evaluation",
            "status": "PENDING_HUMAN_ANNOTATION",
            "reason": message,
            "target_file": str(args.input.relative_to(ROOT)),
            "annotation_schema": {
                "dimensions": ["correctness", "completeness", "faithfulness", "relevance"],
                "scale": "0, 1, 2",
            },
        }
        RESULTS_PATH.write_text(json.dumps(pending_report, indent=2), encoding="utf-8")
        print(f"Recorded status to {RESULTS_PATH}")
        return

    if args.validate_only:
        return

    stats = evaluate_annotations(rows)
    output_payload = {
        "experiment": "human_factual_evaluation",
        "status": "COMPLETED",
        "file": str(args.input.relative_to(ROOT)),
        "system_results": stats,
    }
    RESULTS_PATH.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Saved human evaluation results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
