"""Compare the independent spot-check with the rater-1 scores (preregistration §14).

Kept separate from analyze_corrected_rerun.py so the primary analysis code
stays unchanged.

    python -m src.evaluation.analyze_corrected_rerun_spotcheck
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.evaluation.analyze_corrected_rerun import describe, krippendorff_alpha
from src.evaluation.build_corrected_rerun_human_package import ANNOTATIONS, DIMENSIONS, MAPPING, RERUN, answer_hash
from src.evaluation.build_corrected_rerun_spotcheck import FILLED, select_questions

RATER1 = ANNOTATIONS / "corrected_rerun_human_eval_rater1_filled.csv"
OUTPUT = RERUN / "spotcheck_analysis.json"
ALPHA_THRESHOLD = 0.667
PRIMARY = (("A_CURRENT", "V_VANILLA"), ("C_TWO_STAGE", "V_VANILLA"))


def read_scores(path: Path, question_ids: list[str]) -> dict[tuple[str, str], dict[str, int]]:
    mapping = json.loads(MAPPING.read_text(encoding="utf8"))["mapping"]
    rows = {r["question_id"]: r for r in csv.DictReader(path.open(encoding="utf-8-sig"))}
    missing = [q for q in question_ids if q not in rows]
    if missing:
        raise RuntimeError(f"{path.name}: missing questions {missing}")
    scores = {}
    for qid in question_ids:
        row = rows[qid]
        for label, system in mapping[qid]["label_to_system"].items():
            if answer_hash(row[f"candidate_{label}"]) != mapping[qid]["label_hashes"][label]:
                raise RuntimeError(f"{path.name}: candidate hash mismatch {qid} {label}")
            cell = {}
            for dimension in DIMENSIONS:
                raw = row[f"{label}_{dimension}"].strip()
                value = int(float(raw)) if raw and float(raw).is_integer() else None
                allowed = (0, 1) if dimension == "unsupported_claim" else (1, 2, 3, 4, 5)
                if value not in allowed:
                    raise RuntimeError(f"{path.name}: invalid {dimension}={raw!r} for {qid} {label}")
                cell[dimension] = value
            scores[(qid, system)] = cell
    return scores


def agreement_between(a: dict, b: dict) -> dict:
    keys = sorted(a)
    return {
        "krippendorff_alpha": {
            d: krippendorff_alpha([[a[k][d], b[k][d]] for k in keys], "nominal" if d == "unsupported_claim" else "ordinal")
            for d in DIMENSIONS
        },
        "exact_agreement": {d: round(sum(a[k][d] == b[k][d] for k in keys) / len(keys), 4) for d in DIMENSIONS},
    }


def main(filled: Path = FILLED, output: Path = OUTPUT, compare_with: Path | None = None) -> None:
    ids = select_questions()
    checker, rater1 = read_scores(filled, ids), read_scores(RATER1, ids)
    keys = sorted(checker)
    agreement = {
        d: krippendorff_alpha([[rater1[k][d], checker[k][d]] for k in keys], "nominal" if d == "unsupported_claim" else "ordinal")
        for d in DIMENSIONS
    }
    exact = {d: round(sum(rater1[k][d] == checker[k][d] for k in keys) / len(keys), 4) for d in DIMENSIONS}
    direction = {}
    for t, c in PRIMARY:
        by_rater = {
            name: describe([s[(q, t)]["correctness"] - s[(q, c)]["correctness"] for q in ids])
            for name, s in (("checker", checker), ("rater1", rater1))
        }
        by_rater["same_direction"] = (by_rater["checker"]["mean_diff"] > 0) == (by_rater["rater1"]["mean_diff"] > 0)
        direction[f"{t}_vs_{c}"] = by_rater
    passed = (agreement["correctness"] or 0) >= ALPHA_THRESHOLD and all(v["same_direction"] for v in direction.values())
    result = {
        "preregistration": "reports/preregistration_corrected_rerun.md §14",
        "checker_file": filled.name,
        "questions": ids,
        "units": len(keys),
        "krippendorff_alpha": agreement,
        "exact_agreement": exact,
        "correctness_differences": direction,
        "criterion": f"correctness ordinal alpha >= {ALPHA_THRESHOLD} and checker's A-V and C-V correctness differences in the same direction as rater 1",
        "validation": "PASSED" if passed else "FAILED",
    }
    others = [compare_with] if isinstance(compare_with, Path) else list(compare_with or [])
    if len(others) == 1:
        result["agreement_with_other_checker"] = {"other_checker_file": others[0].name, **agreement_between(checker, read_scores(others[0], ids))}
    elif others:
        scored = {path.name: read_scores(path, ids) for path in others}
        result["agreement_with_other_checkers"] = {name: agreement_between(checker, s) for name, s in scored.items()}
        result["pooled_alpha_all_checkers"] = {
            d: krippendorff_alpha([[checker[k][d], *(s[k][d] for s in scored.values())] for k in sorted(checker)], "nominal" if d == "unsupported_claim" else "ordinal")
            for d in DIMENSIONS
        }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: result[k] for k in ("krippendorff_alpha", "exact_agreement", "validation", "agreement_with_other_checker", "agreement_with_other_checkers", "pooled_alpha_all_checkers") if k in result}, indent=2))
    for key, v in direction.items():
        print(key, "checker", v["checker"]["mean_diff"], v["checker"]["ci95"], "| rater1", v["rater1"]["mean_diff"], v["rater1"]["ci95"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--filled", type=Path, default=FILLED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--compare-with", type=Path, nargs="+", help="Other checkers' filled CSVs, for checker-vs-checker agreement")
    args = parser.parse_args()
    main(args.filled, args.output, args.compare_with)
