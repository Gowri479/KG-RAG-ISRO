"""Build the independent human spot-check package (preregistration §14).

A random 20 of the 60 primary questions, blank, for a rater who has not seen
the model-drafted or rater-1 scores.

    python -m src.evaluation.build_corrected_rerun_spotcheck
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from src.evaluation.build_corrected_rerun_human_package import ANNOTATIONS, BENCHMARK, RERUN, SEED, rater_csv

SPOTCHECK_SIZE = 20
SPOTCHECK_SEED = SEED + 14  # distinct stream, fixed in §14
BLANK = ANNOTATIONS / "corrected_rerun_spotcheck.csv"
FILLED = ANNOTATIONS / "corrected_rerun_spotcheck_filled.csv"
SELECTION = RERUN / "spotcheck_selection.json"


def select_questions() -> list[str]:
    questions = json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
    primary = sorted(q["question_id"] for q in questions if q["kg_required"] == "YES")
    return sorted(random.Random(SPOTCHECK_SEED).sample(primary, SPOTCHECK_SIZE))


def main() -> None:
    selected = set(select_questions())
    # The blank rater-1 package already holds blinded candidates and evidence packs.
    with rater_csv(1).open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [row for row in reader if row["question_id"] in selected]
    if len(rows) != SPOTCHECK_SIZE or any(v for row in rows for k, v in row.items() if k[:2] in ("A_", "B_", "C_")):
        raise RuntimeError("Spot-check source must be the blank package with all selected questions")
    with BLANK.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    SELECTION.write_text(json.dumps({"seed": SPOTCHECK_SEED, "size": SPOTCHECK_SIZE, "question_ids": sorted(selected)}, indent=2) + "\n", encoding="utf8")
    print(f"built {BLANK.relative_to(Path.cwd()) if BLANK.is_relative_to(Path.cwd()) else BLANK} with {len(rows)} questions")


if __name__ == "__main__":
    main()
