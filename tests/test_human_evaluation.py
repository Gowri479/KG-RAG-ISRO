import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "relational_benchmark" / "relational_qa_v1.json"
REVIEW = ROOT / "data" / "results" / "relational_qa_v1" / "human_evaluation_review.csv"


def test_review_package_is_blinded_and_unscored():
    assert hashlib.sha256(BENCHMARK.read_bytes()).hexdigest() == (
        "7f5daf5419b0a4cb70e5c3d5eba021a8a39118b5a0fd235ce19b81f75324ffce"
    )
    with REVIEW.open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 62
    assert len(reader.fieldnames) == 23
    assert all(not row[field] for row in rows for field in reader.fieldnames if field[0] in "ABC" and "_" in field)
