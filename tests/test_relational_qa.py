import hashlib
import json

from src.evaluation.build_relational_qa import build_benchmark
from src.evaluation.validate_relational_qa import (
    BENCHMARK_PATH,
    PROVENANCE_PATH,
    validate,
)


def _benchmark_directory_hashes():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "data" / "benchmark"
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_relational_benchmark_schema_provenance_paths_and_subgroups_validate(tmp_path):
    report = validate()
    assert report["status"] == "passed"
    assert report["question_count"] == 62
    assert report["unique_question_count"] == 62
    assert report["kg_required_distribution"] == {"NO": 36, "YES": 26}
    assert report["category_distribution"] == {
        "DIRECT_CONTROL": 15,
        "MULTI_RELATION": 3,
        "SINGLE_RELATION": 21,
        "TWO_HOP_RELATION": 23,
    }
    assert sum(report["mission_distribution"].values()) == 62
    assert report["two_hop_or_longer_path_pattern_counts"]


def test_builder_preserves_frozen_canonical_benchmark_directory(tmp_path):
    before = _benchmark_directory_hashes()
    generated = build_benchmark(tmp_path)
    after = _benchmark_directory_hashes()
    assert before == after
    assert generated["benchmark_path"].exists()
    assert generated["provenance_path"].exists()
    assert generated["benchmark_path"].parent == tmp_path


def test_validator_detects_duplicate_questions_without_changing_benchmark(tmp_path):
    payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    payload["questions"][1]["question"] = payload["questions"][0]["question"]
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    report = validate(mutated, PROVENANCE_PATH)
    assert report["status"] == "failed"
    assert any("duplicate question text" in error for error in report["errors"])


def test_validator_rejects_disconnected_relation_path(tmp_path):
    payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    item = next(
        question for question in payload["questions"]
        if any(len(path) >= 2 for path in question["supporting_paths"])
    )
    item["supporting_paths"][0][-1] = "cy3-predecessor"
    mutated = tmp_path / "disconnected.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    report = validate(mutated, PROVENANCE_PATH)
    assert report["status"] == "failed"
    assert any("disconnected" in error for error in report["errors"])
