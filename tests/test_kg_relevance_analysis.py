import json

from src.evaluation.analyze_kg_relevance import OUTPUT_PATH
from src.evaluation.validate_kg_relevance import validate


def test_frozen_kg_relevance_annotations_validate_and_partition_180_questions():
    validation = validate()
    assert validation["status"] == "passed"
    assert validation["annotation_count"] == 180
    assert validation["unique_question_count"] == 180

    result = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    assert sum(result["annotation_counts"]["question_type"].values()) == 180
    assert sum(result["annotation_counts"]["kg_relevance"].values()) == 180
    assert sum(result["annotation_counts"]["kg_required"].values()) == 180
    assert result["primary_comparison"]["n"] == 0
    assert result["decision_gate"]["decision"] == "D. FROZEN BENCHMARK INSUFFICIENT"
    assert all(check["match"] for check in result["benchmark"]["aggregate_crosscheck"].values())
