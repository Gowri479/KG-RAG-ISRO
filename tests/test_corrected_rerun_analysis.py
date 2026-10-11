import pytest

from src.evaluation.analyze_corrected_rerun import (
    analysis_sets,
    decision,
    holm,
    item_scores,
    krippendorff_alpha,
    mcnemar_p,
    wilcoxon_p,
)

# Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability": 4 coders x 12 units.
_CODERS = [
    [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
    [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
]
_UNITS = [list(unit) for unit in zip(*_CODERS)]


@pytest.mark.parametrize("metric,expected", [("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849)])
def test_krippendorff_alpha_matches_published_example(metric, expected):
    assert krippendorff_alpha(_UNITS, metric) == pytest.approx(expected, abs=0.001)


def test_krippendorff_alpha_perfect_agreement():
    assert krippendorff_alpha([[1, 1], [3, 3], [5, 5]]) == pytest.approx(1.0)


def test_holm_adjustment():
    assert holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    assert holm({"a": 0.03, "b": 0.02}) == {"a": 0.04, "b": 0.04}


def test_wilcoxon_all_ties_is_one():
    assert wilcoxon_p([0, 0, 0]) == 1.0


def test_mcnemar_counts_direction():
    p, worse, better = mcnemar_p([1, 1, 0, 0], [0, 0, 0, 1])
    assert (worse, better) == (2, 1) and 0 < p <= 1


def test_decision_rule():
    unsupported_ok = {"rate_diff": 0.0, "mcnemar_p": 1.0}
    unsupported_bad = {"rate_diff": 0.2, "mcnemar_p": 0.01}
    assert decision({"holm_p": 0.01, "mean_diff": 0.5}, unsupported_ok) == "SUPPORTED"
    assert decision({"holm_p": 0.01, "mean_diff": 0.5}, unsupported_bad).startswith("NOT SUPPORTED")
    assert decision({"holm_p": 0.01, "mean_diff": -0.5}, unsupported_ok) == "HARMFUL"
    assert decision({"holm_p": 0.20, "mean_diff": 0.5}, unsupported_ok) == "NOT SUPPORTED"


def test_item_scores_mean_and_any_flag():
    ratings = {("q", "V"): {"correctness": [4, 5], "completeness": [3, 3], "groundedness": [5, 4], "relevance": [5, 5], "unsupported_claim": [0, 1]}}
    scores = item_scores(ratings)[("q", "V")]
    assert scores["correctness"] == 4.5 and scores["unsupported_claim"] == 1


def test_analysis_sets_on_frozen_benchmark():
    import json
    from src.evaluation.build_corrected_rerun_human_package import BENCHMARK

    sets = analysis_sets(json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"])
    assert {k: len(v) for k, v in sets.items()} == {
        "primary_kg_required": 60,
        "kg_not_required": 12,
        "sensitivity_deduplicated": 45,
        "sensitivity_excluding_direct_control": 50,
        "heldout_descriptive": 13,
    }
