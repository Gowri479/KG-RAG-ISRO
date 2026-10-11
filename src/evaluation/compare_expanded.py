"""Compare a separately generated expanded-corpus evaluation to the canonical result."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "data" / "results" / "evaluation_results.json"
EXPANDED = ROOT / "data" / "results" / "aditya_l1_expanded_evaluation.json"
OUTPUT = ROOT / "data" / "results" / "aditya_l1_expanded_comparison.json"
METRICS = ("rouge_l", "reference_token_coverage", "exact_match", "idk_rate")


def main() -> None:
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    expanded = json.loads(EXPANDED.read_text(encoding="utf-8"))
    canonical_ids = set(canonical["benchmark"]["test_ids"])
    expanded_ids = set(expanded["benchmark"]["test_ids"])
    if canonical_ids != expanded_ids:
        raise SystemExit("Expanded evaluation does not use the same frozen test IDs")

    systems = {}
    for system in ("bm25_llm", "vanilla_rag", "kg_rag"):
        before = canonical["overall"]["systems"][system]
        after = expanded["overall"]["systems"][system]
        systems[system] = {
            "canonical": {metric: before.get(metric) for metric in METRICS},
            "expanded": {metric: after.get(metric) for metric in METRICS},
            "delta_expanded_minus_canonical": {
                metric: round(after.get(metric, 0.0) - before.get(metric, 0.0), 4)
                for metric in METRICS
            },
        }

    report = {
        "comparison_protocol": "same frozen 180-question test IDs, model, prompts, and lexical metrics",
        "canonical_result": str(CANONICAL.relative_to(ROOT)),
        "expanded_result": str(EXPANDED.relative_to(ROOT)),
        "test_set_size": len(expanded_ids),
        "same_test_ids": True,
        "benchmark_hash_same": canonical["benchmark"].get("benchmark_hash") == expanded["benchmark"].get("benchmark_hash"),
        "systems": systems,
        "notes": "This comparison does not claim factual improvement; human-judged relational evaluation remains pending until labels are verified.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
