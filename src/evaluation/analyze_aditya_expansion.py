"""Compare canonical and expanded evaluations and classify source relevance."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "data" / "results" / "evaluation_results.json"
EXPANDED = ROOT / "data" / "results" / "evaluation_results_aditya_expanded.json"
CHUNKS = ROOT / "data" / "chunks" / "chunks.json"
COMPARISON = ROOT / "data" / "results" / "aditya_l1_expansion_comparison.json"
SOURCE_ANALYSIS = ROOT / "data" / "results" / "aditya_l1_question_source_analysis.json"
METRICS = ("rouge_l", "reference_token_coverage", "exact_match", "idk_rate")
SYSTEMS = ("bm25_llm", "vanilla_rag", "kg_rag")
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and", "or", "in", "on", "at",
    "for", "from", "by", "with", "what", "which", "when", "where", "how", "did", "does", "do", "has",
}


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 2 and token not in STOPWORDS}


def _relative_change(baseline: float, expanded: float) -> float | None:
    if baseline == 0:
        return None
    return round((expanded - baseline) / abs(baseline), 6)


def _source_evidence(reference: str, tokenized_chunks: list[tuple[set[str], str | None]]) -> tuple[bool, float, str | None]:
    reference_tokens = _tokens(reference)
    if not reference_tokens:
        return False, 0.0, None
    threshold = max(1, min(3, len(reference_tokens)))
    best_coverage = 0.0
    best_source = None
    for chunk_tokens, source in tokenized_chunks:
        matched = reference_tokens & chunk_tokens
        coverage = len(matched) / len(reference_tokens)
        if coverage > best_coverage:
            best_coverage = coverage
            best_source = source
        if len(matched) >= threshold and coverage >= 0.75:
            return True, round(coverage, 4), source
    return False, round(best_coverage, 4), best_source


def main() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    expanded = json.loads(EXPANDED.read_text(encoding="utf-8"))
    baseline_test_ids = baseline.get("test_ids") or baseline.get("benchmark", {}).get("test_ids")
    expanded_test_ids = expanded.get("test_ids") or expanded.get("benchmark", {}).get("test_ids")
    if baseline_test_ids != expanded_test_ids:
        raise SystemExit("Canonical and expanded evaluations do not use identical test IDs")

    comparison = {
        "protocol": "same canonical 180-question test IDs, model, prompts, retrieval settings, and lexical metrics; only the corpus/index is expanded",
        "baseline_result": str(BASELINE.relative_to(ROOT)),
        "expanded_result": str(EXPANDED.relative_to(ROOT)),
        "test_size": len(expanded_test_ids),
        "systems": {},
        "tier_results": {},
    }
    for system in SYSTEMS:
        baseline_values = baseline["overall"]["systems"][system]
        expanded_values = expanded["overall"]["systems"][system]
        comparison["systems"][system] = {}
        for metric in METRICS:
            old = baseline_values.get(metric, 0.0)
            new = expanded_values.get(metric, 0.0)
            comparison["systems"][system][metric] = {
                "baseline": old,
                "expanded": new,
                "absolute_change": round(new - old, 4),
                "relative_change": _relative_change(old, new),
            }

        comparison["tier_results"][system] = {}
        for tier in ("tier_1", "tier_2", "tier_3"):
            comparison["tier_results"][system][tier] = {}
            old_tier = baseline["tier_results"][system][tier]
            new_tier = expanded["tier_results"][system][tier]
            for metric in METRICS:
                old = old_tier.get(metric, 0.0)
                new = new_tier.get(metric, 0.0)
                comparison["tier_results"][system][tier][metric] = {
                    "baseline": old,
                    "expanded": new,
                    "absolute_change": round(new - old, 4),
                    "relative_change": _relative_change(old, new),
                }

    all_chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    new_chunks = [
        (_tokens(chunk.get("text", "")), chunk.get("source_url") or chunk.get("source_file"))
        for chunk in all_chunks if str(chunk.get("document_id", "")).startswith("ADITYA_L1_")
    ]
    original_chunks = [
        (_tokens(chunk.get("text", "")), chunk.get("source_url") or chunk.get("source_file"))
        for chunk in all_chunks if not str(chunk.get("document_id", "")).startswith("ADITYA_L1_")
    ]
    benchmark_rows = {row["id"]: row for row in json.loads((ROOT / "data" / "benchmark" / "isro_qa.json").read_text(encoding="utf-8-sig"))}
    source_rows = []
    for qid in expanded_test_ids:
        item = benchmark_rows[qid]
        reference = item.get("answer", "")
        old_hit, old_score, old_source = _source_evidence(reference, original_chunks)
        new_hit, new_score, new_source = _source_evidence(reference, new_chunks)
        label = "both" if old_hit and new_hit else "original corpus only" if old_hit else "new Aditya-L1 corpus" if new_hit else "neither"
        source_rows.append({
            "id": qid,
            "tier": item.get("tier"),
            "reference_answer": reference,
            "association": label,
            "original_evidence": {"matched": old_hit, "best_coverage": old_score, "source": old_source},
            "aditya_l1_evidence": {"matched": new_hit, "best_coverage": new_score, "source": new_source},
        })
    counts = {label: sum(row["association"] == label for row in source_rows) for label in ("original corpus only", "new Aditya-L1 corpus", "both", "neither")}
    source_report = {
        "method": "Conservative lexical evidence: at least 75% of non-stopword reference tokens and at least three matched tokens (or all tokens for shorter references) in one source chunk.",
        "warning": "This is a corpus-association analysis, not a human factuality judgment.",
        "counts": counts,
        "rows": source_rows,
    }
    comparison["source_association_summary"] = counts
    COMPARISON.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    SOURCE_ANALYSIS.write_text(json.dumps(source_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"comparison": str(COMPARISON), "source_analysis": str(SOURCE_ANALYSIS), "source_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
