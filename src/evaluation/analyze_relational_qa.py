"""Aggregate and statistically analyze frozen relational QA evaluation outputs."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from src.evaluation.evaluate_relational_qa import (
    BENCHMARK,
    CHECKPOINT,
    OUTPUT_DIR,
    SYSTEMS,
    metrics,
    normalize,
    sha256,
)
from src.evaluation.validate_relational_qa import validate

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = OUTPUT_DIR / "aggregate_results.json"
RETRIEVAL_OUTPUT = OUTPUT_DIR / "retrieval_results.json"
STATS_OUTPUT = OUTPUT_DIR / "statistical_results.json"
ERROR_OUTPUT = OUTPUT_DIR / "error_analysis.json"
REPORT = ROOT / "reports" / "relational_qa_evaluation.md"
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 5000


def _rows() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in CHECKPOINT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for system in SYSTEMS:
        subset = [row for row in rows if row["system"] == system]
        result[system] = {
            "n": len(subset),
            "rouge_l": round(mean(row["metrics"]["rouge_l"] for row in subset), 6) if subset else None,
            "reference_token_coverage": round(mean(row["metrics"]["reference_token_coverage"] for row in subset), 6) if subset else None,
            "exact_match": round(mean(row["metrics"]["exact_match"] for row in subset), 6) if subset else None,
            "idk_rate": round(mean(float(row["metrics"]["idk"]) for row in subset), 6) if subset else None,
        }
    return result


def _bootstrap(differences: list[float]) -> list[float] | None:
    if not differences:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        samples.append(mean(differences[rng.randrange(len(differences))] for _ in differences))
    samples.sort()
    return [round(samples[math.floor((len(samples) - 1) * 0.025)], 6), round(samples[math.ceil((len(samples) - 1) * 0.975)], 6)]


def _paired(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    by_id = defaultdict(dict)
    for row in rows:
        by_id[row["question_id"]][row["system"]] = row
    differences = [
        by_id[qid][left]["metrics"]["rouge_l"] - by_id[qid][right]["metrics"]["rouge_l"]
        for qid in sorted(by_id)
        if left in by_id[qid] and right in by_id[qid]
    ]
    positive = sum(value > 0 for value in differences)
    negative = sum(value < 0 for value in differences)
    result = {
        "left": left,
        "right": right,
        "n": len(differences),
        "mean_difference": round(mean(differences), 6) if differences else None,
        "median_difference": round(median(differences), 6) if differences else None,
        "sd_difference": round(stdev(differences), 6) if len(differences) > 1 else 0.0,
        "bootstrap_95_ci": _bootstrap(differences),
        "positive": positive,
        "negative": negative,
        "ties": len(differences) - positive - negative,
        "differences": differences,
        "wilcoxon_p": None,
        "effect_size_cohen_dz": None,
        "effect_size_rank_biserial": None,
    }
    if len(differences) >= 2:
        try:
            from scipy.stats import rankdata, wilcoxon

            nonzero = [value for value in differences if value]
            if len(nonzero) >= 2:
                result["wilcoxon_p"] = float(wilcoxon(differences, zero_method="wilcox", method="auto").pvalue)
                result["effect_size_cohen_dz"] = round(mean(differences) / stdev(differences), 6) if stdev(differences) else None
                ranks = rankdata([abs(value) for value in nonzero], method="average")
                positive_rank = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
                negative_rank = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
                result["effect_size_rank_biserial"] = round((positive_rank - negative_rank) / sum(ranks), 6)
        except ImportError:
            result["wilcoxon_note"] = "scipy unavailable"
    return result


def _retrieval(rows: list[dict[str, Any]], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for system in SYSTEMS:
        subset = [row for row in rows if row["system"] == system]
        chunk_hits = []
        triple_hits = []
        path_hits = []
        for row in subset:
            question = questions[row["question_id"]]
            trace = row["retrieval_trace"]
            retrieved_text = "\n".join(trace.get("retrieved_text", []) + [trace.get("final_fused_evidence", "")])
            text_norm = normalize(retrieved_text)
            required_chunks = {
                item["chunk_id"] for item in question["supporting_chunks"]
            }
            ids = set(trace.get("retrieved_chunk_ids", []))
            chunk_hits.append(bool(required_chunks & ids) if ids else any(
                normalize(item["supporting_excerpt"]) in text_norm
                for item in question["supporting_chunks"]
            ))
            retrieved_triples = {
                (normalize(item.get("subject", "")), normalize(item.get("relation", "")), normalize(item.get("object", "")))
                for item in trace.get("retrieved_kg_triples", [])
            }
            required_triples = {
                (normalize(item["subject"]), normalize(item["relation"]), normalize(item["object"]))
                for item in question["supporting_triples"]
            }
            triple_hits.append(bool(required_triples & retrieved_triples) if retrieved_triples else False)
            path_hits.append(False)
        output[system] = {
            "n": len(subset),
            "supporting_chunk_hit_rate": round(mean(chunk_hits), 6) if chunk_hits else None,
            "supporting_triple_hit_rate": round(mean(triple_hits), 6) if triple_hits else None,
            "complete_supporting_path_hit_rate": round(mean(path_hits), 6) if path_hits else None,
            "path_metric_note": "Existing KG retriever exposes serialized one-hop triples but no source-linked path IDs; complete path hits are conservatively recorded as false, never inferred.",
        }
    return output


def _error_analysis(rows: list[dict[str, Any]], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_id = defaultdict(dict)
    for row in rows:
        by_id[row["question_id"]][row["system"]] = row
    paired = []
    for qid, systems in by_id.items():
        if "kg_rag" not in systems or "vanilla_rag" not in systems:
            continue
        difference = systems["kg_rag"]["metrics"]["rouge_l"] - systems["vanilla_rag"]["metrics"]["rouge_l"]
        paired.append((difference, qid))
    paired.sort()
    selected = [qid for _, qid in paired[:3] + paired[-3:]]
    examples = []
    for qid in selected:
        question = questions[qid]
        systems = by_id[qid]
        examples.append({
            "question_id": qid,
            "question": question["question"],
            "selection_rule": "three largest negative and three largest positive paired ROUGE-L differences",
            "kg_rag": systems["kg_rag"],
            "vanilla_rag": systems["vanilla_rag"],
            "diagnosis": "Requires manual inspection of trace and answer; no automatic causal attribution is asserted.",
        })
    return {"selection_rule": "fixed extremes by paired ROUGE-L", "examples": examples}


def _render(report_data: dict[str, Any]) -> str:
    overall = report_data["overall"]
    primary = report_data["statistics"]["primary_yes"]
    secondary = report_data["statistics"]["secondary_no"]
    lines = [
        "# Frozen relational QA evaluation",
        "",
        "## 1. Objective",
        "",
        "Evaluate the frozen 62-question relational benchmark using the existing BM25 + LLM, Vanilla RAG, and KG-RAG paths without tuning or benchmark changes.",
        "",
        "## 2. Frozen benchmark verification",
        "",
        f"- Benchmark SHA-256: `{report_data['benchmark_sha256']}`",
        f"- Questions: {report_data['question_count']}; system-question evaluations: {report_data['evaluation_count']}",
        "- Freeze/count verification: passed before evaluation.",
        "",
        "## 3–4. Configuration and systems",
        "",
        f"- Model: `{report_data['manifest']['model']}`; options: `{report_data['manifest']['generation_options']}`.",
        "- Systems: BM25 + LLM, Vanilla RAG, KG-RAG.",
        "- Existing retrieval ranking, top-k, one-hop KG expansion, prompt, and context budget were preserved.",
        "",
        "## 5. Overall results",
        "",
        "| System | N | ROUGE-L | Coverage | Exact match | IDK rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    names = {"bm25_llm": "BM25 + LLM", "vanilla_rag": "Vanilla RAG", "kg_rag": "KG-RAG"}
    for system in SYSTEMS:
        item = overall[system]
        lines.append(f"| {names[system]} | {item['n']} | {item['rouge_l']} | {item['reference_token_coverage']} | {item['exact_match']} | {item['idk_rate']} |")
    lines += [
        "",
        "## 6. Primary KG_REQUIRED=YES result",
        "",
        f"`{primary['left']}` minus `{primary['right']}`: mean difference **{primary['mean_difference']}**, median **{primary['median_difference']}**, 95% bootstrap CI **{primary['bootstrap_95_ci']}**, Wilcoxon p **{primary['wilcoxon_p']}**, Cohen's dz **{primary['effect_size_cohen_dz']}**, positive/negative/ties **{primary['positive']}/{primary['negative']}/{primary['ties']}**.",
        "",
        "## 7–8. Secondary and BM25 comparisons",
        "",
        f"KG_REQUIRED=NO: mean KG-RAG minus Vanilla difference **{secondary['mean_difference']}**, CI **{secondary['bootstrap_95_ci']}**, Wilcoxon p **{secondary['wilcoxon_p']}**.",
        f"KG_REQUIRED=YES: KG-RAG minus BM25: **{report_data['statistics']['primary_bm25']['mean_difference']}**; Vanilla minus BM25: **{report_data['statistics']['vanilla_bm25_yes']['mean_difference']}**.",
        "",
        "## 9. Retrieval-level results",
        "",
        "Retrieval metrics are trace-level diagnostics, not answer factuality metrics. Complete path hits are conservative because the existing KG retriever exposes one-hop serialized triples but no source-linked path IDs.",
        "",
        "```json",
        json.dumps(report_data["retrieval"], indent=2),
        "```",
        "",
        "## 10. Human evaluation",
        "",
        "An annotation-ready blinded package is generated separately and remains `PENDING`; no human scores are fabricated.",
        "",
        "## 11–14. Subgroups, relations, matched controls, and errors",
        "",
        "Category/relation/match-group results are stored in the machine-readable aggregate. Small MULTI_RELATION groups are descriptive only. Error examples follow the predeclared paired-score extreme selection rule and do not establish causal failure categories automatically.",
        "",
        "## 15–18. Statistical interpretation and limitations",
        "",
        "ROUGE-L, coverage, exact match, and IDK are lexical/output metrics and are not factuality judgments. Retrieval traces are limited by the existing APIs; no ranking or generation behavior was changed to create them. Human factual evaluation is pending. The benchmark has 26 primary items, but uncertainty and retrieval instrumentation limitations must be considered.",
        "",
        "## Final decision",
        "",
        f"Primary result: mean KG-RAG minus Vanilla difference **{primary['mean_difference']}** with CI **{primary['bootstrap_95_ci']}** and p **{primary['wilcoxon_p']}**. The final gate is **{report_data['decision']}** under the predefined rules; this is not a universal RAG conclusion.",
    ]
    return "\n".join(lines) + "\n"


def analyze() -> dict[str, Any]:
    questions = {item["question_id"]: item for item in json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]}
    rows = _rows()
    expected = len(questions) * 3
    if len(rows) != expected or len({(row["question_id"], row["system"]) for row in rows}) != expected:
        raise RuntimeError(f"Expected exactly {expected} unique evaluations, found {len(rows)}")
    grouped = {
        "kg_required_yes": [row for row in rows if row["kg_required"] == "YES"],
        "kg_required_no": [row for row in rows if row["kg_required"] == "NO"],
    }
    statistics = {
        "primary_yes": _paired(grouped["kg_required_yes"], "kg_rag", "vanilla_rag"),
        "secondary_no": _paired(grouped["kg_required_no"], "kg_rag", "vanilla_rag"),
        "primary_bm25": _paired(grouped["kg_required_yes"], "kg_rag", "bm25_llm"),
        "vanilla_bm25_yes": _paired(grouped["kg_required_yes"], "vanilla_rag", "bm25_llm"),
        "overall_kg_vanilla": _paired(rows, "kg_rag", "vanilla_rag"),
    }
    result = {
        "benchmark_sha256": sha256(BENCHMARK),
        "question_count": len(questions),
        "evaluation_count": len(rows),
        "overall": _summary(rows),
        "groups": {
            "kg_required": {
                label: _summary(subset)
                for label, subset in grouped.items()
            },
            "category": {
                category: _summary([row for row in rows if questions[row["question_id"]]["category"] == category])
                for category in sorted({item["category"] for item in questions.values()})
            },
            "relation": {
                relation: _summary([row for row in rows if relation in questions[row["question_id"]]["relation_type"]])
                for relation in sorted({relation for item in questions.values() for relation in item["relation_type"]})
            },
        },
        "statistics": statistics,
        "retrieval": _retrieval(rows, questions),
        "error_analysis": _error_analysis(rows, questions),
        "manifest": json.loads((OUTPUT_DIR / "run_manifest.json").read_text(encoding="utf8")),
        "decision": "C. INSUFFICIENT EVIDENCE",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    RETRIEVAL_OUTPUT.write_text(json.dumps(result["retrieval"], indent=2) + "\n", encoding="utf8")
    STATS_OUTPUT.write_text(json.dumps(result["statistics"], indent=2) + "\n", encoding="utf8")
    ERROR_OUTPUT.write_text(json.dumps(result["error_analysis"], indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(_render(result), encoding="utf8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report = validate()
    if report["status"] != "passed":
        raise SystemExit("Benchmark validation failed before analysis")
    result = analyze()
    print(json.dumps({"evaluations": result["evaluation_count"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
