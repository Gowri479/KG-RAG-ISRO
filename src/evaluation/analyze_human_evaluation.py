"""Analyze the completed blinded human evaluation without altering submitted scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path(r"D:\Users\NANS\Downloads\human_evaluation_review_filled.csv")
BENCHMARK = ROOT / "data" / "relational_benchmark" / "relational_qa_v1.json"
OUTPUT = ROOT / "data" / "results" / "relational_qa_v1" / "human_evaluation_analysis.json"
REPORT = ROOT / "reports" / "human_evaluation_results.md"
SYSTEMS = ("bm25_llm", "vanilla_rag", "kg_rag")
DISPLAY = {"bm25_llm": "BM25 + LLM", "vanilla_rag": "Vanilla RAG", "kg_rag": "KG-RAG"}
METRICS = ("correctness", "completeness", "groundedness", "relevance")
EXPECTED_HASH = "7f5daf5419b0a4cb70e5c3d5eba021a8a39118b5a0fd235ce19b81f75324ffce"


def _bootstrap(values: list[float], seed: int = 42) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    samples = [mean(values[rng.randrange(len(values))] for _ in values) for _ in range(5000)]
    samples.sort()
    return [
        round(samples[math.floor((len(samples) - 1) * 0.025)], 6),
        round(samples[math.ceil((len(samples) - 1) * 0.975)], 6),
    ]


def _paired_stats(values: list[float]) -> dict:
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    result = {
        "n": len(values),
        "mean_difference": round(mean(values), 6),
        "median_difference": round(median(values), 6),
        "positive": positive,
        "negative": negative,
        "ties": len(values) - positive - negative,
        "bootstrap_95_ci": _bootstrap(values),
        "wilcoxon_p": None,
        "cohen_dz": None,
        "rank_biserial": None,
    }
    if len(values) < 2:
        return result
    try:
        from scipy.stats import rankdata, wilcoxon

        nonzero = [value for value in values if value]
        if len(nonzero) >= 2:
            result["wilcoxon_p"] = round(float(wilcoxon(values, zero_method="wilcox", method="auto").pvalue), 6)
            sd = stdev(values)
            result["cohen_dz"] = round(mean(values) / sd, 6) if sd else None
            ranks = rankdata([abs(value) for value in nonzero], method="average")
            positive_rank = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
            negative_rank = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
            result["rank_biserial"] = round((positive_rank - negative_rank) / sum(ranks), 6)
    except ImportError:
        result["statistical_test_note"] = "scipy unavailable"
    return result


def _mapping(question_ids: list[str]) -> dict[str, dict[str, str]]:
    rng = random.Random(20261007)
    output = {}
    for question_id in sorted(question_ids):
        order = list(SYSTEMS)
        rng.shuffle(order)
        output[question_id] = {label: system for label, system in zip(("A", "B", "C"), order)}
    return output


def _load(path: Path) -> tuple[dict[str, dict], dict[str, dict[str, dict]], dict[str, dict[str, str]]]:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf8"))
    questions = {q["question_id"]: q for q in benchmark["questions"]}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {"question_id", "question", "kg_required", "category", "reference_answer"}
    expected |= {f"{label}_{metric}" for label in "ABC" for metric in METRICS + ("unsupported_claim",)}
    missing_columns = sorted(expected - set(rows[0])) if rows else sorted(expected)
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    if len(rows) != 62:
        raise ValueError(f"Expected 62 questions, found {len(rows)}")
    if {row["question_id"] for row in rows} != set(questions):
        raise ValueError("Submitted question IDs do not match the frozen benchmark")
    mapping = _mapping(list(questions))
    by_question = {}
    for row in rows:
        question = questions[row["question_id"]]
        for field in ("question", "kg_required", "category", "reference_answer"):
            if row[field] != question[field]:
                raise ValueError(f"{row['question_id']}: frozen field changed: {field}")
        by_question[row["question_id"]] = row
    return questions, by_question, mapping


def _score(value: str, field: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{field}: invalid score {value!r}") from exc
    if field.endswith("unsupported_claim"):
        if number not in (0.0, 1.0):
            raise ValueError(f"{field}: unsupported claim must be 0/1")
    elif number < 1.0 or number > 5.0:
        raise ValueError(f"{field}: expected submitted 1-5 score")
    return number


def _records(questions, rows, mapping):
    records = []
    missing = []
    for qid, question in questions.items():
        row = rows[qid]
        for label in "ABC":
            system = mapping[qid][label]
            record = {"question_id": qid, "label": label, "system": system, "kg_required": question["kg_required"]}
            for metric in METRICS:
                field = f"{label}_{metric}"
                if not row[field].strip():
                    missing.append(field)
                record[metric] = _score(row[field], field)
            field = f"{label}_unsupported_claim"
            if not row[field].strip():
                missing.append(field)
            record["unsupported_claim"] = _score(row[field], field)
            records.append(record)
    if missing:
        raise ValueError(f"Missing human scores: {missing[:10]}")
    return records


def _summary(records: list[dict]) -> dict:
    return {
        "n_questions": len(records),
        "n_answers": len(records),
        **{
            metric: {
                "mean": round(mean(row[metric] for row in records), 6),
                "median": round(median(row[metric] for row in records), 6),
            }
            for metric in METRICS
        },
        "unsupported_claim_rate": round(mean(row["unsupported_claim"] for row in records), 6),
    }


def _system_summaries(records: list[dict], questions: dict, scope: str | None = None) -> dict:
    output = {}
    for system in SYSTEMS:
        subset = [row for row in records if row["system"] == system and (scope is None or row["kg_required"] == scope)]
        output[system] = _summary(subset)
    return output


def _comparisons(records: list[dict], left: str, right: str, scope: str) -> dict:
    grouped = defaultdict(dict)
    for row in records:
        if row["kg_required"] == scope:
            grouped[row["question_id"]][row["system"]] = row
    result = {}
    for metric in METRICS:
        values = [
            grouped[qid][left][metric] - grouped[qid][right][metric]
            for qid in sorted(grouped)
            if left in grouped[qid] and right in grouped[qid]
        ]
        result[metric] = _paired_stats(values)
    return {
        "left": left,
        "right": right,
        "scope": scope,
        "metrics": result,
        "unsupported_claim_rate": {
            left: round(mean(grouped[qid][left]["unsupported_claim"] for qid in grouped), 6),
            right: round(mean(grouped[qid][right]["unsupported_claim"] for qid in grouped), 6),
        },
    }


def analyze(input_path: Path = DEFAULT_INPUT) -> dict:
    questions, rows, mapping = _load(input_path)
    records = _records(questions, rows, mapping)
    result = {
        "status": "COMPLETED",
        "input_file": str(input_path),
        "evaluation_scope": "FULL_186",
        "annotation_count": 186,
        "question_count": 62,
        "benchmark_sha256": hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(),
        "scale": {
            "observed": "1-5 ordinal",
            "submitted_scores_preserved": True,
            "protocol_deviation": "Original guidance proposed 0-2; supplied annotations use 1-5 and were analyzed unchanged.",
        },
        "missing_values": [],
        "per_question_system_mapping": mapping,
        "system_summaries": {
            "all_62": _system_summaries(records, questions),
            "kg_required_yes_26": _system_summaries(records, questions, "YES"),
            "kg_required_no_36": _system_summaries(records, questions, "NO"),
        },
        "primary_kg_vs_vanilla_yes": _comparisons(records, "kg_rag", "vanilla_rag", "YES"),
        "secondary_yes": {
            "kg_vs_bm25": _comparisons(records, "kg_rag", "bm25_llm", "YES"),
            "vanilla_vs_bm25": _comparisons(records, "vanilla_rag", "bm25_llm", "YES"),
        },
        "secondary_no": {
            "kg_vs_bm25": _comparisons(records, "kg_rag", "bm25_llm", "NO"),
            "vanilla_vs_bm25": _comparisons(records, "vanilla_rag", "bm25_llm", "NO"),
        },
        "lexical_comparison": {
            "phase4_kg_minus_vanilla_yes_mean_rouge_l": -0.018373,
            "phase4_bootstrap_95_ci": [-0.068196, 0.010827],
            "phase4_wilcoxon_p": 1.0,
            "interpretation": "Human scores are factual-quality judgments on a 1-5 ordinal scale and should not be treated as ROUGE equivalents.",
        },
        "methodological_limitations": [
            "Primary comparison has N=26 questions.",
            "Only one completed annotation file is present; inter-annotator agreement is not calculated.",
            "The benchmark is researcher-constructed and human evaluation is not independent external validation.",
            "Unsupported-claim labels are human judgments under the supplied rubric, not automatically measured hallucination.",
            "Anonymization mapping is deterministic per question, not one global A/B/C mapping.",
        ],
    }
    yes = result["primary_kg_vs_vanilla_yes"]["metrics"]
    correctness = yes["correctness"]["mean_difference"]
    completeness = yes["completeness"]["mean_difference"]
    groundedness = yes["groundedness"]["mean_difference"]
    relevance = yes["relevance"]["mean_difference"]
    if all(value > 0 for value in (correctness, completeness, groundedness, relevance)):
        conclusion = "A — YES"
    elif all(value <= 0 for value in (correctness, completeness, groundedness, relevance)):
        conclusion = "B — NO"
    else:
        conclusion = "C — MIXED"
    result["final_conclusion"] = conclusion
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    REPORT.write_text(_render(result), encoding="utf8")
    return result


def _render(result: dict) -> str:
    yes = result["system_summaries"]["kg_required_yes_26"]
    primary = result["primary_kg_vs_vanilla_yes"]
    lines = [
        "# Phase 5C — Completed blinded human evaluation",
        "",
        "## 1. Verification",
        "",
        f"- Evaluation scope: **{result['evaluation_scope']}**",
        f"- Questions: {result['question_count']}; scored answer instances: {result['annotation_count']}",
        f"- Benchmark SHA-256: `{result['benchmark_sha256']}`",
        "- Missing scores: 0",
        "- Submitted scores were preserved unchanged.",
        "",
        "## 2. Actual scoring scale",
        "",
        "The supplied CSV uses a **1–5 ordinal scale** for correctness, completeness, groundedness, and relevance, plus 0/1 unsupported-claim values. This is a deviation from the originally proposed 0–2 rubric. No score conversion was performed.",
        "",
        "## 3. Per-question blinded mapping",
        "",
        "The original package uses a deterministic per-question shuffle with seed `20261007`; there is no single global A/B/C mapping. The exact mapping for every question is stored in `human_evaluation_analysis.json`.",
        "",
        "| System | Correctness mean | Completeness mean | Groundedness mean | Relevance mean | Unsupported-claim rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        item = yes[system]
        lines.append(
            f"| {DISPLAY[system]} | {item['correctness']['mean']:.3f} | "
            f"{item['completeness']['mean']:.3f} | {item['groundedness']['mean']:.3f} | "
            f"{item['relevance']['mean']:.3f} | {item['unsupported_claim_rate']:.3f} |"
        )
    lines += [
        "",
        "## 4. Primary KG-RAG versus Vanilla RAG comparison",
        "",
        "| Dimension | Mean diff | Median diff | + / − / tie | Bootstrap 95% CI | Wilcoxon p | Cohen dz |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for metric in METRICS:
        item = primary["metrics"][metric]
        lines.append(f"| {metric} | {item['mean_difference']:.3f} | {item['median_difference']:.3f} | {item['positive']} / {item['negative']} / {item['ties']} | {item['bootstrap_95_ci']} | {item['wilcoxon_p']} | {item['cohen_dz']} |")
    lines += [
        "",
        f"Unsupported-claim rate on YES: KG-RAG **{primary['unsupported_claim_rate']['kg_rag']:.3f}**; Vanilla RAG **{primary['unsupported_claim_rate']['vanilla_rag']:.3f}**.",
        "",
        "## 5. Human summaries",
        "",
        "Full per-system summaries for all 62 questions, YES=26, and NO=36 are stored in the JSON artifact. The supplied score dimensions are reported on their original 1–5 scale.",
        "",
        "## 6. Human versus lexical results",
        "",
        "Phase 4 ROUGE-L on YES had KG-RAG minus Vanilla mean difference −0.018373, CI [−0.068196, 0.010827], and Wilcoxon p=1.0. The human analysis is more directly relevant to factual quality, but it is a single-annotator, researcher-constructed evaluation. Agreement or contradiction with ROUGE must therefore be described cautiously rather than treating ROUGE as factual correctness.",
        "",
        "## 7. Methodological caveats",
        "",
        "- Primary N=26 questions.",
        "- One completed annotation file is available; inter-annotator agreement is not calculated.",
        "- The benchmark is researcher-constructed, not independent external validation.",
        "- Unsupported-claim judgments are human judgments under the supplied rubric.",
        "- A/B/C system labels are shuffled per question; the full mapping is recorded in the JSON.",
        "",
        "## 8. Conclusion",
        "",
        f"**{result['final_conclusion']}** — based on the submitted human scores and the predeclared paired comparisons. This conclusion is limited to the completed single-annotator evaluation and does not justify universal claims about KG-RAG.",
        "",
        "The paper was not modified. Whether to retain the KG-RAG contribution should be decided after considering these human results together with the Phase 4 lexical results and the documented trace limitations.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input), indent=2))
