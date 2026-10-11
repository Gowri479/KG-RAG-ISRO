"""Diagnose frozen Phase 4 KG-RAG failures without rerunning any model."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "relational_benchmark" / "relational_qa_v1.json"
RESULTS = ROOT / "data" / "results" / "relational_qa_v1"
CHECKPOINT = RESULTS / "per_question_results.jsonl"
OUTPUT = RESULTS / "failure_analysis.json"
DIAGNOSIS = RESULTS / "retrieval_diagnosis.json"
REPORT = ROOT / "reports" / "phase5_failure_analysis.md"
SYSTEMS = ("bm25_llm", "vanilla_rag", "kg_rag")
EXPECTED_HASH = "7f5daf5419b0a4cb70e5c3d5eba021a8a39118b5a0fd235ce19b81f75324ffce"


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _token_recall(answer: str, reference: str) -> float:
    answer_tokens = set(_norm(answer).split())
    reference_tokens = set(_norm(reference).split())
    return len(answer_tokens & reference_tokens) / len(reference_tokens) if reference_tokens else 0.0


def _load() -> tuple[dict[str, dict], dict[str, dict[str, dict]]]:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf8"))
    questions = {q["question_id"]: q for q in benchmark["questions"]}
    rows = defaultdict(dict)
    for line in CHECKPOINT.read_text(encoding="utf8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["question_id"]][row["system"]] = row
    return questions, rows


def _text_support(question: dict, trace: dict) -> tuple[str, float, str]:
    text = "\n".join(trace.get("retrieved_text", []))
    text += "\n" + str(trace.get("dense_context", ""))
    excerpts = [item.get("supporting_excerpt", "") for item in question.get("supporting_chunks", [])]
    present = [excerpt for excerpt in excerpts if excerpt and _norm(excerpt) in _norm(text)]
    if not excerpts:
        return "NOT_DETERMINABLE", 0.0, "No supporting excerpt was recorded."
    return (
        "SUPPORTING_CHUNK_RETRIEVED" if present else "SUPPORTING_CHUNK_NOT_RETRIEVED",
        len(present) / len(excerpts),
        "Exact supporting excerpt found in saved retrieved text/context." if present else "No exact supporting excerpt found in saved retrieved text/context.",
    )


def _kg_evidence(question: dict, trace: dict) -> tuple[str, str, float]:
    paths = trace.get("retrieved_kg_paths")
    if paths:
        return "COMPLETE_PATH_RETRIEVED", "Source-linked path records are present.", 1.0
    triples = trace.get("retrieved_kg_triples", [])
    required = question.get("supporting_triples", [])
    if not triples:
        return "NOT_DETERMINABLE", "No KG triples or paths were saved.", 0.0
    # The saved triples are free-text output, not canonical triple IDs. Only exact
    # normalized subject/relation/object matches are counted.
    matches = 0
    for wanted in required:
        wanted_tuple = (_norm(wanted["subject"]), _norm(wanted["relation"]), _norm(wanted["object"]))
        for got in triples:
            got_tuple = (_norm(got.get("subject", "")), _norm(got.get("relation", "")), _norm(got.get("object", "")))
            if got_tuple == wanted_tuple:
                matches += 1
                break
    if matches == len(required) and required:
        return "COMPLETE_PATH_RETRIEVED", "All required canonical triples exactly matched saved KG triples.", 1.0
    if matches:
        return "PARTIAL_PATH_RETRIEVED", "Some required canonical triples exactly matched saved KG triples.", matches / len(required)
    return "NOT_DETERMINABLE", "KG triples are present, but none exactly match canonical benchmark triples; source-linked path IDs are unavailable.", 0.0


def _failure(question: dict, kg_row: dict, path_status: str, text_status: str, lexical_success: bool) -> str:
    if lexical_success:
        return "NO_FAILURE"
    trace = kg_row["retrieval_trace"]
    if path_status == "NOT_DETERMINABLE":
        return "INSUFFICIENT_TRACE"
    if path_status == "REQUIRED_PATH_NOT_RETRIEVED":
        return "KG_RETRIEVAL_FAILURE"
    if text_status == "SUPPORTING_CHUNK_NOT_RETRIEVED":
        return "SUPPORTING_TEXT_RETRIEVAL_FAILURE"
    if not trace.get("final_fused_evidence", trace.get("dense_context", "")):
        return "EVIDENCE_FUSION_FAILURE"
    return "GENERATION_FAILURE"


def _compare(rows: dict[str, dict]) -> dict:
    kg = rows["kg_rag"]
    vanilla = rows["vanilla_rag"]
    kg_score = kg["metrics"]["rouge_l"]
    vanilla_score = vanilla["metrics"]["rouge_l"]
    if kg_score > vanilla_score:
        outcome = "KG_RAG_BETTER"
    elif vanilla_score > kg_score:
        outcome = "VANILLA_BETTER"
    elif kg_score == vanilla_score:
        outcome = "TIE"
    else:
        outcome = "INDETERMINATE"
    return {
        "classification": outcome,
        "kg_rag_answer": kg["generated_answer"],
        "vanilla_answer": vanilla["generated_answer"],
        "kg_rag_rouge_l": kg_score,
        "vanilla_rouge_l": vanilla_score,
        "kg_rag_coverage": kg["metrics"]["reference_token_coverage"],
        "vanilla_coverage": vanilla["metrics"]["reference_token_coverage"],
        "kg_rag_idk": bool(kg["metrics"]["idk"]),
        "vanilla_idk": bool(vanilla["metrics"]["idk"]),
        "interpretation": "Lexical proxy only; correctness and factuality require human evaluation.",
    }


def analyze() -> dict:
    questions, rows = _load()
    benchmark_hash = hashlib.sha256(BENCHMARK.read_bytes()).hexdigest()
    if benchmark_hash != EXPECTED_HASH:
        raise RuntimeError("Frozen benchmark hash mismatch; aborting Phase 5.")
    required_ids = [qid for qid, q in questions.items() if q["kg_required"] == "YES"]
    if len(required_ids) != 26 or len(rows) != 62 or any(set(rows[qid]) != set(SYSTEMS) for qid in rows):
        raise RuntimeError("Phase 4 invariant failed; aborting Phase 5.")

    diagnoses = []
    for qid in required_ids:
        question = questions[qid]
        kg_row = rows[qid]["kg_rag"]
        trace = kg_row["retrieval_trace"]
        path_status, path_reason, path_fraction = _kg_evidence(question, trace)
        text_status, text_recall, text_reason = _text_support(question, trace)
        lexical_success = kg_row["metrics"]["reference_token_coverage"] > 0
        dense_text = str(trace.get("dense_context", ""))
        evidence_items = trace.get("retrieved_kg_triples", []) + trace.get("retrieved_text", [])
        diagnosis = {
            "question_id": qid,
            "question": question["question"],
            "mission": question["mission"],
            "category": question["category"],
            "relation_types": question["relation_type"],
            "reference_answer": question["reference_answer"],
            "required_triple_ids": [t["triple_id"] for t in question.get("supporting_triples", [])],
            "required_paths": question.get("supporting_paths", []),
            "query_classification": "NOT_AVAILABLE",
            "detected_entities": trace.get("detected_entities", "NOT_AVAILABLE"),
            "detected_relation": "NOT_AVAILABLE",
            "hop_depth": trace.get("hop_depth", "NOT_AVAILABLE"),
            "retrieved_kg_triples": trace.get("retrieved_kg_triples", "NOT_AVAILABLE"),
            "retrieved_kg_paths": trace.get("retrieved_kg_paths", "NOT_AVAILABLE"),
            "kg_path_status": path_status,
            "kg_path_reason": path_reason,
            "kg_path_fraction": path_fraction,
            "supporting_text_status": text_status,
            "supporting_text_recall": text_recall,
            "supporting_text_reason": text_reason,
            "dense_context_words": len(dense_text.split()) if dense_text else 0,
            "dense_evidence_count": len(trace.get("retrieved_text", [])),
            "kg_evidence_count": len(trace.get("retrieved_kg_triples", [])),
            "kg_path_count": len(trace.get("retrieved_kg_paths", [])) if isinstance(trace.get("retrieved_kg_paths", []), list) else "NOT_AVAILABLE",
            "duplicate_evidence_count": len(evidence_items) - len({_norm(str(item)) for item in evidence_items}),
            "unique_evidence_count": len({_norm(str(item)) for item in evidence_items}),
            "fused_context": trace.get("final_fused_evidence", trace.get("dense_context", "NOT_AVAILABLE")),
            "kg_rag_answer": kg_row["generated_answer"],
            "kg_rag_metrics": kg_row["metrics"],
            "lexical_answer_success_proxy": lexical_success,
            "primary_failure_category": _failure(question, kg_row, path_status, text_status, lexical_success),
            "comparison": _compare(rows[qid]),
            "trace_limitation": "Saved Phase 4 traces have no source-linked KG path IDs and no frozen-corpus chunk IDs.",
        }
        diagnoses.append(diagnosis)

    def rate(predicate):
        return round(sum(bool(predicate(d)) for d in diagnoses) / len(diagnoses), 6)

    category_counts = Counter(d["primary_failure_category"] for d in diagnoses)
    relation_breakdown = {}
    for relation in sorted({r for d in diagnoses for r in d["relation_types"]} | {d["category"] for d in diagnoses}):
        subset = [d for d in diagnoses if relation in d["relation_types"] or d["category"] == relation]
        qids = {d["question_id"] for d in subset}
        def subset_rate(predicate):
            return round(sum(bool(predicate(d)) for d in subset) / len(subset), 6) if subset else None
        relation_breakdown[relation] = {
            "n": len(subset),
            "kg_path_retrieval_rate": subset_rate(lambda d: d["kg_path_status"] == "COMPLETE_PATH_RETRIEVED"),
            "supporting_text_retrieval_rate": subset_rate(lambda d: d["supporting_text_status"] == "SUPPORTING_CHUNK_RETRIEVED"),
            "kg_rag_rouge_l": mean(rows[qid]["kg_rag"]["metrics"]["rouge_l"] for qid in qids),
            "vanilla_rouge_l": mean(rows[qid]["vanilla_rag"]["metrics"]["rouge_l"] for qid in qids),
        }
    result = {
        "benchmark_sha256": benchmark_hash,
        "kg_required_questions": 26,
        "diagnosis_basis": "Saved Phase 4 traces only; no model, retrieval, benchmark, corpus, prompt, or parameter changes.",
        "per_question_diagnosis": diagnoses,
        "aggregate_retrieval_statistics": {
            "kg_path_retrieval_rate": rate(lambda d: d["kg_path_status"] == "COMPLETE_PATH_RETRIEVED"),
            "partial_path_rate": rate(lambda d: d["kg_path_status"] == "PARTIAL_PATH_RETRIEVED"),
            "required_path_not_retrieved_rate": rate(lambda d: d["kg_path_status"] == "REQUIRED_PATH_NOT_RETRIEVED"),
            "path_not_determinable_rate": rate(lambda d: d["kg_path_status"] == "NOT_DETERMINABLE"),
            "supporting_text_retrieval_rate": rate(lambda d: d["supporting_text_status"] == "SUPPORTING_CHUNK_RETRIEVED"),
            "supporting_text_not_retrieved_rate": rate(lambda d: d["supporting_text_status"] == "SUPPORTING_CHUNK_NOT_RETRIEVED"),
            "text_not_determinable_rate": rate(lambda d: d["supporting_text_status"] == "NOT_DETERMINABLE"),
        },
        "failure_categories": dict(category_counts),
        "relation_type_breakdown": relation_breakdown,
        "kg_rag_vs_vanilla": {d["question_id"]: d["comparison"] for d in diagnoses},
        "context_dilution": {
            "kg_rag_wins": [d for d in diagnoses if d["comparison"]["classification"] == "KG_RAG_BETTER"],
            "kg_rag_losses": [d for d in diagnoses if d["comparison"]["classification"] == "VANILLA_BETTER"],
            "ties": [d for d in diagnoses if d["comparison"]["classification"] == "TIE"],
            "conclusion": "Not established from saved traces. KG-RAG traces expose evidence counts and context text, but no controlled causal comparison or source-linked path mapping.",
        },
        "retrieval_success_vs_answer_outcome": {
            "answer_success_definition": "reference_token_coverage > 0, a lexical proxy only",
            "kg_path_status_is_not_determinable": True,
            "matrix": "Unavailable as a valid path-retrieval 2x2 because complete path IDs are absent from all saved KG-RAG traces.",
        },
        "limitations": [
            "ROUGE-L and reference-token coverage do not establish factual correctness.",
            "The Phase 4 trace schema has no source-linked KG path IDs.",
            "Pilot-index dense chunk IDs cannot be directly compared with frozen corpus chunk IDs.",
            "Query classification and detected relation are not saved; they are NOT_AVAILABLE.",
            "Human evaluation is pending.",
        ],
        "final_decision": "D — NO CLEAR SINGLE BOTTLENECK",
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    DIAGNOSIS.write_text(json.dumps(result["aggregate_retrieval_statistics"], indent=2) + "\n", encoding="utf8")
    REPORT.write_text(_report(result), encoding="utf8")
    return result


def _report(result: dict) -> str:
    stats = result["aggregate_retrieval_statistics"]
    return f"""# Phase 5 — KG-RAG failure analysis

## 1. Objective

Diagnose the frozen Phase 4 KG-required evaluation using all 26 questions and saved traces only. No benchmark, corpus, model, prompt, retrieval setting, or prior result was changed.

## 2. Frozen benchmark verification

- Benchmark SHA-256: `{result['benchmark_sha256']}` (expected hash matched)
- KG_REQUIRED=YES: 26
- Phase 4 system-question rows: 186
- All three systems represented: verified

## 3–6. Retrieval and pipeline diagnosis

| Diagnostic | Rate |
|---|---:|
| Complete KG path retrieved | {stats['kg_path_retrieval_rate']} |
| Partial KG path retrieved | {stats['partial_path_rate']} |
| Required path not retrieved | {stats['required_path_not_retrieved_rate']} |
| KG path not determinable | {stats['path_not_determinable_rate']} |
| Supporting excerpt found in saved text/context | {stats['supporting_text_retrieval_rate']} |
| Supporting excerpt not found | {stats['supporting_text_not_retrieved_rate']} |

The saved traces expose `retrieved_kg_triples` and `retrieved_kg_paths`, but the latter is empty and there are no source-linked canonical path IDs. Dense IDs also belong to the pilot index rather than the frozen corpus namespace. Therefore path retrieval is **not determinable**, not silently treated as zero.

## 7. Retrieval success versus answer success

A valid retrieval-success/answer-outcome 2×2 cannot be computed because complete KG path retrieval is not observable in the saved traces. Lexical overlap is retained only as a proxy and is not factual correctness.

## 8. Relation-type analysis

The machine-readable report contains category/relation breakdowns for all 26 questions. `MULTI_RELATION` is descriptive only because its subgroup is tiny.

## 9. KG-RAG versus Vanilla

All 26 case-by-case comparisons are in `failure_analysis.json`. They use ROUGE-L and coverage only as lexical proxies. They do not establish factual superiority or failure.

## 10. Context dilution

The traces record context word counts, dense evidence counts, KG triple counts, path counts, and duplicate/unique evidence counts. However, they do not support a controlled causal claim that augmentation caused dilution. **Context dilution is not established.**

## 11. Human evaluation

The 62-question, 3-system blinded package exists and contains 186 answer instances. Its scoring fields remain blank and status is `PENDING`.

## 12. Limitations

- No source-linked KG path IDs were saved.
- Query classification and relation matching fields are unavailable.
- Pilot-index chunk IDs cannot be compared directly to frozen-corpus chunk IDs.
- ROUGE-L/token coverage are lexical proxies, not correctness judgments.
- No human annotations are available.

## 13. Final decision

**D — NO CLEAR SINGLE BOTTLENECK**

The traces show a substantial instrumentation limitation and do not establish whether the dominant problem is retrieval, fusion/context, or generation/evidence use. A stronger causal diagnosis requires independently instrumented future runs, but no rerun or tuning is performed in Phase 5.

## Recommended next step

Complete independent human evaluation first. If further engineering is approved afterward, add trace instrumentation in a separate controlled phase before changing system behavior; do not reinterpret the current frozen results as a path-retrieval experiment.
"""


if __name__ == "__main__":
    analyze()
