"""Run the preregistered blinded human-quality validation on 26 KG-required questions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.evaluate_relational_qa import metrics, sha256
from src.generator.prompt import build_user_prompt
from src.retriever.kg_diagnostic_retriever import CanonicalKG, diagnostic_object

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "relational_benchmark" / "relational_qa_v1.json"
OUTPUT = ROOT / "data" / "results" / "final_structured_kg_eval"
CHECKPOINT = OUTPUT / "generation_results.jsonl"
TRACES = OUTPUT / "traces"
ANNOTATIONS = ROOT / "data" / "annotations"
HUMAN_CSV = ANNOTATIONS / "final_structured_kg_human_eval.csv"
GUIDELINES = ANNOTATIONS / "final_structured_kg_human_eval_guidelines.md"
REPORT = OUTPUT / "final_structured_kg_eval_report.md"
MANIFEST = OUTPUT / "run_manifest.json"
OPTIONS = {"num_predict": 150, "temperature": 0.1, "num_ctx": 2048}
CONTEXT_LIMIT = 1500
SYSTEMS = ("vanilla_rag", "corrected_structured_kg_rag", "bm25_llm")
BLIND_SEED = 20261007


def _questions() -> list[dict[str, Any]]:
    payload = json.loads(BENCHMARK.read_text(encoding="utf8"))
    if sha256(BENCHMARK) != "7f5daf5419b0a4cb70e5c3d5eba021a8a39118b5a0fd235ce19b81f75324ffce":
        raise RuntimeError("Frozen benchmark hash mismatch")
    questions = [item for item in payload["questions"] if item["kg_required"] == "YES"]
    if len(questions) != 26:
        raise RuntimeError("Expected exactly 26 KG-required questions")
    return questions


def _dense_trace(question: str) -> dict[str, Any]:
    from src.evaluation.evaluate_relational_qa import _vanilla_trace

    return _vanilla_trace(question)


def _bm25_trace(question: str) -> dict[str, Any]:
    from src.evaluation.evaluate_relational_qa import _bm25_trace

    return _bm25_trace(question)


def _corrected_trace(question: dict[str, Any], dense: dict[str, Any], kg: CanonicalKG) -> dict[str, Any]:
    base = diagnostic_object(question["question"], True, dense["retrieved_text"])
    paths = base["retrieved_paths"]
    required = [set(path) for path in question.get("supporting_paths", [])]
    retrieved = [set(path["triple_ids"]) for path in paths]
    path_found = any(expected.issubset(actual) for expected in required for actual in retrieved)
    selected = [path for path in paths if any(set(expected).issubset(set(path["triple_ids"])) for expected in required)]
    selected = selected or paths[:10]
    structured = kg.format_paths(selected, include_source=True) if hasattr(kg, "format_paths") else base["kg_evidence"]
    context = " ".join((structured + "\n\nRETRIEVED DOCUMENT EVIDENCE\n" + "\n\n".join(dense["retrieved_text"])).split()[:CONTEXT_LIMIT])
    return {
        "detected_entities": base["detected_entities"],
        "detected_relations": base["detected_relations"],
        "retrieved_paths": [path for path in paths],
        "required_path_found": path_found,
        "source_provenance_available": all(bool(path["source_chunks"]) for path in selected) if selected else False,
        "dense_chunks": dense["retrieved_text"],
        "structured_kg_evidence": structured,
        "final_fused_context": context,
        "retrieved_chunk_ids": dense["retrieved_chunk_ids"],
        "context_token_estimate": len(context.split()),
    }


def _trace(question: dict[str, Any], system: str, kg: CanonicalKG) -> tuple[dict[str, Any], str]:
    text = question["question"]
    if system == "vanilla_rag":
        trace = _dense_trace(text)
        context = " ".join("\n\n".join(trace["retrieved_text"]).split()[:CONTEXT_LIMIT])
    elif system == "bm25_llm":
        trace = _bm25_trace(text)
        context = " ".join("\n\n".join(trace["retrieved_text"]).split()[:CONTEXT_LIMIT])
    else:
        dense = _dense_trace(text)
        trace = _corrected_trace(question, dense, kg)
        context = trace["final_fused_context"]
    trace["final_prompt"] = build_user_prompt(context, text)
    trace["final_context"] = context
    trace["context_size"] = len(context.split())
    return trace, context


def _read_rows() -> dict[tuple[str, str], dict[str, Any]]:
    if not CHECKPOINT.exists():
        return {}
    rows = {}
    for line in CHECKPOINT.read_text(encoding="utf8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = (row["question_id"], row["system"])
            if key in rows:
                raise RuntimeError(f"Duplicate generation row: {key}")
            rows[key] = row
    return rows


def _write_human_package(questions: list[dict[str, Any]], rows: dict[tuple[str, str], dict[str, Any]]) -> None:
    ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(BLIND_SEED)
    output = []
    for question in questions:
        labels = ["A", "B", "C"]
        rng.shuffle(labels)
        for label, system in zip(labels, SYSTEMS):
            row = rows[(question["question_id"], system)]
            output.append({
                "question_id": question["question_id"],
                "blind_system_id": label,
                "question": question["question"],
                "reference_answer": question["reference_answer"],
                "candidate_answer": row["answer"],
                "correctness": "",
                "completeness": "",
                "groundedness": "",
                "relevance": "",
                "unsupported_claim": "",
            })
    with HUMAN_CSV.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def _write_guidelines() -> None:
    GUIDELINES.write_text(
        """# Blinded human evaluation guidelines

## Scope

Evaluate all 78 answer instances independently. The three systems are
identified only as A, B, and C. Do not infer or discuss their identities.
Judge each answer against the question, reference answer, and available
authoritative evidence. Do not use ROUGE, token overlap, or answer length as
a substitute for factual judgment.

## Scoring

For correctness, completeness, groundedness, and relevance use the 1–5 scale:

- 1 = very poor or incorrect
- 2 = mostly incorrect, substantially incomplete, or weakly supported
- 3 = partially correct, complete, supported, or relevant
- 4 = mostly correct, complete, supported, and relevant
- 5 = fully correct, complete, clearly grounded, and directly relevant

For unsupported claim, enter `YES` when the answer contains a factual claim
not supported by the supplied evidence or reference answer; otherwise enter
`NO`. Leave a score blank rather than guessing when the answer cannot be
judged, and record the reason separately if applicable.

Score each question independently. Do not select a winner before all scores
are recorded. Harmless wording, acronym, date-format, and alias differences
should not be penalized. For multi-hop questions, correctness requires the
requested end answer and completeness requires the relevant relationship to
be answered.

Human factual-quality conclusions are pending blinded annotation.
""",
        encoding="utf8",
    )


def _summarize(rows: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for system in SYSTEMS:
        selected = [row for (qid, name), row in rows.items() if name == system]
        result[system] = {
            "n": len(selected),
            "rouge_l": round(sum(row["metrics"]["rouge_l"] for row in selected) / len(selected), 6),
            "reference_token_coverage": round(sum(row["metrics"]["reference_token_coverage"] for row in selected) / len(selected), 6),
            "exact_match_rate": round(sum(row["metrics"]["exact_match"] for row in selected) / len(selected), 6),
            "idk_rate": round(sum(bool(row["metrics"]["idk"]) for row in selected) / len(selected), 6),
        }
    return result


def _write_report(questions: list[dict[str, Any]], rows: dict[tuple[str, str], dict[str, Any]], metrics_summary: dict[str, Any]) -> None:
    corrected = [row["trace"] for (qid, system), row in rows.items() if system == "corrected_structured_kg_rag"]
    path_rate = sum(bool(item["required_path_found"]) for item in corrected) / len(corrected)
    provenance_rate = sum(bool(item["source_provenance_available"]) for item in corrected) / len(corrected)
    REPORT.write_text(
        f"""# Final structured KG human-quality validation

## Experimental objective

This preregistered validation compares vanilla dense RAG, corrected
provenance-preserving structured KG-RAG, and BM25 + LLM on the frozen
26-question `kg_required=YES` subset. It tests whether corrected structured
KG evidence improves human-rated factual quality. No human scores have been
assigned or analyzed.

## Frozen identity and protocol

- Benchmark SHA-256: `7f5daf5419b0a4cb70e5c3d5eba021a8a39118b5a0fd235ce19b81f75324ffce`
- Questions: {len(questions)}
- Generations: {len(rows)}
- Model: `mistral:7b-instruct-q4_K_M`
- Options: `{json.dumps(OPTIONS, sort_keys=True)}`
- Blind randomization seed: `{BLIND_SEED}` (recorded only in the manifest)

## Systems

1. Vanilla RAG
2. Corrected Structured KG-RAG
3. BM25 + LLM

## Preliminary lexical diagnostics

These descriptive metrics are not factual-quality conclusions.

| System | ROUGE-L | Reference-token coverage | Exact match | IDK rate |
|---|---:|---:|---:|---:|
{chr(10).join(f"| {name} | {value['rouge_l']:.6f} | {value['reference_token_coverage']:.6f} | {value['exact_match_rate']:.6f} | {value['idk_rate']:.6f} |" for name, value in metrics_summary.items())}

## Corrected KG retrieval and provenance

- Required paths handled: {sum(bool(item["required_path_found"]) for item in corrected)}/{len(corrected)} ({path_rate:.6f})
- Selected paths with provenance: {sum(bool(item["source_provenance_available"]) for item in corrected)}/{len(corrected)} ({provenance_rate:.6f})

## Files created

- `data/results/final_structured_kg_eval/generation_results.jsonl`
- `data/results/final_structured_kg_eval/preliminary_metrics.json`
- `data/results/final_structured_kg_eval/run_manifest.json`
- `data/results/final_structured_kg_eval/traces/`
- `data/annotations/final_structured_kg_human_eval.csv`
- `data/annotations/final_structured_kg_human_eval_guidelines.md`
- `data/results/final_structured_kg_eval/final_structured_kg_eval_report.md`

Previous benchmark, corpus, Phase 4, Phase 5, Phase 6, and paper artifacts
were not overwritten by this experiment.

**Human factual-quality conclusions are pending blinded annotation.**
""",
        encoding="utf8",
    )


def run() -> None:
    questions = _questions()
    rows = _read_rows()
    kg = CanonicalKG()
    from src.generator.ollama_api import generate_with_metrics

    OUTPUT.mkdir(parents=True, exist_ok=True)
    TRACES.mkdir(parents=True, exist_ok=True)
    for question in questions:
        for system in SYSTEMS:
            key = (question["question_id"], system)
            if key in rows:
                continue
            trace, context = _trace(question, system, kg)
            answer, telemetry = generate_with_metrics(question["question"], context, options=OPTIONS)
            row = {
                "question_id": question["question_id"],
                "question": question["question"],
                "system": system,
                "reference_answer": question["reference_answer"],
                "answer": answer,
                "trace": trace,
                "metrics": metrics(answer, question["reference_answer"]),
                "generation_telemetry": telemetry,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            with CHECKPOINT.open("a", encoding="utf8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            rows[key] = row
            (TRACES / f"{question['question_id']}_{system}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
            print(f"checkpointed {len(rows)} / 78 {question['question_id']} {system}", flush=True)
    if len(rows) != 78:
        raise RuntimeError(f"Expected 78 generation rows, found {len(rows)}")
    summary = _summarize(rows)
    (OUTPUT / "preliminary_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    _write_human_package(questions, rows)
    _write_guidelines()
    _write_report(questions, rows, summary)
    manifest = {
        "benchmark_sha256": sha256(BENCHMARK),
        "question_count": 26,
        "generation_count": 78,
        "systems": SYSTEMS,
        "model": "mistral:7b-instruct-q4_K_M",
        "generation_options": OPTIONS,
        "blind_randomization_seed": BLIND_SEED,
        "code": "src/evaluation/run_final_structured_kg_eval.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
    }
    for path in (CHECKPOINT, OUTPUT / "preliminary_metrics.json", HUMAN_CSV, GUIDELINES, REPORT):
        manifest[f"{path.name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
