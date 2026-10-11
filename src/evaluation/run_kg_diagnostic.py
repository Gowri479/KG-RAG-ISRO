"""Run the isolated five-variant diagnostic on the frozen 26-question subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from src.evaluation.evaluate_relational_qa import metrics, sha256
from src.retriever.kg_diagnostic_retriever import CanonicalKG, diagnostic_object, format_structured_evidence

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "relational_benchmark" / "relational_qa_v1.json"
OUTPUT_DIR = ROOT / "data" / "results"
CHECKPOINT = OUTPUT_DIR / "kg_diagnostic_26.jsonl"
MANIFEST = OUTPUT_DIR / "kg_diagnostic_manifest.json"
MODEL_OPTIONS = {"num_predict": 150, "temperature": 0.1, "num_ctx": 2048}
VARIANTS = ("DENSE_ONLY", "KG_ONLY", "KG_PLUS_SOURCE", "DENSE_PLUS_KG_STRUCTURED", "DENSE_PLUS_KG_UNSTRUCTURED")
CONTEXT_LIMIT = 1500


def _dense(question: str) -> str:
    from src.retriever.faiss_retriever import get_passage_context

    return get_passage_context(question, top_k=3)


def _old_kg(question: str, dense: str) -> str:
    from src.retriever import hybrid, kg_retriever

    entities = hybrid._extract_entities(question)
    keywords = hybrid._query_keywords(question)
    entities = [entity for entity in entities if any(keyword in entity.lower() for keyword in keywords)]
    graph = kg_retriever._load_graph()
    triples = []
    for entity in entities:
        triples.extend(kg_retriever._get_one_hop(graph, entity))
    unique = list(dict.fromkeys(triples))
    kg_context = kg_retriever._serialize_triples(unique)
    return " ".join((kg_context + "\n\n" + dense).split()[:CONTEXT_LIMIT])


def _context(variant: str, dense: str, structured: str, unstructured: str) -> str:
    if variant == "DENSE_ONLY":
        return " ".join(dense.split()[:CONTEXT_LIMIT])
    if variant == "KG_ONLY":
        return " ".join(structured.split()[:CONTEXT_LIMIT])
    if variant == "KG_PLUS_SOURCE":
        return " ".join(structured.split()[:CONTEXT_LIMIT])
    if variant == "DENSE_PLUS_KG_STRUCTURED":
        return " ".join((structured + "\n\nRETRIEVED DOCUMENT EVIDENCE\n" + dense).split()[:CONTEXT_LIMIT])
    return unstructured


def _required_status(question: dict, paths: list[dict]) -> tuple[str, str]:
    required_paths = [set(path) for path in question.get("supporting_paths", [])]
    retrieved = [set(path["triple_ids"]) for path in paths]
    if not required_paths:
        return "unknown", "unknown"
    relation_found = any(
        triple_id in path
        for triple_id in set().union(*required_paths)
        for path in retrieved
    )
    path_found = any(required.issubset(path) for required in required_paths for path in retrieved)
    return str(relation_found).lower(), str(path_found).lower()


def _run(limit: int | None = None) -> None:
    questions = [
        question for question in json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
        if question["kg_required"] == "YES"
    ]
    if len(questions) != 26:
        raise RuntimeError("Frozen KG-required subset is not exactly 26 questions")
    existing = {}
    if CHECKPOINT.exists():
        for line in CHECKPOINT.read_text(encoding="utf8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[(row["question_id"], row["variant"])] = row
    targets = questions[:limit] if limit else questions
    kg = CanonicalKG()
    from src.generator.ollama_api import generate_with_metrics

    for question in targets:
        dense = _dense(question["question"])
        base = diagnostic_object(question["question"], True, dense.split("\n\n"))
        path_status = _required_status(question, base["retrieved_paths"])
        base["required_relation_found"], base["required_path_found"] = path_status
        kg_only = base["kg_path_evidence"]
        structured = base["kg_evidence"]
        unstructured = _old_kg(question["question"], dense)
        for variant in VARIANTS:
            key = (question["question_id"], variant)
            if key in existing:
                continue
            context = _context(variant, dense, kg_only if variant == "KG_ONLY" else structured, unstructured)
            answer, telemetry = generate_with_metrics(question["question"], context, options=MODEL_OPTIONS)
            row = {
                "question_id": question["question_id"],
                "question": question["question"],
                "variant": variant,
                "reference_answer": question["reference_answer"],
                "retrieval": {
                    **base,
                    "fused_evidence": context,
                    "generation_input": context,
                    "context_token_estimate": len(context.split()),
                    "answer": answer,
                },
                "answer": answer,
                "metrics": metrics(answer, question["reference_answer"]),
                "generation_telemetry": telemetry,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with CHECKPOINT.open("a", encoding="utf8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            existing[key] = row
            print(f"checkpointed {len(existing)} / 130 {question['question_id']} {variant}", flush=True)
    if len(existing) == 130:
        MANIFEST.write_text(json.dumps({
            "benchmark_sha256": sha256(BENCHMARK),
            "model": "mistral:7b-instruct-q4_K_M",
            "generation_options": MODEL_OPTIONS,
            "variants": VARIANTS,
            "question_count": 26,
            "evaluation_count": 130,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
            "code": "src/evaluation/run_kg_diagnostic.py",
        }, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    _run(args.limit)
