"""Run the isolated relation-aware A-G experiment with resumable traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.evaluate_relational_qa import metrics
from src.generator.prompt import build_user_prompt
from src.retriever.relation_aware import RelationAwareKG, fuse_relation_evidence

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "relation_aware_benchmark" / "relation_aware_qa_v2.json"
OUTPUT = ROOT / "data" / "results" / "relation_aware"
CHECKPOINT = OUTPUT / "generation_results.jsonl"
OPTIONS = {"temperature": 0.1, "num_predict": 150, "num_ctx": 2048}
CONTEXT_LIMIT = 1500
SYSTEMS = ("vanilla_dense_rag", "corrected_structured_kg_rag", "relation_aware_kg_rag", "relation_aware_kg_only", "relation_aware_kg_source", "relation_aware_kg_dense", "relation_aware_kg_dense_bm25")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_questions() -> list[dict]:
    payload = json.loads(BENCHMARK.read_text(encoding="utf8"))
    return payload["questions"]


def dense_trace(question: str) -> dict:
    from src.evaluation.evaluate_relational_qa import _vanilla_trace
    return _vanilla_trace(question)


def bm25_trace(question: str) -> dict:
    from src.evaluation.evaluate_relational_qa import _bm25_trace
    return _bm25_trace(question)


def structured_trace(question: dict, dense: dict) -> dict:
    from src.retriever.kg_diagnostic_retriever import diagnostic_object
    trace = diagnostic_object(question["question"], question["kg_required"] == "YES", dense["retrieved_text"])
    return trace


def relation_trace(question: dict, kg: RelationAwareKG, dense: dict, bm25: dict) -> dict:
    intent = kg.analyze(question["question"])
    dense_scores = dict(zip(dense["retrieved_chunk_ids"], dense["retrieved_chunk_scores"]))
    bm25_scores = dict(zip(bm25["retrieved_chunk_ids"], bm25["retrieved_chunk_scores"]))
    paths = kg.retrieve(question["question"], dense_scores, bm25_scores)
    required = set(question["supporting_paths"][0])
    required_found = any(required.issubset(set(path.triple_ids)) for path in paths)
    evidence = [path for path in paths if path.source_chunks]
    return {
        "query_intent": {
            "query_type": intent.query_type,
            "detected_entities": list(intent.entity_ids),
            "detected_relations": list(intent.relation_types),
            "relation_direction": intent.direction,
            "hop_depth": intent.hop_depth,
            "explicit_kg_requirement": intent.requires_explicit_path,
        },
        "retrieved_kg_path_ids": [path.path_id for path in paths],
        "retrieved_triple_ids": [list(path.triple_ids) for path in paths],
        "provenance_ids": [list(path.source_chunks) for path in paths],
        "required_path_found": required_found,
        "valid_provenance": all(bool(path.source_chunks and path.source_urls) for path in evidence),
        "evidence": [path.__dict__ for path in paths],
        "dense_chunks": dense["retrieved_text"],
        "bm25_chunks": bm25["retrieved_text"],
    }


def context_for(system: str, relation: dict, dense: dict, bm25: dict) -> str:
    paths = [
        type("PathEvidence", (), path)()
        for path in relation["evidence"]
    ]
    if system == "vanilla_dense_rag":
        text = "\n\n".join(dense["retrieved_text"])
    elif system == "corrected_structured_kg_rag":
        text = "\n\n".join(dense["retrieved_text"])
    else:
        use_dense = system in {"relation_aware_kg_rag", "relation_aware_kg_dense", "relation_aware_kg_dense_bm25", "relation_aware_kg_source"}
        use_bm25 = system == "relation_aware_kg_dense_bm25"
        fused = fuse_relation_evidence(paths, dense["retrieved_text"] if use_dense else [], bm25["retrieved_text"] if use_bm25 else [], CONTEXT_LIMIT)
        text = fused["context_text"]
    return " ".join(text.split()[:CONTEXT_LIMIT])


def read_rows() -> dict[tuple[str, str], dict]:
    if not CHECKPOINT.exists():
        return {}
    rows = {}
    for line in CHECKPOINT.read_text(encoding="utf8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = (row["question_id"], row["system"])
            if key in rows:
                raise RuntimeError(f"Duplicate row: {key}")
            rows[key] = row
    return rows


def main(limit: int | None = None) -> None:
    from src.generator.ollama_api import generate

    questions = load_questions()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    kg = RelationAwareKG()
    rows = read_rows()
    target = questions[:limit] if limit else questions
    for question in target:
        dense = dense_trace(question["question"])
        bm25 = bm25_trace(question["question"])
        relation = relation_trace(question, kg, dense, bm25)
        for system in SYSTEMS:
            key = (question["question_id"], system)
            if key in rows:
                continue
            started = time.perf_counter()
            context = context_for(system, relation, dense, bm25)
            answer = generate(question["question"], context, options=OPTIONS) or "I don't know."
            intent = relation["query_intent"]
            row = {
                "question_id": question["question_id"],
                "question": question["question"],
                "system": system,
                "answer": answer,
                "reference_answer": question["reference_answer"],
                "category": question["category"],
                "kg_required": question["kg_required"],
                "split": question["split"],
                "relation_type": question["relation_type"],
                "detected_query_type": intent["query_type"],
                "detected_entities": intent["detected_entities"],
                "detected_relation": intent["detected_relations"],
                "relation_direction": intent["relation_direction"],
                "hop_depth": intent["hop_depth"],
                "kg_path_ids": relation["retrieved_kg_path_ids"],
                "provenance_ids": relation["provenance_ids"],
                "required_path_found": relation["required_path_found"],
                "valid_provenance": relation["valid_provenance"],
                "selected_evidence": relation["evidence"],
                "context": context,
                "context_token_count": len(context.split()),
                "final_prompt": build_user_prompt(context, question["question"]),
                "metrics": metrics(answer, question["reference_answer"]),
                "generation_configuration": OPTIONS,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            with CHECKPOINT.open("a", encoding="utf8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows[key] = row
            print(f"checkpointed {len(rows)}/{len(target) * len(SYSTEMS)} {key}", flush=True)
    manifest = {
        "benchmark_sha256": sha256(BENCHMARK),
        "benchmark_questions": len(target),
        "systems": list(SYSTEMS),
        "expected_generations": len(target) * len(SYSTEMS),
        "model": "mistral:7b-instruct-q4_K_M",
        "options": OPTIONS,
        "context_token_budget": CONTEXT_LIMIT,
        "seed": 20261008,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "frozen_weights": {"dense": 0.20, "lexical": 0.15, "entity": 0.20, "relation": 0.30, "provenance": 0.15},
        "leakage_control": "Benchmark is generated from canonical triples before any answer generation; no evaluation output is read by retrieval.",
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"complete": len(rows) == len(target) * len(SYSTEMS), "rows": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    main(args.limit)
