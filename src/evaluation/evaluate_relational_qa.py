"""Evaluate the frozen relational QA benchmark without changing retrieval behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "relational_benchmark" / "relational_qa_v1.json"
FREEZE = ROOT / "data" / "relational_benchmark" / "relational_qa_v1.freeze.json"
OUTPUT_DIR = ROOT / "data" / "results" / "relational_qa_v1"
CHECKPOINT = OUTPUT_DIR / "per_question_results.jsonl"
MANIFEST = OUTPUT_DIR / "run_manifest.json"
UNKNOWN = "I don't know."
SYSTEMS = ("bm25_llm", "vanilla_rag", "kg_rag")
OPTIONS = {"num_predict": 150, "temperature": 0.1, "num_ctx": 2048}
CONTEXT_LIMIT = 1500
TOP_K = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", (text or "").lower().strip())


def is_idk(text: str) -> bool:
    value = (text or "").lower().strip()
    return not value or "i don't know" in value or "i do not know" in value or "don't know" in value


def lcs(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, 1):
            current.append(previous[index - 1] + 1 if token == other else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def metrics(answer: str, reference: str) -> dict[str, Any]:
    prediction = normalize(answer).split()
    target = normalize(reference).split()
    if not prediction or not target or is_idk(answer):
        rouge = coverage = exact = 0.0
    else:
        overlap = lcs(prediction, target)
        precision = overlap / len(prediction)
        recall = overlap / len(target)
        rouge = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        coverage = len(set(prediction) & set(target)) / len(set(target))
        exact = float(normalize(answer) == normalize(reference))
    return {
        "rouge_l": round(rouge, 6),
        "reference_token_coverage": round(coverage, 6),
        "exact_match": exact,
        "idk": is_idk(answer),
    }


def _load_chunks():
    from src.baselines.bm25_llm import _load_chunks

    return _load_chunks()


def _bm25_trace(question: str) -> dict[str, Any]:
    from src.baselines.bm25_llm import _load_bm25, _tokenize

    chunks = _load_chunks()
    scores = _load_bm25().get_scores(_tokenize(question))
    indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:TOP_K]
    return {
        "retrieved_chunk_ids": [str(index) for index in indices],
        "retrieved_chunk_scores": [float(scores[index]) for index in indices],
        "retrieved_text": [chunks[index] for index in indices],
    }


def _vanilla_trace(question: str) -> dict[str, Any]:
    import faiss
    from src.retriever.embedding_model import get_embedding_model
    from src.retriever.faiss_retriever import INDEX_PATH, _load_index, _load_chunks as load_faiss_chunks, _query_keywords

    chunks = load_faiss_chunks()
    index = _load_index()
    vector = get_embedding_model().encode([question], convert_to_numpy=True, normalize_embeddings=True)
    limit = min(max(TOP_K * 4, 20), index.ntotal)
    distances, indices = index.search(np.asarray(vector, dtype=np.float32), limit)
    ranked = [(int(index), float(score), chunks[int(index)]) for score, index in zip(distances[0], indices[0]) if 0 <= int(index) < len(chunks)]
    keywords = _query_keywords(question)
    filtered = [item for item in ranked if any(keyword in item[2].lower() for keyword in keywords)]
    selected = filtered or ranked
    selected = selected[:TOP_K]
    return {
        "retrieved_chunk_ids": [str(index) for index, _, _ in selected],
        "retrieved_chunk_scores": [score for _, score, _ in selected],
        "retrieved_text": [text for _, _, text in selected],
    }


def _kg_trace(question: str) -> dict[str, Any]:
    from src.retriever import hybrid, kg_retriever
    from src.retriever.faiss_retriever import get_passage_context

    entities = hybrid._extract_entities(question)
    keywords = hybrid._query_keywords(question)
    entities = [entity for entity in entities if any(keyword in entity.lower() for keyword in keywords)]
    graph = kg_retriever._load_graph()
    triples = []
    for entity in entities:
        triples.extend(kg_retriever._get_one_hop(graph, entity))
    unique = []
    seen = set()
    for triple in triples:
        if triple not in seen:
            unique.append(triple)
            seen.add(triple)
    kg_context = kg_retriever._serialize_triples(unique)
    dense_context = get_passage_context(question, top_k=TOP_K)
    fused = "\n\n".join(part.strip() for part in (kg_context, dense_context) if part.strip())
    fused = " ".join(fused.split()[:CONTEXT_LIMIT])
    return {
        "detected_entities": entities,
        "retrieved_kg_triples": [
            {"subject": subject, "relation": relation, "object": object_}
            for subject, relation, object_ in unique
        ],
        "retrieved_kg_paths": [],
        "hop_depth": 1 if unique else 0,
        "dense_context": dense_context,
        "retrieved_text": dense_context.split("\n\n") if dense_context else [],
        "retrieved_chunk_ids": [],
        "retrieved_chunk_scores": [],
        "kg_context": kg_context,
        "final_fused_evidence": fused,
    }


def _context_for(system: str, question: str, trace: dict[str, Any]) -> str:
    if system == "kg_rag":
        return trace["final_fused_evidence"]
    return " ".join("\n\n".join(trace["retrieved_text"]).split()[:CONTEXT_LIMIT])


def _trace_for(system: str, question: str) -> dict[str, Any]:
    if system == "bm25_llm":
        return _bm25_trace(question)
    if system == "vanilla_rag":
        return _vanilla_trace(question)
    return _kg_trace(question)


def _load_questions() -> list[dict[str, Any]]:
    expected = json.loads(FREEZE.read_text(encoding="utf-8"))
    if sha256(BENCHMARK) != expected["benchmark_sha256"]:
        raise RuntimeError("Frozen relational benchmark hash mismatch")
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    questions = payload["questions"]
    if len(questions) != 62 or Counter(item["kg_required"] for item in questions) != Counter({"YES": 26, "NO": 36}):
        raise RuntimeError("Frozen relational benchmark count mismatch")
    return questions


def _read_checkpoint() -> dict[tuple[str, str], dict[str, Any]]:
    if not CHECKPOINT.exists():
        return {}
    rows = {}
    for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["question_id"], row["system"])
        if key in rows:
            raise RuntimeError(f"Duplicate checkpoint row {key}")
        rows[key] = row
    return rows


def _append(row: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _run_one(question: dict[str, Any], system: str) -> dict[str, Any]:
    from src.generator.ollama_api import generate

    started = time.perf_counter()
    trace = _trace_for(system, question["question"])
    context = _context_for(system, question["question"], trace)
    answer = generate(question["question"], context, options=OPTIONS) or UNKNOWN
    row = {
        "question_id": question["question_id"],
        "question": question["question"],
        "system": system,
        "generated_answer": answer,
        "reference_answer": question["reference_answer"],
        "acceptable_answers": question.get("acceptable_answers", [question["reference_answer"]]),
        "answerable": question.get("answerable", "ANSWERABLE"),
        "category": question["category"],
        "kg_required": question["kg_required"],
        "relation_type": question["relation_type"],
        "mission": question["mission"],
        "match_group_id": question["match_group_id"],
        "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "retrieval_trace": trace,
        "generation_configuration_id": "canonical_mistral_options_v1",
        "generation_configuration": OPTIONS,
        "metrics": metrics(answer, question["reference_answer"]),
        "wall_time_ms": round((time.perf_counter() - started) * 1000, 4),
    }
    return row


def _git_commit() -> str | None:
    result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _write_manifest(questions: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark_sha256": sha256(BENCHMARK),
        "benchmark_size": len(questions),
        "systems": list(SYSTEMS),
        "model": "mistral:7b-instruct-q4_K_M",
        "inference_backend": "Ollama local API",
        "prompt_module": "src/generator/prompt.py",
        "prompt_sha256": sha256(ROOT / "src/generator/prompt.py"),
        "generation_options": OPTIONS,
        "retrieval": {"bm25_top_k": TOP_K, "dense_top_k": TOP_K, "kg_hops": 1, "context_token_budget": CONTEXT_LIMIT},
        "embedding_model": "repository default via src/retriever/embedding_model.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "trace_limitations": [
            "BM25 and dense chunk IDs are indices from the existing pilot index.",
            "The existing KG-RAG API exposes serialized one-hop triples but not source-linked path IDs; retrieved_kg_paths is empty rather than fabricated.",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run(limit: int | None = None) -> None:
    questions = _load_questions()
    _write_manifest(questions)
    existing = _read_checkpoint()
    target = questions[:limit] if limit is not None else questions
    pending = [(question, system) for question in target for system in SYSTEMS if (question["question_id"], system) not in existing]
    if pending:
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(_run_one, question, system) for question, system in pending]
            for future in futures:
                row = future.result()
                _append(row)
                existing[(row["question_id"], row["system"])] = row
                print(f"checkpointed {len(existing)} / {len(target) * 3} {row['question_id']} {row['system']}", flush=True)
    print(json.dumps({"complete": len(existing) == len(target) * 3, "evaluations": len(existing)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.limit)


if __name__ == "__main__":
    main()
