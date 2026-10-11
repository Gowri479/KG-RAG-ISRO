"""Profile the current expanded benchmark implementation on six frozen questions."""

from __future__ import annotations

import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.bm25_llm import _tokenize, retrieve_context as bm25_context
from src.baselines.vanilla_rag import retrieve_context as vanilla_context
from src.generator.ollama_api import generate_with_metrics
from src.retriever import hybrid
from src.retriever.hybrid import retrieve as kgrag_context

BENCHMARK_PATH = ROOT / "data" / "benchmark" / "isro_qa.json"
TEST_IDS_PATH = ROOT / "data" / "benchmark" / "test_ids.json"
OUTPUT_PATH = ROOT / "data" / "results" / "aditya_l1_runtime_diagnostic.json"
OPTIONS = {"num_predict": 150, "temperature": 0.1, "num_ctx": 2048}


def _load_six_questions() -> list[dict]:
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8-sig"))
    test_ids = set(json.loads(TEST_IDS_PATH.read_text(encoding="utf-8-sig")))
    rows = [row for row in benchmark if row["id"] in test_ids]
    by_tier = {tier: [row for row in rows if int(row.get("tier", 0)) == tier] for tier in (1, 2, 3)}
    return [by_tier[tier][0] for tier in (1, 2, 3)] + [by_tier[tier][1] for tier in (1, 2, 3)]


def _profile_system(system: str, question: str) -> dict:
    preprocessing_started = time.perf_counter()
    if system == "bm25_llm":
        _tokenize(question)
    elif system == "kg_rag":
        hybrid._query_keywords(question)
        hybrid._extract_entities(question)
    else:
        question.strip()
    preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000

    retrieval_started = time.perf_counter()
    if system == "bm25_llm":
        context = bm25_context(question, top_k=3)
    elif system == "vanilla_rag":
        context = vanilla_context(question, top_k=3)
    else:
        context = kgrag_context(question, passage_limit=3, max_tokens=1500)
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

    answer, generation = generate_with_metrics(question, " ".join(context.split()[:1500]), options=OPTIONS)
    return {
        "system": system,
        "answer": answer,
        "query_preprocessing_ms": round(preprocessing_ms, 4),
        "retrieval_ms": round(retrieval_ms, 4),
        "context_tokens": len(context.split()),
        "generation": generation,
    }


def main() -> None:
    questions = _load_six_questions()
    per_question = []
    for question in questions:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_profile_system, system, question["question"]) for system in ("bm25_llm", "vanilla_rag", "kg_rag")]
            systems = [future.result() for future in futures]
        per_question.append({
            "id": question["id"],
            "tier": question.get("tier"),
            "question": question["question"],
            "wall_time_ms": round((time.perf_counter() - started) * 1000, 4),
            "systems": systems,
        })

    summaries = {}
    for system in ("bm25_llm", "vanilla_rag", "kg_rag"):
        rows = [row for question in per_question for row in question["systems"] if row["system"] == system]
        request_times = [row["generation"]["request_latency_ms"] for row in rows if row["generation"]["status"] == "ok"]
        summaries[system] = {
            "n": len(rows),
            "query_preprocessing_ms_mean": round(statistics.mean(row["query_preprocessing_ms"] for row in rows), 4),
            "retrieval_ms_mean": round(statistics.mean(row["retrieval_ms"] for row in rows), 4),
            "prompt_construction_ms_mean": round(statistics.mean(row["generation"]["prompt_construction_ms"] for row in rows), 4),
            "ollama_request_ms_mean_successes": round(statistics.mean(request_times), 4) if request_times else None,
            "generation_success_rate": round(sum(row["generation"]["status"] == "ok" for row in rows) / len(rows), 4),
            "timeout_or_error_count": sum(row["generation"]["status"] == "error" for row in rows),
            "generated_tokens_total": sum(row["generation"]["generated_tokens"] or 0 for row in rows),
            "context_tokens_mean": round(statistics.mean(row["context_tokens"] for row in rows), 4),
        }

    report = {
        "experiment": "expanded_runtime_diagnostic",
        "protocol": "six frozen questions; same retrieval settings, prompts, model, and generation options as benchmark runner",
        "questions": [row["id"] for row in per_question],
        "summaries": summaries,
        "per_question": per_question,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
