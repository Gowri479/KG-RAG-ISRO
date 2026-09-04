"""Run short sequential baseline generations over the ISRO benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.bm25_llm import retrieve_context as bm25_context
from src.baselines.vanilla_rag import retrieve_context as vanilla_context
from src.generator.ollama_api import generate
from src.retriever.hybrid import retrieve as kgrag_context

BENCHMARK_PATH = ROOT / "data" / "benchmark" / "isro_qa.json"
RESULTS_PATH = ROOT / "data" / "results" / "baseline_results.json"
UNKNOWN = "I don't know."
OLLAMA_OPTIONS = {
    "num_predict": 150,
    "temperature": 0.1,
    "num_ctx": 2048,
}


def _limit_context(context: str, max_tokens: int = 1500) -> str:
    return " ".join((context or "").split()[:max_tokens])


def _short_answer(question: str, context: str) -> str:
    return generate(question, _limit_context(context), options=OLLAMA_OPTIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KG-RAG and baseline systems over the ISRO benchmark.")
    parser.add_argument("--sample", type=int, default=None, help="Run only the first N benchmark questions.")
    parser.add_argument("--resume", action="store_true", help="Skip questions already saved in baseline_results.json.")
    args = parser.parse_args()

    if args.sample is not None and args.sample < 0:
        parser.error("--sample must be non-negative")

    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8-sig"))
    benchmark = benchmark[:args.sample] if args.sample is not None else benchmark
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    completed_ids = set()
    if args.resume and RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8-sig"))
        completed_ids = {item.get("id") for item in results}

    for index, item in enumerate(benchmark, 1):
        if item["id"] in completed_ids:
            continue

        question = item["question"]
        kgrag_answer = _short_answer(question, kgrag_context(question, passage_limit=3, max_tokens=1500))
        bm25_answer = _short_answer(question, bm25_context(question, top_k=3))
        vanilla_answer = _short_answer(question, vanilla_context(question, top_k=3))

        results.append({
            "id": item["id"],
            "question": question,
            "reference_answer": item.get("answer", ""),
            "tier": item.get("tier"),
            "kgrag_answer": kgrag_answer or UNKNOWN,
            "bm25_answer": bm25_answer or UNKNOWN,
            "vanilla_rag_answer": vanilla_answer or UNKNOWN,
            "graphrag_answer": "N/A",
        })
        completed_ids.add(item["id"])
        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if index % 10 == 0:
            print(f"Processed {index}/{len(benchmark)} questions", flush=True)

    print(f"Saved {len(results)} results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
