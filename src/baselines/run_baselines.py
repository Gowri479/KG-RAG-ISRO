"""Run short sequential baseline generations over the ISRO benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
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


def _run_systems(question: str) -> tuple[str, str, str]:
    """Run the unchanged three system calls concurrently for one question."""
    with ThreadPoolExecutor(max_workers=3) as executor:
        kg_future = executor.submit(_short_answer, question, kgrag_context(question, passage_limit=3, max_tokens=1500))
        bm25_future = executor.submit(_short_answer, question, bm25_context(question, top_k=3))
        vanilla_future = executor.submit(_short_answer, question, vanilla_context(question, top_k=3))
        return kg_future.result(), bm25_future.result(), vanilla_future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KG-RAG and baseline systems over the ISRO benchmark.")
    parser.add_argument("--sample", type=int, default=None, help="Run only the first N benchmark questions.")
    parser.add_argument("--resume", action="store_true", help="Skip questions already saved in baseline_results.json.")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH, help="Output JSON path; defaults to the canonical baseline path.")
    parser.add_argument("--ids-file", type=Path, default=None, help="Optional JSON list of benchmark IDs to run, preserving file order.")
    args = parser.parse_args()

    if args.sample is not None and args.sample < 0:
        parser.error("--sample must be non-negative")

    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8-sig"))
    if args.ids_file is not None:
        requested_ids = json.loads(args.ids_file.read_text(encoding="utf-8-sig"))
        id_order = {question_id: index for index, question_id in enumerate(requested_ids)}
        benchmark = sorted((item for item in benchmark if item["id"] in id_order), key=lambda item: id_order[item["id"]])
    benchmark = benchmark[:args.sample] if args.sample is not None else benchmark
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    completed_ids = set()
    if args.resume and output_path.exists():
        results = json.loads(output_path.read_text(encoding="utf-8-sig"))
        completed_ids = {item.get("id") for item in results}

    for index, item in enumerate(benchmark, 1):
        if item["id"] in completed_ids:
            continue

        question = item["question"]
        kgrag_answer, bm25_answer, vanilla_answer = _run_systems(question)

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
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if index % 10 == 0:
            print(f"Processed {index}/{len(benchmark)} questions", flush=True)

    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
