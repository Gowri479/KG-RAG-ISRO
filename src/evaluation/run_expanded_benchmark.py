"""Run the frozen 180-question benchmark with resumable JSONL checkpointing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.bm25_llm import _load_bm25, retrieve_context as bm25_context
from src.baselines.vanilla_rag import retrieve_context as vanilla_context
from src.generator.ollama_api import generate
from src.retriever import hybrid, kg_retriever
from src.retriever.faiss_retriever import _load_chunks as _load_faiss_chunks, _load_index as _load_faiss_index
from src.retriever.hybrid import retrieve as kgrag_context
from src.retriever.embedding_model import get_embedding_model

BENCHMARK_PATH = ROOT / "data" / "benchmark" / "isro_qa.json"
TEST_IDS_PATH = ROOT / "data" / "benchmark" / "test_ids.json"
DEFAULT_CHECKPOINT = ROOT / "data" / "results" / "evaluation_expanded_checkpoint.jsonl"
DEFAULT_ANSWERS = ROOT / "data" / "results" / "answers_aditya_expanded.json"
DEFAULT_EVALUATION = ROOT / "data" / "results" / "evaluation_results_aditya_expanded.json"
OPTIONS = {"num_predict": 150, "temperature": 0.1, "num_ctx": 2048}
UNKNOWN = "I don't know."


def _path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def _frozen_questions(limit: int | None) -> list[dict]:
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8-sig"))
    by_id = {item["id"]: item for item in benchmark}
    test_ids = json.loads(TEST_IDS_PATH.read_text(encoding="utf-8-sig"))
    if len(test_ids) != 180 or len(set(test_ids)) != 180:
        raise RuntimeError("The canonical test split is not the expected unique 180-question list")
    questions = [by_id[qid] for qid in test_ids]
    return questions[:limit] if limit is not None else questions


def _read_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row.get("id")
        if not qid or qid in rows:
            raise RuntimeError(f"Invalid or duplicate checkpoint row at line {line_number}")
        rows[qid] = row
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _prewarm() -> None:
    """Load all reusable state before the first benchmark question."""
    _load_bm25()
    _load_faiss_chunks()
    _load_faiss_index()
    get_embedding_model()
    kg_retriever._load_graph()
    hybrid._get_nlp()


def _limit_context(context: str, max_tokens: int = 1500) -> str:
    return " ".join((context or "").split()[:max_tokens])


def _run_one_system(system: str, question: str) -> str:
    if system == "kg_rag":
        context = kgrag_context(question, passage_limit=3, max_tokens=1500)
    elif system == "bm25_llm":
        context = bm25_context(question, top_k=3)
    else:
        context = vanilla_context(question, top_k=3)
    return generate(question, _limit_context(context), options=OPTIONS) or UNKNOWN


def _run_question(item: dict) -> dict:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            system: executor.submit(_run_one_system, system, item["question"])
            for system in ("kg_rag", "bm25_llm", "vanilla_rag")
        }
        answers = {system: future.result() for system, future in futures.items()}
    return {
        "id": item["id"],
        "question": item["question"],
        "reference_answer": item.get("answer", ""),
        "tier": item.get("tier"),
        "kgrag_answer": answers["kg_rag"],
        "bm25_answer": answers["bm25_llm"],
        "vanilla_rag_answer": answers["vanilla_rag"],
        "graphrag_answer": "N/A",
        "checkpoint_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_wall_time_ms": round((time.perf_counter() - started) * 1000, 4),
    }


def _write_answers(path: Path, questions: list[dict], rows: dict[str, dict]) -> None:
    ordered = []
    for item in questions:
        row = dict(rows[item["id"]])
        row.pop("checkpoint_completed_at_utc", None)
        row.pop("question_wall_time_ms", None)
        ordered.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen expanded benchmark with JSONL checkpointing.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--answers-output", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--evaluation-output", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--seed-json", type=Path, default=None, help="Optional prior partial JSON array to migrate once into the checkpoint.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N frozen test IDs for validation.")
    parser.add_argument("--question-workers", type=int, default=1, help="Independent questions to process concurrently; 1 preserves serialized question order.")
    parser.add_argument("--finalize", action="store_true", help="Write answers and final evaluation only after all selected IDs are complete.")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.question_workers <= 0:
        parser.error("--question-workers must be positive")

    checkpoint = _path(args.checkpoint)
    answers_output = _path(args.answers_output)
    evaluation_output = _path(args.evaluation_output)
    questions = _frozen_questions(args.limit)
    target_ids = {item["id"] for item in questions}
    rows = _read_jsonl(checkpoint)

    if args.seed_json is not None and not rows:
        seed_path = _path(args.seed_json)
        seeded = json.loads(seed_path.read_text(encoding="utf-8-sig"))
        for row in seeded:
            if row["id"] not in target_ids:
                continue
            _append_jsonl(checkpoint, row)
            rows[row["id"]] = row

    invalid = set(rows) - target_ids
    if invalid:
        raise RuntimeError(f"Checkpoint contains IDs outside this run: {sorted(invalid)}")

    _prewarm()
    pending = [item for item in questions if item["id"] not in rows]
    if args.question_workers == 1:
        completed = ((_run_question(item), item) for item in pending)
        for row, item in completed:
            _append_jsonl(checkpoint, row)
            rows[item["id"]] = row
            print(f"checkpointed {len(rows)}/{len(questions)} {item['id']}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.question_workers) as executor:
            futures = {executor.submit(_run_question, item): item for item in pending}
            for future in as_completed(futures):
                item = futures[future]
                row = future.result()
                _append_jsonl(checkpoint, row)
                rows[item["id"]] = row
                print(f"checkpointed {len(rows)}/{len(questions)} {item['id']}", flush=True)

    complete = all(item["id"] in rows for item in questions)
    if args.finalize and complete:
        _write_answers(answers_output, questions, rows)
        subprocess.run([
            sys.executable,
            str(ROOT / "src" / "evaluation" / "evaluate.py"),
            "--results", str(answers_output),
            "--output", str(evaluation_output),
        ], check=True)
    elif args.finalize:
        raise RuntimeError("Refusing to finalize an incomplete expanded benchmark")

    print(json.dumps({"complete": complete, "completed": len(rows), "selected": len(questions), "checkpoint": str(checkpoint)}, indent=2))


if __name__ == "__main__":
    main()
