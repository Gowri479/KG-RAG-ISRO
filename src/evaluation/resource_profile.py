"""Profile measured retrieval costs without claiming unmeasured LLM costs."""

from __future__ import annotations

import csv
import argparse
import json
import sys
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.analysis_utils import RESULTS_DIR, load_frozen_rows, ROOT


def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _profile_call(fn, question: str, **kwargs) -> tuple[str, float, float | None]:
    before = _rss_mb()
    start = time.perf_counter()
    context = fn(question, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    after = _rss_mb()
    delta = (after - before) if before is not None and after is not None else None
    return context or "", elapsed, delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile measured retrieval costs; generation is never invoked.")
    parser.add_argument("--systems", nargs="+", choices=["bm25_llm", "vanilla_rag", "kg_rag"], default=["bm25_llm", "vanilla_rag", "kg_rag"])
    parser.add_argument("--questions-per-tier", type=int, default=4)
    args = parser.parse_args()
    rows = []
    selected = []
    for tier in (1, 2, 3):
        selected.extend([row for row in load_frozen_rows() if int(row.get("tier", 0)) == tier][:args.questions_per_tier])
    system_specs = {
        "bm25_llm": ("src.baselines.bm25_llm", "retrieve_context", {}),
        "vanilla_rag": ("src.baselines.vanilla_rag", "retrieve_context", {"top_k": 3}),
        "kg_rag": ("src.retriever.hybrid", "retrieve", {"passage_limit": 3, "max_tokens": 1500}),
    }
    errors = {}
    for system, (module_name, function_name, kwargs) in system_specs.items():
        if system not in args.systems:
            continue
        try:
            module = __import__(module_name, fromlist=[function_name])
            fn = getattr(module, function_name)
            for row in selected:
                try:
                    context, elapsed, rss_delta = _profile_call(fn, row["question"], **kwargs)
                    rows.append({
                        "system": system,
                        "id": row["id"],
                        "tier": row.get("tier"),
                        "retrieval_latency_ms": round(elapsed, 4),
                        "context_tokens": len(context.split()),
                        "rss_delta_mb": round(rss_delta, 4) if rss_delta is not None else None,
                    })
                except Exception as exc:
                    errors[f"{system}:{row['id']}"] = str(exc)
        except Exception as exc:
            errors[system] = str(exc)

    summaries = {}
    for system in args.systems:
        values = [row["retrieval_latency_ms"] for row in rows if row["system"] == system]
        rss = [row["rss_delta_mb"] for row in rows if row["system"] == system and row["rss_delta_mb"] is not None]
        summaries[system] = {
            "n_queries": len(values),
            "retrieval_latency_ms_mean": round(statistics.mean(values), 4) if values else None,
            "retrieval_latency_ms_sd": round(statistics.stdev(values), 4) if len(values) > 1 else None,
            "rss_delta_mb_mean": round(statistics.mean(rss), 4) if rss else None,
            "rss_measurement": "process RSS delta per retrieval call; not peak RAM",
            "model_loading_ms": None,
            "generation_latency_ms": None,
            "total_query_latency_ms": None,
            "tokens_per_sec": None,
            "peak_vram_mb": None,
            "kg_load_time_ms": None,
            "faiss_load_time_ms": None,
        }
    result = {
        "experiment": "resource_latency_profile",
        "status": "executed_retrieval_only" if rows and len(summaries) == 3 else ("executed_partial" if rows else "blocked"),
        "subset": f"{len(selected)} frozen test questions ({args.questions_per_tier} per tier), retrieval only",
        "systems": summaries,
        "unmeasured_systems": [system for system in system_specs if system not in summaries],
        "limitations": [
            "LLM generation was not rerun; generation latency and tokens/sec are unavailable.",
            "Peak RAM/VRAM and isolated KG/FAISS load times were not instrumented by the existing implementation.",
            "The measured RSS field is process RSS delta, not peak memory.",
        ],
        "errors": errors,
        "per_query": rows,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = RESULTS_DIR / "resource_profile.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (RESULTS_DIR / "resource_profile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", "id", "tier", "retrieval_latency_ms", "context_tokens", "rss_delta_mb"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")
    print(f"Measured {len(rows)} retrieval calls; errors={len(errors)}")


if __name__ == "__main__":
    main()
