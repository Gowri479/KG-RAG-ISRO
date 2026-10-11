"""Hardware Resource and Efficiency Profiler for Q1 Journal Study.

Measures actual local system performance:
- CPU, RAM, GPU/VRAM configuration
- Indexing / load times (BM25, FAISS, KG)
- Retrieval latency (mean +- sd)
- Generation latency (mean +- sd)
- End-to-end query latency (mean +- sd)
- Tokens per second generation throughput
- Process Peak RAM (RSS MB)
- Model VRAM footprint (MB)

Outputs:
- data/results/resource_profile_final.json
- data/results/resource_profile_final.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import psutil

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.bm25_llm import retrieve_context as bm25_retrieve, _load_bm25
from src.baselines.graphrag import retrieve_context as graphrag_retrieve
from src.baselines.vanilla_rag import retrieve_context as vanilla_retrieve, _load_index
from src.evaluation.analysis_utils import TEST_IDS_PATH, load_frozen_rows, load_json
from src.generator.ollama_api import generate_with_metrics
from src.retriever.hybrid import retrieve as hybrid_retrieve
from src.retriever.kg_retriever import _load_graph

RESULTS_JSON = ROOT / "data" / "results" / "resource_profile_final.json"
RESULTS_CSV = ROOT / "data" / "results" / "resource_profile_final.csv"


def get_vram_info() -> dict:
    # Try querying Ollama running model VRAM
    vram_mb = 0.0
    try:
        req = urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=5)
        data = json.loads(req.read().decode())
        models = data.get("models", [])
        if models:
            vram_bytes = models[0].get("size_vram", 0)
            vram_mb = round(vram_bytes / (1024 * 1024), 2)
    except Exception:
        pass

    # Try nvidia-smi if available
    gpu_name = "N/A"
    total_gpu_mem_mb = None
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = res.stdout.strip().split(",")
            gpu_name = parts[0].strip()
            total_gpu_mem_mb = float(parts[1].strip())
    except Exception:
        pass

    return {
        "gpu_name": gpu_name,
        "total_gpu_memory_mb": total_gpu_mem_mb,
        "active_model_vram_mb": vram_mb,
    }


def measure_indexing_times() -> dict:
    t0 = time.perf_counter()
    _ = _load_bm25()
    bm25_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _ = _load_index()
    faiss_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _ = _load_graph()
    kg_time = (time.perf_counter() - t0) * 1000

    return {
        "bm25_index_load_ms": round(bm25_time, 2),
        "faiss_index_load_ms": round(faiss_time, 2),
        "kg_graph_load_ms": round(kg_time, 2),
    }


def run_profiling(n_repetitions: int = 10) -> None:
    process = psutil.Process()
    ram_total_mb = round(psutil.virtual_memory().total / (1024 * 1024), 2)
    vram_info = get_vram_info()
    cpu_info = {
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "os": f"{platform.system()} {platform.release()}",
    }

    print("Measuring index loading times...")
    indexing_times = measure_indexing_times()

    # Pick representative questions across tiers
    rows = load_frozen_rows()
    selected_queries = [r["question"] for r in rows[:n_repetitions]]

    systems = {
        "BM25 + LLM": lambda q: bm25_retrieve(q, top_k=3),
        "Vanilla RAG": lambda q: vanilla_retrieve(q, top_k=3),
        "GraphRAG-style": lambda q: graphrag_retrieve(q),
        "KG-RAG": lambda q: hybrid_retrieve(q, passage_limit=3, max_tokens=1500),
    }

    profile_records = []
    system_summaries = {}

    for sys_name, ret_fn in systems.items():
        print(f"\nProfiling {sys_name} on {len(selected_queries)} queries...", flush=True)
        ret_latencies = []
        gen_latencies = []
        e2e_latencies = []
        tokens_per_sec_list = []
        peak_rss_list = []

        for q in selected_queries:
            mem_before = process.memory_info().rss / (1024 * 1024)

            # Retrieval
            t_ret0 = time.perf_counter()
            context = ret_fn(q)
            ret_ms = (time.perf_counter() - t_ret0) * 1000
            ret_latencies.append(ret_ms)

            # Generation
            t_gen0 = time.perf_counter()
            ans, metrics = generate_with_metrics(q, context)
            gen_ms = (time.perf_counter() - t_gen0) * 1000
            gen_latencies.append(gen_ms)

            e2e_ms = ret_ms + gen_ms
            e2e_latencies.append(e2e_ms)

            # Telemetry tokens/sec
            eval_count = metrics.get("generated_tokens")
            eval_duration_ns = metrics.get("eval_duration_ns")
            if eval_count and eval_duration_ns and eval_duration_ns > 0:
                tps = eval_count / (eval_duration_ns / 1e9)
                tokens_per_sec_list.append(tps)

            mem_after = process.memory_info().rss / (1024 * 1024)
            peak_rss_list.append(max(mem_before, mem_after))

            profile_records.append({
                "system": sys_name,
                "question": q[:60] + "...",
                "retrieval_ms": round(ret_ms, 2),
                "generation_ms": round(gen_ms, 2),
                "e2e_ms": round(e2e_ms, 2),
                "tokens_per_sec": round(tokens_per_sec_list[-1], 2) if tokens_per_sec_list else None,
                "process_rss_mb": round(mem_after, 2),
            })
            print(f"[{sys_name}] Query {len(ret_latencies)}/{len(selected_queries)} - E2E: {e2e_ms:.1f}ms (Ret: {ret_ms:.1f}ms, Gen: {gen_ms:.1f}ms)", flush=True)

        system_summaries[sys_name] = {
            "n_queries": len(selected_queries),
            "retrieval_latency_ms_mean": round(mean(ret_latencies), 2),
            "retrieval_latency_ms_sd": round(stdev(ret_latencies), 2) if len(ret_latencies) > 1 else 0.0,
            "generation_latency_ms_mean": round(mean(gen_latencies), 2),
            "generation_latency_ms_sd": round(stdev(gen_latencies), 2) if len(gen_latencies) > 1 else 0.0,
            "e2e_latency_ms_mean": round(mean(e2e_latencies), 2),
            "e2e_latency_ms_sd": round(stdev(e2e_latencies), 2) if len(e2e_latencies) > 1 else 0.0,
            "tokens_per_sec_mean": round(mean(tokens_per_sec_list), 2) if tokens_per_sec_list else None,
            "tokens_per_sec_sd": round(stdev(tokens_per_sec_list), 2) if len(tokens_per_sec_list) > 1 else 0.0,
            "peak_process_rss_mb": round(max(peak_rss_list), 2),
            "active_model_vram_mb": vram_info["active_model_vram_mb"],
        }

    # Save CSV
    fieldnames = ["system", "question", "retrieval_ms", "generation_ms", "e2e_ms", "tokens_per_sec", "process_rss_mb"]
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(profile_records)

    # Save JSON
    final_output = {
        "experiment": "hardware_resource_profiling",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_environment": {
            "cpu": cpu_info,
            "total_system_ram_mb": ram_total_mb,
            "gpu": vram_info,
        },
        "indexing_and_load_times_ms": indexing_times,
        "system_benchmarks": system_summaries,
    }
    RESULTS_JSON.write_text(json.dumps(final_output, indent=2), encoding="utf-8")
    print(f"\nResource profiling complete! Saved results to {RESULTS_JSON} and {RESULTS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=8, help="Repetitions per system")
    args = parser.parse_args()
    run_profiling(n_repetitions=args.repetitions)
