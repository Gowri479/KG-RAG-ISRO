"""Plot publication-quality figures for Q1 journal paper.

Generates:
1. paper/figures/kg_hop_ablation_comparison.png (Ablation across 7 variants)
2. paper/figures/graph_baselines_comparison.png (GraphRAG, LightRAG, BM25, Vanilla, KG-RAG)
3. paper/figures/hardware_resource_comparison.png (Latency and memory profiling)
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Styling for academic paper
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})


def plot_hop_ablation() -> None:
    ablation_json = ROOT / "data" / "results" / "kg_hop_ablation_results.json"
    if not ablation_json.exists():
        print(f"Missing {ablation_json}")
        return

    data = json.loads(ablation_json.read_text(encoding="utf-8"))
    summary = data.get("variants_summary", {})

    variants = [
        ("kg_only", "KG-only", "#9b59b6"),
        ("dense_only", "Dense-only\n(Vanilla)", "#3498db"),
        ("dense_kg_1hop", "Dense +\nKG 1-hop", "#2980b9"),
        ("dense_kg_2hop", "Dense +\nKG 2-hop", "#1f618d"),
        ("bm25_only", "BM25-only", "#e67e22"),
        ("bm25_kg", "BM25 +\nKG 1-hop", "#d35400"),
        ("full_kg_rag", "Full KG-RAG\n(Keyword+1H+D)", "#27ae60"),
    ]

    labels = [v[1] for v in variants]
    colors = [v[2] for v in variants]
    rouge_l = [summary[v[0]]["rouge_l_mean"] for v in variants]
    coverage = [summary[v[0]]["coverage_mean"] for v in variants]
    idk_pct = [summary[v[0]]["idk_rate"] * 100 for v in variants]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5.2))

    # Bar chart for ROUGE-L and Coverage
    b1 = ax1.bar(x - width/2, rouge_l, width, label="ROUGE-L", color="#2b5c8f", edgecolor="black", alpha=0.9)
    b2 = ax1.bar(x + width/2, coverage, width, label="Coverage", color="#4ba3c3", edgecolor="black", alpha=0.9)

    ax1.set_ylabel("Score (0 - 1.0)", color="#1a252f", fontweight="bold")
    ax1.set_ylim(0, 0.75)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Line overlay for IDK rate
    ax2 = ax1.twinx()
    l1 = ax2.plot(x, idk_pct, color="#e74c3c", marker="o", linewidth=2.2, markersize=7, label="IDK Rate (%)")
    ax2.set_ylabel("IDK Abstention Rate (%)", color="#c0392b", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.grid(False)

    # Value annotations on ROUGE bars
    for rect in b1:
        height = rect.get_height()
        ax1.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", framealpha=0.95)

    plt.title("Ablation Study: Retrieval Variants & Neighborhood Hop Analysis (Aditya-L1 Benchmark)", pad=15, fontweight="bold")
    plt.tight_layout()
    out_path = FIG_DIR / "kg_hop_ablation_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved hop ablation figure to {out_path}")


def plot_graph_baselines() -> None:
    results_json = ROOT / "data" / "results" / "graph_baselines_comparison.json"
    if not results_json.exists():
        print(f"Missing {results_json}")
        return

    data = json.loads(results_json.read_text(encoding="utf-8"))
    systems_data = data.get("systems", {})

    order = [
        ("GraphRAG-style baseline", "GraphRAG-style\n(Modularity Comm.)", "#95a5a6"),
        ("LightRAG-inspired baseline", "LightRAG-inspired\n(Dual-Level)", "#7f8c8d"),
        ("BM25 + LLM", "BM25 + LLM\n(Sparse Lexical)", "#e67e22"),
        ("Vanilla RAG", "Vanilla RAG\n(MiniLM Dense)", "#3498db"),
        ("KG-RAG (ours)", "KG-RAG (Ours)\n(Hybrid Subgraph)", "#27ae60"),
    ]

    labels = [o[1] for o in order]
    colors = ["#95a5a6", "#7f8c8d", "#e67e22", "#3498db", "#27ae60"]
    rouge_l = [systems_data[o[0]]["rouge_l"] for o in order]
    coverage = [systems_data[o[0]]["coverage"] for o in order]
    idk = [systems_data[o[0]]["idk_rate"] * 100 for o in order]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 5))
    rects1 = ax1.bar(x - width/2, rouge_l, width, label="ROUGE-L", color="#1f4e79", edgecolor="black", alpha=0.9)
    rects2 = ax1.bar(x + width/2, coverage, width, label="Coverage", color="#5dade2", edgecolor="black", alpha=0.9)

    ax1.set_ylabel("Evaluation Metric Score", fontweight="bold")
    ax1.set_ylim(0, 0.70)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(x, idk, color="#c0392b", marker="s", linewidth=2.2, markersize=7, label="IDK Rate (%)")
    ax2.set_ylabel("IDK Abstention Rate (%)", color="#c0392b", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.grid(False)

    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", framealpha=0.95)

    plt.title("Comparative Evaluation of Graph-Based Retrieval Architectures (Aditya-L1)", pad=14, fontweight="bold")
    plt.tight_layout()
    out_path = FIG_DIR / "graph_baselines_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved graph baselines figure to {out_path}")


def plot_hardware_profiling() -> None:
    profile_json = ROOT / "data" / "results" / "resource_profile_final.json"
    if not profile_json.exists():
        print(f"Missing {profile_json}")
        return

    data = json.loads(profile_json.read_text(encoding="utf-8"))
    sys_bench = data.get("system_benchmarks", {})

    systems = ["BM25 + LLM", "Vanilla RAG", "GraphRAG-style", "KG-RAG"]
    ret_lat = [sys_bench[s]["retrieval_latency_ms_mean"] for s in systems]
    gen_lat = [sys_bench[s]["generation_latency_ms_mean"] / 1000.0 for s in systems]
    peak_ram = [sys_bench[s]["peak_process_rss_mb"] for s in systems]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Latency Plot
    x = np.arange(len(systems))
    ax1.bar(x, gen_lat, color="#34495e", edgecolor="black", label="Generation Latency (s)", alpha=0.85)
    ax1.set_ylabel("Generation Latency (seconds)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, rotation=15, ha="right", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.set_title("Per-Query Generation Latency (Mistral-7B Local)", pad=10, fontweight="bold")

    # RAM Footprint Plot
    ax2.bar(x, peak_ram, color="#16a085", edgecolor="black", label="Process Peak RSS (MB)", alpha=0.85)
    ax2.set_ylabel("Process Memory Footprint (MB RSS)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(systems, rotation=15, ha="right", fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    ax2.set_title("Process Memory Consumption (RAM MB)", pad=10, fontweight="bold")

    plt.tight_layout()
    out_path = FIG_DIR / "hardware_resource_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved hardware profiling figure to {out_path}")


if __name__ == "__main__":
    plot_hop_ablation()
    plot_graph_baselines()
    plot_hardware_profiling()
