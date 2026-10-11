"""Generate publication figure for Corpus Distraction / Expansion Analysis."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

NAVY = "#012258"
TEAL = "#0E86D4"
RED = "#C00000"
ORANGE = "#E65100"
GREEN = "#2E7D32"
LIGHT_GREEN = "#81C784"


def plot_corpus_distraction():
    systems = ["BM25 + LLM", "Vanilla RAG", "KG-RAG (ours)"]
    original_rouge = [0.2915, 0.2780, 0.2736]
    expanded_rouge = [0.3135, 0.2483, 0.2568]
    changes_rouge = [e - o for o, e in zip(original_rouge, expanded_rouge)]

    original_cov = [0.4340, 0.3989, 0.3921]
    expanded_cov = [0.4682, 0.3601, 0.3705]
    changes_cov = [e - o for o, e in zip(original_cov, expanded_cov)]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    x = np.arange(len(systems))
    width = 0.32

    # Panel 1: ROUGE-L comparison
    bars1 = axes[0].bar(x - width/2, original_rouge, width, label="Original Corpus (4,557 chunks)", color="#4682B4", edgecolor="white")
    bars2 = axes[0].bar(x + width/2, expanded_rouge, width, label="Expanded Corpus (+58 Aditya chunks)", color="#FF8C00", edgecolor="white")
    axes[0].set_ylabel("ROUGE-L Score")
    axes[0].set_title("(a) Retrieval Competition: ROUGE-L Impact", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(systems)
    axes[0].set_ylim(0, 0.38)
    axes[0].legend(loc="upper right", framealpha=0.9)

    for i, (b1, b2, chg) in enumerate(zip(bars1, bars2, changes_rouge)):
        axes[0].text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.005, f"{original_rouge[i]:.3f}", ha="center", fontsize=7.5)
        axes[0].text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.005, f"{expanded_rouge[i]:.3f}\n({chg:+.3f})", ha="center", fontsize=7.5, color="black" if chg > 0 else "#B71C1C", fontweight="bold")

    # Panel 2: Reference-token Coverage comparison
    bars3 = axes[1].bar(x - width/2, original_cov, width, label="Original Corpus", color="#4682B4", edgecolor="white")
    bars4 = axes[1].bar(x + width/2, expanded_cov, width, label="Expanded Corpus", color="#FF8C00", edgecolor="white")
    axes[1].set_ylabel("Token Coverage")
    axes[1].set_title("(b) Retrieval Competition: Coverage Impact", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(systems)
    axes[1].set_ylim(0, 0.55)

    for i, (b3, b4, chg) in enumerate(zip(bars3, bars4, changes_cov)):
        axes[1].text(b3.get_x() + b3.get_width()/2, b3.get_height() + 0.005, f"{original_cov[i]:.3f}", ha="center", fontsize=7.5)
        axes[1].text(b4.get_x() + b4.get_width()/2, b4.get_height() + 0.005, f"{expanded_cov[i]:.3f}\n({chg:+.3f})", ha="center", fontsize=7.5, color="black" if chg > 0 else "#B71C1C", fontweight="bold")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "corpus_distraction_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_corpus_distraction()
