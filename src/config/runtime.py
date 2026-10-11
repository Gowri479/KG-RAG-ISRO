"""Application runtime configuration.

Centralizes runtime options, model settings, token budgets, and authoritative
artifact paths, decoupling operational services from evaluation scripts.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Model and generation runtime settings
DEFAULT_SEED = 20261008
DEFAULT_TEMPERATURE = 0.1
DEFAULT_NUM_PREDICT = 150
DEFAULT_NUM_CTX = 2048

DEFAULT_GENERATION_OPTIONS = {
    "temperature": DEFAULT_TEMPERATURE,
    "num_predict": DEFAULT_NUM_PREDICT,
    "num_ctx": DEFAULT_NUM_CTX,
    "seed": DEFAULT_SEED,
}

# Context token & word limits
DEFAULT_CONTEXT_LIMIT = 1500  # maximum words in fused context
DEFAULT_TOP_K = 3

# Data artifact paths
CANONICAL_BENCHMARK_PATH = ROOT / "data" / "relation_aware_benchmark" / "relation_aware_qa_v2.json"
CORRECTED_RERUN_RESULTS_PATH = ROOT / "data" / "results" / "corrected_rerun" / "generation_results.jsonl"
CORPUS_CHUNKS_PATH = ROOT / "data" / "corpus" / "chunks.jsonl"
CORPUS_TRIPLES_PATH = ROOT / "data" / "corpus" / "triples.jsonl"
CORPUS_ENTITIES_PATH = ROOT / "data" / "corpus" / "entities.jsonl"
