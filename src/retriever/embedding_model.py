"""Shared lazy cache for the configured sentence-transformer model."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load all-MiniLM-L6-v2 once per benchmark process."""
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
