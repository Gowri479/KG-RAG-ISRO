"""Application search services for dense vector and lexical BM25 retrieval.

Provides isolated runtime retrieval functions used by the API and evaluation runners,
preventing application runtime code from depending on evaluation test scripts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config.runtime import DEFAULT_TOP_K


def retrieve_bm25_chunks(question: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Retrieve top-scoring text chunks using BM25 lexical ranking."""
    from src.baselines.bm25_llm import _load_bm25, _load_chunks, _tokenize

    chunks = _load_chunks()
    scores = _load_bm25().get_scores(_tokenize(question))
    indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
    return {
        "retrieved_chunk_ids": [str(index) for index in indices],
        "retrieved_chunk_scores": [float(scores[index]) for index in indices],
        "retrieved_text": [chunks[index] for index in indices],
    }


def retrieve_dense_chunks(question: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Retrieve top-scoring text chunks using dense vector embeddings and FAISS."""
    from src.retriever.embedding_model import get_embedding_model
    from src.retriever.faiss_retriever import _load_chunks, _load_index, _query_keywords

    chunks = _load_chunks()
    index = _load_index()
    vector = get_embedding_model().encode([question], convert_to_numpy=True, normalize_embeddings=True)
    limit = min(max(top_k * 4, 20), index.ntotal)
    distances, indices = index.search(np.asarray(vector, dtype=np.float32), limit)
    ranked = [
        (int(index_id), float(score), chunks[int(index_id)])
        for score, index_id in zip(distances[0], indices[0])
        if 0 <= int(index_id) < len(chunks)
    ]
    keywords = _query_keywords(question)
    filtered = [item for item in ranked if any(keyword in item[2].lower() for keyword in keywords)]
    selected = filtered or ranked
    selected = selected[:top_k]
    return {
        "retrieved_chunk_ids": [str(index_id) for index_id, _, _ in selected],
        "retrieved_chunk_scores": [score for _, score, _ in selected],
        "retrieved_text": [text for _, _, text in selected],
    }
