"""Dense passage retrieval using FAISS and sentence-transformer embeddings."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import faiss
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "data" / "index" / "faiss_index.index"
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.json"

_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def _load_chunks() -> list[str]:
    if not CHUNKS_PATH.exists():
        return []
    payload = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("chunks", payload.get("items", []))
    else:
        items = []
    texts: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("text") or item.get("content") or ""
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        elif isinstance(item, str):
            text = item.strip()
            if text:
                texts.append(text)
    return texts


def _query_keywords(query: str) -> list[str]:
    return [word.lower() for word in query.split() if len(word) > 3]


def _search(index, query_vector: np.ndarray, chunks: list[str], limit: int = 5) -> list[str]:
    if not chunks:
        return []
    chunk_vectors = _MODEL.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    filtered_index = faiss.IndexFlatL2(chunk_vectors.shape[1])
    filtered_index.add(np.asarray(chunk_vectors, dtype=np.float32))
    _, indices = filtered_index.search(np.asarray(query_vector, dtype=np.float32), min(limit, len(chunks)))
    return [chunks[int(idx)] for idx in indices[0] if 0 <= idx < len(chunks)]


def get_passage_context(query: str, top_k: int = 5) -> str:
    if not query or not INDEX_PATH.exists():
        return ""
    chunks = _load_chunks()
    if not chunks:
        return ""
    query_vector = _MODEL.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    keywords = _query_keywords(query)
    keyword_chunks = [
        chunk for chunk in chunks
        if any(keyword in chunk.lower() for keyword in keywords)
    ]
    if len(keyword_chunks) > 20:
        matches = _search(None, np.asarray(query_vector, dtype=np.float32), keyword_chunks, limit=top_k)
    else:
        index = faiss.read_index(str(INDEX_PATH))
        _, indices = index.search(np.asarray(query_vector, dtype=np.float32), top_k)
        matches = [chunks[int(idx)] for idx in indices[0] if 0 <= idx < len(chunks)]
    cleaned_matches: list[str] = []
    for text in matches:
        text = text.strip()
        if text:
            cleaned_matches.append(text)
    return "\n\n".join(cleaned_matches[:top_k]).strip()


if __name__ == "__main__":
    print(get_passage_context("What is ISRO?"))