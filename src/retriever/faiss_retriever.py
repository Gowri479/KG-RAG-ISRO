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


def get_passage_context(query: str) -> str:
    """Return the top-5 passage texts most similar to the query."""
    if not query or not INDEX_PATH.exists():
        return ""

    chunks = _load_chunks()
    if not chunks:
        return ""

    index = faiss.read_index(str(INDEX_PATH))
    query_vector = _MODEL.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(np.asarray(query_vector, dtype=np.float32), 5)

    matches: list[str] = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(chunks):
            continue
        text = chunks[int(idx)].strip()
        if text:
            matches.append(text)

    return "\n\n".join(matches).strip()


if __name__ == "__main__":
    print(get_passage_context("What is ISRO?"))

