"""Pure dense FAISS + Ollama baseline without KG or keyword filtering."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generator.ollama_api import generate

INDEX_PATH = ROOT / "data" / "index" / "faiss_index.index"
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.json"
MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def _load_chunks() -> list[str]:
    payload = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("chunks", [])
    return [item["text"].strip() for item in items if isinstance(item, dict) and item.get("text")]


def retrieve_context(question: str, top_k: int = 5) -> str:
    """Retrieve dense passages using the prebuilt full FAISS index."""
    chunks = _load_chunks()
    if not chunks or not INDEX_PATH.exists():
        return ""
    index = faiss.read_index(str(INDEX_PATH))
    vector = MODEL.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    _, indices = index.search(np.asarray(vector, dtype=np.float32), top_k)
    passages = [chunks[int(i)] for i in indices[0] if 0 <= i < len(chunks)]
    return "\n\n".join(passages)


def answer(question: str) -> str:
    """Retrieve dense passages and generate an answer."""
    context = retrieve_context(question)
    return generate(question, context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the vanilla dense RAG baseline.")
    parser.add_argument("--question", required=True)
    print(answer(parser.parse_args().question))

