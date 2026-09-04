"""BM25 + Ollama baseline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generator.ollama_api import generate

CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.json"


def _load_chunks() -> list[str]:
    payload = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("chunks", [])
    return [item["text"].strip() for item in items if isinstance(item, dict) and item.get("text")]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def retrieve_context(question: str, top_k: int = 5) -> str:
    """Retrieve the highest-scoring BM25 passages without generation."""
    chunks = _load_chunks()
    if not chunks:
        return ""
    index = BM25Okapi([_tokenize(chunk) for chunk in chunks])
    scores = index.get_scores(_tokenize(question))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return "\n\n".join(chunks[i] for i in top_indices)


def answer(question: str) -> str:
    """Retrieve BM25 passages and generate an answer."""
    context = retrieve_context(question)
    return generate(question, context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BM25 baseline.")
    parser.add_argument("--question", required=True)
    print(answer(parser.parse_args().question))

