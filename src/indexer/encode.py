"""Encode chunk texts into embeddings for the FAISS index."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - fallback when package is absent
    SentenceTransformer = None


def _fallback_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Simple deterministic fallback embedding for environments without sentence-transformers."""
    seed = sum(ord(ch) for ch in text) % 9973
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.zeros(dim, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def encode_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Encode a list of chunk texts. Falls back to deterministic random vectors if the model is unavailable."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    if SentenceTransformer is not None:
        try:
            model = SentenceTransformer(model_name)
            embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(embeddings, dtype=np.float32)
        except Exception:
            pass

    return np.vstack([_fallback_embedding(text) for text in texts]).astype(np.float32)


def encode_chunks(chunk_dir: str | Path, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Read chunk JSON files from a directory and return one embedding per chunk."""
    chunk_path = Path(chunk_dir)
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk directory does not exist: {chunk_path}")

    texts: list[str] = []
    combined_path = chunk_path / "chunks.json"
    files = [combined_path] if combined_path.exists() else sorted(chunk_path.glob("*.json"))

    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and "text" in item:
                    texts.append(str(item["text"]))
        elif isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, str):
                    texts.append(value)

    return encode_texts(texts, model_name=model_name)


if __name__ == "__main__":
    sample = ["ISRO launched Chandrayaan-3 from Sriharikota."]
    emb = encode_texts(sample)
    print(emb.shape)

