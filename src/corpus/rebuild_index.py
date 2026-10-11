"""Rebuild the FAISS index with the configured cached MiniLM encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode all chunks and write the matching FAISS index.")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    chunks_path = ROOT / "data" / "chunks" / "chunks.json"
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("chunks", [])
    texts = [item["text"].strip() for item in items if isinstance(item, dict) and item.get("text")]
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    vectors = model.encode(texts, batch_size=args.batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    vectors = np.asarray(vectors, dtype=np.float32)

    import faiss

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    output_path = ROOT / "data" / "index" / "faiss_index.index"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    print(json.dumps({"chunks": len(texts), "dimension": int(vectors.shape[1]), "index": str(output_path)}))


if __name__ == "__main__":
    main()
