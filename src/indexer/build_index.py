"""Build a local FAISS index for chunk embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - fallback for minimal environments
    faiss = None


def build_faiss_index(vectors: list[list[float]] | np.ndarray, output_dir: str | Path) -> Path:
    """Create a FAISS index file from embedding vectors and return its path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros((1, 1), dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if faiss is not None:
        dim = arr.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(arr)
        index_file = output_path / "faiss_index.index"
        faiss.write_index(index, str(index_file))
        return index_file

    metadata_file = output_path / "faiss_index.json"
    metadata = {"shape": list(arr.shape), "vectors": arr.tolist()}
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_file


if __name__ == "__main__":
    vectors = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]
    out = build_faiss_index(vectors, Path("data/index"))
    print(out)

