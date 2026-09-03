import json
from pathlib import Path

from src.indexer.encode import encode_chunks
from src.indexer.build_index import build_faiss_index


def test_encode_chunks_returns_embeddings(tmp_path):
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    payload = [{"text": "ISRO launched Chandrayaan-3 from Sriharikota."}]
    (chunk_dir / "sample.json").write_text(json.dumps(payload), encoding="utf-8")

    embeddings = encode_chunks(chunk_dir)

    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] > 0


def test_build_faiss_index_creates_index_file(tmp_path):
    vectors = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]
    output_dir = tmp_path / "index_out"

    index_path = build_faiss_index(vectors, output_dir)

    assert output_dir.exists()
    assert index_path.exists()
    assert index_path.suffix in {".faiss", ".index"}
