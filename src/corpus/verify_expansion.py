"""Verify the Aditya-L1 corpus expansion without changing benchmark artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import subprocess
from collections import Counter
from pathlib import Path

import faiss

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_worktree_unchanged(relative_path: str) -> bool:
    result = subprocess.run(["git", "diff", "--quiet", "--", relative_path], cwd=ROOT)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--", relative_path], cwd=ROOT)
    return result.returncode == 0 and staged.returncode == 0


def main() -> None:
    manifest_path = ROOT / "data" / "source_manifest.csv"
    chunks_path = ROOT / "data" / "chunks" / "chunks.json"
    index_path = ROOT / "data" / "index" / "faiss_index.index"
    graph_path = ROOT / "data" / "kg" / "knowledge_graph.json"
    canonical_results = ROOT / "data" / "results" / "evaluation_results.json"

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    aditya_chunks = [chunk for chunk in chunks if str(chunk.get("document_id", "")).startswith("ADITYA_L1_")]
    chunk_counts = Counter(chunk.get("document_id") for chunk in aditya_chunks)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    aditya_edges = [edge for edge in graph.get("edges", []) if str(edge.get("document_id", "")).startswith("ADITYA_L1_")]
    index = faiss.read_index(str(index_path))

    missing_provenance = [
        edge for edge in aditya_edges
        if not edge.get("source_url") or not edge.get("document_id") or not edge.get("page")
    ]
    empty_aditya_chunks = [chunk for chunk in aditya_chunks if not str(chunk.get("text", "")).strip()]
    unavailable = [row["document_id"] for row in manifest if row["status"].startswith("download_failed")]
    downloaded = [row["document_id"] for row in manifest if row["status"] == "downloaded"]

    canonical_hash = sha256(canonical_results) if canonical_results.exists() else None
    canonical_unchanged = git_worktree_unchanged("data/results/evaluation_results.json")
    report = {
        "sources": {
            "manifest_rows": len(manifest),
            "downloaded": downloaded,
            "unavailable_without_substitute": unavailable,
            "extractable_documents": [row["document_id"] for row in manifest if int(row["pages_with_text"] or 0) > 0],
        },
        "chunks": {
            "total": len(chunks),
            "aditya_l1_total": len(aditya_chunks),
            "by_document": dict(sorted(chunk_counts.items())),
            "empty_aditya_l1_chunks": len(empty_aditya_chunks),
        },
        "faiss": {"ntotal": int(index.ntotal), "dimension": int(index.d), "matches_chunk_count": int(index.ntotal) == len(chunks)},
        "knowledge_graph": {
            "nodes": len(graph.get("nodes", [])),
            "edges": len(graph.get("edges", [])),
            "aditya_l1_edges": len(aditya_edges),
            "aditya_l1_edges_missing_provenance": len(missing_provenance),
        },
        "canonical_benchmark": {
            "path": "data/results/evaluation_results.json",
            "sha256": canonical_hash,
            "unchanged_in_git_worktree": canonical_unchanged,
        },
        "checks": {
            "manifest_has_expected_sources": len(manifest) == 7,
            "no_empty_aditya_chunks": not empty_aditya_chunks,
            "index_matches_chunks": int(index.ntotal) == len(chunks),
            "new_graph_edges_have_provenance": not missing_provenance,
            "canonical_benchmark_unchanged": canonical_unchanged,
        },
    }
    report["checks"]["all_passed"] = all(report["checks"].values())
    output = ROOT / "data" / "results" / "aditya_l1_corpus_expansion_report.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["checks"]["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
