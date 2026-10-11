"""Augment the existing knowledge graph with newly ingested Aditya-L1 chunks."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.kg_builder.build_kg import CHUNKS_PATH, KG_PKL_PATH, build_graph, load_chunks, load_nlp, save_graph


def main() -> None:
    chunks = load_chunks()
    new_chunks = [
        chunk for chunk in chunks
        if isinstance(chunk, dict) and str(chunk.get("document_id", "")).startswith("ADITYA_L1_")
    ]
    if not new_chunks:
        raise RuntimeError("No ADITYA_L1_* chunks found; refusing to modify the existing graph")

    if KG_PKL_PATH.exists():
        with KG_PKL_PATH.open("rb") as handle:
            graph = pickle.load(handle)
        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError(f"Expected MultiDiGraph in {KG_PKL_PATH}, found {type(graph).__name__}")
    else:
        graph = nx.MultiDiGraph()

    removed_existing = 0
    for subject, target, key, data in list(graph.edges(keys=True, data=True)):
        if str(data.get("document_id", "")).startswith("ADITYA_L1_"):
            graph.remove_edge(subject, target, key)
            removed_existing += 1

    before = {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}
    addition = build_graph(new_chunks, load_nlp())
    for subject, target, data in addition.edges(data=True):
        graph.add_edge(subject, target, **dict(data))
    save_graph(graph)

    report = {
        "mode": "incremental_augmentation",
        "chunks_path": str(CHUNKS_PATH.relative_to(ROOT)),
        "new_chunk_count": len(new_chunks),
        "new_document_ids": sorted({chunk["document_id"] for chunk in new_chunks}),
        "removed_existing_aditya_edges": removed_existing,
        "added_nodes": graph.number_of_nodes() - before["nodes"],
        "added_edges": graph.number_of_edges() - before["edges"],
        "before": before,
        "after": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()},
        "new_edges_with_source_url": sum(1 for _, _, data in addition.edges(data=True) if data.get("source_url")),
    }
    report_path = ROOT / "data" / "results" / "aditya_l1_kg_augmentation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
