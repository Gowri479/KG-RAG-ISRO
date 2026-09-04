"""Community-summary GraphRAG + Ollama baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generator.ollama_api import generate

KG_PATH = ROOT / "data" / "kg" / "knowledge_graph.json"
MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def _build_community_summaries() -> list[str]:
    payload = json.loads(KG_PATH.read_text(encoding="utf-8"))
    graph = nx.Graph()
    graph.add_nodes_from(node["id"] for node in payload.get("nodes", []) if isinstance(node, dict) and node.get("id"))
    for edge in payload.get("edges", []):
        if isinstance(edge, dict) and edge.get("source") and edge.get("target"):
            graph.add_edge(edge["source"], edge["target"], relation=edge.get("relation", "related_to"))

    if not graph:
        return []
    communities = nx.community.greedy_modularity_communities(graph)
    summaries = []
    for community in communities:
        members = set(community)
        relations = []
        for source, target, data in graph.edges(data=True):
            if source in members and target in members:
                relations.append(f"{source} -{data.get('relation', 'related_to')}-> {target}")
        summaries.append(
            "Entities: " + ", ".join(sorted(members)) +
            "\nRelations: " + ("; ".join(relations) if relations else "none")
        )
    return summaries


COMMUNITY_SUMMARIES = _build_community_summaries()
COMMUNITY_VECTORS = (
    MODEL.encode(COMMUNITY_SUMMARIES, convert_to_numpy=True, normalize_embeddings=True)
    if COMMUNITY_SUMMARIES else np.empty((0, 384), dtype=np.float32)
)


def _query_keywords(question: str) -> list[str]:
    return [word.lower().strip(".,?!") for word in question.split() if len(word.strip(".,?!")) > 3]


def retrieve_context(question: str) -> str:
    """Retrieve the most similar KG community summary without generation."""
    if not COMMUNITY_SUMMARIES:
        return ""
    query_vector = MODEL.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    scores = np.asarray(query_vector, dtype=np.float32) @ np.asarray(COMMUNITY_VECTORS, dtype=np.float32).T
    keywords = _query_keywords(question)
    candidates = [
        index for index, summary in enumerate(COMMUNITY_SUMMARIES)
        if any(keyword in summary.lower() for keyword in keywords)
    ]
    if candidates:
        best = max(candidates, key=lambda index: scores[0][index])
    else:
        best = int(np.argmax(scores[0]))
    return COMMUNITY_SUMMARIES[best]


def answer(question: str) -> str:
    """Retrieve the most similar KG community summary and generate an answer."""
    context = retrieve_context(question)
    return generate(question, context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the GraphRAG baseline.")
    parser.add_argument("--question", required=True)
    print(answer(parser.parse_args().question))

