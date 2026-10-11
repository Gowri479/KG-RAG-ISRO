"""LightRAG-inspired local baseline for ISRO domain question answering.

Inspired by LightRAG (Guo et al., 2024), this baseline implements dual-level
graph retrieval over the local domain knowledge graph:
1. Low-level retrieval: Specific entity and relationship triples matching the query entities.
2. High-level retrieval: Broad community/thematic context from modularity-based graph communities.

Uses local SentenceTransformer embeddings and local Mistral-7B via Ollama.
Labeled explicitly as 'LightRAG-inspired baseline' (local reproduction).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generator.ollama_api import generate, generate_with_metrics
from src.retriever.embedding_model import get_embedding_model
from src.retriever.kg_retriever import _load_graph, get_kg_context
from src.retriever.hybrid import _extract_entities

KG_PATH = ROOT / "data" / "kg" / "knowledge_graph.json"


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return get_embedding_model()


COMMUNITY_CACHE = ROOT / "data" / "kg" / "graphrag_community_summaries.json"


def _build_communities() -> tuple[list[str], np.ndarray]:
    summaries = []
    if COMMUNITY_CACHE.exists():
        try:
            summaries = json.loads(COMMUNITY_CACHE.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not summaries and KG_PATH.exists():
        payload = json.loads(KG_PATH.read_text(encoding="utf-8"))
        G = nx.Graph()
        for node in payload.get("nodes", []):
            if isinstance(node, dict) and node.get("id"):
                G.add_node(node["id"])
        for edge in payload.get("edges", []):
            if isinstance(edge, dict) and edge.get("source") and edge.get("target"):
                G.add_edge(edge["source"], edge["target"], relation=edge.get("relation", "related_to"))

        if G.number_of_nodes() > 0:
            communities = list(nx.community.greedy_modularity_communities(G))
            for comm in communities[:30]:
                members = sorted(list(comm))
                if len(members) < 2:
                    continue
                sub = G.subgraph(comm)
                rel_samples = [f"{u} -{d.get('relation', 'related_to')}-> {v}" for u, v, d in sub.edges(data=True)]
                summaries.append(
                    f"Community Theme: {', '.join(members[:8])}...\n"
                    f"Key Relations: {'; '.join(rel_samples[:6]) if rel_samples else 'structural co-occurrence'}"
                )

    if not summaries:
        return [], np.empty((0, 384), dtype=np.float32)

    model = _load_model()
    vectors = model.encode(summaries, convert_to_numpy=True, normalize_embeddings=True)
    return summaries, np.asarray(vectors, dtype=np.float32)


_SUMMARIES, _VECTORS = _build_communities()


def retrieve_context(query: str, max_tokens: int = 1500) -> str:
    """Perform dual-level retrieval: low-level entity triples + high-level community theme."""
    if not query or not query.strip():
        return ""

    # 1. Low-level retrieval (specific entity triples)
    entities = _extract_entities(query)
    low_level_context = get_kg_context(entities, two_hop=False) if entities else ""

    # 2. High-level retrieval (community theme)
    high_level_context = ""
    if _SUMMARIES:
        model = _load_model()
        q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores = np.asarray(q_vec, dtype=np.float32) @ _VECTORS.T
        best_comm_idx = int(np.argmax(scores[0]))
        high_level_context = f"[High-Level Community Context]\n{_SUMMARIES[best_comm_idx]}"

    parts = []
    if low_level_context:
        parts.append(f"[Low-Level Entity Triples]\n{low_level_context}")
    if high_level_context:
        parts.append(high_level_context)

    combined = "\n\n".join(parts)
    tokens = combined.split()
    if len(tokens) > max_tokens:
        combined = " ".join(tokens[:max_tokens])
    return combined.strip()


def answer(query: str) -> str:
    context = retrieve_context(query)
    return generate(query, context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LightRAG-inspired baseline")
    parser.add_argument("--question", required=True)
    print(answer(parser.parse_args().question))
