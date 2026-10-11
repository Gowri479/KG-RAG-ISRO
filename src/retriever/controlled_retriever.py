"""Controlled, interpretable Knowledge Graph retrieval module.

Performs controlled graph traversal:
- Default: 1-hop neighborhood.
- 2-hop traversal only when query analysis detects a multi-hop compositional query.
- Strict budget on retrieved triples to prevent peripheral graph distraction.
- Full provenance recording: detected entities, hop depth, triples, source chunks, paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx

from src.kg_builder.relations import (
    ProvenanceTriple,
    create_provenance_triple,
    normalize_entity,
)
from src.retriever.kg_retriever import _is_noise, _load_graph
from src.retriever.query_classifier import QueryType, classify_query

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "retrieval_config.json"


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("retrieval_parameters", {})
        except Exception:
            pass
    return {
        "top_k_dense": 3,
        "max_kg_triples": 10,
        "max_context_tokens": 1200,
        "default_hop_depth": 1,
        "multihop_hop_depth": 2,
    }


@dataclass
class ControlledKGResult:
    query: str
    query_type: str
    hop_depth_used: int
    entities_detected: list[str]
    triples: list[ProvenanceTriple] = field(default_factory=list)
    source_chunks: list[str] = field(default_factory=list)
    traversal_paths: list[str] = field(default_factory=list)
    serialized_context: str = ""


def controlled_kg_search(
    query: str,
    entities: list[str],
    force_hop_depth: Optional[int] = None,
    max_triples: Optional[int] = None,
) -> ControlledKGResult:
    """Execute controlled graph retrieval with explicit depth and provenance tracking."""
    cfg = _load_config()
    classification = classify_query(query)

    if force_hop_depth is not None:
        hop_depth = force_hop_depth
    else:
        hop_depth = (
            cfg.get("multihop_hop_depth", 2)
            if classification.query_type == QueryType.MULTI_HOP
            else cfg.get("default_hop_depth", 1)
        )

    limit = max_triples or cfg.get("max_kg_triples", 10)
    G = _load_graph()

    normalized_entities = [normalize_entity(e) for e in entities if e.strip()]
    valid_seed_nodes = [e for e in normalized_entities if e in G]

    retrieved_triples: list[ProvenanceTriple] = []
    source_chunk_ids: set[str] = set()
    paths: list[str] = []
    seen_triple_keys: set[tuple[str, str, str]] = set()

    for seed in valid_seed_nodes:
        # 1-Hop Outgoing
        for _, neighbor, data in G.out_edges(seed, data=True):
            rel = data.get("relation", "related_to")
            if _is_noise(seed) or _is_noise(neighbor):
                continue
            t_key = (seed, rel, neighbor)
            if t_key not in seen_triple_keys and len(retrieved_triples) < limit:
                seen_triple_keys.add(t_key)
                src_chunk = str(data.get("chunk_id", data.get("source", "")))
                if src_chunk:
                    source_chunk_ids.add(src_chunk)
                trip = create_provenance_triple(
                    subject=seed,
                    relation=rel,
                    object_=neighbor,
                    document_id=data.get("document_id", ""),
                    chunk_id=src_chunk,
                    source_url=data.get("source_url", data.get("source", "")),
                )
                retrieved_triples.append(trip)
                paths.append(f"{seed} -[{rel}]-> {neighbor}")

        # 1-Hop Incoming
        for pred, _, data in G.in_edges(seed, data=True):
            rel = data.get("relation", "related_to")
            if _is_noise(pred) or _is_noise(seed):
                continue
            t_key = (pred, rel, seed)
            if t_key not in seen_triple_keys and len(retrieved_triples) < limit:
                seen_triple_keys.add(t_key)
                src_chunk = str(data.get("chunk_id", data.get("source", "")))
                if src_chunk:
                    source_chunk_ids.add(src_chunk)
                trip = create_provenance_triple(
                    subject=pred,
                    relation=rel,
                    object_=seed,
                    document_id=data.get("document_id", ""),
                    chunk_id=src_chunk,
                    source_url=data.get("source_url", data.get("source", "")),
                )
                retrieved_triples.append(trip)
                paths.append(f"{pred} -[{rel}]-> {seed}")

        # 2-Hop only if specified by query analysis and budget remains
        if hop_depth >= 2 and len(retrieved_triples) < limit:
            out_neighbors = [n for _, n, _ in G.out_edges(seed)]
            for hop1 in out_neighbors:
                if len(retrieved_triples) >= limit:
                    break
                for _, hop2, data in G.out_edges(hop1, data=True):
                    if hop2 == seed or _is_noise(hop1) or _is_noise(hop2):
                        continue
                    rel = data.get("relation", "related_to")
                    t_key = (hop1, rel, hop2)
                    if t_key not in seen_triple_keys and len(retrieved_triples) < limit:
                        seen_triple_keys.add(t_key)
                        src_chunk = str(data.get("chunk_id", data.get("source", "")))
                        if src_chunk:
                            source_chunk_ids.add(src_chunk)
                        trip = create_provenance_triple(
                            subject=hop1,
                            relation=rel,
                            object_=hop2,
                            document_id=data.get("document_id", ""),
                            chunk_id=src_chunk,
                            source_url=data.get("source_url", data.get("source", "")),
                        )
                        retrieved_triples.append(trip)
                        paths.append(f"{seed} -> {hop1} -[{rel}]-> {hop2}")

    # Serialize to natural language statements
    serialized = "\n".join(t.to_natural_language() for t in retrieved_triples)

    return ControlledKGResult(
        query=query,
        query_type=classification.query_type.value,
        hop_depth_used=hop_depth,
        entities_detected=valid_seed_nodes,
        triples=retrieved_triples,
        source_chunks=sorted(list(source_chunk_ids)),
        traversal_paths=paths,
        serialized_context=serialized,
    )
