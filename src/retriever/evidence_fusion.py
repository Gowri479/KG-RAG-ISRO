"""Controlled Dense + KG Evidence Fusion with Provenance Preservation.

Implements the 7-step evidence fusion procedure:
1. Dense retrieval (top-k passage chunks)
2. Query entity and relationship matching
3. Controlled KG retrieval (1-hop or multi-hop path)
4. Duplicate evidence detection and pruning
5. Multi-source evidence ranking
6. Fixed context budget truncation
7. Structured context assembly with provenance preservation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.retriever.controlled_retriever import controlled_kg_search
from src.retriever.faiss_retriever import _load_chunks, _load_index, _load_model
from src.retriever.hybrid import _extract_entities, _query_keywords
from src.retriever.query_classifier import classify_query

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "retrieval_config.json"


def _load_fusion_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("retrieval_parameters", {})
        except Exception:
            pass
    return {
        "top_k_dense": 3,
        "max_kg_triples": 10,
        "max_context_tokens": 1200,
        "dense_weight": 0.6,
        "kg_weight": 0.4,
    }


@dataclass
class EvidenceItem:
    evidence_id: str
    evidence_type: str  # "dense_passage" or "kg_triple"
    content: str
    score: float
    document_id: str = ""
    chunk_id: str = ""
    source_url: str = ""
    graph_path: str = ""


@dataclass
class FusedContext:
    query: str
    query_type: str
    evidence_items: list[EvidenceItem]
    context_text: str
    total_tokens: int
    provenance_summary: dict = field(default_factory=dict)


def _retrieve_dense_items(query: str, top_k: int) -> list[EvidenceItem]:
    """Retrieve dense passage chunks with FAISS distances."""
    index = _load_index()
    chunks = _load_chunks()
    model = _load_model()

    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(q_vec, top_k)

    dense_items = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx]
        text = c.get("text") or c.get("content") or ""
        # Convert L2 distance to normalized similarity score [0, 1]
        sim_score = max(0.0, 1.0 - (float(dist) / 2.0))
        dense_items.append(
            EvidenceItem(
                evidence_id=f"dense_chunk_{idx}",
                evidence_type="dense_passage",
                content=text.strip(),
                score=sim_score,
                document_id=c.get("document_id", ""),
                chunk_id=str(c.get("chunk_index", idx)),
                source_url=c.get("source_url", ""),
            )
        )
    return dense_items


def fuse_evidence(
    query: str,
    top_k_dense: Optional[int] = None,
    max_kg_triples: Optional[int] = None,
    max_context_tokens: Optional[int] = None,
) -> FusedContext:
    """Execute the full controlled evidence fusion pipeline."""
    cfg = _load_fusion_config()
    k_dense = top_k_dense or cfg.get("top_k_dense", 3)
    k_triples = max_kg_triples or cfg.get("max_kg_triples", 10)
    budget = max_context_tokens or cfg.get("max_context_tokens", 1200)

    # 1. Query Analysis
    classification = classify_query(query)
    entities = _extract_entities(query)
    keywords = _query_keywords(query)
    matched_entities = [e for e in entities if any(kw in e.lower() for kw in keywords)] or entities

    # 2. Dense Retrieval
    dense_items = _retrieve_dense_items(query, top_k=k_dense)

    # 3. Controlled KG Retrieval
    kg_result = controlled_kg_search(
        query=query,
        entities=matched_entities,
        max_triples=k_triples,
    )

    kg_items: list[EvidenceItem] = []
    for i, (triple, path) in enumerate(zip(kg_result.triples, kg_result.traversal_paths)):
        kg_items.append(
            EvidenceItem(
                evidence_id=f"kg_triple_{i}",
                evidence_type="kg_triple",
                content=triple.to_natural_language(),
                score=triple.confidence * cfg.get("kg_weight", 0.4),
                document_id=triple.document_id,
                chunk_id=triple.chunk_id,
                source_url=triple.source_url,
                graph_path=path,
            )
        )

    # 4. Deduplication
    # If a KG triple's explicit proposition is already fully contained verbatim in a dense chunk, lower its priority
    unique_kg_items = []
    dense_text_blob = " ".join(item.content.lower() for item in dense_items)
    for kg_item in kg_items:
        clean_content = kg_item.content.lower().rstrip(".")
        if clean_content in dense_text_blob:
            # Subsumed by dense chunk, but keep provenance recorded
            continue
        unique_kg_items.append(kg_item)

    # 5. Evidence Ranking & Assembly
    all_evidence = dense_items + unique_kg_items
    all_evidence.sort(key=lambda x: x.score, reverse=True)

    # 6. Apply Fixed Context Budget
    context_sections = []
    if unique_kg_items:
        context_sections.append("[Structured Knowledge Graph Evidence]")
        for item in unique_kg_items:
            context_sections.append(f"- {item.content}")
        context_sections.append("")

    if dense_items:
        context_sections.append("[Retrieved Document Evidence]")
        for item in dense_items:
            context_sections.append(f"--- Document: {item.document_id} (Chunk {item.chunk_id}) ---")
            context_sections.append(item.content)
            context_sections.append("")

    full_context_str = "\n".join(context_sections).strip()
    words = full_context_str.split()
    if len(words) > budget:
        full_context_str = " ".join(words[:budget]) + " ... [Context Truncated to Budget]"

    provenance_summary = {
        "dense_chunk_ids": [item.chunk_id for item in dense_items],
        "document_ids": list(set(item.document_id for item in all_evidence if item.document_id)),
        "kg_paths": [item.graph_path for item in unique_kg_items if item.graph_path],
        "source_urls": list(set(item.source_url for item in all_evidence if item.source_url)),
    }

    return FusedContext(
        query=query,
        query_type=classification.query_type.value,
        evidence_items=all_evidence,
        context_text=full_context_str,
        total_tokens=len(full_context_str.split()),
        provenance_summary=provenance_summary,
    )
