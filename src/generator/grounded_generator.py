"""Grounded generation module with evidence-based explainability and provenance preservation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.generator.ollama_api import generate_with_metrics
from src.generator.prompt import build_user_prompt
from src.retriever.evidence_fusion import FusedContext, fuse_evidence

GROUNDED_SYSTEM_PROMPT = (
    "You are a precise, evidence-grounded domain question-answering assistant for space agency documentation.\n"
    "STRICT INSTRUCTIONS:\n"
    "1. Answer ONLY using the facts explicitly stated in the supplied context.\n"
    "2. Do NOT use outside knowledge, unverified assumptions, or extrapolated conjectures.\n"
    "3. Do NOT invent or assume relationships not explicitly stated.\n"
    "4. Preserve exact dates, payload designations, numerical values, and mission names exactly as written.\n"
    "5. If the context does not contain sufficient factual evidence to answer the question, answer EXACTLY: I don't know.\n"
    "6. Keep your answer factual, direct, and concise without conversational filler or internal thought disclosure."
)


@dataclass
class GroundedAnswerRecord:
    question: str
    query_type: str
    generated_answer: str
    is_abstention: bool
    dense_chunk_ids: list[str]
    kg_triple_ids: list[str]
    source_documents: list[str]
    source_urls: list[str]
    graph_traversal_paths: list[str]
    context_tokens: int
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)
    generation_telemetry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def answer_grounded(question: str) -> GroundedAnswerRecord:
    """End-to-end grounded generation with evidence fusion and full provenance tracking."""
    # 1. Evidence Fusion
    fused: FusedContext = fuse_evidence(question)

    # 2. Generation
    answer_text, telemetry = generate_with_metrics(
        query=question,
        context=fused.context_text,
    )

    is_idk = (answer_text.strip().lower() == "i don't know." or "i don't know" in answer_text.strip().lower())

    dense_ids = [item.chunk_id for item in fused.evidence_items if item.evidence_type == "dense_passage"]
    kg_ids = [item.evidence_id for item in fused.evidence_items if item.evidence_type == "kg_triple"]
    doc_ids = list(set(item.document_id for item in fused.evidence_items if item.document_id))
    urls = list(set(item.source_url for item in fused.evidence_items if item.source_url))
    paths = [item.graph_path for item in fused.evidence_items if item.graph_path]

    return GroundedAnswerRecord(
        question=question,
        query_type=fused.query_type,
        generated_answer=answer_text,
        is_abstention=is_idk,
        dense_chunk_ids=dense_ids,
        kg_triple_ids=kg_ids,
        source_documents=doc_ids,
        source_urls=urls,
        graph_traversal_paths=paths,
        context_tokens=fused.total_tokens,
        retrieval_metadata=fused.provenance_summary,
        generation_telemetry=telemetry,
    )
