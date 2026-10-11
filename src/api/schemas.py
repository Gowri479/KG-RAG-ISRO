"""Request and response models for the KG-RAG HTTP API (the frontend contract)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["live", "mock"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, examples=["Which organization developed the Solar Ultraviolet Imaging Telescope (SUIT)?"])


class Source(BaseModel):
    """Official document a KG fact was curated from."""

    document_id: str
    url: str | None = None
    section: str | None = None
    page: str | None = None
    excerpt: str | None = Field(None, description="Supporting text from the official source.")


class Hop(BaseModel):
    """One fact in a KG path: subject -> relation -> object."""

    subject: str
    relation: str
    object: str
    triple_id: str
    source: Source


class KGPath(BaseModel):
    path_id: str
    score: float = Field(..., description="Retrieval score, 0-1; paths are sorted by it.")
    hops: list[Hop]


class Passage(BaseModel):
    """Text passage from dense retrieval."""

    rank: int
    text: str
    url: str | None = Field(None, description="Source page, when the passage records one.")
    title: str | None = None
    mission: str | None = None
    source_file: str | None = None


class QueryAnalysis(BaseModel):
    query_type: str = Field(..., description="FACTUAL, RELATIONAL or MULTI_HOP.")
    entities: list[str]
    relations: list[str]
    hop_depth: int


class Usage(BaseModel):
    latency_ms: int
    prompt_tokens: int | None = None
    context_tokens_used: int | None = None
    context_trimmed: bool | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    abstained: bool = Field(..., description="True when the system answered \"I don't know.\"")
    mode: Mode = Field(..., description="'mock' answers are canned; never present them as real.")
    system: str = "relation_aware_kg_rag"
    analysis: QueryAnalysis
    kg_paths: list[KGPath] = Field(..., description="Top KG paths shown to the user (max 10).")
    kg_paths_total: int = Field(..., description="All KG paths given to the model.")
    passages: list[Passage]
    usage: Usage


class Example(BaseModel):
    question: str
    mission: str
    category: str


class ExamplesResponse(BaseModel):
    examples: list[Example]


class HealthResponse(BaseModel):
    status: Literal["ready", "warming_up", "error"]
    mode: Mode
    model: str
    ollama_reachable: bool | None = None
    detail: str | None = None


class ErrorResponse(BaseModel):
    detail: str
