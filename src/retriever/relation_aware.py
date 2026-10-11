"""Deterministic relation-aware retrieval over the canonical corpus KG.

This module is isolated from the frozen evaluators. It narrows graph retrieval
by relation intent before ranking provenance-linked paths and source evidence.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
TRIPLES = ROOT / "data" / "corpus" / "triples.jsonl"
ENTITIES = ROOT / "data" / "corpus" / "entities.jsonl"
CHUNKS = ROOT / "data" / "corpus" / "chunks.jsonl"

RELATION_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("DEVELOPED_BY", ("developed by", "developer", "developed", "designed by", "built by")),
    ("HAS_PAYLOAD", ("payload", "payloads", "carries", "carried", "carry", "onboard")),
    ("HAS_OBJECTIVE", ("objective", "objectives", "purpose", "goal")),
    ("OBSERVES", ("observes", "observe", "observed", "measures", "measure")),
    ("LAUNCHED_ON", ("launch date", "launched on", "what date", "when was")),
    ("LAUNCHED_BY", ("launched by", "launch vehicle", "launcher")),
    ("LAUNCHED_FROM", ("launched from", "launch site", "launch location")),
    ("PRECEDED_BY", ("preceded", "before", "earlier mission", "followed")),
    ("OPERATED_BY", ("operated by", "operates")),
    ("OPERATES_AT", ("operates at", "located at")),
    ("STUDIES", ("studies", "study", "scientific target", "target")),
    ("ORBITS", ("orbits", "orbit", "orbital location")),
    ("LED_BY", ("led by", "lead organization")),
)


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


@dataclass(frozen=True)
class RelationIntent:
    query: str
    entity_ids: tuple[str, ...]
    relation_types: tuple[str, ...]
    direction: str
    query_type: str
    requires_explicit_path: bool
    hop_depth: int


@dataclass(frozen=True)
class RelationEvidence:
    path_id: str
    nodes: tuple[str, ...]
    relations: tuple[str, ...]
    triple_ids: tuple[str, ...]
    source_documents: tuple[str, ...]
    source_chunks: tuple[str, ...]
    source_urls: tuple[str, ...]
    source_sections: tuple[str, ...]
    source_pages: tuple[str, ...]
    source_text: tuple[str, ...]
    score: float
    score_components: dict[str, float]


def analyze_relation_query(query: str, entities: Iterable[dict]) -> RelationIntent:
    """Classify relation intent using fixed lexical rules and entity aliases."""
    text = _norm(query)
    matches = []
    for entity in entities:
        name = entity.get("name", "")
        parenthetical = re.findall(r"\(([A-Z][A-Z0-9-]{1,})\)", name)
        aliases = [name, *entity.get("aliases", []), *parenthetical]
        if any(_norm(alias) and _norm(alias) in text for alias in aliases):
            matches.append(entity["entity_id"])
    relation_types = [
        relation
        for relation, terms in RELATION_PATTERNS
        if any(_norm(term) in text for term in terms)
    ]
    relation_types = list(dict.fromkeys(relation_types))
    multi = len(relation_types) > 1 or any(
        marker in text for marker in ("that developed", "payload carried by", "payload developed by")
    )
    if not relation_types:
        query_type = "FACTUAL"
    elif multi:
        query_type = "MULTI_HOP"
    else:
        query_type = "RELATIONAL"
    direction = "OUTGOING"
    if "payload developed by" in text or ("payload carried by" in text and "developed" in text):
        relation_types = ["HAS_PAYLOAD", "DEVELOPED_BY"]
    return RelationIntent(
        query=query,
        entity_ids=tuple(matches),
        relation_types=tuple(relation_types),
        direction=direction,
        query_type=query_type,
        requires_explicit_path=bool(relation_types),
        hop_depth=2 if multi else 1,
    )


class RelationAwareKG:
    """Canonical KG index with relation-constrained path retrieval."""

    def __init__(self) -> None:
        import json

        self.triples = [
            json.loads(line) for line in TRIPLES.read_text(encoding="utf8").splitlines() if line.strip()
        ]
        self.entities = [
            json.loads(line) for line in ENTITIES.read_text(encoding="utf8").splitlines() if line.strip()
        ]
        self.chunks = {
            item["chunk_id"]: item
            for item in (
                json.loads(line) for line in CHUNKS.read_text(encoding="utf8").splitlines() if line.strip()
            )
        }
        self.by_subject: dict[str, list[dict]] = {}
        self.by_object: dict[str, list[dict]] = {}
        for triple in self.triples:
            self.by_subject.setdefault(triple["subject_id"], []).append(triple)
            self.by_object.setdefault(triple["object_id"], []).append(triple)

    def analyze(self, query: str) -> RelationIntent:
        return analyze_relation_query(query, self.entities)

    def retrieve(
        self,
        query: str,
        dense_scores: dict[str, float] | None = None,
        lexical_scores: dict[str, float] | None = None,
        max_paths: int = 50,
    ) -> list[RelationEvidence]:
        intent = self.analyze(query)
        candidates: list[list[dict]] = []
        relations = set(intent.relation_types)
        for entity_id in intent.entity_ids:
            first_relation = intent.relation_types[0] if intent.relation_types else None
            first_edges = self.by_object.get(entity_id, []) if intent.direction == "INCOMING" else self.by_subject.get(entity_id, [])
            first_edges = [edge for edge in first_edges if not first_relation or edge["relation"] == first_relation]
            candidates.extend([[edge] for edge in first_edges])
            if intent.hop_depth >= 2:
                for edge in first_edges:
                    for next_edge in self.by_subject.get(edge["object_id"], []):
                        if len(intent.relation_types) < 2 or next_edge["relation"] == intent.relation_types[1]:
                            candidates.append([edge, next_edge])
        unique: dict[str, RelationEvidence] = {}
        for path in candidates:
            evidence = self._evidence(path, query, dense_scores or {}, lexical_scores or {}, intent)
            unique[evidence.path_id] = evidence
        return sorted(unique.values(), key=lambda item: (-item.score, item.path_id))[:max_paths]

    def _evidence(
        self,
        path: list[dict],
        query: str,
        dense_scores: dict[str, float],
        lexical_scores: dict[str, float],
        intent: RelationIntent,
    ) -> RelationEvidence:
        query_words = set(_norm(query).split())
        entity_score = sum(
            1.0 for edge in path if query_words.intersection(set(_norm(edge["subject"] + " " + edge["object"]).split()))
        ) / len(path)
        relation_score = sum(edge["relation"] in intent.relation_types for edge in path) / len(path) if path else 0.0
        chunks = [edge.get("source_chunk_id", "") for edge in path if edge.get("source_chunk_id")]
        provenance_score = sum(bool(edge.get("provenance") or edge.get("source_chunk_id")) for edge in path) / len(path)
        dense_score = max((dense_scores.get(chunk, 0.0) for chunk in chunks), default=0.0)
        lexical_score = max((lexical_scores.get(chunk, 0.0) for chunk in chunks), default=0.0)
        components = {
            "dense": dense_score,
            "lexical": lexical_score,
            "entity": entity_score,
            "relation": relation_score,
            "provenance": provenance_score,
        }
        score = (
            0.20 * dense_score
            + 0.15 * lexical_score
            + 0.20 * entity_score
            + 0.30 * relation_score
            + 0.15 * provenance_score
        )
        source_chunks = tuple(dict.fromkeys(chunks))
        source_items = [self.chunks[chunk] for chunk in source_chunks if chunk in self.chunks]
        return RelationEvidence(
            path_id="path:" + "+".join(edge["triple_id"] for edge in path),
            nodes=tuple([path[0]["subject"], *[edge["object"] for edge in path]]),
            relations=tuple(edge["relation"] for edge in path),
            triple_ids=tuple(edge["triple_id"] for edge in path),
            source_documents=tuple(dict.fromkeys(edge.get("source_document", "") for edge in path if edge.get("source_document"))),
            source_chunks=source_chunks,
            source_urls=tuple(dict.fromkeys(edge.get("source_url", "") for edge in path if edge.get("source_url"))),
            source_sections=tuple(dict.fromkeys(edge.get("source_section", "") for edge in path if edge.get("source_section"))),
            source_pages=tuple(dict.fromkeys(str(edge.get("source_page")) for edge in path if edge.get("source_page") is not None)),
            source_text=tuple(item.get("text", "") for item in source_items),
            score=round(score, 6),
            score_components=components,
        )


def fuse_relation_evidence(
    paths: list[RelationEvidence],
    dense_items: list[dict],
    bm25_items: list[dict] | None = None,
    token_budget: int = 1200,
) -> dict:
    """Assemble prioritized, deduplicated structured and source evidence."""
    sections: list[str] = ["[RELATION-AWARE KG EVIDENCE]"]
    seen: set[str] = set()
    for path in paths:
        sections.append(f"PATH {path.path_id} SCORE {path.score:.6f}")
        for subject, relation, object_ in zip(path.nodes[:-1], path.relations, path.nodes[1:]):
            sections.append(f"ENTITY: {subject}\nRELATION: {relation}\nVALUE: {object_}")
        sections.append("SOURCE DOCUMENTS: " + ", ".join(path.source_documents or ("NOT_AVAILABLE",)))
        sections.append("SOURCE CHUNKS: " + ", ".join(path.source_chunks or ("NOT_AVAILABLE",)))
        sections.append("SOURCE URLS: " + ", ".join(path.source_urls or ("NOT_AVAILABLE",)))
        for text in path.source_text:
            key = _norm(text)
            if key and key not in seen:
                seen.add(key)
                sections.append("SOURCE TEXT: " + text)
    for heading, items in (("[DENSE EVIDENCE]", dense_items), ("[BM25 EVIDENCE]", bm25_items or [])):
        unique = []
        for item in items:
            text = item if isinstance(item, str) else item.get("text", item.get("content", ""))
            key = _norm(text)
            if key and key not in seen:
                seen.add(key)
                unique.append(text)
        if unique:
            sections.append(heading)
            sections.extend(unique)
    words = "\n".join(sections).split()
    return {
        "context_text": " ".join(words[:token_budget]),
        "total_tokens": min(len(words), token_budget),
        "paths": [asdict(path) if hasattr(path, "__dataclass_fields__") else path for path in paths],
        "token_budget": token_budget,
    }
