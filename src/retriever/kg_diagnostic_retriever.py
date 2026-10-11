"""Isolated, provenance-preserving diagnostic retrieval over the Phase 1 corpus KG."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from src.kg_builder.relations import normalize_entity

ROOT = Path(__file__).resolve().parents[2]
TRIPLES_PATH = ROOT / "data" / "corpus" / "triples.jsonl"
CHUNKS_PATH = ROOT / "data" / "corpus" / "chunks.jsonl"
ENTITIES_PATH = ROOT / "data" / "corpus" / "entities.jsonl"


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


@dataclass(frozen=True)
class DiagnosticPath:
    path_id: str
    nodes: list[str]
    relations: list[str]
    triple_ids: list[str]
    source_documents: list[str]
    source_chunks: list[str]
    retrieval_score: float


class CanonicalKG:
    def __init__(self) -> None:
        self.triples = [json.loads(line) for line in TRIPLES_PATH.read_text(encoding="utf8").splitlines() if line.strip()]
        self.entities = [json.loads(line) for line in ENTITIES_PATH.read_text(encoding="utf8").splitlines() if line.strip()]
        self.chunks = {
            item["chunk_id"]: item
            for item in (json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf8").splitlines() if line.strip())
        }
        self.aliases: dict[str, str] = {}
        for entity in self.entities:
            canonical = entity["name"]
            for value in [canonical, *entity.get("aliases", [])]:
                if value:
                    self.aliases[_norm(value)] = canonical
        self.by_subject: dict[str, list[dict]] = {}
        for triple in self.triples:
            self.by_subject.setdefault(triple["subject_id"], []).append(triple)

    def detect_entities(self, question: str) -> list[dict]:
        text = _norm(question)
        matches = []
        for entity in self.entities:
            candidates = [entity["name"], *entity.get("aliases", [])]
            if any(_norm(candidate) and _norm(candidate) in text for candidate in candidates):
                matches.append(entity)
        return sorted(matches, key=lambda item: len(_norm(item["name"])), reverse=True)

    def detect_relations(self, question: str) -> list[str]:
        q = question.lower()
        relation_terms = {
            "HAS_PAYLOAD": ("payload", "carry", "carried"),
            "DEVELOPED_BY": ("developed", "developer", "develop"),
            "HAS_OBJECTIVE": ("objective", "purpose"),
            "OBSERVES": ("observe", "observes", "observed"),
            "PRECEDED_BY": ("preceded", "before", "preceding"),
        }
        return [relation for relation, terms in relation_terms.items() if any(term in q for term in terms)]

    def paths(self, question: str, max_length: int = 3) -> list[DiagnosticPath]:
        entities = self.detect_entities(question)
        relation_filter = set(self.detect_relations(question))
        seeds = {entity["entity_id"] for entity in entities}
        candidates: list[DiagnosticPath] = []
        def extend(path: list[dict]) -> None:
            candidates.append(self._path(path, question))
            if len(path) >= max_length:
                return
            for next_triple in self.by_subject.get(path[-1]["object_id"], []):
                relations = {item["relation"] for item in path}
                if relation_filter and next_triple["relation"] not in relation_filter and not relations.intersection(relation_filter):
                    continue
                if next_triple["object_id"] in {item["subject_id"] for item in path}:
                    continue
                extend(path + [next_triple])

        for seed in seeds:
            for first in self.by_subject.get(seed, []):
                extend([first])
        unique = {}
        for path in candidates:
            unique[path.path_id] = path
        return sorted(unique.values(), key=lambda path: (-path.retrieval_score, path.path_id))

    def _path(self, triples: list[dict], question: str) -> DiagnosticPath:
        words = set(_norm(question).split())
        score = sum(len(words & set(_norm(t["subject"] + " " + t["object"] + " " + t["relation"]).split())) for t in triples)
        return DiagnosticPath(
            path_id="path:" + "+".join(t["triple_id"] for t in triples),
            nodes=[triples[0]["subject"], *[t["object"] for t in triples]],
            relations=[t["relation"] for t in triples],
            triple_ids=[t["triple_id"] for t in triples],
            source_documents=list(dict.fromkeys(t["source_document"] for t in triples if t["source_document"])),
            source_chunks=list(dict.fromkeys(t["source_chunk_id"] for t in triples if t["source_chunk_id"])),
            retrieval_score=float(score),
        )

    def source_evidence(self, path: DiagnosticPath) -> list[dict]:
        return [self.chunks[chunk_id] for chunk_id in path.source_chunks if chunk_id in self.chunks]


def format_structured_evidence(paths: list[DiagnosticPath], kg: CanonicalKG, include_source: bool = True) -> str:
    sections = ["RELATIONAL EVIDENCE"]
    for path in paths:
        sections.append(f"PATH {path.path_id}")
        for index, relation in enumerate(path.relations):
            sections.append(f"Entity: {path.nodes[index]}")
            sections.append(f"Relation: {relation}")
            sections.append(f"Value: {path.nodes[index + 1]}")
        if include_source:
            sections.append("SOURCE: " + ", ".join(path.source_documents or ["NOT_AVAILABLE"]))
            sections.append("SOURCE CHUNKS: " + ", ".join(path.source_chunks or ["NOT_AVAILABLE"]))
            excerpts = [item.get("text", "") for item in kg.source_evidence(path)]
            if excerpts:
                sections.append("SOURCE EVIDENCE:\n" + "\n".join(excerpts))
    return "\n".join(sections)


def diagnostic_object(question: str, expected: bool, dense_chunks: list[str], kg_only: bool = False) -> dict:
    kg = CanonicalKG()
    entities = kg.detect_entities(question)
    paths = kg.paths(question, max_length=2)
    selected = paths[:10]
    kg_path_evidence = format_structured_evidence(selected, kg, include_source=False)
    kg_evidence = format_structured_evidence(selected, kg, include_source=True)
    return {
        "question": question,
        "detected_entities": [entity["name"] for entity in entities],
        "detected_relations": kg.detect_relations(question),
        "expected_kg_requirement": expected,
        "kg_query": {"entity_ids": [entity["entity_id"] for entity in entities], "max_hops": 3},
        "candidate_triples": [triple for triple in kg.triples if triple["subject_id"] in {e["entity_id"] for e in entities}],
        "retrieved_paths": [asdict(path) for path in selected],
        "required_relation_found": "unknown",
        "required_path_found": "unknown",
        "source_provenance_available": any(path.source_chunks for path in selected),
        "dense_chunks": dense_chunks,
        "kg_evidence": kg_evidence,
        "kg_path_evidence": kg_path_evidence,
        "fused_evidence": kg_evidence if kg_only else "",
        "context_token_estimate": len(kg_evidence.split()),
        "generation_input": "",
        "answer": "",
    }
