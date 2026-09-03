"""Retrieve one-hop knowledge-graph context for entity mentions."""

from __future__ import annotations

import json
from pathlib import Path

GRAPH_PATH = Path(__file__).resolve().parents[2] / "data" / "kg" / "knowledge_graph.json"


def _normalize_entity(value: str) -> str:
    return " ".join(str(value).strip().split()).lower()


def _relation_text(relation: str) -> str:
    if not relation:
        return "is related to"
    relation = str(relation).replace("_", " ").strip()
    if relation in {"related to", "related_to"}:
        return "is related to"
    if relation == "launched_by":
        return "was launched by"
    if relation == "mission_of":
        return "is part of the mission of"
    return relation


def _load_graph() -> dict:
    if not GRAPH_PATH.exists():
        return {"nodes": [], "edges": []}

    payload = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        nodes = payload.get("nodes", []) if isinstance(payload.get("nodes", []), list) else []
        edges = payload.get("edges", []) if isinstance(payload.get("edges", []), list) else []
        return {"nodes": nodes, "edges": edges}
    if isinstance(payload, list):
        return {"nodes": [], "edges": payload}
    return {"nodes": [], "edges": []}


def get_kg_context(entities: list[str]) -> str:
    """Return natural-language one-hop triples for the supplied entities."""
    if not entities:
        return ""

    graph = _load_graph()
    edges = graph.get("edges", [])
    if not edges:
        return ""

    entity_names = {str(entity).strip() for entity in entities if str(entity).strip()}
    if not entity_names:
        return ""

    normalized_lookup = { _normalize_entity(item.get("id", "")): item.get("id", "") for item in graph.get("nodes", []) if isinstance(item, dict) and item.get("id") }
    triples: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source", "")
        target = edge.get("target", "")
        relation = edge.get("relation", "related_to")
        if not source or not target:
            continue

        source_norm = _normalize_entity(source)
        target_norm = _normalize_entity(target)
        if source_norm in {_normalize_entity(e) for e in entity_names} or target_norm in {_normalize_entity(e) for e in entity_names}:
            match_source = source_norm in {_normalize_entity(e) for e in entity_names}
            match_target = target_norm in {_normalize_entity(e) for e in entity_names}
            if not (match_source or match_target):
                continue

            source_label = normalized_lookup.get(source_norm, source)
            target_label = normalized_lookup.get(target_norm, target)
            triple = f"{source_label} {_relation_text(relation)} {target_label}"
            key = (source_label.lower(), relation.lower(), target_label.lower())
            if key not in seen:
                seen.add(key)
                triples.append(triple)

    # Support reverse-direction lookup for entity lists in the target position too.
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source", "")
        target = edge.get("target", "")
        relation = edge.get("relation", "related_to")
        if not source or not target:
            continue

        source_norm = _normalize_entity(source)
        target_norm = _normalize_entity(target)
        if any(_normalize_entity(entity) == source_norm for entity in entity_names) or any(_normalize_entity(entity) == target_norm for entity in entity_names):
            source_label = normalized_lookup.get(source_norm, source)
            target_label = normalized_lookup.get(target_norm, target)
            triple = f"{source_label} {_relation_text(relation)} {target_label}"
            key = (source_label.lower(), relation.lower(), target_label.lower())
            if key not in seen:
                seen.add(key)
                triples.append(triple)

    return "\n".join(triples[:20]).strip()


if __name__ == "__main__":
    print(get_kg_context(["ISRO", "Chandrayaan-3"]))

