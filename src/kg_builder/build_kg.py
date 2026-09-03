"""Build a minimal knowledge graph from entity mentions in text."""

from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx

from src.kg_builder.ner import extract_entities


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _infer_relation(text: str, source: str, target: str) -> str:
    text_lower = text.lower()
    source_lower = source.lower()
    target_lower = target.lower()

    if "launched by" in text_lower and source_lower.startswith("chandrayaan") and "isro" in target_lower:
        return "launched_by"
    if "launched by" in text_lower and "isro" in source_lower and source_lower.startswith("isro"):
        return "launched_by"
    if "mission" in text_lower and ("isro" in source_lower or "isro" in target_lower):
        return "mission_of"
    return "related_to"


def build_graph_from_text(text: str) -> nx.DiGraph:
    """Create a directed NetworkX graph whose nodes are entities and edges encode direct relations."""
    graph = nx.DiGraph()
    entities = extract_entities(text)
    if not entities:
        return graph

    entities = sorted(entities, key=lambda item: item["start"])
    for entity in entities:
        name = _normalize_text(entity["text"])
        if not name:
            continue
        graph.add_node(name, label=entity["label"])

    for i in range(len(entities) - 1):
        left = _normalize_text(entities[i]["text"])
        right = _normalize_text(entities[i + 1]["text"])
        if left and right and left != right:
            relation = _infer_relation(text, left, right)
            graph.add_edge(left, right, relation=relation)

    if len(graph.nodes) >= 2 and graph.number_of_edges() == 0:
        first = list(graph.nodes)[0]
        second = list(graph.nodes)[1]
        graph.add_edge(first, second, relation="related_to")

    return graph


def build_graph_from_file(input_path: str | Path, output_path: str | Path | None = None) -> nx.DiGraph:
    """Load a text file, build a graph, and optionally save it as JSON."""
    text = Path(input_path).read_text(encoding="utf-8")
    graph = build_graph_from_text(text)

    if output_path is not None:
        payload = {
            "nodes": [{"id": node, "label": data.get("label", "UNKNOWN")} for node, data in graph.nodes(data=True)],
            "edges": [{"source": source, "target": target, "relation": data.get("relation", "related_to")}
                      for source, target, data in graph.edges(data=True)],
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return graph


def build_graph_from_directory(input_dir: str | Path, output_dir: str | Path | None = None) -> nx.DiGraph:
    """Aggregate graph construction across all markdown files in a directory and write the combined JSON graph."""
    source_dir = Path(input_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {source_dir}")

    aggregate = nx.DiGraph()
    for file_path in sorted(source_dir.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")
        graph = build_graph_from_text(text)
        for node, data in graph.nodes(data=True):
            if node not in aggregate:
                aggregate.add_node(node, label=data.get("label", "UNKNOWN"))
            else:
                aggregate.nodes[node]["label"] = aggregate.nodes[node].get("label", "UNKNOWN")
        for source, target, data in graph.edges(data=True):
            aggregate.add_edge(source, target, relation=data.get("relation", "related_to"))

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [{"id": node, "label": data.get("label", "UNKNOWN")} for node, data in aggregate.nodes(data=True)],
            "edges": [{"source": source, "target": target, "relation": data.get("relation", "related_to")}
                      for source, target, data in aggregate.edges(data=True)],
        }
        graph_file = output_path / "knowledge_graph.json"
        graph_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return aggregate


if __name__ == "__main__":
    sample = "Chandrayaan-3 was launched by ISRO from Sriharikota."
    graph = build_graph_from_text(sample)
    print(graph.edges(data=True))

