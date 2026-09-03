from src.kg_builder.ner import extract_entities
from src.kg_builder.build_kg import build_graph_from_text


def test_extract_entities_detects_domain_entities():
    text = "Chandrayaan-3 was launched by ISRO from Sriharikota."

    entities = extract_entities(text)

    labels = {entity["label"] for entity in entities}
    texts = {entity["text"] for entity in entities}
    assert {"MISSION", "ORG", "PLACE"}.issubset(labels)
    assert "Chandrayaan-3" in texts
    assert "ISRO" in texts
    assert "Sriharikota" in texts


def test_build_graph_from_text_creates_nodes_and_edges():
    text = "Chandrayaan-3 was launched by ISRO from Sriharikota."

    graph = build_graph_from_text(text)

    assert graph.number_of_nodes() >= 3
    assert graph.number_of_edges() >= 1
    assert any(data.get("relation") == "launched_by" for _, _, data in graph.edges(data=True))
