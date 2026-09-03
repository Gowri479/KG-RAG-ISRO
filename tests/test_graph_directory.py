from pathlib import Path

from src.kg_builder.build_kg import build_graph_from_directory


def test_build_graph_from_directory_creates_aggregate_graph(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "a.md").write_text("Chandrayaan-3 was launched by ISRO from Sriharikota.", encoding="utf-8")
    (source_dir / "b.md").write_text("Aditya-L1 is an Indian mission.", encoding="utf-8")

    output_dir = tmp_path / "kg_out"
    graph = build_graph_from_directory(source_dir, output_dir)

    assert graph.number_of_nodes() >= 4
    assert graph.number_of_edges() >= 2
    assert (output_dir / "knowledge_graph.json").exists()
