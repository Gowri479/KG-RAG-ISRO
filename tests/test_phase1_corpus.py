import json

from src.corpus.build_kg_ready import build_kg_ready
from src.corpus.audit_relational_capacity import audit_relational_capacity
from src.corpus.coverage_analysis import analyze_coverage
from src.corpus.validate_corpus import validate_corpus


def test_kg_ready_facts_are_reproducibly_built_with_provenance():
    result = build_kg_ready()
    assert result["triple_count"] == result["annotation_count"]
    assert result["triple_count"] > 48
    report = validate_corpus()
    assert report["status"] == "passed"
    assert report["provenance_linked_triple_count"] == result["triple_count"]


def test_coverage_reports_distinguish_unflown_gaganyaan_and_direct_temporal_evidence():
    result = analyze_coverage()
    matrix = result["coverage"]["coverage_matrix"]
    assert matrix["Gaganyaan"]["Launch"] == "N/A (unflown)"
    assert matrix["Gaganyaan"]["Orbit"] == "✓"
    assert matrix["Chandrayaan-3"]["Temporal"] == "✓"
    assert matrix["Chandrayaan-1"]["Temporal"] == "—"
    assert result["statistics"]["all_triples_provenance_linked"] is True
    assert result["statistics"]["supported_relation_type_count"] == result["statistics"]["controlled_relation_type_count"]


def test_provenance_outputs_retain_exact_excerpt_and_chunk_text():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    triples = [
        json.loads(line)
        for line in (root / "data" / "corpus" / "triples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert triples
    for triple in triples:
        primary = triple["provenance"][0]
        assert primary["source_chunk_id"] == triple["source_chunk_id"]
        assert primary["source_text"] == triple["source_text"]
        assert primary["supporting_excerpt"]
        assert primary["source_url"].startswith("https://")


def test_relational_capacity_audit_counts_only_provenance_supported_paths():
    audit = audit_relational_capacity()
    assert audit["inventory"]["phase1_statistics_match"] == {
        "documents": True,
        "chunks": True,
        "entities": True,
        "triples": True,
        "provenance": True,
        "relational_coverage_missions": True,
    }
    assert audit["inventory"]["phase1_5_baseline"]["additional_documents_in_phase1_6"] == 0
    assert audit["inventory"]["phase1_5_baseline"]["additional_chunks_in_phase1_6"] == 0
    assert audit["inventory"]["phase1_5_baseline"]["additional_triples_in_phase1_6"] == 36
    relation_stats = {item["relation"]: item for item in audit["relation_statistics"]}
    assert relation_stats["DEVELOPED_BY"]["triple_count"] == 17
    assert relation_stats["OBSERVES"]["triple_count"] == 9
    assert audit["two_hop"]["directed_path_instance_count"] == 69
    patterns = {item["path_pattern"]: item for item in audit["two_hop"]["patterns"]}
    assert patterns[
        "Mission -> HAS_PAYLOAD -> Payload -> DEVELOPED_BY -> Organization"
    ]["distinct_path_count"] + patterns[
        "Mission -> HAS_PAYLOAD -> Payload -> DEVELOPED_BY -> ISROCentre"
    ]["distinct_path_count"] == 17
    assert patterns[
        "Mission -> HAS_PAYLOAD -> Payload -> HAS_OBJECTIVE -> ScientificObjective"
    ]["distinct_path_count"] == 21
    assert all(
        path["all_edges_provenance_supported"]
        for pattern in audit["two_hop"]["patterns"]
        for path in pattern["instances"]
    )
    assert audit["three_hop"]["directed_path_instance_count"] == 8
    assert all(
        path["all_edges_provenance_supported"]
        for pattern in audit["three_hop"]["patterns"]
        for path in pattern["instances"]
    )
    assert audit["decision"]["recommendation"] == "TARGETED COLLECTION REQUIRED"
