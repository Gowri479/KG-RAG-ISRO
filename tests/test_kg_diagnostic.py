from src.retriever.kg_diagnostic_retriever import CanonicalKG, format_structured_evidence
import hashlib
import json
from pathlib import Path


def test_canonical_two_hop_path_and_provenance():
    kg = CanonicalKG()
    paths = kg.paths("Which payload carried by Aditya-L1 was developed by the Indian Institute of Astrophysics?")
    ids = {triple_id for path in paths for triple_id in path.triple_ids}
    assert "aditya-payload-velc" in ids
    assert "aditya-velc-developed-by-iia" in ids
    path = next(path for path in paths if "aditya-velc-developed-by-iia" in path.triple_ids)
    assert path.source_chunks
    assert "RELATIONAL EVIDENCE" in format_structured_evidence([path], kg)


def test_diagnostic_paths_are_deterministic():
    kg = CanonicalKG()
    first = [path.path_id for path in kg.paths("Which payload of Chandrayaan-3 observes lunar soil?")]
    second = [path.path_id for path in kg.paths("Which payload of Chandrayaan-3 observes lunar soil?")]
    assert first == second


def test_diagnostic_artifacts_are_complete_and_hashed():
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "data/results/kg_diagnostic_26.json").read_text(encoding="utf8"))
    manifest = json.loads((root / "data/results/kg_diagnostic_manifest.json").read_text(encoding="utf8"))
    assert result["decision"] == "A. TECHNICAL_FAILURE_IDENTIFIED"
    assert result["question_count"] == 26
    assert result["evaluation_count"] == 130
    assert manifest["benchmark_sha256"] == "7f5daf5419b0a4cb70e5c3d5eba021a8a39118b5a0fd235ce19b81f75324ffce"
    for key, filename in (
        ("diagnostic_json_sha256", "kg_diagnostic_26.json"),
        ("diagnostic_report_sha256", "kg_diagnostic_report.md"),
    ):
        assert manifest[key] == hashlib.sha256((root / "data/results" / filename).read_bytes()).hexdigest()
