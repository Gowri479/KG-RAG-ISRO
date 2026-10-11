"""Unit tests for Q1 journal evaluation components and data integrity."""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_q1_manifest_validity():
    manifest_path = ROOT / "data" / "results" / "q1_experiment_manifest.json"
    assert manifest_path.exists(), "Manifest file missing"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "experiments" in data
    assert len(data["experiments"]) >= 6
    assert data["scientific_constraints_verified"]["frozen_canonical_180_benchmark"] is True


def test_unanswerable_benchmark_integrity():
    unans_path = ROOT / "data" / "benchmark" / "unanswerable_qa.json"
    assert unans_path.exists(), "Unanswerable benchmark missing"
    data = json.loads(unans_path.read_text(encoding="utf-8"))
    assert len(data) >= 30
    for q in data:
        assert q["answerable"] is False
        assert "category" in q
        assert len(q["question"]) > 10


def test_human_eval_template_schema():
    template_path = ROOT / "data" / "annotations" / "human_eval_template.csv"
    assert template_path.exists(), "Human evaluation template missing"
    lines = template_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 50
    header = lines[0].split(",")
    assert "correctness" in header
    assert "faithfulness" in header
    assert "completeness" in header
    assert "relevance" in header


def test_retrieval_eval_template_schema():
    template_path = ROOT / "data" / "annotations" / "retrieval_eval_template.json"
    assert template_path.exists(), "Retrieval evaluation template missing"
    data = json.loads(template_path.read_text(encoding="utf-8"))
    assert len(data) == 100
    for item in data:
        assert "relevant_chunk_ids" in item
        assert "relevant_kg_entities" in item
