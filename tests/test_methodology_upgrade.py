"""Unit tests for Q1 methodology upgrade components:
- Controlled relations vocabulary and entity normalization
- Deterministic query classification
- Controlled graph retrieval
- Dense + KG evidence fusion and context budget
"""

import json
from pathlib import Path
import pytest

from src.kg_builder.relations import (
    ProvenanceTriple,
    create_provenance_triple,
    map_relation,
    normalize_entity,
)
from src.retriever.controlled_retriever import controlled_kg_search
from src.retriever.evidence_fusion import _load_fusion_config, fuse_evidence
from src.retriever.query_classifier import QueryType, classify_query

ROOT = Path(__file__).resolve().parents[1]


def test_entity_normalization():
    assert normalize_entity("aditya l1") == "Aditya-L1"
    assert normalize_entity("the aditya-l1 mission") == "Aditya-L1"
    assert normalize_entity("chandrayaan 2") == "Chandrayaan-2"
    assert normalize_entity("pslv c37") == "PSLV-C37"
    assert normalize_entity("velc") == "VELC"
    assert normalize_entity("isro") == "ISRO"


def test_controlled_relation_vocabulary():
    assert map_relation("carry") == "HAS_PAYLOAD"
    assert map_relation("launched_by") == "LAUNCHED_BY"
    assert map_relation("develop") == "DEVELOPED_BY"
    assert map_relation("orbit") == "ORBITS_AT"
    assert map_relation("observe") == "OBSERVES"
    assert map_relation("study") == "OBSERVES"
    assert map_relation("measure") == "MEASURES"


def test_provenance_triple_creation():
    trip = create_provenance_triple(
        subject="aditya l1",
        relation="carry",
        object_="velc",
        document_id="DOC_001",
        chunk_id="CHUNK_42",
        source_url="https://isro.gov.in/aditya",
    )
    assert trip.subject == "Aditya-L1"
    assert trip.relation == "HAS_PAYLOAD"
    assert trip.object == "VELC"
    assert trip.document_id == "DOC_001"
    assert trip.chunk_id == "CHUNK_42"
    assert "Aditya-L1 has payload VELC." == trip.to_natural_language()


def test_deterministic_query_classifier():
    q_temp = classify_query("When was Chandrayaan-3 launched?")
    assert q_temp.query_type == QueryType.TEMPORAL
    assert q_temp.suggested_hop_depth == 1

    q_rel = classify_query("Which launch vehicle carried Aditya-L1?")
    assert q_rel.query_type == QueryType.RELATIONAL
    assert q_rel.suggested_hop_depth == 1

    q_multi = classify_query("Which center developed the payload onboard Aditya-L1 that observes the solar corona?")
    assert q_multi.query_type == QueryType.MULTI_HOP
    assert q_multi.suggested_hop_depth == 2

    q_fact = classify_query("What is the full form of ISRO?")
    assert q_fact.query_type == QueryType.FACTUAL
    assert q_fact.suggested_hop_depth == 1


def test_retrieval_config_loading():
    cfg = _load_fusion_config()
    assert "top_k_dense" in cfg
    assert "max_kg_triples" in cfg
    assert "max_context_tokens" in cfg
    assert cfg["top_k_dense"] == 3
    assert cfg["max_context_tokens"] == 1200


def test_controlled_kg_search_structure():
    res = controlled_kg_search(
        query="What payloads are on Aditya-L1?",
        entities=["Aditya-L1"],
        max_triples=5,
    )
    assert res.query_type in [t.value for t in QueryType]
    assert res.hop_depth_used in [1, 2]
    assert len(res.triples) <= 5
