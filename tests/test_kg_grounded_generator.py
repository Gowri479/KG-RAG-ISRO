from src.generator.kg_grounded_generator import (
    KGFact,
    build_structured_context,
    classify_conflict,
    serialize_kg_facts,
)


def _fact(target="VSSC"):
    return KGFact("path:1", "SUIT", "DEVELOPED_BY", target, ("doc",), ("chunk",), ("url",))


def test_structured_serialization_preserves_direction_and_provenance():
    text = serialize_kg_facts([_fact()])
    assert "SUIT -> DEVELOPED_BY -> VSSC" in text
    assert "Provenance: chunk" in text


def test_context_budget_is_deterministic():
    context, stats = build_structured_context([_fact()], "dense " * 100, budget=20)
    assert stats["total_token_count"] <= 20
    assert "KG FACTS:" in context


def test_conflict_classification():
    assert classify_conflict([_fact()], "SUIT was developed by VSSC.") == "NO_CONFLICT"
    assert classify_conflict([_fact()], "SUIT was developed by another organization.") == "DENSE_MORE_SPECIFIC"
