from src.retriever.relation_aware import RelationAwareKG, fuse_relation_evidence


def test_relation_intent_detects_entity_relation_and_direction():
    kg = RelationAwareKG()
    intent = kg.analyze("Which organization developed SUIT?")
    assert "payload:aditya-suit" in intent.entity_ids
    assert intent.relation_types == ("DEVELOPED_BY",)
    assert intent.direction == "OUTGOING"
    assert intent.hop_depth == 1


def test_single_relation_retrieval_excludes_unrequested_neighbors():
    kg = RelationAwareKG()
    paths = kg.retrieve("Which organization developed SUIT?")
    assert paths
    assert all(path.relations == ("DEVELOPED_BY",) for path in paths)
    assert all(path.source_chunks for path in paths)
    assert all(path.source_urls for path in paths)


def test_two_hop_retrieval_is_relation_constrained_and_provenance_linked():
    kg = RelationAwareKG()
    paths = kg.retrieve(
        "Which organization developed a payload carried by Aditya-L1?"
    )
    assert paths
    assert any(path.relations == ("HAS_PAYLOAD", "DEVELOPED_BY") for path in paths)
    assert all(path.source_chunks for path in paths)
    assert all(path.source_urls for path in paths)


def test_fusion_is_deterministic_deduplicated_and_budgeted():
    kg = RelationAwareKG()
    paths = kg.retrieve("Which organization developed SUIT?")
    dense = [{"text": paths[0].source_text[0]}, {"text": "independent dense evidence"}]
    first = fuse_relation_evidence(paths, dense, token_budget=40)
    second = fuse_relation_evidence(paths, dense, token_budget=40)
    assert first == second
    assert first["total_tokens"] <= 40
    assert first["context_text"].count(paths[0].source_text[0]) <= 1
