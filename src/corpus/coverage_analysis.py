"""Summarize the curated Phase 1 corpus and its evidence-backed relations."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus"
REPORTS = ROOT / "reports"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def analyze_coverage() -> dict[str, Any]:
    registry = json.loads((CORPUS / "document_registry.json").read_text(encoding="utf-8"))
    chunks = read_jsonl(CORPUS / "chunks.jsonl")
    entities = read_jsonl(CORPUS / "entities.jsonl")
    triples = read_jsonl(CORPUS / "triples.jsonl")
    entity_type_counts = Counter(entity["type"] for entity in entities)
    relation_counts = Counter(triple["relation"] for triple in triples)
    triples_by_mission: dict[str, Counter[str]] = defaultdict(Counter)
    for triple in triples:
        triples_by_mission[triple["mission"]][triple["relation"]] += 1

    documents_by_mission = Counter(document["mission"] for document in registry["documents"])
    documents_by_tier = Counter(document["authority_tier"] for document in registry["documents"])
    chunks_by_mission = Counter(chunk["mission"] for chunk in chunks)
    words_by_mission = Counter()
    for chunk in chunks:
        words_by_mission[chunk["mission"]] += chunk["word_count_estimate"]
    mission_names = sorted(
        {document["mission"] for document in registry["documents"] if document["mission"] != "cross-mission"}
    )

    matrix: dict[str, dict[str, str]] = {}
    mission_coverage: dict[str, dict[str, int]] = {}
    for mission in mission_names:
        relation_data = triples_by_mission[mission]
        payload_ids = {
            triple["object_id"]
            for triple in triples
            if triple["mission"] == mission and triple["relation"] == "HAS_PAYLOAD"
        }
        developer_payloads = {
            triple["subject_id"]
            for triple in triples
            if triple["mission"] == mission and triple["relation"] == "DEVELOPED_BY"
        }
        launch_relations = {
            relation
            for relation in ("LAUNCHED_ON", "LAUNCHED_BY", "LAUNCHED_FROM")
            if relation_data[relation]
        }
        science_count = sum(
            relation_data[relation] for relation in ("HAS_OBJECTIVE", "STUDIES", "OBSERVES")
        )
        orbit_count = relation_data["ORBITS"]
        launched = len(launch_relations) == 3
        launch_state = "✓" if launched else ("N/A (unflown)" if mission == "Gaganyaan" else "partial")
        matrix[mission] = {
            "Payload": "✓" if payload_ids else "—",
            "Developer organization": "✓" if developer_payloads else "—",
            "Launch": launch_state,
            "Orbit": "✓" if orbit_count else "—",
            "Temporal": "✓" if relation_data["PRECEDED_BY"] else "—",
            "Science/objective": "✓" if science_count else "—",
        }
        mission_coverage[mission] = {
            "payloads": len(payload_ids),
            "payloads_with_developer_attribution": len(payload_ids & developer_payloads),
            "developer_relations": relation_data["DEVELOPED_BY"],
            "launch_relations": sum(relation_data[item] for item in launch_relations),
            "launch_relation_kinds_present": len(launch_relations),
            "orbit_relations": orbit_count,
            "explicit_temporal_relations": relation_data["PRECEDED_BY"],
            "science_and_objective_relations": science_count,
            "triple_count": sum(relation_data.values()),
            "chunk_count": chunks_by_mission[mission],
            "document_count": documents_by_mission[mission],
            "word_count_estimate": words_by_mission[mission],
        }

    statistics = {
        "phase": 1,
        "document_count": len(registry["documents"]),
        "documents_by_mission": dict(sorted(documents_by_mission.items())),
        "documents_by_authority_tier": dict(sorted(documents_by_tier.items())),
        "chunks_count": len(chunks),
        "chunks_by_mission": dict(sorted(chunks_by_mission.items())),
        "word_count_estimate": sum(chunk["word_count_estimate"] for chunk in chunks),
        "mission_count": len(mission_names),
        "entity_count": len(entities),
        "entities_by_type": dict(sorted(entity_type_counts.items())),
        "payload_count": entity_type_counts["Payload"],
        "organization_count": entity_type_counts["Organization"] + entity_type_counts["ISROCentre"],
        "launch_vehicle_count": entity_type_counts["LaunchVehicle"],
        "orbit_count": entity_type_counts["Orbit"],
        "controlled_relation_type_count": len(json.loads((CORPUS / "relations.json").read_text(encoding="utf-8"))["relations"]),
        "supported_relation_type_count": len(relation_counts),
        "triple_count": len(triples),
        "triples_by_relation": dict(sorted(relation_counts.items())),
        "triples_by_mission": {
            mission: sum(relations.values())
            for mission, relations in sorted(triples_by_mission.items())
        },
        "provenance_linked_triple_count": sum(bool(triple.get("provenance")) for triple in triples),
        "all_triples_provenance_linked": all(bool(triple.get("provenance")) for triple in triples),
    }
    coverage = {
        "coverage_matrix": matrix,
        "mission_coverage": mission_coverage,
        "status_legend": {
            "✓": "At least one direct, source-linked fact is curated for this category.",
            "partial": "Some but not all expected launch fields are represented.",
            "N/A (unflown)": "No actual launch relationship is asserted for the planned Gaganyaan crewed mission.",
            "—": "No direct fact is currently curated; this is not evidence that the fact is false.",
        },
    }
    (CORPUS / "corpus_statistics.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (CORPUS / "relational_coverage.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_reports(statistics, coverage)
    return {"statistics": statistics, "coverage": coverage}


def write_reports(statistics: dict[str, Any], coverage: dict[str, Any]) -> None:
    mission_rows = [
        f"| {mission} | {values['document_count']} | {values['chunk_count']} | "
        f"{values['payloads']} | {values['payloads_with_developer_attribution']} | "
        f"{values['orbit_relations']} | {values['triple_count']} |"
        for mission, values in coverage["mission_coverage"].items()
    ]
    relation_rows = [
        f"| {relation} | {count} |"
        for relation, count in statistics["triples_by_relation"].items()
    ]
    tier_rows = [
        f"| {tier} | {count} |"
        for tier, count in statistics["documents_by_authority_tier"].items()
    ]
    matrix_columns = ["Payload", "Developer organization", "Launch", "Orbit", "Temporal", "Science/objective"]
    matrix_rows = [
        "| " + mission + " | " + " | ".join(values[column] for column in matrix_columns) + " |"
        for mission, values in coverage["coverage_matrix"].items()
    ]
    coverage_report = f"""# Phase 1 corpus coverage

## Corpus inventory

- Registered documents: {statistics['document_count']}
- Semantic chunks: {statistics['chunks_count']}
- Estimated words: {statistics['word_count_estimate']:,}
- Missions: {statistics['mission_count']}
- Typed entities: {statistics['entity_count']}
- Payload entities: {statistics['payload_count']}
- Organization/ISRO-centre entities: {statistics['organization_count']}
- Launch-vehicle entities: {statistics['launch_vehicle_count']}
- Orbit entities: {statistics['orbit_count']}
- Curated triples: {statistics['triple_count']}
- Controlled relation types represented: {statistics['supported_relation_type_count']} of {statistics['controlled_relation_type_count']}
- Provenance-linked triples: {statistics['provenance_linked_triple_count']} of {statistics['triple_count']}

All registered documents are Tier 1 official ISRO or ISSDC sources. Word counts are whitespace-token estimates, not model token counts. Chunking and annotation were completed without use of benchmark questions, model answers, or downstream QA scores.

| Authority tier | Documents |
|---|---:|
{chr(10).join(tier_rows)}

## Mission coverage

| Mission | Documents | Chunks | Payload entities | Payloads with explicit developer | Orbit relations | Triples |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(mission_rows)}

## Relations represented

| Relation | Triples |
|---|---:|
{chr(10).join(relation_rows)}

## Relational coverage matrix

| Mission | Payload | Developer organization | Launch | Orbit | Temporal | Science/objective |
|---|---|---|---|---|---|---|
{chr(10).join(matrix_rows)}

`✓` means at least one directly supported fact is curated, not that the category is exhaustively covered. A dash means no such fact is currently represented and is not a negative factual claim. Orbit status distinguishes occupied from planned/targeted in entity properties and annotation notes. Gaganyaan launch is not applicable to the unflown crewed mission; its planned launcher is not mislabeled as an actual launch.
"""
    (REPORTS / "phase1_coverage_report.md").write_text(coverage_report, encoding="utf-8")

    gaps = """# Phase 1 corpus gap analysis

## Gap assessment and targeted follow-up

The collection pass prioritized the seven specified mission programmes, and the follow-up collection added mission launch pages and official payload/science publications where initial coverage was thin. This is a bounded collection, not an exhaustive catalogue of ISRO publications.

1. **Payload-developer chains vary by mission.** Aditya-L1 has explicit organization attribution for its seven payloads, and AstroSat's five payload developer attributions are explicit in its mission page. Mars Orbiter Mission has a directly stated developer for TIS. The collected Chandrayaan-1/2/3 sources currently support payload membership, but no developer edges were curated where the source statements did not establish the attribution directly. This leaves uneven multi-hop coverage and is not filled by centre co-occurrence or presumed institutional roles.
2. **Temporal relation coverage is sparse by design.** Chandrayaan-3 is directly described as a follow-on to Chandrayaan-2. No other predecessor edge is asserted from chronology alone. Additional sequence claims require explicit primary-source wording.
3. **Gaganyaan is a programme under development, not a completed crewed launch.** The official source describes the proposed mission and identifies HSFC as lead centre. No actual launch date, launch event, or flown payload relation is asserted.
4. **Science and objective links are selective.** The KG-ready facts preserve direct mission-level targets/objectives where explicit, but payload objectives are not exhaustively converted to triples. The source documents and chunks retain broader science evidence for later reviewed annotation.
5. **Extraction limitations remain visible.** The Chandrayaan-1 brochure is scanned/image-only and produced no text layer; it remains registered and raw-only. One page in the Chandrayaan-2 brochure also produced no text. Extraction flagged 17 PDF pages with private-use glyphs, retaining the text unchanged for review. No OCR-derived claims were introduced.
6. **Source-access limitations are explicit.** Two legacy PRADAN/ISSDC portal URLs recorded in the pre-existing Aditya pilot manifest returned HTTP 504, and some older ISRO routes returned 404. They were not included as evidence; the selected corpus uses the valid current official routes in the registry. Structural validation checks local checksums and URL form but does not make live network requests.
7. **Coverage is not uniform by volume.** Chandrayaan-2 has far more chunks because its large official science-results publication and payload-data handbook were included. Chunk counts should not be interpreted as answer coverage or comparative evidence quality.

## Stop criterion

The second search pass improved launch and payload/science source coverage. Remaining omissions are either facts not directly established in the collected text or deliberately deferred to avoid inferred relations and unbounded collection. The corpus is suitable as a Phase 1 provenance-preserving, KG-ready initial resource; it is not claimed to be exhaustive. Further source collection should be a separately recorded, evidence-led extension rather than driven by benchmark performance.
"""
    (REPORTS / "phase1_gap_analysis.md").write_text(gaps, encoding="utf-8")


if __name__ == "__main__":
    analyze_coverage()
