"""Audit evidence-backed relation and multi-hop capacity in the Phase 1 corpus."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus"
REPORTS = ROOT / "reports"
RELATIONAL_CAPACITY_PATH = CORPUS / "relational_capacity.json"
AUDIT_REPORT_PATH = REPORTS / "phase1_6_relational_curation.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _evidence_summary(edge: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "document_id": item["document_id"],
            "source_chunk_id": item["source_chunk_id"],
            "source_section": item["source_section"],
            "source_url": item["source_url"],
            "supporting_excerpt": item["supporting_excerpt"],
        }
        for item in edge["provenance"]
    ]


def _path_record(edges: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path_pattern": " -> ".join(
            [edges[0]["subject_type"]]
            + [part for edge in edges for part in (edge["relation"], edge["object_type"])]
        ),
        "mission": edges[0]["mission"],
        "nodes": [
            {
                "entity_id": edges[0]["subject_id"],
                "name": edges[0]["subject"],
                "type": edges[0]["subject_type"],
            },
            *[
                {
                    "entity_id": edge["object_id"],
                    "name": edge["object"],
                    "type": edge["object_type"],
                }
                for edge in edges
            ],
        ],
        "triple_ids": [edge["triple_id"] for edge in edges],
        "edges": [
            {
                "relation": edge["relation"],
                "subject": edge["subject"],
                "object": edge["object"],
                "provenance_supported": bool(edge.get("provenance")),
                "evidence": _evidence_summary(edge),
            }
            for edge in edges
        ],
        "source_documents": sorted({
            item["document_id"]
            for edge in edges
            for item in edge["provenance"]
        }),
        "all_edges_provenance_supported": all(bool(edge.get("provenance")) for edge in edges),
    }


def _directed_paths(edges: list[dict[str, Any]], hops: int) -> list[dict[str, Any]]:
    supported_edges = [edge for edge in edges if edge.get("provenance")]
    paths: dict[tuple[str, ...], list[dict[str, Any]]] = {}

    def extend(path: list[dict[str, Any]]) -> None:
        if len(path) == hops:
            paths[tuple(edge["triple_id"] for edge in path)] = path
            return
        previous = path[-1]
        for edge in supported_edges:
            if edge["subject_id"] != previous["object_id"]:
                continue
            if edge["triple_id"] in {part["triple_id"] for part in path}:
                continue
            extend(path + [edge])

    for edge in supported_edges:
        extend([edge])
    return [_path_record(path) for path in paths.values()]


def _shared_payload_motifs(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_payload: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for edge in edges:
        if edge.get("provenance") and edge["subject_type"] == "Payload":
            by_payload[edge["subject_id"]][edge["relation"]].append(edge)
    motifs = []
    for payload_id, relations in by_payload.items():
        for developer in relations["DEVELOPED_BY"]:
            for observation in relations["OBSERVES"]:
                motif = _path_record([developer, observation])
                motif["path_pattern"] = (
                    "Organization <- DEVELOPED_BY - Payload - OBSERVES -> "
                    f"{observation['object_type']}"
                )
                motif["nodes"] = [
                    {"entity_id": developer["object_id"], "name": developer["object"], "type": developer["object_type"]},
                    {"entity_id": payload_id, "name": developer["subject"], "type": "Payload"},
                    {"entity_id": observation["object_id"], "name": observation["object"], "type": observation["object_type"]},
                ]
                motif["mission"] = next(
                    (candidate["mission"] for candidate in edges
                     if candidate["relation"] == "HAS_PAYLOAD" and candidate["object_id"] == payload_id),
                    developer["mission"],
                )
                motif["triple_ids"] = [developer["triple_id"], observation["triple_id"]]
                motif["edges"] = [
                    {
                        "relation": developer["relation"],
                        "subject": developer["subject"],
                        "object": developer["object"],
                        "provenance_supported": True,
                        "evidence": _evidence_summary(developer),
                    },
                    {
                        "relation": observation["relation"],
                        "subject": observation["subject"],
                        "object": observation["object"],
                        "provenance_supported": True,
                        "evidence": _evidence_summary(observation),
                    },
                ]
                motif["source_documents"] = sorted({
                    item["document_id"]
                    for edge in (developer, observation)
                    for item in edge["provenance"]
                })
                motif["all_edges_provenance_supported"] = True
                motifs.append(motif)
    return motifs


def _group_paths(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        grouped[path["path_pattern"]].append(path)
    result = []
    for pattern, instances in sorted(grouped.items()):
        result.append({
            "path_pattern": pattern,
            "distinct_path_count": len(instances),
            "missions": sorted({path["mission"] for path in instances}),
            "mission_count": len({path["mission"] for path in instances}),
            "source_document_count": len({
                document_id for path in instances for document_id in path["source_documents"]
            }),
            "all_edges_provenance_supported": all(
                path["all_edges_provenance_supported"] for path in instances
            ),
            "examples": instances[:2],
            "instances": instances,
        })
    return result


def _mission_question_capacity(
    triples: list[dict[str, Any]],
    relation: str,
) -> dict[str, Any]:
    selected = [edge for edge in triples if edge["relation"] == relation and edge.get("provenance")]
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in selected:
        by_subject[edge["subject_id"]].append(edge)
    return {
        "supported_fact_count": len(selected),
        "unique_subject_query_count": len(by_subject),
        "missions": sorted({edge["mission"] for edge in selected}),
        "source_documents": sorted({
            item["document_id"] for edge in selected for item in edge["provenance"]
        }),
    }


def audit_relational_capacity() -> dict[str, Any]:
    registry = read_json(CORPUS / "document_registry.json")
    ontology = read_json(CORPUS / "ontology.json")
    relation_schema = read_json(CORPUS / "relations.json")
    triples = read_jsonl(CORPUS / "triples.jsonl")
    entities = read_jsonl(CORPUS / "entities.jsonl")
    chunks = read_jsonl(CORPUS / "chunks.jsonl")
    statistics = read_json(CORPUS / "corpus_statistics.json")
    relational_coverage = read_json(CORPUS / "relational_coverage.json")
    validation = read_json(CORPUS / "validation_report.json")
    extraction = read_json(CORPUS / "extraction_report.json")
    missions = sorted({
        document["mission"]
        for document in registry["documents"]
        if document["mission"] != "cross-mission"
    })
    definitions = relation_schema["relations"]
    relation_statistics = []
    by_relation_mission: dict[str, Counter[str]] = defaultdict(Counter)
    for definition in definitions:
        relation = definition["name"]
        edges = [edge for edge in triples if edge["relation"] == relation]
        evidence_records = [
            item for edge in edges for item in edge.get("provenance", [])
        ]
        relation_statistics.append({
            "relation": relation,
            "triple_count": len(edges),
            "unique_subject_count": len({edge["subject_id"] for edge in edges}),
            "unique_object_count": len({edge["object_id"] for edge in edges}),
            "missions": sorted({edge["mission"] for edge in edges}),
            "mission_count": len({edge["mission"] for edge in edges}),
            "distinct_source_document_count": len({item["document_id"] for item in evidence_records}),
            "provenance_supported_instance_count": sum(bool(edge.get("provenance")) for edge in edges),
        })
        for edge in edges:
            by_relation_mission[relation][edge["mission"]] += 1

    mission_relation_matrix = {
        mission: {
            definition["name"]: by_relation_mission[definition["name"]][mission]
            for definition in definitions
        }
        for mission in missions
    }
    two_hop_instances = _directed_paths(triples, 2)
    three_hop_instances = _directed_paths(triples, 3)
    two_hop_patterns = _group_paths(two_hop_instances)
    three_hop_patterns = _group_paths(three_hop_instances)
    shared_payload_motifs = _shared_payload_motifs(triples)

    payload_capacity = _mission_question_capacity(triples, "HAS_PAYLOAD")
    date_edges = [edge for edge in triples if edge["relation"] == "LAUNCHED_ON" and edge.get("provenance")]
    developer_edges = [edge for edge in triples if edge["relation"] == "DEVELOPED_BY" and edge.get("provenance")]
    observes_edges = [edge for edge in triples if edge["relation"] == "OBSERVES" and edge.get("provenance")]
    predecessor_edges = [edge for edge in triples if edge["relation"] == "PRECEDED_BY" and edge.get("provenance")]
    developer_paths = [
        path for path in two_hop_instances
        if path["path_pattern"] in {
            "Mission -> HAS_PAYLOAD -> Payload -> DEVELOPED_BY -> Organization",
            "Mission -> HAS_PAYLOAD -> Payload -> DEVELOPED_BY -> ISROCentre",
        }
    ]
    objective_paths = [
        path for path in two_hop_instances
        if path["path_pattern"] == "Mission -> HAS_PAYLOAD -> Payload -> HAS_OBJECTIVE -> ScientificObjective"
    ]
    observation_paths = [
        path for path in two_hop_instances
        if path["path_pattern"] in {
            "Mission -> HAS_PAYLOAD -> Payload -> OBSERVES -> ScientificTarget",
            "Mission -> HAS_PAYLOAD -> Payload -> OBSERVES -> PlanetaryBody",
            "Mission -> HAS_PAYLOAD -> Payload -> OBSERVES -> Phenomenon",
        }
    ]
    target_motif_queries = len({
        (motif["mission"], motif["nodes"][-1]["entity_id"])
        for motif in shared_payload_motifs
    })
    observation_queries = len({
        (edge["mission"], edge["object_id"]) for edge in observes_edges
    })
    temporal_three_hop_instances = [
        path for path in three_hop_instances
        if path["path_pattern"].startswith("Mission -> PRECEDED_BY -> Mission ->")
    ]
    question_capacity = {
        "estimation_policy": (
            "Counts are evidence-bounded prompt opportunities, not benchmark questions. "
            "A grouped subject query is counted once per supported subject; fact-level "
            "relations are counted separately. No model outputs or absent facts are used."
        ),
        "categories": [
            {
                "category": "A. Direct factual",
                "conservative_question_capacity": len(date_edges),
                "supporting_fact_count": len(date_edges),
                "missions": sorted({edge["mission"] for edge in date_edges}),
                "source_documents": sorted({
                    item["document_id"] for edge in date_edges for item in edge["provenance"]
                }),
                "independently_grounded": True,
                "basis": "One launch-date question per mission with a source-linked LAUNCHED_ON fact.",
            },
            {
                "category": "B. Single relationship",
                "conservative_question_capacity": payload_capacity["unique_subject_query_count"],
                "supporting_fact_count": payload_capacity["supported_fact_count"],
                "missions": payload_capacity["missions"],
                "source_documents": payload_capacity["source_documents"],
                "independently_grounded": True,
                "basis": f"One mission-level payload-list question per mission; {payload_capacity['supported_fact_count']} distinct HAS_PAYLOAD edges support {payload_capacity['unique_subject_query_count']} mission lists.",
            },
            {
                "category": "C. Relational attribute",
                "conservative_question_capacity": len(developer_edges),
                "supporting_fact_count": len(developer_edges),
                "missions": sorted({edge["mission"] for edge in developer_edges}),
                "source_documents": sorted({
                    item["document_id"] for edge in developer_edges for item in edge["provenance"]
                }),
                "independently_grounded": True,
                "basis": "One payload-developer attribution prompt per directly supported DEVELOPED_BY fact.",
            },
            {
                "category": "D. Two-hop",
                "conservative_question_capacity": len({path["mission"] for path in developer_paths}),
                "payload_specific_path_prompt_upper_bound": len(developer_paths),
                "grouped_mission_query_count": len({path["mission"] for path in developer_paths}),
                "supporting_path_count": len(developer_paths),
                "missions": sorted({path["mission"] for path in developer_paths}),
                "source_documents": sorted({
                    document_id for path in developer_paths for document_id in path["source_documents"]
                }),
                "independently_grounded": all(path["all_edges_provenance_supported"] for path in developer_paths),
                "basis": "Mission → HAS_PAYLOAD → Payload → DEVELOPED_BY → Organization/ISROCentre; fact-level count is distinct payload-developer paths, with a lower grouped mission-query count.",
            },
            {
                "category": "E. Scientific relationship",
                "conservative_question_capacity": observation_queries,
                "supporting_fact_count": len(observes_edges),
                "missions": sorted({edge["mission"] for edge in observes_edges}),
                "source_documents": sorted({
                    item["document_id"] for edge in observes_edges for item in edge["provenance"]
                }),
                "independently_grounded": True,
                "basis": "One mission/target-specific 'which payload observes X?' prompt per distinct sourced mission-target pair.",
            },
            {
                "category": "F. Temporal/mission relationship",
                "conservative_question_capacity": len(predecessor_edges),
                "supporting_fact_count": len(predecessor_edges),
                "missions": sorted({edge["mission"] for edge in predecessor_edges}),
                "source_documents": sorted({
                    item["document_id"] for edge in predecessor_edges for item in edge["provenance"]
                }),
                "independently_grounded": True,
                "basis": "Only explicitly sourced PRECEDED_BY edges are counted; chronology inferred solely from dates is excluded.",
            },
            {
                "category": "G. Multi-hop payload/organization/science",
                "conservative_question_capacity": target_motif_queries,
                "grouped_target_query_count": target_motif_queries,
                "payload_specific_motif_count": len(shared_payload_motifs),
                "supporting_path_count": len(shared_payload_motifs),
                "missions": sorted({motif["mission"] for motif in shared_payload_motifs}),
                "source_documents": sorted({
                    document_id for motif in shared_payload_motifs for document_id in motif["source_documents"]
                }),
                "independently_grounded": all(
                    motif["all_edges_provenance_supported"] for motif in shared_payload_motifs
                ),
                "basis": "One mission/target-specific prompt for each sourced target reached through a payload with both DEVELOPED_BY and OBSERVES edges; these are branching joins at the same payload, not directed three-edge paths.",
            },
            {
                "category": "H. Mission → payload → objective",
                "conservative_question_capacity": len({path["mission"] for path in objective_paths}),
                "payload_specific_path_prompt_upper_bound": len(objective_paths),
                "supporting_path_count": len(objective_paths),
                "missions": sorted({path["mission"] for path in objective_paths}),
                "source_documents": sorted({
                    document_id for path in objective_paths for document_id in path["source_documents"]
                }),
                "independently_grounded": all(path["all_edges_provenance_supported"] for path in objective_paths),
                "basis": "A mission-level grouped prompt can ask which objectives are associated with payloads carried by the mission; every path requires a sourced HAS_PAYLOAD and HAS_OBJECTIVE edge.",
            },
        ],
        "unanswerable_capacity": {
            "question_count_estimate": None,
            "status": "Constructible in principle, not counted in this audit.",
            "qualification": (
                "Absence from this bounded corpus can support 'not answerable from this corpus' "
                "only; it cannot establish that a real-world fact is false. Do not turn missing "
                "developer edges or unflown status into unqualified negative answers."
            ),
        },
    }

    registry_by_id = {doc["document_id"]: doc for doc in registry["documents"]}
    document_ids_by_mission = {
        mission: sorted(doc["document_id"] for doc in registry["documents"] if doc["mission"] == mission)
        for mission in ("Chandrayaan-1", "Chandrayaan-2", "Chandrayaan-3")
    }
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    def evidence_locator(chunk_id: str, excerpt: str) -> dict[str, str]:
        chunk = chunk_by_id[chunk_id]
        doc = registry_by_id[chunk["document_id"]]
        return {
            "document_id": chunk["document_id"],
            "title": doc["title"],
            "source_url": doc["source_url"],
            "source_chunk_id": chunk_id,
            "source_section": chunk["section"],
            "supporting_excerpt": excerpt,
        }

    chandrayaan_findings = [
        {
            "mission": "Chandrayaan-1",
            "developer_attribution_status": "B: explicit C1XS/SARA statements in collected source text are now curated; other wording remains ambiguous for DEVELOPED_BY.",
            "developer_evidence": [
                evidence_locator("ISSDC_CY1_PAYLOADS::c00001", "C1XS and SARA are developed by ESA jointly with ISRO"),
                evidence_locator("ISSDC_CY1_PAYLOADS::c00001", "Rutherford Appleton Laboratory, UK and ISRO Satellite Centre, ISRO"),
                evidence_locator("ISSDC_CY1_PAYLOADS::c00001", "Near Infra-Red spectrometer (SIR-2) from Max Plank Institute"),
                evidence_locator("ISSDC_CY1_PAYLOADS::c00002", "Moon Mineralogy Mapper (M3) from Brown University and Jet Propulsion Laboratory, USA through NASA"),
            ],
            "objective_status": "B: payload objectives already present in the collected ISSDC payload document are now curated with chunk-level provenance.",
            "assessment": (
                "The document states five core payload/experiments were indigenously developed and explicitly says C1XS and SARA are developed jointly by ESA and ISRO. "
                "The domestic core statement does not assign each payload to a named centre. 'From ... through ESA/NASA' statements do not by themselves prove DEVELOPED_BY."
            ),
            "extraction_status": "The payload PDF has text on all 17 pages but extraction flagged private-use glyphs on 10 pages. The cited claims are readable excerpts; unrelated glyphs should not be silently normalized.",
        },
        {
            "mission": "Chandrayaan-2",
            "developer_attribution_status": "A: no direct payload-to-developer attribution was found in the collected payload overview, payload document, brochure text, science-results volume, or data handbook during targeted text review.",
            "developer_evidence": [],
            "objective_status": "B: mission/science objectives and selected payload objectives already present in collected ISSDC text are now curated with chunk-level provenance.",
            "objective_evidence": [
                evidence_locator("ISSDC_CY2_SCIENCE_RESULTS::c00035", "Mission objectives are as follows"),
                evidence_locator("ISSDC_CY2_SCIENCE_RESULTS::c00036", "The scientific objective of the mission is to expand the lunar scientific knowledge"),
                evidence_locator("ISSDC_CY2_OVERVIEW::c00011", "science payloads aim to perform detailed study of lunar topography"),
            ],
            "assessment": (
                "Collected payload pages name instruments and describe observations/objectives. The overview has empty 'Mission Objectives' and 'Science Objectives' heading chunks, but the 150-page ISSDC science-results volume (and 53-page payload-data handbook) contains explicit mission-level objective paragraphs. "
                "No developer attribution was confirmed from direct text in this corpus; author affiliations or centre involvement must not be substituted."
            ),
            "extraction_status": "150/150 pages of the science-results volume and 53/53 handbook pages yielded text; 7 PDF pages of the payload document have unresolved private-use glyphs.",
        },
        {
            "mission": "Chandrayaan-3",
            "developer_attribution_status": "A: no explicit payload developer attribution found in the collected mission-detail page, 12-page brochure, and overview text.",
            "developer_evidence": [],
            "objective_status": "B: the mission objective was already curated; selected payload objectives from the collected details page are now curated with chunk-level provenance.",
            "objective_evidence": [
                evidence_locator("ISRO_CY3_DETAILS::c00002", "demonstrate end-to-end capability in safe landing and roving on the lunar surface"),
                evidence_locator("ISRO_CY3_DETAILS::c00023", "Objectives: To carry out the measurements of thermal properties"),
                evidence_locator("ISRO_CY3_DETAILS::c00024", "Objectives: To measure seismicity around the landing site"),
            ],
            "assessment": (
                "Mission details explicitly enumerate payloads and objectives and call the lander/propulsion module/rover indigenous, but do not attribute individual payload development to named centres. "
                "General ISRO ownership or project participation is not a payload-specific developer edge."
            ),
            "extraction_status": "The collected overview, details, and brochure were text-extracted without recorded empty pages or warnings.",
        },
    ]

    source_problems = {
        "verified_additional_sources_found": [],
        "search_attempts": [
            {
                "scope": "Official ISRO/ISSDC search endpoints for Chandrayaan-2/3 payload developers and mission-sequence documents.",
                "outcome": "The attempted official site search routes returned HTTP 404; third-party search endpoints returned challenge or irrelevant/unverifiable output. No additional document was verified or downloaded.",
            }
        ],
        "collection_rule": "Do not collect a candidate without opening the official document and verifying the specific relation text.",
    }
    phase1_5_relation_counts = {
        "HAS_OBJECTIVE": 2,
        "DEVELOPED_BY": 13,
        "OBSERVES": 3,
        "STUDIES": 4,
    }
    inventory = {
        "registered_document_count": len(registry["documents"]),
        "documents_by_host": dict(sorted(Counter(doc["source_domain"] for doc in registry["documents"]).items())),
        "authority_tier_counts": dict(sorted(Counter(doc["authority_tier"] for doc in registry["documents"]).items())),
        "mission_document_ids": document_ids_by_mission,
        "mission_count": len(missions),
        "chunk_count": len(chunks),
        "entity_count": len(entities),
        "triple_count": len(triples),
        "controlled_relation_count": len(definitions),
        "provenance_linked_triple_count": sum(bool(edge.get("provenance")) for edge in triples),
        "validation_status": validation["status"],
        "validation_checks_passed": validation["passed_check_count"],
        "validation_check_count": validation["check_count"],
        "extraction_warnings": sum(len(doc.get("warnings", [])) for doc in extraction["documents"]),
        "pages_without_text": sum(
            doc["page_count"] - doc["pages_with_text"] for doc in extraction["documents"]
        ),
        "ontology_entity_type_count": len(ontology["entity_types"]),
        "phase1_5_baseline": {
            "registered_document_count": 33,
            "chunk_count": 1006,
            "entity_count": 94,
            "triple_count": 98,
            "additional_documents_in_phase1_6": len(registry["documents"]) - 33,
            "additional_chunks_in_phase1_6": len(chunks) - 1006,
            "additional_entities_in_phase1_6": len(entities) - 94,
            "additional_triples_in_phase1_6": len(triples) - 98,
        },
        "phase1_6_added_relation_triples": {
            item["relation"]: item["triple_count"] - phase1_5_relation_counts[item["relation"]]
            for item in relation_statistics
            if item["relation"] in phase1_5_relation_counts
            and item["triple_count"] > phase1_5_relation_counts[item["relation"]]
        },
        "phase1_statistics_match": {
            "documents": statistics["document_count"] == len(registry["documents"]),
            "chunks": statistics["chunks_count"] == len(chunks),
            "entities": statistics["entity_count"] == len(entities),
            "triples": statistics["triple_count"] == len(triples),
            "provenance": statistics["provenance_linked_triple_count"] == sum(bool(edge.get("provenance")) for edge in triples),
            "relational_coverage_missions": set(relational_coverage["coverage_matrix"]) == set(missions),
        },
    }
    result = {
        "audit_version": "1.1",
        "inventory": inventory,
        "relation_statistics": relation_statistics,
        "mission_relation_matrix": mission_relation_matrix,
        "two_hop": {
            "directed_path_instance_count": len(two_hop_instances),
            "pattern_count": len(two_hop_patterns),
            "patterns": two_hop_patterns,
            "shared_payload_developed_by_and_observes_motifs": {
                "count": len(shared_payload_motifs),
                "missions": sorted({motif["mission"] for motif in shared_payload_motifs}),
                "patterns": _group_paths(shared_payload_motifs),
                "instances": shared_payload_motifs,
            },
        },
        "three_hop": {
            "directed_path_instance_count": len(three_hop_instances),
            "pattern_count": len(three_hop_patterns),
            "patterns": three_hop_patterns,
            "provenance_coverage": (
                "Every enumerated path requires provenance on every edge; "
                f"{sum(path['all_edges_provenance_supported'] for path in three_hop_instances)}"
                f"/{len(three_hop_instances)} paths are provenance-supported."
            ),
        },
        "question_construction_capacity": question_capacity,
        "chandrayaan_evidence_audit": chandrayaan_findings,
        "additional_source_search": source_problems,
        "decision": {
            "recommendation": "TARGETED COLLECTION REQUIRED",
            "justification": (
                "The current data can support relation-intensive prompts, but breadth is narrow: the core Mission→Payload→DEVELOPED_BY pattern has only "
                f"{len(developer_paths)} payload-level paths across {len({path['mission'] for path in developer_paths})} missions; "
                f"Mission→Payload→OBSERVES has {len(observation_paths)} paths across "
                f"{len({path['mission'] for path in observation_paths})} missions; Mission→Payload→HAS_OBJECTIVE has "
                f"{len(objective_paths)} paths across {len({path['mission'] for path in objective_paths})} missions; "
                f"mission-level PRECEDED_BY has {len(predecessor_edges)} edge; and {len(three_hop_instances)} directed three-hop paths are all rooted in "
                f"{len(temporal_three_hop_instances)} continuations of that same temporal edge. The curated payload/objective and observation paths now provide meaningful two-hop coverage, "
                "but one explicit temporal edge is not enough to support a temporal question family. Since no additional authoritative sequence source was verified, keep temporal QA out of scope or perform narrowly targeted collection before including it."
            ),
            "collection_boundary": (
                "Explicit Chandrayaan-1/2/3 objective evidence already present in collected sources has been curated; no additional corpus documents were downloaded. "
                "Further collection is limited to an official source explicitly documenting multiple mission-sequence relations, if temporal questions remain in scope. "
                "No new source was verified in this audit, so no unverified document is downloaded or added."
            ),
        },
    }
    RELATIONAL_CAPACITY_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    AUDIT_REPORT_PATH.write_text(render_report(result), encoding="utf-8")
    return result


def render_report(data: dict[str, Any]) -> str:
    inventory = data["inventory"]
    relation_rows = []
    for item in data["relation_statistics"]:
        relation_rows.append(
            f"| {item['relation']} | {item['triple_count']} | {item['unique_subject_count']} | "
            f"{item['unique_object_count']} | {', '.join(item['missions']) or '—'} | "
            f"{item['distinct_source_document_count']} | {item['provenance_supported_instance_count']} |"
        )
    matrix_header = " | ".join(["Mission"] + [item["relation"] for item in data["relation_statistics"]])
    matrix_sep = "|".join(["---"] * (1 + len(data["relation_statistics"])))
    matrix_rows = [
        "| " + mission + " | " + " | ".join(str(count) for count in counts.values()) + " |"
        for mission, counts in data["mission_relation_matrix"].items()
    ]
    two_hop_rows = []
    for pattern in data["two_hop"]["patterns"]:
        example = pattern["examples"][0]
        rendered = " / ".join(
            f"{edge['subject']} —{edge['relation']}→ {edge['object']}"
            for edge in example["edges"]
        )
        evidence = " ; ".join(
            f"{edge['evidence'][0]['document_id']} {edge['evidence'][0]['source_chunk_id']}: "
            f"“{edge['evidence'][0]['supporting_excerpt'].replace(chr(10), ' ')}”"
            for edge in example["edges"]
        )
        two_hop_rows.append(
            f"| {pattern['path_pattern']} | {pattern['distinct_path_count']} | "
            f"{', '.join(pattern['missions'])} | {pattern['source_document_count']} | "
            f"{'yes' if pattern['all_edges_provenance_supported'] else 'no'} | "
            f"{rendered}<br>{evidence} |"
        )
    three_hop_rows = [
        f"| {pattern['path_pattern']} | {pattern['distinct_path_count']} | "
        f"{', '.join(pattern['missions']) or '—'} | {pattern['source_document_count']} | "
        f"{'yes' if pattern['all_edges_provenance_supported'] else 'no'} |"
        for pattern in data["three_hop"]["patterns"]
    ]
    question_rows = []
    for category in data["question_construction_capacity"]["categories"]:
        question_rows.append(
            f"| {category['category']} | {category['conservative_question_capacity']} "
            f"(supporting paths/facts: {category.get('supporting_path_count', category.get('supporting_fact_count'))}) | "
            f"{', '.join(category['missions']) or '—'} | "
            f"{', '.join(category['source_documents']) or '—'} | "
            f"{'yes' if category['independently_grounded'] else 'no'} |"
        )
    necessity_rows = [
        "| HAS_PAYLOAD / LAUNCHED_BY / LAUNCHED_ON / LAUNCHED_FROM / ORBITS | KG-NOT-NECESSARY | Each is typically stated together in one mission or launch-archive chunk; graph traversal may normalize and join them but is not needed to retrieve a single fact. |",
        "| DEVELOPED_BY (payload → organization) | KG-NOT-NECESSARY | Each supported attribution is directly stated in its source chunk(s); KG can normalize repeated organizations across payloads. |",
        "| Mission → HAS_PAYLOAD → Payload → DEVELOPED_BY → Organization/ISROCentre | KG-MULTI-HOP; potentially KG-beneficial | Requires joining payload membership and developer attribution, often across different source documents/chunks. The graph expresses the question naturally, but text retrieval could retrieve both passages. |",
        "| Mission → HAS_PAYLOAD → Payload → OBSERVES → target/body/phenomenon | KG-MULTI-HOP; KG-RELATIONALLY-NATURAL | Explicit linked relations form the query structure across the missions and targets listed above; this is a structural classification, not a claim text retrieval cannot combine the evidence. |",
        "| Payload → DEVELOPED_BY → Organization and Payload → OBSERVES → Target | KG-POTENTIALLY-USEFUL | A branching join at the same payload supports organization-for-observation questions; facts may be in different booklet/page chunks. |",
        "| PRECEDED_BY | KG-NOT-NECESSARY for the one supported instance | The source explicitly states the follow-on relation in one chunk. Current issue is coverage breadth, not retrieval complexity. |",
        "| Mission → HAS_PAYLOAD → Payload → HAS_OBJECTIVE → ScientificObjective | KG-MULTI-HOP; potentially KG-beneficial | Joins mission membership with a payload objective; source material can span a mission payload list and separate objective text. Both edges are directly sourced. |",
        "| Mission → HAS_OBJECTIVE / STUDIES | KG-NOT-NECESSARY for a directly stated single fact | The objective or science statement is directly present in a chunk; graph normalization may still aid cross-document joins. |",
        "| PRECEDED_BY → HAS_PAYLOAD → HAS_OBJECTIVE/OBSERVES | KG-MULTI-HOP, narrowly supported | Eight sourced paths exist, but every one starts from the same Chandrayaan-3 → Chandrayaan-2 predecessor edge; they do not provide broad temporal coverage. |",
    ]
    chandrayaan_lines = []
    for item in data["chandrayaan_evidence_audit"]:
        dev = item["developer_evidence"]
        obj = item.get("objective_evidence", [])
        citations = dev + obj
        citation_lines = [
            f"  - [{citation['document_id']} {citation['source_chunk_id']}]"
            f"({citation['source_url']}): “{citation['supporting_excerpt']}”"
            for citation in citations
        ]
        chandrayaan_lines.append(
            f"### {item['mission']}\n\n"
            f"- Developer finding: **{item['developer_attribution_status']}**\n"
            f"- Objective finding: **{item['objective_status']}**\n"
            f"- Extraction: {item['extraction_status']}\n"
            f"- Assessment: {item['assessment']}\n"
            + ("\n".join(citation_lines) + "\n" if citation_lines else "")
        )
    capacity = data["question_construction_capacity"]
    temporal_edge_count = next(
        item["triple_count"] for item in data["relation_statistics"]
        if item["relation"] == "PRECEDED_BY"
    )
    return f"""# Phase 1.6 — Existing-evidence relational curation and adequacy

## Decision

**{data['decision']['recommendation']}**

{data['decision']['justification']}

This is a corpus-design decision, not a finding that KG augmentation improves QA. Phase 1.6 curated explicit facts already present in collected Chandrayaan sources. It did not add documents, create benchmark questions, run experiments, modify results, or edit the paper.

{data['decision']['collection_boundary']}

## 1. Verified inventory

- Registered official documents: {inventory['registered_document_count']} ({', '.join(f"{host}: {count}" for host, count in inventory['documents_by_host'].items())})
- Authority: {inventory['authority_tier_counts']}
- Missions: {inventory['mission_count']}
- Chunks: {inventory['chunk_count']}
- Entities: {inventory['entity_count']}
- Curated triples: {inventory['triple_count']}
- Controlled relations: {inventory['controlled_relation_count']}
- Provenance-linked triples: {inventory['provenance_linked_triple_count']}/{inventory['triple_count']}
- Structural validation: {inventory['validation_status']} ({inventory['validation_checks_passed']}/{inventory['validation_check_count']})
- Extraction warnings: {inventory['extraction_warnings']}; pages without extractable text: {inventory['pages_without_text']}
- Phase 1 statistics cross-checks: {inventory['phase1_statistics_match']}
- Phase 1.5 → Phase 1.6 changes: {inventory['phase1_5_baseline']}
- New curated triples by relation: {inventory['phase1_6_added_relation_triples']}

All numbers above were recomputed from the registry, ontology, relation vocabulary, entities, triples, chunks, and reports—not copied from the request. The existing full test suite is separately rerun for this audit.

## 2. Relation frequency and provenance

All relation types present in the current controlled vocabulary are listed. Relations not defined in that vocabulary were not added to the analysis.

| Relation | Triples | Unique subjects | Unique objects | Missions | Distinct source documents | Provenance-supported instances |
|---|---:|---:|---:|---|---:|---:|
{chr(10).join(relation_rows)}

The document count is the union of provenance document IDs across that relation's triples. All counted edge instances carry provenance. It does not mean the documents contain only that relation.

## 3. Mission × relation matrix

Counts are provenance-supported triple instances, not coverage marks.

| {matrix_header} |
|{matrix_sep}|
{chr(10).join(matrix_rows)}

## 4. Provenance-supported two-hop paths

There are **{data['two_hop']['directed_path_instance_count']} distinct directed two-hop path instances** across **{data['two_hop']['pattern_count']} type/relation patterns**. Each path below requires both edges to have provenance; source evidence is shown for one example per pattern. Full instances and evidence are in `data/corpus/relational_capacity.json`.

| Pattern | Paths | Missions | Union of source documents | Both edges sourced? | Example path and edge evidence |
|---|---:|---|---:|---|---|
{chr(10).join(two_hop_rows)}

In addition, **{data['two_hop']['shared_payload_developed_by_and_observes_motifs']['count']} shared-payload branching motifs** join a payload's `DEVELOPED_BY` and `OBSERVES` facts. These support the form “which organization developed a payload that observes X”; they are not counted as directed Mission-starting paths. Missions represented: {', '.join(data['two_hop']['shared_payload_developed_by_and_observes_motifs']['missions']) or 'none'}. Both facts in every motif have provenance.

## 5. Three-hop paths

- Provenance-supported directed three-hop paths: **{data['three_hop']['directed_path_instance_count']}**
- Three-hop patterns: **{data['three_hop']['pattern_count']}**
- Mission coverage: {', '.join(sorted({instance['mission'] for pattern in data['three_hop']['patterns'] for instance in pattern['instances']})) or 'none'}.
- Provenance coverage: {data['three_hop']['provenance_coverage']}

| Pattern | Paths | Missions | Source documents | All edges sourced? |
|---|---:|---|---:|---|
{chr(10).join(three_hop_rows) or '| — | 0 | — | 0 | — |'}

The graph does not currently support the example organization → operations-centre/location extension. No relation was inferred to make a longer chain.

## 6. Question-construction capacity (estimate, not a benchmark)

Counts conservatively describe distinct evidence-bounded prompt opportunities or path/fact units. They are not generated benchmark questions and should not be summed across categories as independent samples.

| Category | Conservative prompt capacity (supporting paths/facts) | Missions | Source documents | Independently grounded? |
|---|---|---|---|---|
{chr(10).join(question_rows)}

For the two-hop developer category, {capacity['categories'][3]['grouped_mission_query_count']} grouped mission-level query stems cover {capacity['categories'][3]['supporting_path_count']} distinct payload/developer chains. Objective and observation chains are separately estimated above. Corpus-unanswerable prompts are possible in principle, but are not counted: corpus absence does not establish real-world nonexistence, and such questions require explicit corpus-scope wording in a later benchmark protocol.

## 7. KG-necessity research-design classification

These are structural classifications, not performance claims. “Potentially useful” does not mean text retrieval cannot find the facts.

| Relational pattern | Classification | Rationale |
|---|---|---|
{chr(10).join(necessity_rows)}

## 8. Chandrayaan gap investigation

The key distinction is between absent from the curated graph and absent from collected authoritative text.

{chr(10).join(chandrayaan_lines)}

Summary: explicit payload objective and selected observation facts from CY1/2/3 are now curated from the existing corpus; CY1 C1XS/SARA developer statements were also curated as written. Some institution/agency relationships remain too ambiguous for developer edges. No explicit payload-specific CY2/CY3 developer attribution was found in reviewed extracted text; that is a corpus gap, not evidence that such attribution does not exist elsewhere.

## 9. Additional official sources and access limits

No additional source was verified sufficiently to enter the registry or be downloaded during this audit. Attempted official ISRO/ISSDC site-search routes returned 404; other search endpoints did not yield inspectable primary documents. No search snippet is treated as evidence.

No extra document is required for the two-hop objective/observation capacity established by this curation. If temporal questions remain in scope, the unresolved collection target is:

| Priority | Candidate source/document family | Organization / mission | Document type and relation gap | Potential new entities, relations, and paths | Candidate URL/title status | Why the current corpus is insufficient |
|---|---|---|---|---|---|---|
| 1 | Official mission-history/sequence document explicitly identifying multiple predecessor/follow-on relations | ISRO/ISSDC; multiple missions | Mission history; `PRECEDED_BY` | Likely no new mission entities; number of new temporal relations and derived multi-hop continuations unknown until an opened primary source is verified. | No additional source verified; no URL/title can be responsibly supplied until inspected. | Only one explicit mission-sequence edge is currently supported. Chronology inferred solely from launch dates is excluded. |
| 2 | Payload-specific development/realization sources naming responsible centres/institutions | ISRO/ISSDC; Chandrayaan-2 and Chandrayaan-3 | Payload documentation; `Payload DEVELOPED_BY Organization/ISROCentre` | Potential new organizations/centres and developer edges; new mission→payload→developer paths depend on specific statements and are not estimated without a source. | Official document/title/URL not yet verified; do not download until opened and exact language checked. | Existing reviewed documents support identity/objectives but not payload-specific developer attribution; current developer paths across four other missions already support a bounded question family. |

### In-corpus curation completed in Phase 1.6

The targeted evidence review also found high-value directly supported claims already inside collected documents:

- Chandrayaan-1: ISSDC payload PDF (`ISSDC_CY1_PAYLOADS::c00001`) explicitly says C1XS and SARA were developed jointly by ESA and ISRO; selected payload objectives and observations were also curated. “From institution through agency” is not automatically a developer claim.
- Chandrayaan-2: ISSDC science-results volume (`ISSDC_CY2_SCIENCE_RESULTS::c00035` and `::c00036`) states mission/science objectives; selected payload objectives and observations are now curated from collected payload documentation.
- Chandrayaan-3: ISRO details page (`ISRO_CY3_DETAILS::c00002` and payload-objective table rows) states the mission objective and payload objectives; selected payload objective/observation facts are now curated.

Thus Phase 1.6 has completed the in-corpus curation. No document was added. Priority 1 temporal-source collection is warranted only if temporal QA remains in scope and should stop if no directly explicit authoritative records are found.

## 10. Final rationale and boundaries

The corpus remains Tier 1 authoritative and all curated triples have chunk-level provenance. Phase 1.6 increases the graph from 98 to {inventory['triple_count']} triples without adding documents, and adds multi-hop question capacity across payload/developer, payload/observation, and payload/objective patterns. The graph still has {temporal_edge_count} explicit temporal edge(s) and {data['three_hop']['directed_path_instance_count']} directed three-hop paths. Consequently **TARGETED COLLECTION REQUIRED** only if temporal QA is retained as a major intended category; otherwise, collection can stop for a bounded two-hop-focused experiment, with temporal questions excluded and Chandrayaan-2/3 developer attribution reported as a known coverage limitation.

No benchmark, benchmark answer, result file, prior experimental report, or paper was modified or run as part of this audit.
"""


if __name__ == "__main__":
    audit_relational_capacity()
