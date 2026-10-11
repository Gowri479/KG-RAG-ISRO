"""Build the frozen, provenance-linked relational QA benchmark.

Question specifications are curated independently from system outputs. The
builder reads only the frozen Phase 1 corpus/KG and writes outside
``data/benchmark`` to preserve that directory as an immutable pilot artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "relational_benchmark"


def _specifications() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    def add(
        question: str,
        category: str,
        mission: str,
        triple_ids: list[str],
        *,
        paths: list[list[str]] | None = None,
        kg_required: str = "NO",
        answer_from: str | None = None,
        answer_triples: list[str] | None = None,
        match_key: str | None = None,
        reasoning: str,
    ) -> None:
        specs.append({
            "question": question,
            "category": category,
            "mission": mission,
            "triple_ids": triple_ids,
            "paths": paths or [[triple_id] for triple_id in triple_ids],
            "kg_required": kg_required,
            "answer_from": answer_from or triple_ids[-1],
            "answer_triples": answer_triples or [answer_from or triple_ids[-1]],
            "match_key": match_key,
            "reasoning_requirement": reasoning,
        })

    launched = [
        ("Aditya-L1", "aditya-launch-date", "aditya-launcher"),
        ("AstroSat", "astrosat-launch-date", "astrosat-launcher"),
        ("Chandrayaan-1", "cy1-launch-date", "cy1-launcher"),
        ("Chandrayaan-2", "cy2-launch-date", "cy2-launcher"),
        ("Chandrayaan-3", "cy3-launch-date", "cy3-launcher"),
        ("Mars Orbiter Mission", "mom-launch-date", "mom-launcher"),
    ]
    for mission, date_id, _ in launched:
        add(
            f"On what date was the {mission} mission launched?",
            "DIRECT_CONTROL",
            mission,
            [date_id],
            reasoning="Read the launch date stated in one authoritative source chunk.",
        )
    for mission, _, launcher_id in launched:
        add(
            f"Which launch vehicle carried the {mission} spacecraft?",
            "DIRECT_CONTROL",
            mission,
            [launcher_id],
            reasoning="Read the mission-to-launch-vehicle fact stated in one source chunk.",
        )
    add(
        "What crewed-flight capability is the Gaganyaan programme intended to demonstrate?",
        "DIRECT_CONTROL",
        "Gaganyaan",
        ["gaganyaan-objective"],
        reasoning="Retrieve the explicit programme objective from one source chunk.",
    )
    add(
        "Which centre is identified as leading the Gaganyaan programme?",
        "DIRECT_CONTROL",
        "Gaganyaan",
        ["gaganyaan-led-by-hsfc"],
        reasoning="Retrieve the explicit mission-lead attribution from one source chunk.",
    )
    add(
        "Which organization is responsible for AstroSat operations?",
        "DIRECT_CONTROL",
        "AstroSat",
        ["astrosat-operated-by-istrac"],
        reasoning="Retrieve the explicit operator attribution from one source chunk.",
    )

    payload_lists = {
        "Aditya-L1": [
            "aditya-payload-velc", "aditya-payload-suit", "aditya-payload-solexs",
            "aditya-payload-hel1os", "aditya-payload-aspex", "aditya-payload-papa",
            "aditya-payload-mag",
        ],
        "AstroSat": [
            "astrosat-payload-uvit", "astrosat-payload-laxpc", "astrosat-payload-czti",
            "astrosat-payload-sxt", "astrosat-payload-ssm",
        ],
        "Chandrayaan-1": [
            "cy1-payload-tmc", "cy1-payload-hysi", "cy1-payload-llri",
            "cy1-payload-hex", "cy1-payload-mip", "cy1-payload-cixs",
            "cy1-payload-sir2", "cy1-payload-sara", "cy1-payload-minisar",
            "cy1-payload-m3", "cy1-payload-radom",
        ],
        "Chandrayaan-2": [
            "cy2-payload-tmc2", "cy2-payload-class", "cy2-payload-xsm",
            "cy2-payload-iirs", "cy2-payload-dual-frequency-sar", "cy2-payload-chace2",
            "cy2-payload-dfrs", "cy2-payload-ilsa",
            "cy2-payload-chaste", "cy2-payload-langmuir-probe", "cy2-payload-apxs",
            "cy2-payload-libs",
        ],
        "Chandrayaan-3": [
            "cy3-payload-rambha", "cy3-payload-chaste", "cy3-payload-ilsa",
            "cy3-payload-lra", "cy3-payload-apxs", "cy3-payload-libs",
            "cy3-payload-shape",
        ],
        "Mars Orbiter Mission": [
            "mom-payload-mcc", "mom-payload-tis", "mom-payload-msm",
            "mom-payload-menca", "mom-payload-lap",
        ],
    }
    for mission, ids in payload_lists.items():
        add(
            f"Name one science payload listed for {mission}.",
            "SINGLE_RELATION",
            mission,
            ids,
            answer_from=ids[0],
            answer_triples=ids,
            reasoning=(
                "Select one payload from the mission's explicitly listed payload edges; "
                "the answer is directly stated in the cited mission/payload source."
            ),
        )

    developer_controls = [
        ("Aditya-L1", "Who is credited with developing VELC for Aditya-L1?", "aditya-velc-developed-by-iia", "dev-velc"),
        ("Aditya-L1", "Which organization developed SUIT on Aditya-L1?", "aditya-suit-developed-by-iucaa", "dev-suit"),
        ("Aditya-L1", "Who developed the ASPEX payload carried by Aditya-L1?", "aditya-aspex-developed-by-prl", "dev-aspex"),
        ("Aditya-L1", "Which centre developed PAPA for Aditya-L1?", "aditya-papa-developed-by-vssc", "dev-papa"),
        ("Aditya-L1", "Who is named as the developer of Aditya-L1's magnetometer payload?", "aditya-mag-developed-by-leos", "dev-mag"),
        ("Mars Orbiter Mission", "Which centre developed the TIS payload for the Mars Orbiter Mission?", "mom-tis-developed-by-sac", "dev-tis"),
    ]
    for mission, question, triple_id, match_key in developer_controls:
        add(
            question,
            "SINGLE_RELATION",
            mission,
            [triple_id],
            match_key=match_key,
            reasoning="Retrieve the payload's developer attribution from one source chunk.",
        )

    objective_controls = [
        ("Chandrayaan-1", "What task was assigned to the Terrain Mapping Camera on Chandrayaan-1?", "cy1-tmc-objective", "obj-tmc"),
        ("Chandrayaan-2", "What scientific role is stated for ILSA on Chandrayaan-2?", "cy2-ilsa-objective", "obj-cy2-ilsa"),
        ("Chandrayaan-3", "What is APXS intended to determine on Chandrayaan-3?", "cy3-apxs-objective", "obj-apxs"),
        ("Chandrayaan-3", "What does SHAPE measure from lunar orbit?", "cy3-shape-objective", "obj-shape"),
    ]
    for mission, question, triple_id, match_key in objective_controls:
        add(
            question,
            "SINGLE_RELATION",
            mission,
            [triple_id],
            match_key=match_key,
            reasoning="Retrieve the named payload's objective directly from its cited source chunk.",
        )

    observation_controls = [
        ("Chandrayaan-1", "What phenomenon does SARA image in the lunar environment?", "cy1-sara-observes-lunar-solar-wind-interaction", "obs-sara"),
        ("Chandrayaan-1", "What target is Mini-SAR associated with observing in the lunar polar regions?", "cy1-minisar-observes-water-ice", "obs-minisar"),
        ("Chandrayaan-2", "What phenomenon is ILSA on Chandrayaan-2 designed to observe?", "cy2-ilsa-observes-moonquakes", "obs-cy2-ilsa"),
        ("Chandrayaan-3", "Which planetary body does the Chandrayaan-3 APXS observe?", "cy3-apxs-observes-lunar-soil", "obs-apxs"),
        ("Chandrayaan-3", "Which body does SHAPE observe from the lunar orbit?", "cy3-shape-observes-earth", "obs-shape"),
    ]
    for mission, question, triple_id, match_key in observation_controls:
        add(
            question,
            "SINGLE_RELATION",
            mission,
            [triple_id],
            match_key=match_key,
            reasoning="Retrieve the payload-to-target observation fact directly from its cited source chunk.",
        )

    developer_paths = [
        ("Aditya-L1", "Among the payloads carried by Aditya-L1, which was developed by the Indian Institute of Astrophysics?", ["aditya-payload-velc", "aditya-velc-developed-by-iia"], "dev-velc"),
        ("Aditya-L1", "Which Aditya-L1 payload was developed by the Inter-University Centre for Astronomy and Astrophysics?", ["aditya-payload-suit", "aditya-suit-developed-by-iucaa"], "dev-suit"),
        ("Aditya-L1", "Which payload aboard Aditya-L1 was developed by the Physical Research Laboratory?", ["aditya-payload-aspex", "aditya-aspex-developed-by-prl"], "dev-aspex"),
        ("Aditya-L1", "Which Aditya-L1 payload was developed by Vikram Sarabhai Space Centre?", ["aditya-payload-papa", "aditya-papa-developed-by-vssc"], "dev-papa"),
        ("Aditya-L1", "Which of Aditya-L1's carried payloads was developed by the Laboratory for Electro-Optics Systems?", ["aditya-payload-mag", "aditya-mag-developed-by-leos"], "dev-mag"),
        ("Mars Orbiter Mission", "Which payload carried by the Mars Orbiter Mission was developed by the Space Applications Centre?", ["mom-payload-tis", "mom-tis-developed-by-sac"], "dev-tis"),
    ]
    for mission, question, path, match_key in developer_paths:
        add(
            question,
            "TWO_HOP_RELATION",
            mission,
            path,
            paths=[path],
            kg_required="YES",
            answer_from=path[0],
            match_key=match_key,
            reasoning=(
                "Join the mission-to-payload edge with the payload-to-developer edge; "
                "the cited evidence is split across distinct chunks."
            ),
        )

    objective_paths = [
        ("Chandrayaan-1", "Which Chandrayaan-1 payload was tasked with mapping both lunar hemispheres and producing a three-dimensional topographic atlas?", ["cy1-payload-tmc", "cy1-tmc-objective"], "obj-tmc"),
        ("Chandrayaan-1", "Which Chandrayaan-1 payload was assigned to collect spectra for mapping minerals on the lunar surface?", ["cy1-payload-hysi", "cy1-hysi-objective"], None),
        ("Chandrayaan-1", "Which Chandrayaan-1 payload was intended to improve knowledge of spacecraft altitude, lunar topography, and the Moon's gravity field?", ["cy1-payload-llri", "cy1-llri-objective"], None),
        ("Chandrayaan-1", "Which Chandrayaan-1 payload was assigned to examine lunar chemical and radioactive composition as well as polar volatile transport?", ["cy1-payload-hex", "cy1-hex-objective"], None),
        ("Chandrayaan-1", "Which Chandrayaan-1 payload was designed to demonstrate probe-impact technology relevant to future soft-landing missions?", ["cy1-payload-mip", "cy1-mip-objective"], None),
        ("Chandrayaan-2", "Which Chandrayaan-2 payload was assigned to characterize seismic activity near the landing site?", ["cy2-payload-ilsa", "cy2-ilsa-objective"], "obj-cy2-ilsa"),
        ("Chandrayaan-3", "Which Chandrayaan-3 payload measures thermal properties near the lunar polar region?", ["cy3-payload-chaste", "cy3-chaste-objective"], None),
        ("Chandrayaan-3", "Which Chandrayaan-3 payload was tasked with measuring local seismicity and helping delineate lunar crust and mantle structure?", ["cy3-payload-ilsa", "cy3-ilsa-objective"], None),
        ("Chandrayaan-3", "Which Chandrayaan-3 passive experiment is intended to inform study of the Moon system's dynamics?", ["cy3-payload-lra", "cy3-lra-objective"], None),
        ("Chandrayaan-3", "Which Chandrayaan-3 payload determines the elemental composition of soil and rocks near the landing area?", ["cy3-payload-apxs", "cy3-apxs-objective"], "obj-apxs"),
        ("Chandrayaan-3", "Which Chandrayaan-3 payload analyzes surface elements to infer the Moon's mineralogical composition?", ["cy3-payload-libs", "cy3-libs-objective"], None),
        ("Chandrayaan-3", "Which Chandrayaan-3 payload studies Earth's spectral and polarimetric properties from lunar orbit?", ["cy3-payload-shape", "cy3-shape-objective"], "obj-shape"),
    ]
    for mission, question, path, match_key in objective_paths:
        add(
            question,
            "TWO_HOP_RELATION",
            mission,
            path,
            paths=[path],
            kg_required="YES",
            answer_from=path[0],
            match_key=match_key,
            reasoning=(
                "Identify the payload by joining its mission membership with its "
                "payload-specific objective; these facts have separate source chunks."
            ),
        )

    observation_paths = [
        ("Chandrayaan-1", "Which payload carried by Chandrayaan-1 images the interaction between solar wind and the lunar surface?", ["cy1-payload-sara", "cy1-sara-observes-lunar-solar-wind-interaction"], "obs-sara"),
        ("Chandrayaan-1", "Which Chandrayaan-1 payload observes water ice in permanently shadowed lunar polar regions?", ["cy1-payload-minisar", "cy1-minisar-observes-water-ice"], "obs-minisar"),
        ("Chandrayaan-2", "Which payload on Chandrayaan-2 observes seismic activity around its landing site?", ["cy2-payload-ilsa", "cy2-ilsa-observes-moonquakes"], "obs-cy2-ilsa"),
        ("Chandrayaan-3", "Which Chandrayaan-3 payload observes the Moon?", ["cy3-payload-apxs", "cy3-apxs-observes-lunar-soil"], "obs-apxs"),
        ("Chandrayaan-3", "Which payload aboard Chandrayaan-3 observes Earth from lunar orbit?", ["cy3-payload-shape", "cy3-shape-observes-earth"], "obs-shape"),
    ]
    for mission, question, path, match_key in observation_paths:
        add(
            question,
            "TWO_HOP_RELATION",
            mission,
            path,
            paths=[path],
            kg_required="YES",
            answer_from=path[0],
            match_key=match_key,
            reasoning=(
                "Join the mission-to-payload edge with the payload-to-observation "
                "edge to identify the payload matching the target or phenomenon."
            ),
        )

    add(
        "Which Chandrayaan-1 payload both mapped lunar elemental abundance using X-ray fluorescence and was developed jointly by ESA and ISRO?",
        "MULTI_RELATION",
        "Chandrayaan-1",
        [
            "cy1-payload-cixs", "cy1-cixs-objective",
            "cy1-c1xs-developed-by-esa", "cy1-c1xs-developed-by-isro",
        ],
        paths=[
            ["cy1-payload-cixs", "cy1-cixs-objective"],
            ["cy1-payload-cixs", "cy1-c1xs-developed-by-esa"],
            ["cy1-payload-cixs", "cy1-c1xs-developed-by-isro"],
        ],
        kg_required="YES",
        answer_from="cy1-payload-cixs",
        reasoning=(
            "Join mission membership, the payload objective, and both documented "
            "developer attributions. No single cited chunk states the complete conjunction."
        ),
    )
    add(
        "Which Chandrayaan-1 payload is linked both to imaging solar-wind interaction at the lunar surface and to ESA–ISRO development?",
        "MULTI_RELATION",
        "Chandrayaan-1",
        [
            "cy1-payload-sara", "cy1-sara-observes-lunar-solar-wind-interaction",
            "cy1-sara-developed-by-esa", "cy1-sara-developed-by-isro",
        ],
        paths=[
            ["cy1-payload-sara", "cy1-sara-observes-lunar-solar-wind-interaction"],
            ["cy1-payload-sara", "cy1-sara-developed-by-esa"],
            ["cy1-payload-sara", "cy1-sara-developed-by-isro"],
        ],
        kg_required="YES",
        answer_from="cy1-payload-sara",
        reasoning=(
            "Combine mission membership, a payload observation, and two developer "
            "edges. The evidence is distributed across separate source chunks."
        ),
    )
    add(
        "ISRO describes Chandrayaan-3 as following Chandrayaan-2. Which payload on that earlier mission was assigned to characterize seismic activity near the landing site?",
        "MULTI_RELATION",
        "Chandrayaan-3",
        [
            "cy3-predecessor", "cy2-payload-ilsa", "cy2-ilsa-objective",
        ],
        paths=[["cy3-predecessor", "cy2-payload-ilsa", "cy2-ilsa-objective"]],
        kg_required="YES",
        answer_from="cy2-payload-ilsa",
        reasoning=(
            "Follow the explicit predecessor edge, then join the earlier mission's "
            "payload membership and that payload's objective. This is the sole "
            "incidental temporal item, not a temporal question family."
        ),
    )
    return specs


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_benchmark(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    triples = _read_jsonl(CORPUS / "triples.jsonl")
    chunks = {chunk["chunk_id"]: chunk for chunk in _read_jsonl(CORPUS / "chunks.jsonl")}
    registry = json.loads((CORPUS / "document_registry.json").read_text(encoding="utf-8"))
    documents = {item["document_id"]: item for item in registry["documents"]}
    ontology = json.loads((CORPUS / "relations.json").read_text(encoding="utf-8"))
    relations = {item["name"] for item in ontology["relations"]}
    triple_by_id = {item["triple_id"]: item for item in triples}
    entities = {item["entity_id"]: item for item in _read_jsonl(CORPUS / "entities.jsonl")}

    specs = _specifications()
    counts_by_match: dict[str, int] = {}
    for spec in specs:
        key = spec.get("match_key")
        if key:
            counts_by_match[key] = counts_by_match.get(key, 0) + 1
    match_ids = {
        key: f"match_{index:02d}"
        for index, key in enumerate(sorted(counts_by_match), start=1)
        if counts_by_match[key] == 2
    }

    records = []
    provenance_records = []
    for index, spec in enumerate(specs, start=1):
        support = []
        source_chunks: dict[tuple[str, str], dict[str, Any]] = {}
        relations_for_question = []
        for triple_id in spec["triple_ids"]:
            if triple_id not in triple_by_id:
                raise ValueError(f"Question spec references unknown triple {triple_id}")
            triple = triple_by_id[triple_id]
            if triple["relation"] not in relations:
                raise ValueError(f"Triple {triple_id} has uncontrolled relation {triple['relation']}")
            support.append({
                "triple_id": triple_id,
                "subject_id": triple["subject_id"],
                "subject": triple["subject"],
                "relation": triple["relation"],
                "object_id": triple["object_id"],
                "object": triple["object"],
            })
            relations_for_question.append(triple["relation"])
            for item in triple.get("provenance", []):
                key = (item["document_id"], item["source_chunk_id"])
                source_chunks[key] = {
                    "document_id": item["document_id"],
                    "chunk_id": item["source_chunk_id"],
                    "source_url": item["source_url"],
                    "source_section": item.get("source_section"),
                    "source_page": item.get("source_page"),
                    "supporting_excerpt": item["supporting_excerpt"],
                }
        unique_docs = list(dict.fromkeys(chunk["document_id"] for chunk in source_chunks.values()))
        primary_doc = documents[unique_docs[0]]
        answer_triples = [triple_by_id[item] for item in spec["answer_triples"]]
        answer_entities = list(dict.fromkeys(item["object_id"] for item in answer_triples))
        answer_strings = []
        for entity_id in answer_entities:
            entity = entities.get(entity_id)
            if entity:
                answer_strings.append(entity["name"])
            else:
                matched = next(
                    (item["object"] for item in answer_triples if item["object_id"] == entity_id),
                    None,
                )
                if matched:
                    answer_strings.append(matched)
        reference_answer = triple_by_id[spec["answer_from"]]["object"]
        if spec["category"] == "SINGLE_RELATION" and len(spec["answer_triples"]) > 1:
            reference_answer = answer_strings[0]
        acceptable = []
        for answer in answer_strings:
            acceptable.append(answer)
            entity = next(
                (item for item in entities.values() if item.get("name") == answer),
                None,
            )
            if entity:
                acceptable.extend(alias for alias in entity.get("aliases", []) if alias)
        acceptable.append(reference_answer)
        acceptable = list(dict.fromkeys(acceptable))
        match_group_id = match_ids.get(spec.get("match_key"))
        record = {
            "question_id": f"rqa_{index:03d}",
            "question": spec["question"],
            "category": spec["category"],
            "kg_required": spec["kg_required"],
            "relation_type": list(dict.fromkeys(relations_for_question)),
            "mission": spec["mission"],
            "match_group_id": match_group_id,
            "reference_answer": reference_answer,
            "acceptable_answers": acceptable,
            "answer_entities": answer_entities,
            "supporting_triples": support,
            "supporting_paths": spec["paths"],
            "supporting_chunks": list(source_chunks.values()),
            "source_document": primary_doc["document_id"],
            "source_url": primary_doc["source_url"],
            "source_documents": unique_docs,
            "source_urls": list(dict.fromkeys(
                documents[doc_id]["source_url"] for doc_id in unique_docs
            )),
            "reasoning_requirement": spec["reasoning_requirement"],
            "answerability": "ANSWERABLE",
            "annotation_status": "single_researcher_constructed",
            "system_result_contamination": False,
        }
        records.append(record)
        provenance_records.append({
            "question_id": record["question_id"],
            "supporting_triples": support,
            "supporting_paths": spec["paths"],
            "supporting_chunks": record["supporting_chunks"],
            "source_documents": unique_docs,
            "source_urls": record["source_urls"],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = output_dir / "relational_qa_v1.json"
    provenance_path = output_dir / "relational_qa_provenance.jsonl"
    benchmark_path.write_text(
        json.dumps({
            "benchmark_id": "isro_relational_qa_v1",
            "schema_version": "1.0",
            "created_from": "frozen Phase 1 corpus/KG only",
            "question_construction": "single-researcher, performance-blind, provenance-first",
            "total_questions": len(records),
            "questions": records,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    provenance_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in provenance_records),
        encoding="utf-8",
    )
    return {"benchmark_path": benchmark_path, "provenance_path": provenance_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = build_benchmark(args.output_dir)
    print(f"Saved {result['benchmark_path']}")
    print(f"Saved {result['provenance_path']}")
    print(f"Questions: {len(_specifications())}")


if __name__ == "__main__":
    main()
