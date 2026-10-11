"""Analyze KG relevance on the frozen canonical test split without generation."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from src.evaluation.analysis_utils import (
    BASELINE_PATH,
    BENCHMARK_PATH,
    RESULTS_DIR,
    TEST_IDS_PATH,
    coverage,
    exact_match,
    is_idk,
    load_json,
    rouge_l,
)

ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_PATH = ROOT / "data" / "annotations" / "canonical_180_kg_relevance.jsonl"
GUIDELINES_PATH = ROOT / "data" / "annotations" / "canonical_180_kg_relevance_guidelines.md"
OUTPUT_PATH = RESULTS_DIR / "kg_relevance_analysis.json"
REPORT_PATH = ROOT / "reports" / "kg_relevance_analysis.md"
CORPUS = ROOT / "data" / "corpus"
MIN_INFERENTIAL_N = 10
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 42

QUESTION_TYPES = {
    "DIRECT_FACT",
    "ATTRIBUTE",
    "SINGLE_RELATION",
    "TWO_HOP_RELATION",
    "MULTI_RELATION",
    "OTHER",
}
KG_RELEVANCE = {
    "KG_NOT_RELEVANT",
    "KG_POTENTIALLY_USEFUL",
    "KG_RELATIONALLY_NATURAL",
    "KG_MULTI_HOP",
}
KG_REQUIRED = {"YES", "NO", "UNCERTAIN"}
SYSTEM_ANSWER_KEYS = {
    "BM25 + LLM": "bm25_answer",
    "Vanilla RAG": "vanilla_rag_answer",
    "KG-RAG": "kgrag_answer",
}

ATTRIBUTES = {
    2, 4, 5, 9, 18, 23, 28, 44, 47, 50, 54, 62, 66, 67, 70, 73, 77, 78,
    80, 81, 82, 86, 87, 89, 90, 91, 92, 94, 95, 97, 99, 100,
}
SINGLE_RELATIONS = {
    7, 35, 43, 69, 71, 75, 98, 106, 109, 112, 118, 120, 124, 128, 129,
    131, 134, 138, 139, 144, 147, 149, 150, 152, 154, 158, 160, 173,
}
TWO_HOP_RELATIONS = {101, 102, 105, 110, 116, 121, 130, 143, 153, 156}
MULTI_RELATIONS = {
    103, 107, 108, 111, 113, 114, 115, 119, 122, 123, 125, 127, 132, 133,
    135, 137, 141, 145, 146, 151, 155, 161, 162, 163, 164, 165, 166, 167,
    169, 170, 172, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184,
    185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 200,
}

POTENTIALLY_USEFUL = {
    2, 9, 18, 42, 54, 94, 101, 102, 105, 106, 110, 116, 120, 121, 122,
    124, 128, 132, 133, 150, 161, 162, 173, 175, 178, 192, 194,
}
RELATIONALLY_NATURAL = {7, 35, 43, 69, 71, 75, 98, 107, 112}
MULTI_HOP_RELEVANT = {130, 153, 156}
KG_REQUIRED_UNCERTAIN = {105, 111, 121, 122, 127, 137, 141, 145}

RELATIONS_BY_QUESTION = {
    2: ["HAS_OBJECTIVE"],
    7: ["LAUNCHED_BY"],
    9: ["STUDIES"],
    18: ["ORBITS"],
    35: ["LAUNCHED_BY"],
    42: ["STUDIES"],
    43: ["LAUNCHED_BY"],
    54: ["ORBITS"],
    67: ["ORBITS"],
    69: ["HAS_PAYLOAD"],
    71: ["LAUNCHED_BY"],
    75: ["LAUNCHED_BY"],
    94: ["ORBITS"],
    98: ["OBSERVES"],
    101: ["LAUNCHED_BY", "OBSERVES"],
    102: ["HAS_PAYLOAD", "OBSERVES"],
    105: ["HAS_PAYLOAD", "DEVELOPED_BY"],
    106: ["DEVELOPED_BY"],
    107: ["LAUNCHED_BY", "ORBITS"],
    109: ["OPERATED_BY"],
    110: ["HAS_PAYLOAD", "OBSERVES"],
    112: ["LAUNCHED_BY"],
    116: ["HAS_PAYLOAD", "OBSERVES"],
    118: ["OPERATED_BY"],
    120: ["DEVELOPED_BY"],
    121: ["HAS_PAYLOAD", "HAS_OBJECTIVE"],
    122: ["HAS_PAYLOAD", "OBSERVES"],
    124: ["DEVELOPED_BY"],
    127: ["OPERATED_BY"],
    128: ["LAUNCHED_BY"],
    130: ["HAS_PAYLOAD", "DEVELOPED_BY", "OBSERVES"],
    132: ["PRECEDED_BY"],
    133: ["LAUNCHED_BY", "HAS_PAYLOAD"],
    139: ["HAS_PAYLOAD"],
    147: ["OBSERVES"],
    150: ["LAUNCHED_BY"],
    153: ["HAS_PAYLOAD", "HAS_OBJECTIVE"],
    156: ["HAS_PAYLOAD", "DEVELOPED_BY"],
    161: ["LAUNCHED_ON"],
    162: ["LAUNCHED_ON"],
    165: ["LAUNCHED_ON"],
    170: ["LAUNCHED_ON"],
    173: ["PRECEDED_BY"],
    175: ["LAUNCHED_ON"],
    178: ["PRECEDED_BY"],
    181: ["LAUNCHED_ON"],
    192: ["LAUNCHED_ON"],
    194: ["LAUNCHED_ON"],
}

EVIDENCE_BY_QUESTION = {
    2: ["cy3-objective"],
    7: ["mom-launcher"],
    9: ["aditya-studies-solar-corona"],
    18: ["cy2-orbits-moon"],
    35: ["cy1-launcher"],
    42: ["aditya-studies-solar-corona"],
    43: ["cy3-launcher"],
    54: ["gaganyaan-targets-leo"],
    67: ["aditya-targets-l1-halo-orbit"],
    69: ["aditya-payload-velc"],
    71: ["aditya-launcher"],
    75: ["astrosat-launcher"],
    94: ["cy2-orbits-moon"],
    98: ["cy3-shape-observes-earth"],
    102: ["cy1-payload-mip"],
    107: ["aditya-launcher", "aditya-targets-l1-halo-orbit"],
    110: ["cy1-payload-m3", "cy1-payload-mip", "cy1-payload-minisar",
          "cy1-minisar-objective", "cy1-minisar-observes-water-ice"],
    112: ["aditya-launcher"],
    116: ["cy2-payload-chace2"],
    121: ["cy1-payload-tmc", "cy1-tmc-objective"],
    122: ["cy1-payload-m3", "cy1-payload-mip", "cy1-payload-minisar",
          "cy1-minisar-observes-water-ice"],
    130: ["aditya-payload-aspex", "aditya-aspex-developed-by-prl"],
    153: ["cy2-payload-iirs", "cy2-iirs-objective"],
    156: ["cy1-payload-cixs", "cy1-c1xs-developed-by-esa",
          "cy1-payload-sara", "cy1-sara-developed-by-esa"],
    161: ["cy1-launch-date", "cy2-launch-date"],
    162: ["mom-launch-date", "cy2-launch-date"],
    175: ["astrosat-launch-date", "mom-launch-date"],
    192: ["cy1-launch-date", "mom-launch-date", "astrosat-launch-date",
          "aditya-launch-date"],
    194: ["cy3-launch-date", "aditya-launch-date"],
}

REASON_NOTES = {
    101: "The question's discovery-to-spacecraft/launcher chain is not represented for the unnamed satellite in this graph.",
    102: "The graph confirms that Chandrayaan-1 carried MIP, but its curated MIP edge concerns impact/technology objectives, not the requested discovery result.",
    105: "The CY3 graph does not identify Pragyan as a rover entity with a DEVELOPED_BY edge; the requested developer attribution is missing from this KG.",
    106: "DEVELOPED_BY is controlled, but the graph does not contain the requested general SAC-to-payload responsibility claim.",
    110: "The cited Mini-SAR edge is an objective/target relation, not evidence that it made the historical first confirmation; M3/CHACE discovery-result edges are not curated.",
    116: "The KG lists CHACE-2 on Chandrayaan-2, but does not encode the requested lunar-exosphere observation edge.",
    120: "DEVELOPED_BY is in the schema, but the graph has no GSAT-11–URSC edge.",
    121: "The graph links Chandrayaan-1 to TMC and records its mapping objective; it does not encode a PRODUCES/result relation, so that objective is not treated as proof of the requested output.",
    122: "Some mission–payload and water-ice objective edges exist, but the KG does not establish the requested historical findings by M3 and CHACE.",
    124: "DEVELOPED_BY is controlled, but no SAC-to-RISAT developer edge is present in the frozen graph.",
    127: "The KG contains no CY2–ISTRAC operations edge; its OPERATED_BY instance concerns AstroSat.",
    128: "Gaganyaan is planned and has no LAUNCHED_BY edge in the graph; the question asks about a designated launcher, not an actual launch.",
    130: "The KG joins Aditya-L1→ASPEX→PRL. The evidence concerns ASPEX; the graph does not model the STEPS sub-instrument's observation as a separate edge.",
    132: "The graph has a CY3 PRECEDED_BY CY2 edge, but it does not express the specific orbiter/landing causal relationship asked here.",
    133: "The graph records Chandrayaan-2's launcher and payloads, but not the first GSLV Mk III mission or its CARE payload; the nearby CY2 facts are not substituted.",
    150: "LAUNCHED_BY is controlled, but the graph has no human-rated LVM3–Gaganyaan edge.",
    153: "The graph has a CY2→IIRS payload edge and an IIRS mineralogy objective edge; this supports the relationship asked, but is not evidence for every extra spectral/detail claim in the reference.",
    156: "CY1 payload-membership and ESA developer edges support the mission–payload–developer pattern; this does not assert every international instrument in the reference is represented.",
    173: "PRECEDED_BY is controlled, but the only curated sequence edge is CY3 PRECEDED_BY CY2; the requested CY1→CY2 adjacency is absent.",
    178: "The only curated predecessor edge is CY3 PRECEDED_BY CY2, which is insufficient to count all earlier lunar missions.",
    181: "LAUNCHED_ON facts do not encode the CY3 landing event or elapsed interval asked here.",
    192: "The four launch-date edges can organize the chronology, but sorting dates is possible from text and does not require graph traversal.",
    194: "Both mission launch dates are represented; comparing their year is a date lookup/calculation, not a KG join requirement.",
}

QUESTION_TYPE_RATIONALE = {
    "DIRECT_FACT": "a single explicit fact or definition",
    "ATTRIBUTE": "one property of a named entity",
    "SINGLE_RELATION": "one explicit association between entities",
    "TWO_HOP_RELATION": "a linked two-edge chain between the mission, payload, and requested attribute/result",
    "MULTI_RELATION": "multiple facts/relationships, comparisons, or evidence items rather than one linked two-edge path",
    "OTHER": "does not fit the defined fact, attribute, or relationship categories",
}

RELEVANCE_RATIONALE = {
    "KG_NOT_RELEVANT": "The requested answer is a direct textual fact, an unmodeled relation, or a date/chronology calculation for which this KG's structure is not naturally needed.",
    "KG_POTENTIALLY_USEFUL": "The KG may organize related evidence, but the question is answerable from a passage or the required specific edge is missing/incomplete.",
    "KG_RELATIONALLY_NATURAL": "The wording asks for a relationship instantiated in the controlled KG, although that does not make KG use necessary.",
    "KG_MULTI_HOP": "The answer structure corresponds to a genuine join over two or more explicit, provenance-supported KG edges.",
}


def _qid_number(question_id: str) -> int:
    return int(question_id.rsplit("_", 1)[1])


def _question_type(number: int) -> str:
    if number in ATTRIBUTES:
        return "ATTRIBUTE"
    if number in SINGLE_RELATIONS:
        return "SINGLE_RELATION"
    if number in TWO_HOP_RELATIONS:
        return "TWO_HOP_RELATION"
    if number in MULTI_RELATIONS:
        return "MULTI_RELATION"
    return "DIRECT_FACT"


def _mission(question: str) -> str:
    aliases = [
        ("Chandrayaan-1", r"\bchandrayaan[\s-]*1\b"),
        ("Chandrayaan-2", r"\bchandrayaan[\s-]*2\b"),
        ("Chandrayaan-3", r"\bchandrayaan[\s-]*3\b"),
        ("Mars Orbiter Mission", r"\bmangalyaan\b|\bmars orbiter mission\b"),
        ("Aditya-L1", r"\baditya[\s-]*l1\b"),
        ("AstroSat", r"\bastrosat\b"),
        ("Gaganyaan", r"\bgaganyaan\b"),
    ]
    found = [name for name, pattern in aliases if re.search(pattern, question, re.I)]
    if not found:
        return "not mission-specific"
    if len(found) == 1:
        return found[0]
    return "multiple: " + "; ".join(found)


def _load_relations() -> set[str]:
    return {
        item["name"]
        for item in load_json(CORPUS / "relations.json")["relations"]
    }


def build_annotations(
    questions: list[dict[str, Any]],
    test_ids: set[str],
    triples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    triple_by_id = {triple["triple_id"]: triple for triple in triples}
    records = []
    for question in questions:
        question_id = question["id"]
        if question_id not in test_ids:
            continue
        number = _qid_number(question_id)
        question_type = _question_type(number)
        if number in MULTI_HOP_RELEVANT:
            relevance = "KG_MULTI_HOP"
        elif number in RELATIONALLY_NATURAL:
            relevance = "KG_RELATIONALLY_NATURAL"
        elif number in POTENTIALLY_USEFUL:
            relevance = "KG_POTENTIALLY_USEFUL"
        else:
            relevance = "KG_NOT_RELEVANT"
        required = "UNCERTAIN" if number in KG_REQUIRED_UNCERTAIN else "NO"
        relation_types = RELATIONS_BY_QUESTION.get(number, [])
        evidence_ids = EVIDENCE_BY_QUESTION.get(number, [])
        evidence = []
        for triple_id in evidence_ids:
            triple = triple_by_id.get(triple_id)
            if triple is None:
                raise ValueError(f"Question {question_id} cites unknown triple {triple_id!r}")
            provenance = triple.get("provenance", [])
            if not provenance:
                raise ValueError(f"Question {question_id} cites a triple without provenance: {triple_id}")
            source = provenance[0]
            evidence.append({
                "triple_id": triple_id,
                "subject": triple["subject"],
                "relation": triple["relation"],
                "object": triple["object"],
                "source_document": source["document_id"],
                "chunk_id": source["source_chunk_id"],
                "supporting_excerpt": source["supporting_excerpt"],
            })
        reasons = [
            f"Question wording asks for {QUESTION_TYPE_RATIONALE[question_type]}.",
            RELEVANCE_RATIONALE[relevance],
        ]
        if evidence:
            relations_found = sorted({item["relation"] for item in evidence})
            reasons.append(
                "Cited graph evidence contains provenance-backed relation(s): "
                + ", ".join(relations_found)
                + ". Evidence is a relevance anchor, not an assertion that every clause is supported."
            )
        elif relation_types:
            reasons.append(
                "The requested controlled relation type(s) are "
                + ", ".join(relation_types)
                + ", but no matching triple is cited for this question."
            )
        note = REASON_NOTES.get(number)
        if note:
            reasons.append(note)
        if required == "UNCERTAIN":
            reasons.append(
                "Whether a single consolidated passage suffices cannot be established from the available artifacts; "
                "the result file stores answers but no retrieved contexts."
            )
        elif required == "NO":
            reasons.append(
                "No system-performance outcome was used to assign this label. The annotation does not claim the KG is useless."
            )
        records.append({
            "question_id": question_id,
            "question": question["question"],
            "mission": _mission(question["question"]),
            "tier": question["tier"],
            "question_type": question_type,
            "kg_relevance": relevance,
            "kg_required": required,
            "kg_relation_types": relation_types,
            "reason": " ".join(reasons),
            "supporting_evidence": evidence,
        })
    return records


def _stored_test_rows(
    annotations: list[dict[str, Any]],
    benchmark: list[dict[str, Any]],
    test_ids: set[str],
    stored_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    annotation_by_id = {item["question_id"]: item for item in annotations}
    benchmark_by_id = {item["id"]: item for item in benchmark}
    stored_by_id = {item["id"]: item for item in stored_rows}
    if len(stored_by_id) != len(stored_rows):
        raise ValueError("Stored result rows contain duplicate question IDs")
    if set(annotation_by_id) != test_ids or set(test_ids) - set(stored_by_id):
        raise ValueError("Annotations/results do not exactly match the frozen test IDs")
    rows = []
    for question_id in sorted(test_ids):
        benchmark_item = benchmark_by_id[question_id]
        result_item = stored_by_id[question_id]
        reference = benchmark_item["answer"]
        if result_item.get("reference_answer") != reference:
            raise ValueError(f"Stored reference answer does not match benchmark for {question_id}")
        annotation = annotation_by_id[question_id]
        answers = {
            label: result_item.get(answer_key, "")
            for label, answer_key in SYSTEM_ANSWER_KEYS.items()
        }
        metrics = {}
        for system, answer in answers.items():
            metrics[system] = {
                "rouge_l": 0.0 if is_idk(answer) else rouge_l(answer, reference),
                "reference_token_coverage": 0.0 if is_idk(answer) else coverage(answer, reference),
                "exact_match": exact_match(answer, reference),
                "idk": float(is_idk(answer)),
            }
        rows.append({
            **annotation,
            "reference_answer": reference,
            "system_metrics": metrics,
            "system_answers": answers,
        })
    return rows


def _bootstrap_ci(differences: list[float]) -> list[float] | None:
    if not differences:
        return None
    import random

    rng = random.Random(BOOTSTRAP_SEED)
    n = len(differences)
    means = [
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    means.sort()

    def percentile(p: float) -> float:
        position = (len(means) - 1) * p
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return means[low]
        return means[low] * (high - position) + means[high] * (position - low)

    return [round(percentile(0.025), 6), round(percentile(0.975), 6)]


def _pair_summary(
    rows: list[dict[str, Any]],
    left: str,
    right: str,
    metric: str,
) -> dict[str, Any]:
    left_values = [row["system_metrics"][left][metric] for row in rows]
    right_values = [row["system_metrics"][right][metric] for row in rows]
    differences = [a - b for a, b in zip(left_values, right_values)]
    result: dict[str, Any] = {
        "n": len(differences),
        "metric": metric,
        "left_system": left,
        "right_system": right,
        "left_mean": round(mean(left_values), 6) if left_values else None,
        "left_median": round(median(left_values), 6) if left_values else None,
        "right_mean": round(mean(right_values), 6) if right_values else None,
        "right_median": round(median(right_values), 6) if right_values else None,
        "mean_paired_difference": round(mean(differences), 6) if differences else None,
        "median_paired_difference": round(median(differences), 6) if differences else None,
        "bootstrap_95_percent_ci": _bootstrap_ci(differences) if len(rows) >= MIN_INFERENTIAL_N else None,
        "wilcoxon_signed_rank_p": None,
        "cohen_dz": None,
        "rank_biserial": None,
        "inferential_status": (
            "descriptive_only_insufficient_sample_size"
            if len(rows) < MIN_INFERENTIAL_N else "inferential_test_available"
        ),
    }
    if len(rows) < MIN_INFERENTIAL_N or len([value for value in differences if value]) < 2:
        if len(rows) >= MIN_INFERENTIAL_N:
            result["inferential_status"] = "descriptive_only_fewer_than_two_nonzero_differences"
        return result
    from scipy.stats import rankdata, wilcoxon

    test = wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto")
    result["wilcoxon_signed_rank_p"] = float(test.pvalue)
    sd = stdev(differences)
    result["cohen_dz"] = round(mean(differences) / sd, 6) if sd else None
    nonzero = [value for value in differences if value]
    ranks = rankdata([abs(value) for value in nonzero], method="average")
    positive = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    negative = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
    result["rank_biserial"] = round((positive - negative) / sum(ranks), 6)
    return result


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {}
    for system in SYSTEM_ANSWER_KEYS:
        values = [row["system_metrics"][system] for row in rows]
        groups[system] = {
            "n": len(values),
            "rouge_l_mean": round(mean([x["rouge_l"] for x in values]), 6) if values else None,
            "rouge_l_median": round(median([x["rouge_l"] for x in values]), 6) if values else None,
            "reference_token_coverage_mean": round(
                mean([x["reference_token_coverage"] for x in values]), 6
            ) if values else None,
            "exact_match_mean": round(mean([x["exact_match"] for x in values]), 6) if values else None,
            "idk_rate": round(mean([x["idk"] for x in values]), 6) if values else None,
        }
    return {"n": len(rows), "systems": groups}


def _grouped_results(
    rows: list[dict[str, Any]],
    field: str,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    keys = labels or sorted({str(row[field]) for row in rows})
    result = {}
    for label in keys:
        subset = [row for row in rows if str(row[field]) == label]
        result[label] = _group_metrics(subset)
    return result


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (count - index) * p_value)
        running = max(running, candidate)
        adjusted[name] = round(running, 8)
    return adjusted


def _excerpt(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _comparison_group(
    rows: list[dict[str, Any]],
    group_name: str,
) -> dict[str, Any]:
    outputs = {
        "group": group_name,
        "n": len(rows),
        "minimum_inferential_n": MIN_INFERENTIAL_N,
        "status": (
            "descriptive_only_insufficient_sample_size"
            if len(rows) < MIN_INFERENTIAL_N else "descriptive_and_inferential_if_testable"
        ),
        "kg_rag_vs_vanilla_rag": _pair_summary(rows, "KG-RAG", "Vanilla RAG", "rouge_l"),
        "kg_rag_vs_bm25_llm": _pair_summary(rows, "KG-RAG", "BM25 + LLM", "rouge_l"),
    }
    return outputs


def _assert_overall_parity(rows: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    systems_to_eval = {
        "BM25 + LLM": "bm25_llm",
        "Vanilla RAG": "vanilla_rag",
        "KG-RAG": "kg_rag",
    }
    checks = {}
    for system, eval_name in systems_to_eval.items():
        for metric, key in (
            ("rouge_l", "rouge_l"),
            ("reference_token_coverage", "reference_token_coverage"),
            ("exact_match", "exact_match"),
            ("idk", "idk_rate"),
        ):
            values = [row["system_metrics"][system][metric] for row in rows]
            calculated = round(mean(values), 4)
            expected = evaluation["overall"]["systems"][eval_name][key]
            checks[f"{system}:{metric}"] = {
                "calculated": calculated,
                "stored": expected,
                "match": calculated == expected,
            }
    failed = [name for name, check in checks.items() if not check["match"]]
    if failed:
        raise ValueError(f"Stored per-question outputs do not reproduce aggregate metrics: {failed}")
    return checks


def _failure_pattern_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    relational_rows = [
        row for row in rows
        if row["kg_relevance"] in {"KG_RELATIONALLY_NATURAL", "KG_MULTI_HOP"}
    ]
    identical = [
        row for row in relational_rows
        if row["system_answers"]["KG-RAG"] == row["system_answers"]["Vanilla RAG"]
    ]
    kg_idk = sum(row["system_metrics"]["KG-RAG"]["idk"] for row in relational_rows)
    vanilla_idk = sum(row["system_metrics"]["Vanilla RAG"]["idk"] for row in relational_rows)
    examples = []
    for question_id in ("isro_071", "isro_098", "isro_130", "isro_153", "isro_156"):
        row = next((item for item in relational_rows if item["question_id"] == question_id), None)
        if row is None:
            continue
        examples.append({
            "question_id": question_id,
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            "kg_rag_answer_excerpt": _excerpt(row["system_answers"]["KG-RAG"], 280),
            "vanilla_rag_answer_excerpt": _excerpt(row["system_answers"]["Vanilla RAG"], 280),
            "kg_rag_rouge_l": row["system_metrics"]["KG-RAG"]["rouge_l"],
            "vanilla_rag_rouge_l": row["system_metrics"]["Vanilla RAG"]["rouge_l"],
            "interpretation_limit": "Lexical/output example only; not an adjudication of factual correctness or retrieval cause.",
        })
    return {
        "relational_subset_n": len(relational_rows),
        "identical_kg_and_vanilla_answer_text_count": len(identical),
        "kg_rag_idk_count": int(kg_idk),
        "vanilla_rag_idk_count": int(vanilla_idk),
        "answer_examples": examples,
        "retrieval_contexts_available": False,
    }


def analyze() -> dict[str, Any]:
    benchmark = load_json(BENCHMARK_PATH)
    test_ids = set(load_json(TEST_IDS_PATH))
    stored_rows = load_json(BASELINE_PATH)
    evaluation = load_json(RESULTS_DIR / "evaluation_results.json")
    relation_names = _load_relations()
    triples = [
        json.loads(line)
        for line in (CORPUS / "triples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    questions = [item for item in benchmark if item["id"] in test_ids]
    if len(questions) != 180 or len(test_ids) != 180:
        raise ValueError(f"Expected the frozen 180-question test split; got {len(questions)}")
    annotations = build_annotations(questions, test_ids, triples)
    for record in annotations:
        if not set(record["kg_relation_types"]) <= relation_names:
            raise ValueError(f"Unknown relation type in {record['question_id']}")
        if record["kg_required"] == "YES" and (
            not record["kg_relation_types"] or not record["supporting_evidence"]
        ):
            raise ValueError(f"KG_REQUIRED=YES needs relation/evidence for {record['question_id']}")
    ANNOTATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATION_PATH.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in annotations),
        encoding="utf-8",
    )
    rows = _stored_test_rows(annotations, benchmark, test_ids, stored_rows)
    parity = _assert_overall_parity(rows, evaluation)
    relevance_labels = [
        "KG_NOT_RELEVANT",
        "KG_POTENTIALLY_USEFUL",
        "KG_RELATIONALLY_NATURAL",
        "KG_MULTI_HOP",
    ]
    required_labels = ["YES", "NO", "UNCERTAIN"]
    groups = {
        "overall": _group_metrics(rows),
        "kg_relevance": _grouped_results(rows, "kg_relevance", relevance_labels),
        "kg_required": _grouped_results(rows, "kg_required", required_labels),
        "question_type": _grouped_results(rows, "question_type", sorted(QUESTION_TYPES)),
        "tier": _grouped_results(rows, "tier", ["1", "2", "3"]),
        "mission": _grouped_results(rows, "mission"),
    }
    yes_rows = [row for row in rows if row["kg_required"] == "YES"]
    no_rows = [row for row in rows if row["kg_required"] == "NO"]
    relation_rows = [
        row for row in rows
        if row["kg_relevance"] in {"KG_RELATIONALLY_NATURAL", "KG_MULTI_HOP"}
    ]
    primary = _comparison_group(yes_rows, "KG_REQUIRED=YES")
    secondary_no = _comparison_group(no_rows, "KG_REQUIRED=NO")
    relational = _comparison_group(
        relation_rows, "KG_RELATIONALLY_NATURAL + KG_MULTI_HOP"
    )
    exploratory = {
        "KG_REQUIRED=NO:KG-RAG_vs_Vanilla": secondary_no["kg_rag_vs_vanilla_rag"],
        "KG_REQUIRED=NO:KG-RAG_vs_BM25": secondary_no["kg_rag_vs_bm25_llm"],
        "RELATIONAL:KG-RAG_vs_Vanilla": relational["kg_rag_vs_vanilla_rag"],
        "RELATIONAL:KG-RAG_vs_BM25": relational["kg_rag_vs_bm25_llm"],
    }
    raw_p = {
        name: summary["wilcoxon_signed_rank_p"]
        for name, summary in exploratory.items()
        if summary["wilcoxon_signed_rank_p"] is not None
    }
    adjusted = _holm_adjust(raw_p)
    for name, p_value in adjusted.items():
        exploratory[name]["holm_adjusted_p"] = p_value
    for comparison in (secondary_no, relational):
        for result_key in ("kg_rag_vs_vanilla_rag", "kg_rag_vs_bm25_llm"):
            summary = comparison[result_key]
            name = comparison["group"] + ":" + (
                "KG-RAG_vs_Vanilla" if result_key.endswith("vanilla_rag") else "KG-RAG_vs_BM25"
            )
            if name in adjusted:
                summary["holm_adjusted_p"] = adjusted[name]

    joined = []
    for row in rows:
        joined.append({
            "question_id": row["question_id"],
            "mission": row["mission"],
            "tier": row["tier"],
            "question_type": row["question_type"],
            "kg_relevance": row["kg_relevance"],
            "kg_required": row["kg_required"],
            "system_metrics": row["system_metrics"],
        })
    output = {
        "phase": "Phase 2 — Frozen 180-Question KG-Relevance Analysis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "question_bank": "data/benchmark/isro_qa.json",
            "test_split": "data/benchmark/test_ids.json",
            "total_question_bank_items": len(benchmark),
            "development_items": len(benchmark) - len(test_ids),
            "frozen_test_questions": len(questions),
            "test_ids": sorted(test_ids),
            "frozen_outputs": "data/results/baseline_results.json",
            "aggregate_crosscheck": parity,
            "retrieval_contexts_available": False,
            "stored_artifact_note": (
                "The per-question artifact contains answer strings but no retrieved passages, "
                "chunk IDs, KG context, or retrieval traces."
            ),
        },
        "annotation_counts": {
            "question_type": dict(Counter(row["question_type"] for row in annotations)),
            "kg_relevance": dict(Counter(row["kg_relevance"] for row in annotations)),
            "kg_required": dict(Counter(row["kg_required"] for row in annotations)),
        },
        "groups": groups,
        "primary_comparison": primary,
        "secondary_required_no": secondary_no,
        "relational_subset": relational,
        "secondary_multiple_comparison_correction": {
            "method": "Holm-Bonferroni over testable exploratory subgroup Wilcoxon comparisons",
            "family": sorted(raw_p),
            "adjusted_p_values": adjusted,
        },
        "failure_pattern_audit": _failure_pattern_audit(rows),
        "decision_gate": {
            "decision": "D. FROZEN BENCHMARK INSUFFICIENT",
            "rationale": (
                "The strict KG_REQUIRED=YES subset is empty in this frozen annotation. "
                "No inferential comparison can test the primary conditional-benefit hypothesis. "
                "Relationally natural/multi-hop items are reported descriptively or as exploratory "
                "depending on N; their wording alone does not establish that a KG join is required."
            ),
        },
        "per_question_metrics": joined,
        "per_question_annotation_file": "data/annotations/canonical_180_kg_relevance.jsonl",
        "scoring_note": (
            "ROUGE-L, reference-token coverage, exact match, and IDK are recomputed from stored "
            "answers against frozen reference answers; no LLM or QA benchmark was rerun."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(output, rows), encoding="utf-8")
    return output


def _render_comparison(comparison: dict[str, Any]) -> str:
    vanilla = comparison["kg_rag_vs_vanilla_rag"]
    bm25 = comparison["kg_rag_vs_bm25_llm"]
    if comparison["n"] < MIN_INFERENTIAL_N:
        status = "Descriptive only; insufficient sample size for reliable inferential comparison."
    else:
        status = "Exploratory inference only; interpret with the stated Holm correction."
    vanilla_p = (
        f"raw p {vanilla['wilcoxon_signed_rank_p']}; "
        f"Holm-adjusted p {vanilla.get('holm_adjusted_p')}"
    )
    bm25_p = (
        f"raw p {bm25['wilcoxon_signed_rank_p']}; "
        f"Holm-adjusted p {bm25.get('holm_adjusted_p')}"
    )
    return (
        f"N={comparison['n']}. {status}\n\n"
        f"- KG-RAG vs Vanilla RAG: mean ROUGE-L {vanilla['left_mean']}; "
        f"median {vanilla['left_median']}; Vanilla mean {vanilla['right_mean']}; "
        f"median {vanilla['right_median']}; mean paired difference "
        f"{vanilla['mean_paired_difference']}; median difference "
        f"{vanilla['median_paired_difference']}; 95% bootstrap CI "
        f"{vanilla['bootstrap_95_percent_ci']}; Wilcoxon {vanilla_p}; "
        f"Cohen's dz {vanilla['cohen_dz']}.\n"
        f"- KG-RAG vs BM25 + LLM (secondary): mean paired difference "
        f"{bm25['mean_paired_difference']}; 95% bootstrap CI "
        f"{bm25['bootstrap_95_percent_ci']}; Wilcoxon {bm25_p}."
    )


def render_report(
    output: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    def fmt(value: float | int | None) -> str:
        return f"{value:.4f}" if value is not None else "—"

    counts = output["annotation_counts"]
    all_systems = output["groups"]["overall"]["systems"]
    relevant_ids = {
        row["question_id"] for row in rows
        if row["kg_relevance"] in {"KG_RELATIONALLY_NATURAL", "KG_MULTI_HOP"}
    }
    relevant_rows = [row for row in rows if row["question_id"] in relevant_ids]
    misses = Counter(
        row["kg_relevance"] for row in relevant_rows if not row["supporting_evidence"]
    )
    tier_table = [
        f"| Tier {tier} | {data['n']} | "
        + " | ".join(
            fmt(data["systems"][system][metric])
            for system in ("BM25 + LLM", "Vanilla RAG", "KG-RAG")
            for metric in ("rouge_l_mean",)
        )
        + " |"
        for tier, data in output["groups"]["tier"].items()
    ]
    mission_rows = []
    for mission, data in output["groups"]["mission"].items():
        if data["n"] < 5:
            continue
        mission_rows.append(
            f"| {mission} | {data['n']} | "
            + " | ".join(
                fmt(data["systems"][system]["rouge_l_mean"])
                for system in ("BM25 + LLM", "Vanilla RAG", "KG-RAG")
            ) + " |"
        )
    low_kg_subset = sorted(
        (row for row in relevant_rows),
        key=lambda row: row["system_metrics"]["KG-RAG"]["rouge_l"]
        - row["system_metrics"]["Vanilla RAG"]["rouge_l"],
    )[:5]
    low_kg_text = "\n".join(
        f"- `{row['question_id']}` ({row['kg_relevance']}): "
        f"KG-RAG {row['system_metrics']['KG-RAG']['rouge_l']:.4f}, "
        f"Vanilla {row['system_metrics']['Vanilla RAG']['rouge_l']:.4f}. "
        f"Question: {row['question']}"
        for row in low_kg_subset
    ) or "- No KG-relationally-natural or KG-multi-hop questions."
    failure_audit = output["failure_pattern_audit"]
    output_example_text = "\n".join(
        f"- `{item['question_id']}`: KG-RAG ROUGE-L {item['kg_rag_rouge_l']:.4f}, "
        f"Vanilla {item['vanilla_rag_rouge_l']:.4f}. "
        f"Reference: “{_excerpt(item['reference_answer'], 160)}” "
        f"KG-RAG: “{_excerpt(item['kg_rag_answer_excerpt'], 160)}” "
        f"Vanilla: “{_excerpt(item['vanilla_rag_answer_excerpt'], 160)}”"
        for item in failure_audit["answer_examples"]
    ) or "- No answer examples available."
    overall_table = "\n".join(
        f"| {system} | {item['n']} | {item['rouge_l_mean']:.4f} | "
        f"{item['rouge_l_median']:.4f} | {item['reference_token_coverage_mean']:.4f} | "
        f"{item['exact_match_mean']:.4f} | {item['idk_rate']:.4f} |"
        for system, item in all_systems.items()
    )
    relevant_table = "\n".join(
        f"| {label} | {item['n']} | "
        + " | ".join(
            fmt(item["systems"][system]["rouge_l_mean"])
            for system in ("BM25 + LLM", "Vanilla RAG", "KG-RAG")
        ) + " |"
        for label, item in output["groups"]["kg_relevance"].items()
    )
    required_table = "\n".join(
        f"| {label} | {item['n']} | "
        + " | ".join(
            fmt(item["systems"][system]["rouge_l_mean"])
            for system in ("BM25 + LLM", "Vanilla RAG", "KG-RAG")
        ) + " |"
        for label, item in output["groups"]["kg_required"].items()
    )
    question_type_table = "\n".join(
        f"| {label} | {item['n']} | "
        + " | ".join(
            fmt(item["systems"][system]["rouge_l_mean"])
            for system in ("BM25 + LLM", "Vanilla RAG", "KG-RAG")
        ) + " |"
        for label, item in output["groups"]["question_type"].items()
    )
    return f"""# Frozen canonical benchmark KG-relevance analysis

## 1. Objective

Assess whether the frozen canonical QA set contains a sufficiently large, evidence-supported subset whose question structure naturally corresponds to the frozen Phase 1.6 KG. This is an analysis of existing answers; it is not a new benchmark run and is not a universal RAG-vs-KG-RAG comparison.

## 2. Frozen benchmark definition

- Source bank: `{output['benchmark']['question_bank']}` ({output['benchmark']['total_question_bank_items']} total records; {output['benchmark']['development_items']} development records).
- Evaluated split: `{output['benchmark']['test_split']}`; exactly {output['benchmark']['frozen_test_questions']} frozen test IDs.
- Per-question saved output source: `{output['benchmark']['frozen_outputs']}`.
- Filtered output rows: {len(rows)}; every test question and reference answer was matched by ID.
- Existing aggregate parity: all stored BM25 + LLM, Vanilla RAG, and KG-RAG ROUGE-L, reference-token coverage, exact-match, and IDK metrics were reproduced from the saved answer strings.
- The benchmark item itself has no `mission` field; annotation mission scope is extracted from the question text only. Questions that name multiple missions retain a `multiple: ...` value; unrelated questions are `not mission-specific`.
- The targeted 36-question Aditya-L1 stress test was not merged or analyzed.

## 3–6. Annotation methodology and taxonomy

Each test question was assigned one primary type, one KG-relevance label, and one KG-required label from question wording and the frozen relation vocabulary/triples. System output and score were not used to assign labels. See `data/annotations/canonical_180_kg_relevance_guidelines.md` for detailed rules and examples.

The annotation is a single-researcher classification, not inter-annotator agreement. Multi-relation means combining evidence items; two-hop is reserved for a linked mission→payload→attribute/target chain. A relationship-looking question is not labeled KG-multi-hop merely because it mentions two entities.

## 7. Annotation counts

**Question type**

| Type | N |
|---|---:|
{chr(10).join(f"| {label} | {counts['question_type'].get(label, 0)} |" for label in ("DIRECT_FACT", "ATTRIBUTE", "SINGLE_RELATION", "TWO_HOP_RELATION", "MULTI_RELATION", "OTHER"))}

**KG relevance**

| Label | N |
|---|---:|
{chr(10).join(f"| {label} | {counts['kg_relevance'].get(label, 0)} |" for label in ("KG_NOT_RELEVANT", "KG_POTENTIALLY_USEFUL", "KG_RELATIONALLY_NATURAL", "KG_MULTI_HOP"))}

**KG required**

| Label | N |
|---|---:|
{chr(10).join(f"| {label} | {counts['kg_required'].get(label, 0)} |" for label in ("YES", "NO", "UNCERTAIN"))}

## 8. Overall saved-output results

| System | N | Mean ROUGE-L | Median ROUGE-L | Mean ref-token coverage | Exact match | IDK rate |
|---|---:|---:|---:|---:|---:|---:|
{overall_table}

These reproduce, after rounding, the frozen aggregate artifact: BM25 + LLM ROUGE-L {all_systems['BM25 + LLM']['rouge_l_mean']:.4f}, coverage {all_systems['BM25 + LLM']['reference_token_coverage_mean']:.4f}, exact match {all_systems['BM25 + LLM']['exact_match_mean']:.4f}, IDK {all_systems['BM25 + LLM']['idk_rate']:.4f}; Vanilla RAG ROUGE-L {all_systems['Vanilla RAG']['rouge_l_mean']:.4f}, coverage {all_systems['Vanilla RAG']['reference_token_coverage_mean']:.4f}, exact match {all_systems['Vanilla RAG']['exact_match_mean']:.4f}, IDK {all_systems['Vanilla RAG']['idk_rate']:.4f}; KG-RAG ROUGE-L {all_systems['KG-RAG']['rouge_l_mean']:.4f}, coverage {all_systems['KG-RAG']['reference_token_coverage_mean']:.4f}, exact match {all_systems['KG-RAG']['exact_match_mean']:.4f}, IDK {all_systems['KG-RAG']['idk_rate']:.4f}.

## 9. KG-relevance groups

| KG relevance | N | BM25 + LLM mean ROUGE-L | Vanilla mean ROUGE-L | KG-RAG mean ROUGE-L |
|---|---:|---:|---:|---:|
{relevant_table}

The separate KG_RELATIONALLY_NATURAL (N={counts['kg_relevance'].get('KG_RELATIONALLY_NATURAL', 0)}) and KG_MULTI_HOP (N={counts['kg_relevance'].get('KG_MULTI_HOP', 0)}) groups are each below N={MIN_INFERENTIAL_N}; interpret them descriptively. The combined relational subset N={output['relational_subset']['n']} is exploratory and does not replace the primary KG_REQUIRED=YES test.

## 10. KG-required vs not required

| KG required | N | BM25 + LLM mean ROUGE-L | Vanilla mean ROUGE-L | KG-RAG mean ROUGE-L |
|---|---:|---:|---:|---:|
{required_table}

## 11. Question-type results

| Question type | N | BM25 + LLM mean ROUGE-L | Vanilla mean ROUGE-L | KG-RAG mean ROUGE-L |
|---|---:|---:|---:|---:|
{question_type_table}

## 12. Tier and mission results

| Tier | N | BM25 + LLM mean ROUGE-L | Vanilla mean ROUGE-L | KG-RAG mean ROUGE-L |
|---|---:|---:|---:|---:|
{chr(10).join(tier_table)}

Mission groups with N≥5:

| Mission label | N | BM25 + LLM | Vanilla RAG | KG-RAG |
|---|---:|---:|---:|---:|
{chr(10).join(mission_rows)}

Smaller mission groups remain in the JSON as descriptive counts and scores; they are not interpreted inferentially.

The primary types `TWO_HOP_RELATION` (N={counts['question_type'].get('TWO_HOP_RELATION', 0)}) and `OTHER` (N={counts['question_type'].get('OTHER', 0)}) are also shown with exact N. No separate hypothesis tests are run for taxonomy/tier/mission slices.

## 13–15. Primary comparison, secondary tests, effect sizes, and intervals

### Primary: KG-RAG vs Vanilla RAG on KG_REQUIRED=YES

{_render_comparison(output['primary_comparison'])}

The subset has {output['primary_comparison']['n']} items; this is below the predeclared minimum N={MIN_INFERENTIAL_N}. No Wilcoxon test, effect-size claim, or bootstrap interval is used to make an inferential claim for this group.

### Secondary: KG_REQUIRED=NO

{_render_comparison(output['secondary_required_no'])}

### Relational subset: KG_RELATIONALLY_NATURAL + KG_MULTI_HOP

{_render_comparison(output['relational_subset'])}

Subgroup p-values are exploratory and Holm-adjusted over the testable secondary comparisons in the JSON. The absent/empty primary YES group is not replaced by a more favorable subgroup. Cohen's dz is mean paired difference divided by sample SD; rank-biserial correlation is also included in the machine-readable summaries.

## 16. Failure-pattern / evidence audit

- The stored canonical per-question artifact contains answer strings, but no retrieved chunks, retrieval contexts, graph-expansion traces, or evidence ranking. Therefore it cannot establish whether KG retrieved useful evidence, whether dense RAG retrieved the same evidence, or whether context became noisy.
- On the {failure_audit['relational_subset_n']} relationally natural/multi-hop items, KG-RAG and Vanilla RAG return identical answer strings for {failure_audit['identical_kg_and_vanilla_answer_text_count']} items; stored IDK counts are {failure_audit['kg_rag_idk_count']} for KG-RAG and {failure_audit['vanilla_rag_idk_count']} for Vanilla RAG. Similarity does not identify the retrieval cause.
- The graph-specific annotation found {misses.get('KG_RELATIONALLY_NATURAL', 0) + misses.get('KG_MULTI_HOP', 0)} relationally natural/multi-hop questions without a cited matching edge set (see per-question reasons). Some cited paths are partial: a payload objective is not a mission result, and an observation target edge is not proof that a historical discovery occurred.
- The five most unfavorable paired ROUGE-L differences in the relationally natural/multi-hop subset are listed below as lexical-score examples only. They are not factual-error judgments.

{low_kg_text}

Selected stored answer-text examples (output inspection only):

These fixed illustrative examples are not a random sample or an exhaustive factuality audit; displayed strings are truncated.

{output_example_text}

ROUGE-L and token coverage measure lexical overlap, not factual correctness. No hallucination, faithfulness, or factuality conclusion is drawn.

## 17–18. Interpretation and limitations

The frozen benchmark has {counts['kg_relevance'].get('KG_RELATIONALLY_NATURAL', 0) + counts['kg_relevance'].get('KG_MULTI_HOP', 0)} KG-relationally-natural/multi-hop questions, but **{counts['kg_required'].get('YES', 0)}** meet the strict KG_REQUIRED=YES rule. Thus it cannot test the primary conditional-benefit hypothesis. The analysis uses one annotator; the benchmark's mission field was absent and conservatively derived from question text; graph coverage is incomplete for some benchmark claims; retrieval contexts are unavailable; and automatic lexical metrics do not adjudicate truth.

No questions were added, removed, or rewritten. The 36-question Aditya-L1 evaluation remains separate. No new QA benchmark was run.

## 19. Decision for next phase

**D. FROZEN BENCHMARK INSUFFICIENT**

The primary `KG_REQUIRED=YES` subset is empty. Do not use observed outcomes to relabel questions or revise the canonical benchmark. A future hypothesis test requires a separately approved protocol and independently designed relational benchmark; no such work is started here.
"""


def main() -> None:
    data = analyze()
    print(f"Saved {ANNOTATION_PATH}")
    print(f"Saved {OUTPUT_PATH}")
    print(f"Saved {REPORT_PATH}")
    print(f"Frozen test questions: {data['benchmark']['frozen_test_questions']}")
    print(f"KG_REQUIRED=YES: {data['annotation_counts']['kg_required'].get('YES', 0)}")
    print(f"Decision: {data['decision_gate']['decision']}")


if __name__ == "__main__":
    main()
