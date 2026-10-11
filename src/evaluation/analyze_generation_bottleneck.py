"""Diagnostic audit of evidence composition in the frozen relation-aware run."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "results" / "relation_aware"
GENERATIONS = RESULTS / "generation_results.jsonl"
BENCHMARK = ROOT / "data" / "relation_aware_benchmark" / "relation_aware_qa_v2.json"
SYSTEMS = (
    "vanilla_dense_rag",
    "relation_aware_kg_rag",
    "relation_aware_kg_only",
    "relation_aware_kg_source",
    "relation_aware_kg_dense",
    "relation_aware_kg_dense_bm25",
)
HYBRIDS = SYSTEMS[1:]


def norm(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").casefold()))


def words(text: str) -> list[str]:
    return re.findall(r"\S+", text or "")


def block(context: str, heading: str, next_headings: tuple[str, ...]) -> str:
    start = context.find(heading)
    if start < 0:
        return ""
    end = len(context)
    for candidate in next_headings:
        position = context.find(candidate, start + len(heading))
        if position >= 0:
            end = min(end, position)
    return context[start:end]


def composition(row: dict) -> dict:
    context = row["context"]
    kg = block(context, "[RELATION-AWARE KG EVIDENCE]", ("[DENSE EVIDENCE]", "[BM25 EVIDENCE]"))
    dense = block(context, "[DENSE EVIDENCE]", ("[BM25 EVIDENCE]",))
    bm25 = block(context, "[BM25 EVIDENCE]", ())
    source_count = len(re.findall(r"SOURCE TEXT:", kg))
    kg_tokens = len(words(kg))
    dense_tokens = len(words(dense))
    bm25_tokens = len(words(bm25))
    non_kg_tokens = dense_tokens + bm25_tokens
    total = len(words(context))
    kg_position = context.find("[RELATION-AWARE KG EVIDENCE]")
    overlap = bool(norm(kg) & norm(dense)) if kg and dense else None
    return {
        "context_tokens": total,
        "kg_path_count": len(row.get("selected_evidence", [])) if kg else 0,
        "dense_item_count": len([x for x in dense.split("SOURCE TEXT:") if x.strip()]) if dense else 0,
        "bm25_item_count": len([x for x in bm25.split("SOURCE TEXT:") if x.strip()]) if bm25 else 0,
        "source_evidence_count": source_count,
        "provenance_count": len(row.get("provenance_ids", [])) if kg else 0,
        "kg_tokens": kg_tokens,
        "non_kg_tokens": non_kg_tokens,
        "kg_proportion": kg_tokens / total if total else None,
        "kg_position_fraction": kg_position / len(context) if kg_position >= 0 and context else None,
        "kg_dense_token_overlap": overlap,
        "has_kg_block": bool(kg),
        "has_dense_block": bool(dense),
        "has_bm25_block": bool(bm25),
        "answer_words": len(words(row.get("answer", ""))),
        "idk": bool(row.get("metrics", {}).get("idk")),
    }


def spearman(x: list[float], y: list[float]) -> dict:
    try:
        from scipy.stats import spearmanr

        result = spearmanr(x, y)
        return {"rho": float(result.statistic), "p_value": float(result.pvalue)}
    except Exception as exc:
        return {"rho": None, "p_value": None, "error": str(exc)}


def paired_gap(rows: dict[tuple[str, str], dict], qids: list[str], system: str, metric: str) -> list[dict]:
    result = []
    for qid in qids:
        kg = rows[(qid, "relation_aware_kg_only")]
        current = rows[(qid, system)]
        result.append({
            "question_id": qid,
            "question": current["question"],
            "reference_answer": current["reference_answer"],
            "system": system,
            "rouge_l": current["metrics"]["rouge_l"],
            "coverage": current["metrics"]["reference_token_coverage"],
            "kg_only_rouge_l": kg["metrics"]["rouge_l"],
            "kg_only_coverage": kg["metrics"]["reference_token_coverage"],
            "delta_rouge_l": current["metrics"]["rouge_l"] - kg["metrics"]["rouge_l"],
            "delta_coverage": current["metrics"]["reference_token_coverage"] - kg["metrics"]["reference_token_coverage"],
            "context": composition(current),
            "kg_paths": [x["path_id"] for x in current.get("selected_evidence", [])[:5]],
            "provenance": current.get("provenance_ids", [])[:5],
            "answer": current["answer"],
        })
    return result


def metric_sanity(questions: list[dict], rows: dict[tuple[str, str], dict]) -> dict:
    qmap = {q["question_id"]: q for q in questions}
    values = []
    for (qid, system), item in rows.items():
        if system not in HYBRIDS + ("vanilla_dense_rag",):
            continue
        reference = item["reference_answer"]
        evidence_text = " ".join(
            source for evidence in item.get("selected_evidence", [])
            for source in evidence.get("source_text", [])
        )
        values.append({
            "question_id": qid,
            "system": system,
            "reference_in_kg_source_text": reference.casefold() in evidence_text.casefold(),
            "reference_token_overlap_with_kg": len(norm(reference) & norm(evidence_text)) / len(norm(reference)) if norm(reference) else 0,
            "reference_words": len(words(reference)),
            "answer_words": len(words(item["answer"])),
            "answer_reference_overlap": len(norm(item["answer"]) & norm(reference)) / len(norm(reference)) if norm(reference) else 0,
            "answer_context_overlap": len(norm(item["answer"]) & norm(item["context"])) / len(norm(item["answer"])) if norm(item["answer"]) else 0,
            "supporting_triples": qmap[qid].get("supporting_triples", []),
            "reference_answer": reference,
        })
    return {
        "rows": values,
        "reference_answers_come_from_canonical_triples": True,
        "assessment": "The benchmark references are canonical triple objects, and the relation-aware KG retrieves those same canonical triples. This is legitimate evidence grounding for a KG-supported benchmark, but it makes lexical metrics sensitive to KG/reference wording overlap and is not independent human factual validation.",
    }


def main() -> None:
    questions = json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]
    raw = [json.loads(line) for line in GENERATIONS.read_text(encoding="utf8").splitlines() if line.strip()]
    rows = {(row["question_id"], row["system"]): row for row in raw}
    required = [q for q in questions if q["kg_required"] == "YES"]
    qids = [q["question_id"] for q in required]
    composition_rows = []
    for qid in qids:
        for system in SYSTEMS:
            row = rows[(qid, system)]
            item = {"question_id": qid, "system": system, **composition(row)}
            item["rouge_l"] = row["metrics"]["rouge_l"]
            item["coverage"] = row["metrics"]["reference_token_coverage"]
            composition_rows.append(item)
    correlations = {}
    for system in HYBRIDS:
        subset = [x for x in composition_rows if x["system"] == system and x["kg_proportion"] is not None]
        correlations[system] = {
            feature: {
                target: spearman([x[feature] for x in subset], [x[target] for x in subset])
                for target in ("rouge_l", "coverage")
            }
            for feature in ("kg_proportion", "kg_tokens", "non_kg_tokens", "context_tokens")
        }
    gaps = {}
    for system in HYBRIDS:
        values = paired_gap(rows, qids, system, "rouge_l")
        gaps[system] = sorted(values, key=lambda x: abs(x["delta_rouge_l"]), reverse=True)[:5]
    prompt = rows[(qids[0], "relation_aware_kg_rag")]["final_prompt"]
    prompt_audit = {
        "system_prompt": "You are a factual QA assistant. You must answer ONLY using the context provided below. Do not use any prior knowledge. Do not be conversational. If the answer is not in the context, respond with exactly: I don't know. Never say anything else if the answer is not found.",
        "user_prompt_template": "CONTEXT:\\n<context>\\n\\nQUESTION:\\n<question>\\n\\nANSWER (based only on the context above):",
        "observed_prompt_example_prefix": prompt[:500],
        "explicit_kg_priority": False,
        "explicit_conflict_resolution": False,
        "explicit_relation_direction_instruction": False,
        "explicit_provenance_instruction": False,
        "context_blocks_are_labeled": True,
        "weaknesses": [
            "The system prompt says to use the context but does not prioritize KG facts over other evidence.",
            "It does not define how to resolve conflicting evidence.",
            "It does not explicitly preserve relation direction or use provenance labels.",
            "The user prompt presents one context string rather than a structured instruction hierarchy.",
        ],
    }
    audit = {
        "key_observation": {
            "kg_only_rouge_l": 0.417537,
            "hybrid_relation_aware_rouge_l": 0.123176,
            "kg_only_coverage": 0.810747,
            "hybrid_relation_aware_coverage": 0.315075,
        },
        "trace_integrity": {"rows": len(raw), "required_questions": len(required), "systems": sorted({r["system"] for r in raw})},
        "evidence_composition": composition_rows,
        "hybrid_correlations": correlations,
        "largest_per_question_gaps": gaps,
        "metric_sanity": metric_sanity(questions, rows),
        "prompt_audit": prompt_audit,
        "implementation_note": "The completed runner's relation_aware_kg_source context path also enables dense evidence, while relation_aware_kg_dense does the same; this label/context overlap is recorded rather than corrected after the frozen run.",
        "critical_comparison": {
            "most_supported_explanation": "F. MIXED/UNCERTAIN, with prompt failure and evidence dilution plausible.",
            "reasoning": "KG-only has much higher lexical scores and the hybrid prompt does not prioritize KG facts. The traces do not provide a reliable semantic conflict label, and context size/path count alone cannot establish dilution or conflict. The benchmark references are canonical KG objects, which can amplify lexical overlap.",
            "not_supported": ["A definitive evidence-conflict conclusion", "A causal context-length conclusion", "A factual-superiority conclusion"],
        },
        "recommendation": "Run a small preregistered prompt-only diagnostic with frozen retrieval and deterministic KG-required questions before any new large evaluation.",
    }
    (RESULTS / "generation_bottleneck_analysis.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf8")
    with (RESULTS / "evidence_composition.csv").open("w", newline="", encoding="utf8") as handle:
        fields = list(composition_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(composition_rows)
    (RESULTS / "bottleneck_manifest.json").write_text(json.dumps({
        "inputs": {
            "generation_results_sha256": hashlib.sha256(GENERATIONS.read_bytes()).hexdigest(),
            "benchmark_sha256": hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(),
        },
        "required_questions": len(required),
        "analysis_only": True,
        "generations_rerun": False,
        "frozen_retrieval_modified": False,
        "conclusion": "F. MIXED/UNCERTAIN",
    }, indent=2) + "\n", encoding="utf8")
    lines = [
        "# Generation bottleneck diagnostic",
        "",
        "## Key observation",
        "",
        "KG-only substantially exceeds hybrid lexical scores, but these are automated lexical diagnostics rather than factual-quality evidence.",
        "",
        "## Trace audit",
        "",
        f"- Audited {len(raw)} frozen rows and {len(required)} KG-required questions.",
        "- No answers were regenerated.",
        "- Evidence composition fields are marked unavailable when a context block is absent.",
        "",
        "## Current prompt audit",
        "",
        "The current prompt requires context-only answering, but does not prioritize KG facts, define conflict resolution, preserve relation direction explicitly, or instruct provenance use.",
        "",
        "## Metric sanity check",
        "",
        "Reference answers are canonical triple objects, and KG-required paths point to those same canonical triples. This is legitimate for evidence grounding, but it creates direct lexical overlap between reference answers and KG evidence. It can inflate lexical diagnostics and does not replace human evaluation.",
        "",
        "## Most-supported explanation",
        "",
        "**F. MIXED/UNCERTAIN**",
        "",
        "Prompt failure and evidence dilution are plausible. The frozen traces do not establish whether heterogeneous evidence conflicts semantically, and the benchmark/KG lexical relationship is a confound for ROUGE and coverage. A small prompt-only diagnostic is recommended.",
        "",
        "Detailed composition, correlations, per-question gap examples, and prompt audit are in `generation_bottleneck_analysis.json` and `evidence_composition.csv`.",
    ]
    (RESULTS / "generation_bottleneck_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
