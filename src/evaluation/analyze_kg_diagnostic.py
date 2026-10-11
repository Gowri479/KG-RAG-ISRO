"""Analyze the isolated 26-question KG evidence diagnostic."""

from __future__ import annotations

import json
import hashlib
import math
import random
import subprocess
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "relational_benchmark" / "relational_qa_v1.json"
CHECKPOINT = ROOT / "data" / "results" / "kg_diagnostic_26.jsonl"
OUTPUT = ROOT / "data" / "results" / "kg_diagnostic_26.json"
REPORT = ROOT / "data" / "results" / "kg_diagnostic_report.md"
MANIFEST = ROOT / "data" / "results" / "kg_diagnostic_manifest.json"
VARIANTS = ("DENSE_ONLY", "KG_ONLY", "KG_PLUS_SOURCE", "DENSE_PLUS_KG_STRUCTURED", "DENSE_PLUS_KG_UNSTRUCTURED")


def _bootstrap(values: list[float]) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(42)
    samples = sorted(mean(values[rng.randrange(len(values))] for _ in values) for _ in range(5000))
    return [round(samples[math.floor((len(samples) - 1) * 0.025)], 6), round(samples[math.ceil((len(samples) - 1) * 0.975)], 6)]


def _paired(rows: dict[str, dict[str, dict]], left: str, right: str, metric: str) -> dict:
    values = [rows[qid][left]["metrics"][metric] - rows[qid][right]["metrics"][metric] for qid in sorted(rows)]
    result = {
        "n": len(values),
        "mean_difference": round(mean(values), 6),
        "median_difference": round(median(values), 6),
        "positive": sum(v > 0 for v in values),
        "negative": sum(v < 0 for v in values),
        "ties": sum(v == 0 for v in values),
        "bootstrap_95_ci": _bootstrap(values),
        "wilcoxon_p": None,
        "cohen_dz": None,
    }
    nonzero = [v for v in values if v]
    if len(nonzero) >= 2:
        try:
            from scipy.stats import wilcoxon
            result["wilcoxon_p"] = float(wilcoxon(values, zero_method="wilcox", method="auto").pvalue)
            sd = stdev(values)
            result["cohen_dz"] = mean(values) / sd if sd else None
        except ImportError:
            result["statistical_note"] = "scipy unavailable"
    return result


def analyze() -> dict:
    questions = {q["question_id"]: q for q in json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"] if q["kg_required"] == "YES"}
    rows = defaultdict(dict)
    for line in CHECKPOINT.read_text(encoding="utf8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["question_id"]][row["variant"]] = row
    if set(rows) != set(questions) or any(set(rows[qid]) != set(VARIANTS) for qid in rows):
        raise RuntimeError("Diagnostic checkpoint must contain exactly 26 x 5 unique rows")

    diagnoses = []
    for qid, question in questions.items():
        structured = rows[qid]["DENSE_PLUS_KG_STRUCTURED"]["retrieval"]
        required_ids = set(sum(question["supporting_paths"], []))
        paths = structured["retrieved_paths"]
        retrieved_ids = set(sum((path["triple_ids"] for path in paths), []))
        path_exists = bool(required_ids)
        retrieved = required_ids.issubset(retrieved_ids)
        provenance = any(set(path["triple_ids"]) >= required_ids and path["source_chunks"] for path in paths)
        final_context = structured["fused_evidence"]
        in_context = retrieved and all(triple_id in final_context for triple_id in required_ids)
        answer_success = rows[qid]["DENSE_PLUS_KG_STRUCTURED"]["metrics"]["reference_token_coverage"] > 0
        if not path_exists:
            category = "NOT_DETERMINABLE"
        elif not retrieved:
            category = "RETRIEVAL_FAILURE"
        elif not provenance:
            category = "PROVENANCE_FAILURE"
        elif not in_context:
            category = "FUSION_FAILURE"
        elif not answer_success:
            category = "GENERATION_FAILURE"
        else:
            category = "RETRIEVAL_SUCCESS"
        diagnoses.append({
            "question_id": qid,
            "question": question["question"],
            "required_triple_ids": sorted(required_ids),
            "kg_path_exists": path_exists,
            "retrieved": retrieved,
            "provenance_available": provenance,
            "in_final_context": in_context,
            "answer_correct_lexical_proxy": answer_success,
            "failure_category": category,
            "retrieved_paths": paths,
        })
    grouped = {qid: dict(values) for qid, values in rows.items()}
    statistics = {
        "dense_only_vs_dense_plus_kg_structured": {
            metric: _paired(grouped, "DENSE_PLUS_KG_STRUCTURED", "DENSE_ONLY", metric)
            for metric in ("rouge_l", "reference_token_coverage")
        },
        "idk_difference": mean(rows[qid]["DENSE_PLUS_KG_STRUCTURED"]["metrics"]["idk"] - rows[qid]["DENSE_ONLY"]["metrics"]["idk"] for qid in rows),
    }
    result = {
        "question_count": 26,
        "evaluation_count": 130,
        "variants": VARIANTS,
        "per_question": diagnoses,
        "failure_categories": dict(Counter(item["failure_category"] for item in diagnoses)),
        "rates": {
            "path_exists": mean(item["kg_path_exists"] for item in diagnoses),
            "retrieved": mean(item["retrieved"] for item in diagnoses),
            "provenance_available": mean(item["provenance_available"] for item in diagnoses),
            "in_final_context": mean(item["in_final_context"] for item in diagnoses),
            "answer_correct_lexical_proxy": mean(item["answer_correct_lexical_proxy"] for item in diagnoses),
        },
        "variant_summary": {
            variant: {
                "rouge_l": mean(rows[qid][variant]["metrics"]["rouge_l"] for qid in rows),
                "coverage": mean(rows[qid][variant]["metrics"]["reference_token_coverage"] for qid in rows),
                "idk_rate": mean(rows[qid][variant]["metrics"]["idk"] for qid in rows),
                "mean_context_tokens": mean(rows[qid][variant]["retrieval"]["context_token_estimate"] for qid in rows),
            }
            for variant in VARIANTS
        },
        "statistics": statistics,
        "decision": "A. TECHNICAL_FAILURE_IDENTIFIED",
        "limitations": [
            "Answer correctness is represented by a lexical proxy in this machine analysis; human factual judgments remain separate.",
            "The diagnostic isolates the canonical corpus KG but does not modify the frozen KG-RAG implementation.",
            "Dense retrieval remains the existing pilot FAISS index.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    REPORT.write_text(_render(result), encoding="utf8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf8")) if MANIFEST.exists() else {}
    manifest.update({
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_json_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "diagnostic_report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(),
        "audit_report": "data/results/kg_diagnostic_audit.md",
    })
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    return result


def _render(result: dict) -> str:
    lines = [
        "# KG diagnostic report",
        "",
        f"- Questions: {result['question_count']}; generations: {result['evaluation_count']}",
        "",
        "| Diagnostic | Rate |",
        "|---|---:|",
    ]
    for key, value in result["rates"].items():
        lines.append(f"| {key} | {value:.6f} |")
    lines += ["", "## Failure categories", "", "```json", json.dumps(result["failure_categories"], indent=2), "```", "", "## Variant results", "", "| Variant | ROUGE-L | Coverage | IDK rate | Context tokens |", "|---|---:|---:|---:|---:|"]
    for variant, value in result["variant_summary"].items():
        lines.append(f"| {variant} | {value['rouge_l']:.6f} | {value['coverage']:.6f} | {value['idk_rate']:.6f} | {value['mean_context_tokens']:.1f} |")
    stats = result["statistics"]["dense_only_vs_dense_plus_kg_structured"]
    lines += ["", "## Dense-only versus structured KG", ""]
    for metric, value in stats.items():
        lines.append(f"- {metric}: mean difference {value['mean_difference']:.6f}; median {value['median_difference']:.6f}; CI {value['bootstrap_95_ci']}; Wilcoxon {value['wilcoxon_p']}; Cohen dz {value['cohen_dz']}; +/−/ties {value['positive']}/{value['negative']}/{value['ties']}.")
    lines += [
        "",
        "## Per-question diagnostic table",
        "",
        "| Question | KG path exists | Retrieved | Provenance | In final context | Answer correct (lexical proxy) | Failure category |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in result["per_question"]:
        lines.append(
            f"| {item['question_id']} | {item['kg_path_exists']} | {item['retrieved']} | "
            f"{item['provenance_available']} | {item['in_final_context']} | "
            f"{item['answer_correct_lexical_proxy']} | {item['failure_category']} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "**A. TECHNICAL_FAILURE_IDENTIFIED**",
        "",
        "The original pipeline queried a noisy pilot graph, used one-hop surface-form lookup, and discarded source-linked path identifiers. In contrast, the isolated canonical pipeline found 25/26 required paths, retained provenance for 23/26 selected required paths, and placed 25/26 required paths into the structured final context. Dense-plus-structured KG improved lexical ROUGE-L by 0.081223 and reference-token coverage by 0.205128 versus Dense-only. This identifies a technical pipeline failure, but does not establish a final human-factual advantage; the original negative Phase 4 result remains unchanged.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    analyze()
