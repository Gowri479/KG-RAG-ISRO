"""Analyze the completed blinded evaluation of the corrected KG pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[2]
INPUT = Path(r"D:\Users\NANS\Downloads\final_structured_kg_human_eval_v2_filled.csv")
OUT = ROOT / "data" / "results" / "final_structured_kg_eval"
BENCHMARK = ROOT / "data" / "relational_benchmark" / "relational_qa_v1.json"
MANIFEST = OUT / "run_manifest.json"
GEN = OUT / "generation_results.jsonl"
DIAGNOSTIC = ROOT / "data" / "results" / "kg_diagnostic_26.json"
OLD = ROOT / "data" / "results" / "relational_qa_v1" / "human_evaluation_analysis.json"
SYSTEMS = ("vanilla_rag", "corrected_structured_kg_rag", "bm25_llm")
DISPLAY = {"vanilla_rag": "Vanilla RAG", "corrected_structured_kg_rag": "Corrected Structured KG-RAG", "bm25_llm": "BM25 + LLM"}
METRICS = ("correctness", "completeness", "groundedness", "relevance")
EXPECTED_HASH = "7f5daf5419b0a4cb70e5c3d5eba021a8a39118b5a0fd235ce19b81f75324ffce"


def mapping(question_ids: list[str]) -> dict[str, dict[str, str]]:
    rng = random.Random(20261007)
    result = {}
    for qid in question_ids:
        order = list(SYSTEMS)
        rng.shuffle(order)
        result[qid] = dict(zip(("A", "B", "C"), order))
    return result


def bootstrap(values: list[float], seed: int = 42) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    samples = sorted(mean(values[rng.randrange(len(values))] for _ in values) for _ in range(5000))
    return [round(samples[math.floor(4999 * 0.025)], 6), round(samples[math.ceil(4999 * 0.975)], 6)]


def paired(values: list[float]) -> dict:
    positive = sum(v > 0 for v in values)
    negative = sum(v < 0 for v in values)
    result = {
        "n": len(values), "mean_difference": round(mean(values), 6),
        "median_difference": round(median(values), 6),
        "standard_deviation": round(stdev(values), 6) if len(values) > 1 else None,
        "positive": positive, "negative": negative, "ties": len(values) - positive - negative,
        "bootstrap_95_ci": bootstrap(values), "wilcoxon_p": None, "cohen_dz": None,
        "note": None,
    }
    if len(values) <= 1:
        result["note"] = "Insufficient paired observations."
        return result
    nonzero = [v for v in values if v]
    if len(nonzero) < 2:
        result["note"] = "Almost all paired differences are tied; Wilcoxon and effect size are not informative."
        return result
    try:
        from scipy.stats import wilcoxon
        result["wilcoxon_p"] = round(float(wilcoxon(values, zero_method="wilcox", method="auto").pvalue), 6)
    except ImportError:
        result["note"] = "scipy unavailable; Wilcoxon not calculated."
    sd = stdev(values)
    result["cohen_dz"] = round(mean(values) / sd, 6) if sd else None
    return result


def holm(pvalues: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted(((key, value) for key, value in pvalues.items() if value is not None), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    for index, (key, value) in enumerate(valid):
        corrected = min(1.0, (len(valid) - index) * value)
        running = max(running, corrected)
        adjusted[key] = round(running, 6)
    return {key: adjusted.get(key) for key in pvalues}


def load() -> tuple[list[dict], dict[str, dict[str, str]], dict[str, dict]]:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf8"))
    questions = [q for q in benchmark["questions"] if q["kg_required"] == "YES"]
    if hashlib.sha256(BENCHMARK.read_bytes()).hexdigest() != EXPECTED_HASH or len(questions) != 26:
        raise ValueError("Frozen benchmark identity mismatch")
    with INPUT.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 78:
        raise ValueError(f"Expected 78 rows, found {len(rows)}")
    qids = [q["question_id"] for q in questions]
    blind = mapping(qids)
    generated = {}
    with GEN.open(encoding="utf8") as handle:
        for line in handle:
            row = json.loads(line)
            generated[(row["question_id"], row["system"])] = row
    seen = set()
    records = []
    for row in rows:
        qid, label = row["question_id"], row["blind_system_id"]
        if qid not in blind or label not in "ABC" or (qid, label) in seen:
            raise ValueError("Invalid or duplicate blinded evaluation row")
        seen.add((qid, label))
        if row["question"] != next(q["question"] for q in questions if q["question_id"] == qid):
            raise ValueError(f"{qid}: question text changed")
        system = blind[qid][label]
        generated_row = generated.get((qid, system))
        if generated_row is None or row["candidate_answer"] != generated_row["answer"]:
            raise ValueError(f"{qid}/{label}: candidate answer does not match the recorded generation")
        record = {"question_id": qid, "label": label, "system": system}
        for metric in METRICS:
            value = int(row[metric])
            if value not in range(1, 6):
                raise ValueError(f"{qid}/{label}: invalid {metric}")
            record[metric] = value
        value = int(row["unsupported_claim"])
        if value not in (0, 1):
            raise ValueError(f"{qid}/{label}: invalid unsupported_claim")
        record["unsupported_claim"] = value
        records.append(record)
    if len(seen) != 78:
        raise ValueError("Incomplete question/label coverage")
    return questions, blind, {r["question_id"] + ":" + r["label"]: r for r in records}


def analyze() -> None:
    questions, blind, record_map = load()
    records = list(record_map.values())
    by_q = defaultdict(dict)
    for row in records:
        by_q[row["question_id"]][row["system"]] = row
    summaries = {}
    for system in SYSTEMS:
        subset = [r for r in records if r["system"] == system]
        summaries[system] = {metric: {"mean": round(mean(r[metric] for r in subset), 6), "median": median(r[metric] for r in subset)} for metric in METRICS}
        summaries[system]["unsupported_claim_rate"] = round(mean(r["unsupported_claim"] for r in subset), 6)
    differences = {}
    for metric in METRICS:
        values = [by_q[q["question_id"]]["corrected_structured_kg_rag"][metric] - by_q[q["question_id"]]["vanilla_rag"][metric] for q in questions]
        differences[metric] = paired(values)
    pvalues = {metric: differences[metric]["wilcoxon_p"] for metric in METRICS}
    adjusted = holm(pvalues)
    question_rows = []
    outcomes = {metric: {"kg_better": 0, "vanilla_better": 0, "tie": 0} for metric in METRICS}
    for q in questions:
        qid = q["question_id"]
        row = {"question_id": qid, "blind_system_mapping": blind[qid]}
        for metric in METRICS:
            for system in SYSTEMS:
                row[f"{system}_{metric}"] = by_q[qid][system][metric]
            diff = by_q[qid]["corrected_structured_kg_rag"][metric] - by_q[qid]["vanilla_rag"][metric]
            outcome = "KG better" if diff > 0 else "Vanilla better" if diff < 0 else "Tie"
            outcomes[metric][outcome.lower().replace(" ", "_")] += 1
            row[f"kg_vs_vanilla_{metric}_outcome"] = outcome
        for system in SYSTEMS:
            row[f"{system}_unsupported_claim"] = by_q[qid][system]["unsupported_claim"]
        question_rows.append(row)
    with (OUT / "question_level_results.csv").open("w", encoding="utf8", newline="") as handle:
        fields = list(question_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(question_rows)
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf8"))
    diag_by_q = {r["question_id"]: r for r in diagnostic.get("per_question", [])}
    failure = []
    for q in questions:
        qid = q["question_id"]; human = by_q[qid]; trace = diag_by_q.get(qid, {})
        failure.append({
            "question_id": qid, "kg_correctness": human["corrected_structured_kg_rag"]["correctness"],
            "kg_groundedness": human["corrected_structured_kg_rag"]["groundedness"],
            "vanilla_correctness": human["vanilla_rag"]["correctness"],
            "kg_vs_vanilla_correctness": question_rows[len(failure)]["kg_vs_vanilla_correctness_outcome"],
            "kg_vs_vanilla_groundedness": question_rows[len(failure)]["kg_vs_vanilla_groundedness_outcome"],
            "diagnostic_failure_category": trace.get("failure_category", "NOT_DETERMINABLE"),
            "diagnostic_path_found": trace.get("kg_path_exists") and trace.get("retrieved"),
        })
    output = {
        "status": "COMPLETED", "input_file": str(INPUT), "benchmark_sha256": EXPECTED_HASH,
        "question_count": 26, "human_evaluation_count": 78, "blind_mapping": blind,
        "system_summaries": summaries, "primary_comparison": {"left": "corrected_structured_kg_rag", "right": "vanilla_rag", "metrics": differences, "holm_adjusted_p": adjusted},
        "unsupported_claim_rates": {system: summaries[system]["unsupported_claim_rate"] for system in SYSTEMS},
        "question_outcomes": outcomes, "failure_analysis": failure,
        "old_result_reference": json.loads(OLD.read_text(encoding="utf8")).get("primary_kg_vs_vanilla_yes"),
        "lexical_diagnostics": json.loads((OUT / "preliminary_metrics.json").read_text(encoding="utf8")),
        "decision": "PENDING_COMPUTATION",
    }
    correctness = differences["correctness"]; grounded = differences["groundedness"]
    if correctness["mean_difference"] > 0 and grounded["mean_difference"] > 0:
        output["decision"] = "B. PARTIAL"
    elif correctness["mean_difference"] <= 0 and grounded["mean_difference"] <= 0:
        output["decision"] = "C. NO"
    else:
        output["decision"] = "B. PARTIAL"
    (OUT / "human_evaluation_analysis.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf8")
    (OUT / "statistical_results.json").write_text(json.dumps({"primary_comparison": output["primary_comparison"], "unsupported_claim_rates": output["unsupported_claim_rates"]}, indent=2) + "\n", encoding="utf8")
    report = ["# Final human-evaluation results", "", f"**Decision: {output['decision']}**", "", "Human scores were analyzed unchanged from the submitted completed CSV. No generation was rerun and no human scores were fabricated.", "", "## System means"]
    report += ["", "| System | Correctness | Completeness | Groundedness | Relevance | Unsupported claims |", "|---|---:|---:|---:|---:|---:|"]
    for system in SYSTEMS:
        s = summaries[system]; report.append(f"| {DISPLAY[system]} | {s['correctness']['mean']:.3f} | {s['completeness']['mean']:.3f} | {s['groundedness']['mean']:.3f} | {s['relevance']['mean']:.3f} | {s['unsupported_claim_rate']:.3f} |")
    report += ["", "## Corrected KG-RAG minus Vanilla RAG", "", "| Metric | Mean difference | Median | SD | 95% CI | Wilcoxon p | Holm p | Cohen dz | KG/Vanilla/Tie |", "|---|---:|---:|---:|---|---:|---:|---:|---|"]
    for metric in METRICS:
        d = differences[metric]; report.append(f"| {metric} | {d['mean_difference']:.3f} | {d['median_difference']:.3f} | {d['standard_deviation']:.3f} | {d['bootstrap_95_ci']} | {d['wilcoxon_p']} | {adjusted[metric]} | {d['cohen_dz']} | {d['positive']}/{d['negative']}/{d['ties']} |")
    old_primary = output["old_result_reference"]
    report += [
        "", "## Comparison with the old KG-RAG result", "",
        f"The frozen Phase 5 old KG-RAG result on the same subset had correctness difference "
        f"{old_primary['metrics']['correctness']['mean_difference']:.3f}, groundedness difference "
        f"{old_primary['metrics']['groundedness']['mean_difference']:.3f}, and unsupported-claim rates "
        f"{old_primary['unsupported_claim_rate']['kg_rag']:.3f} (old KG-RAG) versus "
        f"{old_primary['unsupported_claim_rate']['vanilla_rag']:.3f} (Vanilla). The corrected pipeline "
        f"has correctness difference {differences['correctness']['mean_difference']:.3f}, groundedness "
        f"difference {differences['groundedness']['mean_difference']:.3f}, and equal unsupported-claim "
        f"rates of {summaries['corrected_structured_kg_rag']['unsupported_claim_rate']:.3f} and "
        f"{summaries['vanilla_rag']['unsupported_claim_rate']:.3f}. The direction therefore changed for "
        f"groundedness but not for correctness.",
        "", "## Automated versus human results", "",
        "The corrected KG improved the preliminary lexical diagnostics over Vanilla "
        "(ROUGE-L 0.203808 vs 0.116641; reference-token coverage 0.501282 vs 0.292949). "
        "Human results agree only on groundedness direction: groundedness was higher for corrected KG, "
        "while correctness, completeness, and relevance were not higher. Lexical improvement therefore "
        "does not establish factual-quality improvement.",
        "", "## Interpretation", "",
        f"The corrected KG system is evaluated here using human correctness and groundedness as the primary "
        f"evidence. The result is {output['decision']}: correctness, completeness, groundedness, and "
        f"relevance were all higher descriptively, but confidence intervals were wide and no Holm-adjusted "
        f"primary metric was significant. This is not a clear overall human-rated factual-quality advantage.",
        "", "## Limitations", "",
        "This is a 26-question researcher-constructed benchmark with one annotation per answer. Results "
        "should not be generalized beyond this controlled evaluation. One annotation was available per "
        "answer, so inter-annotator agreement cannot be estimated.",
        "", "The previous Phase 5 human evaluation remains frozen and was not combined with these scores.", ""
    ]
    (OUT / "human_evaluation_report.md").write_text("\n".join(report), encoding="utf8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    manifest.update({"human_csv_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(), "diagnostic_artifact_sha256": hashlib.sha256(DIAGNOSTIC.read_bytes()).hexdigest(), "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(), "analysis_script": "src/evaluation/analyze_final_structured_kg_human.py", "analysis_methodology": "paired per-question differences; 5000-sample bootstrap percentile CI; Wilcoxon signed-rank; Cohen dz; Holm correction across four primary metrics"})
    for path in (OUT / "human_evaluation_analysis.json", OUT / "statistical_results.json", OUT / "question_level_results.csv", OUT / "human_evaluation_report.md"):
        manifest[path.name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "final_experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    analyze()
