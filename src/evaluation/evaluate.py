"""Canonical evaluation for the ISRO-QA benchmark.

This script evaluates only the fixed 180-question test set and writes a single
reproducible result file: data/results/evaluation_results.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "data" / "benchmark" / "isro_qa.json"
TEST_IDS_PATH = ROOT / "data" / "benchmark" / "test_ids.json"
DEV_IDS_PATH = ROOT / "data" / "benchmark" / "dev_ids.json"
RESULTS_PATH = ROOT / "data" / "results" / "baseline_results.json"
OUTPUT_PATH = ROOT / "data" / "results" / "evaluation_results.json"

SYSTEM_FIELDS = {
    "bm25_llm": "bm25_answer",
    "vanilla_rag": "vanilla_rag_answer",
    "kg_rag": "kgrag_answer",
}

MODEL_NAME = "mistral:7b-instruct-q4_K_M"
INFERENCE_BACKEND = "Ollama local API"
EVALUATION_PROTOCOL = "canonical_lexical_eval_v1"


def get_git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or None
    except Exception:
        pass
    return None


def normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / n if n else 0.0
    recall = lcs / m if m else 0.0
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 4)


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize(prediction) == normalize(reference) else 0.0


def reference_token_coverage(prediction: str, reference: str) -> float:
    pred_tokens = set(normalize(prediction).split())
    ref_tokens = set(normalize(reference).split())
    if not ref_tokens:
        return 0.0
    return round(len(pred_tokens & ref_tokens) / len(ref_tokens), 4)


def is_idk(answer: str) -> bool:
    text = (answer or "").lower().strip()
    return not text or "i don't know" in text or "i do not know" in text or "don't know" in text


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def evaluate_system(system_name: str, results: list[dict], benchmark: list[dict], test_ids: set[str]) -> dict:
    bench_map = {item["id"]: item for item in benchmark}
    rouge_scores: list[float] = []
    coverage_scores: list[float] = []
    exact_scores: list[float] = []
    idk_count = 0
    evaluated = 0

    for row in results:
        qid = row.get("id")
        if qid not in test_ids:
            continue
        reference = (bench_map.get(qid, {}) or {}).get("answer", "")
        if not reference:
            continue
        answer = (row.get(SYSTEM_FIELDS[system_name], "") or "").strip()
        evaluated += 1
        if is_idk(answer):
            idk_count += 1
            rouge_scores.append(0.0)
            coverage_scores.append(0.0)
            exact_scores.append(0.0)
        else:
            rouge_scores.append(rouge_l(answer, reference))
            coverage_scores.append(reference_token_coverage(answer, reference))
            exact_scores.append(exact_match(answer, reference))

    return {
        "system": system_name,
        "n_evaluated": evaluated,
        "idk_count": idk_count,
        "idk_rate": round(idk_count / evaluated, 4) if evaluated else 0.0,
        "rouge_l": mean(rouge_scores),
        "reference_token_coverage": mean(coverage_scores),
        "coverage": mean(coverage_scores),
        "exact_match": mean(exact_scores),
    }


def evaluate_by_tier(system_name: str, results: list[dict], benchmark: list[dict], test_ids: set[str]) -> dict:
    bench_map = {item["id"]: item for item in benchmark}
    tiers: dict[str, list[tuple[float, float, float]]] = {"tier_1": [], "tier_2": [], "tier_3": []}
    idk: dict[str, int] = {"tier_1": 0, "tier_2": 0, "tier_3": 0}
    n: dict[str, int] = {"tier_1": 0, "tier_2": 0, "tier_3": 0}

    for row in results:
        qid = row.get("id")
        if qid not in test_ids:
            continue
        item = bench_map.get(qid)
        if not item:
            continue
        tier_key = f"tier_{int(item['tier'])}"
        if tier_key not in tiers:
            continue
        reference = item.get("answer", "")
        answer = (row.get(SYSTEM_FIELDS[system_name], "") or "").strip()
        n[tier_key] += 1
        if is_idk(answer):
            idk[tier_key] += 1
            tiers[tier_key].append((0.0, 0.0, 0.0))
        else:
            tiers[tier_key].append((rouge_l(answer, reference), reference_token_coverage(answer, reference), exact_match(answer, reference)))

    output: dict[str, dict] = {}
    for tier_name in ["tier_1", "tier_2", "tier_3"]:
        scores = tiers[tier_name]
        if not scores:
            output[tier_name] = {"n": 0, "rouge_l": 0.0, "reference_token_coverage": 0.0, "coverage": 0.0, "exact_match": 0.0, "idk_rate": 0.0}
            continue
        rouges = [s[0] for s in scores]
        coverages = [s[1] for s in scores]
        exacts = [s[2] for s in scores]
        output[tier_name] = {
            "n": n[tier_name],
            "rouge_l": mean(rouges),
            "reference_token_coverage": mean(coverages),
            "coverage": mean(coverages),
            "exact_match": mean(exacts),
            "idk_rate": round(idk[tier_name] / n[tier_name], 4) if n[tier_name] else 0.0,
        }
    return output


def compute_paired_difference(results: list[dict], benchmark: list[dict], test_ids: set[str], metric: str) -> float:
    bench_map = {item["id"]: item for item in benchmark}
    diffs: list[float] = []
    for row in results:
        qid = row.get("id")
        if qid not in test_ids:
            continue
        item = bench_map.get(qid)
        if item is None:
            continue
        reference = item.get("answer", "")
        kg_answer = (row.get("kgrag_answer", "") or "").strip()
        vanilla_answer = (row.get("vanilla_rag_answer", "") or "").strip()

        if metric == "rouge_l":
            kg_score = 0.0 if is_idk(kg_answer) else rouge_l(kg_answer, reference)
            vanilla_score = 0.0 if is_idk(vanilla_answer) else rouge_l(vanilla_answer, reference)
        elif metric == "coverage":
            kg_score = 0.0 if is_idk(kg_answer) else reference_token_coverage(kg_answer, reference)
            vanilla_score = 0.0 if is_idk(vanilla_answer) else reference_token_coverage(vanilla_answer, reference)
        elif metric == "idk_rate":
            kg_score = 1.0 if is_idk(kg_answer) else 0.0
            vanilla_score = 1.0 if is_idk(vanilla_answer) else 0.0
        else:
            raise ValueError(f"Unsupported metric {metric}")
        diffs.append(kg_score - vanilla_score)

    return round(sum(diffs) / len(diffs), 4) if diffs else 0.0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the fixed ISRO-QA split without changing benchmark definitions.")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH, help="Generated answers JSON path.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Evaluation output JSON path.")
    args = parser.parse_args()
    results_path = args.results if args.results.is_absolute() else ROOT / args.results
    output_path = args.output if args.output.is_absolute() else ROOT / args.output

    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8-sig"))
    test_ids = set(json.loads(TEST_IDS_PATH.read_text(encoding="utf-8-sig")))
    results = json.loads(results_path.read_text(encoding="utf-8-sig"))
    benchmark_hash = hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest()[:12]

    system_results = {}
    tier_results = {}
    for system_name in SYSTEM_FIELDS:
        system_results[system_name] = evaluate_system(system_name, results, benchmark, test_ids)
        tier_results[system_name] = evaluate_by_tier(system_name, results, benchmark, test_ids)

    dev_ids = set(json.loads(DEV_IDS_PATH.read_text(encoding="utf-8-sig")))
    output = {
        "evaluation_protocol": {
            "name": EVALUATION_PROTOCOL,
            "description": "Canonical lexical evaluation over the deterministic 20-dev / 180-test split.",
            "benchmark_size": len(benchmark),
            "dev_size": len(dev_ids),
            "test_size": len(test_ids),
            "split_seed": 42,
            "split_file": str(TEST_IDS_PATH),
            "dev_split_file": str(DEV_IDS_PATH),
            "systems_evaluated": ["bm25_llm", "vanilla_rag", "kg_rag"],
            "metrics": ["rouge_l", "reference_token_coverage", "exact_match", "idk_rate"],
        },
        "benchmark": {
            "path": str(BENCHMARK_PATH),
            "total_questions": len(benchmark),
            "development_questions": len(dev_ids),
            "test_questions": len(test_ids),
            "benchmark_hash": benchmark_hash,
            "tiers": {"tier_1": 100, "tier_2": 60, "tier_3": 40},
            "split_seed": 42,
            "split_file": str(TEST_IDS_PATH),
            "dev_split_file": str(DEV_IDS_PATH),
            "dev_test_overlap": sorted(dev_ids & test_ids),
            "test_ids": sorted(test_ids),
        },
        "execution": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_NAME,
            "inference_backend": INFERENCE_BACKEND,
            "repository_commit": get_git_commit(),
        },
        "metric_definitions": {
            "rouge_l": "Longest common subsequence F-score over normalized token sequences.",
            "reference_token_coverage": "|prediction_tokens ∩ reference_tokens| / |reference_tokens|.",
            "exact_match": "1 when normalized prediction equals normalized reference, otherwise 0.",
            "idk_rate": "Fraction of answers that are blank or abstain with 'I don't know' language.",
        },
        "system_results": system_results,
        "tier_results": tier_results,
        "overall": {
            "test_set_size": len(test_ids),
            "systems": {name: {k: v for k, v in values.items() if k not in {"system", "n_evaluated", "idk_count"}} for name, values in system_results.items()},
        },
        "paired_difference_kg_vs_vanilla": {
            "rouge_l_mean_diff": compute_paired_difference(results, benchmark, test_ids, "rouge_l"),
            "coverage_mean_diff": compute_paired_difference(results, benchmark, test_ids, "coverage"),
            "idk_rate_mean_diff": compute_paired_difference(results, benchmark, test_ids, "idk_rate"),
            "notes": "Mean paired difference: KG-RAG minus Vanilla RAG, using the deterministic 180-question test split.",
        },
        "per_question_results": {
            "notes": "Each row preserves the reference answer, generated answer, lexical metrics, and IDK flag for auditability.",
            "rows": [
                {
                    "id": row.get("id"),
                    "tier": next((item.get("tier") for item in benchmark if item.get("id") == row.get("id")), None),
                    "reference_answer": next((item.get("answer", "") for item in benchmark if item.get("id") == row.get("id")), ""),
                    "bm25_answer": row.get("bm25_answer", ""),
                    "vanilla_rag_answer": row.get("vanilla_rag_answer", ""),
                    "kg_rag_answer": row.get("kgrag_answer", ""),
                    "bm25_rouge_l": 0.0 if is_idk(row.get("bm25_answer", "")) else rouge_l(row.get("bm25_answer", ""), next((item.get("answer", "") for item in benchmark if item.get("id") == row.get("id")), "")),
                    "vanilla_rouge_l": 0.0 if is_idk(row.get("vanilla_rag_answer", "")) else rouge_l(row.get("vanilla_rag_answer", ""), next((item.get("answer", "") for item in benchmark if item.get("id") == row.get("id")), "")),
                    "kg_rouge_l": 0.0 if is_idk(row.get("kgrag_answer", "")) else rouge_l(row.get("kgrag_answer", ""), next((item.get("answer", "") for item in benchmark if item.get("id") == row.get("id")), "")),
                    "bm25_coverage": 0.0 if is_idk(row.get("bm25_answer", "")) else reference_token_coverage(row.get("bm25_answer", ""), next((item.get("answer", "") for item in benchmark if item.get("id") == row.get("id")), "")),
                    "vanilla_coverage": 0.0 if is_idk(row.get("vanilla_rag_answer", "")) else reference_token_coverage(row.get("vanilla_rag_answer", ""), next((item.get("answer", "") for item in benchmark if item.get("id") == row.get("id")), "")),
                    "kg_coverage": 0.0 if is_idk(row.get("kgrag_answer", "")) else reference_token_coverage(row.get("kgrag_answer", ""), next((item.get("answer", "") for item in benchmark if item.get("id") == row.get("id")), "")),
                    "bm25_exact_match": exact_match(row.get("bm25_answer", ""), next((item.get("answer", "") for item in benchmark if item.get("id") == row.get("id")), "")),
                    "vanilla_exact_match": exact_match(row.get("vanilla_rag_answer", ""), next((item.get("answer", "") for item in benchmark if item.get("id") == row.get("id")), "")),
                    "kg_exact_match": exact_match(row.get("kgrag_answer", ""), next((item.get("answer", "") for item in benchmark if item.get("id") == row.get("id")), "")),
                    "bm25_idk": is_idk(row.get("bm25_answer", "")),
                    "vanilla_idk": is_idk(row.get("vanilla_rag_answer", "")),
                    "kg_idk": is_idk(row.get("kgrag_answer", "")),
                }
                for row in results
                if row.get("id") in test_ids
            ],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved evaluation to {output_path}")
    print(json.dumps(output["overall"]["systems"], indent=2))


if __name__ == "__main__":
    main()
