"""Simple evaluation: ROUGE-L and exact match scoring."""

from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "data" / "results" / "baseline_results.json"
BENCHMARK_PATH = ROOT / "data" / "benchmark" / "isro_qa.json"
OUTPUT_PATH = ROOT / "data" / "results" / "ragas_scores.json"


def normalize(text: str) -> str:
    text = text.lower().strip()
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
            if ref_tokens[i-1] == pred_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / n if n else 0
    recall = lcs / m if m else 0
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize(prediction) == normalize(reference) else 0.0


def answer_coverage(prediction: str, reference: str) -> float:
    """How many reference words appear in prediction."""
    pred = set(normalize(prediction).split())
    ref = set(normalize(reference).split())
    if not ref:
        return 0.0
    return round(len(pred & ref) / len(ref), 4)


def evaluate_system(system: str, results: list, bench_map: dict) -> dict:
    rouge_scores, coverage_scores, em_scores = [], [], []
    idk_count = 0
    evaluated = 0

    for row in results:
        qid = row["id"]
        answer = row.get(f"{system}_answer", "").strip()
        reference = bench_map.get(qid, {}).get("answer", "").strip()

        if not reference:
            continue

        if not answer or "don't know" in answer.lower() or "i don't know" in answer.lower():
            idk_count += 1
            rouge_scores.append(0.0)
            coverage_scores.append(0.0)
            em_scores.append(0.0)
        else:
            rouge_scores.append(rouge_l(answer, reference))
            coverage_scores.append(answer_coverage(answer, reference))
            em_scores.append(exact_match(answer, reference))

        evaluated += 1

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "system": system,
        "n_evaluated": evaluated,
        "idk_count": idk_count,
        "idk_rate": round(idk_count / evaluated, 4) if evaluated else 0,
        "rouge_l": avg(rouge_scores),
        "answer_coverage": avg(coverage_scores),
        "exact_match": avg(em_scores),
    }


def main():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8-sig"))
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8-sig"))
    bench_map = {item["id"]: item for item in benchmark}

    systems = ["kgrag", "bm25", "vanilla_rag"]
    all_scores = []

    for system in systems:
        print(f"Evaluating {system}...")
        scores = evaluate_system(system, results, bench_map)
        all_scores.append(scores)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(all_scores, indent=2), encoding="utf-8")

    print("\n=== RESULTS SUMMARY ===")
    print(f"{'System':<15} {'ROUGE-L':<10} {'Coverage':<12} {'ExactMatch':<12} {'IDK%':<8} {'N'}")
    print("-" * 65)
    for s in all_scores:
        print(f"{s['system']:<15} {s['rouge_l']:<10} {s['answer_coverage']:<12} {s['exact_match']:<12} {s['idk_rate']:<8} {s['n_evaluated']}")


if __name__ == "__main__":
    main()