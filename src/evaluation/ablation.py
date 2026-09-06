"""Ablation study: KG-only vs FAISS-only vs Full KG-RAG."""

from __future__ import annotations
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "data" / "benchmark" / "isro_qa.json"
OUTPUT_PATH = ROOT / "data" / "results" / "ablation_results.json"


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


def answer_coverage(prediction: str, reference: str) -> float:
    pred = set(normalize(prediction).split())
    ref = set(normalize(reference).split())
    if not ref:
        return 0.0
    return round(len(pred & ref) / len(ref), 4)


def get_answer_kg_only(question: str) -> str:
    from src.retriever.kg_retriever import get_kg_context
    from src.generator.ollama_api import generate
    import spacy
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(question)
    entities = [ent.text for ent in doc.ents]
    context = get_kg_context(entities)
    if not context.strip():
        return "I don't know."
    return generate(question, context)


def get_answer_faiss_only(question: str) -> str:
    from src.retriever.faiss_retriever import get_passage_context
    from src.generator.ollama_api import generate
    context = get_passage_context(question, top_k=3)
    if not context.strip():
        return "I don't know."
    return generate(question, context)


def get_answer_full_kgrag(question: str) -> str:
    from src.retriever.hybrid import retrieve
    from src.generator.ollama_api import generate
    context = retrieve(question)
    if not context.strip():
        return "I don't know."
    return generate(question, context)


def score_answers(answers: list, references: list) -> dict:
    rouge_scores, coverage_scores = [], []
    idk_count = 0
    for ans, ref in zip(answers, references):
        if not ref:
            continue
        if "don't know" in ans.lower() or not ans.strip():
            idk_count += 1
            rouge_scores.append(0.0)
            coverage_scores.append(0.0)
        else:
            rouge_scores.append(rouge_l(ans, ref))
            coverage_scores.append(answer_coverage(ans, ref))
    n = len(rouge_scores)
    return {
        "rouge_l": round(sum(rouge_scores) / n, 4) if n else 0,
        "coverage": round(sum(coverage_scores) / n, 4) if n else 0,
        "idk_rate": round(idk_count / n, 4) if n else 0,
        "n": n,
    }


def main():
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8-sig"))

    # Run on first 50 questions only for speed (representative sample)
    sample = benchmark[:50]
    questions = [item["question"] for item in sample]
    references = [item["answer"] for item in sample]

    systems = {
        "kg_only": get_answer_kg_only,
        "faiss_only": get_answer_faiss_only,
        "full_kgrag": get_answer_full_kgrag,
    }

    all_results = {}
    for name, fn in systems.items():
        print(f"\nRunning {name} on {len(questions)} questions...")
        answers = []
        for i, q in enumerate(questions):
            try:
                ans = fn(q)
            except Exception as e:
                ans = "I don't know."
                print(f"  Error on q{i}: {e}")
            answers.append(ans)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(questions)}")
        scores = score_answers(answers, references)
        all_results[name] = scores
        print(f"  {name}: {scores}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print("\n=== ABLATION RESULTS ===")
    print(f"{'System':<15} {'ROUGE-L':<10} {'Coverage':<12} {'IDK%':<8} {'N'}")
    print("-" * 50)
    for name, s in all_results.items():
        print(f"{name:<15} {s['rouge_l']:<10} {s['coverage']:<12} {s['idk_rate']:<8} {s['n']}")


if __name__ == "__main__":
    main()