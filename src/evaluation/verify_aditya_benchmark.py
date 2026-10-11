"""Verify support and provenance of the 36-question Aditya-L1 benchmark."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "data" / "benchmark" / "aditya_l1_optional_qa.json"
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.json"

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and", "or", "in", "on", "at",
    "for", "from", "by", "with", "what", "which", "when", "where", "how", "did", "does", "do", "has",
}


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2 and t not in STOPWORDS}


def main() -> None:
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    all_chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    aditya_chunks = [c for c in all_chunks if str(c.get("document_id", "")).startswith("ADITYA_L1_")]

    print(f"Total benchmark questions: {len(benchmark)}")
    print(f"Total Aditya-L1 chunks: {len(aditya_chunks)}")

    report = []
    supported_count = 0
    for q in benchmark:
        qid = q["id"]
        question = q["question"]
        answer = q["answer"]
        ref_tokens = tokens(answer)
        best_cov = 0.0
        best_chunk = None

        for c in aditya_chunks:
            chunk_tokens = tokens(c.get("text", ""))
            matched = ref_tokens & chunk_tokens
            cov = len(matched) / len(ref_tokens) if ref_tokens else 0.0
            if cov > best_cov:
                best_cov = cov
                best_chunk = c

        supported = best_cov >= 0.70 or (len(ref_tokens) <= 3 and best_cov >= 0.60)
        if supported:
            supported_count += 1

        report.append({
            "id": qid,
            "question": question,
            "category": q["category"],
            "expected_answer": answer,
            "source_document_id": q.get("source_document_id"),
            "page": q.get("page"),
            "source_url": q.get("source_url"),
            "best_chunk_id": best_chunk.get("chunk_id") if best_chunk else None,
            "best_chunk_doc": best_chunk.get("document_id") if best_chunk else None,
            "token_coverage": round(best_cov, 4),
            "supported": supported,
        })
        print(f"[{'PASS' if supported else 'WARN'}] {qid} | cov={best_cov:.2f} | doc={best_chunk.get('document_id') if best_chunk else 'NONE'} | {q['category']}")

    print(f"\nSupported questions: {supported_count} / {len(benchmark)}")
    output_path = ROOT / "data" / "results" / "aditya_l1_quality_verification.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved quality verification report to {output_path}")


if __name__ == "__main__":
    main()
