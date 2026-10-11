"""Create section-, table-, and page-aware semantic chunks."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus"
INPUT_PATH = CORPUS / "cleaned_documents.jsonl"
OUTPUT_PATH = CORPUS / "chunks.jsonl"
REPORT_PATH = CORPUS / "chunking_report.json"
LOGGER = logging.getLogger(__name__)


def _word_count(text: str) -> int:
    return len(text.split())


def _split_oversize_text(text: str, max_words: int) -> list[str]:
    if _word_count(text) <= max_words:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 1:
        return [text]
    segments: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        words = _word_count(sentence)
        if current and current_words + words > max_words:
            segments.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words += words
    if current:
        segments.append(" ".join(current))
    return segments


def make_chunks(max_words: int = 350) -> dict[str, Any]:
    if max_words < 50:
        raise ValueError("max_words must be at least 50 to preserve paragraph context")
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Cleaned records do not exist: {INPUT_PATH}")

    records = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks: list[dict[str, Any]] = []
    document_chunk_counts: dict[str, int] = defaultdict(int)
    oversized_chunks: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None

    def emit() -> None:
        nonlocal pending
        if pending is None:
            return
        document_chunk_counts[pending["document_id"]] += 1
        chunk_index = document_chunk_counts[pending["document_id"]]
        chunk = {
            "chunk_id": f"{pending['document_id']}::c{chunk_index:05d}",
            "document_id": pending["document_id"],
            "mission": pending["mission"],
            "section": pending["section"],
            "subsection": pending.get("subsection", ""),
            "page": pending.get("page"),
            "text": pending["text"].strip(),
            "source_url": pending["source_url"],
            "authority_tier": pending["authority_tier"],
            "content_kind": pending["content_kind"],
            "word_count_estimate": _word_count(pending["text"]),
        }
        chunks.append(chunk)
        if chunk["word_count_estimate"] > max_words:
            oversized_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "word_count_estimate": chunk["word_count_estimate"],
                "reason": "A single sentence, table row, or semantic record exceeded the target and was not split internally.",
            })
        pending = None

    for record in records:
        text = record["text"].strip()
        if not text:
            continue
        parts = [text] if record["content_kind"] == "table_row" else _split_oversize_text(text, max_words)
        for part in parts:
            key = (
                record["document_id"],
                record.get("mission"),
                record.get("section", ""),
                record.get("subsection", ""),
                record.get("page"),
                record.get("content_kind"),
                record.get("source_url"),
                record.get("authority_tier"),
            )
            same_context = pending is not None and pending["context_key"] == key
            can_append = (
                same_context
                and record["content_kind"] != "table_row"
                and _word_count(pending["text"]) + _word_count(part) <= max_words
            )
            if can_append:
                pending["text"] += "\n\n" + part
            else:
                emit()
                pending = {
                    "context_key": key,
                    "document_id": record["document_id"],
                    "mission": record.get("mission"),
                    "section": record.get("section", ""),
                    "subsection": record.get("subsection", ""),
                    "page": record.get("page"),
                    "source_url": record["source_url"],
                    "authority_tier": record["authority_tier"],
                    "content_kind": record["content_kind"],
                    "text": part,
                }
            if record["content_kind"] == "table_row":
                emit()
    emit()

    OUTPUT_PATH.write_text(
        "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    counts: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        counts[chunk["mission"]] += 1
    report = {
        "target_chunk_words": max_words,
        "overlap_words": 0,
        "chunk_count": len(chunks),
        "word_count_estimate": sum(chunk["word_count_estimate"] for chunk in chunks),
        "documents_with_chunks": len(document_chunk_counts),
        "chunks_by_mission": dict(sorted(counts.items())),
        "oversized_chunks": oversized_chunks,
        "policy": "Chunks do not cross document, section, or PDF page boundaries. Table rows remain atomic with their headers. Long prose is split at sentence boundaries; no arbitrary overlap is added.",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Created %d semantic chunks from %d extracted records.", len(chunks), len(records))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-words", type=int, default=350, help="Upper target for a semantic chunk; not tuned against QA results.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    make_chunks(args.max_words)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
