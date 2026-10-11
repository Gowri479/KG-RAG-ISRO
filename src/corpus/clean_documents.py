"""Normalize extracted records and log repeated PDF margin removal."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus"
INPUT_PATH = CORPUS / "extracted_records.jsonl"
OUTPUT_PATH = CORPUS / "cleaned_documents.jsonl"
REPORT_PATH = CORPUS / "cleaning_report.json"
LOGGER = logging.getLogger(__name__)


def _normalize(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def clean_records() -> dict[str, Any]:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Extracted records do not exist: {INPUT_PATH}")
    records = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    page_records: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.get("content_kind") == "pdf_text" and record.get("page") is not None:
            page_records[(record["document_id"], int(record["page"]))].append(index)

    margin_occurrences: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for (document_id, page_number), indices in page_records.items():
        first_record = records[indices[0]]
        last_record = records[indices[-1]]
        first_line = _normalize(first_record["text"]).splitlines()[0]
        last_line = _normalize(last_record["text"]).splitlines()[-1]
        for candidate in {first_line, last_line}:
            if 4 <= len(candidate) <= 120 and not re.fullmatch(r"[\W\d_]+", candidate):
                margin_occurrences[document_id][candidate].add(page_number)

    remove_from_page: dict[tuple[str, int], set[str]] = defaultdict(set)
    removed_lines: list[dict[str, Any]] = []
    pages_per_document = Counter(document_id for document_id, _ in page_records)
    for document_id, candidates in margin_occurrences.items():
        threshold = max(3, math.ceil(pages_per_document[document_id] * 0.25))
        for line, page_numbers in candidates.items():
            if len(page_numbers) < threshold:
                continue
            for page_number in page_numbers:
                remove_from_page[(document_id, page_number)].add(line)
            removed_lines.append({
                "document_id": document_id,
                "text": line,
                "page_numbers": sorted(page_numbers),
                "reason": "Exact repeated line found only at the first/last extracted line of PDF pages.",
            })

    cleaned: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        text = _normalize(item["text"])
        page = item.get("page")
        removals = remove_from_page.get((item["document_id"], int(page)), set()) if page is not None else set()
        if removals:
            lines = text.splitlines()
            while lines and lines[0] in removals:
                lines.pop(0)
            while lines and lines[-1] in removals:
                lines.pop()
            text = "\n".join(lines).strip()
            item["removed_margin_lines"] = sorted(removals)
        if text:
            item["text"] = text
            cleaned.append(item)

    OUTPUT_PATH.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in cleaned),
        encoding="utf-8",
    )
    report = {
        "input_records": len(records),
        "output_records": len(cleaned),
        "removed_repeated_margin_lines": removed_lines,
        "policy": "Whitespace is normalized. Only exact lines repeated at PDF page margins are removed, and every removal is recorded. HTML navigation/header/footer is excluded during semantic extraction.",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info(
        "Cleaned %d records to %d; recorded %d repeated margin-line removals.",
        len(records),
        len(cleaned),
        len(removed_lines),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    clean_records()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
