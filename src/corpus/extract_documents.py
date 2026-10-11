"""Extract section- and page-aware records from the Phase 1 raw corpus."""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus"
REGISTRY_PATH = CORPUS / "document_registry.json"
EXTRACTED_PATH = CORPUS / "extracted_records.jsonl"
REPORT_PATH = CORPUS / "extraction_report.json"
LOGGER = logging.getLogger(__name__)
_BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li"}
_IGNORED_TAGS = {"script", "style", "noscript", "nav", "footer", "header", "aside", "svg"}
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
_IGNORED_CLASS = re.compile(r"(?:^|[\s_-])(breadcrumb|breadcrumbs|breadcr?umb|navbar|navigation|site-footer|site-header|social|share)(?:$|[\s_-])", re.I)


def _clean_whitespace(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


class SemanticHTMLExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.records: list[dict[str, Any]] = []
        self.heading_path: list[tuple[int, str]] = []
        self.title_parts: list[str] = []
        self.in_title = False
        self.ignored_stack: list[str] = []
        self.active_block_tag: str | None = None
        self.active_block: list[str] = []
        self.table_depth = 0
        self.row_depth = 0
        self.cell_tag: str | None = None
        self.cell_buffer: list[str] = []
        self.current_row: list[tuple[str, str]] = []
        self.table_headers: list[str] = []

    @property
    def section(self) -> str:
        return " > ".join(label for _, label in self.heading_path) or "Document"

    def _record(self, text: str, kind: str) -> None:
        cleaned = _clean_whitespace(text)
        if not cleaned:
            return
        self.records.append({
            "content_kind": kind,
            "section": self.section,
            "subsection": self.heading_path[-1][1] if self.heading_path else "",
            "page": None,
            "text": cleaned,
        })

    def _finish_block(self) -> None:
        if self.active_block_tag:
            text = "".join(self.active_block)
            tag = self.active_block_tag
            if tag.startswith("h") and tag[1:].isdigit():
                level = int(tag[1:])
                label = _clean_whitespace(text)
                self.heading_path = [(current, value) for current, value in self.heading_path if current < level]
                if label:
                    self.heading_path.append((level, label))
                    self._record(label, "heading")
            else:
                self._record(text, "paragraph" if tag == "p" else "list_item")
        self.active_block_tag = None
        self.active_block = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name.lower(): value or "" for name, value in attrs}
        if self.ignored_stack:
            if tag not in _VOID_TAGS:
                self.ignored_stack.append(tag)
            return
        identifier = f"{attributes.get('id', '')} {attributes.get('class', '')}"
        if tag in _IGNORED_TAGS or _IGNORED_CLASS.search(identifier):
            if tag not in _VOID_TAGS:
                self.ignored_stack.append(tag)
            return
        if tag == "title":
            self.in_title = True
        elif tag in _BLOCK_TAGS:
            if self.active_block_tag is not None:
                self._finish_block()
            self.active_block_tag = tag
            self.active_block = []
        elif tag == "table":
            self.table_depth += 1
        elif tag == "tr" and self.table_depth:
            self.row_depth += 1
            self.current_row = []
        elif tag in {"th", "td"} and self.row_depth:
            self.cell_tag = tag
            self.cell_buffer = []
        elif tag == "br":
            if self.cell_tag:
                self.cell_buffer.append("\n")
            elif self.active_block_tag:
                self.active_block.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.ignored_stack:
            for index in range(len(self.ignored_stack) - 1, -1, -1):
                if self.ignored_stack[index] == tag:
                    del self.ignored_stack[index:]
                    break
            return
        if tag == "title":
            self.in_title = False
        elif tag in _BLOCK_TAGS and self.active_block_tag == tag:
            self._finish_block()
        elif tag in {"th", "td"} and self.cell_tag == tag:
            self.current_row.append((tag, _clean_whitespace("".join(self.cell_buffer))))
            self.cell_tag = None
            self.cell_buffer = []
        elif tag == "tr" and self.row_depth:
            cells = [text for _, text in self.current_row if text]
            if cells and all(cell_tag == "th" for cell_tag, _ in self.current_row):
                self.table_headers = cells
            elif cells:
                if self.table_headers and len(cells) == len(self.table_headers):
                    row_text = " | ".join(
                        f"{header}: {value}" for header, value in zip(self.table_headers, cells)
                    )
                    row_text = f"Table columns — {row_text}"
                else:
                    row_text = " | ".join(cells)
                self._record(row_text, "table_row")
            self.current_row = []
            self.row_depth -= 1
        elif tag == "table" and self.table_depth:
            self.table_depth -= 1
            self.table_headers = []

    def handle_data(self, data: str) -> None:
        if self.ignored_stack:
            return
        if self.in_title:
            self.title_parts.append(data)
        elif self.cell_tag:
            self.cell_buffer.append(data)
        elif self.active_block_tag:
            self.active_block.append(data)


def _is_pdf_heading(line: str) -> bool:
    if len(line) > 100 or len(line.split()) > 14 or re.search(r"\d{2,}", line):
        return False
    if line.endswith((".", "?", "!", ";", ":")):
        return False
    letters = [char for char in line if char.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
    title_case_words = sum(word[:1].isupper() for word in line.split() if word[:1].isalpha())
    word_count = max(1, sum(word[:1].isalpha() for word in line.split()))
    return uppercase_ratio >= 0.72 or (len(line) <= 70 and title_case_words / word_count >= 0.85)


def _extract_pdf(path: Path) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    reader = PdfReader(str(path))
    if reader.is_encrypted and reader.decrypt("") == 0:
        raise ValueError(f"Encrypted PDF cannot be opened without a password: {path}")
    records: list[dict[str, Any]] = []
    pages_with_text = 0
    warnings: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            warnings.append(f"Page {page_number} produced no extractable text.")
            continue
        suspect_counts: dict[int, int] = {}
        for character in page_text:
            if character == "\ufffd" or unicodedata.category(character) in {"Co", "Cs"}:
                codepoint = ord(character)
                suspect_counts[codepoint] = suspect_counts.get(codepoint, 0) + 1
        if suspect_counts:
            glyphs = ", ".join(
                f"U+{codepoint:04X} ({count})"
                for codepoint, count in sorted(suspect_counts.items())
            )
            warnings.append(
                f"Page {page_number} contains unresolved extraction glyphs {glyphs}; "
                "text was retained unchanged for review."
            )
        pages_with_text += 1
        section = f"Page {page_number}"
        paragraph_lines: list[str] = []

        def flush() -> None:
            nonlocal paragraph_lines
            text = _clean_whitespace("\n".join(paragraph_lines))
            if text:
                records.append({
                    "content_kind": "pdf_text",
                    "section": section,
                    "subsection": section.removeprefix(f"Page {page_number}").strip(" >"),
                    "page": page_number,
                    "text": text,
                })
            paragraph_lines = []

        for raw_line in page_text.replace("\r", "\n").splitlines():
            line = _clean_whitespace(raw_line)
            if not line:
                flush()
            elif _is_pdf_heading(line):
                flush()
                section = f"Page {page_number} > {line}"
            else:
                paragraph_lines.append(line)
        flush()

    report = {
        "page_count": len(reader.pages),
        "pages_with_text": pages_with_text,
        "warnings": warnings,
    }
    return "", records, report


def _extract_html(path: Path) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    parser = SemanticHTMLExtractor()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    parser.close()
    title = _clean_whitespace(" ".join(parser.title_parts))
    return title, parser.records, {
        "page_count": 1,
        "pages_with_text": 1 if parser.records else 0,
        "warnings": [] if parser.records else ["No semantic content blocks were extracted from the HTML source."],
    }


def extract_registry(registry_path: Path = REGISTRY_PATH) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    document_reports: list[dict[str, Any]] = []
    failures: list[str] = []

    for document in registry["documents"]:
        source_path = ROOT / Path(document["local_path"].replace("/", "\\"))
        if not source_path.is_file():
            failures.append(f"{document['document_id']}: missing file {document['local_path']}")
            document["extraction_status"] = "missing_source"
            continue
        try:
            if document["file_type"] == "pdf":
                extracted_title, document_records, report = _extract_pdf(source_path)
            elif document["file_type"] == "html":
                extracted_title, document_records, report = _extract_html(source_path)
            else:
                raise ValueError(f"Unsupported file type {document['file_type']!r}")

            for index, record in enumerate(document_records, start=1):
                records.append({
                    "record_id": f"{document['document_id']}::r{index:05d}",
                    "document_id": document["document_id"],
                    "title": extracted_title or document["title"],
                    "mission": document["mission"],
                    "document_type": document["document_type"],
                    "source_url": document["source_url"],
                    "authority_tier": document["authority_tier"],
                    **record,
                })
            report.update({
                "document_id": document["document_id"],
                "record_count": len(document_records),
                "extracted_title": extracted_title,
            })
            retain_raw_only = not document_records and document.get("extraction_policy") == "retain_raw_only"
            document["extraction_status"] = (
                "raw_only_no_text" if retain_raw_only else "extracted" if document_records else "empty"
            )
            document["extracted_pages"] = report["page_count"]
            document["pages_with_text"] = report["pages_with_text"]
            document["extraction_warnings"] = report["warnings"]
            document_reports.append(report)
            if not document_records and not retain_raw_only:
                failures.append(f"{document['document_id']}: no semantic text records extracted")
        except Exception as exc:
            document["extraction_status"] = "failed"
            document["extraction_warnings"] = [f"{type(exc).__name__}: {exc}"]
            failures.append(f"{document['document_id']}: {type(exc).__name__}: {exc}")
            LOGGER.exception("Failed to extract %s", document["document_id"])

    EXTRACTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXTRACTED_PATH.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        json.dumps({
            "document_count": len(registry["documents"]),
            "record_count": len(records),
            "documents": document_reports,
            "failures": failures,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("Extraction was incomplete:\n" + "\n".join(failures))
    return {"document_count": len(registry["documents"]), "record_count": len(records)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    result = extract_registry(args.registry)
    LOGGER.info("Extracted %d records from %d source documents.", result["record_count"], result["document_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
