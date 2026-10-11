"""Collect and extract the official Aditya-L1 text corpus.

Only the official ISSDC, PRADAN, and ALPPS URLs listed in ``SOURCES`` are
used. PDF text is emitted one page per Markdown file so the existing chunker
can retain document and page provenance. Scientific data products are never
downloaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = ROOT / "data" / "documents" / "aditya_l1" / "official"
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = ROOT / "data" / "source_manifest.csv"

SOURCES = [
    {"document_id": "ADITYA_L1_MISSION_BOOKLET", "title": "Aditya-L1 Mission Booklet", "organization": "ISSDC", "document_type": "mission booklet", "url": "https://www.issdc.gov.in/docs/Aditya/Aditya_L1_second_booklet.pdf", "filename": "Aditya_L1_second_booklet.pdf", "authority": "AUTHORITATIVE CORE: ISSDC", "publication_date": ""},
    {"document_id": "ADITYA_L1_PAYLOAD", "title": "Aditya-L1 Payloads", "organization": "ISSDC", "document_type": "payload document", "url": "https://www.issdc.gov.in/docs/Aditya/Al1_Payload.pdf", "filename": "Al1_Payload.pdf", "authority": "AUTHORITATIVE CORE: ISSDC", "publication_date": ""},
    {"document_id": "ADITYA_L1_ALPPS_V1", "title": "Aditya-L1 Proposal Processing System (ALPPS) User Guide", "organization": "ALPPS/ISRO", "document_type": "user guide", "url": "https://alpps.issdc.gov.in/web/extensions/Aditya-l1/resources/docs/ALPPS_UserGuide_V1.0.pdf", "filename": "ALPPS_UserGuide_V1.0.pdf", "authority": "AUTHORITATIVE CORE: ALPPS/ISRO", "publication_date": "2026-01-01"},
    {"document_id": "ADITYA_L1_ISSDC_PAGE", "title": "Aditya-L1 Mission", "organization": "ISSDC", "document_type": "official mission page", "url": "https://www.issdc.gov.in/adityal1.html", "filename": "adityal1.html", "authority": "SUPPLEMENTARY OFFICIAL SOURCE: ISSDC", "publication_date": ""},
    {"document_id": "ADITYA_L1_PRADAN_ARCHIVE", "title": "Aditya-L1 PRADAN Archive", "organization": "PRADAN/ISSDC", "document_type": "official data archive landing page", "url": "https://pradan1.issdc.gov.in/al1/", "filename": "pradan_al1.html", "authority": "SUPPLEMENTARY OFFICIAL SOURCE: PRADAN/ISSDC", "publication_date": ""},
    {"document_id": "ADITYA_L1_PRADAN_FAQ", "title": "Aditya-L1 PRADAN FAQ", "organization": "PRADAN/ISSDC", "document_type": "official FAQ", "url": "https://pradan1.issdc.gov.in/al1/faq.xhtml", "filename": "pradan_al1_faq.xhtml", "authority": "SUPPLEMENTARY OFFICIAL SOURCE: PRADAN/ISSDC", "publication_date": ""},
    {"document_id": "ADITYA_L1_ALPPS_HOME", "title": "Aditya-L1 Proposal Processing System", "organization": "ALPPS/ISRO", "document_type": "official portal landing page", "url": "https://alpps.issdc.gov.in/", "filename": "alpps_home.html", "authority": "SUPPLEMENTARY OFFICIAL SOURCE: ALPPS/ISRO", "publication_date": ""},
]


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip_depth and data.strip():
            self.parts.append(" ".join(data.split()))


def fetch_missing_sources() -> dict[str, str]:
    statuses: dict[str, str] = {}
    OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        destination = OFFICIAL_DIR / source["filename"]
        if destination.exists() and destination.stat().st_size > 0:
            statuses[source["document_id"]] = "downloaded"
            continue
        try:
            request = Request(source["url"], headers={"User-Agent": "KG-RAG-ISRO research corpus collector"})
            with urlopen(request, timeout=120) as response, destination.open("wb") as output:
                output.write(response.read())
            statuses[source["document_id"]] = "downloaded"
        except Exception as exc:
            statuses[source["document_id"]] = f"download_failed: {exc}"
    return statuses


def _header(source: dict, page: int, section: str) -> str:
    return "\n".join([
        f"Document ID: {source['document_id']}",
        f"Title: {source['title']}",
        "Mission: Aditya-L1",
        f"Organization: {source['organization']}",
        f"Document Type: {source['document_type']}",
        f"Source URL: {source['url']}",
        f"Page: {page}",
        f"Section: {section}",
        "Authority: " + source["authority"],
        "---",
    ])


def extract_pdf(source: dict) -> tuple[int, int, str]:
    from pypdf import PdfReader

    pdf_path = OFFICIAL_DIR / source["filename"]
    if not pdf_path.exists():
        return 0, 0, "missing local PDF"
    reader = PdfReader(str(pdf_path))
    pages_with_text = 0
    for page_number, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            pages_with_text += 1
        output = RAW_DIR / f"{source['document_id']}_page_{page_number:04d}.md"
        if text:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(_header(source, page_number, f"PDF page {page_number}") + "\n" + text + "\n", encoding="utf-8")
        elif output.exists():
            output.unlink()
    error = "" if pages_with_text else "image-only PDF; archived but not ingested"
    return len(reader.pages), pages_with_text, error


def extract_html(source: dict) -> tuple[int, int, str]:
    html_path = OFFICIAL_DIR / source["filename"]
    if not html_path.exists():
        return 0, 0, "missing local HTML source"
    parser = VisibleTextParser()
    parser.feed(html_path.read_text(encoding="utf-8", errors="ignore"))
    text = "\n".join(parser.parts).strip()
    output = RAW_DIR / f"{source['document_id']}_page_0001.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_header(source, 1, "Official web page") + "\n" + text + "\n", encoding="utf-8")
    return 1, int(bool(text)), ""


def write_manifest(statuses: dict[str, str], extraction: dict[str, dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = ["document_id", "title", "mission", "organization", "document_type", "publication_date", "source_url", "local_filename", "authority", "status", "pages", "pages_with_text", "extraction_error", "usage_notes"]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in SOURCES:
            item = extraction.get(source["document_id"], {})
            status = statuses.get(source["document_id"], "not_processed")
            writer.writerow({
                "document_id": source["document_id"], "title": source["title"], "mission": "Aditya-L1", "organization": source["organization"],
                "document_type": source["document_type"], "publication_date": source["publication_date"], "source_url": source["url"],
                "local_filename": str((OFFICIAL_DIR / source["filename"]).relative_to(ROOT)) if (OFFICIAL_DIR / source["filename"]).exists() else "",
                "authority": source["authority"], "status": status, "pages": item.get("pages", 0), "pages_with_text": item.get("pages_with_text", 0),
                "extraction_error": item.get("error", ""), "usage_notes": "Text corpus only; scientific data products were not downloaded.",
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and extract official Aditya-L1 text sources.")
    parser.add_argument("--download", action="store_true", help="Download missing official sources; existing files are reused.")
    args = parser.parse_args()
    if args.download:
        statuses = fetch_missing_sources()
    else:
        previous_report_path = ROOT / "data" / "results" / "aditya_l1_collection_report.json"
        previous_statuses = {}
        if previous_report_path.exists():
            previous = json.loads(previous_report_path.read_text(encoding="utf-8"))
            previous_statuses = previous.get("download_status", {})
        statuses = {
            source["document_id"]: (
                "downloaded" if (OFFICIAL_DIR / source["filename"]).exists()
                else previous_statuses.get(source["document_id"], "not_downloaded")
            )
            for source in SOURCES
        }
    extraction: dict[str, dict] = {}
    for source in SOURCES:
        if not (OFFICIAL_DIR / source["filename"]).exists():
            extraction[source["document_id"]] = {"pages": 0, "pages_with_text": 0, "error": "source unavailable; no substitute used"}
            continue
        try:
            if source["filename"].lower().endswith(".pdf"):
                pages, pages_with_text, error = extract_pdf(source)
            else:
                pages, pages_with_text, error = extract_html(source)
            extraction[source["document_id"]] = {"pages": pages, "pages_with_text": pages_with_text, "error": error}
        except Exception as exc:
            extraction[source["document_id"]] = {"pages": 0, "pages_with_text": 0, "error": str(exc)}
    write_manifest(statuses, extraction)
    report = {"sources": SOURCES, "download_status": statuses, "extraction": extraction, "raw_output": "data/raw/ADITYA_L1_*_page_*.md", "manifest": str(MANIFEST_PATH.relative_to(ROOT))}
    report_path = ROOT / "data" / "results" / "aditya_l1_collection_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
