"""Clean raw markdown scraped from ISRO pages before chunking."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace to a single space while preserving line breaks."""
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_markdown_noise(text: str) -> str:
    """Remove boilerplate markdown and page elements that do not carry content."""
    if not text:
        return ""
    cleaned = text
    cleaned = re.sub(r"(?is)<!--.*?-->", " ", cleaned)
    cleaned = re.sub(r"(?is)<script.*?</script>", " ", cleaned)
    cleaned = re.sub(r"(?is)<style.*?</style>", " ", cleaned)
    cleaned = re.sub(r"(?m)^#+\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*(?:\[\d+\]|\(\d+\)|[•\-\*]|\d+\.)\s*", "", cleaned)
    cleaned = re.sub(r"(?i)\b(page|home|search|skip to content|last updated|print|share|menu)\b", " ", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "").replace("*", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def deduplicate_lines(text: str) -> str:
    """Drop repeated lines while keeping the first occurrence of each unique content block."""
    if not text:
        return ""
    seen: set[str] = set()
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)

    merged_text = "\n".join(lines).strip()
    merged_text = re.sub(r"(?i)\b([A-Za-z0-9-]+)\b(?:\s+\1\b)+", r"\1", merged_text)
    return merged_text.strip()


def clean_markdown(raw_text: str, source_url: str | None = None) -> str:
    """Return a clean, deduplicated markdown string suitable for chunking and indexing."""
    if raw_text is None:
        raise ValueError("raw_text must not be None")
    cleaned = normalize_whitespace(raw_text)
    cleaned = strip_markdown_noise(cleaned)
    cleaned = deduplicate_lines(cleaned)
    if source_url:
        cleaned = f"Source: {source_url}\n\n{cleaned}".strip()
    return cleaned


def clean_directory(input_dir: Path, output_dir: Path) -> list[Path]:
    """Clean all markdown files in a directory and save grouped output files."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_files: list[Path] = []
    for file_path in sorted(input_dir.glob("*.md")):
        try:
            raw_text = file_path.read_text(encoding="utf-8")
            cleaned = clean_markdown(raw_text, source_url=file_path.stem)
            output_path = output_dir / file_path.name
            output_path.write_text(cleaned, encoding="utf-8")
            cleaned_files.append(output_path)
        except Exception as exc:
            logger.exception("Failed to clean %s: %s", file_path, exc)
    return cleaned_files


def main() -> None:
    """CLI entry point for cleaning raw markdown files."""
    import argparse

    parser = argparse.ArgumentParser(description="Clean raw ISRO markdown before indexing.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"), help="Directory containing raw Markdown files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/chunks"), help="Output directory for cleaned text")
    args = parser.parse_args()
    clean_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main()

