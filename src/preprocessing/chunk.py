"""Chunk cleaned text into overlapping windows approximating 512-token blocks."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List

from tqdm import tqdm

logger = logging.getLogger(__name__)


def tokenize_for_chunking(text: str) -> list[str]:
    """Split text into tokens using a simple whitespace-based approximation."""
    return text.split()


def chunk_text(text: str, chunk_size: int = 512, stride: int = 128) -> list[str]:
    """Create overlapping word-based chunks with a stride, approximating token windows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if stride >= chunk_size:
        raise ValueError("stride must be smaller than chunk_size")

    tokens = tokenize_for_chunking(text)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunk = " ".join(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(tokens):
            break
        start += max(1, chunk_size - stride)
    return chunks


def chunk_markdown_file(input_path: Path, output_path: Path, chunk_size: int = 512, stride: int = 128) -> list[dict]:
    """Read a markdown file, chunk it, and save a JSON array of chunk metadata objects."""
    text = input_path.read_text(encoding="utf-8")
    chunks = chunk_text(text, chunk_size=chunk_size, stride=stride)
    payload = [
        {
            "chunk_index": index,
            "source_file": str(input_path),
            "chunk_size": len(chunk.split()),
            "text": chunk,
        }
        for index, chunk in enumerate(chunks)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def chunk_directory(input_dir: Path, output_dir: Path, chunk_size: int = 512, stride: int = 128) -> list[dict]:
    """Chunk every markdown file in the directory into JSON chunk records and a combined corpus file."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []
    for file_path in tqdm(sorted(input_dir.glob("*.md")), desc="Chunking markdown files", unit="file"):
        file_payload = chunk_markdown_file(file_path, output_dir / f"{file_path.stem}_chunks.json", chunk_size=chunk_size, stride=stride)
        all_chunks.extend(file_payload)

    combined_path = output_dir / "chunks.json"
    combined_path.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    return all_chunks


def main() -> None:
    """CLI entry point for chunk generation."""
    parser = argparse.ArgumentParser(description="Chunk cleaned Markdown files into overlapping passages.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"), help="Directory containing cleaned Markdown files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/chunks"), help="Directory for chunk JSON files")
    parser.add_argument("--chunk-size", type=int, default=512, help="Approximate chunk size in words")
    parser.add_argument("--stride", type=int, default=128, help="Overlap between adjacent chunks in words")
    args = parser.parse_args()
    chunk_directory(args.input_dir, args.output_dir, chunk_size=args.chunk_size, stride=args.stride)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main()

