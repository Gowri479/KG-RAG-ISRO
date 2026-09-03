"""CLI query entrypoint for local KG + vector retrieval and generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.generator.ollama_api import generate
    from src.retriever.hybrid import retrieve
except ImportError:  # pragma: no cover
    from generator.ollama_api import generate
    from retriever.hybrid import retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve and answer a question with the local KG + FAISS pipeline.")
    parser.add_argument("--question", required=True, help="Question to answer from the local corpus.")
    args = parser.parse_args()

    context = retrieve(args.question)
    answer = generate(args.question, context)
    print(answer)


if __name__ == "__main__":
    main()

