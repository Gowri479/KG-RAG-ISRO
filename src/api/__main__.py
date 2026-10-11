"""Run the API: python -m src.api [--mock] [--host 127.0.0.1] [--port 8000]"""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="KG-RAG ISRO API")
    parser.add_argument("--mock", action="store_true", help="No language model; canned answers with real KG paths")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ["KG_RAG_API_MODE"] = "mock" if args.mock else os.environ.get("KG_RAG_API_MODE", "live")
    uvicorn.run("src.api.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
