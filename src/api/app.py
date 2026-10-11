"""KG-RAG HTTP API for the frontend.

    python -m src.api                # live: needs Ollama + local index/chunks
    python -m src.api --mock         # mock: no model, git-tracked files only

Interactive docs at http://127.0.0.1:8000/docs. Environment:
KG_RAG_API_MODE (live|mock), KG_RAG_CORS_ORIGINS (comma-separated),
KG_RAG_MOCK_DELAY_MS (simulated latency in mock mode).
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.pipeline import GenerationUnavailable, LivePipeline, MockPipeline, ollama_reachable
from src.api.schemas import AskRequest, AskResponse, ErrorResponse, Example, ExamplesResponse, HealthResponse
from src.evaluation.build_corrected_rerun_human_package import BENCHMARK
from src.generator.ollama_api import MODEL_NAME

DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"


def make_pipeline(mode: str | None = None):
    mode = (mode or os.environ.get("KG_RAG_API_MODE", "live")).lower()
    if mode == "mock":
        return MockPipeline(delay_ms=int(os.environ.get("KG_RAG_MOCK_DELAY_MS", "800")))
    if mode == "live":
        return LivePipeline()
    raise ValueError(f"KG_RAG_API_MODE must be 'live' or 'mock', not {mode!r}")


def create_app(mode: str | None = None) -> FastAPI:
    pipeline = make_pipeline(mode)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        threading.Thread(target=pipeline.warm_up, daemon=True).start()
        yield

    app = FastAPI(
        title="KG-RAG ISRO API",
        version="1.0.0",
        description="Question answering over official ISRO/ISSDC documents with a provenance-linked knowledge graph "
        "(relation-aware KG-RAG). Every answer comes with the KG paths and official sources behind it.",
        lifespan=lifespan,
    )
    app.state.pipeline = pipeline
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in os.environ.get("KG_RAG_CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        """Poll until status is 'ready' before enabling the ask button."""
        return HealthResponse(
            status=pipeline.status,
            mode=pipeline.mode,
            model=MODEL_NAME,
            ollama_reachable=ollama_reachable() if pipeline.mode == "live" else None,
            detail=pipeline.detail,
        )

    @app.get("/api/examples", response_model=ExamplesResponse, tags=["questions"])
    def examples() -> ExamplesResponse:
        """Example questions from the evaluation benchmark, for suggestion chips."""
        return ExamplesResponse(examples=_examples())

    @app.post(
        "/api/ask",
        response_model=AskResponse,
        tags=["questions"],
        responses={503: {"model": ErrorResponse, "description": "Still warming up, or the language model is unavailable."}},
    )
    def ask(request: AskRequest) -> AskResponse:
        """Answer a question. Live answers take roughly 10-30 seconds on the local GPU."""
        if pipeline.status != "ready":
            raise HTTPException(503, pipeline.detail or "The service is still warming up.")
        try:
            return pipeline.answer(request.question.strip())
        except GenerationUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc

    return app


@lru_cache(maxsize=1)
def _examples() -> list[Example]:
    seen, examples = set(), []
    for q in json.loads(BENCHMARK.read_text(encoding="utf8"))["questions"]:
        key = " ".join(q["question"].casefold().split())
        if key not in seen:
            seen.add(key)
            examples.append(Example(question=q["question"], mission=q["mission"], category=q["category"]))
    return examples


app = create_app()
