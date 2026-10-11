"""Question answering behind the API: relation-aware KG-RAG (system A).

Live mode reproduces system A from the pre-registered corrected rerun:
dense + BM25 retrieval, relation-aware KG paths, KG-first evidence fusion,
a 1,500-token evidence budget and the same generation settings.

Mock mode needs only git-tracked files and no Ollama: KG retrieval is real,
answers are canned. Benchmark questions return system A's actual rerun answer.
"""

from __future__ import annotations

import json
import re
import threading
import time
from functools import lru_cache

import requests

from src.api.schemas import AskResponse, Hop, KGPath, Passage, QueryAnalysis, Source, Usage
from src.evaluation.run_corrected_rerun import OPTIONS, RESULTS as RERUN_RESULTS
from src.evaluation.run_relation_aware_experiment import CONTEXT_LIMIT
from src.generator.kg_grounded_generator import _evidence_budget
from src.generator.ollama_api import MODEL_NAME, OLLAMA_URL, UNKNOWN
from src.generator.prompt import SYSTEM_PROMPT
from src.generator.token_budget import EVIDENCE, trim_to_tokens
from src.retriever.relation_aware import RelationAwareKG, RelationEvidence, fuse_relation_evidence

MAX_PATHS_SHOWN = 10
EXCERPT_CHARS = 600
SOURCE_LINE = re.compile(r"^#\s*Source:\s*(\S+)")


class GenerationUnavailable(RuntimeError):
    """The language model could not be reached or failed to answer."""


def _norm(text: str) -> str:
    return " ".join((text or "").casefold().split())


@lru_cache(maxsize=1)
def knowledge_graph() -> RelationAwareKG:
    return RelationAwareKG()


@lru_cache(maxsize=1)
def triples_by_id() -> dict[str, dict]:
    return {t["triple_id"]: t for t in knowledge_graph().triples}


def _source(triple: dict) -> Source:
    page = triple.get("source_page")
    return Source(
        document_id=triple.get("source_document") or "unknown",
        url=triple.get("source_url") or None,
        section=triple.get("source_section") or None,
        page=None if page in (None, "None", "") else str(page),
        excerpt=" ".join((triple.get("source_text") or "").split())[:EXCERPT_CHARS] or None,
    )


def to_kg_paths(paths: list[RelationEvidence]) -> list[KGPath]:
    triples = triples_by_id()
    shown = []
    for path in paths[:MAX_PATHS_SHOWN]:
        hops = []
        for triple_id in path.triple_ids:
            t = triples[triple_id]
            hops.append(Hop(subject=t["subject"], relation=t["relation"], object=t["object"], triple_id=triple_id, source=_source(t)))
        shown.append(KGPath(path_id=path.path_id, score=path.score, hops=hops))
    return shown


def analysis_for(question: str) -> QueryAnalysis:
    intent = knowledge_graph().analyze(question)
    return QueryAnalysis(
        query_type=intent.query_type,
        entities=list(intent.entity_ids),
        relations=list(intent.relation_types),
        hop_depth=intent.hop_depth,
    )


def ollama_reachable() -> bool:
    try:
        return requests.get(OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=2).ok
    except requests.RequestException:
        return False


class LivePipeline:
    mode = "live"

    def __init__(self) -> None:
        self.status = "warming_up"
        self.detail: str | None = "Loading knowledge graph, index and embedding model."
        self._generation_lock = threading.Lock()

    def warm_up(self) -> None:
        """Load every heavy resource once; the first question is then as fast as the rest."""
        try:
            from src.baselines.bm25_llm import _load_bm25
            from src.retriever.embedding_model import get_embedding_model
            from src.retriever.faiss_retriever import _load_chunk_records, _load_index

            knowledge_graph()
            triples_by_id()
            _load_chunk_records()
            _load_index()
            _load_bm25()
            get_embedding_model()
            self.status, self.detail = "ready", None
        except Exception as exc:  # surfaced through /api/health
            self.status, self.detail = "error", f"{type(exc).__name__}: {exc}"

    def answer(self, question: str) -> AskResponse:
        from src.evaluation.evaluate_relational_qa import _bm25_trace, _vanilla_trace
        from src.generator.ollama_api import generate_with_metrics
        from src.retriever.faiss_retriever import _load_chunk_records

        started = time.perf_counter()
        dense, bm25 = _vanilla_trace(question), _bm25_trace(question)
        paths = knowledge_graph().retrieve(
            question,
            dict(zip(dense["retrieved_chunk_ids"], dense["retrieved_chunk_scores"])),
            dict(zip(bm25["retrieved_chunk_ids"], bm25["retrieved_chunk_scores"])),
        )
        fused = fuse_relation_evidence(paths, dense["retrieved_text"], [], CONTEXT_LIMIT)
        context = " ".join(fused["context_text"].split()[:CONTEXT_LIMIT])
        context = trim_to_tokens(context, _evidence_budget(SYSTEM_PROMPT, question, EVIDENCE, OPTIONS))
        with self._generation_lock:
            answer, telemetry = generate_with_metrics(question, context, options=OPTIONS)
        if telemetry.get("status") != "ok":
            raise GenerationUnavailable(f"Language model unavailable ({telemetry.get('failure') or telemetry.get('status')}). Is Ollama running?")

        records = _load_chunk_records()
        passages = []
        for rank, (chunk_id, text) in enumerate(zip(dense["retrieved_chunk_ids"], dense["retrieved_text"]), 1):
            meta = records[int(chunk_id)] if chunk_id.isdigit() and int(chunk_id) < len(records) else {}
            url = SOURCE_LINE.match(text)
            passages.append(Passage(rank=rank, text=text, url=url.group(1) if url else None, title=meta.get("title"), mission=meta.get("mission"), source_file=meta.get("source_file")))
        return AskResponse(
            question=question,
            answer=answer,
            abstained=_norm(answer) == _norm(UNKNOWN),
            mode="live",
            analysis=analysis_for(question),
            kg_paths=to_kg_paths(paths),
            kg_paths_total=len(paths),
            passages=passages,
            usage=Usage(
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt_tokens=telemetry.get("prompt_tokens"),
                context_tokens_used=telemetry.get("context_tokens_used"),
                context_trimmed=telemetry.get("context_trimmed"),
            ),
        )


class MockPipeline:
    mode = "mock"
    status = "ready"
    detail = "Mock mode: real KG retrieval, canned answers, no language model."

    def __init__(self, delay_ms: int = 800) -> None:
        self.delay_ms = delay_ms

    def warm_up(self) -> None:
        knowledge_graph()
        triples_by_id()

    @staticmethod
    @lru_cache(maxsize=1)
    def _rerun_answers() -> dict[str, str]:
        answers = {}
        for line in RERUN_RESULTS.read_text(encoding="utf8").splitlines():
            row = json.loads(line) if line.strip() else None
            if row and row["system"] == "A_CURRENT":
                answers.setdefault(_norm(row["question"]), row["answer"])
        return answers

    def answer(self, question: str) -> AskResponse:
        started = time.perf_counter()
        paths = knowledge_graph().retrieve(question)
        answer = self._rerun_answers().get(_norm(question))
        if answer is None:
            if paths:
                top = paths[0]
                facts = "; ".join(f"{s} {r.replace('_', ' ').lower()} {o}" for s, r, o in zip(top.nodes[:-1], top.relations, top.nodes[1:]))
                answer = f"[MOCK ANSWER] Based on the knowledge graph: {facts}."
            else:
                answer = UNKNOWN
        time.sleep(self.delay_ms / 1000)
        return AskResponse(
            question=question,
            answer=answer,
            abstained=_norm(answer) == _norm(UNKNOWN),
            mode="mock",
            analysis=analysis_for(question),
            kg_paths=to_kg_paths(paths),
            kg_paths_total=len(paths),
            passages=[],
            usage=Usage(latency_ms=int((time.perf_counter() - started) * 1000)),
        )
