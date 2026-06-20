"""Configuration for the document knowledge-base agent.

Every value can be overridden with an environment variable of the same name in
upper case (see ``.env.example``). The module-level ``DEFAULT_*`` constants are
used by ingestion + retrieval; the ``Configuration`` dataclass exposes the same
knobs to the LangGraph runtime via ``configurable``.
"""

import json
import os
from dataclasses import dataclass, fields
from typing import Any

from langchain_core.runnables import RunnableConfig

# --- Embeddings & vector store ---
DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
DEFAULT_COLLECTION = os.environ.get("CHROMA_COLLECTION", "citelocal_agent")

# --- Chunking (ingest) ---
DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))

# --- Hybrid retrieval + rerank ---
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "4"))
# How many candidates each retriever (dense + BM25) contributes before fusion.
DEFAULT_CANDIDATE_K = int(os.environ.get("CANDIDATE_K", "20"))
# Cross-encoder used to re-rank fused candidates. Selected via a documented
# bake-off (docs/reranker-selection.md, scripts/rerank_bakeoff.py): bge-reranker-v2-m3
# tops multi-hop ranking. Its scores are ~[0,1] (sigmoid), NOT ms-marco's wide
# logits — so SCORE_THRESHOLD / SUPPORT_THRESHOLD below are calibrated for this scale.
DEFAULT_RERANKER_MODEL = os.environ.get(
    "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
)
# Abstention GATE: the top chunk must clear this for search() to return anything,
# which is what lets the agent honestly say "not in the docs". bge-reranker-v2-m3
# scores are ~[0,1] (sigmoid), NOT ms-marco logits. Calibrated with
# scripts/calibrate_threshold.py on offline_sample: in-scope top scores run ~0.14–0.99,
# refuse ~0.00–0.05 (two topically-near no_answer outliers at 0.52 / 0.86). 0.2 keeps
# answer recall ~0.99 while abstaining on ~0.86 of refuse cases (matching the prior
# model); pushing higher (the f1+abstain argmax was ~0.525) wrongly refuses ~10% of
# answerable questions. Re-run the calibration script if you change corpus/reranker.
DEFAULT_SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.2"))
# Looser bar for admitting *supporting* chunks AFTER the gate has opened (question is
# in-scope). Recovers multi-hop second sources the reranker scores low, without
# weakening abstention — the gate alone decides whether anything is returned. Clamped
# to <= SCORE_THRESHOLD in search(). 0.0 admits any non-zero-scored chunk into the
# top-k once answering; on offline_sample this lifts multi-hop single-shot recall to
# 0.79 -> 0.89 with refusal-leak unchanged. Set equal to SCORE_THRESHOLD to disable.
DEFAULT_SUPPORT_THRESHOLD = float(os.environ.get("SUPPORT_THRESHOLD", "0.0"))
# RRF constant for reciprocal-rank fusion.
DEFAULT_RRF_K = int(os.environ.get("RRF_K", "60"))

# --- LLM (only the answer step may need a key) ---
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "openai:gpt-4.1")

# --- Web search tools (opt-in; the project is local-first, so this is OFF by
# default — when off, the toolset/routing/tests are exactly as before). When on,
# the agent may also reach the public web; web results carry verifiable
# `web:<url>` citations, checked the same way as document locators. ---
DEFAULT_ENABLE_WEB_SEARCH = os.environ.get("ENABLE_WEB_SEARCH", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# "ddg" (DuckDuckGo, keyless) or "tavily" (higher quality, needs TAVILY_API_KEY).
DEFAULT_WEB_SEARCH_BACKEND = os.environ.get("WEB_SEARCH_BACKEND", "ddg")
DEFAULT_WEB_SEARCH_RESULTS = int(os.environ.get("WEB_SEARCH_RESULTS", "5"))
# Cap on characters of page text fetch_url returns (keeps the context bounded).
DEFAULT_WEB_FETCH_CHARS = int(os.environ.get("WEB_FETCH_CHARS", "4000"))


def llm_call_kwargs() -> dict:
    """Extra constructor kwargs for the chat model, from the environment.

    ``$LLM_EXTRA_BODY`` (a JSON object) is forwarded as ``extra_body`` to
    OpenAI-compatible providers — an escape hatch for provider-specific request
    fields (e.g. a reasoning/thinking toggle, sampling knobs some gateways expose)
    without touching code. Empty for stock OpenAI, so the default path is unchanged.
    """
    raw = os.environ.get("LLM_EXTRA_BODY")
    return {"extra_body": json.loads(raw)} if raw else {}

# --- Graph execution budgets ---
# ReAct loop steps for the retrieval loop: the simple path's response_agent AND
# each researcher inside the orchestrator. The orchestrator's own
# planner -> fan-out -> verify -> synth subgraph gets a separate, larger budget.
# The top-level graph (router -> one node) is tiny and uses LangGraph's default,
# so callers no longer pass recursion_limit by hand.
DEFAULT_RECURSION_LIMIT = int(os.environ.get("RECURSION_LIMIT", "12"))
DEFAULT_ORCHESTRATOR_RECURSION_LIMIT = int(
    os.environ.get("ORCHESTRATOR_RECURSION_LIMIT", "24")
)

# --- Claim verification (per-sentence citation entailment) ---
# Backend that checks each answer sentence is actually entailed by the retrieved
# evidence: "off" (default, no-op), "nli" (a local cross-encoder — offline, no
# API key), or "llm" (one structured grading call, reuses the answer model).
DEFAULT_ENTAILMENT_BACKEND = os.environ.get("ENTAILMENT_BACKEND", "off")
# NLI cross-encoder for the "nli" backend. Labels: contradiction/entailment/neutral.
DEFAULT_NLI_MODEL = os.environ.get("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")


def _coerce(value: Any, default: Any) -> Any:
    """Coerce a (possibly string) override to the type of the field's default."""
    if not isinstance(value, str):
        return value
    if isinstance(default, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, dict):
        return json.loads(value)
    return value


@dataclass(kw_only=True)
class Configuration:
    """Runtime-configurable parameters for the agent graph."""

    llm_model: str = DEFAULT_LLM_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    chroma_path: str = DEFAULT_CHROMA_PATH
    collection_name: str = DEFAULT_COLLECTION
    top_k: int = DEFAULT_TOP_K
    candidate_k: int = DEFAULT_CANDIDATE_K
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD
    recursion_limit: int = DEFAULT_RECURSION_LIMIT
    orchestrator_recursion_limit: int = DEFAULT_ORCHESTRATOR_RECURSION_LIMIT
    entailment_backend: str = DEFAULT_ENTAILMENT_BACKEND
    nli_model: str = DEFAULT_NLI_MODEL
    enable_web_search: bool = DEFAULT_ENABLE_WEB_SEARCH
    web_search_backend: str = DEFAULT_WEB_SEARCH_BACKEND
    web_search_results: int = DEFAULT_WEB_SEARCH_RESULTS
    web_fetch_chars: int = DEFAULT_WEB_FETCH_CHARS

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "Configuration":
        """Create a Configuration from env vars first, then ``configurable``.

        Env vars arrive as strings, so values are coerced to each field's type
        (e.g. ``RECURSION_LIMIT=20`` -> int 20, ``SCORE_THRESHOLD=0.5`` -> float).
        """
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {}
        for f in fields(cls):
            if not f.init:
                continue
            raw = os.environ.get(f.name.upper(), configurable.get(f.name))
            if raw is None or raw == "":
                continue
            values[f.name] = _coerce(raw, f.default)
        return cls(**values)
