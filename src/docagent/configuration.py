"""Configuration for the document knowledge-base agent.

Every value can be overridden with an environment variable of the same name in
upper case (see ``.env.example``). The module-level ``DEFAULT_*`` constants are
used by ingestion + retrieval; the ``Configuration`` dataclass exposes the same
knobs to the LangGraph runtime via ``configurable``.
"""

import os
from dataclasses import dataclass, fields
from typing import Any

from langchain_core.runnables import RunnableConfig

# --- Embeddings & vector store ---
DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
DEFAULT_COLLECTION = os.environ.get("CHROMA_COLLECTION", "docagent")

# --- Chunking (ingest) ---
DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))

# --- Hybrid retrieval + rerank ---
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "4"))
# How many candidates each retriever (dense + BM25) contributes before fusion.
DEFAULT_CANDIDATE_K = int(os.environ.get("CANDIDATE_K", "20"))
# Cross-encoder used to re-rank fused candidates.
DEFAULT_RERANKER_MODEL = os.environ.get(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
# Minimum cross-encoder relevance score (a logit) to keep a chunk. Chunks below
# this are dropped, which is what lets the agent honestly say "not in the docs".
# Calibrated with scripts/calibrate_threshold.py: on the validation set in/out-of
# -scope rerank scores are well separated (in-scope ~2.6–7.2, out-of-scope ~ -11),
# so any threshold in [-2.5, 2.5] gives precision/recall/abstention = 1.0; 0.0 is
# a safe midpoint. Re-run the calibration script if you change corpus/reranker.
DEFAULT_SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.0"))
# RRF constant for reciprocal-rank fusion.
DEFAULT_RRF_K = int(os.environ.get("RRF_K", "60"))

# --- LLM (only the answer step may need a key) ---
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "openai:gpt-4.1")

# --- Claim verification (per-sentence citation entailment) ---
# Backend that checks each answer sentence is actually entailed by the retrieved
# evidence: "off" (default, no-op), "nli" (a local cross-encoder — offline, no
# API key), or "llm" (one structured grading call, reuses the answer model).
DEFAULT_ENTAILMENT_BACKEND = os.environ.get("ENTAILMENT_BACKEND", "off")
# NLI cross-encoder for the "nli" backend. Labels: contradiction/entailment/neutral.
DEFAULT_NLI_MODEL = os.environ.get("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")


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
    entailment_backend: str = DEFAULT_ENTAILMENT_BACKEND
    nli_model: str = DEFAULT_NLI_MODEL

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "Configuration":
        """Create a Configuration from env vars first, then ``configurable``."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
