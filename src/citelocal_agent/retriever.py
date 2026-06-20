"""Hybrid retriever — citelocal_agent's retrieval core.

Pipeline:  dense (vector) + sparse (BM25)  ->  Reciprocal-Rank-Fusion
           ->  cross-encoder rerank  ->  relevance-threshold filter.

This lifts results above naive top-k cosine similarity, and the threshold gives
the agent a principled way to say "not in the docs" (everything fell below it).
Each result carries precise provenance (file + line range, or PDF page) for
exact citations.

**Scale:** the sparse side is a persistent, memory-mapped ``bm25s`` index built at
ingest (see ``bm25_index``), and chunk text is fetched **only for the fused
candidates** via ``vs.get(ids=...)`` — so startup no longer loads the whole corpus
into RAM. When no persistent index exists (a throwaway/small KB, or a store built
before the index existed), it transparently falls back to the in-memory build.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from citelocal_agent import bm25_index
from citelocal_agent.configuration import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RRF_K,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_SUPPORT_THRESHOLD,
    DEFAULT_TOP_K,
)
from citelocal_agent.vectorstore import get_vectorstore


@dataclass
class RetrievedChunk:
    """A retrieved chunk with provenance and a cross-encoder relevance score."""

    text: str
    source: str
    chunk_id: str
    score: float
    start_line: int | None = None
    end_line: int | None = None
    page: int | None = None

    @property
    def locator(self) -> str:
        """Precise, human-readable source locator used in citations."""
        if self.page is not None:
            return f"{self.source} (p.{self.page})"
        if self.start_line is not None:
            return f"{self.source}:L{self.start_line}-{self.end_line}"
        return self.source


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"\W+", text.lower()) if t]


@lru_cache(maxsize=2)
def _get_reranker(model_name: str = DEFAULT_RERANKER_MODEL) -> CrossEncoder:
    return CrossEncoder(model_name)


class HybridRetriever:
    """Dense + BM25 retrieval, RRF fusion, cross-encoder rerank, threshold."""

    def __init__(
        self,
        persist_directory: str = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vs = get_vectorstore(
            persist_directory=persist_directory, collection_name=collection_name
        )

        if bm25_index.exists(persist_directory, collection_name):
            # Scalable path: memory-mapped sparse index, text fetched on demand.
            self._sparse: bm25_index.SparseIndex | None = bm25_index.SparseIndex(
                persist_directory, collection_name
            )
            self._sources = bm25_index.load_sources(persist_directory, collection_name)
            self._num_chunks = self._sparse.num_docs
            self._fallback_by_id: dict | None = None
            self._fallback_bm25 = None
            self._fallback_ids: List[str] = []
        else:
            # Fallback: build in memory from the full collection (old behaviour;
            # fine for throwaway/small KBs or stores predating the sparse index).
            data = self.vs.get()
            ids = data.get("ids", []) or []
            docs = data.get("documents", []) or []
            metas = data.get("metadatas", []) or []
            self._sparse = None
            self._fallback_ids = ids
            self._fallback_by_id = {
                cid: (text, meta or {}) for cid, text, meta in zip(ids, docs, metas)
            }
            self._fallback_bm25 = (
                BM25Okapi([_tokenize(d) for d in docs]) if docs else None
            )
            self._sources = sorted({(m or {}).get("source", "unknown") for m in metas})
            self._num_chunks = len(ids)

        # The cross-encoder is loaded lazily (only when a search actually runs),
        # so cheap guards like `is_empty` don't drag in the reranker model.
        self._reranker_model = reranker_model
        self._reranker_obj: CrossEncoder | None = None

    @property
    def _reranker(self) -> CrossEncoder:
        if self._reranker_obj is None:
            self._reranker_obj = _get_reranker(self._reranker_model)
        return self._reranker_obj

    @property
    def num_chunks(self) -> int:
        return self._num_chunks

    @property
    def is_empty(self) -> bool:
        return self._num_chunks == 0

    def _dense_ids(self, query: str, candidate_k: int) -> List[str]:
        ids: List[str] = []
        for d in self.vs.similarity_search(query, k=candidate_k):
            cid = (d.metadata or {}).get("chunk_id")
            if cid is not None:
                ids.append(cid)
        return ids

    def _bm25_ids(self, query: str, candidate_k: int) -> List[str]:
        if self._sparse is not None:
            return self._sparse.query(query, candidate_k)
        if self._fallback_bm25 is None:
            return []
        scores = self._fallback_bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._fallback_ids[i] for i in ranked[:candidate_k]]

    def _text_meta(self, ids: List[str]) -> dict:
        """Return ``{chunk_id: (text, meta)}`` for the given ids only.

        Scalable path fetches just these ids from Chroma (no full-corpus load);
        fallback path reads the in-memory map.
        """
        if self._sparse is None:
            by_id = self._fallback_by_id or {}
            return {cid: by_id[cid] for cid in ids if cid in by_id}
        if not ids:
            return {}
        data = self.vs.get(ids=ids)
        return {
            cid: (text or "", meta or {})
            for cid, text, meta in zip(
                data.get("ids", []), data.get("documents", []), data.get("metadatas", [])
            )
        }

    @staticmethod
    def _rrf(rank_lists: List[List[str]], rrf_k: int) -> List[str]:
        fused: dict = {}
        for ids in rank_lists:
            for rank, cid in enumerate(ids):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        return sorted(fused, key=lambda c: fused[c], reverse=True)

    def search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        rrf_k: int = DEFAULT_RRF_K,
        support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
    ) -> List[RetrievedChunk]:
        if self.is_empty:
            return []
        dense = self._dense_ids(query, candidate_k)
        sparse = self._bm25_ids(query, candidate_k)
        fused = self._rrf([dense, sparse], rrf_k)[:candidate_k]
        if not fused:
            return []

        by_id = self._text_meta(fused)
        fused = [c for c in fused if c in by_id]
        if not fused:
            return []

        pairs = [(query, by_id[cid][0]) for cid in fused]
        scores = self._reranker.predict(pairs)  # type: ignore[arg-type]
        ranked = sorted(zip(fused, scores), key=lambda x: float(x[1]), reverse=True)

        # Two-threshold selection. The strict `score_threshold` is the abstention
        # GATE: unless the top chunk clears it, return nothing (the agent then says
        # "not in the docs"). Once the gate is open the question is in-scope, so we
        # admit *supporting* chunks down to the looser `support_threshold` — this is
        # what lets a multi-hop second source the reranker scored mildly negative in
        # (it never widens abstention, since the gate alone decides empty-vs-not).
        if not ranked or float(ranked[0][1]) < score_threshold:
            return []
        admit = min(support_threshold, score_threshold)  # never stricter than gate

        out: List[RetrievedChunk] = []
        for cid, score in ranked:
            if float(score) < admit:
                continue
            text, meta = by_id[cid]
            out.append(
                RetrievedChunk(
                    text=text,
                    source=meta.get("source", "unknown"),
                    chunk_id=cid,
                    score=float(score),
                    start_line=meta.get("start_line"),
                    end_line=meta.get("end_line"),
                    page=meta.get("page"),
                )
            )
            if len(out) >= k:
                break
        return out

    def list_sources(self) -> List[str]:
        return sorted(self._sources)


@lru_cache(maxsize=2)
def get_retriever(
    persist_directory: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION,
) -> HybridRetriever:
    """Cached retriever (loads the mmap sparse index + lazy cross-encoder once)."""
    return HybridRetriever(
        persist_directory=persist_directory, collection_name=collection_name
    )
