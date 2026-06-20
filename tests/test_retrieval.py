"""Retrieval tests — no LLM API key required.

These exercise the full retrieval stack (hybrid dense+BM25 -> RRF ->
cross-encoder rerank -> relevance threshold) against the bundled `sample_notes`,
so the suite always runs green offline / in CI without downloading any papers.
"""

from pathlib import Path

import pytest

from citelocal_agent import bm25_index
from citelocal_agent.ingest import chunk_documents, load_documents
from citelocal_agent.retriever import HybridRetriever
from citelocal_agent.vectorstore import get_vectorstore

NOTES = Path(__file__).parent.parent / "sample_notes"


def test_load_notes():
    docs = load_documents(NOTES)
    assert len(docs) >= 3
    sources = {d.metadata["source"] for d in docs}
    assert "transformers-and-attention.md" in sources


@pytest.fixture(scope="module", params=["persistent", "fallback"])
def kb(request, tmp_path_factory):
    """Build a throwaway KB from sample_notes; exercise BOTH retriever paths.

    ``persistent`` builds the bm25s index, which the retriever memory-maps at
    query time (the production path); ``fallback`` skips it so the retriever
    builds BM25 in memory. Every retrieval test below thus runs on both.
    """
    chroma_path = str(tmp_path_factory.mktemp("chroma"))
    docs = load_documents(NOTES)
    chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=120)
    vs = get_vectorstore(persist_directory=chroma_path, collection_name="test_kb")
    vs.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    if request.param == "persistent":
        data = vs.get()
        metas = data.get("metadatas", []) or []
        bm25_index.build(
            chroma_path,
            "test_kb",
            data.get("ids", []) or [],
            data.get("documents", []) or [],
            [(m or {}).get("source", "unknown") for m in metas],
        )
    return HybridRetriever(persist_directory=chroma_path, collection_name="test_kb")


def test_num_chunks_and_sources(kb):
    assert kb.num_chunks > 0 and not kb.is_empty
    assert "transformers-and-attention.md" in kb.list_sources()


def test_hybrid_retrieval_hits_relevant_note(kb):
    hits = kb.search("what is scaled dot-product attention", k=3)
    assert hits, "expected at least one hit"
    assert any("attention" in h.source for h in hits)
    assert all(h.start_line is not None and h.end_line is not None for h in hits)


def test_hybrid_retrieval_bm25_keyword(kb):
    # BM25 should help match the exact term "BM25" even if dense misses it
    hits = kb.search("BM25 sparse retrieval", k=3)
    assert any("retrieval" in h.source for h in hits)


def test_threshold_filters_out_of_scope(kb):
    hits = kb.search("what is the capital of France", k=3)
    assert hits == []


def test_gate_protects_abstention_even_with_loose_support(kb):
    # The strict score_threshold is the abstention GATE: even a very loose
    # support_threshold must not make an out-of-scope query retrieve anything.
    assert kb.search("what is the capital of France", k=4, support_threshold=-100.0) == []


def test_support_threshold_admits_supporting_chunks_after_gate(kb):
    # Once the gate is open (in-scope), a looser support_threshold admits further,
    # mildly-negative supporting chunks the strict gate alone would drop.
    q = "what is scaled dot-product attention"
    strict = kb.search(q, k=6, support_threshold=0.0)        # gate only (>= 0.0)
    loose = kb.search(q, k=6, support_threshold=-100.0)       # admit down to -100
    assert loose and loose[0].score >= 0.0                    # gate opened
    assert all(h.score >= 0.0 for h in strict)                # strict keeps only >= gate
    assert len(loose) >= len(strict)                          # loosening never removes
    if len(loose) > len(strict):                              # extras are sub-gate
        assert any(h.score < 0.0 for h in loose)
