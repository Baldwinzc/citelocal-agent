"""Persistent, memory-mappable BM25 sparse index (backed by ``bm25s``).

The original retriever loaded **every** chunk into RAM and built an in-memory
``BM25Okapi`` at construction — fine for a few thousand chunks, fatal at 100k+.
This module builds the sparse index **once at ingest** and memory-maps it at query
time, so the retriever no longer pays an O(corpus) RAM cost just to start up.

Layout (next to the Chroma store): ``{chroma_path}/bm25s/{collection}/``
    - the bm25s index files (vocab, scores, params)
    - ``chunk_ids.json``  — row index -> Chroma chunk_id (== Chroma doc id)
    - ``sources.json``    — distinct source paths (so list_sources stays O(1))

``HybridRetriever`` falls back to an in-memory build when this index is absent
(small/throwaway KBs, or a store created before the index existed), so nothing
breaks if ingest hasn't built it yet.
"""

import json
import shutil
from pathlib import Path

import bm25s

# Tokenizer config — must match between build and query. ``rank-bm25``'s old
# tokenizer differed slightly, so re-validate retrieval/threshold after switching.
_STOPWORDS = "en"
_IDS_FILE = "chunk_ids.json"
_SOURCES_FILE = "sources.json"
_PARAMS_FILE = "params.index.json"  # written by bm25s.save; used as an existence marker


def index_dir(persist_directory: str, collection_name: str) -> Path:
    return Path(persist_directory) / "bm25s" / collection_name


def exists(persist_directory: str, collection_name: str) -> bool:
    d = index_dir(persist_directory, collection_name)
    return (d / _IDS_FILE).exists() and (d / _PARAMS_FILE).exists()


def clear(persist_directory: str, collection_name: str) -> None:
    d = index_dir(persist_directory, collection_name)
    if d.exists():
        shutil.rmtree(d)


def build(
    persist_directory: str,
    collection_name: str,
    chunk_ids: list[str],
    texts: list[str],
    sources: list[str],
) -> None:
    """Build + persist the sparse index for the whole collection (full rebuild)."""
    d = index_dir(persist_directory, collection_name)
    clear(persist_directory, collection_name)
    if not chunk_ids:
        return  # empty collection -> no index (retriever treats it as empty)
    d.mkdir(parents=True, exist_ok=True)
    retriever = bm25s.BM25()
    retriever.index(
        bm25s.tokenize(texts, stopwords=_STOPWORDS, show_progress=False),
        show_progress=False,
    )
    retriever.save(str(d), show_progress=False)
    (d / _IDS_FILE).write_text(json.dumps(chunk_ids), encoding="utf-8")
    (d / _SOURCES_FILE).write_text(json.dumps(sorted(set(sources))), encoding="utf-8")


def load_sources(persist_directory: str, collection_name: str) -> list[str]:
    d = index_dir(persist_directory, collection_name)
    f = d / _SOURCES_FILE
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


class SparseIndex:
    """A memory-mapped bm25s index; maps query -> top candidate chunk_ids."""

    def __init__(self, persist_directory: str, collection_name: str):
        d = index_dir(persist_directory, collection_name)
        self.chunk_ids: list[str] = json.loads(
            (d / _IDS_FILE).read_text(encoding="utf-8")
        )
        self.bm25 = bm25s.BM25.load(str(d), mmap=True, load_corpus=False)

    @property
    def num_docs(self) -> int:
        return len(self.chunk_ids)

    def query(self, text: str, k: int) -> list[str]:
        if not self.chunk_ids:
            return []
        k = min(k, len(self.chunk_ids))
        tokens = bm25s.tokenize(text, stopwords=_STOPWORDS, show_progress=False)
        results, _scores = self.bm25.retrieve(tokens, k=k, show_progress=False)
        return [self.chunk_ids[i] for i in results[0]]
