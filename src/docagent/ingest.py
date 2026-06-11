#!/usr/bin/env python
"""Ingest local documents (Markdown / RST / text / PDF) into the Chroma KB.

Each chunk stores precise provenance: the **relative path** as ``source`` (so two
same-named files in different folders don't collide), a line range (or PDF page),
and a stable ``chunk_id`` used as the Chroma id.

Usage:
    python -m docagent.ingest --path ./corpus
    python -m docagent.ingest --path ./corpus --reset
"""

import argparse
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docagent import bm25_index
from docagent.configuration import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION,
)
from docagent.vectorstore import get_vectorstore

load_dotenv()

TEXT_EXTS = {".md", ".markdown", ".txt", ".rst"}
PDF_EXTS = {".pdf"}
SKIP_NAMES = {"LICENSE", "SOURCE.md"}  # corpus attribution files, not content


def _load_text(path: Path, source: str) -> List[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": source})]


def _load_pdf(path: Path, source: str) -> List[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": source, "page": i + 1},
                )
            )
    return docs


def load_documents(root: Path) -> List[Document]:
    """Load supported files; ``source`` is each file's path relative to ``root``."""
    if root.is_file():
        files, base = [root], root.parent
    else:
        files, base = sorted(root.rglob("*")), root

    docs: List[Document] = []
    for f in files:
        if not f.is_file() or f.name in SKIP_NAMES:
            continue
        source = f.relative_to(base).as_posix()  # relative path => unique identity
        ext = f.suffix.lower()
        if ext in TEXT_EXTS:
            docs.extend(_load_text(f, source))
        elif ext in PDF_EXTS:
            docs.extend(_load_pdf(f, source))
    return docs


def chunk_documents(
    raw_docs: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """Split docs and attach a unique chunk_id + line-range provenance."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks: List[Document] = []
    per_source: dict = {}
    for doc in raw_docs:
        text = doc.page_content
        source = doc.metadata.get("source", "unknown")
        for ch in splitter.split_documents([doc]):
            start_index = ch.metadata.get("start_index", 0)
            start_line = text[:start_index].count("\n") + 1
            end_line = start_line + ch.page_content.count("\n")
            idx = per_source.get(source, 0)
            per_source[source] = idx + 1
            # source is a relative path => chunk_id is globally unique
            ch.metadata["chunk_id"] = f"{source}::{idx}"
            ch.metadata["start_line"] = start_line
            ch.metadata["end_line"] = end_line
            chunks.append(ch)
    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Chroma knowledge base."
    )
    parser.add_argument("--path", required=True, help="File or directory to ingest.")
    parser.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--reset", action="store_true", help="Clear the collection before ingesting."
    )
    args = parser.parse_args()

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    print(f"Loading documents from {root} ...")
    raw_docs = load_documents(root)
    if not raw_docs:
        raise SystemExit("No supported documents found (.md/.markdown/.txt/.rst/.pdf).")
    print(f"Loaded {len(raw_docs)} raw document section(s).")

    chunks = chunk_documents(raw_docs, args.chunk_size, args.chunk_overlap)
    print(f"Split into {len(chunks)} chunks.")

    vs = get_vectorstore(
        persist_directory=args.chroma_path, collection_name=args.collection
    )

    if args.reset:
        try:
            existing = vs.get()
            ids = existing.get("ids", [])
            if ids:
                vs.delete(ids=ids)
                print(f"Reset: removed {len(ids)} existing chunks.")
        except Exception as e:  # noqa: BLE001
            print(f"Reset skipped: {e}")

    vs.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    print(
        f"Ingested {len(chunks)} chunks into collection "
        f"'{args.collection}' at {args.chroma_path}."
    )

    # Rebuild the persistent sparse index over the *whole* collection, so the
    # retriever memory-maps it at query time instead of loading every chunk and
    # building BM25 in RAM at startup. A full rebuild keeps append and reset
    # correct without incremental-update bookkeeping (ingest is a batch job).
    all_data = vs.get()
    all_metas = all_data.get("metadatas", []) or []
    bm25_index.build(
        args.chroma_path,
        args.collection,
        all_data.get("ids", []) or [],
        all_data.get("documents", []) or [],
        [(m or {}).get("source", "unknown") for m in all_metas],
    )
    print(f"Built sparse BM25 index over {len(all_metas)} chunks.")
    print("Done.")


if __name__ == "__main__":
    main()
