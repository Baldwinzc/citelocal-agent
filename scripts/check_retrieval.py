#!/usr/bin/env python
"""Dev probe: inspect what the hybrid retriever returns for a few queries.

No LLM / API key needed — this exercises only the retrieval stack
(dense + BM25 -> RRF -> cross-encoder rerank -> threshold).

    python scripts/check_retrieval.py
"""

from docagent.retriever import get_retriever

QUERIES = [
    "How does FastAPI handle a normal def (sync) path operation function?",
    "How do I declare a path parameter with an integer type?",
    "What does PEP 484 specify about type hints?",        # -> peps/*.rst
    "What are protocols and structural subtyping?",       # -> pdf (page-cited)
    "What is the capital of France?",  # out of scope -> should be empty
]


def main():
    r = get_retriever()
    print(f"Knowledge base: {len(r.docs)} chunks across {len(r.list_sources())} docs\n")
    for q in QUERIES:
        print(f"Q: {q}")
        hits = r.search(q, k=3)
        if not hits:
            print("   (no chunk passed the relevance threshold)\n")
            continue
        for c in hits:
            preview = c.text.strip().replace("\n", " ")[:90]
            print(f"   {c.locator:<40} score={c.score:6.2f}  {preview!r}")
        print()


if __name__ == "__main__":
    main()
