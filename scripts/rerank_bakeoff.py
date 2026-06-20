#!/usr/bin/env python
"""Reranker model-selection bake-off (deterministic, no LLM).

Why: multi-hop is the weakest retrieval cell. Single-shot recall on the compound
question plateaus because the cross-encoder scores the second source low. This
script compares candidate rerankers *fairly* — the dense+BM25+RRF candidate pool
is identical across models (it doesn't depend on the reranker), so only the
reranking order changes — and reports the numbers that decide model selection:

    recall@k  - is each gold source ranked into the top-k? (threshold DISABLED, so
                this is pure ranking quality, comparable across models / score scales)
    AUC       - how well the top-1 score separates answerable (in_scope) from
                should-refuse (out_of_scope / no_answer) questions; this is the
                abstention headroom, scale-independent (1.0 = perfectly separable)
    latency   - mean wall-clock per search() on this machine (CPU cost is real)

Run after ingesting the matching corpus (e.g. sample_notes for offline_sample):
    python scripts/rerank_bakeoff.py --split offline_sample
    python scripts/rerank_bakeoff.py --split offline_sample --models BAAI/bge-reranker-base

The default RERANKER_MODEL is NOT changed by this script — it only measures.
"""

import argparse
import time

from sentence_transformers import CrossEncoder

from citelocal_agent.eval.qa_dataset import load_qa_cases
from citelocal_agent.retriever import get_retriever

DEFAULT_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",   # current default (~22M)
    "cross-encoder/ms-marco-MiniLM-L-12-v2",  # cheap upgrade (~33M)
    "BAAI/bge-reranker-base",                 # ~278M
    "BAAI/bge-reranker-v2-m3",                # ~568M
    "mixedbread-ai/mxbai-rerank-base-v1",     # ~184M
]
NEG_INF = float("-inf")


def _auc(pos: list[float], neg: list[float]) -> float | None:
    """P(score(in_scope) > score(refuse)) over all pairs (Mann-Whitney); 0.5 ties."""
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser(description="Compare candidate rerankers on the QA set.")
    ap.add_argument("--split", default="offline_sample",
                    choices=["full_corpus", "offline_sample"])
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--k", type=int, default=4, help="primary recall@k cutoff")
    ap.add_argument("--k2", type=int, default=10, help="secondary recall@k cutoff")
    args = ap.parse_args()

    cases = load_qa_cases(split=args.split)
    sourced = [c for c in cases if c["expected_sources"]]
    retriever = get_retriever()
    print(f"Corpus: {retriever.num_chunks} chunks / {len(retriever.list_sources())} docs")
    print(f"Split={args.split}  cases={len(cases)}  sourced={len(sourced)}\n")

    cats = sorted({c["category"] for c in sourced})
    hdr = (f"{'model':<40}{'recall@'+str(args.k):>10}{'recall@'+str(args.k2):>11}"
           f"{'multi_hop':>11}{'abst.AUC':>10}{'ms/q':>8}")
    print(hdr)
    print("-" * len(hdr))

    for name in args.models:
        try:
            retriever._reranker_obj = CrossEncoder(name)  # swap reranker, fixed pools
        except Exception as e:  # noqa: BLE001
            print(f"{name:<40}  LOAD FAILED: {type(e).__name__}: {str(e)[:40]}")
            continue

        rec_k = {c: [] for c in cats}
        rec_k2_all, rec_k_all = [], []
        pos, neg = [], []
        t0 = time.perf_counter()
        for c in cases:
            # thresholds disabled -> top-k purely by rerank score
            hits = retriever.search(c["question"], k=max(args.k, args.k2),
                                    score_threshold=NEG_INF, support_threshold=NEG_INF)
            if hits:
                (pos if c["intent"] == "in_scope" else neg).append(hits[0].score)
            if c["expected_sources"]:
                exp = set(c["expected_sources"])
                bn = [h.source.rsplit("/", 1)[-1] for h in hits]
                r1 = len(exp & set(bn[:args.k])) / len(exp)
                r2 = len(exp & set(bn[:args.k2])) / len(exp)
                rec_k_all.append(r1)
                rec_k2_all.append(r2)
                rec_k[c["category"]].append(r1)
        ms = (time.perf_counter() - t0) / max(len(cases), 1) * 1000

        def avg(xs):
            return sum(xs) / len(xs) if xs else 0.0

        auc = _auc(pos, neg)
        mh = rec_k.get("multi_hop", [])
        print(f"{name:<40}{avg(rec_k_all):>10.3f}{avg(rec_k2_all):>11.3f}"
              f"{avg(mh):>11.3f}{(auc if auc is not None else float('nan')):>10.3f}{ms:>8.0f}")

    print("\nrecall@k uses NO relevance threshold (pure ranking). Swapping the default")
    print("RERANKER_MODEL also requires re-running scripts/calibrate_threshold.py — the")
    print("score scale changes (ms-marco = wide logits, bge/mxbai ~ [0,1]).")


if __name__ == "__main__":
    main()
