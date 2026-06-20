#!/usr/bin/env python
"""Calibrate the retrieval relevance threshold on a small validation set.

For each labelled QA case, take the top cross-encoder rerank score (threshold
disabled). "Should answer" = in_scope; "should abstain" = out_of_scope / no_answer.
Sweep candidate thresholds and report precision / recall / abstention, so
``SCORE_THRESHOLD`` is chosen from data instead of by feel.

    python scripts/calibrate_threshold.py
    python scripts/calibrate_threshold.py --split offline_sample
"""

import argparse

from citelocal_agent.eval.qa_dataset import load_qa_cases
from citelocal_agent.retriever import get_retriever


def main():
    parser = argparse.ArgumentParser(description="Calibrate SCORE_THRESHOLD on QA cases.")
    parser.add_argument("--split", default="full_corpus",
                        choices=["full_corpus", "offline_sample"])
    args = parser.parse_args()

    r = get_retriever()
    scored = []
    for c in load_qa_cases(split=args.split):
        hits = r.search(c["question"], k=1, score_threshold=float("-inf"))
        top = hits[0].score if hits else float("-inf")
        scored.append((c["question"][:46], top, c["intent"] == "in_scope"))

    print(f"{'top_score':>10}  {'answerable':>10}  question")
    for q, top, sa in sorted(scored, key=lambda x: x[1], reverse=True):
        print(f"{top:>10.2f}  {str(sa):>10}  {q}")

    # Sweep a range derived from the observed scores, so this works for any
    # reranker scale (ms-marco's wide logits OR bge/mxbai's ~[0,1] sigmoid).
    tops = [top for _, top, _ in scored if top != float("-inf")]
    lo, hi = (min(tops), max(tops)) if tops else (0.0, 1.0)
    span = (hi - lo) or 1.0
    steps = 40
    candidates = [lo + span * i / steps for i in range(steps + 1)]

    print(f"\n{'thresh':>9}{'precision':>11}{'recall':>9}{'abstain':>9}{'f1':>7}")
    best = None
    for th in candidates:
        tp = fp = fn = tn = 0
        for _, top, sa in scored:
            pred = top >= th
            tp += sa and pred
            fn += sa and not pred
            fp += (not sa) and pred
            tn += (not sa) and not pred
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        abst = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{th:>9.3f}{prec:>11.2f}{rec:>9.2f}{abst:>9.2f}{f1:>7.2f}")
        objective = f1 + abst  # balance answering correctly and abstaining correctly
        if best is None or objective > best[1]:
            best = (th, objective, prec, rec, abst)

    print(
        f"\nSuggested SCORE_THRESHOLD ≈ {best[0]:.3f} "
        f"(precision {best[2]:.2f}, recall {best[3]:.2f}, abstain {best[4]:.2f}). "
        "Set it in .env / configuration.py."
    )


if __name__ == "__main__":
    main()
