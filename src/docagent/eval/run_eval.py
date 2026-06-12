#!/usr/bin/env python
"""Quantitative evaluation of docagent over docagent.eval.qa_dataset.

Run from the project root, after ingesting the corpus that matches the split:
    python -m docagent.ingest --path ./papers --reset
    python -m docagent.eval.run_eval                       # full_corpus (default)
    python -m docagent.eval.run_eval --split offline_sample --categories multi_hop

Metrics (reported overall AND broken down by category — the per-category view is
what lets us prove later milestones, e.g. multi-hop, actually improve):
    intent    - router decision matches expected (in_scope vs out_of_scope)
    recall    - expected source doc(s) present in retriever top-k
    answer    - LLM judges the final answer meets the criterion (in_scope cases)
    citation  - verified citations point to an expected source (in_scope sourced)
    refusal   - out_of_scope / no_answer cases correctly declined
Also reported: hallucinated citations (emitted but never retrieved) — should be 0.

A machine-readable summary is written to ``eval_results.json`` (gitignored) so
deltas between milestones can be tracked.
"""

import argparse
import json
from collections import defaultdict
from typing import cast

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from docagent.agent import get_default_agent
from docagent.configuration import DEFAULT_LLM_MODEL, DEFAULT_TOP_K, llm_call_kwargs
from docagent.eval.qa_dataset import CATEGORIES, load_qa_cases
from docagent.logging_config import configure_logging
from docagent.retriever import get_retriever
from docagent.utils import extract_outcome, source_of

load_dotenv()


class Grade(BaseModel):
    correct: bool = Field(description="Does the answer satisfy the criterion?")
    justification: str = Field(description="Brief justification.")


_judge = init_chat_model(DEFAULT_LLM_MODEL, **llm_call_kwargs()).with_structured_output(Grade)

JUDGE_SYS = (
    "You grade a document-QA assistant's answer against a single criterion. "
    "Return correct=true only if the answer satisfies it. When the criterion "
    "requires declining, correct=true means the answer appropriately declines or "
    "says the documents don't cover it (and does not fabricate)."
)


def _judge_answer(criteria: str, question: str, answer: str) -> bool:
    g = cast(
        Grade,
        _judge.invoke(
            [
                {"role": "system", "content": JUDGE_SYS},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nCriterion: {criteria}\n\nAnswer:\n{answer}",
                },
            ]
        ),
    )
    return bool(g.correct)


def _mark(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "ok" if v else "X"
    return f"{v:.2f}"


def _blank_stats():
    return {
        "n": 0, "intent_c": 0, "intent_t": 0, "recalls": [],
        "ans_c": 0, "ans_t": 0, "cite_c": 0, "cite_t": 0,
        "ref_c": 0, "ref_t": 0, "hallucinated": 0, "errors": 0,
    }


def _summarise(s: dict) -> dict:
    def pct(c, t):
        return None if not t else round(c / t, 4)

    mean_recall = sum(s["recalls"]) / len(s["recalls"]) if s["recalls"] else None
    return {
        "cases": s["n"],
        "intent_accuracy": pct(s["intent_c"], s["intent_t"]),
        "recall_mean": round(mean_recall, 4) if mean_recall is not None else None,
        "answer_correctness": pct(s["ans_c"], s["ans_t"]),
        "citation_grounding": pct(s["cite_c"], s["cite_t"]),
        "refusal_accuracy": pct(s["ref_c"], s["ref_t"]),
        "hallucinated_citations": s["hallucinated"],
    }


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description="Evaluate docagent over the QA dataset.")
    parser.add_argument("--split", default="full_corpus",
                        choices=["full_corpus", "offline_sample"])
    parser.add_argument("--categories", nargs="*", default=None, choices=sorted(CATEGORIES))
    parser.add_argument("--json-out", default="eval_results.json")
    args = parser.parse_args()

    cases = load_qa_cases(split=args.split)
    if args.categories:
        cases = [c for c in cases if c["category"] in args.categories]
    if not cases:
        raise SystemExit(f"No cases for split={args.split} categories={args.categories}.")

    retriever = get_retriever()
    agent = get_default_agent()
    print(f"Corpus: {retriever.num_chunks} chunks / {len(retriever.list_sources())} docs")
    print(f"Eval split={args.split}  cases={len(cases)}\n")

    overall = _blank_stats()
    per_cat: dict[str, dict] = defaultdict(_blank_stats)

    header = f"{'case':<26}{'cat':<14}{'intent':<8}{'recall':<8}{'answer':<8}{'citation':<10}{'refusal':<8}"
    print(header)
    print("-" * len(header))

    for case in cases:
        q = case["question"]
        exp_sources = set(case["expected_sources"])
        exp_intent = case["intent"]
        cat = case["category"]
        exp_router = "out_of_scope" if exp_intent == "out_of_scope" else "in_scope"
        buckets = (overall, per_cat[cat])

        hits = retriever.search(q, k=DEFAULT_TOP_K)
        retrieved = {h.source.rsplit("/", 1)[-1] for h in hits}  # compare by basename
        recall = None
        if exp_sources:
            recall = len(exp_sources & retrieved) / len(exp_sources)
            for b in buckets:
                b["recalls"].append(recall)

        try:
            result = agent.invoke({"question_input": {"question": q}})
            o = extract_outcome(result)
            answer_ok = _judge_answer(case["criteria"], q, o["answer"])
        except Exception as e:  # noqa: BLE001 — a flaky provider call shouldn't abort the run
            for b in buckets:
                b["n"] += 1
                b["errors"] += 1
            print(f"{case['id']:<26}{cat:<14}ERROR: {str(e)[:50]}")
            continue
        intent = o["intent"]
        cited = {source_of(c).rsplit("/", 1)[-1] for c in o["citations"]}

        intent_ok = intent == exp_router

        cite_ok = None
        if exp_intent == "in_scope" and exp_sources:
            cite_ok = bool(cited & exp_sources) and not o["unsupported"]

        refusal_ok = None
        if exp_intent in ("out_of_scope", "no_answer"):
            refusal_ok = answer_ok

        for b in buckets:
            b["n"] += 1
            b["intent_c"] += int(intent_ok)
            b["intent_t"] += 1
            b["hallucinated"] += len(o["unsupported"])
            if cite_ok is not None:
                b["cite_c"] += int(cite_ok)
                b["cite_t"] += 1
            if refusal_ok is not None:
                b["ref_c"] += int(refusal_ok)
                b["ref_t"] += 1
            else:
                b["ans_c"] += int(answer_ok)
                b["ans_t"] += 1

        ans_cell = answer_ok if exp_intent == "in_scope" else None
        print(
            f"{case['id']:<26}{cat:<14}{_mark(intent_ok):<8}{_mark(recall):<8}"
            f"{_mark(ans_cell):<8}{_mark(cite_ok):<10}{_mark(refusal_ok):<8}"
        )

    print("-" * len(header))

    def pct(c, t):
        return f"{c}/{t} ({100 * c / t:.0f}%)" if t else "n/a"

    mean_recall = sum(overall["recalls"]) / len(overall["recalls"]) if overall["recalls"] else 0.0
    print("\n=== OVERALL ===")
    print(f"Intent routing accuracy : {pct(overall['intent_c'], overall['intent_t'])}")
    print(f"Retrieval recall (mean) : {mean_recall:.2f}")
    print(f"Answer correctness      : {pct(overall['ans_c'], overall['ans_t'])}")
    print(f"Citation grounding      : {pct(overall['cite_c'], overall['cite_t'])}")
    print(f"Refusal accuracy        : {pct(overall['ref_c'], overall['ref_t'])}")
    print(f"Hallucinated citations  : {overall['hallucinated']} (lower is better)")
    if overall["errors"]:
        print(f"Errored cases (skipped) : {overall['errors']}/{overall['n']}")

    print("\n=== BY CATEGORY ===")
    cat_header = f"{'category':<16}{'n':>4}{'intent':>9}{'recall':>9}{'answer':>9}{'citation':>10}{'refusal':>9}"
    print(cat_header)
    print("-" * len(cat_header))
    for cat in sorted(per_cat):
        s = _summarise(per_cat[cat])
        print(
            f"{cat:<16}{s['cases']:>4}"
            f"{_mark(s['intent_accuracy']):>9}{_mark(s['recall_mean']):>9}"
            f"{_mark(s['answer_correctness']):>9}{_mark(s['citation_grounding']):>10}"
            f"{_mark(s['refusal_accuracy']):>9}"
        )

    payload = {
        "split": args.split,
        "categories": args.categories,
        "overall": _summarise(overall),
        "by_category": {cat: _summarise(per_cat[cat]) for cat in sorted(per_cat)},
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote machine-readable results to {args.json_out}")


if __name__ == "__main__":
    main()
