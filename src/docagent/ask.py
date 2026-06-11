#!/usr/bin/env python
"""CLI: ask the local knowledge base a question.

Usage:
    python -m docagent.ask "How do I declare an integer path parameter?"
    python -m docagent.ask --trace "..."     # also print the retrieval trace
"""

import argparse

from dotenv import load_dotenv

from docagent.agent import get_default_agent
from docagent.configuration import Configuration
from docagent.utils import extract_outcome


def _print_trace(result: dict) -> None:
    trace = result.get("trace") or []
    if not trace:
        return
    print("=== trace ===")
    for i, t in enumerate(trace, 1):
        if t.get("step") == "search_docs":
            print(f"  {i}. search_docs  query={t.get('query')!r}")
        else:
            print(f"  {i}. {t.get('step')}")
    print()


def _print_outcome(o: dict) -> None:
    if o["kind"] == "question":
        print("=== Clarifying question ===")
        print(o["question"].strip())
        return
    print("=== Answer ===")
    print(o["answer"].strip())
    if o["citations"]:
        print("\n=== Citations ===")
        for c in o["citations"]:
            print(f"- {c}")
    if o["unsupported"]:
        print("\n=== Unsupported citations (dropped — not actually retrieved) ===")
        for c in o["unsupported"]:
            print(f"- {c}")
    if o.get("unsupported_sentences"):
        print("\n=== Unverified sentences (not entailed by the evidence) ===")
        for s in o["unsupported_sentences"]:
            print(f"- {s}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Ask the local document knowledge base a question."
    )
    cfg = Configuration.from_runnable_config()
    parser.add_argument("question", help="The question to ask.")
    parser.add_argument(
        "--trace", action="store_true", help="Print the agent's retrieval trace."
    )
    parser.add_argument(
        "--verify",
        choices=["off", "nli", "llm"],
        default=cfg.entailment_backend,
        help="Per-sentence entailment check of the answer against the evidence "
        "(nli = local cross-encoder, offline; llm = one grading call). "
        f"Default: {cfg.entailment_backend} (ENTAILMENT_BACKEND).",
    )
    args = parser.parse_args()

    result = get_default_agent().invoke(
        {"question_input": {"question": args.question}},
        config={"recursion_limit": 12},
    )

    verify_llm = None
    if args.verify == "llm":
        from langchain.chat_models import init_chat_model

        verify_llm = init_chat_model(cfg.llm_model, temperature=0.0)

    if args.trace:
        _print_trace(result)
    _print_outcome(
        extract_outcome(
            result,
            verify_backend=args.verify,
            llm=verify_llm,
            nli_model=cfg.nli_model,
        )
    )


if __name__ == "__main__":
    main()
