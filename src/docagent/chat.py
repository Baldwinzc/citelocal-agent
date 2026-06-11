#!/usr/bin/env python
"""Interactive multi-turn chat over the local knowledge base.

Keeps one conversation thread so follow-up questions ("tell me more", "why is
that?") resolve against earlier turns. Each turn is still routed (scope +
simple/complex) and answered with verified citations.

Usage:
    python -m docagent.chat
    python -m docagent.chat --verify nli     # also flag ungrounded sentences

Type 'exit' / 'quit' (or Ctrl-D) to leave; 'reset' starts a fresh thread.
"""

import argparse
import uuid

from dotenv import load_dotenv

from docagent.agent import get_chat_agent
from docagent.ask import _print_outcome
from docagent.configuration import Configuration
from docagent.logging_config import configure_logging
from docagent.utils import extract_outcome


def main():
    load_dotenv()
    configure_logging()
    cfg = Configuration.from_runnable_config()
    parser = argparse.ArgumentParser(description="Chat with the local knowledge base.")
    parser.add_argument(
        "--verify",
        choices=["off", "nli", "llm"],
        default=cfg.entailment_backend,
        help="Per-sentence entailment check of each answer (default from "
        f"ENTAILMENT_BACKEND={cfg.entailment_backend}).",
    )
    args = parser.parse_args()

    verify_llm = None
    if args.verify == "llm":
        from langchain.chat_models import init_chat_model

        verify_llm = init_chat_model(cfg.llm_model, temperature=0.0)

    agent = get_chat_agent()
    thread_id = uuid.uuid4().hex
    print("docagent chat — ask a question ('exit' to quit, 'reset' for a new thread)\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "reset":
            thread_id = uuid.uuid4().hex
            print("(started a new conversation)\n")
            continue

        result = agent.invoke(
            {"question_input": {"question": q}},
            config={"configurable": {"thread_id": thread_id}},
        )
        _print_outcome(
            extract_outcome(
                result,
                verify_backend=args.verify,
                llm=verify_llm,
                nli_model=cfg.nli_model,
            )
        )
        print()


if __name__ == "__main__":
    main()
