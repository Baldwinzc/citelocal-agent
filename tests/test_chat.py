#!/usr/bin/env python
"""Multi-turn conversation tests — require an LLM API key.

Drives the checkpointed chat agent across two turns on one thread, where the
second turn ("how does IT differ ...") only resolves if the conversation memory
carried the first turn's topic. Two different threads must NOT share state.

Run after ingesting sample_notes into the default collection:
    python -m docagent.ingest --path ./sample_notes --reset
    pytest tests/test_chat.py -v
"""

import uuid

import pytest
from dotenv import load_dotenv

from docagent.agent import get_chat_agent
from docagent.utils import extract_outcome

load_dotenv(override=True)


@pytest.fixture(scope="module")
def agent():
    return get_chat_agent()


def _thread():
    return {"configurable": {"thread_id": uuid.uuid4().hex}}


def test_followup_resolves_against_prior_turn(agent):
    cfg = _thread()
    r1 = agent.invoke({"question_input": {"question": "What is BM25?"}}, config=cfg)
    o1 = extract_outcome(r1)
    assert o1["kind"] == "answer" and o1["answer"]

    # "it" has no referent without the first turn's memory
    r2 = agent.invoke(
        {"question_input": {"question": "How does it differ from dense retrieval?"}},
        config=cfg,
    )
    o2 = extract_outcome(r2)
    assert o2["kind"] == "answer"
    text = o2["answer"].lower()
    assert "bm25" in text or "sparse" in text, f"follow-up lost context: {o2['answer']}"
    # state persisted: turn 2 sees more conversation history than turn 1 did
    assert len(r2["messages"]) > len(r1["messages"])


def test_threads_are_isolated(agent):
    ta, tb = _thread(), _thread()
    agent.invoke({"question_input": {"question": "What is RAG?"}}, config=ta)
    a2 = agent.invoke({"question_input": {"question": "What is BM25?"}}, config=ta)
    b1 = agent.invoke({"question_input": {"question": "What is attention?"}}, config=tb)
    # thread A holds two turns; thread B holds one -> B didn't inherit A's history
    assert len(a2["messages"]) > len(b1["messages"])
