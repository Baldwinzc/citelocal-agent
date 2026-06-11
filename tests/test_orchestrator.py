#!/usr/bin/env python
"""End-to-end multi-agent (orchestrator) tests — require an LLM API key.

Runs the multi_hop offline_sample cases through the full graph and checks the
orchestrator actually did its job: decomposed into >=2 sub-questions, retrieved
across >=2 distinct sources, and produced a grounded answer with no hallucinated
citations.

Run from the project root after ingesting sample_notes into the default collection:
    python -m docagent.ingest --path ./sample_notes --reset
    pytest tests/test_orchestrator.py -v
"""

import pytest
from dotenv import load_dotenv

from docagent.agent import get_default_agent
from docagent.eval.qa_dataset import load_qa_cases
from docagent.utils import extract_outcome, source_of

load_dotenv(override=True)


def _multi_hop_cases():
    return [
        (c["question"], c["id"])
        for c in load_qa_cases(split="offline_sample")
        if c["category"] == "multi_hop"
    ]


@pytest.fixture(scope="module")
def agent():
    return get_default_agent()


@pytest.mark.parametrize("question,name", _multi_hop_cases())
def test_multi_hop_orchestration(agent, question, name):
    result = agent.invoke(
        {"question_input": {"question": question}}, config={"recursion_limit": 30}
    )

    # 1. router sent it down the complex path and the planner decomposed it
    planner_steps = [t for t in result.get("trace", []) if t.get("step") == "planner"]
    assert planner_steps, f"[{name}] orchestrator/planner did not run"
    assert len(planner_steps[0]["sub_questions"]) >= 2, (
        f"[{name}] expected >=2 sub-questions, got {planner_steps[0]['sub_questions']}"
    )

    # 2. researchers retrieved across at least two distinct sources
    sources = {source_of(loc) for loc in result.get("retrieved_locators", [])}
    assert len(sources) >= 2, f"[{name}] retrieval spanned <2 sources: {sources}"

    # 3. final synthesized answer is grounded, with no hallucinated citations
    o = extract_outcome(result)
    assert o["citations"], f"[{name}] synthesized answer has no citations"
    assert not o["unsupported"], f"[{name}] hallucinated citations: {o['unsupported']}"
