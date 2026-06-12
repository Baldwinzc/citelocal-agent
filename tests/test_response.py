#!/usr/bin/env python
"""End-to-end agent tests (require an LLM API key; no LangSmith needed).

For each in-scope question, one agent run is checked for:
1. expected tool calls (search_docs -> Answer),
2. grounded citations (non-empty AND no hallucinated/unsupported locators),
3. answer quality (LLM-judged against the criterion).

Run from the project root after ingesting the corpus.
"""

import pytest
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from docagent.agent import build_agent
from docagent.configuration import DEFAULT_LLM_MODEL
from docagent.eval.prompts import RESPONSE_CRITERIA_SYSTEM_PROMPT
from docagent.eval.qa_dataset import load_qa_cases
from docagent.utils import extract_outcome, extract_tool_calls, format_messages_string

load_dotenv(override=True)


class CriteriaGrade(BaseModel):
    grade: bool = Field(description="Does the answer satisfy the criterion?")
    justification: str = Field(description="Justification, with specific examples.")


_judge = init_chat_model(DEFAULT_LLM_MODEL).with_structured_output(CriteriaGrade)


def _in_scope_cases():
    # Anchored to the bundled sample_notes so this LLM suite can run without
    # downloading the papers (ingest sample_notes into the default collection).
    return [
        (c["question"], c["id"], c["criteria"])
        for c in load_qa_cases(split="offline_sample")
        if c["intent"] == "in_scope"
    ]


@pytest.fixture
def agent():
    # Fresh agent per test (own LLM http client) — the cached singleton can be
    # closed by pytest teardown and leak across test modules.
    return build_agent()


@pytest.mark.parametrize("question,name,criteria", _in_scope_cases())
def test_in_scope_case(agent, question, name, criteria):
    result = agent.invoke({"question_input": {"question": question}})

    # 1. expected tool calls
    tools = extract_tool_calls(result["messages"])
    assert "search_docs" in tools, f"[{name}] no search_docs; got {tools}"
    assert "answer" in tools, f"[{name}] never called Answer; got {tools}"

    # 2. grounded citations
    o = extract_outcome(result)
    assert o["citations"], f"[{name}] answer has no citations"
    assert not o["unsupported"], f"[{name}] hallucinated citations: {o['unsupported']}"

    # 3. answer quality (LLM-judged)
    transcript = o["answer"] + "\n\n" + format_messages_string(result["messages"])
    grade = _judge.invoke(
        [
            {"role": "system", "content": RESPONSE_CRITERIA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Response criteria: {criteria}\n\nAssistant's response:\n{transcript}\n\nDoes it meet the criteria?",
            },
        ]
    )
    assert grade.grade, f"[{name}] {grade.justification}"
