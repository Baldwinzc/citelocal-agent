"""State and structured-output schemas for the document QA agent."""

import operator
from typing import Annotated

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict


class IntentSchema(BaseModel):
    """Decide whether a question can be answered from the knowledge base, and how."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["in_scope", "out_of_scope"] = Field(
        description="'in_scope' if the question is about the knowledge base contents "
        "and should be answered by retrieving documents; 'out_of_scope' for "
        "chit-chat or questions clearly unrelated to the documents.",
    )
    complexity: Literal["simple", "complex"] = Field(
        default="simple",
        description="'simple' if the question is a single fact answerable from one "
        "place; 'complex' if it spans multiple documents/topics, asks to compare or "
        "combine several facts, or needs to be decomposed into sub-questions. Only "
        "meaningful when classification is 'in_scope'.",
    )


class PlanSchema(BaseModel):
    """Decompose a complex question into focused, independent sub-questions."""

    reasoning: str = Field(
        description="Why these sub-questions together cover the original question."
    )
    sub_questions: list[str] = Field(
        description="1-4 self-contained sub-questions, each answerable on its own "
        "by searching the documents."
    )


class StateInput(TypedDict):
    """The external input to the graph."""

    question_input: dict  # e.g. {"question": "How do I declare a path param?"}


class State(MessagesState):
    """Graph state. ``messages`` is inherited from MessagesState.

    - ``trace`` accumulates observability events (one per retrieval / decision).
    - ``retrieved_locators`` accumulates every locator the agent actually
      retrieved this run, so the final answer's citations can be *verified*
      against what was really seen (not just trusted because the LLM emitted them).
    - ``evidence`` accumulates ``{locator, text}`` for each retrieved chunk, so a
      claim can be checked for entailment against the *text* that supports it
      (used by the multi-agent Verifier and per-sentence citation verification).
    - ``sub_results`` accumulates one entry per parallel Researcher in the
      multi-agent path: ``{sub_id, sub_question, answer, citations, evidence}``.
      Researchers write here (not to ``messages``) so N parallel branches merge
      cleanly via ``operator.add`` instead of racing the ``add_messages`` reducer.
    - ``sub_questions`` / ``verified_results`` are single-writer channels (the
      planner / verifier nodes), so they need no reducer.

    The ``operator.add`` channels are append-merged so the whole run is inspectable.
    """

    question_input: dict
    classification_decision: Literal["in_scope", "out_of_scope"]
    trace: Annotated[list, operator.add]
    retrieved_locators: Annotated[list, operator.add]
    evidence: Annotated[list, operator.add]
    # --- multi-agent (orchestrator) channels ---
    sub_questions: list
    sub_results: Annotated[list, operator.add]
    verified_results: list
