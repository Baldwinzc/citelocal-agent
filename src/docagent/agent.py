r"""Document knowledge-base agent (agentic RAG over a local Chroma store).

Two-layer LangGraph, with the response layer now routed by question complexity::

    START -> intent_router --in_scope + simple --> response_agent  --> END
                            \--in_scope + complex--> orchestrator   --> END
                            \--out_of_scope / empty KB-------------> END

``response_agent`` is the single ReAct retrieval loop (``build_research_loop``);
``orchestrator`` (see ``orchestrator.py``) is the multi-agent path that decomposes
a complex question, researches sub-questions in parallel **reusing the same loop**,
verifies, and synthesises. The router picks between them so simple questions keep
their low latency/cost.

The agent is constructed by ``build_agent(config)`` — **nothing is initialised at
import time** (no LLM, no reranker), so importing this module is cheap and the
model/retriever lifecycles are explicit. ``make_graph`` is the factory used by
``langgraph.json``; ``get_default_agent`` is a lazily-built, cached default for
the CLI and web server.

Every retrieval is recorded in ``state["trace"]``; the locators it returned are
accumulated in ``state["retrieved_locators"]`` (so citations can be *verified*
against what was actually retrieved) and the chunk texts in ``state["evidence"]``
(so claims can be checked for entailment against the supporting text).
"""

import logging
import re
from functools import lru_cache
from typing import Literal, cast

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from docagent.configuration import Configuration, llm_call_kwargs
from docagent.orchestrator import build_orchestrator
from docagent.prompts import (
    AGENT_TOOLS_PROMPT,
    agent_system_prompt,
    default_intent_instructions,
    default_kb_description,
    intent_system_prompt,
    intent_user_prompt,
)
from docagent.retriever import get_retriever
from docagent.schemas import IntentSchema, State, StateInput
from docagent.tools import get_tools_by_name, make_retrieval_tools
from docagent.utils import extract_message_content

logger = logging.getLogger(__name__)

TERMINAL_TOOLS = {"Answer", "Question"}
# Parse each search_docs result block: "[i] locator: <loc>  (relevance <s>)\n<text>".
# Captures both the locator (verified against citations) and the chunk text
# (kept as evidence for entailment checks). DOTALL + a lookahead to the next
# block header lets a chunk's text contain blank lines safely.
_BLOCK_RE = re.compile(
    r"\[\d+\] locator: (?P<loc>.+?)\s{2,}\(relevance [^)]*\)\n"
    r"(?P<text>.*?)(?=\n\n\[\d+\] locator: |\Z)",
    re.DOTALL,
)


def _parse_search_results(observation: str) -> list[tuple[str, str]]:
    """Extract ``(locator, chunk_text)`` pairs from a search_docs observation."""
    return [
        (m.group("loc").strip(), m.group("text").strip())
        for m in _BLOCK_RE.finditer(observation)
    ]


def build_research_loop(llm_with_tools, tools_by_name, system_prompt):
    """Compile the ReAct retrieval loop (llm_call -> tools -> llm_call).

    Returned graph is used **both** as the simple-path ``response_agent`` node and,
    invoked per sub-question, as each Researcher's engine in the orchestrator.
    """

    def llm_call(state: State):
        """LLM decides which retrieval tool to call next."""
        return {
            "messages": [
                llm_with_tools.invoke(
                    [{"role": "system", "content": system_prompt}] + state["messages"]
                )
            ]
        }

    def tool_node(state: State):
        """Run non-terminal tools; record trace + retrieved locators + evidence."""
        result, trace, locators, evidence = [], [], [], []
        for tool_call in getattr(state["messages"][-1], "tool_calls", None) or []:
            name = tool_call["name"]
            try:
                observation = tools_by_name[name].invoke(tool_call["args"])
            except Exception as e:  # noqa: BLE001 — never crash the graph
                observation = f"Tool '{name}' failed: {e}"
            result.append(
                {"role": "tool", "content": observation, "tool_call_id": tool_call["id"]}
            )
            if name == "search_docs":
                trace.append(
                    {"step": "search_docs", "query": tool_call["args"].get("query", "")}
                )
                for loc, text in _parse_search_results(observation):
                    locators.append(loc)
                    evidence.append({"locator": loc, "text": text})
            elif name == "list_sources":
                trace.append({"step": "list_sources"})
        return {
            "messages": result,
            "trace": trace,
            "retrieved_locators": locators,
            "evidence": evidence,
        }

    def should_continue(state: State) -> str:
        tool_calls = getattr(state["messages"][-1], "tool_calls", None)
        if tool_calls:
            if any(tc["name"] in TERMINAL_TOOLS for tc in tool_calls):
                return END
            return "environment"
        return END

    builder = StateGraph(State)
    builder.add_node("llm_call", llm_call)
    builder.add_node("environment", tool_node)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call", should_continue, {"environment": "environment", END: END}
    )
    builder.add_edge("environment", "llm_call")
    return builder.compile()


_SIMPLE_PREFIX = "Answer this question using the knowledge base: "


def _recent_dialogue(messages: list, max_msgs: int = 4) -> list[dict]:
    """Recent user/assistant turns as ``{role, content}`` dicts, for the router.

    Lets the intent router classify a follow-up ("tell me more", "why?") in the
    context of the conversation. Tool messages and tool-call-only (empty-content)
    assistant messages are skipped; the simple-path instruction prefix is stripped.
    """
    dialogue = []
    for m in messages:
        if isinstance(m, dict):
            role, content = m.get("role"), m.get("content", "") or ""
        else:
            content = extract_message_content(m)
            cls = type(m).__name__.lower()
            role = "user" if "human" in cls else "assistant" if "ai" in cls else None
        if role in ("user", "assistant") and content and content.strip():
            dialogue.append(
                {"role": role, "content": content.replace(_SIMPLE_PREFIX, "").strip()}
            )
    return dialogue[-max_msgs:]


def _merge_subgraph_result(out: dict, prior_msg_count: int) -> dict:
    """Merge a subgraph's final state back into the parent.

    Returns only the messages the subgraph *added* (so the user message the
    router already placed in parent state isn't re-appended), plus the retrieval
    side-channels the subgraph accumulated.
    """
    update = {
        k: out[k] for k in ("trace", "retrieved_locators", "evidence") if k in out
    }
    update["messages"] = (out.get("messages", []) or [])[prior_msg_count:]
    return update


def build_agent(config: Configuration | None = None, checkpointer=None):
    """Build and compile the agent graph for a given configuration.

    All model/tool wiring lives here (not at module import), captured in node
    closures, so the same module can serve multiple configs and import stays cheap.
    Pass a ``checkpointer`` (e.g. ``InMemorySaver``) to persist state per
    ``thread_id`` for multi-turn conversations (see ``get_chat_agent``).
    """
    config = config or Configuration.from_runnable_config()

    retriever = get_retriever(config.chroma_path, config.collection_name)
    tools = make_retrieval_tools(retriever, config.top_k, config.score_threshold)
    tools_by_name = get_tools_by_name(tools)
    llm = init_chat_model(config.llm_model, temperature=0.0, **llm_call_kwargs())
    llm_router = llm.with_structured_output(IntentSchema)
    llm_with_tools = llm.bind_tools(tools, tool_choice="any")

    system_prompt = agent_system_prompt.format(
        tools_prompt=AGENT_TOOLS_PROMPT, kb_description=default_kb_description
    )

    research_loop = build_research_loop(llm_with_tools, tools_by_name, system_prompt)
    orchestrator = build_orchestrator(
        llm,
        research_loop,
        verify_backend=config.entailment_backend,
        research_recursion_limit=config.recursion_limit,
    )

    # Each path invokes its inner graph with its OWN recursion budget (instead of
    # adding the compiled graph as a node, which would force both to inherit the
    # top-level limit). The top-level graph is then just router -> one node.
    def response_agent(state: State):
        msgs = state["messages"]
        out = research_loop.invoke(
            {"messages": msgs}, config={"recursion_limit": config.recursion_limit}
        )
        return _merge_subgraph_result(out, len(msgs))

    def orchestrator_node(state: State):
        try:
            out = orchestrator.invoke(
                state, config={"recursion_limit": config.orchestrator_recursion_limit}
            )
            return _merge_subgraph_result(out, len(state.get("messages", []) or []))
        except Exception as e:  # noqa: BLE001
            # The complex path can fail in many ways (a model emitting malformed
            # structured output, a researcher exhausting its step budget). Degrade
            # to the simple retrieval loop rather than crashing the request.
            logger.warning("orchestrator failed (%s); falling back to the simple path", e)
            question = state["question_input"].get("question", "")
            out = research_loop.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"recursion_limit": config.recursion_limit},
            )
            return _merge_subgraph_result(out, 1)

    def intent_router(
        state: State,
    ) -> Command[Literal["response_agent", "orchestrator", "__end__"]]:
        """Guard an empty KB, then decide scope and (for in_scope) the path."""
        question = state["question_input"].get("question", "")

        if retriever.is_empty:
            return Command(
                goto=END,  # type: ignore[arg-type]
                update={
                    "classification_decision": "out_of_scope",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "The knowledge base is empty. Run "
                            "`python -m docagent.ingest --path <your-docs>` first.",
                        }
                    ],
                },
            )

        # Recent turns give the router context so follow-ups classify correctly.
        router_messages = [
            {
                "role": "system",
                "content": intent_system_prompt.format(
                    kb_description=default_kb_description,
                    intent_instructions=default_intent_instructions,
                ),
            },
            *_recent_dialogue(state.get("messages", []) or []),
            {"role": "user", "content": intent_user_prompt.format(question=question)},
        ]
        result = cast(IntentSchema, llm_router.invoke(router_messages))

        if result.classification == "in_scope":
            if result.complexity == "complex":
                logger.info("intent: in_scope (complex) — multi-agent orchestrator")
                return Command(
                    goto="orchestrator",
                    # record the human turn so multi-turn history stays coherent
                    # (the orchestrator itself reads question_input, not messages)
                    update={
                        "classification_decision": "in_scope",
                        "messages": [{"role": "user", "content": question}],
                    },
                )
            logger.info("intent: in_scope (simple) — retrieving from knowledge base")
            return Command(
                goto="response_agent",
                update={
                    "classification_decision": "in_scope",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Answer this question using the knowledge base: {question}",
                        }
                    ],
                },
            )

        logger.info("intent: out_of_scope — declining")
        return Command(
            goto=END,  # type: ignore[arg-type]
            update={
                "classification_decision": "out_of_scope",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "This question is outside the scope of the local "
                        "knowledge base, so I can't answer it from the available documents.",
                    }
                ],
            },
        )

    overall_workflow = (
        StateGraph(State, input_schema=StateInput)
        .add_node("intent_router", intent_router)
        .add_node("response_agent", response_agent)
        .add_node("orchestrator", orchestrator_node)
        .add_edge(START, "intent_router")
    )
    return overall_workflow.compile(checkpointer=checkpointer)


def make_graph():
    """Graph factory for ``langgraph.json`` / LangGraph dev."""
    return build_agent()


@lru_cache(maxsize=1)
def get_default_agent():
    """Lazily-built, cached single-shot agent (no memory across invocations)."""
    return build_agent()


@lru_cache(maxsize=1)
def get_chat_agent():
    """Cached multi-turn agent backed by an in-process checkpointer.

    Invoke with ``config={"configurable": {"thread_id": <id>}}``; state (the
    conversation messages) persists per thread, so follow-up questions resolve
    against earlier turns. Swap in a persistent saver (e.g. SqliteSaver) for
    durability across process restarts.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    return build_agent(checkpointer=InMemorySaver())
