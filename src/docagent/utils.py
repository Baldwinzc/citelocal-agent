"""Utility helpers for the document QA agent."""

import re
from typing import Any, List


def extract_message_content(message) -> str:
    """Extract content from different message types as a clean string."""
    content = message.content
    if isinstance(content, str) and "<Recursion on AIMessage with id=" in content:
        return "[Recursive content]"
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [i["text"] for i in content if isinstance(i, dict) and "text" in i]
        return "\n".join(parts)
    return str(content)


def extract_tool_calls(messages: List[Any]) -> List[str]:
    """Extract tool-call names (lower-cased) from a list of messages."""
    names: list[str] = []
    for message in messages:
        if isinstance(message, dict) and message.get("tool_calls"):
            names.extend(c["name"].lower() for c in message["tool_calls"])
        elif hasattr(message, "tool_calls") and message.tool_calls:
            names.extend(c["name"].lower() for c in message.tool_calls)
    return names


def format_messages_string(messages: List[Any]) -> str:
    """Format messages into a single string for analysis / evaluation."""
    return "\n".join(message.pretty_repr() for message in messages)


def source_of(locator: str) -> str:
    """``file.md:L1-29`` or ``file.pdf (p.3)`` -> ``file...`` (the source file)."""
    return re.split(r"[:(]", locator.strip())[0].strip()


def extract_outcome(
    result: dict,
    *,
    verify_backend: str = "off",
    llm=None,
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    entail_fn=None,
) -> dict:
    """Extract and **verify** the agent's outcome from a final graph state.

    Citations are checked against ``state['retrieved_locators']`` (what the agent
    actually retrieved this run): a citation whose locator/source was never
    retrieved is moved to ``unsupported`` instead of being trusted blindly.

    Optionally (opt-in, off by default so the common path stays cheap/offline) it
    also runs **per-sentence entailment** of the answer against ``state['evidence']``
    via ``verify_claims``: set ``verify_backend`` to "nli"/"llm" (or inject an
    ``entail_fn``) to populate ``unsupported_sentences`` + ``claim_verdicts``.

    Returns dict with keys:
        kind               : "answer" | "refusal" | "question"
        intent             : the router decision
        answer             : the answer text ("" for a clarifying question)
        question           : the clarifying question text (or None)
        citations          : verified citations (locator or source matched retrieval)
        unsupported        : citations the agent emitted but never actually retrieved
        unsupported_sentences : answer sentences not entailed by evidence (if verified)
        claim_verdicts     : per-sentence {sentence, supported} (if verified)
        trace              : the retrieval trace
    """
    intent = result.get("classification_decision", "") or ""
    retrieved_set = set(result.get("retrieved_locators", []) or [])
    retrieved_sources = {source_of(r) for r in retrieved_set}
    messages = result.get("messages", [])

    answer, question = "", None
    raw_citations: list[str] = []
    for msg in reversed(messages):
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc["name"] == "Answer":
                answer = tc["args"].get("answer", "") or ""
                raw_citations = tc["args"].get("citations", []) or []
            elif tc["name"] == "Question":
                question = tc["args"].get("content", "") or ""
        if answer or question:
            break

    if question:
        return {
            "kind": "question", "intent": intent, "question": question,
            "answer": "", "citations": [], "unsupported": [],
            "unsupported_sentences": [], "claim_verdicts": [],
            "trace": result.get("trace", []) or [],
        }

    if not answer and messages:  # refusal / out-of-scope
        answer = str(messages[-1].content)

    # Verify each citation against what was retrieved. Match on the exact locator,
    # or the source path, or the source *basename* (tolerates the agent citing
    # "x.md:L1-9" while retrieval stored "dir/x.md:L1-9").
    def _base(s: str) -> str:
        return source_of(s).rsplit("/", 1)[-1]

    retrieved_bases = {_base(r) for r in retrieved_set}
    supported, unsupported = [], []
    for c in raw_citations:
        if c in retrieved_set or source_of(c) in retrieved_sources or _base(c) in retrieved_bases:
            supported.append(c)
        else:
            unsupported.append(c)

    # Optional per-sentence entailment check of the answer against the evidence.
    unsupported_sentences, claim_verdicts = [], []
    if answer and intent == "in_scope" and (entail_fn is not None or verify_backend != "off"):
        from docagent.verify import verify_claims

        v = verify_claims(
            answer,
            result.get("evidence", []) or [],
            backend=verify_backend,
            llm=llm,
            nli_model=nli_model,
            entail_fn=entail_fn,
        )
        unsupported_sentences = v["unsupported"]
        claim_verdicts = v["verdicts"]

    return {
        "kind": "answer" if intent == "in_scope" else "refusal",
        "intent": intent, "question": None, "answer": answer,
        "citations": supported, "unsupported": unsupported,
        "unsupported_sentences": unsupported_sentences,
        "claim_verdicts": claim_verdicts,
        "trace": result.get("trace", []) or [],
    }


def show_graph(graph, xray=False):
    """Render a LangGraph mermaid diagram (for use inside notebooks/IPython)."""
    from IPython.display import Image

    return Image(graph.get_graph(xray=xray).draw_mermaid_png())
