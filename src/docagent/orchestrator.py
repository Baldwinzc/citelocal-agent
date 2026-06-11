"""Multi-agent orchestrator subgraph — the 'complex' path.

    planner --(Send fan-out)--> researcher × N (parallel) --> verifier --> synthesizer

Built for multi-hop questions: the Planner splits the question into focused
sub-questions, each Researcher runs the *same* retrieval loop the simple path uses
(scoped to one sub-question) and writes its finding into ``sub_results`` — **not**
``messages`` — so N parallel branches merge cleanly via the ``operator.add``
reducer instead of racing ``add_messages``. The Verifier checks each finding's
sentences are entailed by the evidence it actually retrieved; the Synthesizer
combines the verified findings into ONE final ``Answer`` tool call, so
``extract_outcome`` (and every caller) keeps working unchanged.

``build_orchestrator`` takes the already-compiled ``research_loop`` from
``agent.build_research_loop`` (reuse, no duplication) — it does not import
``agent``, so there is no import cycle.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from docagent.prompts import (
    planner_system_prompt,
    planner_user_prompt,
    synthesizer_system_prompt,
)
from docagent.schemas import PlanSchema, State
from docagent.tools import Answer
from docagent.utils import extract_outcome
from docagent.verify import verify_claims

# Each Researcher runs an independent retrieval loop; cap its steps like the CLI.
RESEARCH_RECURSION_LIMIT = 12
MAX_SUB_QUESTIONS = 4


def build_orchestrator(llm, research_loop, *, verify_backend: str = "llm"):
    """Compile the planner→researchers→verifier→synthesizer subgraph.

    ``research_loop`` is the compiled retrieval-loop graph (shared with the simple
    path). ``verify_backend`` is passed to ``verify_claims`` for the Verifier.
    """
    planner_llm = llm.with_structured_output(PlanSchema)
    # Only the Answer tool is bound + forced, so the synthesizer always finishes
    # with a single Answer tool call that extract_outcome can read.
    synth_llm = llm.bind_tools([Answer], tool_choice="any")

    def planner(state: State):
        question = state["question_input"].get("question", "")
        plan = planner_llm.invoke(
            [
                {"role": "system", "content": planner_system_prompt},
                {"role": "user", "content": planner_user_prompt.format(question=question)},
            ]
        )
        subs = [s.strip() for s in plan.sub_questions if s and s.strip()] or [question]
        subs = subs[:MAX_SUB_QUESTIONS]
        return {
            "sub_questions": subs,
            "trace": [{"step": "planner", "sub_questions": subs}],
        }

    def fan_out(state: State):
        """Conditional edge: one parallel Researcher per sub-question."""
        return [
            Send("researcher", {"sub_question": sq, "sub_id": i})
            for i, sq in enumerate(state["sub_questions"])
        ]

    def researcher(state: dict):
        """Run the retrieval loop on one sub-question (private Send state)."""
        sq = state["sub_question"]
        sub_id = state["sub_id"]
        result = research_loop.invoke(
            {"messages": [
                {"role": "user",
                 "content": f"Answer this question using the knowledge base: {sq}"}
            ]},
            config={"recursion_limit": RESEARCH_RECURSION_LIMIT},
        )
        o = extract_outcome({**result, "classification_decision": "in_scope"})
        evidence = result.get("evidence", []) or []
        return {
            "sub_results": [
                {
                    "sub_id": sub_id,
                    "sub_question": sq,
                    "answer": o["answer"],
                    "citations": o["citations"],
                    "evidence": evidence,
                }
            ],
            # Bubble these up to the parent so the final answer's citations verify
            # against everything every researcher actually retrieved.
            "trace": result.get("trace", []) or [],
            "retrieved_locators": result.get("retrieved_locators", []) or [],
            "evidence": evidence,
        }

    def verifier(state: State):
        """Per-finding sentence-level entailment check against its evidence."""
        verified = []
        for sr in sorted(state["sub_results"], key=lambda r: r["sub_id"]):
            v = verify_claims(
                sr["answer"], sr["evidence"], backend=verify_backend, llm=llm
            )
            verified.append({**sr, "verification": v})
        return {
            "verified_results": verified,
            "trace": [{"step": "verifier", "findings": len(verified)}],
        }

    def synthesizer(state: State):
        """Combine verified findings into one final Answer tool call."""
        question = state["question_input"].get("question", "")
        findings = []
        for sr in state["verified_results"]:
            supported = sr["verification"]["supported"]
            # Prefer the entailment-supported sentences; fall back to the full
            # sub-answer if the verifier flagged none (avoid dropping everything).
            body = " ".join(supported) if supported else sr["answer"]
            findings.append(
                f"Sub-question: {sr['sub_question']}\n"
                f"Finding: {body}\n"
                f"Citations: {', '.join(sr['citations']) or '(none)'}"
            )
        context = "\n\n".join(findings) or "(no findings)"
        msg = synth_llm.invoke(
            [
                {"role": "system", "content": synthesizer_system_prompt},
                {
                    "role": "user",
                    "content": f"Original question:\n{question}\n\n"
                    f"Verified findings:\n{context}\n\n"
                    "Write the final answer and call the Answer tool.",
                },
            ]
        )
        return {"messages": [msg], "trace": [{"step": "synthesizer"}]}

    builder = StateGraph(State)
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("verifier", verifier)
    builder.add_node("synthesizer", synthesizer)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", fan_out, ["researcher"])
    builder.add_edge("researcher", "verifier")
    builder.add_edge("verifier", "synthesizer")
    builder.add_edge("synthesizer", END)
    return builder.compile()
