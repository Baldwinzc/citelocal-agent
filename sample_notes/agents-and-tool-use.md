# Agents and Tool Use

## What an agent is

An LLM **agent** is a model placed in a loop where it can call **tools** (search,
calculators, code, APIs), observe the results, and decide what to do next, instead
of producing a single answer in one shot. The loop continues until the model
decides it has enough to answer (or hits a step budget).

## Tool calling

Tool use works by giving the model a set of tool schemas; the model emits a
structured **tool call** (name + arguments), the system executes it and returns
the result as an observation, and the model continues. Forcing a tool call (rather
than free text) is how a system guarantees, e.g., that an answer always comes with
a structured payload such as citations.

## ReAct: reason + act

The **ReAct** pattern interleaves reasoning and acting: the model thinks about
what it needs, calls a tool, reads the observation, and repeats. This lets it
gather evidence incrementally and reformulate when a tool result is unhelpful,
rather than committing to one query.

## Agentic RAG

Plain RAG retrieves once and generates. **Agentic RAG** puts retrieval inside the
agent loop: the model searches, inspects what came back, reformulates the query,
searches again, and answers only when it has enough evidence — or declines.
Because the loop can re-query, retrieval quality (hybrid search + reranking)
largely determines how well the agent does.

## Stopping and step budgets

An agent needs a **terminal action** (e.g. an "answer" or "ask-clarification"
tool) to end the loop, and a **step budget** (a recursion limit) so a model that
keeps searching without converging fails safely instead of looping forever.

## When agents help — and when they don't

Agents shine when a task needs multiple steps, tools, or reformulation (multi-hop
questions, gathering evidence across sources). For a simple, single-fact lookup
the extra loop is wasted latency and cost, so a good system **routes** simple
questions to a direct path and reserves the agent loop for genuinely complex ones.
