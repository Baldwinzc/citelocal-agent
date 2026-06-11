# docagent — guide for AI coding assistants

## Overview

docagent is an **agentic-RAG document question-answering agent** built on
LangGraph. It ingests local documents (Markdown / text / PDF) into a Chroma
vector store and answers questions about them, always returning citations.

## Environment setup

```bash
conda create -n docagent python=3.11 -c conda-forge
conda activate docagent
pip install -e .
```

The answer LLM needs an API key (`OPENAI_API_KEY` in `.env`), or set
`LLM_MODEL=ollama:llama3.1` for a fully local setup. Embeddings always run
locally and need no key.

## Key modules

- `src/docagent/agent.py` — the LangGraph graph, built by the `build_agent(config)`
  factory (nothing initialised at import). `intent_router` classifies scope **and
  complexity**, then routes: `in_scope + simple` → `response_agent` (the single RAG
  loop: `llm_call` → `should_continue` → `environment`, built by
  `build_research_loop`); `in_scope + complex` → `orchestrator` (multi-agent).
  `get_default_agent()` is single-shot; `get_chat_agent()` compiles with an
  `InMemorySaver` checkpointer for multi-turn memory (invoke with a `thread_id`).
  Recursion budgets are per-path, from `Configuration` (wrapper nodes invoke each
  inner graph with its own limit).
- `src/docagent/orchestrator.py` — the complex path: `planner` decomposes the
  question → parallel `researcher`s (each reuses the same retrieval loop, via the
  `Send` API) → `verifier` (per-sentence entailment) → `synthesizer` (one final
  `Answer`). Researchers write to `sub_results` (not `messages`) so parallel
  branches merge via `operator.add`.
- `src/docagent/verify.py` — `verify_claims()`: split an answer into sentences and
  check each is entailed by the retrieved evidence text (pluggable backend:
  LLM judge / injected `entail_fn` / off).
- `src/docagent/ingest.py` — ingestion CLI (`python -m docagent.ingest --path ...`).
- `src/docagent/ask.py` — single-shot query CLI (`python -m docagent.ask "..."`);
  `src/docagent/chat.py` — multi-turn REPL (`python -m docagent.chat`).
- `src/docagent/vectorstore.py` — shared embeddings + Chroma backend, imported by
  both ingest and the retrieval tools so they stay in sync.
- `src/docagent/tools/retrieval_tools.py` — `search_docs`, `list_sources`,
  `Answer` (terminal tool that forces citations), `Question`.

## Running

```bash
python -m docagent.ingest --path ./sample_docs   # build the knowledge base
python -m docagent.ask "What vector store does docagent use?"
langgraph dev                                     # LangGraph Studio
```

## Testing

```bash
python tests/run_all_tests.py         # local retrieval tests (no key)
python tests/run_all_tests.py --all   # + LLM end-to-end (needs key)
```

## Conventions

- The agent must answer **only** from retrieved content and cite its sources.
- `Answer` and `Question` are terminal tools — calling them ends the loop.
- Keep embeddings local; only the answer step may call an external LLM.
