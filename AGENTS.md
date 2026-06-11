# AGENTS.md

Guidance for AI agents working in this repository. See `CLAUDE.md` for the full
module-by-module guide; this is the short version.

## What this is

docagent: an agentic-RAG document QA agent on LangGraph. Ingests local docs into
Chroma, answers questions with forced citations.

## Setup

```bash
conda create -n docagent python=3.11 -c conda-forge && conda activate docagent
pip install -e .
```

## Where things live

- Graph: `src/docagent/agent.py` — `build_agent(config)` factory; router picks
  `response_agent` (simple) vs `orchestrator` (complex) by question complexity.
- Multi-agent: `src/docagent/orchestrator.py` (planner→researchers→verifier→synthesizer),
  `src/docagent/verify.py` (per-sentence citation entailment).
- Tools: `src/docagent/tools/retrieval_tools.py`
- Ingest / ask CLIs: `src/docagent/ingest.py`, `src/docagent/ask.py`
- Eval: `src/docagent/eval/` (`data/qa_cases.jsonl` dataset + `run_eval.py`)
- Tests: `tests/` (`test_unit.py`/`test_retrieval.py` need no key; `test_response.py`
  and `test_orchestrator.py` do)

## Rules

- Answer only from retrieved content; always cite sources.
- Embeddings run locally (no API key); only the answer LLM needs a key.
- Run `python tests/run_all_tests.py` before claiming tests pass.
