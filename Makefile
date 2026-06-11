.PHONY: install lint format typecheck test test-online eval ingest-demo clean

install:               ## editable install with dev extras
	pip install -e ".[dev]"

lint:                  ## ruff lint
	ruff check src tests scripts

format:                ## ruff autofix + format
	ruff check --fix src tests scripts && ruff format src tests scripts

typecheck:             ## mypy (config in pyproject)
	mypy

test:                  ## offline tests — no API key, no network
	python -m pytest tests/test_unit.py tests/test_retrieval.py -q

test-online:           ## LLM-gated tests — need OPENAI_API_KEY + ingested sample_notes
	python -m pytest tests/test_response.py tests/test_orchestrator.py tests/test_chat.py -q

eval:                  ## per-category evaluation table (needs API key)
	python -m docagent.eval.run_eval

ingest-demo:           ## fetch the 8 demo papers and ingest them
	python scripts/fetch_arxiv.py --demo && python -m docagent.ingest --path ./papers --reset

clean:
	rm -rf dist build .pytest_cache .mypy_cache **/__pycache__
