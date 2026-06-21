#!/usr/bin/env sh
# One-command setup for citelocal-agent using uv (handles Python + venv + deps).
#
#   ./install.sh
#   .venv/bin/citelocal doctor      # verify, then edit .env (set OPENAI_API_KEY)
#
# Re-runnable and safe: it won't overwrite an existing .env.
set -eu

if ! command -v uv >/dev/null 2>&1; then
    echo "==> installing uv (Python toolchain)…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv lands in ~/.local/bin (or ~/.cargo/bin); make it visible for this run.
    PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    export PATH
fi

echo "==> creating .venv (Python 3.12) and installing citelocal-agent…"
uv venv --python 3.12
uv pip install -e .

if [ ! -f .env ]; then
    cp .env.example .env
    echo "==> wrote .env from .env.example — set OPENAI_API_KEY (or use an ollama: model)"
fi

echo ""
echo "Done. Next:"
echo "  1. edit .env  (set OPENAI_API_KEY; optionally OPENAI_BASE_URL / LLM_MODEL)"
echo "  2. .venv/bin/citelocal doctor"
echo "  3. .venv/bin/citelocal ingest --path ./papers --reset   # after fetching papers"
echo "  4. .venv/bin/citelocal web     # or: citelocal ask \"...\""
