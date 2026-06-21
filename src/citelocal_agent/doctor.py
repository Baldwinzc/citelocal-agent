"""``citelocal doctor`` — a fast preflight check for the common setup pitfalls.

Checks, in order: Python version, that the heavy deps import, that an answer model
is configured (API key for an OpenAI-style gateway, or an ``ollama:`` model), that
the gateway is reachable and actually serves the configured model (the classic
"default is gpt-4.1 but my gateway doesn't have it" trap), and whether a corpus is
ingested. Prints ✓ / ⚠ / ✗ per check; exits non-zero only if a *critical* check
(Python, imports) fails, so it's safe to run right after install, before any key.
"""

import json
import os
import sys
import urllib.request

CRITICAL = "critical"


def _check_python() -> tuple[str, str, str]:
    v = sys.version_info
    ok = v >= (3, 11)
    return (
        "ok" if ok else CRITICAL,
        f"Python {v.major}.{v.minor}.{v.micro}",
        "" if ok else "need >= 3.11 (langchain 1.x requires it)",
    )


def _check_imports() -> tuple[str, str, str]:
    missing = []
    for mod in ("langchain", "langgraph", "chromadb", "sentence_transformers",
                "bm25s", "pypdf", "fastapi"):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            missing.append(mod)
    if missing:
        return CRITICAL, "core dependencies", f"missing: {', '.join(missing)} — run pip install -e ."
    return "ok", "core dependencies importable", ""


def _model_id(llm_model: str) -> tuple[str, str]:
    """Split 'openai:gpt-5.4-mini' -> ('openai', 'gpt-5.4-mini')."""
    provider, _, name = llm_model.partition(":")
    return provider, name or provider


def _check_answer_model(env: dict) -> tuple[str, str, str]:
    llm_model = env.get("LLM_MODEL", "openai:gpt-4.1")
    provider, _ = _model_id(llm_model)
    if provider == "ollama":
        return "ok", f"answer model {llm_model} (local; no key needed)", ""
    if env.get("OPENAI_API_KEY"):
        return "ok", f"answer model {llm_model}; OPENAI_API_KEY set", ""
    return "warn", f"answer model {llm_model}", "OPENAI_API_KEY not set — add it to .env"


def _check_gateway(env: dict) -> tuple[str, str, str]:
    llm_model = env.get("LLM_MODEL", "openai:gpt-4.1")
    provider, model_name = _model_id(llm_model)
    key = env.get("OPENAI_API_KEY")
    if provider == "ollama" or not key:
        return "skip", "gateway reachability", "skipped (no OpenAI key / local model)"
    base = (env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    req = urllib.request.Request(
        f"{base}/models", headers={"Authorization": f"Bearer {key}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310 (trusted base url)
            data = json.loads(resp.read().decode())
    except Exception as e:  # noqa: BLE001
        return "warn", f"gateway {base}", f"unreachable: {type(e).__name__} ({str(e)[:60]})"
    ids = {m.get("id") for m in (data.get("data") or [])}
    if ids and model_name not in ids:
        return "warn", f"gateway {base} reachable", (
            f"but '{model_name}' is NOT in /models — set LLM_MODEL to one it serves"
        )
    return "ok", f"gateway {base} reachable; serves '{model_name}'", ""


def _check_corpus(env: dict) -> tuple[str, str, str]:
    try:
        from citelocal_agent.retriever import get_retriever

        r = get_retriever(
            env.get("CHROMA_PATH", "./chroma_db"),
            env.get("CHROMA_COLLECTION", "citelocal_agent"),
        )
        n = r.num_chunks
    except Exception as e:  # noqa: BLE001
        return "warn", "corpus", f"could not open ({type(e).__name__}: {str(e)[:60]})"
    if n == 0:
        return "warn", "corpus is empty", "run: citelocal ingest --path ./papers --reset"
    return "ok", f"corpus has {n} chunks", ""


_MARK = {"ok": "✓", "warn": "⚠", "skip": "–", CRITICAL: "✗"}


def main() -> int:
    env = dict(os.environ)
    checks = [
        _check_python(),
        _check_imports(),
        _check_answer_model(env),
        _check_gateway(env),
        _check_corpus(env),
    ]
    print("citelocal doctor\n")
    failed_critical = False
    for status, label, detail in checks:
        line = f"  {_MARK.get(status, '?')} {label}"
        if detail:
            line += f"  — {detail}"
        print(line)
        if status == CRITICAL:
            failed_critical = True
    print()
    if failed_critical:
        print("Critical checks failed — fix the ✗ items above.")
        return 1
    print("Ready. Try: citelocal ask \"...\"   (or: citelocal web)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
