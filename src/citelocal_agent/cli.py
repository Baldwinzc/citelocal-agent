"""Unified ``citelocal`` command — a thin dispatcher over the module entry points.

    citelocal ask "How is BERT related to the Transformer?"
    citelocal chat
    citelocal ingest --path ./papers --reset
    citelocal web
    citelocal doctor

Each subcommand delegates to that module's own ``main()`` (and its argparse), so
flags are identical to the ``python -m citelocal_agent.<x>`` form — this just gives
users a single installed command instead of remembering module paths.
"""

import sys
from importlib import import_module

from citelocal_agent import version

# subcommand -> "module:function" (imported lazily so `citelocal --version` is instant)
_COMMANDS = {
    "ask": "citelocal_agent.ask:main",
    "chat": "citelocal_agent.chat:main",
    "ingest": "citelocal_agent.ingest:main",
    "web": "citelocal_agent.web:main",
    "doctor": "citelocal_agent.doctor:main",
}

_USAGE = """usage: citelocal <command> [args]

commands:
  ask QUESTION        answer one question (add --trace to show retrieval)
  chat                interactive multi-turn REPL
  ingest --path DIR   index a folder of PDFs / Markdown / text
  web                 serve the FastAPI app + chat UI on :8000
  doctor              check the install, config, corpus and LLM gateway

  -V, --version       print version
Run 'citelocal <command> --help' for a command's own flags."""


def _resolve(command: str):
    """Return a subcommand's ``main`` callable, or ``None`` if unknown."""
    target = _COMMANDS.get(command)
    if target is None:
        return None
    mod, attr = target.split(":")
    return getattr(import_module(mod), attr)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_USAGE)
        return 0
    if argv[0] in ("-V", "--version", "version"):
        print(f"citelocal {version}")
        return 0

    command, rest = argv[0], argv[1:]
    fn = _resolve(command)
    if fn is None:
        print(f"citelocal: unknown command {command!r}\n\n{_USAGE}", file=sys.stderr)
        return 2

    # Hand the subcommand's own argparse a clean argv: prog name + just its flags.
    sys.argv = [f"citelocal {command}"] + rest
    fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
