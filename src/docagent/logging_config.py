"""Minimal logging setup.

Library code logs through module loggers (``logging.getLogger(__name__)``) and
never configures handlers itself. Entrypoints (the CLIs and the web server) call
``configure_logging()`` once at startup to attach a console handler; its level
comes from ``$LOG_LEVEL`` (default INFO). Importing the library with no entrypoint
leaves logging at Python's default, so embedding docagent stays quiet.
"""

import logging
import os

# Chatty third-party loggers — pinned to WARNING so our own INFO stays readable.
_NOISY = ("httpx", "httpcore", "urllib3", "sentence_transformers", "chromadb")


def configure_logging(level: str | None = None) -> None:
    """Attach a console handler at ``$LOG_LEVEL`` (default INFO). Idempotent-ish:
    safe to call once per entrypoint."""
    logging.basicConfig(
        level=(level or os.environ.get("LOG_LEVEL", "INFO")).upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
