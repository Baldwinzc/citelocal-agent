"""Claim-level citation verification.

``extract_outcome`` already drops citations whose *locator* was never retrieved.
This module goes one level deeper: it splits an answer into sentences and checks
each one is actually **entailed by the evidence text** that was retrieved — so a
fluent sentence citing a real chunk that doesn't actually support it is caught.

The backend is pluggable:
    - ``backend="llm"``  : one structured LLM call grades all sentences at once
                           (used by the multi-agent Verifier — M2).
    - ``entail_fn=...``  : inject a ``(claim, evidence_text) -> bool`` scorer; this
                           is how an offline stub (tests) or an NLI cross-encoder
                           (M3b) plugs in without touching callers.
    - ``backend="off"``  : no-op (everything supported) — keeps CI free/offline.

``verify_claims`` returns ``{supported, unsupported, verdicts}`` where ``verdicts``
is one ``{sentence, supported}`` per sentence, in order.
"""

import re
from typing import Callable

from pydantic import BaseModel, Field

# Sentence splitter: break on ., !, ? followed by whitespace. Good enough for
# answer prose and stays offline (no nltk/spacy download).
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# Inline locator markers like "[bert.pdf (p.3)]" — stripped before entailment so
# the check is about the claim, not the citation syntax.
_LOCATOR_MARKER_RE = re.compile(r"\[[^\]]*\]")


def split_sentences(text: str) -> list[str]:
    """Split answer text into non-empty, stripped sentences."""
    if not text or not text.strip():
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _claim_text(sentence: str) -> str:
    """The sentence with inline [locator] markers removed."""
    return _LOCATOR_MARKER_RE.sub("", sentence).strip()


class _SentenceVerdicts(BaseModel):
    """Per-sentence grounding verdicts, aligned to the numbered input sentences."""

    supported: list[bool] = Field(
        description="One boolean per numbered sentence, in order: true iff the "
        "sentence is supported by (entailed by) the provided evidence."
    )


_VERIFY_SYS = (
    "You verify that each sentence of an answer is grounded in the provided "
    "evidence passages. A sentence is supported ONLY if the evidence states or "
    "directly implies it; general knowledge does not count. Return one boolean "
    "per numbered sentence, in the same order."
)


def _llm_verify(llm, sentences: list[str], evidence_text: str) -> list[bool]:
    """One structured LLM call grading all sentences against the evidence."""
    numbered = "\n".join(f"{i + 1}. {_claim_text(s)}" for i, s in enumerate(sentences))
    out = llm.with_structured_output(_SentenceVerdicts).invoke(
        [
            {"role": "system", "content": _VERIFY_SYS},
            {
                "role": "user",
                "content": f"Evidence:\n{evidence_text}\n\nSentences:\n{numbered}",
            },
        ]
    )
    flags = list(out.supported)
    # Be robust to a wrong-length list: pad missing with False, truncate extra.
    if len(flags) < len(sentences):
        flags += [False] * (len(sentences) - len(flags))
    return flags[: len(sentences)]


def verify_claims(
    answer_text: str,
    evidence: list[dict] | None,
    *,
    backend: str = "llm",
    llm=None,
    entail_fn: Callable[[str, str], bool] | None = None,
) -> dict:
    """Verify each sentence of ``answer_text`` against ``evidence`` chunk texts.

    ``evidence`` is a list of ``{"locator", "text"}`` dicts (as accumulated in
    ``State['evidence']``). Returns ``{supported, unsupported, verdicts}``.
    """
    sentences = split_sentences(answer_text)
    if not sentences:
        return {"supported": [], "unsupported": [], "verdicts": []}

    evidence_text = "\n\n".join((e.get("text") or "") for e in (evidence or [])).strip()

    if entail_fn is not None:
        flags = [entail_fn(_claim_text(s), evidence_text) for s in sentences]
    elif backend == "off":
        flags = [True] * len(sentences)
    elif not evidence_text:
        # No evidence at all -> nothing can be grounded.
        flags = [False] * len(sentences)
    elif backend == "llm":
        if llm is None:
            raise ValueError("verify_claims(backend='llm') requires an llm")
        flags = _llm_verify(llm, sentences, evidence_text)
    else:
        raise ValueError(f"unknown verify backend: {backend!r}")

    verdicts = [{"sentence": s, "supported": bool(f)} for s, f in zip(sentences, flags)]
    return {
        "supported": [v["sentence"] for v in verdicts if v["supported"]],
        "unsupported": [v["sentence"] for v in verdicts if not v["supported"]],
        "verdicts": verdicts,
    }
