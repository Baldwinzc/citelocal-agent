"""Fast unit tests — no network, no model downloads, no API key.

These cover the pure logic (RRF fusion, citation verification, chunk provenance,
Question extraction) so CI has a hard, fast, offline gate independent of any LLM
or embedding model.
"""

import pytest
from langchain_core.documents import Document

from citelocal_agent.agent import _parse_search_results, build_research_loop
from citelocal_agent.eval.qa_dataset import (
    CATEGORIES,
    INTENTS,
    SPLITS,
    load_qa_cases,
    qa_names,
)
from citelocal_agent.ingest import chunk_documents
from citelocal_agent.orchestrator import build_orchestrator
from citelocal_agent.retriever import HybridRetriever
from citelocal_agent.schemas import IntentSchema
from citelocal_agent.utils import extract_outcome, source_of
from citelocal_agent.verify import split_sentences, verify_claims


class _Msg:
    """Minimal stand-in for an AIMessage with tool_calls."""

    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls or []
        self.content = content


def test_source_of():
    assert source_of("async.md:L10-20") == "async.md"
    assert source_of("guide.pdf (p.3)") == "guide.pdf"
    assert source_of("fastapi/async.md:L1-9") == "fastapi/async.md"


def test_rrf_fusion_ranks_consensus_first():
    # 'b' is high in both lists -> should win RRF
    fused = HybridRetriever._rrf([["a", "b", "c"], ["b", "c", "a"]], 60)
    assert set(fused) == {"a", "b", "c"}
    assert fused[0] == "b"


def test_chunk_provenance_unique_ids():
    text = "\n".join(f"line {i} with some words here" for i in range(60))
    docs = [Document(page_content=text, metadata={"source": "fastapi/x.md"})]
    chunks = chunk_documents(docs, chunk_size=120, chunk_overlap=20)
    assert chunks
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))  # unique
    assert all(i.startswith("fastapi/x.md::") for i in ids)
    assert all(c.metadata["start_line"] >= 1 for c in chunks)


def test_citation_verification_drops_hallucinated():
    result = {
        "classification_decision": "in_scope",
        "retrieved_locators": ["async.md:L10-20"],
        "messages": [
            _Msg(tool_calls=[{
                "name": "Answer",
                "args": {
                    "answer": "Runs in a threadpool [async.md:L10-20].",
                    "citations": ["async.md:L10-20", "made-up.md:L1-2"],
                },
            }])
        ],
    }
    o = extract_outcome(result)
    assert o["kind"] == "answer"
    assert o["citations"] == ["async.md:L10-20"]
    assert o["unsupported"] == ["made-up.md:L1-2"]  # hallucinated -> dropped


def test_unsupported_marker_scrubbed_from_answer_prose():
    # A hallucinated inline [locator] must not leak into the user-facing answer,
    # while the supported one stays. (ask.py / web.py print o["answer"] verbatim.)
    result = {
        "classification_decision": "in_scope",
        "retrieved_locators": ["async.md:L10-20"],
        "messages": [
            _Msg(tool_calls=[{
                "name": "Answer",
                "args": {
                    "answer": "Runs in a threadpool [async.md:L10-20] [made-up.md:L1-2].",
                    "citations": ["async.md:L10-20", "made-up.md:L1-2"],
                },
            }])
        ],
    }
    o = extract_outcome(result)
    assert "made-up.md:L1-2" not in o["answer"]      # bogus marker scrubbed
    assert "[async.md:L10-20]" in o["answer"]          # supported marker kept
    assert "  " not in o["answer"]                     # no double-space artifact
    assert o["citations"] == ["async.md:L10-20"]
    assert o["unsupported"] == ["made-up.md:L1-2"]     # still recorded (metric honest)


def test_orchestrator_scratch_label_scrubbed():
    # The orchestrator-leak failure mode: an internal scratch label cited inline.
    result = {
        "classification_decision": "in_scope",
        "retrieved_locators": ["bert.pdf (p.1)"],
        "messages": [
            _Msg(tool_calls=[{
                "name": "Answer",
                "args": {
                    "answer": "BERT is an encoder [bert.pdf (p.1)] [verified findings: sub-question 1].",
                    "citations": ["bert.pdf (p.1)", "verified findings: sub-question 1"],
                },
            }])
        ],
    }
    o = extract_outcome(result)
    assert "verified findings" not in o["answer"]
    assert "[bert.pdf (p.1)]" in o["answer"]
    assert o["unsupported"] == ["verified findings: sub-question 1"]


def test_unrelated_brackets_not_stripped():
    # Brackets that are not declared (unsupported) citations must be left alone.
    result = {
        "classification_decision": "in_scope",
        "retrieved_locators": ["async.md:L10-20"],
        "messages": [
            _Msg(tool_calls=[{
                "name": "Answer",
                "args": {
                    "answer": "Step [1] runs in a threadpool [async.md:L10-20].",
                    "citations": ["async.md:L10-20"],
                },
            }])
        ],
    }
    o = extract_outcome(result)
    assert o["answer"] == "Step [1] runs in a threadpool [async.md:L10-20]."


def test_filter_citations_to_allowed():
    from citelocal_agent.utils import filter_citations_to_allowed

    # The synthesizer's only legitimate sources are the union of researcher
    # citations; a fabricated label outside it is dropped and its marker scrubbed.
    answer = (
        "BERT is an encoder [bert.pdf (p.1)] and is pre-trained "
        "[verified findings: sub-question 2]."
    )
    allowed = ["bert.pdf (p.1)", "gpt.pdf (p.3)"]
    cleaned, kept = filter_citations_to_allowed(
        answer, ["bert.pdf (p.1)", "verified findings: sub-question 2"], allowed
    )
    assert kept == ["bert.pdf (p.1)"]
    assert "verified findings" not in cleaned
    assert "[bert.pdf (p.1)]" in cleaned
    assert "  " not in cleaned


def test_filter_citations_to_allowed_noop_when_all_allowed():
    from citelocal_agent.utils import filter_citations_to_allowed

    answer = "BERT is an encoder [bert.pdf (p.1)]."
    cleaned, kept = filter_citations_to_allowed(
        answer, ["bert.pdf (p.1)"], ["bert.pdf (p.1)", "other.md:L1-2"]
    )
    assert cleaned == answer  # nothing dropped -> prose untouched
    assert kept == ["bert.pdf (p.1)"]


def test_research_loop_budget_fallback():
    # A runaway loop (GraphRecursionError) degrades to an honest message instead
    # of crashing the request; extract_outcome then renders it as a normal answer.
    from langgraph.errors import GraphRecursionError

    from citelocal_agent.agent import _BUDGET_FALLBACK_MSG, _run_research_loop

    class _Boom:
        def invoke(self, *a, **k):
            raise GraphRecursionError("limit reached")

    msgs = [{"role": "user", "content": "some unanswerable, sprawling question"}]
    out = _run_research_loop(_Boom(), msgs, 12)
    assert out["messages"][-1].content == _BUDGET_FALLBACK_MSG
    assert out["trace"] == [{"step": "budget_exhausted"}]

    o = extract_outcome({**out, "classification_decision": "in_scope"})
    assert o["answer"] == _BUDGET_FALLBACK_MSG
    assert o["citations"] == [] and o["unsupported"] == []


def test_research_loop_passthrough_on_success():
    from citelocal_agent.agent import _run_research_loop

    class _Ok:
        def invoke(self, *a, **k):
            return {"messages": ["ok"], "trace": [], "retrieved_locators": []}

    assert _run_research_loop(_Ok(), [], 12) == {
        "messages": ["ok"], "trace": [], "retrieved_locators": []
    }


def test_cli_version_and_usage(capsys):
    from citelocal_agent import version
    from citelocal_agent.cli import main

    assert main(["--version"]) == 0
    assert version in capsys.readouterr().out
    assert main([]) == 0
    assert "usage: citelocal" in capsys.readouterr().out


def test_cli_unknown_command_exits_nonzero(capsys):
    from citelocal_agent.cli import main

    assert main(["frobnicate"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_cli_resolve_maps_known_commands():
    from citelocal_agent import doctor
    from citelocal_agent.cli import _resolve

    assert _resolve("doctor") is doctor.main      # known -> callable
    assert callable(_resolve("ask"))
    assert _resolve("frobnicate") is None          # unknown -> None


def test_doctor_python_and_model_checks():
    from citelocal_agent.doctor import _check_answer_model, _check_python

    assert _check_python()[0] == "ok"  # CI runs >= 3.11

    assert _check_answer_model({"LLM_MODEL": "ollama:llama3.1"})[0] == "ok"  # local, no key
    assert _check_answer_model(
        {"LLM_MODEL": "openai:gpt-5.4-mini", "OPENAI_API_KEY": "sk-x"}
    )[0] == "ok"
    assert _check_answer_model({"LLM_MODEL": "openai:gpt-5.4-mini"})[0] == "warn"  # no key


def test_question_extraction():
    result = {
        "classification_decision": "in_scope",
        "retrieved_locators": [],
        "messages": [
            _Msg(tool_calls=[{"name": "Question", "args": {"content": "Which version?"}}])
        ],
    }
    o = extract_outcome(result)
    assert o["kind"] == "question"
    assert o["question"] == "Which version?"


def test_qa_dataset_loads_and_validates():
    cases = load_qa_cases()
    assert len(cases) >= 8, "expected at least the migrated seed cases"
    # required keys + valid enum values on every row (load already validates)
    for c in cases:
        assert c["intent"] in INTENTS
        assert c["category"] in CATEGORIES
        assert c["split"] in SPLITS
        assert isinstance(c["expected_sources"], list)
        # in_scope cases must name at least one expected source; refusals must not
        if c["intent"] == "in_scope":
            assert c["expected_sources"], f"{c['id']} in_scope but no expected_sources"
        else:
            assert not c["expected_sources"], f"{c['id']} refusal but has expected_sources"


def test_qa_dataset_ids_unique_and_views_aligned():
    assert len(qa_names) == len(set(qa_names)), "case ids must be unique"
    # the offline_sample split must be non-empty (it anchors the offline LLM suite)
    assert load_qa_cases(split="offline_sample"), "offline_sample split is empty"


def test_qa_dataset_split_filter():
    full = load_qa_cases(split="full_corpus")
    offline = load_qa_cases(split="offline_sample")
    assert all(c["split"] == "full_corpus" for c in full)
    assert all(c["split"] == "offline_sample" for c in offline)
    assert len(full) + len(offline) == len(load_qa_cases())


# --- M2: multi-agent helpers (all offline, no LLM) ---

# Exactly the format search_docs emits (retrieval_tools.py), incl. a blank line
# inside a chunk, to prove the parser captures locator + full text per block.
_SEARCH_OUTPUT = (
    "[1] locator: bert.pdf (p.3)  (relevance 2.60)\n"
    "BERT is a bidirectional Transformer encoder.\n\n"
    "Pre-trained on masked LM.\n\n"
    "[2] locator: notes/attention.md:L1-9  (relevance 1.20)\n"
    "Attention relates any two positions in one step."
)


def test_parse_search_results_captures_locator_and_text():
    pairs = _parse_search_results(_SEARCH_OUTPUT)
    assert [loc for loc, _ in pairs] == ["bert.pdf (p.3)", "notes/attention.md:L1-9"]
    # text includes the chunk's internal blank line, not the next block
    assert "Pre-trained on masked LM." in pairs[0][1]
    assert "[2]" not in pairs[0][1]
    assert pairs[1][1] == "Attention relates any two positions in one step."


def test_parse_search_results_empty_on_no_hits():
    assert _parse_search_results("No sufficiently relevant chunks found.") == []


def test_intent_schema_complexity_default_and_set():
    assert IntentSchema(reasoning="x", classification="in_scope").complexity == "simple"
    s = IntentSchema(reasoning="x", classification="in_scope", complexity="complex")
    assert s.complexity == "complex"


def test_split_sentences():
    sents = split_sentences("First fact. Second fact! Third fact?")
    assert sents == ["First fact.", "Second fact!", "Third fact?"]
    assert split_sentences("") == []


def test_verify_claims_drops_unentailed_sentence():
    # stub entailment: a claim is supported iff its key term appears in evidence
    def stub(claim, evidence_text):
        return "threadpool" in claim.lower()

    evidence = [{"locator": "async.md:L1-9", "text": "irrelevant"}]
    out = verify_claims(
        "It runs in a threadpool [async.md:L1-9]. It also cures cancer.",
        evidence,
        entail_fn=stub,
    )
    assert out["supported"] == ["It runs in a threadpool [async.md:L1-9]."]
    assert out["unsupported"] == ["It also cures cancer."]


def test_verify_claims_off_backend_is_noop():
    out = verify_claims("Any claim.", [{"locator": "x", "text": ""}], backend="off")
    assert out["unsupported"] == [] and out["supported"] == ["Any claim."]


def test_verify_claims_no_evidence_unsupported():
    out = verify_claims("A claim.", [], backend="llm")
    assert out["unsupported"] == ["A claim."]  # nothing to ground against


# --- M3b: NLI backend (label logic + per-chunk aggregation, no model download) ---

def test_argmax_and_row_is_entailment():
    from citelocal_agent.verify import _argmax, _row_is_entailment

    assert _argmax([0.9, 0.1, 0.0]) == 0
    assert _argmax([0.1, 0.9, 0.0]) == 1
    # entailment is index 1 for cross-encoder/nli-* models
    assert _row_is_entailment([0.1, 0.9, 0.0]) is True
    assert _row_is_entailment([0.9, 0.1, 0.0]) is False


class _FakeNLI:
    """Stub cross-encoder: entailment (idx 1) iff claim and chunk share 'threadpool'."""

    def predict(self, pairs):
        rows = []
        for chunk, claim in pairs:
            ent = "threadpool" in claim.lower() and "threadpool" in chunk.lower()
            rows.append([0.0, 1.0, 0.0] if ent else [1.0, 0.0, 0.0])
        return rows


def test_verify_claims_nli_any_chunk_entails(monkeypatch):
    monkeypatch.setattr("citelocal_agent.verify._get_nli", lambda name: _FakeNLI())
    evidence = [
        {"locator": "a.md:L1-9", "text": "the call runs in a threadpool"},
        {"locator": "b.md:L1-9", "text": "unrelated noise"},
    ]
    out = verify_claims(
        "It runs in a threadpool [a.md:L1-9]. It also cures cancer.",
        evidence,
        backend="nli",
    )
    # supported because the FIRST chunk entails it; second sentence entailed by none
    assert out["supported"] == ["It runs in a threadpool [a.md:L1-9]."]
    assert out["unsupported"] == ["It also cures cancer."]


def test_extract_outcome_entailment_optin():
    from citelocal_agent.utils import extract_outcome

    class _Msg:
        def __init__(self, tool_calls):
            self.tool_calls = tool_calls
            self.content = ""

    answer_tc = {
        "name": "Answer",
        "args": {
            "answer": "It runs in a threadpool. It also cures cancer.",
            "citations": ["async.md:L1-9"],
        },
        "id": "1",
    }
    result = {
        "classification_decision": "in_scope",
        "retrieved_locators": ["async.md:L1-9"],
        "evidence": [{"locator": "async.md:L1-9", "text": "runs in a threadpool"}],
        "messages": [_Msg([answer_tc])],
    }

    # default (off): citations verified, but no sentence-level check runs
    base = extract_outcome(result)
    assert base["citations"] == ["async.md:L1-9"]
    assert base["unsupported_sentences"] == [] and base["claim_verdicts"] == []

    # opt-in with an injected stub scorer (offline, no model)
    verified = extract_outcome(
        result, entail_fn=lambda claim, ev: "threadpool" in claim.lower()
    )
    assert verified["unsupported_sentences"] == ["It also cures cancer."]
    assert len(verified["claim_verdicts"]) == 2


class _FakeLLM:
    """Build-time stand-in: only the wiring methods are exercised, never invoke."""

    def with_structured_output(self, schema):
        return self

    def bind_tools(self, tools, tool_choice=None):
        return self


def test_research_loop_graph_compiles():
    g = build_research_loop(_FakeLLM(), {}, "system prompt").get_graph()
    assert {"llm_call", "environment"} <= set(g.nodes)


def test_orchestrator_graph_wiring():
    g = build_orchestrator(_FakeLLM(), object()).get_graph()
    assert {"planner", "researcher", "verifier", "synthesizer"} <= set(g.nodes)


def test_merge_subgraph_result_slices_new_messages():
    from citelocal_agent.agent import _merge_subgraph_result

    out = {
        "messages": ["u", "a1", "a2"],
        "trace": [{"step": "x"}],
        "retrieved_locators": ["l"],
        "evidence": [{"locator": "l", "text": "t"}],
    }
    upd = _merge_subgraph_result(out, prior_msg_count=1)
    assert upd["messages"] == ["a1", "a2"]  # the 1 prior (input) message is dropped
    assert upd["trace"] == [{"step": "x"}]
    assert upd["retrieved_locators"] == ["l"]


def test_merge_subgraph_result_omits_absent_keys():
    from citelocal_agent.agent import _merge_subgraph_result

    upd = _merge_subgraph_result({"messages": ["only"]}, 0)
    assert upd["messages"] == ["only"]
    assert "trace" not in upd and "evidence" not in upd


# --- C: multi-turn — router conversation context helper (offline) ---

def test_recent_dialogue_filters_and_strips_prefix():
    from citelocal_agent.agent import _recent_dialogue

    msgs = [
        {"role": "user", "content": "Answer this question using the knowledge base: What is BM25?"},
        {"role": "assistant", "content": ""},      # tool-call-only -> skipped
        {"role": "tool", "content": "observation"},  # tool msg -> skipped
        {"role": "assistant", "content": "BM25 is a sparse ranking function."},
    ]
    assert _recent_dialogue(msgs) == [
        {"role": "user", "content": "What is BM25?"},  # instruction prefix stripped
        {"role": "assistant", "content": "BM25 is a sparse ranking function."},
    ]


def test_recent_dialogue_caps_and_handles_message_objects():
    from citelocal_agent.agent import _recent_dialogue

    class HumanMessage:  # name drives the role mapping
        def __init__(self, content):
            self.content = content

    assert len(_recent_dialogue([{"role": "user", "content": f"q{i}"} for i in range(6)], 3)) == 3
    assert _recent_dialogue([HumanMessage("hello")]) == [{"role": "user", "content": "hello"}]


# --- M3d: API hardening primitives (framework-free, offline) ---

def test_rate_limiter_fixed_window():
    from citelocal_agent.security import RateLimiter

    rl = RateLimiter(max_requests=2, window_seconds=10)
    assert rl.allow("ip", 0.0)
    assert rl.allow("ip", 1.0)
    assert not rl.allow("ip", 2.0)   # 3rd request within the window -> blocked
    assert rl.allow("ip", 12.0)      # window elapsed -> allowed again
    assert rl.allow("other-ip", 2.0)  # keys are independent


def test_api_key_ok(monkeypatch):
    from citelocal_agent.security import api_key_ok

    monkeypatch.delenv("DOCAGENT_API_KEY", raising=False)
    assert api_key_ok(None) is True   # auth disabled when no key configured
    monkeypatch.setenv("DOCAGENT_API_KEY", "secret")
    assert api_key_ok("secret") is True
    assert api_key_ok("wrong") is False
    assert api_key_ok(None) is False


# --- Web tools (opt-in): all offline via an injected fake backend, no network ---

from citelocal_agent.tools import get_web_backend, make_web_tools  # noqa: E402


class _FakeWebBackend:
    """Deterministic stand-in for a web backend — no network, no key."""

    def search(self, query, k):
        results = [
            {"url": "https://example.com/a", "title": "Title A", "snippet": "Snippet A about X."},
            {"url": "https://example.com/b", "title": "Title B", "snippet": "Snippet B.", "score": 0.42},
        ]
        return results[:k]

    def fetch(self, url, max_chars):
        return (f"Full readable page text for {url}. " * 5)[:max_chars]


def _tools_by_name(backend, max_results=5, fetch_chars=4000):
    return {t.name: t for t in make_web_tools(backend, max_results, fetch_chars)}


def test_web_search_emits_parseable_web_locators():
    by = _tools_by_name(_FakeWebBackend())
    out = by["web_search"].invoke({"query": "anything"})
    pairs = _parse_search_results(out)  # the SAME parser tool_node uses
    assert [loc for loc, _ in pairs] == [
        "web:https://example.com/a",
        "web:https://example.com/b",
    ]
    # title + snippet both land in the evidence text for the first result
    assert "Title A" in pairs[0][1] and "Snippet A about X." in pairs[0][1]


def test_fetch_url_emits_single_web_block_truncated():
    by = _tools_by_name(_FakeWebBackend(), fetch_chars=40)
    out = by["fetch_url"].invoke({"url": "https://example.com/a"})
    pairs = _parse_search_results(out)
    assert len(pairs) == 1
    assert pairs[0][0] == "web:https://example.com/a"
    assert pairs[0][1] and len(pairs[0][1]) <= 40  # respects fetch_chars cap


def test_web_search_no_results_message():
    class _Empty:
        def search(self, q, k):
            return []

        def fetch(self, u, m):
            return ""

    by = _tools_by_name(_Empty())
    assert "No web results" in by["web_search"].invoke({"query": "x"})


def test_source_of_web_locator_is_exact():
    # web locators verify by full URL; files keep basename/source behavior
    assert source_of("web:https://x.com/a?b=1") == "web:https://x.com/a?b=1"
    assert source_of("async.md:L10-20") == "async.md"


def test_extract_outcome_verifies_web_citations():
    result = {
        "classification_decision": "in_scope",
        "retrieved_locators": ["web:https://example.com/a"],  # only this was fetched
        "messages": [
            _Msg(tool_calls=[{
                "name": "Answer",
                "args": {
                    "answer": "The release is 2.1 [web:https://example.com/a].",
                    "citations": [
                        "web:https://example.com/a",      # fetched -> supported
                        "web:https://never-fetched.com",  # not fetched -> dropped
                    ],
                },
            }])
        ],
    }
    o = extract_outcome(result)
    assert o["citations"] == ["web:https://example.com/a"]
    assert o["unsupported"] == ["web:https://never-fetched.com"]


def test_get_web_backend_factory():
    from citelocal_agent.tools.web_tools import DuckDuckGoBackend, TavilyBackend

    assert isinstance(get_web_backend("ddg"), DuckDuckGoBackend)
    assert isinstance(get_web_backend(), DuckDuckGoBackend)  # default
    assert isinstance(get_web_backend("tavily"), TavilyBackend)
    with pytest.raises(ValueError):
        get_web_backend("nope")


def test_intent_schema_accepts_web_answerable():
    s = IntentSchema(reasoning="x", classification="web_answerable")
    assert s.classification == "web_answerable"


def test_research_loop_records_web_evidence():
    """tool_node records web_search locators + evidence (just like search_docs)."""
    from langchain_core.messages import AIMessage

    tools_by_name = {t.name: t for t in make_web_tools(_FakeWebBackend(), 5, 4000)}

    class _TwoStepLLM:
        """Step 1: call web_search. Step 2: no tool call, so the loop ends."""

        def __init__(self):
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "web_search", "args": {"query": "x"},
                                 "id": "1", "type": "tool_call"}],
                )
            return AIMessage(content="done")  # no tool_calls -> should_continue -> END

    loop = build_research_loop(_TwoStepLLM(), tools_by_name, "sys")
    out = loop.invoke({"messages": [{"role": "user", "content": "q"}]})
    assert "web:https://example.com/a" in out["retrieved_locators"]
    assert any(e["locator"] == "web:https://example.com/a" for e in out["evidence"])
    assert any(t.get("step") == "web_search" for t in out["trace"])
