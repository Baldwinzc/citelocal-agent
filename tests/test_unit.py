"""Fast unit tests — no network, no model downloads, no API key.

These cover the pure logic (RRF fusion, citation verification, chunk provenance,
Question extraction) so CI has a hard, fast, offline gate independent of any LLM
or embedding model.
"""

from langchain_core.documents import Document

from docagent.agent import _parse_search_results, build_research_loop
from docagent.eval.qa_dataset import (
    CATEGORIES,
    INTENTS,
    SPLITS,
    load_qa_cases,
    qa_names,
)
from docagent.ingest import chunk_documents
from docagent.orchestrator import build_orchestrator
from docagent.retriever import HybridRetriever
from docagent.schemas import IntentSchema
from docagent.utils import extract_outcome, source_of
from docagent.verify import split_sentences, verify_claims


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
    from docagent.verify import _argmax, _row_is_entailment

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
    monkeypatch.setattr("docagent.verify._get_nli", lambda name: _FakeNLI())
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
    from docagent.utils import extract_outcome

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
    from docagent.agent import _merge_subgraph_result

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
    from docagent.agent import _merge_subgraph_result

    upd = _merge_subgraph_result({"messages": ["only"]}, 0)
    assert upd["messages"] == ["only"]
    assert "trace" not in upd and "evidence" not in upd


# --- C: multi-turn — router conversation context helper (offline) ---

def test_recent_dialogue_filters_and_strips_prefix():
    from docagent.agent import _recent_dialogue

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
    from docagent.agent import _recent_dialogue

    class HumanMessage:  # name drives the role mapping
        def __init__(self, content):
            self.content = content

    assert len(_recent_dialogue([{"role": "user", "content": f"q{i}"} for i in range(6)], 3)) == 3
    assert _recent_dialogue([HumanMessage("hello")]) == [{"role": "user", "content": "hello"}]
