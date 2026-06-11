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
