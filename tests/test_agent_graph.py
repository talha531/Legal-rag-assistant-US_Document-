"""End-to-end agent test with an in-memory Qdrant, a hashing embedder and a scripted fake LLM
(no network, no API key)."""
import dataclasses
import json

import pytest

pytest.importorskip("qdrant_client")
pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from qdrant_client import QdrantClient  # noqa: E402

from app.agent.graph import build_graph, initial_state, run_agent  # noqa: E402
from app.config import Settings  # noqa: E402
from app.embeddings import HashingEmbedder  # noqa: E402
from app.ingestion.chunker import LegalChunker  # noqa: E402
from app.models import DocumentInfo, PageElement  # noqa: E402
from app.vectorstore import LegalVectorStore  # noqa: E402

DOC = DocumentInfo("doc1", "Sample Regs", "sample.pdf", "April 1, 1996", "99")


def el(page, text, etype="text", heading=False):
    return PageElement(page, etype, text, (0, 0, 1, 1), is_heading=heading)


class FakeLLM:
    model = "fake"

    def __init__(self, sufficient=True):
        self.sufficient = sufficient
        self.calls = []

    def complete(self, system, user, json_mode=False, max_tokens=None):
        self.calls.append(system[:30])
        if "query-analysis" in system:
            return json.dumps({"intent": "legal_question", "standalone_question": "How long must batch records be kept?",
                               "search_query": "batch records retention period years", "section_ref": None, "part_ref": None})
        if "evidence-validation" in system:
            return json.dumps({"sufficient": self.sufficient, "confidence": 0.9, "reason": "test", "missing": "x", "suggested_query": ""})
        if "improve failed search queries" in system:
            return json.dumps({"search_query": f"alternative wording {len(self.calls)}", "strategy": "synonyms"})
        return "Batch records must be kept for the period in the table [S1]."


@pytest.fixture()
def env():
    settings = dataclasses.replace(Settings.from_env(), min_score=0.05, max_retries=2)
    store = LegalVectorStore(settings, HashingEmbedder(), client=QdrantClient(":memory:"))
    store.ensure_collection(recreate=True)
    chunks = LegalChunker(DOC).chunk([
        el(1, "PART 900\u2014RECORDS", "heading", True),
        el(1, "\u00a7 900.3 Record retention.", "heading", True),
        el(1, "(a) Each manufacturer shall keep batch records for two years and complaint files for three years."),
        el(2, "\u00a7 900.4 Labeling.", "heading", True),
        el(2, "(a) Every package shall bear the manufacturer name and a batch number."),
    ])
    store.upsert(list(chunks))
    return settings, store


def test_answered_with_valid_citations(env):
    settings, store = env
    graph = build_graph(store, FakeLLM(), settings)
    out = run_agent(graph, initial_state("How long must batch records be kept?", [], 4, 2), "t1")
    assert out["status"] == "answered"
    cited = [c for c in out["citations"] if c["cited"]]
    assert cited and cited[0]["revision_date"] == "April 1, 1996"
    assert "April 1, 1996" in out["disclaimer"]
    assert [e["node"] for e in out["trace"]][:3] == ["analyze", "retrieve", "validate"]


def test_specific_section_lookup(env):
    settings, store = env
    graph = build_graph(store, FakeLLM(), settings)
    out = run_agent(graph, initial_state("What does \u00a7 900.4 say?", [], 4, 2), "t2")
    assert out["section_ref"] == "900.4"
    assert any(h["match"] == "exact_section" and h["section"] == "900.4" for h in out["hits"])


def test_insufficient_evidence_retries_then_gives_up(env):
    settings, store = env
    graph = build_graph(store, FakeLLM(sufficient=False), settings)
    out = run_agent(graph, initial_state("How long must batch records be kept?", [], 4, 2), "t3")
    assert out["status"] == "insufficient"
    assert out["retries"] == 2
    assert [e["node"] for e in out["trace"]].count("rewrite") == 2
    assert "won't guess" in out["answer"]


def test_smalltalk_skips_retrieval(env):
    settings, store = env
    graph = build_graph(store, FakeLLM(), settings)
    out = run_agent(graph, initial_state("hello", [], 4, 2), "t4")
    assert out["status"] == "direct" and not out["citations"]
