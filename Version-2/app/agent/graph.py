"""LangGraph assembly + streaming helpers."""
from __future__ import annotations

from typing import Iterator

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import AgentNodes
from app.agent.state import AgentState
from app.config import Settings
from app.llm import LLMClient
from app.vectorstore import LegalVectorStore


def build_graph(store: LegalVectorStore, llm: LLMClient, settings: Settings, checkpointer=None):
    n = AgentNodes(store, llm, settings)
    g = StateGraph(AgentState)
    g.add_node("analyze", n.analyze)
    g.add_node("retrieve", n.retrieve)
    g.add_node("validate", n.validate)
    g.add_node("rewrite", n.rewrite)
    g.add_node("generate", n.generate)
    g.add_node("citations", n.citations)
    g.add_node("insufficient", n.insufficient)
    g.add_node("direct", n.direct)

    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", n.route_after_analyze, {"retrieve": "retrieve", "direct": "direct"})
    g.add_edge("retrieve", "validate")
    g.add_conditional_edges(
        "validate", n.route_after_validate,
        {"generate": "generate", "rewrite": "rewrite", "insufficient": "insufficient"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "citations")
    g.add_edge("citations", END)
    g.add_edge("insufficient", END)
    g.add_edge("direct", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def initial_state(question: str, history: list[dict], top_k: int, max_retries: int, part_filter: str = "") -> dict:
    """Every per-turn field is reset here, because the checkpointer keeps state per thread."""
    return {
        "question": question,
        "chat_history": history,
        "top_k": top_k,
        "max_retries": max_retries,
        "user_part_filter": part_filter,
        "retries": 0,
        "trace": [],
        "hits": [],
        "citations": [],
        "warnings": [],
        "disclaimer": "",
        "answer": "",
        "status": "",
        "error": "",
        "validation": {},
        "context_hits": [],
        "query_history": [],
        "section_ref": "",
        "filters": {},
    }


def stream_agent(graph, state: dict, thread_id: str) -> Iterator[tuple[str, dict]]:
    """Yield (node_name, state_update) as each node finishes (used for live status in the UI)."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    for event in graph.stream(state, config=config, stream_mode="updates"):
        for node, update in event.items():
            yield node, (update or {})


def run_agent(graph, state: dict, thread_id: str) -> dict:
    final = dict(state)
    for _, update in stream_agent(graph, state, thread_id):
        final.update(update)
    return final
