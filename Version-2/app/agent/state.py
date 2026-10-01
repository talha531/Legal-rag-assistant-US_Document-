from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # inputs
    question: str
    chat_history: list[dict]
    top_k: int
    max_retries: int
    user_part_filter: str
    # analysis
    intent: str
    standalone_question: str
    search_query: str
    section_ref: str
    filters: dict
    # retrieval / validation loop
    hits: list[dict]
    validation: dict
    retries: int
    query_history: list[str]
    # generation
    context_hits: list[dict]
    answer: str
    citations: list[dict]
    warnings: list[str]
    disclaimer: str
    status: str            # answered | insufficient | direct | error
    error: str
    # observability
    trace: list[dict]
