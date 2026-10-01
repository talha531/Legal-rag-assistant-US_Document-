"""LangChain @tool wrappers around the vector store (the agent's retrieval tools)."""
from __future__ import annotations

import re

from langchain_core.tools import tool

from app.vectorstore import LegalVectorStore

_STOP = {
    "the", "and", "for", "that", "this", "with", "what", "which", "are", "does", "must", "shall",
    "how", "when", "who", "under", "any", "all", "can", "from", "into", "about", "have", "has",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9\.]+", text.lower()) if len(t) > 2 and t not in _STOP}


def rerank(query: str, hits: list[dict], top_k: int) -> list[dict]:
    """Dense score (85%) + lexical overlap (15%): helps with exact legal terms and numbers."""
    q = _tokens(query)
    for h in hits:
        body = f'{h.get("text", "")} {h.get("section_heading", "")}'.lower()
        overlap = (sum(1 for t in q if t in body) / len(q)) if q else 0.0
        h["lexical"] = round(overlap, 3)
        h["rank_score"] = 0.85 * h["score"] + 0.15 * overlap
    hits.sort(key=lambda h: h["rank_score"], reverse=True)
    return hits[:top_k]


def make_tools(store: LegalVectorStore):
    @tool("search_legal_documents")
    def search_legal_documents(
        query: str, top_k: int = 5, part: str = "", section: str = "", element_type: str = ""
    ) -> list[dict]:
        """Semantic search over the ingested legal document chunks.
        Optional metadata filters: part (e.g. "101"), section (e.g. "101.9"),
        element_type ("text", "table", "image", "footnote", "ocr")."""
        filters = {"part": part, "section": section, "element_type": element_type}
        raw = store.search(query, limit=max(top_k * 3, 12), filters=filters)
        return rerank(query, raw, top_k)

    @tool("get_legal_section")
    def get_legal_section(section: str, part: str = "", limit: int = 3) -> list[dict]:
        """Fetch the opening chunk(s) of a specific CFR section by its number, e.g. "101.9"."""
        return store.get_section(section, part=part, limit=limit)

    return search_legal_documents, get_legal_section
