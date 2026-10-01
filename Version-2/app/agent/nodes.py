"""LangGraph node implementations.

Flow:  analyze -> retrieve -> validate --(sufficient)--> generate -> citations
                     ^            |
                     |            +--(insufficient, retries left)--> rewrite --+
                     +---------------------------------------------------------+
                                  +--(insufficient, no retries)--> insufficient
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone

from app.agent import prompts
from app.agent.citations import build_citations, build_context, revision_disclaimer
from app.agent.state import AgentState
from app.agent.tools import make_tools
from app.config import Settings
from app.ingestion.legal_structure import extract_part_ref, extract_section_ref, normalize_section
from app.llm import LLMClient, LLMError, parse_json
from app.vectorstore import LegalVectorStore

logger = logging.getLogger(__name__)

SMALLTALK_RE = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening)|who are you|what can you do)\b", re.I)

NODE_LABELS = {
    "analyze": "Query analysis",
    "retrieve": "Retrieval",
    "validate": "Evidence validation",
    "rewrite": "Query rewrite",
    "generate": "Answer generation",
    "citations": "Citation generation",
    "insufficient": "Insufficient evidence",
    "direct": "Direct reply",
}


class AgentNodes:
    def __init__(self, store: LegalVectorStore, llm: LLMClient, settings: Settings):
        self.store, self.llm, self.s = store, llm, settings
        self.search_tool, self.section_tool = make_tools(store)

    # ------------------------------------------------------------------ utilities
    @staticmethod
    def _trace(state: AgentState, node: str, detail: str, t0: float, ok: bool = True) -> list[dict]:
        event = {
            "node": node,
            "label": NODE_LABELS.get(node, node),
            "detail": detail,
            "ok": ok,
            "ms": int((time.perf_counter() - t0) * 1000),
            "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        }
        return [*state.get("trace", []), event]

    # ----------------------------------------------------------------- 1. analyze
    def analyze(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        q = state["question"].strip()
        det_section = extract_section_ref(q)
        det_part = extract_part_ref(q)
        user_part = (state.get("user_part_filter") or "").strip()

        if SMALLTALK_RE.match(q) and not det_section:
            return {"intent": "smalltalk", "trace": self._trace(state, "analyze", "Small talk detected", t0)}

        analysis: dict = {}
        note = ""
        try:
            raw = self.llm.complete(prompts.ANALYZE_SYSTEM, prompts.analyze_user(q, state.get("chat_history", [])), json_mode=True, max_tokens=300)
            analysis = parse_json(raw)
        except LLMError as exc:
            logger.warning("analyze fallback: %s", exc)
            note = " (LLM analysis unavailable - used the raw question)"

        intent = str(analysis.get("intent") or "legal_question")
        standalone = str(analysis.get("standalone_question") or q).strip() or q
        search_query = str(analysis.get("search_query") or standalone).strip() or standalone
        section_ref = det_section or normalize_section(str(analysis.get("section_ref") or ""))
        if section_ref:
            intent = "specific_section"
        if intent == "smalltalk" and not section_ref:
            return {"intent": "smalltalk", "trace": self._trace(state, "analyze", "Small talk detected", t0)}

        filters = {}
        part = user_part or det_part
        if part:
            filters["part"] = part
        detail = f"intent={intent}; query=\"{search_query}\""
        if section_ref:
            detail += f"; section=\u00a7 {section_ref}"
        if filters:
            detail += f"; filter={filters}"
        return {
            "intent": intent if intent in {"legal_question", "specific_section", "definition"} else "legal_question",
            "standalone_question": standalone,
            "search_query": search_query,
            "section_ref": section_ref,
            "filters": filters,
            "retries": 0,
            "query_history": [search_query],
            "trace": self._trace(state, "analyze", detail + note, t0),
        }

    # ---------------------------------------------------------------- 2. retrieve
    def retrieve(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        top_k = int(state.get("top_k") or self.s.top_k)
        query = state["search_query"]
        filters = state.get("filters", {})
        part = filters.get("part", "")
        section = state.get("section_ref", "")
        merged: dict[str, dict] = {}

        def add(items: list[dict], match: str) -> None:
            for h in items:
                if h["id"] not in merged:
                    merged[h["id"]] = {**h, "match": match}

        if section:
            add(self.section_tool.invoke({"section": section, "part": part, "limit": 1}), "exact_section")
            add(self.search_tool.invoke({"query": query, "top_k": top_k, "part": part, "section": section}), "section_semantic")
        if len([h for h in merged.values()]) < top_k:
            add(self.search_tool.invoke({"query": query, "top_k": top_k, "part": part, "section": ""}), "semantic")

        hits = list(merged.values())
        exact = [h for h in hits if h["match"] == "exact_section"]
        rest = sorted((h for h in hits if h["match"] != "exact_section"), key=lambda h: h.get("rank_score", h["score"]), reverse=True)
        hits = (exact + rest)[: top_k + len(exact)]
        best = max((h["score"] for h in hits if h["match"] != "exact_section"), default=0.0)
        detail = f"{len(hits)} chunks; best similarity {best:.2f}"
        if state.get("retries"):
            detail = f"retry {state['retries']}: " + detail
        return {"hits": hits, "trace": self._trace(state, "retrieve", detail, t0, ok=bool(hits))}

    # ---------------------------------------------------------------- 3. validate
    def validate(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        hits = state.get("hits", [])
        has_exact = any(h["match"] == "exact_section" for h in hits)
        best = max((h["score"] for h in hits if h["match"] != "exact_section"), default=0.0)

        if not hits:
            v = {"sufficient": False, "confidence": 0.0, "reason": "No passages were retrieved.", "missing": "everything", "suggested_query": ""}
        elif has_exact and state.get("intent") == "specific_section":
            v = {"sufficient": True, "confidence": 0.9, "reason": "The requested section was found directly.", "missing": "", "suggested_query": ""}
        elif best < self.s.min_score and not has_exact:
            v = {"sufficient": False, "confidence": best, "reason": f"Best similarity {best:.2f} is below the threshold {self.s.min_score:.2f}.", "missing": "relevant passages", "suggested_query": ""}
        else:
            snippet = self.s.validation_snippet_chars
            evidence = "\n\n".join(
                f"[{i}] ({h.get('hierarchy_path', '')}) {(h.get('text') or '')[:snippet]}" for i, h in enumerate(hits[:6], start=1)
            )
            try:
                raw = self.llm.complete(prompts.VALIDATE_SYSTEM, prompts.validate_user(state["standalone_question"], evidence), json_mode=True, max_tokens=250)
                v = parse_json(raw)
                v["sufficient"] = bool(v.get("sufficient"))
            except LLMError as exc:
                logger.warning("validate fallback: %s", exc)
                v = {"sufficient": best >= self.s.min_score * 1.4, "confidence": best, "reason": "Validator unavailable; used similarity heuristic.", "missing": "", "suggested_query": ""}

        detail = ("Sufficient" if v["sufficient"] else "Insufficient") + f" - {v.get('reason', '')}"
        return {"validation": v, "trace": self._trace(state, "validate", detail, t0, ok=v["sufficient"])}

    # ------------------------------------------------------------------ 4. rewrite
    def rewrite(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        v = state.get("validation", {})
        history = state.get("query_history", [])
        retries = state.get("retries", 0) + 1
        try:
            raw = self.llm.complete(
                prompts.REWRITE_SYSTEM,
                prompts.rewrite_user(state["standalone_question"], history, v.get("missing", ""), v.get("suggested_query", "")),
                json_mode=True, max_tokens=150,
            )
            new_q = str(parse_json(raw).get("search_query") or "").strip()
        except LLMError as exc:
            logger.warning("rewrite fallback: %s", exc)
            new_q = ""
        if not new_q or new_q.lower() in {q.lower() for q in history}:
            new_q = (v.get("suggested_query") or f"{state['standalone_question']} requirements definition scope").strip()
        filters = dict(state.get("filters", {}))
        if retries >= 2:
            filters = {}  # broaden: drop metadata filters on later retries
        return {
            "search_query": new_q,
            "filters": filters,
            "retries": retries,
            "query_history": [*history, new_q],
            "trace": self._trace(state, "rewrite", f"retry {retries}: \"{new_q}\"" + (" (filters dropped)" if retries >= 2 else ""), t0),
        }

    # ----------------------------------------------------------------- 5. generate
    def generate(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        context, used = build_context(state["hits"], self.s.max_context_chars)
        try:
            answer = self.llm.complete(prompts.ANSWER_SYSTEM, prompts.answer_user(state["standalone_question"], context)).strip()
        except LLMError as exc:
            return {
                "answer": f"I retrieved relevant passages but the language model call failed: {exc}",
                "context_hits": used, "status": "error", "error": str(exc),
                "trace": self._trace(state, "generate", f"LLM error: {exc}", t0, ok=False),
            }
        return {"answer": answer, "context_hits": used, "trace": self._trace(state, "generate", f"{len(answer)} chars from {len(used)} sources", t0)}

    # --------------------------------------------------------------- 6. citations
    def citations(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        if state.get("status") == "error":
            return {"citations": [], "warnings": [], "disclaimer": "", "trace": state.get("trace", [])}
        clean, cites, warns = build_citations(state["answer"], state["context_hits"])
        n_cited = sum(1 for c in cites if c["cited"])
        return {
            "answer": clean,
            "citations": cites,
            "warnings": warns,
            "disclaimer": revision_disclaimer([c for c in cites if c["cited"]] or cites),
            "status": "answered",
            "trace": self._trace(state, "citations", f"{n_cited} cited / {len(cites)} consulted", t0, ok=bool(n_cited)),
        }

    # ----------------------------------------------------------- terminal branches
    def insufficient(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        v = state.get("validation", {})
        answer = prompts.INSUFFICIENT_REPLY
        if v.get("reason"):
            answer += f"\n\n*Why:* {v['reason']}"
        _, used = build_context(state.get("hits", []), self.s.max_context_chars)
        from app.agent.citations import build_citation

        cites = [build_citation(h, f"S{i}", False) for i, h in enumerate(used[:4], start=1)]
        return {
            "answer": answer, "citations": cites, "warnings": [], "status": "insufficient",
            "disclaimer": revision_disclaimer(cites),
            "trace": self._trace(state, "insufficient", f"gave up after {state.get('retries', 0)} retries", t0, ok=False),
        }

    def direct(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        return {
            "answer": prompts.SMALLTALK_REPLY, "citations": [], "warnings": [], "disclaimer": "",
            "status": "direct", "trace": self._trace(state, "direct", "no retrieval needed", t0),
        }

    # -------------------------------------------------------------------- routing
    @staticmethod
    def route_after_analyze(state: AgentState) -> str:
        return "direct" if state.get("intent") == "smalltalk" else "retrieve"

    @staticmethod
    def route_after_validate(state: AgentState) -> str:
        if state.get("validation", {}).get("sufficient"):
            return "generate"
        max_retries = state.get("max_retries", 2)
        return "rewrite" if state.get("retries", 0) < max_retries else "insufficient"
