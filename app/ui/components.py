"""Reusable Streamlit components: header, status pills, agent trace, citation/source cards."""
from __future__ import annotations

import html
import time
from typing import Iterator

import streamlit as st

from app.ui.styles import CSS

ICONS = {
    "analyze": "\U0001F9E0", "retrieve": "\U0001F50E", "validate": "\u2696\uFE0F", "rewrite": "\u270D\uFE0F",
    "generate": "\U0001F4DD", "citations": "\U0001F4CE", "insufficient": "\u26A0\uFE0F", "direct": "\U0001F4AC",
}


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "") -> str:
    return f'<span class="pill {kind}">{html.escape(text)}</span>'


def render_trace(trace: list[dict], expanded: bool = False) -> None:
    if not trace:
        return
    total = sum(e.get("ms", 0) for e in trace)
    with st.expander(f"\U0001F916 Agent steps ({len(trace)} steps, {total / 1000:.1f}s)", expanded=expanded):
        for e in trace:
            icon = ICONS.get(e["node"], "\u2022")
            mark = "\u2705" if e.get("ok", True) else "\u26A0\uFE0F"
            st.markdown(
                f'<div class="trace-row"><span class="n">{icon} {html.escape(e["label"])}</span>'
                f'<span style="flex:1">{mark} {html.escape(str(e["detail"]))}</span>'
                f'<span class="ms">{e.get("ms", 0)} ms</span></div>',
                unsafe_allow_html=True,
            )


def render_citations(citations: list[dict], key_prefix: str = "c") -> None:
    if not citations:
        return
    cited = [c for c in citations if c["cited"]]
    other = [c for c in citations if not c["cited"]]
    if cited:
        st.markdown("**Sources cited**")
        for c in cited:
            source_card(c, f"{key_prefix}-{c['marker']}")
    if other:
        with st.expander(f"Other passages consulted ({len(other)})"):
            for c in other:
                source_card(c, f"{key_prefix}-o-{c['marker']}")


def source_card(c: dict, key: str) -> None:
    title = f"[{c['marker']}] {c['label']}"
    if c.get("section_heading"):
        title += f" \u2014 {c['section_heading']}"
    with st.expander(title):
        chips = [
            f"Document: {c['document_name']}",
            f"Revision: {c['revision_date']}",
            f"Title {c['title']}" if c.get("title") else "",
            f"Part {c['part']}" if c.get("part") else "",
            f"Subpart {c['subpart']}" if c.get("subpart") else "",
            f"\u00a7 {c['section']}{c.get('subsection', '')}" if c.get("section") else "",
            f"Page {c['page']}" if c.get("page") else "",
            f"Type: {c['element_type']}",
            f"Score {c['score']}",
        ]
        st.markdown("".join(f'<span class="chip">{html.escape(x)}</span>' for x in chips if x), unsafe_allow_html=True)
        if c.get("hierarchy_path"):
            st.markdown(f'<div class="src-meta">{html.escape(c["hierarchy_path"])} \u00b7 file: {html.escape(c["source_file"])}</div>', unsafe_allow_html=True)
        if c["element_type"] == "table":
            st.markdown(c["text"])
        else:
            st.markdown(f'<div class="evidence">{html.escape(c["text"])}</div>', unsafe_allow_html=True)
        for p in (c.get("image_paths") or [])[:3]:
            try:
                st.image(p, caption="Extracted image on this page range", width=320)
            except Exception:
                pass
        with st.expander("Raw metadata", expanded=False):
            st.json({k: v for k, v in c.items() if k not in ("text",)})


def render_answer_extras(msg_meta: dict, key_prefix: str) -> None:
    for w in msg_meta.get("warnings", []):
        st.warning(w, icon="\u26A0\uFE0F")
    render_citations(msg_meta.get("citations", []), key_prefix)
    if msg_meta.get("disclaimer"):
        st.markdown(f'<div class="disclaimer">{msg_meta["disclaimer"]}</div>', unsafe_allow_html=True)
    render_trace(msg_meta.get("trace", []))


def typewriter(text: str) -> Iterator[str]:
    for tok in text.split(" "):
        yield tok + " "
        time.sleep(0.004)
