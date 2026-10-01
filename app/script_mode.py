"""Script Studio: turn a topic (or a CFR section) into a ~4-minute video script using ONLY the document.

Topic lock: if the topic names a section (e.g. "§1303.13" or just "1303.13"), the script is built from
the chunks of exactly that section - never from "similar" passages of other sections.

Safety checks:
- The prompt forbids naming agencies / acts / sections that are not written in the sources.
- After generation, every acronym and section number in the script is checked against the sources.
  Unsupported ones trigger one corrective rewrite, and any that remain are returned as warnings.
"""
from __future__ import annotations

import re
from collections import Counter

from app.agent.citations import build_citations, build_context, revision_disclaimer
from app.agent.tools import make_tools
from app.config import Settings
from app.ingestion.legal_structure import extract_section_ref
from app.llm import LLMClient, LLMError
from app.vectorstore import LegalVectorStore

WORDS_PER_MINUTE = 150

SCRIPT_SYSTEM = """You are a scriptwriter for short educational videos about regulations.
Write ONLY from the numbered SOURCES. Never add facts, numbers, dates, agencies, acts, programs or section numbers
that are not written in the sources. Do not use outside knowledge, even if you think you know the regulation.

NAMING RULE: name an agency, act, or law ONLY if its exact name appears in the sources. If the sources only say
"the Administrator", say "the Administrator". Never guess the agency, the act, or a section number.

TOPIC LOCK: every segment must be about the requested TOPIC. If a source passage is about a different section or
subject, ignore it. If the sources do not actually cover the TOPIC, reply with exactly: INSUFFICIENT_SOURCES

Structure (Markdown):
## Title
**Hook** (0:00-0:20) ...
**Segment 1 - <name>** (time range) ...
(...as many segments as needed)
**Wrap-up** (time range) ...
Rules: spoken, conversational sentences a narrator can read aloud; plain language; define legal terms briefly.
After each factual sentence add its source marker like [S1]. Add short "(On screen: ...)" cues naming the section shown.
Do not describe the sources as current law. In the wrap-up, say once that the script is based on the edition of the
regulations provided, and mention a date ONLY if that date is written in the sources.
Do not mention appeals, penalties, deadlines, or procedures unless they are written in the sources."""

# --- post-generation check: acronyms and section numbers must appear in the sources -------------------------
_TOKEN_RE = re.compile(
    r"\b\d{3,4}\.\d+(?:\([a-z0-9]+\))*"            # 1303.31, 1303.31(a)
    r"|(?:§+\s*|[Ss]ections?\s+)\d+(?:\.\d+)?"      # § 1303.31, section 306
    r"|\b[A-Z]{2,5}\b"                              # FDA, DEA, APA
)
_ALLOWED_TOKENS = {"CFR", "TXT", "PDF", "DOCX", "USC", "US", "USA", "PART", "ON"}
_MARKER_RE = re.compile(r"\[S\d+(?:[,;\s]+S\d+)*\]")


def find_unsupported(script: str, source_text: str) -> list[str]:
    """Return acronyms / section numbers used in the script that never appear in the sources."""
    body = _MARKER_RE.sub("", script)
    src = source_text
    bad: set[str] = set()
    for m in _TOKEN_RE.finditer(body):
        tok = m.group(0).strip()
        if tok in _ALLOWED_TOKENS:
            continue
        core = re.sub(r"^(?:§+\s*|[Ss]ections?\s+)", "", tok)
        core = re.sub(r"\([a-z0-9]+\)", "", core)  # ignore paragraph letters like (a)
        if core and core not in src:
            bad.add(tok)
    return sorted(bad)


# --- helpers ------------------------------------------------------------------------------------------------
_TIME_RE = re.compile(r"\(\s*\d+:\d{2}\s*[-\u2010-\u2015]\s*\d+:\d{2}\s*\)")
_ONSCREEN_RE = re.compile(r"\(On screen:[^)]*\)", re.I)
_HEADING_RE = re.compile(r"^\s*#+.*$", re.M)


def _words(text: str) -> int:
    """Count spoken words only (no markers, on-screen cues, headings, or time ranges)."""
    t = _MARKER_RE.sub("", text)
    t = _ONSCREEN_RE.sub("", t)
    t = _TIME_RE.sub("", t)
    t = _HEADING_RE.sub("", t)
    t = t.replace("**", "")
    return len(re.findall(r"\b\w+\b", t))


def _fail(msg: str) -> dict:
    return {
        "script": "", "citations": [], "warnings": [], "words": 0, "target_words": 0,
        "est_minutes": 0.0, "section": None, "disclaimer": "", "error": msg,
    }


def _dominant_section(hits: list[dict]) -> str | None:
    """If most retrieved passages belong to one section, return it (else None)."""
    counts = Counter(h.get("section") for h in hits if h.get("section"))
    if not counts:
        return None
    section, n = counts.most_common(1)[0]
    return section if n >= max(2, len(hits) // 2) else None


def _is_insufficient(script: str) -> bool:
    return script.strip().upper().startswith("INSUFFICIENT_SOURCES")


def generate_script(
    topic: str, store: LegalVectorStore, llm: LLMClient, settings: Settings,
    minutes: float = 4.0, tone: str = "clear and engaging", top_k: int = 8,
) -> dict:
    section = extract_section_ref(topic)
    extra_warnings: list[str] = []

    if not section:
        # ---- free-topic mode: several searches + similarity floor
        search, _ = make_tools(store)
        seen: dict[str, dict] = {}
        for q in (topic, f"{topic} requirements", f"{topic} definitions scope applicability"):
            for h in search.invoke({"query": q, "top_k": top_k, "part": "", "section": ""}):
                if h["id"] not in seen or h["rank_score"] > seen[h["id"]]["rank_score"]:
                    seen[h["id"]] = h
        found = sorted(seen.values(), key=lambda h: h["rank_score"], reverse=True)
        found = [h for h in found if h["score"] >= settings.min_score][:top_k]
        if not found:
            return _fail("No relevant passages were found in the document for this topic.")

        # If the passages mostly come from one section, lock onto that whole section
        # (avoids mixing "similar" passages from different sections).
        dominant = _dominant_section(found)
        if dominant:
            section = dominant
            extra_warnings.append(f"Topic matched \u00a7 {section}; the script is built from that section only.")
        else:
            hits = found
            context, used = build_context(hits, settings.max_context_chars)
            topic_line = f"TOPIC: {topic}"

    if section:
        # ---- exact section mode: use every chunk of that section, nothing else
        hits = store.get_section(section, limit=60)
        if not hits:
            return _fail(
                f"\u00a7 {section} was not found in the indexed document(s). "
                "Check the number, or ingest the PDF that contains it."
            )
        context, used = build_context(hits, settings.max_context_chars)
        if len(used) < len(hits):
            extra_warnings.append(
                f"\u00a7 {section} is long: the script covers its first {len(used)} of {len(hits)} passages."
            )
        heading = next((h.get("section_heading") for h in hits if h.get("section_heading")), "")
        topic_line = f"TOPIC: 21 CFR-style section \u00a7 {section}" + (f" - {heading}" if heading else "")
        topic_line += f"\n(Use ONLY the sources below; they are the text of \u00a7 {section}.)"

    target = int(minutes * WORDS_PER_MINUTE)
    user = (
        f"{topic_line}\nTARGET LENGTH: about {target} spoken words (~{minutes:g} minutes). TONE: {tone}.\n\n"
        f"SOURCES:\n{context}\n\nWrite the script now."
    )

    insufficient_msg = (
        "The retrieved passages do not cover this topic, so no script was written (to avoid inventing content). "
        "Try naming the section, e.g. \u00a7 1303.31."
    )

    try:
        script = llm.complete(SCRIPT_SYSTEM, user, max_tokens=2500).strip()
        if _is_insufficient(script):
            return _fail(insufficient_msg)

        # one corrective pass for length
        n = _words(script)
        if n < target * 0.75 or n > target * 1.25:
            direction = "Expand" if n < target else "Tighten"
            script = llm.complete(
                SCRIPT_SYSTEM,
                user + f"\n\nYour previous draft had {n} words. {direction} it to about {target} words, keeping every "
                       f"source marker and using only the sources.\n\nPREVIOUS DRAFT:\n{script}",
                max_tokens=2500,
            ).strip()
            if _is_insufficient(script):
                return _fail(insufficient_msg)

        # one corrective pass for names / numbers that are not in the sources
        unsupported = find_unsupported(script, context)
        if unsupported:
            script = llm.complete(
                SCRIPT_SYSTEM,
                user + "\n\nYour previous draft used names or numbers that do NOT appear in the sources: "
                       + ", ".join(unsupported)
                       + ". Rewrite the script without them (say \"the Administrator\" or describe the rule in plain "
                         "words instead), keeping every source marker and the target length.\n\n"
                         f"PREVIOUS DRAFT:\n{script}",
                max_tokens=2500,
            ).strip()
            if _is_insufficient(script):
                return _fail(insufficient_msg)
            unsupported = find_unsupported(script, context)

        n = _words(script)
    except LLMError as exc:
        return _fail(str(exc))

    if unsupported:
        extra_warnings.append(
            "Check before publishing - these names/numbers are not in the source text: " + ", ".join(unsupported)
        )

    clean, cites, warns = build_citations(script, used)
    return {
        "script": clean, "citations": cites, "warnings": extra_warnings + warns, "words": n, "target_words": target,
        "est_minutes": round(n / WORDS_PER_MINUTE, 1), "section": section,
        "disclaimer": revision_disclaimer([c for c in cites if c["cited"]] or cites),
        "error": "",
    }