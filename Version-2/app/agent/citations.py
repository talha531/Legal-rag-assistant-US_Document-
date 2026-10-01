"""Deterministic citation building + grounding checks (no LLM involved => cannot hallucinate)."""
from __future__ import annotations

import re

from app.ingestion.legal_structure import find_all_section_refs

MARKER_RE = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)*)\]")


def citation_label(hit: dict) -> str:
    """'Title 21, Part 101, \u00a7 101.9(c), p. 45'."""
    bits = []
    if hit.get("title"):
        bits.append(f"Title {hit['title']}")
    if hit.get("part"):
        bits.append(f"Part {hit['part']}")
    if hit.get("section"):
        bits.append(f"\u00a7 {hit['section']}{hit.get('subsection', '')}")
    page = hit.get("page_start") or hit.get("page")
    end = hit.get("page_end")
    if page:
        bits.append(f"pp. {page}-{end}" if end and end != page else f"p. {page}")
    return ", ".join(bits) or "Unstructured passage"


def build_context(hits: list[dict], max_chars: int, per_hit: int = 1800) -> tuple[str, list[dict]]:
    """Number the evidence [S1], [S2] ... within a character budget. Returns (context, used_hits)."""
    blocks: list[str] = []
    used: list[dict] = []
    budget = max_chars
    for hit in hits:
        text = (hit.get("text") or "")[:per_hit]
        header = (
            f"[S{len(used) + 1}] {hit.get('document_name', '')} | revision date: {hit.get('revision_date', 'Unknown')} | "
            f"{citation_label(hit)} | type: {hit.get('element_type', 'text')}"
        )
        block = f"{header}\n{text}"
        if used and len(block) > budget:
            break
        blocks.append(block)
        used.append(hit)
        budget -= len(block)
    return "\n\n".join(blocks), used


def extract_markers(answer: str) -> list[int]:
    out: list[int] = []
    for m in MARKER_RE.finditer(answer or ""):
        out += [int(x) for x in re.findall(r"S(\d+)", m.group(1))]
    return out


def sanitize_markers(answer: str, valid: set[int]) -> str:
    """Drop markers that point at sources that do not exist."""

    def repl(m: re.Match) -> str:
        ids = [int(x) for x in re.findall(r"S(\d+)", m.group(1))]
        keep = [i for i in ids if i in valid]
        return "[" + ", ".join(f"S{i}" for i in keep) + "]" if keep else ""

    return re.sub(r"[ ]{2,}", " ", MARKER_RE.sub(repl, answer or ""))


def find_unsupported_sections(answer: str, hits: list[dict]) -> list[str]:
    """Section numbers named in the answer that appear nowhere in the retrieved evidence."""
    known: set[str] = set()
    corpus = ""
    for h in hits:
        if h.get("section"):
            known.add(h["section"])
        corpus += " " + (h.get("text") or "")
    unsupported = []
    for ref in find_all_section_refs(answer):
        if ref and ref not in known and ref not in corpus and ref not in unsupported:
            unsupported.append(ref)
    return unsupported


def build_citation(hit: dict, marker: str, cited: bool) -> dict:
    return {
        "marker": marker,
        "cited": cited,
        "document_name": hit.get("document_name", ""),
        "document_id": hit.get("document_id", ""),
        "source_file": hit.get("source_file", ""),
        "revision_date": hit.get("revision_date", "Unknown"),
        "title": hit.get("title", ""),
        "chapter": hit.get("chapter", ""),
        "part": hit.get("part", ""),
        "part_name": hit.get("part_name", ""),
        "subpart": hit.get("subpart", ""),
        "section": hit.get("section", ""),
        "section_heading": hit.get("section_heading", ""),
        "subsection": hit.get("subsection", ""),
        "page": hit.get("page_start") or hit.get("page"),
        "page_end": hit.get("page_end"),
        "element_type": hit.get("element_type", "text"),
        "hierarchy_path": hit.get("hierarchy_path", ""),
        "score": round(float(hit.get("score", 0.0)), 3),
        "image_paths": hit.get("image_paths", []),
        "bbox": hit.get("bbox"),
        "label": citation_label(hit),
        "text": hit.get("text", ""),
    }


def build_citations(answer: str, used_hits: list[dict]) -> tuple[str, list[dict], list[str]]:
    """Validate markers, build citation objects. Returns (clean_answer, citations, warnings)."""
    valid = set(range(1, len(used_hits) + 1))
    warnings: list[str] = []
    cited_ids = sorted({i for i in extract_markers(answer) if i in valid})
    invalid = {i for i in extract_markers(answer) if i not in valid}
    clean = sanitize_markers(answer, valid)
    if invalid:
        warnings.append("Some citation markers pointed to non-existent sources and were removed.")
    if not cited_ids:
        warnings.append(
            "The answer did not attach source markers, so the passages below are listed as consulted, "
            "not confirmed as support. Verify before relying on it."
        )
    citations = [build_citation(used_hits[i - 1], f"S{i}", True) for i in cited_ids]
    citations += [
        build_citation(h, f"S{i}", False) for i, h in enumerate(used_hits, start=1) if i not in cited_ids
    ]
    bad = find_unsupported_sections(clean, used_hits)
    if bad:
        warnings.append(
            "The answer mentions section(s) " + ", ".join(f"\u00a7 {b}" for b in bad)
            + " that do not appear in the retrieved evidence - treat with caution."
        )
    return clean, citations, warnings


def revision_disclaimer(citations: list[dict]) -> str:
    dates = sorted({c["revision_date"] for c in citations if c.get("revision_date")})
    if not dates:
        return ""
    shown = ", ".join(d for d in dates)
    return (
        f"Source text is from the document revision dated **{shown}**. Later amendments are not reflected; "
        "verify current law in the Federal Register / eCFR. Informational research only - not legal advice."
    )


def format_citation_line(c: dict) -> str:
    return f"[{c['marker']}] {c['document_name']} - {c['label']} (revision {c['revision_date']})"
