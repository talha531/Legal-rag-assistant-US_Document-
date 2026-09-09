"""
src/section_retrieval.py
=========================
SINGLE ACTION: find an *exact* CFR section (e.g. "§1308.45") in the indexed
collection — locate the real section heading, pull in adjacent chunks that
belong to the same section, and trim the text down to just that section
(stopping before the next section heading starts).

Notebook origin: Cells 10B/10C/10D ("EXACT CFR SECTION RETRIEVAL",
"SECTION-AWARE EXACT RETRIEVAL", "EXTRACT ONLY THE REQUESTED CFR SECTION")
plus the CFR-detector used by the chat cell.

Run directly:
    python -m src.section_retrieval "§386.48"
"""

import re
import sys

import config
from src.retriever import get_qdrant_client


def extract_section_refs(query: str) -> list:
    """Find every CFR-style section number mentioned in `query`."""
    patterns = [
        r'§\s*(\d+(?:\.\d+)+)',
        r'\bsection\s+(\d+(?:\.\d+)+)\b',
        r'\bsec\.?\s+(\d+(?:\.\d+)+)\b',
    ]
    refs = []
    for pattern in patterns:
        for match in re.findall(pattern, query, flags=re.IGNORECASE):
            if match not in refs:
                refs.append(match)
    return refs


def extract_section_number(query: str):
    """Return the *first* CFR section number found in `query`, or None."""
    refs = extract_section_refs(query)
    return refs[0] if refs else None


def detect_cfr_section(text: str):
    """Return a normalized '§X.Y' string if `text` names a CFR section."""
    section = extract_section_number(text)
    return f"§{section}" if section else None


def find_section_heading(text: str, section_ref: str):
    """Return the character offset where `section_ref` appears as a real heading, or None."""
    sec = re.escape(section_ref)
    patterns = [
        rf'§\s*{sec}\s+[A-Z]',
        rf'§\s*{sec}\s+\d',
        rf'\n\s*{sec}\s+[A-Z]',
        rf'\n\s*{sec}\s+\d',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.start()
    return None


def load_all_qdrant_chunks(collection_name: str = config.COLLECTION_NAME) -> list:
    """Scroll through every point in Qdrant and return their payloads as dicts."""
    qdrant = get_qdrant_client()
    all_chunks = []
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=collection_name, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            if payload.get("text"):
                all_chunks.append({
                    "chunk_id": payload.get("chunk_id"),
                    "document": payload.get("document"),
                    "page": payload.get("page"),
                    "text": payload.get("text"),
                    "word_count": payload.get("word_count"),
                })
        if offset is None:
            break
    return all_chunks


def section_aware_retrieval(query: str, max_section_chunks: int = 5) -> list:
    """Find the real heading for the section in `query`, then collect that
    section's chunks (stopping once the next section heading appears)."""
    refs = extract_section_refs(query)
    if not refs:
        return []
    section_ref = refs[0]

    all_chunks = load_all_qdrant_chunks()
    heading_matches = [
        {"index": i, "chunk": c, "heading_pos": pos}
        for i, c in enumerate(all_chunks)
        if (pos := find_section_heading(c["text"], section_ref)) is not None
    ]
    if not heading_matches:
        return []

    target_chunk = heading_matches[0]["chunk"]
    document = target_chunk["document"]

    same_document = [(i, c) for i, c in enumerate(all_chunks) if c["document"] == document]
    same_document.sort(key=lambda x: (x[1]["page"] if x[1]["page"] is not None else 0, str(x[1]["chunk_id"])))

    target_position = next(
        (pos for pos, (_, c) in enumerate(same_document) if c["chunk_id"] == target_chunk["chunk_id"]), None
    )
    if target_position is None:
        return [target_chunk]

    selected = []
    for pos in range(target_position, min(target_position + max_section_chunks, len(same_document))):
        chunk = same_document[pos][1]
        selected.append(chunk)
        if pos > target_position and re.search(r'§\s*\d+(?:\.\d+)+\s+[A-Z]', chunk["text"], flags=re.IGNORECASE):
            selected.pop()
            break

    return [
        {
            "rank": rank, "score": 1.0, "exact": True, "section_heading": rank == 1,
            "chunk_id": c["chunk_id"], "document": c["document"], "page": c["page"],
            "text": c["text"], "word_count": c["word_count"],
        }
        for rank, c in enumerate(selected, start=1)
    ]


def extract_exact_section(text: str, section_number: str):
    """Trim `text` down to just the requested section (up to the next heading)."""
    sec = re.escape(section_number)
    start_match = re.search(rf'§\s*{sec}\s+[A-Z]', text, re.IGNORECASE)
    if not start_match:
        return None

    remaining = text[start_match.start():]
    next_match = re.search(
        r'§\s*\d+(?:\.\d+)+\s+[A-Z]', remaining[len(start_match.group()):], re.IGNORECASE
    )
    if next_match:
        end = len(start_match.group()) + next_match.start()
        return remaining[:end].strip()
    return remaining.strip()


def section_exact_answer_context(query: str) -> list:
    """
    High-level entry point: given a query that names a CFR section, return
    the cleaned, exact-section-only passages ready to hand to an LLM.
    """
    section_number = extract_section_number(query)
    if not section_number:
        return []

    results = section_aware_retrieval(query, max_section_chunks=5)

    cleaned_results = []
    for result in results:
        cleaned_text = extract_exact_section(result["text"], section_number)
        if cleaned_text:
            new_result = dict(result)
            new_result["text"] = cleaned_text
            new_result["exact_section"] = True
            cleaned_results.append(new_result)
    return cleaned_results


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("Section (e.g. §386.48): ").strip()
    for r in section_exact_answer_context(q):
        print("-" * 75)
        print(f"{r['document']} | p.{r['page']} | exact={r.get('exact_section')}")
        print(r["text"][:1000])
