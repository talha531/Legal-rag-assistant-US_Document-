"""Structure-aware chunking.

Rules
-----
* A chunk never crosses a Title / Chapter / Part / Subpart / Section boundary.
* Inside a section, text is packed up to CHUNK_MAX_CHARS, preferring to break at
  subsection / paragraph markers, e.g. (a), (1), (i).
* Tables become their own chunks (Markdown, header row repeated when split).
* Images with OCR text or captions become their own chunks; other images are attached
  to the next chunk as `image_paths` so nothing extracted is lost.
* Footnotes become their own chunks (type "footnote") so they never interrupt a paragraph.
* Every chunk carries the full hierarchy + document metadata, and its embedding text is
  prefixed with the hierarchy path (contextual chunk headers improve retrieval a lot).
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Iterator, Optional

from app.ingestion.legal_structure import (
    STRUCTURAL_LEVELS,
    SUB_LEVELS,
    LegalStructureTracker,
)
from app.models import Chunk, DocumentInfo, PageElement

_SENTENCE_SPLIT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(\u00a7\d])")


# --------------------------------------------------------------------- helpers
def split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split one oversized paragraph on sentence boundaries with a small overlap."""
    if len(text) <= max_chars:
        return [text]
    sentences = _SENTENCE_SPLIT.split(text)
    pieces: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for sent in sentences:
        # a single monster sentence: hard split
        while len(sent) > max_chars:
            if cur:
                pieces.append(" ".join(cur))
                cur, cur_len = [], 0
            pieces.append(sent[:max_chars])
            sent = sent[max_chars - overlap:] if overlap < max_chars else sent[max_chars:]
        if cur_len + len(sent) + 1 > max_chars and cur:
            pieces.append(" ".join(cur))
            # overlap: carry trailing sentences that fit within `overlap` chars
            carry: list[str] = []
            carry_len = 0
            for prev in reversed(cur):
                if carry_len + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                carry_len += len(prev) + 1
            cur, cur_len = carry, carry_len
        cur.append(sent)
        cur_len += len(sent) + 1
    if cur:
        pieces.append(" ".join(cur))
    return [p for p in pieces if p.strip()]


def split_table_markdown(md: str, max_chars: int) -> list[str]:
    """Split a Markdown table by rows, repeating the header (and caption) in every piece."""
    if len(md) <= max_chars:
        return [md]
    lines = md.split("\n")
    # header = optional caption lines + first two table lines (header row + separator)
    first_table_line = next((i for i, ln in enumerate(lines) if ln.startswith("|")), 0)
    head = lines[: first_table_line + 2]
    body = lines[first_table_line + 2:]
    pieces: list[str] = []
    cur = list(head)
    cur_len = sum(len(x) + 1 for x in cur)
    for row in body:
        if cur_len + len(row) + 1 > max_chars and len(cur) > len(head):
            pieces.append("\n".join(cur))
            cur = list(head)
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(row)
        cur_len += len(row) + 1
    if len(cur) > len(head):
        pieces.append("\n".join(cur))
    return pieces or [md]


@dataclass
class _Part:
    text: str
    page: int
    etype: str
    ref: str
    bbox: Optional[tuple]
    is_body: bool


@dataclass
class _Buffer:
    snapshot: dict = field(default_factory=dict)
    parts: list[_Part] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(p.text) + 2 for p in self.parts)

    @property
    def has_body(self) -> bool:
        return any(p.is_body for p in self.parts)


# ---------------------------------------------------------------------- chunker
class LegalChunker:
    def __init__(self, doc: DocumentInfo, max_chars: int = 1800, min_chars: int = 350, overlap: int = 200):
        self.doc = doc
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap = overlap
        self.tracker = LegalStructureTracker(doc.title_number, doc.title_name)
        self._index = 0
        self._carry_images: list[str] = []
        self._ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---------------------------------------------------------------- public API
    def chunk(self, elements: Iterable[PageElement]) -> Iterator[Chunk]:
        buf = _Buffer()
        fn_parts: list[str] = []
        fn_page = 0
        fn_snapshot: dict = {}

        for el in elements:
            text = (el.text or "").strip()
            has_image = bool(el.extra.get("path"))
            if not text and not has_image:
                continue

            # flush accumulated footnotes when the page changes or a non-footnote arrives
            if fn_parts and (el.element_type != "footnote" or el.page != fn_page):
                yield self._make_chunk(
                    fn_snapshot, [_Part("\n".join(fn_parts), fn_page, "footnote", "", None, True)], "footnote"
                )
                fn_parts = []

            et = el.element_type
            if et == "footnote":
                if not fn_parts:
                    fn_snapshot, fn_page = self.tracker.snapshot(), el.page
                fn_parts.append(text)
                continue

            if et == "table":
                yield from self._flush(buf)
                buf = _Buffer()
                yield from self._emit_special(el, "table")
                continue

            if et == "image":
                if has_image:
                    self._carry_images.append(str(el.extra["path"]))
                if text:
                    yield from self._flush(buf)
                    buf = _Buffer()
                    yield from self._emit_special(el, "image")
                continue

            # ---- text-like elements: text, heading, caption, ocr
            changed = self.tracker.update(text, el.is_heading)
            if changed in STRUCTURAL_LEVELS:
                if buf.has_body:
                    yield from self._flush(buf)
                    buf = _Buffer()
                buf.snapshot = self.tracker.snapshot()  # heading-only buffers roll forward
            elif changed in SUB_LEVELS and buf.chars >= self.min_chars:
                yield from self._flush(buf)
                buf = _Buffer()

            if not buf.parts:
                buf.snapshot = self.tracker.snapshot()
            ref = LegalStructureTracker.subsection_ref(self.tracker.state)
            buf.parts.append(
                _Part(text, el.page, et, ref, el.bbox, is_body=(not el.is_heading))
            )

            if buf.chars >= self.max_chars:
                snapshot = buf.snapshot
                tail = buf.parts[-1].text
                yield from self._flush(buf)
                buf = _Buffer(snapshot=snapshot)
                seed = self._overlap_seed(tail)
                if seed:
                    buf.parts.append(_Part(seed, el.page, "text", ref, None, is_body=False))

        if fn_parts:
            yield self._make_chunk(
                fn_snapshot, [_Part("\n".join(fn_parts), fn_page, "footnote", "", None, True)], "footnote"
            )
        yield from self._flush(buf)

    # ------------------------------------------------------------------ internals
    def _overlap_seed(self, text: str) -> str:
        if self.overlap <= 0 or len(text) <= self.overlap:
            return ""
        tail = text[-self.overlap:]
        cut = re.search(r"(?<=[.;:])\s+", tail)
        return tail[cut.end():] if cut else ""

    def _flush(self, buf: _Buffer) -> list[Chunk]:
        if not buf.parts or not buf.has_body:
            return []
        groups: list[list[_Part]] = []
        cur: list[_Part] = []
        cur_len = 0
        for p in buf.parts:
            if len(p.text) > self.max_chars:
                if cur:
                    groups.append(cur)
                    cur, cur_len = [], 0
                for piece in split_long_text(p.text, self.max_chars, self.overlap):
                    groups.append([_Part(piece, p.page, p.etype, p.ref, p.bbox, p.is_body)])
                continue
            if cur and cur_len + len(p.text) > self.max_chars:
                groups.append(cur)
                cur, cur_len = [], 0
            cur.append(p)
            cur_len += len(p.text) + 2
        if cur:
            groups.append(cur)
        return [self._make_chunk(buf.snapshot, g, None) for g in groups]

    def _emit_special(self, el: PageElement, kind: str) -> list[Chunk]:
        snapshot = self.tracker.snapshot()
        ref = LegalStructureTracker.subsection_ref(snapshot)
        if kind == "table":
            pieces = split_table_markdown(el.text.strip(), self.max_chars)
        else:
            pieces = split_long_text(el.text.strip(), self.max_chars, self.overlap)
        out = []
        for piece in pieces:
            part = _Part(piece, el.page, kind, ref, el.bbox, True)
            out.append(self._make_chunk(snapshot, [part], kind, extra={"caption": el.extra.get("caption", "")}))
        return out

    def _make_chunk(self, snapshot: dict, parts: list[_Part], etype: Optional[str], extra: Optional[dict] = None) -> Chunk:
        text = "\n\n".join(p.text for p in parts).strip()
        pages = sorted({p.page for p in parts})
        types = sorted({p.etype for p in parts})
        if etype is None:
            etype = "ocr" if types == ["ocr"] else "text"
        first = parts[0]
        subsection_ref = first.ref or LegalStructureTracker.subsection_ref(snapshot)
        path_state = dict(snapshot)
        path = LegalStructureTracker.hierarchy_path(path_state)
        heading = snapshot.get("section_heading", "")
        images, self._carry_images = self._carry_images, []

        payload = {
            # document
            "document_id": self.doc.document_id,
            "document_name": self.doc.document_name,
            "source_file": self.doc.source_file,
            "revision_date": self.doc.revision_date,
            # legal hierarchy
            "title": snapshot.get("title", ""),
            "title_name": snapshot.get("title_name", ""),
            "chapter": snapshot.get("chapter", ""),
            "chapter_name": snapshot.get("chapter_name", ""),
            "subchapter": snapshot.get("subchapter", ""),
            "part": snapshot.get("part", ""),
            "part_name": snapshot.get("part_name", ""),
            "subpart": snapshot.get("subpart", ""),
            "subpart_name": snapshot.get("subpart_name", ""),
            "section": snapshot.get("section", ""),
            "section_heading": heading,
            "subsection": subsection_ref,
            "hierarchy_path": path,
            # location / layout
            "page": pages[0],
            "page_start": pages[0],
            "page_end": pages[-1],
            "bbox": list(first.bbox) if first.bbox else None,
            # content
            "element_type": etype,
            "element_types": types,
            "caption": (extra or {}).get("caption", ""),
            "image_paths": images,
            "text": text,
            "char_count": len(text),
            "chunk_index": self._index,
            "ingested_at": self._ingested_at,
        }
        header = " | ".join(x for x in (self.doc.document_name, path, heading) if x)
        embed_text = f"{header}\n{text}" if header else text
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.doc.document_id}:{self._index}"))
        self._index += 1
        return Chunk(chunk_id=chunk_id, text=text, embed_text=embed_text, payload=payload)
