"""Legal-hierarchy detection for US Code of Federal Regulations style documents.

Hierarchy preserved (top -> bottom):
    Title -> Chapter -> (Subchapter) -> Part -> Subpart -> Section
          -> Subsection (a) -> Paragraph (1) -> Subparagraph (i) -> Sub-subparagraph (A)

This module is pure Python (no PDF / ML dependencies) so it is easy to unit-test.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from typing import Optional

LEVELS = [
    "title", "chapter", "subchapter", "part", "subpart", "section",
    "subsection", "paragraph", "subparagraph", "subsubparagraph",
]
STRUCTURAL_LEVELS = set(LEVELS[:6])            # title .. section
SUB_LEVELS = set(LEVELS[6:])                   # (a) (1) (i) (A)
NAMED_LEVELS = {"title", "chapter", "subchapter", "part", "subpart"}

_DASH = "[\u2014\u2013\u2012\u2015\\-]"

RE_TITLE = re.compile(rf"^TITLE\s+(\d+)\s*{_DASH}*\s*(.*)$", re.I)
RE_CHAPTER = re.compile(rf"^CHAPTER\s+([IVXLCDM]+|\d+[A-Z]?)\s*{_DASH}*\s*(.*)$", re.I)
RE_SUBCHAPTER = re.compile(rf"^SUBCHAPTER\s+([A-Z]{{1,2}})\s*{_DASH}*\s*(.*)$", re.I)
RE_PART = re.compile(rf"^PART\s+(\d+[A-Za-z]?)\s*{_DASH}*\s*(.*)$", re.I)
RE_SUBPART = re.compile(rf"^SUBPART\s+([A-Z]{{1,2}}(?:-\d+)?)\s*{_DASH}*\s*(.*)$", re.I)
RE_SECTION = re.compile(r"^(?:\u00a7|Sec\.)\s*(\d+[A-Za-z]?\.\d+[A-Za-z0-9\-]*)\s*(.*)$")
RE_PAREN = re.compile(r"^\(([A-Za-z]{1,4}|\d{1,3})\)\s*(?=\S)")
RE_TOC = re.compile(r"(\.{4,}|(?:\s\.){4,})\s*\d*\s*$")

# Section / part references inside user questions and answers
RE_SECTION_REF = re.compile(
    r"(?:\u00a7+|\bsec(?:tion)?s?\.?|\bCFR)\s*(\d+[A-Za-z]?\.\d+[A-Za-z0-9\-]*)", re.I
)
RE_PART_REF = re.compile(r"\bpart\s+(\d+[A-Za-z]?)\b", re.I)

_ROMANS = {
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
}


@dataclass
class Marker:
    level: str
    ident: str
    name: str = ""


def normalize_section(value: str) -> str:
    """'§ 101.9(c)(2)' -> '101.9'."""
    m = re.search(r"(\d+[A-Za-z]?\.\d+[A-Za-z0-9\-]*)", value or "")
    if not m:
        return ""
    sec = m.group(1)
    # drop trailing "(...)" leftovers are already excluded by the character class
    return sec.rstrip("-")


RE_BARE_SECTION = re.compile(r"(?<![\d.])(\d{2,4}\.\d{1,4}[A-Za-z]?)(?![\d.]*\d)")


def extract_section_ref(text: str) -> str:
    """Find a CFR section number: '§ 101.9', 'section 101.9', '21 CFR 101.9' or a bare '1303.13'."""
    m = RE_SECTION_REF.search(text or "")
    if m:
        return normalize_section(m.group(1))
    m = RE_BARE_SECTION.search(text or "")
    return normalize_section(m.group(1)) if m else ""


def extract_part_ref(text: str) -> str:
    m = RE_PART_REF.search(text or "")
    return m.group(1) if m else ""


def find_all_section_refs(text: str) -> list[str]:
    return [normalize_section(m.group(1)) for m in RE_SECTION_REF.finditer(text or "")]


def looks_like_structural_heading(text: str) -> bool:
    t = (text or "").strip()
    return any(rx.match(t) for rx in (RE_TITLE, RE_CHAPTER, RE_SUBCHAPTER, RE_PART, RE_SUBPART, RE_SECTION))


def _next_letter(current: str) -> str:
    if not current:
        return "a"
    if current == "z":
        return "aa"
    if len(current) == 1:
        return chr(ord(current) + 1)
    return ""


def _clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", name or "").strip(" \u2014\u2013-.")


class LegalStructureTracker:
    """Stateful tracker: feed text blocks in reading order, read the current hierarchy."""

    def __init__(self, title_hint: str = "", title_name_hint: str = ""):
        self.state: dict[str, str] = {}
        for lvl in LEVELS:
            self.state[lvl] = ""
            if lvl in NAMED_LEVELS:
                self.state[f"{lvl}_name"] = ""
        self.state["section_heading"] = ""
        self.state["title"] = title_hint
        self.state["title_name"] = title_name_hint
        self._pending: Optional[str] = None  # level waiting for its (wrapped) name line

    # ------------------------------------------------------------------ classify
    def classify(self, text: str, is_heading: bool = False) -> Optional[Marker]:
        t = (text or "").strip()
        if not t or RE_TOC.search(t):
            return None

        def gated(keyword: str) -> bool:
            return is_heading or t[: len(keyword)].isupper()

        m = RE_TITLE.match(t)
        if m and gated("TITLE") and len(t) < 200:
            return Marker("title", m.group(1), _clean_name(m.group(2)))
        m = RE_SUBCHAPTER.match(t)
        if m and gated("SUBCHAPTER") and len(t) < 250:
            return Marker("subchapter", m.group(1).upper(), _clean_name(m.group(2)))
        m = RE_CHAPTER.match(t)
        if m and gated("CHAPTER") and len(t) < 250:
            return Marker("chapter", m.group(1).upper(), _clean_name(m.group(2)))
        m = RE_SUBPART.match(t)
        if m and (is_heading or t[:7] in ("Subpart", "SUBPART")) and len(t) < 200:
            return Marker("subpart", m.group(1).upper(), _clean_name(m.group(2)))
        m = RE_PART.match(t)
        if m and gated("PART") and len(t) < 200:
            return Marker("part", m.group(1), _clean_name(m.group(2)))

        if not t.startswith("\u00a7\u00a7"):
            m = RE_SECTION.match(t)
            if m:
                rest = m.group(2).strip()
                # Run-in heading: "§ 1303.13 Adjustments of aggregate quotas. (a) The Administrator may..."
                # The heading is everything up to the first sentence end; the block itself may be long.
                head = re.split(r"(?<=[a-z\]\)])\.\s", rest, maxsplit=1)[0]
                looks_like_heading = (
                    rest[:1].isupper() and len(head) <= 160 and " of this " not in head
                )
                if is_heading or looks_like_heading or rest.startswith("["):
                    return Marker("section", m.group(1).rstrip("-"), _clean_name(head))

        m = RE_PAREN.match(t)
        if m:
            label = m.group(1)
            level = self._classify_paren(label)
            if level:
                return Marker(level, label, "")
        return None

    def _classify_paren(self, label: str) -> Optional[str]:
        if label.isdigit():
            return "paragraph"
        if label.isupper():
            return "subsubparagraph"
        if label.islower():
            expected = _next_letter(self.state.get("subsection", ""))
            if label == expected:
                return "subsection"
            if label in _ROMANS:
                return "subparagraph"
            return "subsection"
        return None

    # -------------------------------------------------------------------- update
    def update(self, text: str, is_heading: bool = False) -> Optional[str]:
        """Consume a block; returns the hierarchy level that changed (or None)."""
        marker = self.classify(text, is_heading)
        if marker is None:
            if self._pending and is_heading and len(text.strip()) <= 220:
                self._set_name(self._pending, _clean_name(text))
            self._pending = None
            return None
        self._pending = marker.level if (marker.level in NAMED_LEVELS and not marker.name) else None
        self._apply(marker)
        return marker.level

    def _set_name(self, level: str, name: str) -> None:
        self.state[f"{level}_name"] = name

    def _apply(self, m: Marker) -> None:
        idx = LEVELS.index(m.level)
        for lvl in LEVELS[idx:]:
            self.state[lvl] = ""
            if lvl in NAMED_LEVELS:
                self.state[f"{lvl}_name"] = ""
        self.state["section_heading"] = "" if idx <= LEVELS.index("section") else self.state["section_heading"]
        self.state[m.level] = m.ident
        if m.level in NAMED_LEVELS:
            self.state[f"{m.level}_name"] = m.name
        if m.level == "section":
            self.state["section_heading"] = m.name
            derived_part = m.ident.split(".")[0]
            if self.state["part"] != derived_part:
                # Part heading missed (or wrapped): trust the section number.
                self.state["part"] = derived_part
                self.state["part_name"] = ""
                self.state["subpart"] = ""
                self.state["subpart_name"] = ""

    # ------------------------------------------------------------------- outputs
    def snapshot(self) -> dict[str, str]:
        return dict(self.state)

    @staticmethod
    def subsection_ref(state: dict[str, str]) -> str:
        out = ""
        if state.get("subsection"):
            out += f"({state['subsection']})"
        if state.get("paragraph"):
            out += f"({state['paragraph']})"
        if state.get("subparagraph"):
            out += f"({state['subparagraph']})"
        if state.get("subsubparagraph"):
            out += f"({state['subsubparagraph']})"
        return out

    @classmethod
    def hierarchy_path(cls, s: dict[str, str]) -> str:
        parts: list[str] = []
        if s.get("title"):
            parts.append(f"Title {s['title']}")
        if s.get("chapter"):
            parts.append(f"Chapter {s['chapter']}")
        if s.get("subchapter"):
            parts.append(f"Subchapter {s['subchapter']}")
        if s.get("part"):
            parts.append(f"Part {s['part']}")
        if s.get("subpart"):
            parts.append(f"Subpart {s['subpart']}")
        if s.get("section"):
            parts.append(f"\u00a7 {s['section']}")
        ref = cls.subsection_ref(s)
        if ref:
            parts.append(ref)
        return " > ".join(parts)


# ---------------------------------------------------------------- document facts
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
RE_REVISED = re.compile(r"revised\s+as\s+of\s+([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", re.I)
RE_EDITION = re.compile(r"\((\d{1,2})-(\d{1,2})-(\d{2,4})\s+Edition\)", re.I)


def parse_revision_date(text: str) -> str:
    """Find 'Revised as of April 1, 1996' (or '(4-1-96 Edition)'). Returns '' if absent."""
    m = RE_REVISED.search(text or "")
    if m and m.group(1).lower() in _MONTHS:
        return f"{calendar.month_name[_MONTHS[m.group(1).lower()]]} {int(m.group(2))}, {m.group(3)}"
    m = RE_EDITION.search(text or "")
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 1900 if year >= 50 else 2000
        if 1 <= month <= 12:
            return f"{calendar.month_name[month]} {day}, {year}"
    return ""


def parse_title_info(text: str) -> tuple[str, str]:
    """Return (title_number, title_name) if found, e.g. ('21', 'Food and Drugs')."""
    m = re.search(rf"\bTitle\s+(\d{{1,2}})\s*{_DASH}\s*([A-Z][A-Za-z ,&]{{2,60}})", text or "")
    if m:
        return m.group(1), _clean_name(m.group(2))
    m = re.search(r"\b(\d{1,2})\s+CFR\b", text or "")
    if m:
        return m.group(1), ""
    return "", ""