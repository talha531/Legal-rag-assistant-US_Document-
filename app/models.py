"""Plain data containers shared across ingestion, retrieval and the agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DocumentInfo:
    document_id: str
    document_name: str
    source_file: str
    revision_date: str = "Unknown"
    title_number: str = ""
    title_name: str = ""
    page_count: int = 0
    pdf_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageElement:
    """One extracted element of a PDF page, in reading order.

    element_type: text | heading | footnote | caption | table | image | ocr
    """

    page: int
    element_type: str
    text: str
    bbox: Optional[tuple[float, float, float, float]] = None
    is_heading: bool = False
    starts_bold: bool = False
    font_size: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    text: str          # original content (stored in the payload, shown in citations)
    embed_text: str    # context-enriched text that is embedded
    payload: dict[str, Any]
