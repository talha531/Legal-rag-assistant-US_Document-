"""End-to-end ingestion: PDF -> elements -> structure-aware chunks -> embeddings -> Qdrant."""
from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from app.config import Settings
from app.ingestion.chunker import LegalChunker
from app.ingestion.pdf_parser import PDFParser
from app.storage import Storage
from app.vectorstore import LegalVectorStore

logger = logging.getLogger(__name__)
BATCH = 64


@dataclass
class IngestReport:
    document_id: str
    document_name: str
    revision_date: str
    pages: int
    chunks: int
    by_type: dict = field(default_factory=dict)
    sections: int = 0
    images: int = 0
    seconds: float = 0.0


def ingest_pdf(
    pdf_path: str | Path,
    store: LegalVectorStore,
    settings: Settings,
    storage: Optional[Storage] = None,
    document_name: str = "",
    ocr: Optional[bool] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> IngestReport:
    t0 = time.perf_counter()
    parser = PDFParser(settings, ocr_enabled=ocr)
    info = parser.read_info(pdf_path, document_name)
    logger.info("Ingesting %s (%s pages, revision %s)", info.document_name, info.page_count, info.revision_date)

    store.ensure_collection()
    store.delete_document(info.document_id)  # re-ingesting the same file replaces it

    chunker = LegalChunker(info, settings.chunk_max_chars, settings.chunk_min_chars, settings.chunk_overlap)
    types: Counter = Counter()
    sections: set[str] = set()
    images = 0
    total = 0
    batch = []

    for chunk in chunker.chunk(parser.iter_elements(pdf_path, info, progress)):
        batch.append(chunk)
        types[chunk.payload["element_type"]] += 1
        if chunk.payload["section"]:
            sections.add(f'{chunk.payload["part"]}:{chunk.payload["section"]}')
        images += len(chunk.payload["image_paths"])
        if len(batch) >= BATCH:
            total += store.upsert(batch)
            batch = []
            if progress:
                progress(0, 0, f"Embedded and stored {total} chunks...")
    total += store.upsert(batch)

    report = IngestReport(
        info.document_id, info.document_name, info.revision_date, info.page_count,
        total, dict(types), len(sections), images, round(time.perf_counter() - t0, 1),
    )
    if storage:
        storage.upsert_document(
            {**asdict(info), "page_count": info.page_count}, chunk_count=total
        )
    logger.info("Ingest finished: %s", report)
    return report
