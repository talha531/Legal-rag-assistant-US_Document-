"""Parser test on the synthetic sample PDF (skipped when PyMuPDF is not installed)."""
import dataclasses
from pathlib import Path

import pytest

pytest.importorskip("fitz")

from app.config import Settings  # noqa: E402
from app.ingestion.chunker import LegalChunker  # noqa: E402
from app.ingestion.pdf_parser import PDFParser  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_cfr_excerpt.pdf"


def test_sample_pdf_end_to_end(tmp_path):
    settings = dataclasses.replace(Settings.from_env(), images_dir=tmp_path)
    parser = PDFParser(settings, ocr_enabled=False)
    info = parser.read_info(SAMPLE)
    assert info.revision_date == "April 1, 1996"
    assert info.title_number == "99"
    chunks = list(LegalChunker(info).chunk(parser.iter_elements(SAMPLE, info)))
    sections = {c.payload["section"] for c in chunks}
    assert {"900.1", "900.2", "900.3", "900.4"} <= sections
    assert any(c.payload["element_type"] == "table" for c in chunks)
    assert all(c.payload["revision_date"] == "April 1, 1996" for c in chunks)
    sec4 = next(c for c in chunks if c.payload["section"] == "900.4")
    assert sec4.payload["page"] == 2 and sec4.payload["part"] == "900"
