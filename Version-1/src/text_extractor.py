"""
src/text_extractor.py
======================
SINGLE ACTION: extract text from PDFs, one page at a time, straight to disk.

Notebook origin: Cell 4 ("MEMORY-EFFICIENT PDF TEXT EXTRACTION").
Keeps RAM usage low: only file paths are kept in memory, not page text.

Run directly:
    python -m src.text_extractor
"""

import json
import re
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF
from tqdm.auto import tqdm

import config
from src.pdf_loader import collect_pdfs


def clean_pdf_text(text: str) -> str:
    """Normalize line endings/whitespace extracted from a PDF page."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdfs_to_disk(pdf_paths: List[Path]) -> Dict[str, dict]:
    """
    Extract every page of every PDF to its own .txt file under
    config.TEXT_DIR, and write metadata (page index + document info)
    to config.META_DIR. Returns the DOCUMENT_INFO dict.
    """
    if not pdf_paths:
        raise RuntimeError("No PDF files given. Run src.pdf_loader first.")

    config.ensure_directories()

    page_texts: Dict[str, Dict[int, str]] = {}
    document_info: Dict[str, dict] = {}

    total_pages = total_characters = total_documents = 0

    print("=" * 75)
    print("📚 PDF TEXT EXTRACTION (disk-based, low RAM)")
    print("=" * 75)
    print(f"PDF files: {len(pdf_paths)}")

    for doc_index, pdf_path in enumerate(pdf_paths, start=1):
        print(f"\nDOCUMENT {doc_index}/{len(pdf_paths)}: {pdf_path.name}")

        try:
            pdf_doc = fitz.open(str(pdf_path))
        except Exception as e:
            print(f"❌ Could not open PDF: {e}")
            continue

        num_pages = len(pdf_doc)
        document_text_dir = config.TEXT_DIR / pdf_path.stem
        document_text_dir.mkdir(parents=True, exist_ok=True)

        page_index: Dict[int, str] = {}
        document_characters = 0

        for page_num in tqdm(range(num_pages), desc=f"Extracting {pdf_path.name}", unit="page"):
            text = clean_pdf_text(pdf_doc[page_num].get_text("text"))

            page_path = document_text_dir / f"page_{page_num + 1:06d}.txt"
            page_path.write_text(text, encoding="utf-8")

            page_index[page_num] = str(page_path)
            document_characters += len(text)

        pdf_doc.close()

        page_texts[pdf_path.name] = page_index
        document_info[pdf_path.name] = {
            "path": str(pdf_path),
            "filename": pdf_path.name,
            "pages": num_pages,
            "characters": document_characters,
            "text_dir": str(document_text_dir),
        }

        metadata_path = config.META_DIR / f"{pdf_path.stem}_metadata.json"
        metadata_path.write_text(json.dumps(document_info[pdf_path.name], indent=2), encoding="utf-8")

        total_pages += num_pages
        total_characters += document_characters
        total_documents += 1
        print(f"✅ Extracted {document_characters:,} characters -> {document_text_dir}")

    (config.META_DIR / "page_index.json").write_text(json.dumps(page_texts, indent=2), encoding="utf-8")
    (config.META_DIR / "document_info.json").write_text(json.dumps(document_info, indent=2), encoding="utf-8")

    print("\n" + "=" * 75)
    print("✅ EXTRACTION COMPLETE")
    print(f"Documents : {total_documents:,} | Pages : {total_pages:,} | Characters : {total_characters:,}")

    return document_info


if __name__ == "__main__":
    config.ensure_directories()
    pdfs = collect_pdfs()
    extract_pdfs_to_disk(pdfs)
