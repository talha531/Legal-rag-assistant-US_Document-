"""
src/pdf_loader.py
==================
SINGLE ACTION: find every valid PDF file to process.

Notebook origin: Cell 3A/3B ("CHOOSE MULTIPLE" / "CONFIRM FINALIZE") which used
Colab's files.upload() or a Jupyter ipywidgets FileUpload button. VS Code has
no upload widget, so this module simply scans a local folder
(config.INPUT_DIR) — put your PDFs there, or pass any folder/file path.

Run directly:
    python -m src.pdf_loader
    python -m src.pdf_loader --input /path/to/pdfs
"""

import argparse
from pathlib import Path
from typing import List

import config


def collect_pdfs(input_path: Path = config.INPUT_DIR) -> List[Path]:
    """
    Return every valid, de-duplicated .pdf file found under input_path.
    input_path may be a single PDF file or a folder containing PDFs.
    """
    input_path = Path(input_path).resolve()

    if input_path.is_file():
        candidates = [input_path] if input_path.suffix.lower() == ".pdf" else []
    elif input_path.is_dir():
        candidates = sorted(input_path.glob("*.pdf"))
    else:
        candidates = []

    valid_books: List[Path] = []
    seen = set()

    for pdf in candidates:
        pdf = pdf.resolve()
        if pdf.suffix.lower() == ".pdf" and pdf.exists() and pdf not in seen:
            seen.add(pdf)
            valid_books.append(pdf)

    return valid_books


def print_summary(pdfs: List[Path]) -> None:
    print("=" * 70)
    print("📚 SELECTED PDF DOCUMENTS")
    print("=" * 70)

    if not pdfs:
        print("❌ No valid PDF files found.")
        print(f"   Put PDF files in: {config.INPUT_DIR}")
        return

    for i, pdf in enumerate(pdfs, start=1):
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"[{i}] {pdf.name}  ({size_mb:.2f} MB)")

    print("-" * 70)
    print(f"✅ Total PDFs selected: {len(pdfs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover PDF files to process.")
    parser.add_argument("--input", type=str, default=str(config.INPUT_DIR),
                         help="Folder (or single .pdf file) to scan.")
    args = parser.parse_args()

    config.ensure_directories()
    books = collect_pdfs(Path(args.input))
    print_summary(books)
