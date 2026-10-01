"""CLI ingestion:  python scripts/ingest.py --pdf data/pdfs/cfr_title21.pdf [--ocr] [--recreate]

NOTE: in embedded Qdrant mode only ONE process may open the storage folder at a time.
Stop the Streamlit app before running this script (or use Qdrant via Docker, see README).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings, setup_logging  # noqa: E402
from app.embeddings import FastEmbedEmbedder  # noqa: E402
from app.ingestion.pipeline import ingest_pdf  # noqa: E402
from app.storage import Storage  # noqa: E402
from app.vectorstore import LegalVectorStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a legal PDF into Qdrant")
    ap.add_argument("--pdf", required=True, help="Path to the PDF")
    ap.add_argument("--name", default="", help="Display name for the document")
    ap.add_argument("--ocr", action="store_true", help="Enable OCR (needs Tesseract)")
    ap.add_argument("--recreate", action="store_true", help="Drop and recreate the whole collection first")
    args = ap.parse_args()

    setup_logging()
    settings = get_settings()
    embedder = FastEmbedEmbedder(settings.embedding_model)
    store = LegalVectorStore(settings, embedder)
    store.ensure_collection(recreate=args.recreate)
    storage = Storage(settings.sqlite_path)

    last = {"pct": -1}

    def progress(done: int, total: int, msg: str) -> None:
        if total:
            pct = int(done * 100 / total)
            if pct != last["pct"] and pct % 5 == 0:
                last["pct"] = pct
                print(f"  {pct:3d}%  {msg}")
        else:
            print(f"        {msg}")

    report = ingest_pdf(args.pdf, store, settings, storage, args.name, ocr=args.ocr or None, progress=progress)
    print("\nDone.")
    for k, v in report.__dict__.items():
        print(f"  {k:14s}: {v}")


if __name__ == "__main__":
    main()
