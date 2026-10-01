"""
src/chunker.py
===============
SINGLE ACTION: turn the per-page text files into overlapping word chunks,
written straight to a single chunks.jsonl file on disk.

Notebook origin: Cell 5 ("DISK-BASED CHUNKING").

Run directly:
    python -m src.chunker
"""

import gc
import json
import re
from pathlib import Path

from tqdm.auto import tqdm

import config


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_page_index() -> dict:
    page_index_path = config.META_DIR / "page_index.json"
    if not page_index_path.exists():
        raise RuntimeError(
            f"❌ {page_index_path} not found. Run src.text_extractor first."
        )
    return json.loads(page_index_path.read_text(encoding="utf-8"))


def chunk_pages_to_disk(page_texts: dict = None,
                         chunk_size: int = config.CHUNK_SIZE,
                         chunk_overlap: int = config.CHUNK_OVERLAP) -> dict:
    """
    Read page text from disk, build overlapping word chunks, and write
    each chunk immediately to config.CHUNK_FILE (chunks.jsonl).
    Returns the chunk metadata summary dict.
    """
    if page_texts is None:
        page_texts = load_page_index()

    if not page_texts:
        raise RuntimeError("❌ No page text available. Run src.text_extractor first.")

    step_size = chunk_size - chunk_overlap
    if step_size <= 0:
        raise ValueError("❌ CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    config.ensure_directories()
    if config.CHUNK_FILE.exists():
        config.CHUNK_FILE.unlink()

    chunk_id = total_words = total_chunks = total_pages = total_empty_pages = 0
    document_statistics = {}

    print("=" * 75)
    print("📚 DISK-BASED TEXT CHUNKING")
    print(f"Chunk size: {chunk_size} words | Overlap: {chunk_overlap} words")

    with open(config.CHUNK_FILE, "w", encoding="utf-8") as chunk_writer:
        for doc_index, (document_name, pages) in enumerate(page_texts.items(), start=1):
            print(f"\nDOCUMENT {doc_index}/{len(page_texts)}: {document_name}")

            document_chunk_count = document_word_count = 0

            for page_num, page_path_string in tqdm(pages.items(), desc="Chunking pages", unit="page"):
                page_path = Path(page_path_string)
                total_pages += 1

                if not page_path.exists():
                    print(f"⚠️ Missing page file: {page_path}")
                    continue

                page_text = clean_text(page_path.read_text(encoding="utf-8"))
                if not page_text:
                    total_empty_pages += 1
                    continue

                words = page_text.split()
                word_count = len(words)
                if word_count == 0:
                    total_empty_pages += 1
                    continue

                start = 0
                page_num_int = int(page_num)
                while start < word_count:
                    end = min(start + chunk_size, word_count)
                    chunk_words = words[start:end]
                    chunk_text = " ".join(chunk_words).strip()

                    if chunk_text:
                        chunk = {
                            "chunk_id": chunk_id,
                            "document": document_name,
                            "page": page_num_int + 1,
                            "text": chunk_text,
                            "word_count": len(chunk_words),
                        }
                        chunk_writer.write(json.dumps(chunk, ensure_ascii=False) + "\n")

                        chunk_id += 1
                        total_chunks += 1
                        document_chunk_count += 1
                        total_words += len(chunk_words)
                        document_word_count += len(chunk_words)

                    if end >= word_count:
                        break
                    start += step_size

                if page_num_int % 100 == 0:
                    gc.collect()

            document_statistics[document_name] = {
                "pages": len(pages),
                "chunks": document_chunk_count,
                "words": document_word_count,
            }
            print(f"Chunks: {document_chunk_count:,} | Words: {document_word_count:,}")
            gc.collect()

    average_words = total_words / total_chunks if total_chunks > 0 else 0
    chunk_metadata = {
        "chunk_file": str(config.CHUNK_FILE),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "step_size": step_size,
        "documents": len(page_texts),
        "pages": total_pages,
        "empty_pages": total_empty_pages,
        "chunks": total_chunks,
        "total_words": total_words,
        "average_words_per_chunk": average_words,
        "documents_statistics": document_statistics,
    }
    config.CHUNK_META_FILE.write_text(json.dumps(chunk_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 75)
    print("✅ CHUNKING COMPLETE")
    print(f"Chunks: {total_chunks:,} | Avg words/chunk: {average_words:.1f}")

    return chunk_metadata


if __name__ == "__main__":
    chunk_pages_to_disk()
