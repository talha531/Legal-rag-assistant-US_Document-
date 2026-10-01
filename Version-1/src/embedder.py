"""
src/embedder.py
================
SINGLE ACTION: turn chunks.jsonl into a disk-backed embeddings.npy matrix
using a BGE sentence-transformers model. Only one small batch of text is
held in RAM at a time.

Notebook origin: Cell 6 ("BGE EMBEDDINGS — DISK-BASED + MEMORY EFFICIENT").

Run directly:
    python -m src.embedder
"""

import gc
import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

import config


def _count_lines(path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def embed_chunks_to_disk(model_name: str = config.EMBEDDING_MODEL_NAME) -> dict:
    """
    Encode every chunk in config.CHUNK_FILE and write the resulting
    vectors to config.EMBEDDING_FILE (a real, memory-mappable .npy file).
    Returns the metadata summary dict.
    """
    if not config.CHUNK_FILE.exists():
        raise FileNotFoundError(f"❌ {config.CHUNK_FILE} not found. Run src.chunker first.")

    config.ensure_directories()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32 if device == "cuda" else 8

    print("=" * 75)
    print("🧠 BGE EMBEDDINGS — DISK BASED")
    print(f"Model: {model_name} | Device: {device} | Batch: {batch_size}")

    embedding_model = SentenceTransformer(model_name, device=device)
    embed_dim = embedding_model.get_sentence_embedding_dimension()

    total_chunks = _count_lines(config.CHUNK_FILE)
    if total_chunks == 0:
        raise ValueError("❌ chunks.jsonl contains no chunks.")

    for f in (config.EMBEDDING_FILE, config.EMBED_METADATA_JSON, config.EMBED_METADATA_JSONL):
        if f.exists():
            f.unlink()

    embeddings_disk = np.lib.format.open_memmap(
        config.EMBEDDING_FILE, dtype=np.float32, mode="w+", shape=(total_chunks, embed_dim)
    )

    metadata_file = open(config.EMBED_METADATA_JSONL, "w", encoding="utf-8")

    def process_batch(texts, metadata, start_index):
        if not texts:
            return 0
        batch_embeddings = embedding_model.encode(
            texts, batch_size=len(texts), normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32)
        end_index = start_index + len(batch_embeddings)
        embeddings_disk[start_index:end_index] = batch_embeddings

        for meta in metadata:
            metadata_file.write(json.dumps(meta, ensure_ascii=False) + "\n")

        embeddings_disk.flush()
        metadata_file.flush()
        del batch_embeddings
        gc.collect()
        return len(texts)

    embedding_index = 0
    batch_texts, batch_metadata = [], []

    with open(config.CHUNK_FILE, "r", encoding="utf-8") as f:
        progress = tqdm(total=total_chunks, desc="Embedding chunks", unit="chunk")
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            batch_texts.append(chunk["text"])
            batch_metadata.append({
                "index": embedding_index + len(batch_texts) - 1,
                "chunk_id": chunk["chunk_id"],
                "document": chunk["document"],
                "page": chunk["page"],
                "word_count": chunk["word_count"],
            })
            if len(batch_texts) >= batch_size:
                processed = process_batch(batch_texts, batch_metadata, embedding_index)
                embedding_index += processed
                progress.update(processed)
                batch_texts.clear()
                batch_metadata.clear()

        if batch_texts:
            processed = process_batch(batch_texts, batch_metadata, embedding_index)
            embedding_index += processed
            progress.update(processed)

        progress.close()

    metadata_file.close()
    embeddings_disk.flush()
    del embeddings_disk
    gc.collect()

    metadata_summary = {
        "model": model_name,
        "device": device,
        "embedding_dimension": embed_dim,
        "total_chunks": total_chunks,
        "batch_size": batch_size,
        "normalized": True,
        "embedding_file": str(config.EMBEDDING_FILE),
        "metadata_jsonl": str(config.EMBED_METADATA_JSONL),
    }
    config.EMBED_METADATA_JSON.write_text(json.dumps(metadata_summary, indent=2), encoding="utf-8")

    # verify
    test_embeddings = np.load(config.EMBEDDING_FILE, mmap_mode="r")
    if test_embeddings.shape != (total_chunks, embed_dim):
        raise ValueError("❌ Embedding shape verification failed.")
    del test_embeddings

    print("\n✅ EMBEDDING COMPLETE:", total_chunks, "vectors of dim", embed_dim)
    return metadata_summary


if __name__ == "__main__":
    embed_chunks_to_disk()
