"""
src/vector_store.py
====================
SINGLE ACTION: load chunks + embeddings into a persistent, local, on-disk
Qdrant collection.

Notebook origin: Cell 7 ("DISK-BASED QDRANT VECTOR DATABASE").

Run directly:
    python -m src.vector_store
"""

import gc
import json

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import config


def build_qdrant_collection(collection_name: str = config.COLLECTION_NAME,
                             batch_size: int = 128) -> dict:
    """
    Read embeddings.npy + chunks.jsonl and upsert every vector into a
    fresh local Qdrant collection stored under config.QDRANT_DIR.
    Returns the saved qdrant_config dict.
    """
    for f in (config.CHUNK_FILE, config.EMBEDDING_FILE, config.EMBED_METADATA_JSONL):
        if not f.exists():
            raise FileNotFoundError(f"❌ Required file not found: {f}")

    config.ensure_directories()

    print("=" * 75)
    print("🗄️ QDRANT — DISK BASED VECTOR DATABASE")

    embeddings = np.load(config.EMBEDDING_FILE, mmap_mode="r")
    total_embeddings, embed_dim = embeddings.shape
    print(f"Embeddings: {total_embeddings:,} x {embed_dim}")

    qdrant = QdrantClient(path=str(config.QDRANT_DIR))

    if qdrant.collection_exists(collection_name):
        print("⚠️ Existing collection found — deleting it")
        qdrant.delete_collection(collection_name=collection_name)

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
    )
    print(f"✅ Collection '{collection_name}' created (dim={embed_dim}, distance=COSINE)")

    batch, point_id = [], 0
    with open(config.CHUNK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            vector = embeddings[point_id].tolist()
            payload = {
                "chunk_id": chunk["chunk_id"],
                "document": chunk["document"],
                "page": chunk["page"],
                "text": chunk["text"],
                "word_count": chunk["word_count"],
            }
            batch.append(PointStruct(id=point_id, vector=vector, payload=payload))
            point_id += 1

            if len(batch) >= batch_size:
                qdrant.upsert(collection_name=collection_name, points=batch)
                batch.clear()
                gc.collect()
                if point_id % 512 == 0:
                    print(f"Inserted: {point_id:,} / {total_embeddings:,}")

    if batch:
        qdrant.upsert(collection_name=collection_name, points=batch)
        batch.clear()

    collection_info = qdrant.get_collection(collection_name=collection_name)
    points_count = collection_info.points_count
    if points_count != total_embeddings:
        raise RuntimeError(f"❌ Verification failed: expected {total_embeddings}, found {points_count}")

    print(f"✅ All {points_count:,} vectors stored")

    qdrant_config = {
        "qdrant_path": str(config.QDRANT_DIR),
        "collection_name": collection_name,
        "vector_dimension": int(embed_dim),
        "distance": "COSINE",
        "total_vectors": int(points_count),
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "embedding_file": str(config.EMBEDDING_FILE),
        "chunk_file": str(config.CHUNK_FILE),
        "metadata_file": str(config.EMBED_METADATA_JSONL),
    }
    config.QDRANT_CONFIG_FILE.write_text(json.dumps(qdrant_config, indent=2), encoding="utf-8")

    qdrant.close()
    print("\n✅ QDRANT VECTOR DATABASE COMPLETE")
    return qdrant_config


if __name__ == "__main__":
    build_qdrant_collection()
