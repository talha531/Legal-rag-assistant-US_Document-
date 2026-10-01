"""
src/retriever.py
=================
SINGLE ACTION: plain semantic (vector) search — embed a query with BGE,
search Qdrant, return the top-K matching chunks.

Notebook origin: Cell 8 ("SEMANTIC RETRIEVAL FROM DISK-BASED QDRANT").

This module lazily loads the embedding model and the Qdrant client once,
then reuses them (get_embedding_model() / get_qdrant_client()), so other
modules (self_rag_answer, section_retrieval, chat) can import
semantic_search() without re-loading the model every call.

Run directly:
    python -m src.retriever "What authority does the Secretary have?"
"""

import sys

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

import config

_embedding_model = None
_qdrant_client = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
    return _embedding_model


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        if not config.QDRANT_DIR.exists():
            raise FileNotFoundError(f"❌ Qdrant database not found: {config.QDRANT_DIR}")
        _qdrant_client = QdrantClient(path=str(config.QDRANT_DIR))
    return _qdrant_client


def semantic_search(query: str, top_k: int = config.TOP_K_DEFAULT,
                     collection_name: str = config.COLLECTION_NAME) -> list:
    """Embed `query`, search Qdrant, return a list of result dicts."""
    query = str(query).strip()
    if not query:
        return []

    model = get_embedding_model()
    qdrant = get_qdrant_client()

    query_embedding = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    query_embedding = query_embedding.astype(np.float32)

    response = qdrant.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=top_k,
        with_payload=True,
    )

    results = []
    for rank, point in enumerate(response.points, start=1):
        payload = point.payload or {}
        results.append({
            "rank": rank,
            "score": float(point.score),
            "chunk_id": payload.get("chunk_id"),
            "document": payload.get("document"),
            "page": payload.get("page"),
            "text": payload.get("text"),
            "word_count": payload.get("word_count"),
        })
    return results


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("Question: ").strip()
    for r in semantic_search(q, top_k=config.TOP_K_DEFAULT):
        print("-" * 75)
        print(f"Rank {r['rank']} | Score {r['score']:.4f} | {r['document']} p.{r['page']}")
        print(r["text"][:400])
