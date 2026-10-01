"""Qdrant wrapper: collection management, upsert, filtered semantic search, section lookup."""
from __future__ import annotations

import logging
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import Settings
from app.embeddings import Embedder
from app.models import Chunk

logger = logging.getLogger(__name__)

KEYWORD_FIELDS = ("document_id", "title", "chapter", "part", "subpart", "section", "element_type")


class VectorStoreError(RuntimeError):
    pass


class LegalVectorStore:
    def __init__(self, settings: Settings, embedder: Embedder, client: Optional[QdrantClient] = None):
        self.s = settings
        self.embedder = embedder
        self.collection = settings.collection
        if client is None:
            if settings.qdrant_url:
                client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=60)
            else:
                settings.qdrant_path.mkdir(parents=True, exist_ok=True)
                client = QdrantClient(path=str(settings.qdrant_path))
        self.client = client

    # ------------------------------------------------------------------ collection
    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                self.collection,
                vectors_config=qm.VectorParams(size=self.embedder.dim, distance=qm.Distance.COSINE),
            )
            for field in KEYWORD_FIELDS:
                try:
                    self.client.create_payload_index(self.collection, field, qm.PayloadSchemaType.KEYWORD)
                except Exception as exc:  # local mode may ignore indexes
                    logger.debug("payload index %s skipped: %s", field, exc)
            return
        try:
            size = self.client.get_collection(self.collection).config.params.vectors.size
            if size != self.embedder.dim:
                raise VectorStoreError(
                    f"Collection '{self.collection}' has vector size {size} but the embedding model "
                    f"outputs {self.embedder.dim}. Use a new QDRANT_COLLECTION or re-ingest with recreate."
                )
        except AttributeError:
            pass

    def count(self) -> int:
        try:
            return int(self.client.count(self.collection, exact=True).count)
        except Exception:
            return 0

    # --------------------------------------------------------------------- writes
    def upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.embed_documents([c.embed_text for c in chunks])
        points = [
            qm.PointStruct(id=c.chunk_id, vector=vec, payload=c.payload) for c, vec in zip(chunks, vectors)
        ]
        self.client.upsert(self.collection, points=points, wait=True)
        return len(points)

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=qm.FilterSelector(filter=self._filter({"document_id": document_id})),
            wait=True,
        )

    # ---------------------------------------------------------------------- reads
    @staticmethod
    def _filter(filters: Optional[dict[str, Any]]) -> Optional[qm.Filter]:
        if not filters:
            return None
        must = []
        for key, value in filters.items():
            if value in (None, "", []):
                continue
            match = qm.MatchAny(any=list(value)) if isinstance(value, (list, tuple, set)) else qm.MatchValue(value=value)
            must.append(qm.FieldCondition(key=key, match=match))
        return qm.Filter(must=must) if must else None

    def search(self, query: str, limit: int = 10, filters: Optional[dict[str, Any]] = None) -> list[dict]:
        vec = self.embedder.embed_query(query)
        res = self.client.query_points(
            self.collection, query=vec, limit=limit, query_filter=self._filter(filters), with_payload=True
        )
        return [{**(p.payload or {}), "id": str(p.id), "score": float(p.score)} for p in res.points]

    def get_section(self, section: str, part: str = "", limit: int = 5, document_id: str = "") -> list[dict]:
        flt = self._filter({"section": section, "part": part, "document_id": document_id})
        records, _ = self.client.scroll(
            self.collection, scroll_filter=flt, limit=500, with_payload=True, with_vectors=False
        )
        hits = [{**(r.payload or {}), "id": str(r.id), "score": 1.0} for r in records]
        hits.sort(key=lambda h: h.get("chunk_index", 0))
        return hits[:limit]
