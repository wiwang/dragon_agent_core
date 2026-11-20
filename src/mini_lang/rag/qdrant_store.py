"""Qdrant-backed vector store.

Honours the same VectorStore protocol as the in-memory one. Uses the
dependency-free HashingEmbedder so RAG works without an external embedding
API; swap in a real embedder by passing one to ``create``. ``qdrant-client``
is an optional dependency (``pip install .[infra]``).
"""

from __future__ import annotations

import uuid
from typing import Any

from ..config import Settings
from .embeddings import Embedder, HashingEmbedder
from .vectorstore import SEED_DOCS


class QdrantVectorStore:
    def __init__(self, client, collection: str, embedder: Embedder):
        self._client = client
        self._collection = collection
        self.embedder = embedder

    @classmethod
    async def create(
        cls, settings: Settings, embedder: Embedder | None = None, seed: bool = True
    ) -> "QdrantVectorStore":
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams

        embedder = embedder or HashingEmbedder()
        client = AsyncQdrantClient(url=settings.qdrant_url)
        collection = settings.qdrant_collection

        existing = {c.name for c in (await client.get_collections()).collections}
        if collection not in existing:
            await client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
            )
        store = cls(client, collection, embedder)
        if seed and (await client.count(collection)).count == 0:
            await store.add(SEED_DOCS)
        return store

    async def add(self, docs: list[dict[str, Any]]) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=uuid.uuid4().hex,
                vector=self.embedder.embed(doc["text"]),
                payload={"text": doc["text"], "id": doc.get("id"), "metadata": doc.get("metadata", {})},
            )
            for doc in docs
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        hits = await self._client.query_points(
            collection_name=self._collection,
            query=self.embedder.embed(query),
            limit=k,
        )
        return [
            {
                "id": h.payload.get("id"),
                "text": h.payload.get("text", ""),
                "metadata": h.payload.get("metadata", {}),
                "score": h.score,
            }
            for h in hits.points
        ]
