"""Vector store protocol + an in-memory implementation.

The Qdrant-backed store (Phase 4) honours the same Protocol, so RetrieverNode
never changes. A document is ``{"id", "text", "metadata"}``; search returns the
top-k with a ``score`` attached.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .embeddings import Embedder, HashingEmbedder, cosine


@runtime_checkable
class VectorStore(Protocol):
    async def add(self, docs: list[dict[str, Any]]) -> None: ...

    async def search(self, query: str, k: int = 3) -> list[dict[str, Any]]: ...


class InMemoryVectorStore:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or HashingEmbedder()
        self._docs: list[dict[str, Any]] = []
        self._vectors: list[list[float]] = []

    async def add(self, docs: list[dict[str, Any]]) -> None:
        for doc in docs:
            self._docs.append(doc)
            self._vectors.append(self.embedder.embed(doc["text"]))

    async def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        if not self._docs:
            return []
        q = self.embedder.embed(query)
        scored = (
            {**doc, "score": cosine(q, vec)}
            for doc, vec in zip(self._docs, self._vectors)
        )
        return sorted(scored, key=lambda d: d["score"], reverse=True)[:k]


# A tiny seed corpus so the RAG branch returns something out of the box.
SEED_DOCS = [
    {"id": "d1", "text": "Refunds are processed within 5 business days.", "metadata": {"topic": "refund"}},
    {"id": "d2", "text": "Standard shipping takes 3 to 5 business days.", "metadata": {"topic": "shipping"}},
    {"id": "d3", "text": "Our support hours are 9am to 6pm on weekdays.", "metadata": {"topic": "support"}},
    {"id": "d4", "text": "Premium members get free express shipping.", "metadata": {"topic": "shipping"}},
]


async def build_seeded_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    await store.add(SEED_DOCS)
    return store
