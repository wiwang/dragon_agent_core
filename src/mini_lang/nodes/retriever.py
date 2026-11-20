"""RetrieverNode — RAG. Pulls top-k docs from the vector store."""

from __future__ import annotations

from ..graph import Node
from ..rag.vectorstore import VectorStore
from ..state import AgentState


class RetrieverNode(Node):
    name = "retriever"

    def __init__(self, store: VectorStore, k: int = 3):
        self.store = store
        self.k = k

    async def __call__(self, state: AgentState) -> dict:
        docs = await self.store.search(state.user_input, k=self.k)
        return {"retrieved_docs": docs}
