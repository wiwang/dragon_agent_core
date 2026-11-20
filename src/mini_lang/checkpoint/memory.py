"""In-memory checkpointer — keeps the full state history per thread."""

from __future__ import annotations

from collections import defaultdict

from ..state import AgentState


class InMemoryCheckpointer:
    def __init__(self) -> None:
        self._store: dict[str, list[AgentState]] = defaultdict(list)

    async def put(self, thread_id: str, state: AgentState) -> None:
        self._store[thread_id].append(state.model_copy(deep=True))

    async def get(self, thread_id: str) -> AgentState | None:
        history = self._store.get(thread_id)
        return history[-1] if history else None

    async def history(self, thread_id: str) -> list[AgentState]:
        return list(self._store.get(thread_id, []))
