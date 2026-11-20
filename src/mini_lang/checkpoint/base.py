"""Checkpointer protocol — persistence for graph state.

The graph calls ``put`` after every super-step and ``get`` to resume. The
in-memory implementation lives next door; the Redis one arrives in Phase 4.
Both honour this Protocol, so the graph never imports a concrete backend.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..state import AgentState


@runtime_checkable
class Checkpointer(Protocol):
    async def put(self, thread_id: str, state: AgentState) -> None: ...

    async def get(self, thread_id: str) -> AgentState | None: ...

    async def history(self, thread_id: str) -> list[AgentState]: ...
