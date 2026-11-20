"""Conversation storage protocol.

Distinct from the checkpointer: the checkpointer holds *graph execution* state
(for resume); the conversation store holds the durable *user-facing* record —
turns and run summaries — that a product surface would query. In-memory and
PostgreSQL implementations both honour this Protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..state import AgentState


@dataclass
class RunRecord:
    run_id: str
    thread_id: str
    route: str | None
    score: float | None
    answer: str | None
    messages: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_state(cls, state: AgentState) -> "RunRecord":
        return cls(
            run_id=state.run_id or "",
            thread_id=state.thread_id,
            route=state.route,
            score=state.scores.get("quality"),
            answer=state.answer,
            messages=[m.model_dump() for m in state.messages],
        )


@runtime_checkable
class ConversationStore(Protocol):
    async def save_run(self, record: RunRecord) -> None: ...

    async def get_history(self, thread_id: str) -> list[RunRecord]: ...

    async def get_run(self, run_id: str) -> "RunRecord | None": ...

    async def close(self) -> None: ...
