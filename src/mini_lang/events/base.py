"""Event bus protocol + message types.

A ``RunRequest`` is enqueued by the API; a worker consumes it, executes the
graph, and produces a ``RunResult``. The in-memory bus (asyncio.Queue) and the
Kafka bus (Phase 4) both honour this Protocol, so the worker code is identical
whether events stay in-process or cross the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable


@dataclass
class RunRequest:
    run_id: str
    thread_id: str
    query: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    run_id: str
    thread_id: str
    answer: str | None
    score: float | None
    route: str | None
    error: str | None = None


@runtime_checkable
class EventBus(Protocol):
    async def publish(self, request: RunRequest) -> None: ...

    def consume(self) -> AsyncIterator[RunRequest]: ...

    async def close(self) -> None: ...
