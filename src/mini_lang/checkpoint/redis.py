"""Redis checkpointer.

Stores the per-thread state history as a Redis list of JSON snapshots — the
same shape as the in-memory checkpointer, just durable and shared across
processes. ``redis`` is an optional dependency (``pip install .[infra]``).
"""

from __future__ import annotations

from ..state import AgentState

_PREFIX = "agent:thread:"


class RedisCheckpointer:
    def __init__(self, url: str, prefix: str = _PREFIX):
        from redis import asyncio as aioredis  # lazy: optional dependency

        self._redis = aioredis.from_url(url, decode_responses=True)
        self._prefix = prefix

    def _key(self, thread_id: str) -> str:
        return f"{self._prefix}{thread_id}"

    async def put(self, thread_id: str, state: AgentState) -> None:
        await self._redis.rpush(self._key(thread_id), state.model_dump_json())

    async def get(self, thread_id: str) -> AgentState | None:
        raw = await self._redis.lindex(self._key(thread_id), -1)
        return AgentState.model_validate_json(raw) if raw else None

    async def history(self, thread_id: str) -> list[AgentState]:
        raw = await self._redis.lrange(self._key(thread_id), 0, -1)
        return [AgentState.model_validate_json(r) for r in raw]

    async def close(self) -> None:
        await self._redis.aclose()
