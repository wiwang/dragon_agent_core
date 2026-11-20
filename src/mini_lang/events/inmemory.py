"""In-memory event bus backed by asyncio.Queue."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .base import RunRequest


class InMemoryEventBus:
    def __init__(self, maxsize: int = 0):
        self._queue: asyncio.Queue[RunRequest] = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    async def publish(self, request: RunRequest) -> None:
        await self._queue.put(request)

    async def consume(self) -> AsyncIterator[RunRequest]:
        while not self._closed:
            request = await self._queue.get()
            try:
                yield request
            finally:
                self._queue.task_done()

    async def close(self) -> None:
        self._closed = True

    @property
    def depth(self) -> int:
        return self._queue.qsize()
