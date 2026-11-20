"""Kafka-backed event bus (aiokafka).

Same EventBus protocol as the in-memory bus, so the API and worker code don't
change — events just cross the network now, enabling multiple worker processes
to share the load. ``aiokafka`` is an optional dependency (``pip install .[infra]``).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import AsyncIterator

from .base import RunRequest


class KafkaEventBus:
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str = "agent-workers"):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self._producer = None
        self._consumer = None
        self._closed = False

    async def _ensure_producer(self):
        if self._producer is None:
            from aiokafka import AIOKafkaProducer

            self._producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap_servers)
            await self._producer.start()
        return self._producer

    async def publish(self, request: RunRequest) -> None:
        producer = await self._ensure_producer()
        await producer.send_and_wait(
            self.topic, json.dumps(asdict(request)).encode("utf-8")
        )

    async def consume(self) -> AsyncIterator[RunRequest]:
        from aiokafka import AIOKafkaConsumer

        self._consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            enable_auto_commit=True,
            auto_offset_reset="earliest",
        )
        await self._consumer.start()
        try:
            async for msg in self._consumer:
                if self._closed:
                    break
                yield RunRequest(**json.loads(msg.value.decode("utf-8")))
        finally:
            await self._consumer.stop()

    async def close(self) -> None:
        self._closed = True
        if self._producer is not None:
            await self._producer.stop()
