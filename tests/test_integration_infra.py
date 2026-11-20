"""Integration tests against real infra (Redis, Postgres, Qdrant, Kafka).

Skipped unless RUN_INFRA_TESTS=1 and `docker compose up` is running, so the
default suite stays hermetic. Run with:

    RUN_INFRA_TESTS=1 uv run pytest tests/test_integration_infra.py -q
"""

import asyncio
import os
import uuid

import pytest

from mini_lang.state import AgentState

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INFRA_TESTS") != "1", reason="set RUN_INFRA_TESTS=1 to run"
)


async def test_redis_checkpointer_roundtrip():
    from mini_lang.checkpoint.redis import RedisCheckpointer

    cp = RedisCheckpointer("redis://localhost:6379/0")
    tid = f"it-{uuid.uuid4().hex}"
    await cp.put(tid, AgentState(thread_id=tid, answer="one"))
    await cp.put(tid, AgentState(thread_id=tid, answer="two"))
    latest = await cp.get(tid)
    assert latest.answer == "two"
    assert len(await cp.history(tid)) == 2
    await cp.close()


async def test_postgres_conversation_store_roundtrip():
    from mini_lang.persistence.base import RunRecord
    from mini_lang.persistence.postgres import PostgresConversationStore

    dsn = "postgresql+asyncpg://agent:agent@localhost:5432/agent"
    store = await PostgresConversationStore.create(dsn)
    tid = f"it-{uuid.uuid4().hex}"
    rec = RunRecord(run_id=uuid.uuid4().hex, thread_id=tid, route="tool", score=0.9, answer="hi")
    await store.save_run(rec)
    history = await store.get_history(tid)
    assert len(history) == 1 and history[0].answer == "hi"
    await store.close()


async def test_qdrant_vectorstore_search():
    from mini_lang.config import Settings
    from mini_lang.rag.qdrant_store import QdrantVectorStore

    settings = Settings(qdrant_collection=f"it_{uuid.uuid4().hex[:8]}")
    store = await QdrantVectorStore.create(settings)
    docs = await store.search("refund policy", k=2)
    assert docs and any("refund" in d["text"].lower() for d in docs)


async def test_kafka_eventbus_publish_consume():
    from mini_lang.events.base import RunRequest
    from mini_lang.events.kafka import KafkaEventBus

    topic = f"it-{uuid.uuid4().hex[:8]}"
    bus = KafkaEventBus("localhost:9092", topic, group_id=f"g-{uuid.uuid4().hex[:8]}")
    received: list[RunRequest] = []

    async def consume_one():
        async for req in bus.consume():
            received.append(req)
            break

    consumer = asyncio.create_task(consume_one())
    await asyncio.sleep(2)  # let the consumer join the group
    await bus.publish(RunRequest(run_id="r1", thread_id="t1", query="hello"))
    await asyncio.wait_for(consumer, timeout=20)
    await bus.close()
    assert received and received[0].query == "hello"
