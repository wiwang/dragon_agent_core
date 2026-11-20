"""Conversation store (in-memory) + the /conversations endpoint."""

import httpx
import pytest

from mini_lang.persistence.base import RunRecord
from mini_lang.persistence.memory import InMemoryConversationStore
from mini_lang.state import AgentState
from service.app import app


async def test_in_memory_store_roundtrip():
    store = InMemoryConversationStore()
    state = AgentState(
        thread_id="t", run_id="r1", route="tool", answer="42", scores={"quality": 0.9}
    )
    await store.save_run(RunRecord.from_state(state))
    history = await store.get_history("t")
    assert len(history) == 1
    assert history[0].route == "tool" and history[0].score == 0.9


async def test_conversations_endpoint_records_runs():
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            await c.post("/chat", json={"query": "compute 2 * 21", "thread_id": "conv"})
            resp = await c.get("/conversations/conv")
            assert resp.status_code == 200
            runs = resp.json()["runs"]
            assert len(runs) == 1 and runs[0]["route"] == "tool"
