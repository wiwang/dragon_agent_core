"""API integration tests against the FastAPI app via in-process ASGI transport."""

import json

import httpx
import pytest

from service.app import app


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_chat_runs_inline(client):
    resp = await client.post("/chat", json={"query": "compute 2 * 21", "thread_id": "a"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] == "tool"
    assert body["score"] == 0.9
    assert body["run_id"]


async def test_chat_async_goes_through_queue_and_worker(client):
    import asyncio

    resp = await client.post("/chat/async", json={"query": "compute 5 * 5", "thread_id": "async-t"})
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    assert resp.json()["status"] == "queued"

    # Poll the store until the in-process worker finishes the run.
    for _ in range(50):
        poll = (await client.get(f"/runs/by-id/{run_id}")).json()
        if poll["status"] == "done":
            break
        await asyncio.sleep(0.05)
    assert poll["status"] == "done"
    assert poll["route"] == "tool" and poll["score"] == 0.9


async def test_run_history_endpoint(client):
    await client.post("/chat", json={"query": "explain the refund policy", "thread_id": "b"})
    resp = await client.get("/runs/b")
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] == "retrieve"
    assert body["steps"] >= 1
    assert any(m["role"] == "assistant" for m in body["messages"])


async def test_chat_stream_emits_sse_events(client):
    async with client.stream(
        "POST", "/chat/stream", json={"query": "compute 7 + 7", "thread_id": "c"}
    ) as resp:
        assert resp.status_code == 200
        events = []
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))
    kinds = [e["event"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "end"
    assert events[-1]["answer"]


async def test_missing_thread_returns_404(client):
    resp = await client.get("/runs/does-not-exist")
    assert resp.status_code == 404
