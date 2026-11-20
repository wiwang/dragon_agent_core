"""FastAPI service.

Endpoints:
- ``POST /chat``          run the graph inline and return the result (reliable everywhere)
- ``POST /chat/async``    enqueue a run on the bus; a worker processes it (scales out)
- ``POST /chat/stream``   stream node-by-node events as SSE
- ``GET  /runs/{thread}`` inspect the checkpointed state history for a thread
- ``GET  /runs/by-id/{id}`` poll for an async run's result (from the store)
- ``GET  /conversations/{thread}`` durable run history from the conversation store
- ``GET  /healthz``       liveness + queue depth

``/chat`` executes synchronously so request/response is correct in any topology.
``/chat/async`` exercises the queue→worker path; the worker (in-process for the
memory bus, or a separate Deployment for Kafka) writes results to the shared
store, which the client polls — this is what scales horizontally in Kubernetes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from mini_lang.config import get_settings
from mini_lang.events.base import RunRequest
from mini_lang.persistence.base import RunRecord

from .deps import build_conversation_store, build_eventbus, build_graph
from .runner import AgentRunner
from .worker import run_worker

logger = logging.getLogger("agent.api")


class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    run_id: str
    thread_id: str
    answer: str | None
    score: float | None
    route: str | None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.tracing_enabled:
        from mini_lang.observability.tracing import instrument_fastapi, setup_tracing

        setup_tracing(service_name="mini-lang-api", endpoint=settings.otlp_endpoint)
        instrument_fastapi(app)
    graph = await build_graph(settings)
    runner = AgentRunner(graph)
    bus = build_eventbus(settings)
    store = await build_conversation_store(settings)

    workers = [
        asyncio.create_task(run_worker(bus, runner, store))
        for _ in range(settings.api_inproc_workers)
    ]

    app.state.settings = settings
    app.state.graph = graph
    app.state.runner = runner
    app.state.bus = bus
    app.state.store = store
    logger.info(
        "service ready (llm=%s, bus=%s, inproc_workers=%d)",
        settings.llm_provider,
        settings.eventbus,
        settings.api_inproc_workers,
    )
    try:
        yield
    finally:
        await bus.close()
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await store.close()


app = FastAPI(title="mini-lang agent", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    bus = app.state.bus
    return {"status": "ok", "queue_depth": getattr(bus, "depth", None)}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Run synchronously and return the result. Correct in any topology."""
    runner: AgentRunner = app.state.runner
    run_id = runner.new_run_id()
    state = await runner.run(req.query, req.thread_id, run_id)
    await app.state.store.save_run(RunRecord.from_state(state))
    return ChatResponse(
        run_id=run_id,
        thread_id=req.thread_id,
        answer=state.answer,
        score=state.scores.get("quality"),
        route=state.route,
    )


@app.post("/chat/async")
async def chat_async(req: ChatRequest):
    """Enqueue a run on the bus; a worker processes it. Poll /runs/by-id/{id}."""
    runner: AgentRunner = app.state.runner
    run_id = runner.new_run_id()
    await app.state.bus.publish(
        RunRequest(run_id=run_id, thread_id=req.thread_id, query=req.query)
    )
    return {"run_id": run_id, "thread_id": req.thread_id, "status": "queued"}


@app.get("/runs/by-id/{run_id}")
async def run_by_id(run_id: str):
    """Poll for an async run's result from the conversation store."""
    record = await app.state.store.get_run(run_id)
    if record is None:
        return {"run_id": run_id, "status": "pending"}
    return {
        "run_id": run_id,
        "status": "done",
        "thread_id": record.thread_id,
        "route": record.route,
        "score": record.score,
        "answer": record.answer,
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    runner: AgentRunner = app.state.runner

    async def event_gen():
        async for event in runner.astream(req.query, req.thread_id):
            state = event["state"]
            node = event["node"]
            payload = {
                "event": event["event"],
                "node": node if isinstance(node, (str, type(None))) else list(node),
                "route": state.route,
                "answer": state.answer,
                "score": state.scores.get("quality"),
                "iteration": state.iteration,
            }
            yield {"event": event["event"], "data": json.dumps(payload)}

    return EventSourceResponse(event_gen())


@app.get("/runs/{thread_id}")
async def get_run(thread_id: str):
    graph = app.state.graph
    history = await graph.checkpointer.history(thread_id)
    if not history:
        raise HTTPException(status_code=404, detail="no such thread")
    latest = history[-1]
    return {
        "thread_id": thread_id,
        "steps": len(history),
        "answer": latest.answer,
        "score": latest.scores.get("quality"),
        "route": latest.route,
        "messages": [m.model_dump() for m in latest.messages],
    }


@app.get("/conversations/{thread_id}")
async def get_conversation(thread_id: str):
    """Durable run history from the conversation store (in-memory or Postgres)."""
    records = await app.state.store.get_history(thread_id)
    return {
        "thread_id": thread_id,
        "runs": [
            {"run_id": r.run_id, "route": r.route, "score": r.score, "answer": r.answer}
            for r in records
        ],
    }
