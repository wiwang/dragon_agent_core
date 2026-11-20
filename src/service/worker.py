"""Queue worker.

Consumes ``RunRequest`` events from the bus, executes the graph, and writes the
result to the shared conversation store. Results are correlated by ``run_id``
through the store (not an in-process future), so the worker can run in its own
process/pod and scale horizontally — clients poll ``/conversations`` or
``/runs/by-id`` for the outcome.

Run standalone (against Kafka)::

    uv run python -m service.worker
"""

from __future__ import annotations

import asyncio
import logging

from mini_lang.events.base import EventBus
from mini_lang.persistence.base import ConversationStore, RunRecord

from .runner import AgentRunner

logger = logging.getLogger("agent.worker")


async def run_worker(
    bus: EventBus,
    runner: AgentRunner,
    store: ConversationStore | None = None,
) -> None:
    """Consume run requests until the bus closes."""
    async for request in bus.consume():
        try:
            state = await runner.run(request.query, request.thread_id, request.run_id)
            if store is not None:
                await store.save_run(RunRecord.from_state(state))
            logger.info("processed run %s (route=%s)", request.run_id, state.route)
        except Exception:  # never let one bad run kill the worker
            logger.exception("run %s failed", request.run_id)


async def _main() -> None:  # pragma: no cover - standalone entrypoint
    from mini_lang.config import get_settings

    from .deps import build_conversation_store, build_eventbus, build_graph

    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    bus = build_eventbus(settings)
    runner = AgentRunner(await build_graph(settings))
    store = await build_conversation_store(settings)
    logger.info("worker started; consuming events from %s", settings.eventbus)
    await run_worker(bus, runner, store)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_main())
