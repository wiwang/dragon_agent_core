"""AgentRunner — thin wrapper that turns a query into a graph execution.

Used both by the queue worker (``run``) and the SSE endpoint (``astream``).
"""

from __future__ import annotations

import uuid
from typing import AsyncIterator

from mini_lang.graph import Graph
from mini_lang.state import AgentState


class AgentRunner:
    def __init__(self, graph: Graph):
        self.graph = graph

    @staticmethod
    def new_run_id() -> str:
        return uuid.uuid4().hex

    async def run(self, query: str, thread_id: str, run_id: str | None = None) -> AgentState:
        state = AgentState(
            user_input=query, thread_id=thread_id, run_id=run_id or self.new_run_id()
        )
        return await self.graph.run(state, thread_id=thread_id)

    async def astream(
        self, query: str, thread_id: str, run_id: str | None = None
    ) -> AsyncIterator[dict]:
        state = AgentState(
            user_input=query, thread_id=thread_id, run_id=run_id or self.new_run_id()
        )
        async for event in self.graph.astream(state, thread_id=thread_id):
            yield event
