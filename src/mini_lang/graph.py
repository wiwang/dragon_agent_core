"""The graph engine — mini-LangGraph's core.

A :class:`Graph` is a set of named :class:`Node` objects connected by edges.
Edges are either *static* (always go to node X) or *conditional* (a router
inspects the state and picks the next node). A successor may be a *list* of
node names, in which case those nodes run concurrently (``asyncio.gather``) and
their partial updates are merged before execution converges on a join node.

Execution is exposed as an async generator (:meth:`astream`) yielding an event
per super-step; :meth:`run` simply drains it. After every step the current
state is handed to the checkpointer, so a run can be resumed.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Awaitable, Callable

from .checkpoint.base import Checkpointer
from .checkpoint.memory import InMemoryCheckpointer
from .observability.tracing import span
from .state import AgentState

# Successor specification returned by an edge / router.
Successor = "str | list[str]"
# A router maps state -> next node name, a list of names (parallel), or END.
Router = Callable[[AgentState], "str | list[str]"]


class _End:
    """Sentinel marking graph termination."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "END"


END = _End()


class GraphRecursionError(RuntimeError):
    """Raised when a run exceeds ``max_steps`` (usually an unbounded loop)."""


class Node(ABC):
    """Base class for all nodes.

    A node is an async callable that receives the current state and returns a
    *partial update* dict (merged via :meth:`AgentState.apply`). Returning
    ``None`` or ``{}`` means "no change".
    """

    name: str = "node"

    @abstractmethod
    async def __call__(self, state: AgentState) -> dict[str, Any] | None: ...


class FnNode(Node):
    """Adapt a plain async function into a Node."""

    def __init__(self, name: str, fn: Callable[[AgentState], Awaitable[dict | None]]):
        self.name = name
        self._fn = fn

    async def __call__(self, state: AgentState) -> dict[str, Any] | None:
        return await self._fn(state)


class Graph:
    def __init__(self, checkpointer: Checkpointer | None = None):
        self.nodes: dict[str, Node] = {}
        self.static_edges: dict[str, "str | list[str] | _End"] = {}
        self.conditional_edges: dict[str, Router] = {}
        self.entry: str | None = None
        self.checkpointer: Checkpointer = checkpointer or InMemoryCheckpointer()

    # ---- construction -------------------------------------------------
    def add_node(self, node: Node | Callable, name: str | None = None) -> "Graph":
        if not isinstance(node, Node):
            if name is None:
                raise ValueError("a name is required when adding a plain function")
            node = FnNode(name, node)
        elif name:
            node.name = name
        self.nodes[node.name] = node
        return self

    def add_edge(self, src: str, dst: "str | list[str] | _End") -> "Graph":
        self.static_edges[src] = dst
        return self

    def add_conditional_edges(self, src: str, router: Router) -> "Graph":
        self.conditional_edges[src] = router
        return self

    def set_entry_point(self, name: str) -> "Graph":
        self.entry = name
        return self

    # ---- execution ----------------------------------------------------
    def _successor(self, name: str, state: AgentState) -> "str | list[str] | _End":
        if name in self.conditional_edges:
            return self.conditional_edges[name](state)
        if name in self.static_edges:
            return self.static_edges[name]
        return END

    async def _invoke(self, node: Node, state: AgentState) -> dict[str, Any] | None:
        with span(f"node.{node.name}", node=node.name, thread_id=state.thread_id):
            return await node(state)

    async def astream(
        self,
        state: AgentState,
        thread_id: str | None = None,
        max_steps: int = 50,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield one event per super-step: {'event', 'node', 'state'}."""
        if self.entry is None:
            raise ValueError("graph has no entry point")
        thread_id = thread_id or state.thread_id
        current: "str | list[str] | _End" = self.entry
        steps = 0

        yield {"event": "start", "node": None, "state": state}

        while not isinstance(current, _End):
            steps += 1
            if steps > max_steps:
                raise GraphRecursionError(f"exceeded max_steps={max_steps}")

            if isinstance(current, list):
                # Parallel fan-out: run every branch on the same input state.
                updates = await asyncio.gather(
                    *(self._invoke(self.nodes[n], state) for n in current)
                )
                for update in updates:
                    state = state.apply(update)
                await self.checkpointer.put(thread_id, state)
                yield {"event": "parallel", "node": list(current), "state": state}
                # Join: every branch must converge on the same successor.
                succs = {self._render(self._successor(n, state)) for n in current}
                if len(succs) != 1:
                    raise GraphRecursionError(
                        f"parallel branches {current} diverge to {succs}; they must join"
                    )
                current = self._successor(current[0], state)
            else:
                node = self.nodes[current]
                update = await self._invoke(node, state)
                state = state.apply(update)
                await self.checkpointer.put(thread_id, state)
                yield {"event": "node", "node": current, "state": state}
                current = self._successor(current, state)

        yield {"event": "end", "node": None, "state": state}

    async def run(
        self, state: AgentState, thread_id: str | None = None, max_steps: int = 50
    ) -> AgentState:
        final = state
        async for event in self.astream(state, thread_id, max_steps):
            final = event["state"]
        return final

    async def resume(self, thread_id: str) -> AgentState | None:
        """Return the last checkpointed state for a thread (or None)."""
        return await self.checkpointer.get(thread_id)

    @staticmethod
    def _render(succ: "str | list[str] | _End") -> str:
        if isinstance(succ, _End):
            return "END"
        if isinstance(succ, list):
            return "|".join(sorted(succ))
        return succ
