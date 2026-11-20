"""AgentState — the single value that flows through the graph.

Mirrors LangGraph's "state" channel concept: nodes don't mutate the state in
place, they return a *partial update* (a dict) that the graph merges in via
per-channel reducers (see :meth:`AgentState.apply`).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Channels that accumulate (list-extend) rather than overwrite.
_ADDITIVE_LIST = {"messages"}
# Channels that deep-merge (dict-update) rather than overwrite.
_MERGE_DICT = {"scratchpad", "scores", "metadata"}


class Message(BaseModel):
    """A single conversation message."""

    role: str  # system | user | assistant | tool
    content: str
    name: str | None = None


class AgentState(BaseModel):
    """Working memory for one agent run.

    A node returns a partial dict; ``apply`` folds it in using the reducer for
    each channel. Everything is JSON-serialisable so it can be checkpointed.
    """

    thread_id: str = "default"
    run_id: str | None = None

    user_input: str = ""
    messages: list[Message] = Field(default_factory=list)

    # Planner output.
    plan: str | None = None
    route: str | None = None  # which branch(es) the router chose

    # Tool calling.
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)

    # RAG.
    retrieved_docs: list[dict[str, Any]] = Field(default_factory=list)

    # Final answer + evaluation.
    answer: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)

    # Loop control + free-form scratch.
    iteration: int = 0
    scratchpad: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def apply(self, update: dict[str, Any] | None) -> "AgentState":
        """Return a new state with ``update`` merged in (channel reducers applied)."""
        if not update:
            return self
        new = self.model_copy(deep=True)
        for key, value in update.items():
            if key in _ADDITIVE_LIST:
                items = value if isinstance(value, list) else [value]
                if key == "messages":
                    items = [m if isinstance(m, Message) else Message.model_validate(m) for m in items]
                current = list(getattr(new, key))
                current.extend(items)
                setattr(new, key, current)
            elif key in _MERGE_DICT and isinstance(value, dict):
                merged = {**getattr(new, key), **value}
                setattr(new, key, merged)
            else:
                setattr(new, key, value)
        return new

    def add_message(self, role: str, content: str, name: str | None = None) -> "AgentState":
        return self.apply({"messages": [Message(role=role, content=content, name=name)]})
