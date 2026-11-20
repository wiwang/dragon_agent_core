"""Conditional routing helpers.

``add_conditional_edges`` accepts any ``Callable[[AgentState], str | list[str]]``.
These classes are optional sugar for the two most common shapes:

- :class:`Router`  — wrap a function, with a name for tracing/debugging.
- :class:`RouteMap` — read a value off the state and look it up in a table.
"""

from __future__ import annotations

from typing import Callable

from .state import AgentState


class Router:
    def __init__(self, fn: Callable[[AgentState], "str | list[str]"], name: str = "router"):
        self.fn = fn
        self.name = name

    def __call__(self, state: AgentState) -> "str | list[str]":
        return self.fn(state)


class RouteMap:
    """Route on a single state attribute via a lookup table.

    >>> RouteMap("route", {"tool": "tool_node", "rag": ["retriever", "tool_node"]}, default="answer")
    """

    def __init__(self, key: str, table: dict[str, "str | list[str]"], default: "str | list[str]"):
        self.key = key
        self.table = table
        self.default = default

    def __call__(self, state: AgentState) -> "str | list[str]":
        value = getattr(state, self.key, None)
        return self.table.get(value, self.default)
