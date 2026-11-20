"""Tool protocol + registry.

A tool is a named async callable with a JSON-describable signature. The
registry is what the ToolNode consults to dispatch a planned tool call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable


@dataclass
class ToolResult:
    tool: str
    output: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str

    async def run(self, **kwargs: Any) -> Any: ...


class FnTool:
    """Wrap an async function as a Tool."""

    def __init__(self, name: str, description: str, fn: Callable[..., Awaitable[Any]]):
        self.name = name
        self.description = description
        self._fn = fn

    async def run(self, **kwargs: Any) -> Any:
        return await self._fn(**kwargs)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> "ToolRegistry":
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    async def call(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(tool=name, error=f"unknown tool '{name}'")
        try:
            output = await tool.run(**kwargs)
            return ToolResult(tool=name, output=output)
        except Exception as exc:  # surface tool errors as data, don't crash the graph
            return ToolResult(tool=name, error=str(exc))
