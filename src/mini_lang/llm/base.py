"""LLM client protocol.

Nodes depend only on this interface, never on a concrete provider. The ``tag``
argument names the *purpose* of the call ("plan", "tool_select", "answer",
"evaluate"); the mock uses it to return deterministic responses and tests use
it to script exact outputs.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from ..state import Message


@runtime_checkable
class LLMClient(Protocol):
    async def acomplete(
        self, messages: list[Message], *, tag: str | None = None, **kwargs
    ) -> str: ...

    def astream(
        self, messages: list[Message], *, tag: str | None = None, **kwargs
    ) -> AsyncIterator[str]: ...
