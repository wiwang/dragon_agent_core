"""Deterministic mock LLM.

Lets the entire agent run — and the whole test suite — execute with no API key
and no network. Behaviour is driven by the call ``tag`` plus simple keyword
heuristics on the latest user message. Pass ``responses={tag: text}`` to script
exact outputs in a test.
"""

from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator

from ..state import Message

_MATH_RE = re.compile(r"[-+*/]|\bcalculate\b|\bsum\b|\bmultiply\b", re.IGNORECASE)
_LOOKUP_RE = re.compile(r"\b(who|what|when|where|explain|define|docs?|policy)\b", re.IGNORECASE)


class MockLLM:
    def __init__(self, responses: dict[str, str] | None = None, latency: float = 0.0):
        self.responses = responses or {}
        self.latency = latency
        self.calls: list[tuple[str | None, list[Message]]] = []

    async def acomplete(
        self, messages: list[Message], *, tag: str | None = None, **kwargs
    ) -> str:
        self.calls.append((tag, messages))
        if self.latency:
            await asyncio.sleep(self.latency)
        if tag in self.responses:
            return self.responses[tag]
        return self._heuristic(tag, messages)

    async def astream(
        self, messages: list[Message], *, tag: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        text = await self.acomplete(messages, tag=tag, **kwargs)
        for token in text.split(" "):
            if self.latency:
                await asyncio.sleep(self.latency)
            yield token + " "

    # ---- deterministic heuristics ------------------------------------
    def _heuristic(self, tag: str | None, messages: list[Message]) -> str:
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        context = "\n".join(m.content for m in messages)

        if tag == "plan":
            has_math = bool(_MATH_RE.search(user))
            has_lookup = bool(_LOOKUP_RE.search(user))
            if has_math and has_lookup:
                return "both"
            if has_math:
                return "tool"
            if has_lookup:
                return "retrieve"
            return "answer"

        if tag == "tool_select":
            # Extract an arithmetic expression for the calculator, else echo.
            expr = re.search(r"[-+/*\d().\s]{3,}", user)
            if expr:
                return f'calculator|{{"expression": "{expr.group().strip()}"}}'
            return f'search|{{"query": "{user[:80]}"}}'

        if tag == "evaluate":
            # "Grounded" if the answer drew on tool results or retrieved docs.
            grounded = "TOOL_RESULTS:" in context and "[" in context
            has_docs = "RETRIEVED:" in context and "- " in context
            return "0.9" if (grounded or has_docs) else "0.5"

        # tag == "answer" (or unknown): synthesise from available context.
        bits: list[str] = []
        for line in context.splitlines():
            if line.startswith("TOOL_RESULTS:") or line.startswith("RETRIEVED:"):
                bits.append(line)
        prefix = f"Answer to '{user[:60]}': "
        return prefix + (" ".join(bits) if bits else "based on the request, here is a response.")
