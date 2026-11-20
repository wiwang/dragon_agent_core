"""Anthropic adapter.

Active only when ``AGENT_LLM_PROVIDER=anthropic`` and an API key is set. Uses
prompt caching on the system prompt to cut cost on repeated calls. The
``anthropic`` package is an optional dependency (``pip install .[llm]``).
"""

from __future__ import annotations

from typing import AsyncIterator

from ..state import Message


class AnthropicLLM:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        from anthropic import AsyncAnthropic  # lazy: optional dependency

        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model

    @staticmethod
    def _split(messages: list[Message]) -> tuple[list[dict], str]:
        system = "\n".join(m.content for m in messages if m.role == "system")
        turns = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        if not turns:  # the API requires at least one turn
            turns = [{"role": "user", "content": system or "Continue."}]
        return turns, system

    def _system_blocks(self, system: str) -> list[dict]:
        if not system:
            return []
        # cache_control lets Anthropic reuse the system prefix across calls.
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    async def acomplete(
        self, messages: list[Message], *, tag: str | None = None, max_tokens: int = 1024, **kwargs
    ) -> str:
        turns, system = self._split(messages)
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self._system_blocks(system),
            messages=turns,
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    async def astream(
        self, messages: list[Message], *, tag: str | None = None, max_tokens: int = 1024, **kwargs
    ) -> AsyncIterator[str]:
        turns, system = self._split(messages)
        async with self._client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=self._system_blocks(system),
            messages=turns,
        ) as stream:
            async for text in stream.text_stream:
                yield text
