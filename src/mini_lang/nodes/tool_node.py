"""ToolNode — selects and executes a tool.

Asks the LLM (tag="tool_select") for ``name|{json args}``, then dispatches via
the registry. The dispatch is wrapped in timeout + retry to model real,
flaky tool I/O. Tool errors are captured as data (never crash the graph).
"""

from __future__ import annotations

import json

from ..graph import Node
from ..llm.base import LLMClient
from ..reliability import RetryError, with_retry, with_timeout
from ..state import AgentState, Message
from ..tools.base import ToolRegistry, ToolResult

_SYSTEM = (
    "Select one tool to satisfy the request. Reply as 'tool_name|{\"arg\": value}'. "
    "Available tools: calculator(expression), search(query)."
)


class ToolNode(Node):
    name = "tool"

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        timeout: float = 5.0,
        max_attempts: int = 3,
    ):
        self.llm = llm
        self.registry = registry
        self.timeout = timeout
        self.max_attempts = max_attempts

    @staticmethod
    def _parse(selection: str) -> tuple[str, dict]:
        name, _, raw = selection.partition("|")
        try:
            args = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            args = {}
        return name.strip(), args

    async def __call__(self, state: AgentState) -> dict:
        prompt = [
            Message(role="system", content=_SYSTEM),
            Message(role="user", content=state.user_input),
        ]
        selection = await self.llm.acomplete(prompt, tag="tool_select")
        name, args = self._parse(selection)

        @with_retry(max_attempts=self.max_attempts, base_delay=0.05)
        @with_timeout(self.timeout)
        async def _run() -> ToolResult:
            return await self.registry.call(name, **args)

        try:
            result = await _run()
        except RetryError as exc:
            result = ToolResult(tool=name, error=f"exhausted retries: {exc}")

        call_record = {"tool": name, "args": args}
        result_record = {"tool": name, "output": result.output, "error": result.error}
        return {
            "tool_calls": [call_record],
            "tool_results": [result_record],
        }
