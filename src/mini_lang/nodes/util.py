"""Shared helpers for nodes."""

from __future__ import annotations

import json

from ..state import AgentState


def format_context(state: AgentState) -> str:
    """Render tool results + retrieved docs into a stable text block.

    The markers (``TOOL_RESULTS:`` / ``RETRIEVED:``) are what the MockLLM keys
    off to synthesise grounded answers and grade groundedness, so both the
    answer and evaluator nodes must format context the same way.
    """
    lines: list[str] = []
    if state.tool_results:
        lines.append("TOOL_RESULTS: " + json.dumps(state.tool_results))
    if state.retrieved_docs:
        lines.append("RETRIEVED:")
        for doc in state.retrieved_docs:
            lines.append(f"- {doc['text']}")
    return "\n".join(lines)
