"""Tiny CLI demo: drive the agent on the mock LLM and print the node trace.

    uv run python -m mini_lang.demo "explain the refund policy and add 2 + 2"
"""

from __future__ import annotations

import asyncio
import sys

from .agent import build_default_agent
from .state import AgentState


async def main(query: str) -> None:
    graph = await build_default_agent()
    state = AgentState(user_input=query, thread_id="demo")
    print(f"\n=== query: {query!r} ===")
    async for event in graph.astream(state):
        node = event["node"]
        if event["event"] in ("node", "parallel") and node:
            label = node if isinstance(node, str) else " ‖ ".join(node)
            s = event["state"]
            print(f"  [{label}] route={s.route} docs={len(s.retrieved_docs)} "
                  f"tools={len(s.tool_results)} score={s.scores.get('quality')}")
        final = event["state"]
    print(f"\nanswer: {final.answer}")
    print(f"score:  {final.scores.get('quality')}  iterations: {final.iteration}")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "explain the refund policy and add 2 + 2"
    asyncio.run(main(query))
