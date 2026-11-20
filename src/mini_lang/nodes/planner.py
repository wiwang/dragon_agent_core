"""PlannerNode — decides how to handle the request.

Asks the LLM (tag="plan") to choose a route: use a tool, retrieve from the
knowledge base, do both (in parallel), or answer directly. Records the user
turn on first entry so re-entries from the evaluator loop don't duplicate it.
"""

from __future__ import annotations

from ..graph import Node
from ..llm.base import LLMClient
from ..state import AgentState, Message

_VALID_ROUTES = {"tool", "retrieve", "both", "answer"}

_SYSTEM = (
    "You are a planner. Choose exactly one route for the user's request, "
    "replying with a single word:\n"
    "- 'tool' for calculations or actions,\n"
    "- 'retrieve' for questions answerable from the knowledge base,\n"
    "- 'both' when you need a calculation AND a lookup,\n"
    "- 'answer' for everything else."
)


class PlannerNode(Node):
    name = "planner"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def __call__(self, state: AgentState) -> dict:
        prompt = [
            Message(role="system", content=_SYSTEM),
            Message(role="user", content=state.user_input),
        ]
        choice = (await self.llm.acomplete(prompt, tag="plan")).strip().lower()
        route = choice if choice in _VALID_ROUTES else "answer"

        update: dict = {"route": route, "plan": f"route={route}"}
        # Record the user turn only on the first pass through the planner.
        if state.iteration == 0 and not any(
            m.role == "user" and m.content == state.user_input for m in state.messages
        ):
            update["messages"] = [Message(role="user", content=state.user_input)]
        return update
