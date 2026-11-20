"""AnswerNode — synthesises the final answer from gathered context.

Sits between the tool/retrieve branches and the evaluator. Feeds the LLM
(tag="answer") the formatted context so the response is grounded in whatever
the branches produced.
"""

from __future__ import annotations

from ..graph import Node
from ..llm.base import LLMClient
from ..state import AgentState, Message
from .util import format_context

_SYSTEM = "Answer the user using the provided context. Be concise and grounded."


class AnswerNode(Node):
    name = "answer"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def __call__(self, state: AgentState) -> dict:
        context = format_context(state)
        user = state.user_input
        prompt = [
            Message(role="system", content=_SYSTEM),
            Message(role="user", content=f"{user}\n\n{context}" if context else user),
        ]
        answer = await self.llm.acomplete(prompt, tag="answer")
        return {
            "answer": answer,
            "messages": [Message(role="assistant", content=answer)],
        }
