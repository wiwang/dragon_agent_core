"""EvaluatorNode — grades the answer and drives the retry loop.

Asks the LLM (tag="evaluate") for a 0-1 quality score given the answer and the
context it was supposed to use. Writes ``scores['quality']`` and increments the
iteration counter; the conditional edge out of this node loops back to the
planner when the score is below threshold (bounded by ``max_iterations``).

In Phase 5 this node also logs the score to MLflow — see ``eval/mlflow_tracker``.
"""

from __future__ import annotations

from ..graph import Node
from ..llm.base import LLMClient
from ..state import AgentState, Message
from .util import format_context

_SYSTEM = (
    "Grade the assistant's answer for groundedness and helpfulness. "
    "Reply with only a number between 0 and 1."
)


class EvaluatorNode(Node):
    name = "evaluator"

    def __init__(self, llm: LLMClient, tracker=None):
        self.llm = llm
        self.tracker = tracker  # optional MLflow tracker (Phase 5)

    async def __call__(self, state: AgentState) -> dict:
        context = format_context(state)
        prompt = [
            Message(role="system", content=_SYSTEM),
            Message(
                role="user",
                content=f"QUESTION: {state.user_input}\nANSWER: {state.answer}\n{context}",
            ),
        ]
        raw = await self.llm.acomplete(prompt, tag="evaluate")
        try:
            score = max(0.0, min(1.0, float(raw.strip())))
        except ValueError:
            score = 0.0

        if self.tracker is not None:
            self.tracker.log_evaluation(state, score)

        return {"scores": {"quality": score}, "iteration": state.iteration + 1}
