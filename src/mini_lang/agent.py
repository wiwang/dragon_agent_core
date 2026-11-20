"""Assemble the default agent graph.

Topology::

                       ┌─ tool ─┐
    planner ─ route ─┤          ├─ answer ─ evaluator ─┬─ (score<thr) ─ planner
                       └ retriever┘                     └─ END

The planner's route selects one of: ``tool``, ``retriever``, both in parallel,
or straight to ``answer``. The evaluator loops back to the planner while the
quality score is below threshold, bounded by ``max_iterations``.
"""

from __future__ import annotations

from .checkpoint.base import Checkpointer
from .config import Settings, get_settings
from .graph import END, Graph
from .llm import build_llm
from .llm.base import LLMClient
from .nodes.answer import AnswerNode
from .nodes.evaluator import EvaluatorNode
from .nodes.planner import PlannerNode
from .nodes.retriever import RetrieverNode
from .nodes.tool_node import ToolNode
from .rag.vectorstore import VectorStore, build_seeded_store
from .router import RouteMap
from .state import AgentState
from .tools.base import ToolRegistry
from .tools.builtin import default_registry


def _eval_router(settings: Settings):
    def route(state: AgentState):
        score = state.scores.get("quality", 0.0)
        if score >= settings.eval_threshold or state.iteration >= settings.max_iterations:
            return END
        return "planner"

    return route


def build_agent_graph(
    *,
    llm: LLMClient | None = None,
    registry: ToolRegistry | None = None,
    store: VectorStore | None = None,
    checkpointer: Checkpointer | None = None,
    settings: Settings | None = None,
    tracker=None,
) -> Graph:
    settings = settings or get_settings()
    llm = llm or build_llm(settings)
    registry = registry or default_registry()
    store = store  # may be None; caller can inject a seeded store

    g = Graph(checkpointer=checkpointer)
    g.add_node(PlannerNode(llm))
    g.add_node(ToolNode(llm, registry))
    if store is not None:
        g.add_node(RetrieverNode(store))
    g.add_node(AnswerNode(llm))
    g.add_node(EvaluatorNode(llm, tracker=tracker))

    g.set_entry_point("planner")

    # Planner route -> branch(es). "both" fans out to a parallel group.
    route_table = {
        "tool": "tool",
        "retrieve": "retriever" if store is not None else "answer",
        "both": ["retriever", "tool"] if store is not None else "tool",
        "answer": "answer",
    }
    g.add_conditional_edges("planner", RouteMap("route", route_table, default="answer"))

    # Branches converge on the answer node.
    g.add_edge("tool", "answer")
    if store is not None:
        g.add_edge("retriever", "answer")
    g.add_edge("answer", "evaluator")

    # Evaluator loops back to the planner or ends.
    g.add_conditional_edges("evaluator", _eval_router(settings))
    return g


async def build_default_agent(settings: Settings | None = None, **kwargs) -> Graph:
    """Convenience builder with a seeded in-memory vector store for RAG."""
    settings = settings or get_settings()
    store = kwargs.pop("store", None) or await build_seeded_store()
    return build_agent_graph(store=store, settings=settings, **kwargs)
