"""End-to-end agent graph on the mock LLM: each route, parallel branch, eval loop."""

import pytest

from mini_lang.agent import build_agent_graph, build_default_agent
from mini_lang.config import Settings
from mini_lang.state import AgentState


async def _run(query: str, **kwargs):
    graph = await build_default_agent(**kwargs)
    return await graph.run(AgentState(user_input=query, thread_id="t"))


async def test_tool_route_runs_calculator():
    final = await _run("compute 2 * 21")
    assert final.route == "tool"
    assert final.tool_results[0]["tool"] == "calculator"
    assert final.tool_results[0]["output"] == 42
    assert final.scores["quality"] == 0.9  # grounded -> high score, ends


async def test_retrieve_route_pulls_docs():
    final = await _run("explain the refund policy")
    assert final.route == "retrieve"
    assert final.retrieved_docs
    assert any("refund" in d["text"].lower() for d in final.retrieved_docs)
    assert final.scores["quality"] == 0.9


async def test_both_route_runs_in_parallel():
    final = await _run("what is 12 / 4")
    assert final.route == "both"
    assert final.tool_results  # tool branch ran
    assert final.retrieved_docs  # retriever branch ran
    assert final.answer


async def test_answer_route_loops_to_max_iterations():
    # A plain greeting isn't grounded -> mock grades it low -> loop until bound.
    final = await _run("hello there")
    assert final.route == "answer"
    assert final.scores["quality"] < 0.7
    assert final.iteration == 2  # default max_iterations
    # The user turn is recorded exactly once despite looping.
    assert sum(1 for m in final.messages if m.role == "user") == 1


async def test_high_score_ends_immediately():
    # Script the evaluator to pass on the first try.
    from mini_lang.llm.mock import MockLLM

    llm = MockLLM(responses={"plan": "answer", "evaluate": "0.95"})
    graph = build_agent_graph(llm=llm, store=None)  # no RAG needed
    final = await graph.run(AgentState(user_input="hi", thread_id="t"))
    assert final.iteration == 1
    assert final.scores["quality"] == 0.95


async def test_graph_without_store_falls_back_to_answer():
    # 'retrieve' route with no store should degrade to the answer branch.
    from mini_lang.llm.mock import MockLLM

    llm = MockLLM(responses={"plan": "retrieve", "evaluate": "0.9"})
    graph = build_agent_graph(llm=llm, store=None)
    final = await graph.run(AgentState(user_input="explain docs", thread_id="t"))
    assert final.answer is not None
