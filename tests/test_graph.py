"""Graph engine: linear traversal, conditional routing, parallel fan-out, checkpointing."""

import pytest

from mini_lang import END, AgentState, Graph, GraphRecursionError
from mini_lang.checkpoint.memory import InMemoryCheckpointer


def _appender(label: str):
    async def fn(state: AgentState):
        return {"messages": [{"role": "assistant", "content": label}]}

    return fn


async def test_linear_traversal_runs_nodes_in_order():
    g = Graph()
    g.add_node(_appender("a"), "a").add_node(_appender("b"), "b")
    g.set_entry_point("a").add_edge("a", "b").add_edge("b", END)

    final = await g.run(AgentState(user_input="hi"))
    assert [m.content for m in final.messages] == ["a", "b"]


async def test_conditional_routing_picks_branch():
    g = Graph()
    g.add_node(_appender("left"), "left").add_node(_appender("right"), "right")
    g.set_entry_point("router_src")
    # entry node just records a choice in scratchpad
    async def chooser(state: AgentState):
        return {"scratchpad": {"pick": state.user_input}}

    g.add_node(chooser, "router_src")
    g.add_conditional_edges("router_src", lambda s: s.scratchpad["pick"])
    g.add_edge("left", END).add_edge("right", END)

    final = await g.run(AgentState(user_input="right"))
    assert final.messages[-1].content == "right"


async def test_parallel_fanout_merges_then_joins():
    g = Graph()

    async def start(state: AgentState):
        return {"scratchpad": {"started": True}}

    async def branch_x(state: AgentState):
        return {"scratchpad": {"x": 1}}

    async def branch_y(state: AgentState):
        return {"scratchpad": {"y": 2}}

    g.add_node(start, "start")
    g.add_node(branch_x, "x").add_node(branch_y, "y").add_node(_appender("joined"), "join")
    g.set_entry_point("start")
    g.add_conditional_edges("start", lambda s: ["x", "y"])  # fan out
    g.add_edge("x", "join").add_edge("y", "join")  # both converge
    g.add_edge("join", END)

    final = await g.run(AgentState())
    assert final.scratchpad == {"started": True, "x": 1, "y": 2}
    assert final.messages[-1].content == "joined"


async def test_parallel_branches_must_join():
    g = Graph()
    g.add_node(_appender("s"), "s")
    g.add_node(_appender("x"), "x").add_node(_appender("y"), "y")
    g.add_node(_appender("p"), "p").add_node(_appender("q"), "q")
    g.set_entry_point("s")
    g.add_conditional_edges("s", lambda _: ["x", "y"])
    g.add_edge("x", "p").add_edge("y", "q")  # diverge -> error
    g.add_edge("p", END).add_edge("q", END)

    with pytest.raises(GraphRecursionError):
        await g.run(AgentState())


async def test_max_steps_guards_infinite_loop():
    g = Graph()
    g.add_node(_appender("loop"), "loop").set_entry_point("loop")
    g.add_edge("loop", "loop")  # never terminates
    with pytest.raises(GraphRecursionError):
        await g.run(AgentState(), max_steps=5)


async def test_checkpoint_and_resume():
    cp = InMemoryCheckpointer()
    g = Graph(checkpointer=cp)
    g.add_node(_appender("a"), "a").add_node(_appender("b"), "b")
    g.set_entry_point("a").add_edge("a", "b").add_edge("b", END)

    await g.run(AgentState(thread_id="t1"), thread_id="t1")
    resumed = await g.resume("t1")
    assert resumed is not None
    assert [m.content for m in resumed.messages] == ["a", "b"]
    # full history: start checkpoint after each of the 2 nodes
    assert len(await cp.history("t1")) == 2


async def test_astream_emits_events():
    g = Graph()
    g.add_node(_appender("a"), "a").set_entry_point("a").add_edge("a", END)
    events = [ev["event"] async for ev in g.astream(AgentState())]
    assert events == ["start", "node", "end"]
