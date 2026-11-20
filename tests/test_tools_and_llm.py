"""Tool registry, safe calculator, and the deterministic mock LLM."""

import pytest

from mini_lang.llm.mock import MockLLM
from mini_lang.state import Message
from mini_lang.tools.builtin import calculator, default_registry


async def test_calculator_evaluates_arithmetic():
    assert await calculator("2 + 3 * 4") == 14
    assert await calculator("(10 - 2) / 4") == 2


async def test_calculator_rejects_unsafe_input():
    with pytest.raises(ValueError):
        await calculator("__import__('os').system('ls')")


async def test_registry_dispatch_and_error_handling():
    reg = default_registry()
    ok = await reg.call("calculator", expression="6 * 7")
    assert ok.ok and ok.output == 42

    missing = await reg.call("nope")
    assert not missing.ok and "unknown tool" in missing.error

    bad = await reg.call("calculator", expression="a + b")
    assert not bad.ok  # error surfaced as data, not raised


async def test_mock_llm_plan_routing():
    llm = MockLLM()
    msgs = lambda t: [Message(role="user", content=t)]
    assert await llm.acomplete(msgs("what is 2 + 2"), tag="plan") == "both"
    assert await llm.acomplete(msgs("2 * 21"), tag="plan") == "tool"
    assert await llm.acomplete(msgs("explain the refund policy"), tag="plan") == "retrieve"
    assert await llm.acomplete(msgs("say hello"), tag="plan") == "answer"


async def test_mock_llm_scripted_responses():
    llm = MockLLM(responses={"answer": "scripted"})
    out = await llm.acomplete([Message(role="user", content="x")], tag="answer")
    assert out == "scripted"
    assert llm.calls  # call was recorded


async def test_mock_llm_streaming():
    llm = MockLLM(responses={"answer": "one two three"})
    chunks = [c async for c in llm.astream([Message(role="user", content="x")], tag="answer")]
    assert "".join(chunks).strip() == "one two three"
