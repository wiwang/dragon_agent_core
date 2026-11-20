"""Retry (backoff) and timeout decorators."""

import asyncio

import pytest

from mini_lang import RetryError, with_retry, with_timeout


async def test_retry_succeeds_after_transient_failures():
    attempts = {"n": 0}

    @with_retry(max_attempts=3, base_delay=0, jitter=False)
    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("transient")
        return "ok"

    assert await flaky() == "ok"
    assert attempts["n"] == 3


async def test_retry_exhausts_and_raises_retryerror():
    @with_retry(max_attempts=2, base_delay=0, jitter=False)
    async def always_fails():
        raise ValueError("nope")

    with pytest.raises(RetryError) as exc:
        await always_fails()
    assert isinstance(exc.value.__cause__, ValueError)


async def test_retry_only_on_listed_exceptions():
    @with_retry(max_attempts=3, base_delay=0, retry_on=(KeyError,))
    async def wrong_error():
        raise ValueError("not retried")

    with pytest.raises(ValueError):
        await wrong_error()


async def test_timeout_cancels_slow_call():
    @with_timeout(0.05)
    async def slow():
        await asyncio.sleep(1)

    with pytest.raises(asyncio.TimeoutError):
        await slow()


async def test_timeout_passes_fast_call():
    @with_timeout(0.5)
    async def fast():
        return 42

    assert await fast() == 42
