"""Reliability primitives: retry with exponential backoff + jitter, and timeout.

Both are async decorators meant to wrap anything that touches the network — LLM
calls, tool I/O, vector-store queries. Kept tiny and dependency-free so the
behaviour is obvious and testable.
"""

from __future__ import annotations

import asyncio
import random
from functools import wraps
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class RetryError(RuntimeError):
    """Raised when all retry attempts are exhausted (chains the last error)."""


def with_timeout(seconds: float) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Cancel the wrapped coroutine if it runs longer than ``seconds``."""

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(fn)
        async def wrapper(*args, **kwargs) -> T:
            return await asyncio.wait_for(fn(*args, **kwargs), timeout=seconds)

        return wrapper

    return decorator


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 5.0,
    jitter: bool = True,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Retry on ``retry_on`` with capped exponential backoff.

    Delay before attempt *n* (1-indexed) is ``min(max_delay, base_delay * 2**(n-1))``,
    optionally multiplied by a random factor in ``[0.5, 1.0]`` to avoid
    thundering-herd alignment. ``asyncio.CancelledError`` is never retried.
    """

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(fn)
        async def wrapper(*args, **kwargs) -> T:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except asyncio.CancelledError:
                    raise
                except retry_on as exc:  # type: ignore[misc]
                    last_exc = exc
                    if attempt >= max_attempts:
                        break
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    if jitter:
                        delay *= 0.5 + random.random() * 0.5
                    await asyncio.sleep(delay)
            raise RetryError(f"{fn.__name__} failed after {max_attempts} attempts") from last_exc

        return wrapper

    return decorator
