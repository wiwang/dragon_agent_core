"""LLM factory — selects a client from Settings."""

from __future__ import annotations

from ..config import Settings, get_settings
from .base import LLMClient
from .mock import MockLLM

__all__ = ["LLMClient", "MockLLM", "build_llm"]


def build_llm(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("AGENT_ANTHROPIC_API_KEY is required for the anthropic provider")
        from .anthropic import AnthropicLLM

        return AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model)
    return MockLLM()
