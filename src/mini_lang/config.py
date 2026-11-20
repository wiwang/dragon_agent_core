"""Runtime configuration.

Every pluggable seam (LLM, checkpointer, vector store, event bus, tracing,
eval tracking) is selected here from an env var, defaulting to the in-memory /
mock implementation so the whole system runs with zero external services.
Flip a value (e.g. ``AGENT_CHECKPOINTER=redis``) to use a real backend.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_", env_file=".env", extra="ignore"
    )

    # LLM: mock | anthropic
    llm_provider: str = "mock"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"

    # Checkpointer: memory | redis
    checkpointer: str = "memory"
    redis_url: str = "redis://localhost:6379/0"

    # Vector store: memory | qdrant
    vectorstore: str = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "agent_docs"

    # Conversation store: memory | postgres
    conversation_store: str = "memory"
    postgres_dsn: str = "postgresql+asyncpg://agent:agent@localhost:5432/agent"

    # Event bus: memory | kafka
    eventbus: str = "memory"
    kafka_bootstrap: str = "localhost:9092"
    kafka_topic: str = "agent.runs"

    # Observability + eval
    tracing_enabled: bool = False
    otlp_endpoint: str | None = None
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Number of in-process queue workers the API starts. Set to 0 in Kubernetes
    # where a separate worker Deployment consumes the (Kafka) bus and scales.
    api_inproc_workers: int = 2

    # Agent behaviour
    max_iterations: int = 2  # evaluator-driven retry loop bound
    eval_threshold: float = 0.7  # below this, loop back to the planner

    service_url: str = "http://localhost:8000"  # used by the Gradio UI


@lru_cache
def get_settings() -> Settings:
    return Settings()
