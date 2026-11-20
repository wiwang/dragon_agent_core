"""Build runtime components from Settings.

Each factory defaults to the in-memory/mock implementation and switches to a
real backend by env var. Phase 4 fills in the redis/qdrant/kafka/postgres
branches; the selection logic already lives here so nothing else changes.
"""

from __future__ import annotations

from mini_lang.agent import build_agent_graph
from mini_lang.checkpoint.base import Checkpointer
from mini_lang.checkpoint.memory import InMemoryCheckpointer
from mini_lang.config import Settings, get_settings
from mini_lang.events.base import EventBus
from mini_lang.events.inmemory import InMemoryEventBus
from mini_lang.graph import Graph
from mini_lang.llm import build_llm
from mini_lang.persistence.base import ConversationStore
from mini_lang.persistence.memory import InMemoryConversationStore
from mini_lang.rag.vectorstore import VectorStore, build_seeded_store
from mini_lang.tools.builtin import default_registry


def build_checkpointer(settings: Settings) -> Checkpointer:
    if settings.checkpointer == "redis":
        from mini_lang.checkpoint.redis import RedisCheckpointer

        return RedisCheckpointer(settings.redis_url)
    return InMemoryCheckpointer()


async def build_vectorstore(settings: Settings) -> VectorStore:
    if settings.vectorstore == "qdrant":
        from mini_lang.rag.qdrant_store import QdrantVectorStore

        return await QdrantVectorStore.create(settings)
    return await build_seeded_store()


def build_eventbus(settings: Settings) -> EventBus:
    if settings.eventbus == "kafka":
        from mini_lang.events.kafka import KafkaEventBus

        return KafkaEventBus(settings.kafka_bootstrap, settings.kafka_topic)
    return InMemoryEventBus()


async def build_conversation_store(settings: Settings) -> ConversationStore:
    if settings.conversation_store == "postgres":
        from mini_lang.persistence.postgres import PostgresConversationStore

        return await PostgresConversationStore.create(settings.postgres_dsn)
    return InMemoryConversationStore()


async def build_tracker(settings: Settings):
    if settings.mlflow_enabled:
        from mini_lang.eval.mlflow_tracker import MlflowTracker

        return MlflowTracker(settings.mlflow_tracking_uri)
    return None


async def build_graph(settings: Settings | None = None) -> Graph:
    settings = settings or get_settings()
    return build_agent_graph(
        llm=build_llm(settings),
        registry=default_registry(),
        store=await build_vectorstore(settings),
        checkpointer=build_checkpointer(settings),
        settings=settings,
        tracker=await build_tracker(settings),
    )
