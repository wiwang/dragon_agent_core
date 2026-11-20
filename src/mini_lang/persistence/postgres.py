"""PostgreSQL conversation store (SQLAlchemy async + asyncpg)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from .base import RunRecord
from .models import Base, Conversation, Run


class PostgresConversationStore:
    def __init__(self, dsn: str):
        self._engine = create_async_engine(dsn, future=True)
        self._session = async_sessionmaker(self._engine, expire_on_commit=False)

    @classmethod
    async def create(cls, dsn: str) -> "PostgresConversationStore":
        store = cls(dsn)
        await store.init_models()
        return store

    async def init_models(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def save_run(self, record: RunRecord) -> None:
        async with self._session() as session:
            # Upsert the parent conversation, then insert the run.
            await session.execute(
                insert(Conversation)
                .values(thread_id=record.thread_id)
                .on_conflict_do_nothing(index_elements=["thread_id"])
            )
            session.add(
                Run(
                    run_id=record.run_id,
                    thread_id=record.thread_id,
                    route=record.route,
                    score=record.score,
                    answer=record.answer,
                    messages=record.messages,
                )
            )
            await session.commit()

    async def get_history(self, thread_id: str) -> list[RunRecord]:
        async with self._session() as session:
            rows = (
                await session.execute(
                    select(Run).where(Run.thread_id == thread_id).order_by(Run.created_at)
                )
            ).scalars().all()
        return [
            RunRecord(
                run_id=r.run_id,
                thread_id=r.thread_id,
                route=r.route,
                score=r.score,
                answer=r.answer,
                messages=r.messages or [],
            )
            for r in rows
        ]

    async def get_run(self, run_id: str) -> RunRecord | None:
        async with self._session() as session:
            r = (
                await session.execute(select(Run).where(Run.run_id == run_id))
            ).scalar_one_or_none()
        if r is None:
            return None
        return RunRecord(
            run_id=r.run_id,
            thread_id=r.thread_id,
            route=r.route,
            score=r.score,
            answer=r.answer,
            messages=r.messages or [],
        )

    async def close(self) -> None:
        await self._engine.dispose()
