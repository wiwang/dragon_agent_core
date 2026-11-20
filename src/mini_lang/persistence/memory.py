"""In-memory conversation store (default)."""

from __future__ import annotations

from collections import defaultdict

from .base import RunRecord


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._runs: dict[str, list[RunRecord]] = defaultdict(list)

    async def save_run(self, record: RunRecord) -> None:
        self._runs[record.thread_id].append(record)

    async def get_history(self, thread_id: str) -> list[RunRecord]:
        return list(self._runs.get(thread_id, []))

    async def get_run(self, run_id: str) -> RunRecord | None:
        for runs in self._runs.values():
            for record in runs:
                if record.run_id == run_id:
                    return record
        return None

    async def close(self) -> None:  # nothing to release
        return None
