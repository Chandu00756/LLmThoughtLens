"""EventBus — async pub/sub that drives every live dashboard update.

Each subscriber gets its own bounded ``asyncio.Queue``.  Publishers call
:meth:`publish` (sync-safe) with a JSON-serialisable event dict; subscribers
``async for`` over :meth:`subscribe`.  Slow subscribers drop their oldest
events rather than blocking the whole bus.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any


class EventBus:
    """A tiny in-process fan-out bus for live trace events."""

    def __init__(self, max_queue: int = 1000) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._max_queue = int(max_queue)
        self._history: list[dict[str, Any]] = []
        self._history_limit = 200
        self._lock = asyncio.Lock()

    async def subscribe(self) -> AsyncIterator[dict[str, Any]]:
        """Yield events as they are published, starting with recent history."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._max_queue)
        async with self._lock:
            # Replay recent history so a freshly-opened dashboard isn't blank.
            for event in self._history[-50:]:
                with _suppress_full(queue):
                    queue.put_nowait(event)
            self._subscribers.add(queue)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            async with self._lock:
                self._subscribers.discard(queue)

    def publish(self, kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Publish an event to every subscriber. Safe to call from any thread/loop."""
        event = {
            "kind": kind,
            "ts": time.time(),
            "data": payload or {},
        }
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit :]
        for queue in list(self._subscribers):
            if queue.full():
                # Drop the oldest event for this slow subscriber.
                with _suppress_empty(queue):
                    queue.get_nowait()
            with _suppress_full(queue):
                queue.put_nowait(event)
        return event

    @property
    def n_subscribers(self) -> int:
        return len(self._subscribers)

    def recent(self, n: int = 50) -> list[dict[str, Any]]:
        return self._history[-n:]


class _suppress_full:
    """Context manager: ignore ``asyncio.QueueFull``."""

    def __init__(self, queue: asyncio.Queue) -> None:
        self.queue = queue

    def __enter__(self) -> _suppress_full:
        return self

    def __exit__(self, exc_type: Any, *_: Any) -> bool:
        return exc_type is asyncio.QueueFull


class _suppress_empty:
    """Context manager: ignore ``asyncio.QueueEmpty``."""

    def __init__(self, queue: asyncio.Queue) -> None:
        self.queue = queue

    def __enter__(self) -> _suppress_empty:
        return self

    def __exit__(self, exc_type: Any, *_: Any) -> bool:
        return exc_type is asyncio.QueueEmpty


# Module-level singleton used by the FastAPI app + proxy + white-box stream.
_GLOBAL_BUS: EventBus | None = None


def get_bus() -> EventBus:
    """Return the process-wide :class:`EventBus`, creating it on first use."""
    global _GLOBAL_BUS
    if _GLOBAL_BUS is None:
        _GLOBAL_BUS = EventBus()
    return _GLOBAL_BUS
