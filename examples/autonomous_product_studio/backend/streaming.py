"""Run events: persisted in order, woken live, and streamed as Server-Sent Events."""

from __future__ import annotations

import asyncio
import json
import threading
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any, Protocol
from uuid import UUID

from .auth import Principal, principal_scope
from .store import EVENT_PAGE_SIZE

TERMINAL_EVENTS = frozenset({"completed", "failed", "cancelled", "rejected"})


class EventStore(Protocol):
    def append_event(
        self, run_id: UUID, event_type: str, payload: dict[str, Any]
    ) -> dict[str, Any]: ...

    def read_events(self, run_id: UUID, *, after: int = 0) -> list[dict[str, Any]]: ...


class RunEventBroker:
    """Appends run events to the store and streams them to SSE clients.

    The store is the single ordered source of events, so API and worker
    processes share one stream and reconnecting clients resume with
    `Last-Event-ID`. Publishing in this process wakes local subscribers at once;
    events from other processes arrive on the next poll.
    """

    def __init__(
        self,
        store: EventStore,
        *,
        poll_interval: float = 0.5,
        keepalive_interval: float = 15.0,
    ) -> None:
        self.store = store
        self.poll_interval = poll_interval
        self.keepalive_interval = keepalive_interval
        self._subscribers: dict[
            UUID, set[tuple[asyncio.AbstractEventLoop, asyncio.Event]]
        ] = defaultdict(set)
        self._lock = threading.Lock()

    def publish(self, run_id: UUID, event_type: str, **payload: Any) -> dict[str, Any]:
        """Persist one event for the current principal's tenant and wake streams."""
        event = self.store.append_event(
            run_id, event_type, json.loads(json.dumps(payload, default=str))
        )
        with self._lock:
            waiters = tuple(self._subscribers.get(run_id, ()))
        for loop, wake in waiters:
            try:
                loop.call_soon_threadsafe(wake.set)
            except RuntimeError:
                continue
        return event

    async def stream(
        self, run_id: UUID, *, principal: Principal, after: int = 0
    ) -> AsyncIterator[str]:
        """Yield SSE frames after `after` until a terminal event is sent."""
        loop = asyncio.get_running_loop()
        wake = asyncio.Event()
        subscriber = (loop, wake)
        with self._lock:
            self._subscribers[run_id].add(subscriber)
        last_frame = loop.time()
        try:
            while True:
                wake.clear()
                events = await asyncio.to_thread(self._read, principal, run_id, after)
                for event in events:
                    yield _encode(event)
                    after = int(event["id"])
                    last_frame = loop.time()
                    if event["type"] in TERMINAL_EVENTS:
                        return
                if len(events) >= EVENT_PAGE_SIZE:
                    continue
                try:
                    await asyncio.wait_for(wake.wait(), timeout=self.poll_interval)
                except TimeoutError:
                    if loop.time() - last_frame >= self.keepalive_interval:
                        last_frame = loop.time()
                        yield ": keep-alive\n\n"
        finally:
            with self._lock:
                self._subscribers[run_id].discard(subscriber)
                if not self._subscribers[run_id]:
                    del self._subscribers[run_id]

    def _read(
        self, principal: Principal, run_id: UUID, after: int
    ) -> list[dict[str, Any]]:
        with principal_scope(principal):
            return self.store.read_events(run_id, after=after)


def _encode(event: dict[str, Any]) -> str:
    return f"id: {event['id']}\ndata: {json.dumps(event, default=str)}\n\n"
