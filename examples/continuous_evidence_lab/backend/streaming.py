"""Server-sent events backed by the persisted, ordered run event log."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Mapping
from typing import Any, Protocol
from uuid import UUID

# The stream ends on these events; after a reviewer decides, the client
# reconnects with `Last-Event-ID` to follow the resumed run.
STREAM_END = frozenset({"awaiting_approval", "completed", "failed", "cancelled"})


class EventLog(Protocol):
    async def append_event(
        self, run_id: UUID, event_type: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]: ...

    async def read_events(
        self, run_id: UUID, after: int = 0
    ) -> list[dict[str, Any]]: ...


class RunEventBroker:
    """Publishes run events and streams them, replaying from `Last-Event-ID`.

    Events are written by the API and by workers in other processes, so the
    stream polls the event log instead of relying on in-process notification.
    """

    def __init__(
        self,
        log: EventLog,
        *,
        poll_interval: float = 0.5,
        keepalive_interval: float = 15.0,
    ) -> None:
        self.log = log
        self.poll_interval = poll_interval
        self.keepalive_interval = keepalive_interval

    async def publish(
        self, run_id: UUID, event_type: str, **payload: Any
    ) -> dict[str, Any]:
        return await self.log.append_event(run_id, event_type, payload)

    async def stream(self, run_id: UUID, after: int = 0) -> AsyncGenerator[str, None]:
        idle = 0.0
        while True:
            events = await self.log.read_events(run_id, after=after)
            for event in events:
                yield self.encode(event)
                after = int(event["id"])
                if event["type"] in STREAM_END:
                    return
            if events:
                idle = 0.0
            else:
                idle += self.poll_interval
                if idle >= self.keepalive_interval:
                    idle = 0.0
                    yield ": keep-alive\n\n"
            await asyncio.sleep(self.poll_interval)

    @staticmethod
    def encode(event: Mapping[str, Any]) -> str:
        identifier = f"id: {event['id']}\n" if "id" in event else ""
        return f"{identifier}data: {json.dumps(dict(event), default=str)}\n\n"
