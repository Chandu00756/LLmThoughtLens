"""Ingest API — let an external process (the SDK) push events into the dashboard.

The SDK running inside someone else's app posts ``{kind, data}`` here; the
server republishes it on the :class:`EventBus`, so a trace produced in another
process appears live in the browser dashboard.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from LLmThoughtLens.server.bus import get_bus


class IngestEvent(BaseModel):
    kind: str
    data: dict[str, Any] = {}


def build_router() -> APIRouter:
    router = APIRouter(prefix="/api", tags=["ingest"])
    bus = get_bus()

    @router.post("/ingest")
    def ingest(ev: IngestEvent) -> dict[str, Any]:
        published = bus.publish(ev.kind, ev.data)
        return {"ok": True, "ts": published["ts"]}

    return router
