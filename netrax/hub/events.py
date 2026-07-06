"""Events API: timeline with filters by type, date and label."""

from __future__ import annotations

import time
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter()
TYPES = ("seen", "unknown", "gesture", "object", "posture", "system")


def _day_start(s: str) -> float:
    try:
        return datetime.strptime(s, "%Y-%m-%d").timestamp()      # local midnight
    except ValueError:
        raise HTTPException(422, f"bad date {s!r}, use YYYY-MM-DD") from None


@router.get("/api/events")
def list_events(request: Request,
                type: str | None = Query(None, description="comma list: seen,unknown,gesture,object,posture"),
                date_from: str | None = None, date_to: str | None = None,
                since: float | None = None, until: float | None = None,
                q: str | None = None, person_id: int | None = None,
                limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    types = [t for t in (type or "").split(",") if t]
    bad = [t for t in types if t not in TYPES]
    if bad:
        raise HTTPException(422, f"unknown event type(s): {', '.join(bad)}")
    if date_from:
        since = _day_start(date_from)
    if date_to:
        until = (datetime.fromtimestamp(_day_start(date_to)) + timedelta(days=1)).timestamp()
    store = request.app.state.rt.store
    items, total = store.query_events(types or None, since, until, person_id, q, limit, offset)
    for e in items:
        e["snapshot_url"] = f"/media/{e['snapshot_path']}" if e["snapshot_path"] else None
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/api/events/recent")
def recent(request: Request, limit: int = Query(20, ge=1, le=50)):
    return list(request.app.state.rt.events.recent)[:limit]


@router.get("/api/events/summary")
def summary(request: Request):
    store = request.app.state.rt.store
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    return {"today": store.event_counts(today), "all": store.event_counts(), "now": time.time()}


@router.delete("/api/events")
def clear(request: Request):
    return {"deleted": request.app.state.rt.store.clear_events()}
