"""Helpers shared by the tool modules.

Every tool returns the same envelope - data, the records it consulted, and the
rules it evaluated - so the UI, the grounding validator and the audit log all
read the same shape regardless of which tool produced it.
"""
from __future__ import annotations

from datetime import date

from backend.domain.data import store
from backend.domain.rules import MAX_DUTY_7D, MAX_FLIGHT_28D
from backend.domain.world import World


def _s():
    return store()


def _ev(prefix: str, ids) -> list:
    """Evidence entries: "file.json#record_id" for everything consulted."""
    if isinstance(ids, str):
        ids = [ids]
    return [f"{prefix}#{i}" for i in ids]


def _clock_summary(cid: str, on_date: str | None = None) -> dict:
    s = _s()
    w = World(s)
    d = date.fromisoformat(on_date) if on_date else s.snapshot.date()
    d7 = w.window_hours(cid, d, 7, "duty")
    f28 = w.window_hours(cid, d, 28, "flight")
    return {
        "crew_id": cid,
        "window_end": d.isoformat(),
        "duty_hours_7d": d7,
        "duty_headroom_7d": round(MAX_DUTY_7D - d7, 2),
        "flight_hours_28d": f28,
        "flight_headroom_28d": round(MAX_FLIGHT_28D - f28, 2),
        "last_rest_ended": s.clocks.get(cid, {}).get("last_rest_ended"),
    }

