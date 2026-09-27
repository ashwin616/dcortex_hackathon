"""Duty-period arithmetic. Pure functions, no data access beyond what's passed in.

Conventions (rules.json):
  report  = first departure - 60 min
  release = last arrival    + 30 min
  FDP     = release - report, in hours
"""
from __future__ import annotations

from datetime import datetime, timedelta

from backend.domain.data import fmt_dt, hours, parse_dt

REPORT_LEAD_MIN = 60
RELEASE_LAG_MIN = 30


def report_release(legs: list[dict]) -> tuple[datetime, datetime]:
    """Derive report/release from an ordered list of flight records."""
    dep = parse_dt(legs[0]["dep_utc"])
    arr = parse_dt(legs[-1]["arr_utc"])
    return dep - timedelta(minutes=REPORT_LEAD_MIN), arr + timedelta(minutes=RELEASE_LAG_MIN)


def fdp_limit(sectors: int) -> float:
    """RULE-FDP-01: 13h, minus 0.5h for each sector beyond the 2nd."""
    return 13.0 - 0.5 * max(0, sectors - 2)


def duty_len(day: dict, delay_h: float = 0.0) -> tuple[datetime, datetime, float]:
    """(report, release, hours) for a pairing-day, optionally shifted by a delay.

    A delay before the first departure moves the whole duty later but does not
    change its length; a delay *during* the duty (crew already reported)
    extends it. We model the latter by shifting release only when
    `delay_h` is applied via `extend_duty`.
    """
    rep = parse_dt(day["report_utc"]) + timedelta(hours=delay_h)
    rel = parse_dt(day["release_utc"]) + timedelta(hours=delay_h)
    return rep, rel, hours(rel - rep)


def extend_duty(day: dict, delay_h: float) -> tuple[datetime, datetime, float]:
    """Crew has already reported; the delay pushes release out and lengthens FDP."""
    rep = parse_dt(day["report_utc"])
    rel = parse_dt(day["release_utc"]) + timedelta(hours=delay_h)
    return rep, rel, hours(rel - rep)


def synth_day(store, flight_ids: list[str]) -> dict:
    """Build a pairing-day-shaped dict from an arbitrary set of legs."""
    legs = [store.flight_by_id[f] for f in flight_ids]
    legs.sort(key=lambda f: f["dep_utc"])
    rep, rel = report_release(legs)
    return {
        "date": legs[0]["date"],
        "flights": [l["flight_id"] for l in legs],
        "report_utc": fmt_dt(rep),
        "release_utc": fmt_dt(rel),
    }
