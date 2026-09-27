"""Money. Every figure is itemised so the UI can show the arithmetic.

Rates come from costs.json - nothing here is hardcoded except the structure.
"""
from __future__ import annotations

from datetime import timedelta

from backend.domain.data import Store, hours, parse_dt

PILOT_RANKS = ("Captain", "First Officer")
POSITIONING_TRANSIT_MIN = 15   # deadhead arrival -> report
REPORT_LEAD_MIN = 60           # report -> departure


def callout_cost(s: Store, rank: str, is_reserve: bool) -> tuple[int, str]:
    pilot = rank in PILOT_RANKS
    if is_reserve:
        key = "reserve_callout_pilot" if pilot else "reserve_callout_cabin"
        label = "reserve callout"
    else:
        key = "dayoff_callout_pilot" if pilot else "dayoff_callout_cabin"
        label = "day-off callout"
    return int(s.costs[key]), label


def positioning(s: Store, from_station: str, to_station: str, on_date: str,
                required_dep_utc: str):
    """Cheapest same-day deadhead that gets the crew to `to_station` in time.

    Returns (flight_record, new_report_dt, delay_hours) or None if no leg exists.
    New report = positioning arrival + 15 min transit; the duty's first
    departure moves to report + 60 min, and the difference is the delay.
    """
    legs = [f for f in s.flights_by_date.get(on_date, [])
            if f["dep_station"] == from_station and f["arr_station"] == to_station]
    if not legs:
        return None
    best = None
    for f in sorted(legs, key=lambda x: x["arr_utc"]):
        arr = parse_dt(f["arr_utc"])
        new_report = arr + timedelta(minutes=POSITIONING_TRANSIT_MIN)
        new_dep = new_report + timedelta(minutes=REPORT_LEAD_MIN)
        delay = round(max(0.0, hours(new_dep - parse_dt(required_dep_utc))), 2)
        if best is None or delay < best[2]:
            best = (f, new_report, delay)
    return best


def deadhead_cost(s: Store, delay_h: float) -> tuple[int, dict]:
    pos = int(s.costs["deadhead_positioning"])
    delay = int(round(delay_h * s.costs["delay_cost_per_duty_hour"]))
    return pos + delay, {"deadhead_positioning": pos,
                         "delay": delay,
                         "delay_hours": delay_h,
                         "delay_rate_per_hour": int(s.costs["delay_cost_per_duty_hour"])}


def cancellation_cost(s: Store, n_flights: int) -> int:
    return int(s.costs["cancellation_per_flight"]) * n_flights


def hotel_cost(s: Store, nights: int = 1) -> int:
    return int(s.costs["hotel_overnight"]) * nights


def inr(x: float | int) -> str:
    """Indian digit grouping: 41200 -> 'INR 41,200'."""
    n = int(round(x))
    sign = "-" if n < 0 else ""
    sn = str(abs(n))
    if len(sn) > 3:
        head, tail = sn[:-3], sn[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        sn = ",".join(parts) + "," + tail
    return f"{sign}INR {sn}"
