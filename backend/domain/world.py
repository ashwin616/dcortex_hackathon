"""Copy-on-write disruption overlay.

The base snapshot is never mutated. A disruption produces a `World` that
answers the same questions as the base store but with the event applied, so a
Tier-2 "what if" is a real simulation rather than a prompt asking the model to
imagine consequences. Worlds do not chain: each starts from the base snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from backend.domain.data import Store, store as base_store
from backend.domain.models import Duty


@dataclass
class SickCall:
    crew_id: str
    pairing_id: str | None = None
    from_date: str | None = None
    reported_utc: str | None = None


@dataclass
class Closure:
    station: str
    date: str
    start_utc: str   # "HH:MM"
    end_utc: str     # "HH:MM"


class World:
    """Base snapshot + a list of applied events."""

    def __init__(self, st: Store | None = None):
        self.s = st or base_store()
        self.sick: list[SickCall] = []
        self.closures: list[Closure] = []
        self.delays: dict[tuple[str, str], float] = {}   # (pairing_id, date) -> hours
        self.cover: dict[str, list[Duty]] = {}           # crew_id -> extra duties
        self.events: list[str] = []

    # ---- event application -------------------------------------------
    def with_sick(self, crew_id: str, pairing_id: str | None = None,
                  from_date: str | None = None, reported_utc: str | None = None) -> "World":
        self.sick.append(SickCall(crew_id, pairing_id, from_date, reported_utc))
        self.events.append(f"SICK_CREW {crew_id}" + (f" on {pairing_id}" if pairing_id else ""))
        return self

    def with_closure(self, station: str, on_date: str, start: str, end: str) -> "World":
        self.closures.append(Closure(station, on_date, start, end))
        self.events.append(f"STATION_CLOSURE {station} {on_date} {start}-{end}Z")
        return self

    def with_delay(self, pairing_id: str, on_date: str, hours_: float) -> "World":
        self.delays[(pairing_id, on_date)] = hours_
        self.events.append(f"DELAY {pairing_id} {on_date} +{hours_}h")
        return self

    def with_cover(self, crew_id: str, duties: list[Duty]) -> "World":
        self.cover.setdefault(crew_id, []).extend(duties)
        self.events.append(f"COVER {crew_id} x{len(duties)}")
        return self

    # ---- queries ------------------------------------------------------
    def is_sick(self, crew_id: str, pairing_id: str, on_date: str) -> bool:
        for s in self.sick:
            if s.crew_id != crew_id:
                continue
            if s.pairing_id and s.pairing_id != pairing_id:
                continue
            if s.from_date and on_date < s.from_date:
                continue
            return True
        return False

    def duties(self, crew_id: str, exclude_pairing: str | None = None) -> list[Duty]:
        """This crew's planned duties in this world."""
        out = []
        for d in self.s.week_duties(crew_id):
            if exclude_pairing and d.pairing_id == exclude_pairing:
                continue
            if self.is_sick(crew_id, d.pairing_id, d.date):
                continue
            out.append(d)
        out.extend(self.cover.get(crew_id, []))
        out.sort(key=lambda d: d.report_utc)
        return out

    def window_hours(self, crew_id: str, end_d: date, days: int, kind: str = "duty",
                     exclude_pairing: str | None = None) -> float:
        """Calendar-day window sum: daily_history (to 14 Sep) + planned week duties.

        Verified to reproduce duty_hours_7d / flight_hours_28d for all 150 crew.
        """
        start = end_d - timedelta(days=days - 1)
        idx = 0 if kind == "duty" else 1
        total = 0.0
        for d, v in self.s.history.get(crew_id, {}).items():
            if start <= d <= end_d:
                total += v[idx]
        for du in self.duties(crew_id, exclude_pairing=exclude_pairing):
            dd = date.fromisoformat(du.date)
            if start <= dd <= end_d:
                total += du.duty_hours if idx == 0 else du.flight_hours
        return round(total, 2)

    def uncovered_flights(self) -> dict[str, list[str]]:
        """flight_id -> [roles now missing], from every applied sick call."""
        out: dict[str, list[str]] = {}
        for sc in self.sick:
            pids = [sc.pairing_id] if sc.pairing_id else self.s.pairings_of.get(sc.crew_id, [])
            for pid in pids:
                p = self.s.pairing_by_id.get(pid)
                if not p:
                    continue
                role = self.s.roles_in[pid].get(sc.crew_id, "Crew")
                for day in p["days"]:
                    if sc.from_date and day["date"] < sc.from_date:
                        continue
                    for fid in day["flights"]:
                        out.setdefault(fid, []).append(role)
        return out

    def closure_hits(self) -> list[dict]:
        """Flights whose departure or arrival falls inside a closure window."""
        hits = []
        for c in self.closures:
            for f in self.s.flights_by_date.get(c.date, []):
                dep_t = f["dep_utc"][11:16]
                arr_t = f["arr_utc"][11:16]
                dep_hit = f["dep_station"] == c.station and c.start_utc <= dep_t < c.end_utc
                arr_hit = f["arr_station"] == c.station and c.start_utc <= arr_t < c.end_utc
                if dep_hit or arr_hit:
                    hits.append({"flight_id": f["flight_id"], "flight_no": f["flight_no"],
                                 "reason": "departure" if dep_hit else "arrival",
                                 "station": c.station, "seats": f["seats"],
                                 "sched_utc": f["dep_utc"] if dep_hit else f["arr_utc"]})
        return hits

    def describe(self) -> str:
        return "; ".join(self.events) if self.events else "base snapshot (no disruption applied)"


def base_world() -> World:
    return World()
