"""Load the nine JSON files once, index them, and derive the week's duties.

Nothing here reasons about legality - it only makes the data addressable.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from functools import cached_property
from pathlib import Path

from backend.domain.models import Duty

FMT = "%Y-%m-%dT%H:%M:%SZ"

# The provided starter dataset stays at the repository root, unmodified: the
# supplied validate.py and generate.py both expect it there. Look for it beside
# the backend too, so the package still works if it is vendored.
_HERE = Path(__file__).resolve()
DATA_DIR = next(
    (p for p in (_HERE.parents[3] / "data", _HERE.parents[2] / "data",
                 _HERE.parents[1] / "data") if (p / "flights.json").exists()),
    _HERE.parents[3] / "data")


def parse_dt(s: str) -> datetime:
    return datetime.strptime(s, FMT)


def fmt_dt(d: datetime) -> str:
    return d.strftime(FMT)


def hours(td: timedelta) -> float:
    return round(td.total_seconds() / 3600.0, 2)


def hm(x: float) -> str:
    """0.33 -> '0h20m' - used in rule detail strings."""
    neg = x < 0
    x = abs(x)
    h = int(x)
    m = int(round((x - h) * 60))
    if m == 60:
        h, m = h + 1, 0
    return f"{'-' if neg else ''}{h}h{m:02d}m"


class Store:
    """The immutable base snapshot."""

    def __init__(self, data_dir: Path | str = DATA_DIR):
        self.dir = Path(data_dir)
        j = lambda n: json.loads((self.dir / f"{n}.json").read_text(encoding="utf-8"))

        self.flights: list[dict] = j("flights")
        self.crew_list: list[dict] = j("crew")
        rosters = j("rosters")
        self.pairings: list[dict] = rosters["pairings"]
        self.flagged_exceptions: list[dict] = rosters.get("flagged_exceptions", [])
        self.clocks_list: list[dict] = j("duty_clocks")
        self.reserve_list: list[dict] = j("reserve_pool")
        self.cert_list: list[dict] = j("certifications")
        self.rules: dict = j("rules")
        self.costs: dict = j("costs")
        self.risk_list: list[dict] = j("risk_signals")
        self.scenarios: list[dict] = j("scenarios")
        self.questions: list[dict] = j("questions")

        self.snapshot_utc: str = self.clocks_list[0]["as_of_utc"]
        self.snapshot: datetime = parse_dt(self.snapshot_utc)

        # ---- indexes -------------------------------------------------
        self.flight_by_id = {f["flight_id"]: f for f in self.flights}
        self.crew = {c["crew_id"]: c for c in self.crew_list}
        self.pairing_by_id = {p["pairing_id"]: p for p in self.pairings}
        self.clocks = {c["crew_id"]: c for c in self.clocks_list}
        self.reserves = {r["crew_id"]: r for r in self.reserve_list}
        self.risk = {r["crew_id"]: r for r in self.risk_list}

        self.certs: dict[str, dict[str, date]] = {}
        for c in self.cert_list:
            self.certs.setdefault(c["crew_id"], {})[c["cert_type"]] = date.fromisoformat(c["valid_to"])
        self.cert_from: dict[str, dict[str, date]] = {}
        for c in self.cert_list:
            self.cert_from.setdefault(c["crew_id"], {})[c["cert_type"]] = date.fromisoformat(c["valid_from"])

        # daily_history: crew_id -> {date: (duty_h, flight_h)}
        self.history: dict[str, dict[date, tuple[float, float]]] = {}
        for c in self.clocks_list:
            self.history[c["crew_id"]] = {
                date.fromisoformat(d["date"]): (d["duty_hours"], d["flight_hours"])
                for d in c.get("daily_history", [])
            }

        self.flights_by_date: dict[str, list[dict]] = {}
        for f in self.flights:
            self.flights_by_date.setdefault(f["date"], []).append(f)

        # crew_id -> [pairing_id]; pairing_id -> {crew_id: role}
        self.pairings_of: dict[str, list[str]] = {}
        self.roles_in: dict[str, dict[str, str]] = {}
        for p in self.pairings:
            self.roles_in[p["pairing_id"]] = {m["crew_id"]: m["role"] for m in p["crew"]}
            for m in p["crew"]:
                self.pairings_of.setdefault(m["crew_id"], []).append(p["pairing_id"])

        self._week_duties = self._derive_week_duties()

    # ---- derived ------------------------------------------------------
    def _derive_week_duties(self) -> dict[str, list[Duty]]:
        """Rostered duties for the planned week, per crew, straight from rosters.json."""
        out: dict[str, list[Duty]] = {}
        for p in self.pairings:
            for day in p["days"]:
                legs = [self.flight_by_id[fid] for fid in day["flights"]]
                rep, rel = parse_dt(day["report_utc"]), parse_dt(day["release_utc"])
                dh = hours(rel - rep)
                fh = round(sum(l["block_hours"] for l in legs), 2)
                for m in p["crew"]:
                    out.setdefault(m["crew_id"], []).append(
                        Duty(crew_id=m["crew_id"], date=day["date"],
                             report_utc=day["report_utc"], release_utc=day["release_utc"],
                             duty_hours=dh, flight_hours=fh,
                             pairing_id=p["pairing_id"], flights=list(day["flights"]))
                    )
        for v in out.values():
            v.sort(key=lambda d: d.report_utc)
        return out

    def week_duties(self, crew_id: str) -> list[Duty]:
        return list(self._week_duties.get(crew_id, []))

    def pairing_day(self, pairing_id: str, day_date: str) -> dict | None:
        p = self.pairing_by_id.get(pairing_id)
        if not p:
            return None
        return next((d for d in p["days"] if d["date"] == day_date), None)

    def pairing_of_flight(self, flight_id: str) -> tuple[str, dict] | tuple[None, None]:
        for p in self.pairings:
            for day in p["days"]:
                if flight_id in day["flights"]:
                    return p["pairing_id"], day
        return None, None

    def duty_of(self, day: dict) -> tuple[datetime, datetime, float, int]:
        rep, rel = parse_dt(day["report_utc"]), parse_dt(day["release_utc"])
        return rep, rel, hours(rel - rep), len(day["flights"])

    def seats_on(self, flight_ids) -> int:
        return sum(self.flight_by_id[f]["seats"] for f in flight_ids if f in self.flight_by_id)

    @cached_property
    def reserve_ids(self) -> set[str]:
        return set(self.reserves)

    @cached_property
    def stations(self) -> set[str]:
        return {f["dep_station"] for f in self.flights} | {f["arr_station"] for f in self.flights}

    @cached_property
    def dates(self) -> list[str]:
        return sorted(self.flights_by_date)

    def is_reserve(self, crew_id: str) -> bool:
        return crew_id in self.reserves


_STORE: Store | None = None


def store(data_dir: Path | str | None = None) -> Store:
    """Process-wide singleton - the dataset is ~1 MB, load it once."""
    global _STORE
    if _STORE is None or data_dir is not None:
        _STORE = Store(data_dir or DATA_DIR)
    return _STORE
