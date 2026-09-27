"""Tier 1 - lookup and retrieval.

Answerable straight from the indexed data. No rule evaluation.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from backend.domain.candidates import cover_options, cover_role_on_pairing
from backend.domain.costing import callout_cost, cancellation_cost, inr, positioning
from backend.domain.data import fmt_dt, hm, hours, parse_dt, store
from backend.domain.duty import extend_duty, fdp_limit, synth_day
from backend.domain.models import ToolResult
from backend.domain.rules import (MAX_DUTY_7D, MAX_FLIGHT_28D, MIN_REST_H, check_cover,
                                  duty_02, fdp_01, flt_03, rest_04)
from backend.domain.world import World
from backend.tools.common import _clock_summary, _ev, _s

def find_crew(crew_id: str | None = None, rank: str | None = None, base: str | None = None,
              rating: str | None = None, status: str | None = None,
              name: str | None = None, limit: int = 60) -> ToolResult:
    """Look up crew by id, or filter the roster by rank / base / rating / status.

    `status` accepts several - "leave, training" answers "who is unavailable".
    """
    s = _s()
    rows = []
    for cid, c in s.crew.items():
        if crew_id and cid.upper() != crew_id.upper():
            continue
        if rank and c["rank"].lower() != rank.lower():
            continue
        if base and c["base"].upper() != base.upper():
            continue
        if rating and rating.upper() not in [r.upper() for r in c["ratings"]]:
            continue
        if status:
            # "on leave or in training" is one question about two statuses.
            wanted = status if isinstance(status, list) else \
                [x.strip() for x in str(status).replace("/", ",").split(",")]
            if c.get("status", "active").lower() not in {w.lower() for w in wanted if w}:
                continue
        if name and name.lower() not in c["name"].lower():
            continue
        row = dict(c)
        row["is_reserve"] = s.is_reserve(cid)
        if s.is_reserve(cid):
            row["oncall_window_utc"] = s.reserves[cid]["oncall_window_utc"]
        row["pairings"] = s.pairings_of.get(cid, [])
        if crew_id:
            row["duty_clock"] = _clock_summary(cid)
            row["certifications"] = {k: v.isoformat() for k, v in sorted(s.certs.get(cid, {}).items())}
            row["risk"] = s.risk.get(cid)
        rows.append(row)
    rows.sort(key=lambda r: r["crew_id"])
    ev = _ev("crew.json", [r["crew_id"] for r in rows[:limit]])
    return ToolResult(tool="find_crew", data={"count": len(rows), "crew": rows[:limit]},
                      evidence=ev,
                      note=None if len(rows) <= limit else f"{len(rows)} matched, showing {limit}")


def duty_clock(crew_id: str, on_date: str | None = None) -> ToolResult:
    """Duty and flight-hour accruals for one crew member, with headroom under
    RULE-DUTY-02 and RULE-FLT-03. Calendar-day windows, inclusive of `on_date`."""
    s = _s()
    cid = crew_id.upper()
    if cid not in s.crew:
        return ToolResult(tool="duty_clock", data={"error": f"unknown crew_id {crew_id}"}, evidence=[])
    summary = _clock_summary(cid, on_date)
    w = World(s)
    d = date.fromisoformat(summary["window_end"])
    rules = [duty_02(summary["duty_hours_7d"], d.isoformat()).to_dict(),
             flt_03(summary["flight_hours_28d"], d.isoformat()).to_dict()]
    hist = [h for h in s.clocks[cid].get("daily_history", []) if h["duty_hours"] > 0][-10:]
    duties = [x.to_dict() for x in w.duties(cid)]
    return ToolResult(
        tool="duty_clock",
        data={**summary, "rank": s.crew[cid]["rank"], "base": s.crew[cid]["base"],
              "recent_duty_days": hist, "planned_duties": duties},
        evidence=_ev("duty_clocks.json", cid) + _ev("rosters.json", s.pairings_of.get(cid, [])),
        rules_evaluated=rules)


def list_reserves(on_date: str, base: str | None = None, rank: str | None = None,
                  covers_time_utc: str | None = None,
                  through_date: str | None = None) -> ToolResult:
    """Reserves on call for a date, or across a range.

    `covers_time_utc` ("HH:MM" or full ISO) filters to those whose on-call
    window contains that report time. `through_date` makes it a range, and any
    day of that range the dataset does not hold is reported rather than
    quietly dropped - answering a three-day question with one day's data and
    no warning is how a controller ends up short-crewed.
    """
    s = _s()
    if through_date and through_date != on_date:
        return _reserves_over_range(on_date, through_date, base, rank, covers_time_utc)
    t = None
    if covers_time_utc:
        t = covers_time_utc[11:16] if len(covers_time_utc) > 10 else covers_time_utc
    rows = []
    for cid, r in s.reserves.items():
        if on_date not in r.get("dates", []):
            continue
        c = s.crew[cid]
        if base and c["base"].upper() != base.upper():
            continue
        if rank and c["rank"].lower() != rank.lower():
            continue
        w = r["oncall_window_utc"]
        if t and not (w["start"] <= t <= w["end"]):
            continue
        rows.append({"crew_id": cid, "name": c["name"], "rank": c["rank"], "base": c["base"],
                     "ratings": c["ratings"], "window": w,
                     "reachability_minutes": c["reachability_minutes"],
                     "status": c.get("status")})
    rows.sort(key=lambda r: (r["rank"], r["crew_id"]))
    outside = on_date < s.dates[0] or on_date > s.dates[-1]
    note = (f"{on_date} is outside the dataset, which covers "
            f"{s.dates[0]} to {s.dates[-1]}." if outside else None)
    return ToolResult(tool="list_reserves",
                      data={"date": on_date, "covers_time_utc": t,
                            "count": len(rows), "reserves": rows,
                            "outside_dataset": outside, "note": note},
                      evidence=_ev("reserve_pool.json", [r["crew_id"] for r in rows]),
                      note=note)


def _reserves_over_range(start: str, end: str, base, rank, covers_time_utc) -> ToolResult:
    """Per-day reserve cover across a date range, saying which days are missing."""
    from datetime import date as _date, timedelta as _td

    s = _s()
    d0, d1 = sorted([start, end])
    days, missing = [], []
    cur = _date.fromisoformat(d0)
    last = _date.fromisoformat(d1)
    while cur <= last:
        iso = cur.isoformat()
        if iso in s.flights_by_date or any(iso in r.get("dates", [])
                                           for r in s.reserve_list):
            one = list_reserves(iso, base=base, rank=rank,
                                covers_time_utc=covers_time_utc)
            days.append({"date": iso, "count": one.data["count"],
                         "reserves": one.data["reserves"]})
        else:
            missing.append(iso)
        cur += _td(days=1)

    everyone = sorted({r["crew_id"] for day in days for r in day["reserves"]})
    note = None
    if missing:
        note = (f"No data for {', '.join(missing)} - the dataset covers "
                f"{s.dates[0]} to {s.dates[-1]}, so those days are outside it.")
    return ToolResult(
        tool="list_reserves",
        data={"range": [d0, d1], "days_covered": [d["date"] for d in days],
              "dates_outside_dataset": missing, "by_date": days,
              "crew_on_reserve_at_some_point": everyone,
              "count": len(everyone), "note": note},
        evidence=_ev("reserve_pool.json", everyone),
        note=note)


def list_flights(on_date: str | None = None, dep_station: str | None = None,
                 arr_station: str | None = None, flight_no: str | None = None,
                 aircraft: str | None = None, after_utc: str | None = None,
                 before_utc: str | None = None, limit: int = 60) -> ToolResult:
    """Filter the schedule. Times are UTC "HH:MM"."""
    s = _s()
    pool = s.flights_by_date.get(on_date, []) if on_date else s.flights
    rows = []
    for f in pool:
        if dep_station and f["dep_station"].upper() != dep_station.upper():
            continue
        if arr_station and f["arr_station"].upper() != arr_station.upper():
            continue
        if flight_no and f["flight_no"].upper() != flight_no.upper():
            continue
        if aircraft and f["aircraft"].upper() != aircraft.upper():
            continue
        t = f["dep_utc"][11:16]
        if after_utc and t < after_utc[-5:]:
            continue
        if before_utc and t > before_utc[-5:]:
            continue
        rows.append(f)
    rows.sort(key=lambda f: f["dep_utc"])
    out = []
    for f in rows[:limit]:
        pid, _ = s.pairing_of_flight(f["flight_id"])
        out.append({**f, "pairing_id": pid})
    outside = bool(on_date) and (on_date < s.dates[0] or on_date > s.dates[-1])
    note = (f"{on_date} is outside the dataset, which covers {s.dates[0]} to "
            f"{s.dates[-1]}." if outside else
            (None if len(rows) <= limit else f"{len(rows)} matched, showing {limit}"))
    return ToolResult(tool="list_flights",
                      data={"count": len(rows), "flights": out,
                            "outside_dataset": outside, "note": note},
                      evidence=_ev("flights.json", [f["flight_id"] for f in out]),
                      note=note)


def certifications_expiring(as_of: str, within_days: int = 30,
                            crew_id: str | None = None) -> ToolResult:
    """Certifications expiring within N days of a date (RULE-CERT-06 exposure)."""
    s = _s()
    d0 = date.fromisoformat(as_of)
    cutoff = d0 + timedelta(days=within_days)
    rows = []
    for cid, certs in s.certs.items():
        if crew_id and cid.upper() != crew_id.upper():
            continue
        for ct, valid_to in certs.items():
            if valid_to <= cutoff:
                rows.append({"crew_id": cid, "name": s.crew[cid]["name"], "rank": s.crew[cid]["rank"],
                             "cert_type": ct, "valid_to": valid_to.isoformat(),
                             "days_left": (valid_to - d0).days,
                             "already_expired": valid_to < d0,
                             "rostered_dates": [x.date for x in s.week_duties(cid)]})
    rows.sort(key=lambda r: (r["valid_to"], r["crew_id"]))
    return ToolResult(tool="certifications_expiring",
                      data={"as_of": as_of, "within_days": within_days,
                            "count": len(rows), "certifications": rows},
                      evidence=_ev("certifications.json", [f"{r['crew_id']}:{r['cert_type']}" for r in rows]))


def get_pairing(pairing_id: str | None = None, flight_id: str | None = None,
                crew_id: str | None = None, on_date: str | None = None,
                aircraft: str | None = None) -> ToolResult:
    """A pairing with its days, legs, report/release and full crew complement.
    Findable by pairing id, by a flight on it, by a crew member, or by tail+date."""
    s = _s()
    pids = []
    if pairing_id:
        pids = [pairing_id.upper()]
    elif flight_id:
        pid, _ = s.pairing_of_flight(flight_id)
        pids = [pid] if pid else []
    elif crew_id:
        pids = s.pairings_of.get(crew_id.upper(), [])
    elif aircraft:
        pids = [p["pairing_id"] for p in s.pairings if p["aircraft"].upper() == aircraft.upper()
                and (not on_date or any(d["date"] == on_date for d in p["days"]))]
    elif on_date:
        pids = [p["pairing_id"] for p in s.pairings if any(d["date"] == on_date for d in p["days"])]

    out = []
    for pid in pids:
        p = s.pairing_by_id.get(pid)
        if not p:
            continue
        days = []
        for day in p["days"]:
            if on_date and day["date"] != on_date and (pairing_id or crew_id):
                pass  # keep all days for an explicit pairing/crew lookup
            legs = [s.flight_by_id[f] for f in day["flights"]]
            rep, rel, dh, n = s.duty_of(day)
            days.append({**day, "duty_hours": dh, "sectors": n,
                         "fdp_limit": fdp_limit(n),
                         "block_hours": round(sum(l["block_hours"] for l in legs), 2),
                         "seats": sum(l["seats"] for l in legs),
                         "legs": [{"flight_id": l["flight_id"], "flight_no": l["flight_no"],
                                   "route": f"{l['dep_station']}-{l['arr_station']}",
                                   "dep_utc": l["dep_utc"], "arr_utc": l["arr_utc"],
                                   "seats": l["seats"]} for l in legs]})
        out.append({"pairing_id": pid, "aircraft": p["aircraft"],
                    "aircraft_type": s.flight_by_id[p["days"][0]["flights"][0]]["aircraft_type"],
                    "days": days,
                    "crew": [{**m, "name": s.crew[m["crew_id"]]["name"],
                              "base": s.crew[m["crew_id"]]["base"]} for m in p["crew"]]})
    return ToolResult(tool="get_pairing", data={"count": len(out), "pairings": out},
                      evidence=_ev("rosters.json", [o["pairing_id"] for o in out]))


def schedule_stats(metric: str = "overview", on_date: str | None = None,
                   station: str | None = None) -> ToolResult:
    """Aggregates over the schedule: 'overview', 'daily_counts', 'longest_block',
    'destinations' (nonstop from `station`), 'largest_leg'. With `on_date` it
    also returns block hours and legs per aircraft for that day."""
    s = _s()
    data: dict = {}
    ev: list = []
    if metric in ("overview", "daily_counts"):
        data["flights_per_date"] = {d: len(v) for d, v in sorted(s.flights_by_date.items())}
        data["total_flights"] = len(s.flights)
        data["stations"] = sorted(s.stations)
        data["aircraft"] = sorted({f["aircraft"] for f in s.flights})
    if metric in ("overview", "longest_block"):
        mx = max(f["block_hours"] for f in s.flights)
        legs = [f for f in s.flights if f["block_hours"] == mx]
        data["longest_block_hours"] = mx
        data["longest_block_flights"] = sorted({f["flight_no"] for f in legs})
        data["longest_block_route"] = f"{legs[0]['dep_station']}-{legs[0]['arr_station']}"
        ev += _ev("flights.json", [f["flight_id"] for f in legs[:6]])
    if metric in ("overview", "largest_leg"):
        mx = max(f["seats"] for f in s.flights)
        legs = [f for f in s.flights if f["seats"] == mx]
        data["max_seats_per_leg"] = mx
        data["max_seats_aircraft_type"] = legs[0]["aircraft_type"]
        data["max_seats_leg_count"] = len(legs)
        data["seats_by_aircraft_type"] = {f["aircraft_type"]: f["seats"] for f in s.flights}
    if metric in ("overview", "destinations") and station:
        dests = sorted({f["arr_station"] for f in s.flights if f["dep_station"] == station.upper()})
        data["nonstop_from"] = station.upper()
        data["destinations"] = dests
    if on_date:
        day = s.flights_by_date.get(on_date, [])
        data["on_date"] = on_date
        data["flights_on_date"] = len(day)
        data["seats_on_date"] = sum(f["seats"] for f in day)
        data["block_hours_on_date"] = round(sum(f["block_hours"] for f in day), 2)
        legs_by_tail: dict = {}
        for f in day:
            legs_by_tail[f["aircraft"]] = legs_by_tail.get(f["aircraft"], 0) + 1
        data["legs_by_aircraft"] = dict(sorted(legs_by_tail.items()))
    return ToolResult(tool="schedule_stats", data=data, evidence=ev or ["flights.json#all"])


def risk_signal(crew_id: str | None = None, min_score: float | None = None,
                limit: int = 20) -> ToolResult:
    """Pre-computed disruption-risk scores (a provided input, not a prediction)."""
    s = _s()
    rows = []
    for cid, r in s.risk.items():
        if crew_id and cid.upper() != crew_id.upper():
            continue
        if min_score is not None and r["disruption_risk_score"] < min_score:
            continue
        rows.append({**r, "rank": s.crew[cid]["rank"], "base": s.crew[cid]["base"],
                     "pairings": s.pairings_of.get(cid, [])})
    rows.sort(key=lambda r: -r["disruption_risk_score"])
    return ToolResult(tool="risk_signal", data={"count": len(rows), "signals": rows[:limit]},
                      evidence=_ev("risk_signals.json", [r["crew_id"] for r in rows[:limit]]),
                      note="Provided pre-computed input; the Advisor does not predict risk.")

