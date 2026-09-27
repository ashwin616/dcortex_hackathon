"""Planner and duty-manager questions.

The first tool set was built around the disruption a controller is handling
right now. These answer the other half of the desk's work - the questions asked
before anything has gone wrong: where is the plan thin, who is carrying the
load, what does an option cost, and what do the rules actually permit.

Same envelope as every other tool: data, the records consulted, and any rule
arithmetic, so the grounding and completeness passes work on them unchanged.
"""
from __future__ import annotations

from collections import Counter
from datetime import date

from backend.domain.costing import inr
from backend.domain.data import parse_dt, store
from backend.domain.duty import fdp_limit
from backend.domain.models import ToolResult
from backend.domain.rules import (MAX_DUTY_7D, MAX_FLIGHT_28D, MIN_REST_H,
                                  fdp_01)
from backend.domain.world import World
from backend.tools.common import _ev, _s


def cost_reference() -> ToolResult:
    """The callout, deadhead, delay and cancellation rates, with the deltas.

    A duty manager comparing options wants the difference between calling a
    reserve and calling someone in on a day off, not to be told to look it up.
    """
    s = _s()
    c = s.costs
    delta_pilot = c["dayoff_callout_pilot"] - c["reserve_callout_pilot"]
    delta_cabin = c["dayoff_callout_cabin"] - c["reserve_callout_cabin"]
    return ToolResult(
        tool="cost_reference",
        data={
            "currency": c.get("currency", "INR"),
            "reserve_callout_pilot": c["reserve_callout_pilot"],
            "dayoff_callout_pilot": c["dayoff_callout_pilot"],
            "pilot_delta": delta_pilot,
            "reserve_callout_cabin": c["reserve_callout_cabin"],
            "dayoff_callout_cabin": c["dayoff_callout_cabin"],
            "cabin_delta": delta_cabin,
            "deadhead_positioning": c["deadhead_positioning"],
            "delay_cost_per_duty_hour": c["delay_cost_per_duty_hour"],
            "cancellation_per_flight": c["cancellation_per_flight"],
            "hotel_overnight": c["hotel_overnight"],
            "summary": (f"A pilot on a day off costs {inr(delta_pilot)} more than a "
                        f"reserve; cabin crew {inr(delta_cabin)} more."),
            "notes": c.get("notes"),
        },
        evidence=["costs.json#rates"])


def rules_reference(sectors: int | None = None) -> ToolResult:
    """The seven rules and their parameters, with the FDP ladder worked out.

    `sectors` takes one number or several - "four, five or six sectors" is one
    question, not three.
    """
    s = _s()
    ladder = {n: fdp_limit(n) for n in range(1, 9)}
    asked = sectors if isinstance(sectors, list) else \
        ([sectors] if sectors is not None else [])
    asked_ladder = {}
    for x in asked:
        try:
            n = int(x)
        except (TypeError, ValueError):
            continue
        asked_ladder[n] = fdp_limit(n)
    return ToolResult(
        tool="rules_reference",
        data={
            "rules": s.rules["rules"],
            "time_convention": s.rules["time_convention"],
            "definitions": s.rules["definitions"],
            "fdp_ladder_hours_by_sectors": ladder,
            "fdp_for_sectors": asked_ladder or None,
            "max_duty_hours_7d": MAX_DUTY_7D,
            "max_flight_hours_28d": MAX_FLIGHT_28D,
            "min_rest_hours": MIN_REST_H,
        },
        evidence=[f"rules.json#{r['rule_id']}" for r in s.rules["rules"]])


def pairings_overnight(on_or_after: str | None = None) -> ToolResult:
    """Pairings that night-stop away from base, and where.

    An overnight is where a single sick call costs two days instead of one, so
    the planner wants them listed before the week starts.
    """
    s = _s()
    rows = []
    for p in s.pairings:
        if len(p["days"]) < 2:
            continue
        dates = [d["date"] for d in p["days"]]
        if on_or_after and dates[-1] < on_or_after:
            continue
        last_leg = s.flight_by_id[p["days"][0]["flights"][-1]]
        if last_leg["arr_station"] == "BLR":
            continue
        rows.append({
            "pairing_id": p["pairing_id"], "aircraft": p["aircraft"],
            "dates": dates, "overnight_station": last_leg["arr_station"],
            "crew": [m["crew_id"] for m in p["crew"]],
            "seats_exposed": sum(s.seats_on(d["flights"]) for d in p["days"]),
        })
    rows.sort(key=lambda r: (r["dates"][0], r["pairing_id"]))
    return ToolResult(tool="pairings_overnight",
                      data={"count": len(rows), "pairings": rows,
                            "note": "A sick call on one of these costs both days."},
                      evidence=_ev("rosters.json", [r["pairing_id"] for r in rows]))


def roster_load(top: int = 10, base: str | None = None) -> ToolResult:
    """Who is working the most this week - duty days, hours and pairings.

    Concentration risk: if one person holds four of the week's duties, they are
    a single point of failure regardless of how legal every duty is.
    """
    s = _s()
    w = World(s)
    rows = []
    for cid, c in s.crew.items():
        if base and c["base"].upper() != base.upper():
            continue
        duties = w.duties(cid)
        if not duties:
            continue
        rows.append({
            "crew_id": cid, "rank": c["rank"], "base": c["base"],
            "duty_days": len({d.date for d in duties}),
            "duty_hours_week": round(sum(d.duty_hours for d in duties), 2),
            "flight_hours_week": round(sum(d.flight_hours for d in duties), 2),
            "pairings": sorted({d.pairing_id for d in duties}),
            "dates": sorted({d.date for d in duties}),
            "risk": (s.risk.get(cid) or {}).get("disruption_risk_score"),
        })
    rows.sort(key=lambda r: (-r["duty_days"], -r["duty_hours_week"], r["crew_id"]))
    busiest = rows[0]["duty_days"] if rows else 0
    tied = sum(1 for r in rows if r["duty_days"] == busiest)
    shown = rows[:top]
    # "Who is working the most days?" cut to ten looked like a ranking when it
    # was a slice of a 30-way tie - and reading the top of a tie as a ranking is
    # how a controller ends up "protecting" an arbitrary ten people. The tie is
    # the answer, so it is stated.
    note = (f"{tied} crew tie at {busiest} duty days - showing the {len(shown)} "
            f"with the most hours; duty days alone does not separate them"
            if tied > len(shown) else None)
    return ToolResult(tool="roster_load",
                      data={"count": len(rows), "crew": shown,
                            "busiest_duty_days": busiest,
                            "crew_tied_at_busiest": tied},
                      evidence=_ev("rosters.json", sorted(
                          {p for r in shown for p in r["pairings"]})),
                      note=note)


def fdp_headroom(max_slack_hours: float = 1.0, on_date: str | None = None) -> ToolResult:
    """Rostered duties running closest to their FDP ceiling.

    These are the duties where an hour of delay turns into a re-crew, so they
    are the ones to watch when weather is forecast.
    """
    s = _s()
    rows = []
    for p in s.pairings:
        for day in p["days"]:
            if on_date and day["date"] != on_date:
                continue
            _rep, _rel, dh, n = s.duty_of(day)
            limit = fdp_limit(n)
            slack = round(limit - dh, 2)
            if slack > max_slack_hours:
                continue
            rows.append({
                "pairing_id": p["pairing_id"], "aircraft": p["aircraft"],
                "date": day["date"], "sectors": n, "duty_hours": dh,
                "fdp_limit": limit, "slack_hours": slack,
                "report_utc": day["report_utc"], "release_utc": day["release_utc"],
                "crew": [m["crew_id"] for m in p["crew"]],
            })
    rows.sort(key=lambda r: (r["slack_hours"], r["date"]))
    return ToolResult(
        tool="fdp_headroom",
        data={"max_slack_hours": max_slack_hours, "count": len(rows), "duties": rows,
              "note": "Slack is how much delay the duty absorbs before RULE-FDP-01 bites."},
        evidence=_ev("rosters.json", sorted({r["pairing_id"] for r in rows})))


def pairing_headroom(pairing_id: str, on_date: str | None = None) -> ToolResult:
    """Duty-clock headroom for every crew member rostered on a pairing.

    "Can this crew absorb a delay?" is a per-person question - the pairing is
    only as flexible as whoever on it has the least room left.
    """
    s = _s()
    w = World(s)
    pid = pairing_id.upper()
    p = s.pairing_by_id.get(pid)
    if not p:
        return ToolResult(tool="pairing_headroom",
                          data={"error": f"unknown pairing {pairing_id}"}, evidence=[])
    days = [d for d in p["days"] if not on_date or d["date"] == on_date]
    assumed = None
    if not days:
        # "day 1" and "the first day" arrive as the snapshot date more often
        # than as the pairing's actual first date. Answering about day one beats
        # refusing over an off-by-one the controller did not make.
        days = p["days"]
        assumed = (f"{on_date} is not a duty date for {pid}; answering for "
                   f"{days[0]['date']}, its first day")

    day = days[0]
    d = date.fromisoformat(day["date"])
    _rep, _rel, dh, n = s.duty_of(day)
    rows, rules = [], []
    for m in p["crew"]:
        cid = m["crew_id"]
        d7 = w.window_hours(cid, d, 7, "duty")
        f28 = w.window_hours(cid, d, 28, "flight")
        # Two readings of "headroom", and a controller means different ones on
        # different days: how much room is left once they have flown this duty,
        # or how much they had walking into it. Reporting one figure under an
        # ambiguous name answered the other question half the time, so both are
        # named. The narrator is forbidden from computing differences, which
        # means anything it should be able to say has to be here already.
        before = round(d7 - dh, 2)
        rows.append({
            "crew_id": cid, "role": m["role"], "rank": s.crew[cid]["rank"],
            "duty_hours_7d_before_this_duty": before,
            "duty_headroom_7d_before_this_duty": round(MAX_DUTY_7D - before, 2),
            "duty_hours_7d_including_this_duty": d7,
            "duty_headroom_7d_after_this_duty": round(MAX_DUTY_7D - d7, 2),
            "duty_headroom_7d": round(MAX_DUTY_7D - d7, 2),
            "flight_hours_28d": f28,
            "flight_headroom_28d": round(MAX_FLIGHT_28D - f28, 2),
        })
    rows.sort(key=lambda r: r["duty_headroom_7d"])
    r_fdp = fdp_01(n, dh, day["date"])
    rules.append(r_fdp.to_dict())
    return ToolResult(
        tool="pairing_headroom",
        data={"pairing_id": pid, "date": day["date"], "sectors": n,
              "assumption": assumed,
              "duty_hours": dh, "fdp_limit": r_fdp.limit,
              "fdp_slack_hours": round(r_fdp.limit - dh, 2),
              "crew": rows,
              "tightest": rows[0] if rows else None,
              "note": "The pairing is only as flexible as its tightest clock."},
        evidence=_ev("rosters.json", pid)
                 + _ev("duty_clocks.json", [r["crew_id"] for r in rows]),
        rules_evaluated=rules)


def aircraft_day(aircraft: str, on_date: str) -> ToolResult:
    """Everything one tail flies on one day, with seats and cancellation exposure."""
    s = _s()
    legs = [f for f in s.flights_by_date.get(on_date, [])
            if f["aircraft"].upper() == aircraft.upper()]
    legs.sort(key=lambda f: f["dep_utc"])
    pid = None
    for p in s.pairings:
        if p["aircraft"].upper() == aircraft.upper() and \
                any(d["date"] == on_date for d in p["days"]):
            pid = p["pairing_id"]
            break
    seats = sum(f["seats"] for f in legs)
    return ToolResult(
        tool="aircraft_day",
        data={"aircraft": aircraft.upper(), "date": on_date, "pairing_id": pid,
              "legs": len(legs),
              "flights": [{"flight_id": f["flight_id"], "flight_no": f["flight_no"],
                           "route": f"{f['dep_station']}-{f['arr_station']}",
                           "dep_utc": f["dep_utc"], "seats": f["seats"],
                           "block_hours": f["block_hours"]} for f in legs],
              "flight_ids": [f["flight_id"] for f in legs],
              "seats": seats,
              "block_hours": round(sum(f["block_hours"] for f in legs), 2),
              "cancellation_cost_inr": int(s.costs["cancellation_per_flight"]) * len(legs),
              "crew": [m["crew_id"] for m in s.pairing_by_id[pid]["crew"]] if pid else []},
        evidence=_ev("flights.json", [f["flight_id"] for f in legs])
                 + (_ev("rosters.json", pid) if pid else []))


def operating_snapshot() -> ToolResult:
    """When "now" is, and what the dataset covers.

    The system knew the snapshot all along - it is in the planner prompt - and
    still refused "what is today's date?", because no operation could return
    it. Anything the advisor reasons from should be answerable *about*, not
    just usable silently.
    """
    from datetime import timedelta

    s = _s()
    now = s.snapshot
    today = now.date()
    return ToolResult(
        tool="operating_snapshot",
        data={
            "now_utc": s.snapshot_utc,
            "today": today.isoformat(),
            "today_weekday": today.strftime("%A"),
            "time_utc": s.snapshot_utc[11:16],
            "yesterday": (today - timedelta(days=1)).isoformat(),
            "tomorrow": (today + timedelta(days=1)).isoformat(),
            "schedule_week": {"first": s.dates[0], "last": s.dates[-1],
                              "days": len(s.dates)},
            "hub": "BLR",
            "stations": sorted(s.stations),
            "timezone": "UTC (all times in this system)",
            "currency": "INR",
            "counts": {"flights": len(s.flights), "crew": len(s.crew),
                       "pairings": len(s.pairings)},
            "note": ("This is a fixed operating snapshot, not wall-clock time - "
                     "the dataset is a planned week frozen at that instant."),
        },
        evidence=["duty_clocks.json#as_of", "flights.json#week"])
