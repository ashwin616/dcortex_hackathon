"""Tier 2 - consequence and simulation.

These run the kernel against a disruption overlay, so a what-if is an
actual simulation rather than a guess about one.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from backend.domain.candidates import cover_options, cover_role_on_pairing
from backend.domain.costing import (callout_cost, cancellation_cost,
                                    deadhead_cost, inr, positioning)
from backend.domain.data import fmt_dt, hm, hours, parse_dt, store
from backend.domain.duty import extend_duty, fdp_limit, synth_day
from backend.domain.models import ToolResult
from backend.domain.rules import (MAX_DUTY_7D, MAX_FLIGHT_28D, MIN_REST_H, check_cover,
                                  duty_02, fdp_01, flt_03, rest_04)
from backend.domain.world import World
from backend.tools.common import _clock_summary, _ev, _s

def sick_call_impact(crew_id: str, from_date: str | None = None,
                     pairing_id: str | None = None,
                     reported_utc: str | None = None) -> ToolResult:
    """Which flights lose a crew member, and what else in the pairing is at risk."""
    s = _s()
    cid = crew_id.upper()
    if cid not in s.crew:
        return ToolResult(tool="sick_call_impact", data={"error": f"unknown crew_id {crew_id}"}, evidence=[])
    w = World(s).with_sick(cid, pairing_id, from_date, reported_utc)
    pids = [pairing_id] if pairing_id else s.pairings_of.get(cid, [])
    role = None
    days_out = []
    immediate, downstream, seats_imm = [], [], 0
    for pid in pids:
        p = s.pairing_by_id.get(pid)
        if not p:
            continue
        role = s.roles_in[pid].get(cid, role)
        for i, day in enumerate(p["days"]):
            if from_date and day["date"] < from_date:
                continue
            rep, rel, dh, n = s.duty_of(day)
            entry = {"pairing_id": pid, "date": day["date"], "flights": day["flights"],
                     "seats": s.seats_on(day["flights"]), "duty_hours": dh, "sectors": n,
                     "report_utc": day["report_utc"], "release_utc": day["release_utc"],
                     "overnight_station": s.flight_by_id[day["flights"][-1]]["arr_station"]}
            days_out.append(entry)
            if not immediate:
                immediate = list(day["flights"])
                seats_imm = entry["seats"]
            else:
                downstream += list(day["flights"])
    return ToolResult(
        tool="sick_call_impact",
        data={"crew_id": cid, "name": s.crew[cid]["name"], "rank": s.crew[cid]["rank"],
              "role_on_pairing": role, "pairings_affected": pids,
              "reported_utc": reported_utc,
              "uncrewed_now": immediate, "passengers_affected_now": seats_imm,
              "also_at_risk": downstream,
              "passengers_at_risk_total": sum(d["seats"] for d in days_out),
              "days": days_out,
              "remaining_complement": [m for m in s.pairing_by_id[pids[0]]["crew"]
                                       if m["crew_id"] != cid] if pids else [],
              "world": w.describe()},
        evidence=_ev("crew.json", cid) + _ev("rosters.json", pids)
                 + _ev("flights.json", immediate + downstream))


def check_assignment(crew_id: str, pairing_id: str | None = None,
                     flight_ids: list | None = None, on_date: str | None = None,
                     delay_hours: float = 0.0, replacing: str | None = None) -> ToolResult:
    """Would assigning this crew member to this pairing / these legs breach a rule?

    Returns every rule with its limit, actual value and margin - the numbers the
    explanation is built from.
    """
    s = _s()
    cid = crew_id.upper()
    if cid not in s.crew:
        return ToolResult(tool="check_assignment", data={"error": f"unknown crew_id {crew_id}"}, evidence=[])

    if pairing_id:
        p = s.pairing_by_id.get(pairing_id.upper())
        if not p:
            return ToolResult(tool="check_assignment",
                              data={"error": f"unknown pairing {pairing_id}"}, evidence=[])
        pdays = [d for d in p["days"] if not on_date or d["date"] >= on_date]
        exclude = p["pairing_id"]
    elif flight_ids:
        missing = [f for f in flight_ids if f not in s.flight_by_id]
        if missing:
            return ToolResult(tool="check_assignment",
                              data={"error": f"unknown flight ids: {missing}"}, evidence=[])
        pdays = [synth_day(s, flight_ids)]
        exclude = None
    else:
        return ToolResult(tool="check_assignment",
                          data={"error": "need pairing_id or flight_ids"}, evidence=[])

    w = World(s)
    if replacing:
        w = w.with_sick(replacing.upper(), pairing_id, on_date)

    # An out-of-base crew member cannot simply appear at the gate. Checking the
    # rules while ignoring how they get there answers "is it legal" and misses
    # "what does it cost you" - which is the half a controller acts on.
    station = s.flight_by_id[pdays[0]["flights"][0]]["dep_station"]
    positioning_plan = None
    if s.crew[cid]["base"] != station and not delay_hours:
        pos = positioning(s, s.crew[cid]["base"], station, pdays[0]["date"],
                          s.flight_by_id[pdays[0]["flights"][0]]["dep_utc"])
        if pos:
            leg, new_report, delay_hours = pos
            extra, breakdown = deadhead_cost(s, delay_hours)
            callout, label = callout_cost(s, s.crew[cid]["rank"], s.is_reserve(cid))
            positioning_plan = {
                "required": True, "from_base": s.crew[cid]["base"], "to": station,
                "deadhead_flight": leg["flight_no"], "arrives_utc": leg["arr_utc"],
                "new_report_utc": fmt_dt(new_report),
                "first_departure_delayed_hours": delay_hours,
                "cost_inr": callout + extra,
                "cost_breakdown": {label.replace(" ", "_"): callout, **breakdown},
            }
        else:
            positioning_plan = {"required": True, "from_base": s.crew[cid]["base"],
                                "to": station, "deadhead_flight": None,
                                "note": "no same-day positioning flight"}

    v = check_cover(w, cid, pdays, exclude_pairing=exclude, delay_h=delay_hours,
                    deadhead_ok=bool(positioning_plan is None
                                     or positioning_plan.get("deadhead_flight")),
                    needed_station=station)
    return ToolResult(
        tool="check_assignment",
        data={"crew_id": cid, "rank": s.crew[cid]["rank"], "base": s.crew[cid]["base"],
              "target": pairing_id or flight_ids, "delay_hours": delay_hours,
              "legal": v.legal, "issues": v.issues,
              "rules_checked": v.rules_checked,
              "positioning": positioning_plan,
              "days": [{"date": d["date"], "sectors": len(d["flights"]),
                        "flights": d["flights"]} for d in pdays]},
        evidence=_ev("crew.json", cid) + _ev("duty_clocks.json", cid)
                 + (_ev("rosters.json", pairing_id) if pairing_id else [])
                 + _ev("flights.json", [f for d in pdays for f in d["flights"]]),
        rules_evaluated=[r.to_dict() for r in v.rules])


def station_closure_impact(station: str, on_date: str, start_utc: str, end_utc: str,
                           turnaround_minutes: int = 30) -> ToolResult:
    """Flights caught by a station closure, and whether the rostered crew can
    still legally operate once the flight is delayed to the reopening.

    Each hit flight is pushed to reopen + `turnaround_minutes`; the duty's
    release moves with it and RULE-FDP-01 is re-evaluated for that crew.
    """
    s = _s()
    w = World(s).with_closure(station.upper(), on_date, start_utc[-5:], end_utc[-5:])
    hits = w.closure_hits()
    reopen = parse_dt(f"{on_date}T{end_utc[-5:]}:00Z") + timedelta(minutes=turnaround_minutes)

    pairings: dict = {}
    per_flight, rules_eval = [], []
    for h in hits:
        pid, day = s.pairing_of_flight(h["flight_id"])
        if pid:
            pairings.setdefault(pid, []).append(h["flight_id"])
        if not day:
            continue
        rep, rel, _, n = s.duty_of(day)
        shift = round(hours(reopen - parse_dt(h["sched_utc"])), 2)
        new_fdp = hours((rel + timedelta(hours=shift)) - rep)
        r = fdp_01(n, new_fdp, on_date)
        rules_eval.append(r.to_dict())
        per_flight.append({
            "flight_id": h["flight_id"], "flight_no": h["flight_no"], "pairing_id": pid,
            "hit_on": h["reason"], "seats": h["seats"],
            "min_delay_hours": shift, "crew_fdp_after_delay": new_fdp,
            "fdp_limit": r.limit, "crew_legal_after_delay": r.passed,
            "action": ("delay (crew legal)" if r.passed else
                       "delay exceeds crew FDP - re-crew tail legs from reserves or cancel"),
        })
    per_flight.sort(key=lambda x: (not x["crew_legal_after_delay"], x["flight_id"]))
    crew_affected = sorted({c for pid in pairings for c in s.roles_in[pid]})
    breaches = [p for p in per_flight if not p["crew_legal_after_delay"]]
    return ToolResult(
        tool="station_closure_impact",
        data={"station": station.upper(), "date": on_date,
              "window_utc": f"{start_utc[-5:]}-{end_utc[-5:]}",
              "reopen_plus_turnaround_utc": fmt_dt(reopen),
              "affected_flight_count": len(hits),
              "affected_flights": [h["flight_id"] for h in hits],
              "per_flight_assessment": per_flight,
              "flights_needing_recrew_or_cancel": [b["flight_id"] for b in breaches],
              "seats_affected": sum(h["seats"] for h in hits),
              "pairings_affected": pairings,
              "crew_affected_count": len(crew_affected),
              "crew_affected": crew_affected,
              "cancellation_cost_if_all_cancelled": cancellation_cost(s, len(hits)),
              "world": w.describe()},
        evidence=_ev("flights.json", [h["flight_id"] for h in hits])
                 + _ev("rosters.json", list(pairings)),
        rules_evaluated=rules_eval)


def delay_impact(on_date: str, delay_minutes: float, pairing_id: str | None = None,
                 aircraft: str | None = None, flight_no: str | None = None) -> ToolResult:
    """A delay before the first departure: does the rostered crew's FDP still fit?

    The crew has already reported, so release moves out and the FDP lengthens.
    """
    s = _s()
    pid = pairing_id
    if not pid:
        for p in s.pairings:
            if aircraft and p["aircraft"].upper() != aircraft.upper():
                continue
            for day in p["days"]:
                if day["date"] != on_date:
                    continue
                if flight_no and not any(s.flight_by_id[f]["flight_no"].upper() == flight_no.upper()
                                         for f in day["flights"]):
                    continue
                pid = p["pairing_id"]
                break
            if pid:
                break
    if not pid or pid not in s.pairing_by_id:
        return ToolResult(tool="delay_impact",
                          data={"error": "no pairing matched", "date": on_date,
                                "aircraft": aircraft, "flight_no": flight_no}, evidence=[])

    p = s.pairing_by_id[pid]
    day = next(d for d in p["days"] if d["date"] == on_date)
    dh = delay_minutes / 60.0
    rep, rel, fdp_after = extend_duty(day, dh)
    _, _, fdp_before, n = s.duty_of(day)
    r_after = fdp_01(n, fdp_after, on_date)

    # how many legs can they legally still operate?
    legal_prefix, legs = [], [s.flight_by_id[f] for f in day["flights"]]
    for i in range(1, len(legs) + 1):
        sub = legs[:i]
        sub_fdp = hours((parse_dt(sub[-1]["arr_utc"]) + timedelta(minutes=30, hours=dh))
                        - parse_dt(day["report_utc"]))
        if sub_fdp <= fdp_limit(i) + 1e-6:
            legal_prefix = [l["flight_id"] for l in sub]
    dropped = [f for f in day["flights"] if f not in legal_prefix]

    return ToolResult(
        tool="delay_impact",
        data={"pairing_id": pid, "aircraft": p["aircraft"], "date": on_date,
              "delay_minutes": delay_minutes, "sectors": n,
              "fdp_before": fdp_before, "fdp_after_delay": fdp_after,
              "fdp_limit": r_after.limit, "breach": not r_after.passed,
              "over_by_hours": None if r_after.passed else round(-r_after.margin, 2),
              "legs_crew_can_still_operate": legal_prefix,
              "legs_needing_recrew_or_cancellation": dropped,
              "seats_on_dropped_legs": s.seats_on(dropped),
              "crew": p["crew"]},
        evidence=_ev("rosters.json", pid) + _ev("flights.json", day["flights"]),
        rules_evaluated=[r_after.to_dict()])


def earliest_next_report(release_utc: str) -> ToolResult:
    """RULE-REST-04: release + 12h."""
    rel = parse_dt(release_utc if len(release_utc) > 10 else release_utc + "T00:00:00Z")
    nxt = rel + timedelta(hours=MIN_REST_H)
    r = rest_04(MIN_REST_H, "next duty", nxt.date().isoformat())
    return ToolResult(tool="earliest_next_report",
                      data={"released_utc": fmt_dt(rel), "min_rest_hours": MIN_REST_H,
                            "earliest_report_utc": fmt_dt(nxt)},
                      evidence=["rules.json#RULE-REST-04"],
                      rules_evaluated=[r.to_dict()])


def crew_near_limits(on_date: str, duty_threshold_7d: float = 45.0,
                     flight_threshold_28d: float | None = None,
                     base: str | None = None, rank_by: str = "duty",
                     top: int | None = None, limit: int = 40) -> ToolResult:
    """Crew at or above a duty/flight-hour threshold for the window ending
    `on_date`, including that day's planned duty. Also drives the morning brief."""
    s = _s()
    w = World(s)
    d = date.fromisoformat(on_date)
    rows = []
    for cid, c in s.crew.items():
        if base and c["base"].upper() != base.upper():
            continue
        d7 = w.window_hours(cid, d, 7, "duty")
        f28 = w.window_hours(cid, d, 28, "flight")
        hit = d7 >= duty_threshold_7d - 1e-9
        if flight_threshold_28d is not None:
            hit = hit or f28 >= flight_threshold_28d - 1e-9
        if top and rank_by == "flight":
            hit = True          # a top-N ranking filters by position, not threshold
        if not hit:
            continue
        rows.append({"crew_id": cid, "name": c["name"], "rank": c["rank"], "base": c["base"],
                     "duty_hours_7d": d7, "duty_headroom_7d": round(MAX_DUTY_7D - d7, 2),
                     "flight_hours_28d": f28,
                     "flight_headroom_28d": round(MAX_FLIGHT_28D - f28, 2),
                     "pairings": s.pairings_of.get(cid, []),
                     "risk": (s.risk.get(cid) or {}).get("disruption_risk_score")})
    # "closest to the 100-hour limit" is a flight-hour question; ranking it by
    # duty hours answers a different one.
    if rank_by == "flight":
        rows.sort(key=lambda r: (-r["flight_hours_28d"], r["crew_id"]))
    else:
        rows.sort(key=lambda r: (-r["duty_hours_7d"], r["crew_id"]))
    if top:
        rows = rows[:int(top)]
    return ToolResult(tool="crew_near_limits",
                      data={"window_end": on_date, "duty_threshold_7d": duty_threshold_7d,
                            "ranked_by": rank_by,
                            "flight_threshold_28d": flight_threshold_28d,
                            "count": len(rows), "crew": rows[:limit]},
                      evidence=_ev("duty_clocks.json", [r["crew_id"] for r in rows[:limit]]))


def cancellation_impact(flight_ids: list | None = None, aircraft: str | None = None,
                        on_date: str | None = None) -> ToolResult:
    """Seats and direct cost of cancelling specific legs, or a whole tail's day."""
    s = _s()
    if not flight_ids and aircraft and on_date:
        flight_ids = [f["flight_id"] for f in s.flights_by_date.get(on_date, [])
                      if f["aircraft"].upper() == aircraft.upper()]
    flight_ids = flight_ids or []
    known = [f for f in flight_ids if f in s.flight_by_id]
    unknown = [f for f in flight_ids if f not in s.flight_by_id]
    rows = [{"flight_id": f, "flight_no": s.flight_by_id[f]["flight_no"],
             "route": f"{s.flight_by_id[f]['dep_station']}-{s.flight_by_id[f]['arr_station']}",
             "seats": s.flight_by_id[f]["seats"],
             "aircraft_type": s.flight_by_id[f]["aircraft_type"]} for f in known]
    return ToolResult(tool="cancellation_impact",
                      data={"flights": rows, "unknown_flight_ids": unknown,
                            "passengers_affected": sum(r["seats"] for r in rows),
                            "rate_per_flight": int(s.costs["cancellation_per_flight"]),
                            "cost_inr": cancellation_cost(s, len(known)),
                            "cost_formatted": inr(cancellation_cost(s, len(known)))},
                      evidence=_ev("flights.json", known) + ["costs.json#cancellation_per_flight"])

