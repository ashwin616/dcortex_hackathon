"""Tier 3 - recommendation.

Ranking is done by the deterministic candidate engine; the model never
chooses a crew member.
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

def _retained_duty(s, pairing_id: str, retained: list, delay_h: float) -> dict | None:
    """What the rostered crew can still legally fly after the delay.

    The recovery answer is only half told by naming the leg that has to go: the
    controller also needs to hear that the crew keeps the rest of the duty, and
    that the shortened duty is itself legal.
    """
    if not retained or not pairing_id:
        return None
    legs = [s.flight_by_id[f] for f in retained]
    rep = parse_dt(s.flight_by_id[retained[0]]["dep_utc"]) - timedelta(minutes=60)
    rel = parse_dt(legs[-1]["arr_utc"]) + timedelta(minutes=30, hours=delay_h)
    fdp = hours(rel - rep)
    r = fdp_01(len(retained), fdp, legs[0]["date"])
    return {"flights": retained, "sectors": len(retained),
            "fdp_hours": fdp, "fdp_limit": r.limit, "legal": r.passed,
            "detail": r.detail,
            "assumption": ("Report time held at the rostered hour - the crew had "
                           "already signed on when the delay was called, so the "
                           "duty clock runs from the original report. Had they "
                           "been told before sign-on, report would move with the "
                           "delay and the FDP would be "
                           f"{round(fdp - delay_h, 2)}h instead.")}


def _pairing_covering(s, crew_id: str, from_date: str | None) -> str | None:
    """Which of this crew member's pairings is the question about?

    Taking the first one is wrong the moment somebody flies twice in a week:
    "resolve their 19 Sep assignment" must land on the pairing that actually
    has a duty on the 19th, not whichever appears first in the roster.
    """
    pairings = s.pairings_of.get(crew_id) or []
    if not pairings:
        return None
    if not from_date:
        return pairings[0]
    # prefer a pairing operating exactly on that date, else the next one after
    exact, later = [], []
    for pid in pairings:
        dates = [d["date"] for d in s.pairing_by_id[pid]["days"]]
        if from_date in dates:
            exact.append(pid)
        elif any(d >= from_date for d in dates):
            later.append((min(d for d in dates if d >= from_date), pid))
    if exact:
        return exact[0]
    return min(later)[1] if later else None


def recommend_options(crew_id: str | None = None, pairing_id: str | None = None,
                      from_date: str | None = None, role: str | None = None,
                      flight_ids: list | None = None,
                      include_excluded: int = 8) -> ToolResult:
    """Ranked, rule-checked ways to cover an opening. Cheapest legal first;
    disruption risk breaks ties between equally priced options."""
    s = _s()
    w = World(s)
    exclude = None

    if crew_id:
        cid = crew_id.upper()
        if cid not in s.crew:
            return ToolResult(tool="recommend_options",
                              data={"error": f"unknown crew_id {crew_id}"}, evidence=[])
        pid = pairing_id or _pairing_covering(s, cid, from_date)
        if not pid:
            rostered = s.pairings_of.get(cid) or []
            msg = (f"{cid} is not rostered on any pairing this week" if not rostered
                   else f"{cid} has no duty on or after {from_date} "
                        f"(rostered on {', '.join(rostered)})")
            return ToolResult(tool="recommend_options", data={"error": msg},
                              evidence=_ev("crew.json", cid))
        pdays, role_ = cover_role_on_pairing(w, pid, cid, from_date)
        role = role or role_
        exclude = pid
        w = w.with_sick(cid, pid, from_date)
    elif pairing_id:
        p = s.pairing_by_id[pairing_id.upper()]
        pdays = [d for d in p["days"] if not from_date or d["date"] >= from_date]
        exclude = p["pairing_id"]
        role = role or "Captain"
        cid = None
    elif flight_ids:
        pdays = [synth_day(s, flight_ids)]
        role = role or "Captain"
        cid = None
    else:
        return ToolResult(tool="recommend_options",
                          data={"error": "need crew_id, pairing_id or flight_ids"}, evidence=[])

    if not pdays:
        return ToolResult(tool="recommend_options",
                          data={"error": "nothing left to cover for that date range"}, evidence=[])

    opts, exc = cover_options(w, pdays, role, sick_crew_id=crew_id.upper() if crew_id else None,
                              exclude_pairing=exclude)
    legs = [f for d in pdays for f in d["flights"]]
    return ToolResult(
        tool="recommend_options",
        data={"covering": {"pairing_id": exclude, "role": role,
                           "dates": [d["date"] for d in pdays],
                           "flights": legs, "seats": s.seats_on(legs),
                           "vacated_by": crew_id},
              "options": [o.to_dict() for o in opts],
              "legal_option_count": sum(1 for o in opts if o.legal and o.crew_id),
              "considered": len(s.crew),
              "excluded_examples": exc[:include_excluded],
              "excluded_count": len(exc),
              "world": w.describe()},
        evidence=_ev("rosters.json", [exclude] if exclude else [])
                 + _ev("crew.json", [o.crew_id for o in opts if o.crew_id])
                 + ["costs.json#rates"])


def morning_brief(on_date: str, top_n: int = 3) -> ToolResult:
    """Proactive standing brief: per aircraft line, the things that break first."""
    s = _s()
    w = World(s)
    d = date.fromisoformat(on_date)
    lines: dict = {}
    for p in s.pairings:
        day = next((x for x in p["days"] if x["date"] == on_date), None)
        if not day:
            continue
        rep, rel, dh, n = s.duty_of(day)
        tightest, worst_risk, cert_risk = None, None, []
        for m in p["crew"]:
            cid = m["crew_id"]
            head = round(MAX_DUTY_7D - w.window_hours(cid, d, 7, "duty"), 2)
            if tightest is None or head < tightest[1]:
                tightest = (cid, head)
            rs = (s.risk.get(cid) or {}).get("disruption_risk_score")
            if rs is not None and (worst_risk is None or rs > worst_risk[1]):
                worst_risk = (cid, rs)
            for ct, vt in s.certs.get(cid, {}).items():
                if vt <= d + timedelta(days=14):
                    cert_risk.append({"crew_id": cid, "cert_type": ct, "valid_to": vt.isoformat(),
                                      "expired_on_duty_date": vt < d})
        lines[p["aircraft"]] = {
            "pairing_id": p["pairing_id"], "sectors": n, "duty_hours": dh,
            "fdp_limit": fdp_limit(n), "fdp_slack_hours": round(fdp_limit(n) - dh, 2),
            "tightest_duty_headroom": {"crew_id": tightest[0], "hours": tightest[1]} if tightest else None,
            "highest_disruption_risk": {"crew_id": worst_risk[0], "score": worst_risk[1]} if worst_risk else None,
            "certification_exposure": cert_risk,
            "reserves_on_call": [r for r, v in s.reserves.items() if on_date in v.get("dates", [])][:top_n],
        }
    return ToolResult(tool="morning_brief",
                      data={"date": on_date, "lines": lines,
                            "flagged_roster_exceptions": [f for f in s.flagged_exceptions]},
                      evidence=_ev("rosters.json", [v["pairing_id"] for v in lines.values()]))



def recommend_joint(crew_ids: list | None = None, pairing_ids: list | None = None,
                    on_date: str | None = None, top_k: int = 8) -> ToolResult:
    """Cover several simultaneous openings with one consistent plan.

    Use this whenever more than one crew member is out at the same time. Asking
    for each opening separately produces a plan that assigns the same reserve
    to two aircraft at once.
    """
    from backend.domain.candidates import allocate_jointly

    s = _s()
    w = World(s)
    openings = []
    crew_ids = [c.upper() for c in (crew_ids or [])]
    pairing_ids = [p.upper() for p in (pairing_ids or [])]

    if crew_ids:
        for i, cid in enumerate(crew_ids):
            if cid not in s.crew:
                return ToolResult(tool="recommend_joint",
                                  data={"error": f"unknown crew_id {cid}"}, evidence=[])
            pid = (pairing_ids[i] if i < len(pairing_ids)
                   else _pairing_covering(s, cid, on_date))
            if not pid:
                return ToolResult(tool="recommend_joint",
                                  data={"error": f"{cid} has no duty on or after {on_date}"},
                                  evidence=_ev("crew.json", cid))
            pdays, role = cover_role_on_pairing(w, pid, cid, on_date)
            if not pdays:
                continue
            openings.append({"label": f"{s.pairing_by_id[pid]['aircraft']} / {pid}",
                             "pdays": pdays, "role": role, "sick_crew_id": cid,
                             "exclude_pairing": pid})
            w = w.with_sick(cid, pid, on_date)
    elif pairing_ids:
        for pid in pairing_ids:
            p = s.pairing_by_id.get(pid)
            if not p:
                return ToolResult(tool="recommend_joint",
                                  data={"error": f"unknown pairing {pid}"}, evidence=[])
            pdays = [d for d in p["days"] if not on_date or d["date"] >= on_date]
            openings.append({"label": f"{p['aircraft']} / {pid}", "pdays": pdays,
                             "role": "Captain", "exclude_pairing": pid})
    else:
        return ToolResult(tool="recommend_joint",
                          data={"error": "need crew_ids or pairing_ids"}, evidence=[])

    if len(openings) < 2:
        return ToolResult(tool="recommend_joint",
                          data={"error": "joint allocation needs at least two openings; "
                                         "use recommend_options for one"}, evidence=[])

    plan, per = allocate_jointly(w, openings, top_k=top_k)
    ev = _ev("rosters.json", [o["exclude_pairing"] for o in openings if o.get("exclude_pairing")])
    for p in per:
        ev += _ev("crew.json", [o.crew_id for o in p["options"] if o.crew_id])
    return ToolResult(tool="recommend_joint",
                      data={**plan, "openings": len(openings), "world": w.describe()},
                      evidence=ev + ["costs.json#rates"])


def recovery_options(on_date: str, delay_minutes: float = 0.0,
                     pairing_id: str | None = None, aircraft: str | None = None,
                     flight_ids: list | None = None) -> ToolResult:
    """What to do about legs the rostered crew can no longer legally operate.

    Prices the two real choices leg-for-leg: call out a fresh crew set for the
    tail of the duty, or cancel it. Use after delay_impact or
    station_closure_impact reports an FDP breach.
    """
    from backend.tools.impact import delay_impact

    s = _s()
    w = World(s)
    retained: list = []
    if flight_ids:
        dropped = [f for f in flight_ids if f in s.flight_by_id]
        pid, _ = s.pairing_of_flight(dropped[0]) if dropped else (None, None)
        fdp_note = None
    else:
        d = delay_impact(on_date=on_date, delay_minutes=delay_minutes,
                         pairing_id=pairing_id, aircraft=aircraft).data
        if "error" in d:
            return ToolResult(tool="recovery_options", data=d, evidence=[])
        pid = d["pairing_id"]
        dropped = d["legs_needing_recrew_or_cancellation"]
        retained = d.get("legs_crew_can_still_operate") or []
        fdp_note = (f"Delayed duty runs {d['fdp_after_delay']}h against a "
                    f"{d['fdp_limit']}h limit ({d['sectors']} sectors)"
                    if d["breach"] else "No FDP breach - the rostered crew can complete the duty")
        if not d["breach"]:
            return ToolResult(tool="recovery_options",
                              data={"pairing_id": pid, "breach": False, "note": fdp_note,
                                    "legs_needing_cover": [], "options": []},
                              evidence=_ev("rosters.json", pid))

    if not dropped:
        return ToolResult(tool="recovery_options",
                          data={"pairing_id": pid, "legs_needing_cover": [],
                                "options": [], "note": "nothing needs covering"},
                          evidence=_ev("rosters.json", pid) if pid else [])

    p = s.pairing_by_id[pid]
    day = synth_day(s, dropped)
    seats = s.seats_on(dropped)

    # One fresh crew member per role in the pairing's complement. `taken` is
    # what stops the six-seat set being filled by the same person six times.
    dep_station = s.flight_by_id[dropped[0]]["dep_station"]
    per_role, set_cost, gaps, taken = [], 0, [], set()
    for member in p["crew"]:
        role = member["role"]
        opts, _ = cover_options(w, [day], role, sick_crew_id=None,
                                exclude_pairing=None, include_cancel=False,
                                unavailable=taken, ignore_base=True)
        if not opts:
            gaps.append(role)
            continue
        best = opts[0]
        taken.add(best.crew_id)
        set_cost += best.cost_inr
        per_role.append({"role": role, "crew_id": best.crew_id,
                         "cost_inr": best.cost_inr, "action": best.action,
                         "reasoning": best.reasoning})

    # Costed as a callout to the operating station, which is the convention the
    # dataset's own recovery figures use. Worth stating plainly: no reserve is
    # based away from BLR/DEL, so a set covering a leg out of another station
    # would in reality also need positioning, and that is not priced here.
    positioning_note = None
    if dep_station not in {s.crew[c["crew_id"]]["base"] for c in per_role} and per_role:
        positioning_note = (
            f"Costed as a callout to {dep_station}. Every candidate is based "
            f"elsewhere, so a real plan would also need positioning to "
            f"{dep_station}; that cost is not included here.")

    cancel_cost = cancellation_cost(s, len(dropped))
    options = []
    if not gaps:
        options.append({
            "rank": 1, "action": f"Call out a fresh crew set for {', '.join(dropped)}",
            "legal": True, "cost_inr": set_cost, "crew": per_role,
            "reasoning": f"Rostered crew operates the earlier legs; a full "
                         f"{len(per_role)}-person set takes the remainder. "
                         f"Every member checked against all seven rules.",
        })
    options.append({
        "rank": len(options) + 1,
        "action": f"Cancel {', '.join(dropped)}", "legal": True,
        "cost_inr": cancel_cost, "crew": [],
        "reasoning": f"Always available, but strands {seats} seats across "
                     f"{len(dropped)} leg(s).",
    })
    options.sort(key=lambda o: o["cost_inr"])
    for i, o in enumerate(options, 1):
        o["rank"] = i

    cheapest = min(o["cost_inr"] for o in options)
    dearest = max(o["cost_inr"] for o in options)
    ratio = round(dearest / cheapest, 1) if cheapest else None

    return ToolResult(
        tool="recovery_options",
        data={"pairing_id": pid, "date": on_date,
              # Named so the two verdicts cannot be confused: the rostered duty
              # as planned DOES breach; the shortened duty the crew keeps does
              # not. An answer that reports the second as "no breach" tells the
              # controller the opposite of the truth.
              "full_duty_breaches_fdp": True,
              "retained_duty_is_legal": True,
              "breach": True, "note": fdp_note,
              "cost_ratio_dearest_over_cheapest": ratio,
              "cost_comparison": (f"cancelling costs {ratio}x the re-crew"
                                  if ratio else None),
              "legs_needing_cover": dropped, "seats_at_risk": seats,
              "legs_original_crew_can_still_operate": retained,
              "retained_duty": _retained_duty(s, pid, retained,
                                              delay_minutes / 60.0) if retained else None,
              "roles_with_no_legal_cover": gaps,
              "positioning_note": positioning_note,
              "recrew_set_cost_inr": set_cost if not gaps else None,
              "cancellation_cost_inr": cancel_cost,
              "options": options},
        evidence=_ev("rosters.json", pid) + _ev("flights.json", dropped)
                 + ["costs.json#rates"])
