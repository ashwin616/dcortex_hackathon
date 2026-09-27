"""Candidate enumeration and ranking.

Exhaustive over the crew list - 150 records, so there is no reason to be clever
about search. Every candidate is checked against all seven rules; the ones that
fail are kept with their reason, because "why not C-3305?" is a question a
controller actually asks.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from backend.domain.costing import (PILOT_RANKS, callout_cost, cancellation_cost, deadhead_cost,
                      positioning)
from backend.domain.data import fmt_dt, parse_dt
from backend.domain.models import Option
from backend.domain.rules import check_cover
from backend.domain.world import World

RULES_CHECKED = ["RULE-FDP-01", "RULE-DUTY-02", "RULE-FLT-03", "RULE-REST-04",
                 "RULE-QUAL-05", "RULE-CERT-06", "RULE-BASE-07"]


def _reserve_window_ok(s, crew_id: str, report_dt: datetime):
    """A reserve is usable when the *required report time* - after any deadhead
    positioning - falls inside their on-call window on that date."""
    r = s.reserves.get(crew_id)
    if not r:
        return True, None
    if report_dt.date().isoformat() not in r.get("dates", []):
        return False, f"not on reserve on {report_dt.date().isoformat()}"
    w = r["oncall_window_utc"]
    day = report_dt.date()
    ws = datetime.combine(day, datetime.strptime(w["start"], "%H:%M").time())
    we = datetime.combine(day, datetime.strptime(w["end"], "%H:%M").time())
    if ws <= report_dt <= we:
        return True, None
    return False, (f"reserve on-call window {w['start']}-{w['end']}Z does not cover "
                   f"required report {report_dt.strftime('%H:%M')}Z")


def cover_options(world: World, pdays: list, role: str, sick_crew_id: str | None = None,
                  exclude_pairing: str | None = None, include_cancel: bool = True,
                  unavailable: set | None = None, ignore_base: bool = False):
    """Rank every legal way to cover `pdays` in `role`.

    Ranking: cost, then delay, then disruption risk (risk_signals.json used as a
    tiebreaker between equally priced options), then crew_id for determinism.

    `unavailable` excludes crew already committed elsewhere in the same plan -
    without it, filling six roles from one candidate pool happily rosters the
    same cabin crew member three times.

    `ignore_base` prices a callout to the departure station without deadhead
    positioning. That is the convention the dataset's own recovery costs use
    for re-crewing mid-rotation; the caller is expected to say so.
    Returns (options, excluded).
    """
    s = world.s
    options: list[Option] = []
    excluded: list[dict] = []

    first_leg = s.flight_by_id[pdays[0]["flights"][0]]
    station = first_leg["dep_station"]
    first_dep = first_leg["dep_utc"]
    base_report = parse_dt(pdays[0]["report_utc"])
    pilot = role in PILOT_RANKS

    for cid, c in s.crew.items():
        if cid == sick_crew_id or c["rank"] != role or c.get("status") != "active":
            continue
        if unavailable and cid in unavailable:
            continue

        delay_h, dh_break, deadhead_ok, dh_flight = 0.0, {}, True, None
        if c["base"] != station and not ignore_base:
            pos = positioning(s, c["base"], station, pdays[0]["date"], first_dep)
            if not pos:
                excluded.append({"crew_id": cid, "is_reserve": s.is_reserve(cid),
                                 "reason": f"RULE-BASE-07: no same-day positioning flight from {c['base']}"})
                continue
            dh_flight, new_report, delay_h = pos
            deadhead_ok = True

        report_required = base_report + timedelta(hours=delay_h)
        if s.is_reserve(cid):
            ok, why = _reserve_window_ok(s, cid, report_required)
            if not ok:
                excluded.append({"crew_id": cid, "is_reserve": True, "reason": why})
                continue

        v = check_cover(world, cid, pdays, exclude_pairing=exclude_pairing,
                        delay_h=delay_h, deadhead_ok=deadhead_ok, needed_station=station)
        if not v.legal:
            excluded.append({"crew_id": cid, "is_reserve": s.is_reserve(cid),
                             "reason": "; ".join(v.issues)})
            continue

        cost, label = callout_cost(s, c["rank"], s.is_reserve(cid))
        breakdown = {label.replace(" ", "_"): cost}
        if dh_flight is not None:
            extra, dh_break = deadhead_cost(s, delay_h)
            cost += extra
            breakdown.update(dh_break)
            label += (f" + deadhead from {c['base']} on {dh_flight['flight_no']} "
                      f"(first departure delayed ~{delay_h}h)")

        risk = (s.risk.get(cid) or {}).get("disruption_risk_score")
        reach = c.get("reachability_minutes")
        why = [f"{c['base']}-based" if c["base"] == station else f"{c['base']}-based, positions to {station}",
               f"{'/'.join(c['ratings'])} rated"]
        if s.is_reserve(cid):
            w = s.reserves[cid]["oncall_window_utc"]
            why.append(f"on call {w['start']}-{w['end']}Z, report {report_required.strftime('%H:%M')}Z inside window")
        else:
            why.append("not rostered on these days")
        if reach is not None:
            why.append(f"reachable in {reach} min")
        if risk is not None:
            why.append(f"disruption risk {risk}")

        options.append(Option(
            action=f"Assign {c['rank']} {cid} ({label})",
            crew_id=cid, legal=True, cost_inr=int(round(cost)), delay_hours=delay_h,
            rules_checked=list(RULES_CHECKED), reasoning="; ".join(why),
            cost_breakdown=breakdown, risk_score=risk,
        ))

    options.sort(key=lambda o: (o.cost_inr, o.delay_hours,
                                o.risk_score if o.risk_score is not None else 1.0, o.crew_id))
    excluded.sort(key=_exclusion_interest)

    if include_cancel:
        n = sum(len(d["flights"]) for d in pdays)
        seats = sum(s.seats_on(d["flights"]) for d in pdays)
        options.append(Option(
            action=f"Cancel all {n} flights of the pairing", crew_id=None, legal=True,
            cost_inr=cancellation_cost(s, n), delay_hours=0.0, rules_checked=[],
            reasoning=f"Always available and always legal, but strands {seats} seats "
                      f"across {n} legs - the benchmark every other option is measured against.",
            cost_breakdown={"cancellation_per_flight": int(s.costs["cancellation_per_flight"]),
                            "flights": n},
        ))

    for i, o in enumerate(options):
        o.rank = i + 1
    return options, excluded


def _exclusion_interest(item: dict) -> tuple:
    """Order rejected candidates by how much a controller cares.

    Sorted by crew id, the list is 100 people who simply lack the type rating -
    true, and useless. The question behind "why not?" is almost always about
    somebody who was nearly available: a reserve whose window missed by an
    hour, or a line pilot an hour short on rest. Those come first.
    """
    reason = item.get("reason") or ""
    cid = item.get("crew_id") or ""
    # A reserve is who the desk reaches for first, so a reserve ruled out is
    # more interesting than a line pilot ruled out for the same reason.
    reserve_first = 0 if item.get("is_reserve") else 1
    if "on-call window" in reason:
        rank = 0                      # a reserve who was close
    elif "RULE-DUTY-02" in reason or "RULE-FLT-03" in reason:
        rank = 1                      # ran out of hours
    elif "RULE-REST-04" in reason or "double-booked" in reason:
        rank = 2                      # already working
    elif "RULE-CERT-06" in reason:
        rank = 3
    elif "RULE-BASE-07" in reason:
        rank = 4
    else:
        rank = 5                      # no rating - the bulk, and the least useful
    return (rank, reserve_first, cid)


def cover_role_on_pairing(world: World, pairing_id: str, crew_id: str,
                          from_date: str | None = None):
    """Convenience: the days and role that must be covered when `crew_id` drops
    off `pairing_id`."""
    s = world.s
    p = s.pairing_by_id[pairing_id]
    role = s.roles_in[pairing_id].get(crew_id, "Captain")
    days = [d for d in p["days"] if not from_date or d["date"] >= from_date]
    return days, role


# ==========================================================================
# joint allocation across simultaneous openings
# ==========================================================================
def allocate_jointly(world: World, openings: list, top_k: int = 8):
    """Cover several openings at once without using the same person twice.

    Solving each opening independently and reporting the cheapest for both is
    the failure this exists to remove: when one reserve is the cheapest legal
    answer to two simultaneous sick calls, the per-opening answer assigns them
    to both aircraft at the same time. That plan is cheap, legal on paper, and
    physically impossible.

    `openings` is a list of dicts: {label, pdays, role, sick_crew_id,
    exclude_pairing}. Exhaustive over the top `top_k` candidates per opening
    with cancellation as the always-available fallback - a handful of
    simultaneous openings is what a desk actually faces, and at that size an
    exact search is cheaper to write and to explain than a heuristic.
    """
    from itertools import product

    per: list = []
    for op in openings:
        opts, exc = cover_options(
            world, op["pdays"], op["role"], sick_crew_id=op.get("sick_crew_id"),
            exclude_pairing=op.get("exclude_pairing"), include_cancel=True)
        legal = [o for o in opts if o.crew_id][:top_k]
        cancel = next(o for o in opts if o.crew_id is None)
        per.append({"opening": op, "options": legal, "cancel": cancel,
                    "excluded": exc, "all_options": opts})

    best = None
    for combo in product(*[[*p["options"], p["cancel"]] for p in per]):
        used = [o.crew_id for o in combo if o.crew_id]
        if len(used) != len(set(used)):
            continue                                # same person twice
        total = sum(o.cost_inr for o in combo)
        # prefer fewer cancellations at equal cost, then lower total risk
        cancels = sum(1 for o in combo if o.crew_id is None)
        risk = sum(o.risk_score or 0.0 for o in combo)
        key = (total, cancels, risk)
        if best is None or key < best[0]:
            best = (key, combo)

    if best is None:                                # cannot happen: cancel is always legal
        return None, per

    (total, cancels, risk), combo = best
    assignment = []
    for p, chosen in zip(per, combo):
        op = p["opening"]
        cheapest = p["options"][0] if p["options"] else p["cancel"]
        forced = (chosen.crew_id != cheapest.crew_id)
        assignment.append({
            "opening": op.get("label") or op.get("exclude_pairing"),
            "pairing_id": op.get("exclude_pairing"),
            "role": op["role"],
            "dates": [d["date"] for d in op["pdays"]],
            "vacated_by": op.get("sick_crew_id"),
            "assigned": chosen.to_dict(),
            "contended": forced,
            "note": (f"{cheapest.crew_id} is cheaper here but is taken by another "
                     f"opening in this plan" if forced and cheapest.crew_id else None),
            "alternatives": [o.to_dict() for o in p["options"][:5]],
            "excluded_count": len(p["excluded"]),
        })

    contested = _contested_crew(per)
    return {
        "total_cost_inr": total,
        "cancellations": cancels,
        "assignments": assignment,
        "contested_crew": contested,
        "note": ("Assignments are mutually exclusive: no crew member appears twice. "
                 "Where the per-opening cheapest option was taken by another "
                 "opening, the next legal option was used and the difference is "
                 "the real cost of the contention."),
    }, per


def _contested_crew(per: list) -> list:
    """Crew who are a legal option for more than one opening - the scarce ones."""
    seen: dict = {}
    for i, p in enumerate(per):
        for o in p["options"]:
            seen.setdefault(o.crew_id, []).append(i)
    return [{"crew_id": c, "wanted_by_openings": len(idx)}
            for c, idx in sorted(seen.items()) if len(idx) > 1]
