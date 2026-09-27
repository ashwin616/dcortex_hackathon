"""The seven rules, as pure predicates that return their arithmetic.

Each returns a RuleResult carrying limit / actual / margin, so the explanation
shown to a controller is *derived* from the numbers rather than narrated by a
language model. A rule that passes still reports its headroom.
"""
from __future__ import annotations

from datetime import date, timedelta

from backend.domain.data import hm, hours, parse_dt
from backend.domain.duty import duty_len, fdp_limit
from backend.domain.models import RuleResult, Verdict
from backend.domain.world import World

ALL_RULES = ["RULE-FDP-01", "RULE-DUTY-02", "RULE-FLT-03", "RULE-REST-04",
             "RULE-QUAL-05", "RULE-CERT-06", "RULE-BASE-07"]

EPS = 1e-6
MAX_DUTY_7D = 60.0
MAX_FLIGHT_28D = 100.0
MIN_REST_H = 12.0


# --------------------------------------------------------------------------
# individual rules
# --------------------------------------------------------------------------
def fdp_01(sectors: int, fdp_hours: float, on_date: str | None = None) -> RuleResult:
    lim = fdp_limit(sectors)
    margin = round(lim - fdp_hours, 2)
    ok = fdp_hours <= lim + EPS
    tail = f"{hm(margin)} headroom" if ok else f"exceeds by {hm(-margin)}"
    return RuleResult(
        rule_id="RULE-FDP-01", passed=ok, limit=lim, actual=fdp_hours, margin=margin,
        on_date=on_date,
        detail=f"FDP {fdp_hours}h vs {lim}h limit ({sectors} sectors): {tail}",
    )


def duty_02(total_7d: float, on_date: str) -> RuleResult:
    margin = round(MAX_DUTY_7D - total_7d, 2)
    ok = total_7d <= MAX_DUTY_7D + EPS
    d = date.fromisoformat(on_date)
    win = f"{(d - timedelta(days=6)).isoformat()}..{on_date}"
    tail = f"{hm(margin)} headroom" if ok else f"exceeds by {hm(-margin)}"
    return RuleResult(
        rule_id="RULE-DUTY-02", passed=ok, limit=MAX_DUTY_7D, actual=total_7d, margin=margin,
        on_date=on_date, window=win,
        detail=f"{total_7d}h duty in 7 days ending {on_date} vs 60h limit: {tail}",
    )


def flt_03(total_28d: float, on_date: str) -> RuleResult:
    margin = round(MAX_FLIGHT_28D - total_28d, 2)
    ok = total_28d <= MAX_FLIGHT_28D + EPS
    d = date.fromisoformat(on_date)
    win = f"{(d - timedelta(days=27)).isoformat()}..{on_date}"
    tail = f"{hm(margin)} headroom" if ok else f"exceeds by {hm(-margin)}"
    return RuleResult(
        rule_id="RULE-FLT-03", passed=ok, limit=MAX_FLIGHT_28D, actual=total_28d, margin=margin,
        on_date=on_date, window=win,
        detail=f"{total_28d}h block in 28 days ending {on_date} vs 100h limit: {tail}",
    )


def rest_04(rest_h: float, before_label: str, on_date: str) -> RuleResult:
    margin = round(rest_h - MIN_REST_H, 2)
    ok = rest_h >= MIN_REST_H - EPS
    tail = "ok" if ok else f"short by {hm(-margin)}"
    return RuleResult(
        rule_id="RULE-REST-04", passed=ok, limit=MIN_REST_H, actual=rest_h, margin=margin,
        on_date=on_date,
        detail=f"{rest_h}h rest before {before_label} on {on_date} vs 12h minimum: {tail}",
    )


def qual_05(crew: dict, aircraft_type: str) -> RuleResult:
    ok = aircraft_type in crew.get("ratings", [])
    held = ", ".join(crew.get("ratings") or ["none"])
    detail = (f"{crew['crew_id']} holds {aircraft_type} rating" if ok
              else f"no {aircraft_type} rating (holds {held})")
    return RuleResult(rule_id="RULE-QUAL-05", passed=ok, detail=detail)


def cert_06(certs: dict, on_date: str) -> RuleResult:
    d = date.fromisoformat(on_date)
    bad = {k: v for k, v in certs.items() if v < d}
    ok = not bad
    if ok:
        nearest = min(certs.values()) if certs else None
        detail = f"all certifications valid on {on_date}"
        if nearest:
            detail += f" (earliest expiry {nearest.isoformat()})"
    else:
        listed = ", ".join(f"{k} ({v.isoformat()})" for k, v in sorted(bad.items()))
        detail = f"expired on {on_date}: {listed}"
    return RuleResult(rule_id="RULE-CERT-06", passed=ok, on_date=on_date, detail=detail)


def base_07(crew: dict, needed_station: str, deadhead_ok: bool, delay_h: float) -> RuleResult:
    same_base = crew["base"] == needed_station
    ok = same_base or deadhead_ok
    if same_base:
        detail = f"based at {needed_station}, no positioning needed"
    elif deadhead_ok:
        detail = (f"based at {crew['base']}; deadhead positioning to {needed_station} required "
                  f"(first departure delayed ~{delay_h}h, cost applies)")
    else:
        detail = f"based at {crew['base']}; no same-day positioning flight to {needed_station}"
    return RuleResult(rule_id="RULE-BASE-07", passed=ok, actual=delay_h, detail=detail)


# --------------------------------------------------------------------------
# composite: can this crew member cover these pairing-days?
# --------------------------------------------------------------------------
def check_cover(world: World, crew_id: str, pdays: list, exclude_pairing: str | None = None,
                delay_h: float = 0.0, deadhead_ok: bool = True,
                needed_station: str | None = None) -> Verdict:
    """Evaluate all seven rules for `crew_id` taking on `pdays`.

    A candidate must be legal on *every* day of the cover - the 7-day duty
    window is re-evaluated for each day, which is where multi-day pairings
    usually fail.
    """
    s = world.s
    c = s.crew[crew_id]
    rules = []
    issues = []

    first_leg = s.flight_by_id[pdays[0]["flights"][0]]
    actype = first_leg["aircraft_type"]
    needed_station = needed_station or first_leg["dep_station"]

    # RULE-QUAL-05 - hard gate, nothing else matters without the rating
    r = qual_05(c, actype)
    rules.append(r)
    if not r.passed:
        return Verdict(crew_id=crew_id, legal=False, rules=rules,
                       issues=[f"RULE-QUAL-05: no {actype} rating"])

    # RULE-BASE-07
    rb = base_07(c, needed_station, deadhead_ok, delay_h)
    rules.append(rb)
    if not rb.passed:
        issues.append(f"RULE-BASE-07: no same-day positioning flight from {c['base']}")

    # per-day: RULE-CERT-06 and RULE-FDP-01
    for day in pdays:
        rc = cert_06(s.certs.get(crew_id, {}), day["date"])
        rules.append(rc)
        if not rc.passed:
            issues.append(f"RULE-CERT-06: certification invalid on {day['date']}")

        _, _, fdp = duty_len(day, delay_h)
        rf = fdp_01(len(day["flights"]), fdp, day["date"])
        rules.append(rf)
        if not rf.passed:
            issues.append(f"RULE-FDP-01: FDP {fdp}h > {rf.limit}h limit "
                          f"({len(day['flights'])} sectors)")

    # RULE-REST-04 and double-booking, over the simulated timeline
    sim = [(parse_dt(d.report_utc), parse_dt(d.release_utc), d.pairing_id, d.date)
           for d in world.duties(crew_id, exclude_pairing=exclude_pairing)]
    for day in pdays:
        rep, rel, _ = duty_len(day, delay_h)
        sim.append((rep, rel, "COVER", day["date"]))
    sim.sort(key=lambda x: x[0])
    for a, b in zip(sim, sim[1:]):
        rest = hours(b[0] - a[1])
        rr = rest_04(rest, b[2], b[3])
        if a[2] == "COVER" or b[2] == "COVER":
            rules.append(rr)
        if not rr.passed:
            tag = "downstream" if b[2] != "COVER" and a[2] == "COVER" else "rest"
            issues.append(f"RULE-REST-04: only {rest}h rest before {b[2]} on {b[3]} ({tag} conflict)")
        if b[0] < a[1]:
            issues.append(f"double-booked: {a[2]} overlaps {b[2]} on {b[3]}")

    # RULE-DUTY-02 / RULE-FLT-03 - re-checked on every day of the cover
    for day in pdays:
        d = date.fromisoformat(day["date"])
        base7 = world.window_hours(crew_id, d, 7, "duty", exclude_pairing=exclude_pairing)
        base28 = world.window_hours(crew_id, d, 28, "flight", exclude_pairing=exclude_pairing)
        add_d = sum(duty_len(x, delay_h)[2] for x in pdays
                    if date.fromisoformat(x["date"]) <= d)
        add_f = sum(round(sum(s.flight_by_id[f]["block_hours"] for f in x["flights"]), 2)
                    for x in pdays if date.fromisoformat(x["date"]) <= d)
        tot7 = round(base7 + add_d, 2)
        tot28 = round(base28 + add_f, 2)

        rd = duty_02(tot7, day["date"])
        rules.append(rd)
        if not rd.passed:
            issues.append(f"RULE-DUTY-02: would exceed 60h/7d by {hm(-rd.margin)} "
                          f"on {day['date']} (total {tot7}h)")

        rl = flt_03(tot28, day["date"])
        rules.append(rl)
        if not rl.passed:
            issues.append(f"RULE-FLT-03: would exceed 100h/28d by {hm(-rl.margin)} "
                          f"on {day['date']} (total {tot28}h)")

    return Verdict(crew_id=crew_id, legal=not issues, rules=rules, issues=issues)
