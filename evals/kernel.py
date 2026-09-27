"""Kernel evaluation - runs the deterministic layer against the shipped answer keys.

No LLM involved. If this passes, the arithmetic underneath the chat is right;
whatever the model does on top can only be a presentation problem.

    python eval.py kernel            # summary
    python eval.py kernel -v         # show every check
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from backend import tools as T
from backend.domain.data import store

VERBOSE = "-v" in sys.argv
ROOT = Path(__file__).resolve().parent.parent

PASS, FAIL = [], []


def check(name: str, got, want, note: str = ""):
    ok = got == want
    (PASS if ok else FAIL).append(name)
    if VERBOSE or not ok:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}{(' - ' + note) if note else ''}")
        if not ok:
            print(f"       got : {got}")
            print(f"       want: {want}")


def check_set(name: str, got, want, note: str = ""):
    check(name, sorted(set(got)), sorted(set(want)), note)


# ==========================================================================
# scenarios S1 - S6
# ==========================================================================
def eval_scenarios(s):
    for sc in s.scenarios:
        sid, ev, key = sc["scenario_id"], sc["event"], sc["answer_key"]

        if ev["type"] == "SICK_CREW":
            d = ev["reported_utc"][:10]
            imp = T.call("sick_call_impact", crew_id=ev["crew_id"],
                         pairing_id=ev["pairing_id"], from_date=d).data
            if "uncovered_flights" in key:
                check_set(f"{sid} uncovered flights",
                          imp["uncrewed_now"] + imp["also_at_risk"], key["uncovered_flights"])
            if "day1_uncovered" in key:
                check_set(f"{sid} day-1 uncovered", imp["uncrewed_now"], key["day1_uncovered"])
            if "passengers_day1" in key:
                check(f"{sid} passengers day 1", imp["passengers_affected_now"],
                      key["passengers_day1"])

            rec = T.call("recommend_options", crew_id=ev["crew_id"],
                         pairing_id=ev["pairing_id"], from_date=d).data
            _compare_options(sid, rec["options"], key.get("options", []))

        elif ev["type"] == "STATION_CLOSURE":
            w = ev["window_utc"]
            imp = T.call("station_closure_impact", station=ev["station"],
                         on_date=w["start"][:10], start_utc=w["start"][11:16],
                         end_utc=w["end"][11:16]).data
            check_set(f"{sid} affected flights", imp["affected_flights"], key["affected_flights"])
            got = {p["flight_id"]: (p["min_delay_hours"], p["crew_fdp_after_delay"], p["fdp_limit"])
                   for p in imp["per_flight_assessment"]}
            want = {p["flight_id"]: (p["min_delay_hours"], p["crew_fdp_after_delay"], p["fdp_limit"])
                    for p in key["per_flight_assessment"]}
            check(f"{sid} per-flight delay/FDP assessment", got, want)

        elif ev["type"] == "DELAY":
            imp = T.call("delay_impact", on_date=ev["date"], aircraft=ev["aircraft"],
                         delay_minutes=ev["delay_hours"] * 60).data
            check(f"{sid} FDP after delay", imp["fdp_after_delay"], key["fdp_after_delay"])
            check(f"{sid} FDP limit", imp["fdp_limit"], key["fdp_limit"])
            check(f"{sid} breach", imp["breach"], key["breach"])
            check(f"{sid} legs needing re-crew", imp["legs_needing_recrew_or_cancellation"],
                  ["DX404-2026-09-16"])

            rec = T.call("recovery_options", on_date=ev["date"], aircraft=ev["aircraft"],
                         delay_minutes=ev["delay_hours"] * 60).data
            want_costs = sorted(o["cost_inr"] for o in key["options"])
            check(f"{sid} recovery option costs",
                  sorted(o["cost_inr"] for o in rec["options"]), want_costs)
            check(f"{sid} cheapest recovery", rec["options"][0]["cost_inr"],
                  min(want_costs))

        elif ev["type"] == "CERT_EXPIRY":
            chk = T.call("check_assignment", crew_id=ev["crew_id"],
                         pairing_id=ev["pairing_id"],
                         on_date=key["illegal_assignment"]["date"]).data
            check(f"{sid} assignment illegal", chk["legal"], False)
            check(f"{sid} cited rule",
                  any(key["illegal_assignment"]["rule"] in i for i in chk["issues"]), True)
            rec = T.call("recommend_options", crew_id=ev["crew_id"],
                         pairing_id=ev["pairing_id"],
                         from_date=key["illegal_assignment"]["date"]).data
            _compare_options(sid, rec["options"], key.get("options", []))

        elif ev["type"] == "MULTI_SICK":
            for i, sub in enumerate(ev["events"], start=1):
                d = sub["reported_utc"][:10]
                rec = T.call("recommend_options", crew_id=sub["crew_id"],
                             pairing_id=sub["pairing_id"], from_date=d).data
                want = key.get(f"options_dx{'ab'[i-1]}") or key.get(
                    "options_dxa" if i == 1 else "options_dxb") or []
                _compare_options(f"{sid}.{i}", rec["options"], want)

            # The per-opening lists above are each correct in isolation; the
            # joint plan is what makes them a plan rather than a conflict.
            d = ev["events"][0]["reported_utc"][:10]
            joint = T.call("recommend_joint",
                           crew_ids=[e["crew_id"] for e in ev["events"]],
                           pairing_ids=[e["pairing_id"] for e in ev["events"]],
                           on_date=d).data
            assigned = [a["assigned"]["crew_id"] for a in joint["assignments"]]
            check(f"{sid} joint plan assigns distinct crew",
                  len(assigned) == len(set(assigned)), True, str(assigned))
            cheapest = [min(o["cost_inr"] for o in key[k])
                        for k in ("options_dxa", "options_dxb") if k in key]
            check(f"{sid} joint plan costs more than the naive sum",
                  joint["total_cost_inr"] > sum(cheapest), True,
                  f"joint {joint['total_cost_inr']} vs naive {sum(cheapest)}")


def _compare_options(sid, got_opts, want_opts):
    """Compare on the facts that matter: the legal candidate set, their costs,
    and the cheapest option. Equal-cost orderings are equally correct (see the
    dataset README), so rank is only enforced at rank 1.

    Some keys ship only the top few options. Those are compared as a prefix:
    every listed candidate must appear in ours at the same cost.
    """
    if not want_opts:
        return
    got = {o["crew_id"]: o["cost_inr"] for o in got_opts if o["crew_id"]}
    want = {o["crew_id"]: o["cost_inr"] for o in want_opts if o["crew_id"]}
    truncated = not any(o["crew_id"] is None for o in want_opts) and len(want) < len(got)
    if truncated:
        check(f"{sid} listed candidates present at stated cost",
              {c: got.get(c) for c in want}, want, "key lists only the top options")
    else:
        check(f"{sid} legal candidate set", sorted(got), sorted(want))
        check(f"{sid} candidate costs", got, want)
    g1 = min(got_opts, key=lambda o: (o["cost_inr"], o["crew_id"] or "zz"))
    w1 = min(want_opts, key=lambda o: (o["cost_inr"], o["crew_id"] or "zz"))
    check(f"{sid} cheapest option", (g1["crew_id"], g1["cost_inr"]),
          (w1["crew_id"], w1["cost_inr"]))
    gc = next((o["cost_inr"] for o in got_opts if o["crew_id"] is None), None)
    wc = next((o["cost_inr"] for o in want_opts if o["crew_id"] is None), None)
    if wc is not None:
        check(f"{sid} cancellation fallback cost", gc, wc)


# ==========================================================================
# questions Q01 - Q38 (the ones with a machine-comparable key)
# ==========================================================================
def eval_questions(s):
    q = {x["question_id"]: x["expected_answer"] for x in s.questions}

    # --- Tier 1 -----------------------------------------------------------
    r = T.call("list_reserves", on_date="2026-09-15", base="BLR").data
    check_set("Q01 BLR reserves", [x["crew_id"] for x in r["reserves"]],
              [x["crew_id"] for x in q["Q01"]])
    check("Q01 windows", {x["crew_id"]: x["window"] for x in r["reserves"]},
          {x["crew_id"]: x["window"] for x in q["Q01"]})

    r = T.call("duty_clock", crew_id="C-1042").data
    check("Q02 duty 7d", r["duty_hours_7d"], q["Q02"]["duty_hours_7d"])
    check("Q02 headroom", r["duty_headroom_7d"], q["Q02"]["headroom_hours"])

    r = T.call("list_flights", on_date="2026-09-15", dep_station="DEL").data
    check_set("Q03 DEL departures", [f["flight_no"] for f in r["flights"]],
              [x["flight_no"] if isinstance(x, dict) else x for x in q["Q03"]])

    r = T.call("certifications_expiring", as_of="2026-09-15", within_days=30).data
    check("Q04 expiring certifications",
          sorted((c["crew_id"], c["cert_type"], c["valid_to"]) for c in r["certifications"]),
          sorted((c["crew_id"], c["cert_type"], c["valid_to"]) for c in q["Q04"]))

    f = T.call("list_flights", on_date="2026-09-15", flight_no="DX412").data["flights"][0]
    check("Q05 aircraft/type/seats",
          {"aircraft": f["aircraft"], "aircraft_type": f["aircraft_type"], "seats": f["seats"]},
          q["Q05"])

    r = T.call("find_crew", crew_id="C-2210").data["crew"][0]
    check("Q07 base/rating", {"base": r["base"], "ratings": r["ratings"]},
          {"base": q["Q07"]["base"], "ratings": q["Q07"]["ratings"]})

    r = T.call("get_pairing", pairing_id="P-2291").data["pairings"][0]
    check("Q08 pairing crew", {m["crew_id"]: m["role"] for m in r["crew"]},
          {m["crew_id"]: m["role"] for m in q["Q08"]})

    r = T.call("schedule_stats", metric="daily_counts", on_date="2026-09-16").data
    check("Q10 flights on 16 Sep", r["flights_on_date"],
          q["Q10"] if isinstance(q["Q10"], int) else q["Q10"].get("count"))

    r = T.call("find_crew", rank="Captain", base="DEL").data
    check_set("Q11 DEL captains", [c["crew_id"] for c in r["crew"]],
              q["Q11"] if isinstance(q["Q11"], list) else q["Q11"]["crew_ids"])

    r = T.call("schedule_stats", metric="longest_block").data
    check("Q12 longest block hours", r["longest_block_hours"], q["Q12"]["block_hours"])
    check_set("Q12 longest block flights", r["longest_block_flights"], q["Q12"]["flights"])

    r = T.call("duty_clock", crew_id="C-2087").data
    check("Q13 C-2087 flight hours 28d", r["flight_hours_28d"], q["Q13"]["flight_hours_28d"])
    check("Q13 C-2087 rank", r["rank"], q["Q13"]["rank"])

    r = T.call("schedule_stats", metric="destinations", station="BLR").data
    check_set("Q14 nonstop from BLR", r["destinations"],
              q["Q14"] if isinstance(q["Q14"], list) else q["Q14"]["stations"])

    r = T.call("risk_signal", crew_id="C-1042").data["signals"][0]
    check("Q16 risk score", r["disruption_risk_score"], q["Q16"]["score"])
    check("Q16 drivers", r["drivers"], q["Q16"]["drivers"])

    # --- Tier 2 -----------------------------------------------------------
    r = T.call("sick_call_impact", crew_id="C-1042", pairing_id="P-2291",
               from_date="2026-09-15").data
    check_set("Q17 day-1 uncrewed", r["uncrewed_now"], q["Q17"]["day1"])
    check_set("Q17 day-2 at risk", r["also_at_risk"], q["Q17"]["day2_also_at_risk"])
    check("Q17 passengers day 1", r["passengers_affected_now"], q["Q17"]["passengers_day1"])

    r = T.call("check_assignment", crew_id="C-2087", pairing_id="P-2291",
               on_date="2026-09-15").data
    check("Q18 legal", r["legal"], q["Q18"]["legal"])
    check("Q18 issues", r["issues"], q["Q18"]["issues"])

    r = T.call("station_closure_impact", station="BLR", on_date="2026-09-17",
               start_utc="08:00", end_utc="14:00").data
    want19 = q["Q19"] if isinstance(q["Q19"], list) else q["Q19"].get("affected_flights", [])
    check_set("Q19 closure-affected flights", r["affected_flights"], want19)

    r = T.call("delay_impact", on_date="2026-09-16", aircraft="VT-DXA", delay_minutes=90).data
    check("Q20 breach", r["breach"], q["Q20"]["breach"])
    check("Q20 FDP after delay", r["fdp_after_delay"], q["Q20"]["fdp_after_delay"])
    check("Q20 FDP limit", r["fdp_limit"], q["Q20"]["fdp_limit"])

    r = T.call("earliest_next_report", release_utc="2026-09-16T15:30:00Z").data
    check("Q23 earliest report", r["earliest_report_utc"], q["Q23"])

    r = T.call("check_assignment", crew_id="C-3305", pairing_id="P-2291").data
    check("Q24 C-3305 full pairing legal", r["legal"], q["Q24"]["legal"])
    check("Q24 issues", r["issues"], q["Q24"]["issues"])

    r = T.call("cancellation_impact", flight_ids=["DX404-2026-09-16"]).data
    check("Q25 passengers", r["passengers_affected"], q["Q25"]["passengers"])
    check("Q25 cost", r["cost_inr"], q["Q25"]["cost_inr"])

    r = T.call("crew_near_limits", on_date="2026-09-15", duty_threshold_7d=45).data
    check("Q26 crew at/over 45h", {c["crew_id"]: c["duty_hours_7d"] for c in r["crew"]},
          {c["crew_id"]: c["duty_hours_7d_incl_15sep_plan"] for c in q["Q26"]})

    r = T.call("recommend_options", pairing_id="P-2224", from_date="2026-09-16",
               role="Captain").data
    elig = [o["crew_id"] for o in r["options"] if o["crew_id"] and o["cost_inr"] <= 18500]
    check_set("Q27 eligible reserve captains", elig, q["Q27"]["eligible"])

    # --- Tier 3 -----------------------------------------------------------
    r = T.call("recommend_options", crew_id="C-1042", pairing_id="P-2291",
               from_date="2026-09-15").data
    _compare_options("Q31", r["options"], q["Q31"])

    r = T.call("recommend_options", crew_id="C-5417", pairing_id="P-2213",
               from_date="2026-09-19").data
    _compare_options("Q34", r["options"], q["Q34"] if isinstance(q["Q34"], list)
                     else q["Q34"].get("options", []))

    want37 = q["Q37"]
    fo = T.call("get_pairing", aircraft="VT-DXF", on_date="2026-09-20").data["pairings"]
    if fo:
        pid = fo[0]["pairing_id"]
        sick_fo = next((c for c in fo[0]["crew"] if c["role"] == "First Officer"), None)
        if sick_fo:
            r = T.call("recommend_options", crew_id=sick_fo["crew_id"], pairing_id=pid,
                       from_date="2026-09-20").data
            top = r["options"][0]
            check("Q37 cheapest FO cover",
                  {"crew_id": top["crew_id"], "cost_inr": top["cost_inr"]},
                  {"crew_id": want37["crew_id"], "cost_inr": want37["cost_inr"]})


def main():
    s = store()
    print(f"Crew Ops Advisor - kernel evaluation")
    print(f"dataset: {s.dir}  snapshot: {s.snapshot_utc}\n")
    eval_scenarios(s)
    eval_questions(s)
    total = len(PASS) + len(FAIL)
    print(f"\n{'-' * 58}")
    print(f"kernel checks: {len(PASS)}/{total} passed"
          + (f", {len(FAIL)} FAILED" if FAIL else " - all green"))
    if FAIL:
        print("failed:", ", ".join(FAIL))
    return 1 if FAIL else 0


