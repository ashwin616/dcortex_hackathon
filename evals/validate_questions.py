"""Are the questions themselves right?

`questions_user_perspective.json` ships answer keys. Before scoring the Advisor
against them it is worth checking they agree with the dataset, because grading a
correct system against a wrong key wastes a day and teaches the wrong lesson.

Every check here recomputes the expected value from the raw JSON, independently
of the tool layer, so a shared bug cannot make both sides agree. Questions whose
answer is a judgement (the Tier-3 ones carrying `grading_note`) are reported as
unverifiable rather than guessed at.

    python eval.py validate
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from backend.domain.data import DATA_DIR, parse_dt, store

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OK, BAD, SKIP = [], [], []


def check(qid: str, label: str, got, want):
    if got == want:
        OK.append(qid)
        return
    BAD.append((qid, label, got, want))


def approx(qid: str, label: str, got, want, tol=0.011):
    try:
        if abs(float(got) - float(want)) <= tol:
            OK.append(qid)
            return
    except (TypeError, ValueError):
        pass
    BAD.append((qid, label, got, want))


def hhmm_in(window: dict, t: str) -> bool:
    return window["start"] <= t <= window["end"]


def main() -> int:
    s = store()
    path = DATA_DIR / "questions_user_perspective.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    qs = {q["question_id"]: q for q in doc["questions"]}
    exp = {k: v["expected_answer"] for k, v in qs.items()}

    print(f"Validating {len(qs)} question keys in {path.name}\n")

    # ---- Q39 leave / training ------------------------------------------
    want = sorted((c["crew_id"], c["status"]) for c in exp["Q39"])
    got = sorted((c["crew_id"], c["status"]) for c in s.crew_list
                 if c.get("status") in ("leave", "training"))
    check("Q39", "crew on leave or training", got, want)

    # ---- Q40 cabin crew at BLR -----------------------------------------
    counts = Counter(c["rank"] for c in s.crew_list if c["base"] == "BLR"
                     and "Cabin" in c["rank"])
    check("Q40", "Senior Cabin Crew at BLR",
          counts["Senior Cabin Crew"], exp["Q40"]["Senior Cabin Crew"])
    check("Q40", "Cabin Crew at BLR", counts["Cabin Crew"], exp["Q40"]["Cabin Crew"])
    check("Q40", "total cabin at BLR", sum(counts.values()), exp["Q40"]["total"])

    # ---- Q41 P-2204 on 17 Sep ------------------------------------------
    day = next(d for d in s.pairing_by_id["P-2204"]["days"] if d["date"] == "2026-09-17")
    dur = (parse_dt(day["release_utc"]) - parse_dt(day["report_utc"])).total_seconds() / 3600
    check("Q41", "report", day["report_utc"], exp["Q41"]["report_utc"])
    check("Q41", "release", day["release_utc"], exp["Q41"]["release_utc"])
    approx("Q41", "duty hours", dur, exp["Q41"]["duty_hours"])
    check("Q41", "sectors", len(day["flights"]), exp["Q41"]["sectors"])
    approx("Q41", "fdp limit", 13.0 - 0.5 * max(0, len(day["flights"]) - 2),
           exp["Q41"]["fdp_limit_hours"])

    # ---- Q42 C-2091 rating ---------------------------------------------
    c = s.crew["C-2091"]
    check("Q42", "A320 cleared", "A320" in c["ratings"], exp["Q42"]["cleared"])
    check("Q42", "ratings", c["ratings"], exp["Q42"]["ratings"])
    check("Q42", "rank", c["rank"], exp["Q42"]["rank"])

    # ---- Q43 reserves inside window at 02:00Z on 16 Sep -----------------
    want = sorted(r["crew_id"] for r in exp["Q43"])
    got = sorted(cid for cid, r in s.reserves.items()
                 if "2026-09-16" in r.get("dates", [])
                 and hhmm_in(r["oncall_window_utc"], "02:00"))
    check("Q43", "reserves in window at 02:00Z", got, want)

    # ---- Q44 network block hours on 18 Sep ------------------------------
    legs = s.flights_by_date["2026-09-18"]
    approx("Q44", "block hours", round(sum(f["block_hours"] for f in legs), 2),
           exp["Q44"]["block_hours"])
    check("Q44", "legs", len(legs), exp["Q44"]["legs"])

    # ---- Q45 top five closest to the 100h/28d limit ---------------------
    top = sorted(s.clocks_list, key=lambda c: -c["flight_hours_28d"])[:5]
    check("Q45", "top five crew", [c["crew_id"] for c in top],
          [e["crew_id"] for e in exp["Q45"]])
    for e in exp["Q45"]:
        approx("Q45", f"{e['crew_id']} 28d hours",
               s.clocks[e["crew_id"]]["flight_hours_28d"], e["flight_hours_28d"])
        approx("Q45", f"{e['crew_id']} headroom",
               round(100 - s.clocks[e["crew_id"]]["flight_hours_28d"], 2),
               e["headroom_hours"])

    # ---- Q46 pairings that night-stop away from base --------------------
    got = []
    for p in s.pairings:
        if len(p["days"]) < 2:
            continue
        last = s.flight_by_id[p["days"][0]["flights"][-1]]
        if last["arr_station"] != "BLR":
            got.append((p["pairing_id"], last["arr_station"]))
    check("Q46", "overnighting pairings", sorted(got),
          sorted((e["pairing_id"], e["overnight_station"]) for e in exp["Q46"]))

    # ---- Q47 callout rates ----------------------------------------------
    for key in ("reserve_callout_pilot", "dayoff_callout_pilot",
                "reserve_callout_cabin", "dayoff_callout_cabin"):
        check("Q47", key, s.costs[key], exp["Q47"][key])
    check("Q47", "pilot delta",
          s.costs["dayoff_callout_pilot"] - s.costs["reserve_callout_pilot"],
          exp["Q47"]["pilot_delta"])
    check("Q47", "cabin delta",
          s.costs["dayoff_callout_cabin"] - s.costs["reserve_callout_cabin"],
          exp["Q47"]["cabin_delta"])

    # ---- Q48 risk >= 0.70 ------------------------------------------------
    got = sorted((r["crew_id"], r["disruption_risk_score"]) for r in s.risk_list
                 if r["disruption_risk_score"] >= 0.70)
    check("Q48", "high-risk crew", got,
          sorted((e["crew_id"], e["score"]) for e in exp["Q48"]))

    # ---- Q49 C-2087 clock ------------------------------------------------
    clock = s.clocks["C-2087"]
    check("Q49", "last rest ended", clock["last_rest_ended"], exp["Q49"]["last_rest_ended"])
    approx("Q49", "duty 7d", clock["duty_hours_7d"], exp["Q49"]["duty_hours_7d"])
    approx("Q49", "headroom", round(60 - clock["duty_hours_7d"], 2),
           exp["Q49"]["headroom_hours"])

    # ---- Q50 legs per aircraft on 15 Sep ---------------------------------
    got = dict(Counter(f["aircraft"] for f in s.flights_by_date["2026-09-15"]))
    check("Q50", "legs per aircraft", got, exp["Q50"])

    # ---- Q53 rest arithmetic ---------------------------------------------
    if "Q53" in exp and isinstance(exp["Q53"], dict):
        rel = parse_dt("2026-09-18T23:15:00Z")
        earliest = rel + timedelta(hours=12)
        e = exp["Q53"]
        if "earliest_report_utc" in e:
            check("Q53", "earliest report",
                  earliest.strftime("%Y-%m-%dT%H:%M:%SZ"), e["earliest_report_utc"])
        if "usable" in e:
            check("Q53", "usable for 10:00Z report",
                  parse_dt("2026-09-19T10:00:00Z") >= earliest, e["usable"])

    # ---- Q54 FDP ladder ---------------------------------------------------
    if "Q54" in exp and isinstance(exp["Q54"], dict):
        for k, v in exp["Q54"].items():
            digits = "".join(ch for ch in str(k) if ch.isdigit())
            if digits:
                n = int(digits)
                approx("Q54", f"{n} sectors", 13.0 - 0.5 * max(0, n - 2), v)

    # ---- Q56 DEL closure 06:00-10:00Z on 16 Sep --------------------------
    if "Q56" in exp:
        want56 = exp["Q56"] if isinstance(exp["Q56"], list) else \
            exp["Q56"].get("affected_flights", [])
        want_ids = sorted(x["flight_id"] if isinstance(x, dict) else x for x in want56)
        got56 = sorted(
            f["flight_id"] for f in s.flights_by_date["2026-09-16"]
            if (f["dep_station"] == "DEL" and "06:00" <= f["dep_utc"][11:16] < "10:00")
            or (f["arr_station"] == "DEL" and "06:00" <= f["arr_utc"][11:16] < "10:00"))
        check("Q56", "DEL closure flights", got56, want_ids)

    # ---- Q59 VT-DXC whole 15 Sep programme -------------------------------
    if "Q59" in exp and isinstance(exp["Q59"], dict):
        legs59 = [f for f in s.flights_by_date["2026-09-15"] if f["aircraft"] == "VT-DXC"]
        e = exp["Q59"]
        if "passengers" in e:
            check("Q59", "passengers", sum(f["seats"] for f in legs59), e["passengers"])
        if "flights" in e and isinstance(e["flights"], int):
            check("Q59", "flights", len(legs59), e["flights"])
        for key in ("cancellation_cost_inr", "cost_inr"):
            if key in e:
                check("Q59", key, 250000 * len(legs59), e[key])

    # ---- unverifiable by construction ------------------------------------
    for qid, q in qs.items():
        if q.get("grading_note") and qid not in set(OK) | {b[0] for b in BAD}:
            SKIP.append(qid)

    # ---- report -----------------------------------------------------------
    checked = sorted(set(OK) | {b[0] for b in BAD})
    print(f"machine-checked : {len(checked)} questions, {len(OK)} assertions passed")
    if BAD:
        print(f"\nDISAGREEMENTS ({len(BAD)}):")
        for qid, label, got, want in BAD:
            print(f"  {qid} {label}")
            print(f"      dataset says : {got}")
            print(f"      key says     : {want}")
    else:
        print("every machine-checkable key agrees with the dataset")
    if SKIP:
        print(f"\njudgement-based, graded by note not by key: {', '.join(sorted(SKIP))}")
    unchecked = [q for q in qs if q not in checked and q not in SKIP]
    if unchecked:
        print(f"not independently checked: {', '.join(sorted(unchecked))}")
    return 1 if BAD else 0
