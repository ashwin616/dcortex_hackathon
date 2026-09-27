"""Tier 3 - crew notification.

Returns the verified facts a callout message must contain. The model writes
the prose; every figure in it has to come from here.
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

def notification_facts(crew_id: str, pairing_id: str, from_date: str | None = None) -> ToolResult:
    """The verified facts a callout message must contain. The model writes the
    prose; every number in it has to come from here."""
    s = _s()
    cid = crew_id.upper()
    p = s.pairing_by_id[pairing_id.upper()]
    days = [d for d in p["days"] if not from_date or d["date"] >= from_date]
    c = s.crew[cid]
    legs = [s.flight_by_id[f] for d in days for f in d["flights"]]
    res = s.reserves.get(cid)
    cost, label = None, None
    cost, label = callout_cost(s, c["rank"], s.is_reserve(cid))
    return ToolResult(
        tool="notification_facts",
        data={"crew_id": cid, "name": c["name"], "rank": c["rank"], "base": c["base"],
              "reachability_minutes": c["reachability_minutes"],
              "callout_type": label, "callout_cost_inr": cost,
              "oncall_window_utc": res["oncall_window_utc"] if res else None,
              "pairing_id": p["pairing_id"], "aircraft": p["aircraft"],
              "days": [{"date": d["date"], "report_utc": d["report_utc"],
                        "release_utc": d["release_utc"], "flights": d["flights"]} for d in days],
              "routing": " / ".join(f"{l['flight_no']} {l['dep_station']}-{l['arr_station']} "
                                    f"{l['dep_utc'][11:16]}Z" for l in legs),
              "overnight_station": legs[-1]["arr_station"] if len(days) > 1 else None,
              "first_report_utc": days[0]["report_utc"]},
        evidence=_ev("crew.json", cid) + _ev("rosters.json", p["pairing_id"])
                 + _ev("flights.json", [l["flight_id"] for l in legs]))

