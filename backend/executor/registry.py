"""The operation catalogue - the single source of truth for what the LLM may ask for.

Both the planner prompt and the plan validator are generated from this list, so
they cannot drift apart. Adding a tool in one place and forgetting the other is
a class of bug this file removes.

The catalogue is deliberately terse. It is rendered into the planner prompt on
every request, and on a 200,000-tokens-per-day budget prose in a prompt is a
recurring cost, not a one-off.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field

from backend.tools import TOOLS

# Parameters the model must never set: paging limits, internal knobs, and
# anything that would let a plan quietly widen its own blast radius.
# `min_score` and `top` are how a planner asks "who is worst" - hiding them
# forced the model to abstain on questions the data answers easily.
INTERNAL_PARAMS = {"limit", "include_excluded", "include_cancel", "top_n",
                   "turnaround_minutes", "number_floor",
                   "deadhead_ok", "needed_station", "exclude_pairing"}

TIER = {
    "query_data": 1, "data_schema": 1,
    "ask_sql": 1, "run_sql": 1, "sql_schema": 1,
    "find_crew": 1, "duty_clock": 1, "list_reserves": 1, "list_flights": 1,
    "certifications_expiring": 1, "get_pairing": 1, "schedule_stats": 1,
    "risk_signal": 1, "cost_reference": 1, "operating_snapshot": 1,
    "rules_reference": 1, "pairings_overnight": 1, "aircraft_day": 1,
    "roster_load": 2, "fdp_headroom": 2, "pairing_headroom": 2,
    "sick_call_impact": 2, "check_assignment": 2, "station_closure_impact": 2,
    "delay_impact": 2, "earliest_next_report": 2, "crew_near_limits": 2,
    "cancellation_impact": 2,
    "recommend_options": 3, "recommend_joint": 3, "recovery_options": 3,
    "notification_facts": 3, "morning_brief": 3,
}

# One line each, written for a planner choosing between them - not user-facing
# documentation. Where two operations overlap, the line says which wins.
SUMMARY = {
    "query_data": "GENERAL data question: filter/sort/aggregate any table. Use for superlatives (longest OR shortest, most OR fewest), counts, totals, averages, top-N and group-by, on any field",
    "data_schema": "which entities and fields query_data accepts",
    "ask_sql": "answer a data question by writing and running SQL against the snapshot database - use for joins, multi-table aggregates and anything query_data cannot shape",
    "run_sql": "run one read-only SELECT you have already written",
    "sql_schema": "the SQL tables, columns and row counts",
    "find_crew": "crew record(s) by id, or filtered by rank/base/rating/status - includes their pairings, clock and certificates when an id is given",
    "duty_clock": "one crew member's hours used and how much headroom is left this week - answers 'have they got room'",
    "list_reserves": "reserves on call for a date; covers_time_utc filters by window",
    "list_flights": "schedule filtered by date/station/flight_no/aircraft/time",
    "certifications_expiring": "certs expiring within N days of a date",
    "get_pairing": "a pairing and WHO IS ON IT: days, legs, report/release, every crew member and their role",
    "schedule_stats": "network aggregates: counts, longest block, destinations, seats",
    "risk_signal": "disruption-risk scores; min_score filters to the high ones",
    "cost_reference": "callout/deadhead/delay/cancellation rates and the deltas between them",
    "operating_snapshot": "what 'now' is: today's date, the schedule week, hub, units",
    "rules_reference": "the seven rules, their parameters, and the FDP ladder by sector count",
    "pairings_overnight": "pairings that night-stop away from base, and where",
    "aircraft_day": "everything one tail flies on one day: legs, seats, block, cancel cost",
    "roster_load": "who is working most this week - duty days, hours, concentration risk",
    "fdp_headroom": "rostered duties closest to their FDP ceiling (least delay tolerance)",
    "pairing_headroom": "duty-clock headroom for every crew member on a pairing",
    "sick_call_impact": "crew is out: which legs lose them now and what is at risk after",
    "check_assignment": "is assigning this crew to this pairing/legs legal? all 7 rules",
    "station_closure_impact": "closure window: flights hit, delay to reopen, crew FDP after",
    "delay_impact": "delay before first departure: does the rostered crew's FDP still fit",
    "earliest_next_report": "RULE-REST-04 only: release time -> earliest next report",
    "crew_near_limits": "who is closest to their duty/flight ceiling on a date (threshold optional, defaults to 45h)",
    "cancellation_impact": "seats and direct cost of cancelling specific legs",
    "recommend_options": "ranked legal ways to cover ONE opening, with cost and reasons",
    "recommend_joint": "two or more crew out at once: one plan, nobody assigned twice",
    "recovery_options": "after an FDP breach: price re-crewing the tail legs vs cancelling",
    "notification_facts": "verified facts for a callout message (then write the message)",
    "morning_brief": "per aircraft line: duty headroom, FDP slack, cert exposure",
}


@dataclass(frozen=True)
class Operation:
    name: str
    tier: int
    summary: str
    params: tuple = ()
    required: tuple = ()

    def signature(self) -> str:
        parts = [f"{p}*" if p in self.required else p for p in self.params]
        return f"{self.name}({', '.join(parts)})"


def _build() -> dict:
    ops = {}
    for name, fn in TOOLS.items():
        sig = inspect.signature(fn)
        params, required = [], []
        for pname, p in sig.parameters.items():
            if pname in INTERNAL_PARAMS:
                continue
            params.append(pname)
            if p.default is inspect._empty:
                required.append(pname)
        ops[name] = Operation(name=name, tier=TIER.get(name, 1),
                              summary=SUMMARY.get(name, (inspect.getdoc(fn) or "").split("\n")[0]),
                              params=tuple(params), required=tuple(required))
    return ops


OPERATIONS = _build()
ALLOWED_OPERATIONS = frozenset(OPERATIONS)


def query_schema_block() -> str:
    """The queryable tables, rendered for the planner prompt.

    Cheaper than a tool per question shape, and it is the difference between
    answering "the shortest block time" and refusing it.
    """
    from backend.tools.query import (AGGREGATES, ENUM_FIELDS, OPERATORS, SCHEMA,
                                     enum_values)
    lines = ["QUERYABLE TABLES for query_data(entity, where, order_by, direction,",
             "                                aggregate, field, group_by, limit)"]
    for entity, fields in SCHEMA.items():
        lines.append(f"  {entity}: {', '.join(fields)}")
    # The values, not just the field names. A filter on status="on_leave" when
    # the data says "leave" returns [] - which reads as "nobody is on leave".
    # Grouped by value set: rank means the same four things in four tables, and
    # this prompt is rendered on every request.
    groups: dict = {}
    for entity in SCHEMA:
        for f in ENUM_FIELDS.get(entity, ()):
            if vals := enum_values(entity, f):
                groups.setdefault(" | ".join(vals), []).append(f"{entity}.{f}")
    if groups:
        lines.append("  these fields only ever hold these values:")
        for vals, names in groups.items():
            lines.append(f"    {', '.join(names)} = {vals}")
    lines.append(f"  operators: {', '.join(sorted(OPERATORS))}")
    lines.append(f"  aggregates: {', '.join(AGGREGATES)}  ·  direction: asc | desc")
    lines.append('  where entries look like {"field": "block_hours", "op": "gte", "value": 2}')
    return "\n".join(lines)


def catalogue(max_tier: int = 3) -> str:
    """The operation list as it appears in the planner prompt. `*` marks required."""
    lines = []
    for tier in (1, 2, 3):
        if tier > max_tier:
            break
        label = {1: "LOOKUP", 2: "CONSEQUENCE", 3: "RECOMMENDATION"}[tier]
        lines.append(f"TIER {tier} - {label}")
        for op in sorted((o for o in OPERATIONS.values() if o.tier == tier),
                         key=lambda o: o.name):
            lines.append(f"  {op.signature()}")
            lines.append(f"      {op.summary}")
    return "\n".join(lines)
