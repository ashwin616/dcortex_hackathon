"""One typed query operation, instead of a tool per question shape.

`schedule_stats(metric="longest_block")` answers exactly one question. Asked for
the *shortest* block time the system had nothing, and said so - correctly and
uselessly. The same gap swallows "average block hours", "top five by seats",
"how many legs per aircraft type", and every other phrasing a controller
invents. Writing a tool per shape never catches up.

So data questions are answered by one parameterised operation. The planner emits
a typed query - entity, filters, ordering, aggregation - and deterministic code
executes it. That keeps the boundary the whole system rests on:

  * the entity and every field are allowlisted, so there is no attribute access
    the schema does not name and no way to reach the filesystem or the network
  * operators are a fixed set; no expression is ever evaluated
  * the query that ran is returned with the answer, so it is auditable and a
    controller can see exactly what was asked

Rule evaluation stays where it belongs. `check_assignment` and
`recommend_options` are judgements against a rulebook, not queries, and they
keep their own tools. The split is: query the data generically, evaluate the
rules specifically.
"""
from __future__ import annotations

import statistics
from datetime import date

from backend.domain.data import parse_dt
from backend.domain.duty import fdp_limit
from backend.domain.models import ToolResult
from backend.domain.rules import MAX_DUTY_7D, MAX_FLIGHT_28D
from backend.domain.world import World
from backend.tools.common import _ev, _s

# --------------------------------------------------------------------------
# schema - the allowlist, and what the planner is told it may ask for
# --------------------------------------------------------------------------
SCHEMA: dict[str, list[str]] = {
    "flights": ["flight_id", "flight_no", "date", "dep_station", "arr_station",
                "dep_utc", "arr_utc", "dep_time", "arr_time", "block_hours",
                "aircraft", "aircraft_type", "seats", "pairing_id", "route"],
    "crew": ["crew_id", "name", "rank", "base", "ratings", "seniority",
             "reachability_minutes", "status", "is_reserve", "duty_hours_7d",
             "flight_hours_28d", "duty_headroom_7d", "flight_headroom_28d",
             "risk_score", "duty_days_week", "duty_hours_week", "pairing_count"],
    "duties": ["pairing_id", "aircraft", "aircraft_type", "date", "sectors",
               "duty_hours", "block_hours", "seats", "fdp_limit", "fdp_slack",
               "report_utc", "release_utc", "overnight_station", "crew_size"],
    "reserves": ["crew_id", "rank", "base", "ratings", "window_start",
                 "window_end", "window_hours", "reachability_minutes", "days_on_call"],
    "certifications": ["crew_id", "rank", "base", "cert_type", "valid_from",
                       "valid_to", "days_to_expiry", "expired_at_snapshot"],
    "risk": ["crew_id", "rank", "base", "score", "drivers"],
}

# Fields whose values come from a small fixed set. The planner guessed
# `status = "on_leave"` when the data says `"leave"`, and a filter that matches
# nothing returns an empty list that looks exactly like a true negative - the
# worst failure shape available, because it is silent. Listing the real values
# in the prompt prevents the guess; echoing them back on an empty result makes
# the mistake self-correcting when one slips through anyway.
ENUM_FIELDS = {
    "flights": ["dep_station", "arr_station", "aircraft", "aircraft_type"],
    "crew": ["rank", "base", "status"],
    "duties": ["aircraft", "aircraft_type", "overnight_station"],
    "reserves": ["rank", "base"],
    "certifications": ["cert_type", "rank", "base"],
    "risk": ["rank", "base"],
}
_ENUM_CACHE: dict = {}
MAX_ENUM_VALUES = 12


def _matches_any(value, valid: list) -> bool:
    """Did the filter name at least one value the data actually holds?

    Case-insensitive, and it looks inside a list because `in` filters pass one.
    """
    wanted = value if isinstance(value, (list, tuple, set)) else [value]
    low = {v.lower() for v in valid}
    return any(isinstance(w, str) and w.lower() in low for w in wanted)


def enum_values(entity: str, field: str) -> list | None:
    """The distinct values of a categorical field, or None if it is not one."""
    if field not in ENUM_FIELDS.get(entity, ()):
        return None
    key = (entity, field)
    if key not in _ENUM_CACHE:
        seen = {r.get(field) for r in _rows(entity)}
        vals = sorted(str(v) for v in seen if isinstance(v, str) and v)
        _ENUM_CACHE[key] = vals if len(vals) <= MAX_ENUM_VALUES else []
    return _ENUM_CACHE[key] or None


# The columns that identify a row of each entity. Everything else is detail a
# narrator does not need in order to answer, and the evidence panel keeps.
IDENTITY = {
    "flights": ["flight_id", "flight_no", "date", "route"],
    "crew": ["crew_id", "name", "rank", "base"],
    "duties": ["pairing_id", "aircraft", "date"],
    "reserves": ["crew_id", "rank", "base"],
    "certifications": ["crew_id", "cert_type", "valid_to"],
    "risk": ["crew_id", "rank", "score"],
}
# Beyond this many rows, sending every column is paying tokens for detail
# nobody reads.
PROJECT_ABOVE = 6

EVIDENCE_FILE = {
    "flights": ("flights.json", "flight_id"),
    "crew": ("crew.json", "crew_id"),
    "duties": ("rosters.json", "pairing_id"),
    "reserves": ("reserve_pool.json", "crew_id"),
    "certifications": ("certifications.json", "crew_id"),
    "risk": ("risk_signals.json", "crew_id"),
}

OPERATORS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: _num(a) < _num(b),
    "lte": lambda a, b: _num(a) <= _num(b),
    "gt": lambda a, b: _num(a) > _num(b),
    "gte": lambda a, b: _num(a) >= _num(b),
    "in": lambda a, b: a in b if isinstance(b, (list, tuple, set)) else a == b,
    "nin": lambda a, b: a not in b if isinstance(b, (list, tuple, set)) else a != b,
    "contains": lambda a, b: _contains(a, b),
    "startswith": lambda a, b: str(a or "").upper().startswith(str(b).upper()),
    "between": lambda a, b: (isinstance(b, (list, tuple)) and len(b) == 2
                             and _num(b[0]) <= _num(a) <= _num(b[1])),
}
AGGREGATES = ("count", "sum", "avg", "min", "max", "distinct")

# Words a controller uses for a direction, so "shortest" needs no new tool.
DIRECTION_WORDS = {"asc": "asc", "ascending": "asc", "up": "asc",
                   "smallest": "asc", "shortest": "asc", "lowest": "asc",
                   "least": "asc", "min": "asc",
                   "desc": "desc", "descending": "desc", "down": "desc",
                   "largest": "desc", "longest": "desc", "highest": "desc",
                   "most": "desc", "max": "desc", "biggest": "desc"}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("-inf") if v is None else str(v)


def _contains(a, b) -> bool:
    if isinstance(a, (list, tuple, set)):
        return b in a or str(b).upper() in {str(x).upper() for x in a}
    return str(b).lower() in str(a or "").lower()


# --------------------------------------------------------------------------
# flattened rows, one builder per entity
# --------------------------------------------------------------------------
def _rows(entity: str) -> list[dict]:
    s = _s()
    if entity == "flights":
        out = []
        for f in s.flights:
            pid, _ = s.pairing_of_flight(f["flight_id"])
            out.append({**f, "pairing_id": pid,
                        "dep_time": f["dep_utc"][11:16], "arr_time": f["arr_utc"][11:16],
                        "route": f"{f['dep_station']}-{f['arr_station']}"})
        return out

    if entity == "crew":
        w = World(s)
        out = []
        for cid, c in s.crew.items():
            clock = s.clocks.get(cid, {})
            duties = w.duties(cid)
            out.append({
                **c, "is_reserve": s.is_reserve(cid),
                "duty_hours_7d": clock.get("duty_hours_7d"),
                "flight_hours_28d": clock.get("flight_hours_28d"),
                "duty_headroom_7d": round(MAX_DUTY_7D - (clock.get("duty_hours_7d") or 0), 2),
                "flight_headroom_28d": round(
                    MAX_FLIGHT_28D - (clock.get("flight_hours_28d") or 0), 2),
                "risk_score": (s.risk.get(cid) or {}).get("disruption_risk_score"),
                "duty_days_week": len({d.date for d in duties}),
                "duty_hours_week": round(sum(d.duty_hours for d in duties), 2),
                "pairing_count": len(s.pairings_of.get(cid, [])),
            })
        return out

    if entity == "duties":
        out = []
        for p in s.pairings:
            for day in p["days"]:
                _rep, _rel, dh, n = s.duty_of(day)
                legs = [s.flight_by_id[f] for f in day["flights"]]
                out.append({
                    "pairing_id": p["pairing_id"], "aircraft": p["aircraft"],
                    "aircraft_type": legs[0]["aircraft_type"], "date": day["date"],
                    "sectors": n, "duty_hours": dh,
                    "block_hours": round(sum(l["block_hours"] for l in legs), 2),
                    "seats": sum(l["seats"] for l in legs),
                    "fdp_limit": fdp_limit(n),
                    "fdp_slack": round(fdp_limit(n) - dh, 2),
                    "report_utc": day["report_utc"], "release_utc": day["release_utc"],
                    "overnight_station": legs[-1]["arr_station"],
                    "crew_size": len(p["crew"]),
                })
        return out

    if entity == "reserves":
        out = []
        for cid, r in s.reserves.items():
            c = s.crew[cid]
            w = r["oncall_window_utc"]
            start, end = w["start"], w["end"]
            hours = (int(end[:2]) * 60 + int(end[3:]) - int(start[:2]) * 60
                     - int(start[3:])) / 60.0
            out.append({"crew_id": cid, "rank": c["rank"], "base": c["base"],
                        "ratings": c["ratings"], "window_start": start,
                        "window_end": end, "window_hours": round(hours, 2),
                        "reachability_minutes": c["reachability_minutes"],
                        "days_on_call": len(r.get("dates", []))})
        return out

    if entity == "certifications":
        snap = s.snapshot.date()
        out = []
        for c in s.cert_list:
            cid = c["crew_id"]
            valid_to = date.fromisoformat(c["valid_to"])
            out.append({**c, "rank": s.crew[cid]["rank"], "base": s.crew[cid]["base"],
                        "days_to_expiry": (valid_to - snap).days,
                        "expired_at_snapshot": valid_to < snap})
        return out

    if entity == "risk":
        return [{"crew_id": r["crew_id"], "score": r["disruption_risk_score"],
                 "drivers": r.get("drivers", []),
                 "rank": s.crew[r["crew_id"]]["rank"],
                 "base": s.crew[r["crew_id"]]["base"]} for r in s.risk_list]

    return []


# --------------------------------------------------------------------------
# the operation
# --------------------------------------------------------------------------
def query_data(entity: str, where: list | None = None, order_by: str | None = None,
               direction: str = "desc", aggregate: str | None = None,
               field: str | None = None, group_by: str | None = None,
               limit: int = 20, select: list | None = None) -> ToolResult:
    """Filter, sort, aggregate or group any dataset table.

    entity: flights | crew | duties | reserves | certifications | risk
    where:  [{"field": ..., "op": "eq|ne|lt|lte|gt|gte|in|nin|contains|
              startswith|between", "value": ...}, ...]
    order_by + direction ("asc" for shortest/smallest, "desc" for longest/largest)
    aggregate: count | sum | avg | min | max | distinct, applied to `field`,
    optionally split by `group_by`.
    """
    entity = (entity or "").strip().lower()
    if entity not in SCHEMA:
        return ToolResult(tool="query_data", data={
            "error": f"unknown entity {entity!r}; choose from "
                     f"{', '.join(sorted(SCHEMA))}"}, evidence=[])
    fields = SCHEMA[entity]

    def known(name: str | None, label: str):
        if name and name not in fields:
            return (f"{entity} has no field {name!r} ({label}); available: "
                    f"{', '.join(fields)}")
        return None

    conditions = []
    for cond in (where or []):
        if not isinstance(cond, dict):
            return ToolResult(tool="query_data",
                              data={"error": f"each `where` entry must be an object "
                                             f"with field/op/value, got {cond!r}"},
                              evidence=[])
        f, op, val = cond.get("field"), (cond.get("op") or "eq").lower(), cond.get("value")
        if (msg := known(f, "where")):
            return ToolResult(tool="query_data", data={"error": msg}, evidence=[])
        if op not in OPERATORS:
            return ToolResult(tool="query_data", data={
                "error": f"unknown operator {op!r}; choose from "
                         f"{', '.join(sorted(OPERATORS))}"}, evidence=[])
        conditions.append((f, op, val))

    direction = DIRECTION_WORDS.get(str(direction).lower(), "desc")
    for name, label in ((order_by, "order_by"), (field, "field"), (group_by, "group_by")):
        if (msg := known(name, label)):
            return ToolResult(tool="query_data", data={"error": msg}, evidence=[])
    if aggregate and aggregate.lower() not in AGGREGATES:
        return ToolResult(tool="query_data", data={
            "error": f"unknown aggregate {aggregate!r}; choose from "
                     f"{', '.join(AGGREGATES)}"}, evidence=[])

    rows = _rows(entity)
    for f, op, val in conditions:
        rows = [r for r in rows if _safe(OPERATORS[op], r.get(f), val)]
    matched = len(rows)

    interpreted = {"entity": entity,
                   "where": [{"field": f, "op": o, "value": v} for f, o, v in conditions],
                   "order_by": order_by, "direction": direction,
                   "aggregate": aggregate, "field": field, "group_by": group_by,
                   "limit": limit}

    # A filter naming a value the data never holds is the worst failure shape
    # available, because it is silent: `status = "on_leave"` when the data says
    # "leave" returns an empty list, and an empty list reads as "nobody is on
    # leave". Worse, `in ["on_leave", "training"]` matches the second value and
    # returns a partial answer that looks complete. So every unrecognised value
    # is named, whether or not anything matched.
    unknown: dict = {}
    for f, _op, val in conditions:
        if (vals := enum_values(entity, f)) is None:
            continue
        wanted = val if isinstance(val, (list, tuple, set)) else [val]
        low = {v.lower() for v in vals}
        bad = [w for w in wanted if isinstance(w, str) and w.lower() not in low]
        if bad:
            unknown[f] = {"not_in_data": bad, "valid_values": vals}
    if unknown and not rows:
        return ToolResult(
            tool="query_data",
            data={"query": interpreted, "matched": 0, "rows": [],
                  "error": "no rows matched, and the filter names values the "
                           "data does not use",
                  "unrecognised_filter_values": unknown},
            evidence=[])

    # ---- aggregation -----------------------------------------------------
    if aggregate:
        agg = aggregate.lower()
        if group_by:
            groups: dict = {}
            for r in rows:
                groups.setdefault(r.get(group_by), []).append(r)
            result = {str(k): _apply(agg, v, field) for k, v in sorted(
                groups.items(), key=lambda kv: str(kv[0]))}
        else:
            result = _apply(agg, rows, field)
        return ToolResult(
            tool="query_data",
            data={"query": interpreted, "matched": matched, "result": result,
                  "note": f"{agg} of {field or 'rows'}"
                          + (f" grouped by {group_by}" if group_by else "")},
            evidence=_evidence(entity, rows[:40]))

    # ---- ordering and the top slice --------------------------------------
    if order_by:
        rows = sorted(rows, key=lambda r: _sort_key(r.get(order_by)),
                      reverse=(direction == "desc"))
        # A superlative usually means "and everything tied with it".
        if rows and limit:
            best = rows[0].get(order_by)
            ties = [r for r in rows if r.get(order_by) == best]
            if len(ties) > limit:
                limit = len(ties)

    shown = rows[:max(1, int(limit or 20))]
    projected = False
    if select:
        keep = [c for c in select if c in fields] or fields
    elif len(shown) > PROJECT_ABOVE:
        # Long answers get the identifying columns plus whatever was ordered on.
        keep = list(dict.fromkeys(
            [c for c in IDENTITY.get(entity, []) if c in fields]
            + ([order_by] if order_by else [])))
        projected = bool(keep)
    else:
        keep = None
    if keep:
        shown = [{k: r.get(k) for k in keep} for r in shown]

    return ToolResult(
        tool="query_data",
        data={"query": interpreted, "matched": matched, "returned": len(shown),
              "columns": keep, "projected": projected,
              "rows": shown,
              "extreme_value": rows[0].get(order_by) if (order_by and rows) else None,
              **({"unrecognised_filter_values": unknown} if unknown else {})},
        evidence=_evidence(entity, rows[:40]),
        note=_note(projected, unknown))


def _note(projected: bool, unknown: dict) -> str | None:
    parts = []
    if projected:
        parts.append("showing identifying columns only; the full records are in "
                     "the evidence panel")
    for f, d in unknown.items():
        parts.append(f"the filter on {f} used {d['not_in_data']}, which the data "
                     f"never holds; valid values are {d['valid_values']} - this "
                     "result may therefore be incomplete")
    return " · ".join(parts) or None


def _safe(fn, a, b) -> bool:
    try:
        return bool(fn(a, b))
    except (TypeError, ValueError):
        return False


def _sort_key(v):
    if v is None:
        return (2, 0, "")
    if isinstance(v, bool):
        return (1, int(v), "")
    if isinstance(v, (int, float)):
        return (0, v, "")
    return (1, 0, str(v))


def _apply(agg: str, rows: list, field: str | None):
    if agg == "count":
        return len(rows)
    if agg == "distinct":
        return sorted({str(r.get(field)) for r in rows if r.get(field) is not None})
    vals = [r.get(field) for r in rows if isinstance(r.get(field), (int, float))]
    if not vals:
        return None
    if agg == "sum":
        return round(sum(vals), 2)
    if agg == "avg":
        return round(statistics.fmean(vals), 2)
    if agg == "min":
        return min(vals)
    if agg == "max":
        return max(vals)
    return None


def _evidence(entity: str, rows: list) -> list:
    file, key = EVIDENCE_FILE[entity]
    return _ev(file, [str(r.get(key)) for r in rows if r.get(key)])


def data_schema() -> ToolResult:
    """What can be queried, and with which fields. Use when unsure."""
    return ToolResult(tool="data_schema",
                      data={"entities": SCHEMA, "operators": sorted(OPERATORS),
                            "aggregates": list(AGGREGATES)},
                      evidence=["schema#query_data"])
