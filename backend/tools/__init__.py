"""The tool layer - the operations a validated plan may execute.

Split by tier so the boundary between "look it up", "work out what happens" and
"tell me what to do" is visible in the file tree, not just the docstrings.
"""
from __future__ import annotations

from backend.cache import memoise
from backend.domain.models import ToolResult
from backend.tools.impact import (cancellation_impact, check_assignment,
                                  crew_near_limits, delay_impact,
                                  earliest_next_report, sick_call_impact,
                                  station_closure_impact)
from backend.tools.lookup import (certifications_expiring, duty_clock, find_crew,
                                  get_pairing, list_flights, list_reserves,
                                  risk_signal, schedule_stats)
from backend.tools.notification import notification_facts
from backend.tools.query import data_schema, query_data
from backend.tools.sql import ask_sql, run_sql, sql_schema
from backend.tools.planning import (aircraft_day, cost_reference,
                                    operating_snapshot,
                                    fdp_headroom, pairing_headroom,
                                    pairings_overnight, roster_load,
                                    rules_reference)
from backend.tools.recommendation import (morning_brief, recommend_joint,
                                          recommend_options, recovery_options)

TOOLS = {
    # Tier 1 - lookup
    "query_data": query_data,
    "ask_sql": ask_sql,
    "run_sql": run_sql,
    "sql_schema": sql_schema,
    "data_schema": data_schema,
    "find_crew": find_crew,
    "duty_clock": duty_clock,
    "list_reserves": list_reserves,
    "list_flights": list_flights,
    "certifications_expiring": certifications_expiring,
    "get_pairing": get_pairing,
    "schedule_stats": schedule_stats,
    "risk_signal": risk_signal,
    "cost_reference": cost_reference,
    "operating_snapshot": operating_snapshot,
    "rules_reference": rules_reference,
    "pairings_overnight": pairings_overnight,
    "aircraft_day": aircraft_day,
    # Tier 2 - planning
    "roster_load": roster_load,
    "fdp_headroom": fdp_headroom,
    "pairing_headroom": pairing_headroom,
    # Tier 2 - consequence
    "sick_call_impact": sick_call_impact,
    "check_assignment": check_assignment,
    "station_closure_impact": station_closure_impact,
    "delay_impact": delay_impact,
    "earliest_next_report": earliest_next_report,
    "crew_near_limits": crew_near_limits,
    "cancellation_impact": cancellation_impact,
    # Tier 3 - recommendation
    "recommend_options": recommend_options,
    "recommend_joint": recommend_joint,
    "recovery_options": recovery_options,
    "notification_facts": notification_facts,
    "morning_brief": morning_brief,
}


def call(name: str, **kwargs) -> ToolResult:
    """Invoke one operation. Never raises - the deterministic layer must not be
    able to take the chat down, so a failure comes back as data."""
    fn = TOOLS.get(name)
    if fn is None:
        return ToolResult(tool=name, data={"error": f"no such tool: {name}"}, evidence=[])
    try:
        return memoise(name, kwargs, lambda: fn(**kwargs))
    except TypeError as e:
        return ToolResult(tool=name, data={"error": f"bad arguments: {e}"}, evidence=[])
    except Exception as e:  # noqa: BLE001 - deliberate containment
        return ToolResult(tool=name, data={"error": f"{type(e).__name__}: {e}"}, evidence=[])


__all__ = ["TOOLS", "call"]
