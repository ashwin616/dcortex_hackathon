"""Deterministic workflow engine.

Runs a validated plan: topological order, output of one step feeding the
arguments of the next, evidence accumulated as it goes. No LLM, no network, no
surprises - given the same plan and the same dataset it produces the same
result, which is what makes the audit trail worth keeping.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from backend import tools as T
from backend.domain.models import ToolResult
from backend.executor.registry import OPERATIONS
from backend.planner.schemas import ExecutionPlan, StepResult

# What each operation may inherit from an upstream step, declared per operation
# rather than globally. The distinction matters: in recommend_options `crew_id`
# is the person being replaced, but in check_assignment it is the candidate
# being tested. Carrying a sick crew member's id into a legality check would
# silently answer a different question than the one asked, so operations that
# would misread an inherited value simply do not accept it.
# Plural parameters are gathered across every dependency, not taken from the
# first: "both captains are sick" is a two-step lookup feeding one joint plan,
# and a single pairing_id would silently solve half the problem.
PLURAL_OF = {"pairing_ids": "pairing_id", "crew_ids": "crew_id"}

CARRY_FORWARD = {
    "recommend_options": ("crew_id", "pairing_id", "from_date", "role", "flight_ids"),
    "recovery_options": ("pairing_id", "on_date", "flight_ids"),
    "recommend_joint": ("pairing_ids", "crew_ids", "on_date"),
    "notification_facts": ("crew_id", "pairing_id", "from_date"),
    "sick_call_impact": ("crew_id", "pairing_id", "from_date"),
    "check_assignment": ("pairing_id", "flight_ids", "on_date"),
    "cancellation_impact": ("flight_ids",),
    "get_pairing": ("pairing_id", "crew_id", "on_date"),
    "duty_clock": ("crew_id", "on_date"),
    "risk_signal": ("crew_id",),
    "find_crew": ("crew_id",),
    "crew_near_limits": ("on_date",),
    "delay_impact": ("pairing_id", "on_date"),
    "morning_brief": ("on_date",),
}


class ExecutionContext:
    """Everything one plan produced, in the order it happened."""

    def __init__(self, plan: ExecutionPlan):
        self.plan = plan
        self.results: dict[str, ToolResult] = {}
        self.steps: list[StepResult] = []
        self.evidence: list[str] = []
        self.rules_evaluated: list[dict] = []
        self.duration_ms: float = 0.0
        self.parallel_waves: int = 0

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    @property
    def tool_results(self) -> list:
        return list(self.results.values())

    def errors(self) -> list[str]:
        return [f"{s.operation}: {s.error}" for s in self.steps if not s.ok]

    def to_dict(self) -> dict:
        return {
            "intent": self.plan.intent,
            "plan": self.plan.summary(),
            "steps": [s.model_dump() for s in self.steps],
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "evidence": sorted(set(self.evidence)),
            "rules_evaluated": self.rules_evaluated,
            "duration_ms": round(self.duration_ms, 2),
            "parallel_waves": self.parallel_waves,
            "ok": self.ok,
        }

    def for_model(self) -> dict:
        """The slim view handed to the narrator."""
        return {
            "intent": self.plan.intent,
            "operations": [{"id": s.id, "operation": s.operation, "args": s.args,
                            "ok": s.ok, "error": s.error} for s in self.steps],
            "results": {k: v.for_model() for k, v in self.results.items()},
        }


def execution_levels(plan: ExecutionPlan) -> list:
    """Steps grouped into waves that may run concurrently.

    "Both captains are sick" is two independent pairing lookups feeding one
    joint plan. Run in sequence that is two round trips of latency for no
    reason - they share no state and neither can affect the other.
    """
    by_id = {s.id: s for s in plan.steps}
    remaining = {s.id: {d for d in s.depends_on if d in by_id} for s in plan.steps}
    levels: list = []
    done: set = set()
    while remaining:
        ready = [sid for sid, deps in remaining.items() if deps <= done]
        if not ready:                       # cycle: validator rejects these,
            ready = list(remaining)         # but never trust it twice
        levels.append([by_id[sid] for sid in ready])
        done |= set(ready)
        for sid in ready:
            remaining.pop(sid)
    return levels


def topological_order(plan: ExecutionPlan) -> list:
    """Steps in dependency order, stable for steps that do not depend on each other."""
    by_id = {s.id: s for s in plan.steps}
    ordered, seen, visiting = [], set(), set()

    def visit(sid: str):
        if sid in seen or sid not in by_id:
            return
        if sid in visiting:                     # validator rejects cycles, but
            return                              # never trust a second time
        visiting.add(sid)
        for dep in by_id[sid].depends_on:
            visit(dep)
        visiting.discard(sid)
        seen.add(sid)
        ordered.append(by_id[sid])

    for s in plan.steps:
        visit(s.id)
    return ordered


def resolve_args(step, context: ExecutionContext) -> dict:
    """Fill arguments this step left blank from the steps it depends on.

    Only keys this operation declares as inheritable are carried, and an
    argument the plan set explicitly is never overwritten.
    """
    args = dict(step.args)
    op = OPERATIONS[step.operation]
    inheritable = [k for k in CARRY_FORWARD.get(step.operation, ())
                   if k in op.params and not args.get(k)]
    if not inheritable:
        return args

    collected: dict = {}
    for dep in step.depends_on:
        prev = context.results.get(dep)
        if prev is None:
            continue
        source = prev.data if isinstance(prev.data, dict) else {}
        for key in list(inheritable):
            singular = PLURAL_OF.get(key)
            val = _extract(source, singular or key)
            if val is None:
                continue
            if singular:
                bucket = collected.setdefault(key, [])
                if val not in bucket:
                    bucket.append(val)
            else:
                args[key] = val
                inheritable.remove(key)
    for key, values in collected.items():
        if values:
            args[key] = values
    return args


def _extract(data: dict, key: str):
    """Pull one carry-forward value out of an upstream result."""
    if key in data and data[key] not in (None, [], ""):
        return data[key]
    aliases = {
        "pairing_id": ("pairings_affected", "pairing_id"),
        "crew_id": ("crew_id",),
        "from_date": ("date", "window_end"),
        "flight_ids": ("uncrewed_now", "affected_flights",
                       "flights_needing_recrew_or_cancel"),
        "role": ("role_on_pairing", "role"),
    }
    for alt in aliases.get(key, ()):
        v = data.get(alt)
        if isinstance(v, list) and v:
            return v[0] if key in ("pairing_id", "crew_id") else v
        if v:
            return v
    # one level down: {"covering": {"pairing_id": ...}}
    for nested in ("covering", "impact"):
        sub = data.get(nested)
        if isinstance(sub, dict) and sub.get(key):
            return sub[key]
    # a list of records: {"pairings": [{"pairing_id": ...}]}. Only unambiguous
    # when the upstream query returned exactly one - picking the first of many
    # would answer a question nobody asked.
    for nested in ("pairings", "flights", "crew", "reserves"):
        rows = data.get(nested)
        if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
            if rows[0].get(key):
                return rows[0][key]
    return None


def _run_step(step, ctx: ExecutionContext):
    s0 = time.perf_counter()
    args = resolve_args(step, ctx)
    result = T.call(step.operation, **args)
    elapsed = (time.perf_counter() - s0) * 1000
    failed = isinstance(result.data, dict) and "error" in result.data
    return step, args, result, elapsed, failed


def execute(plan: ExecutionPlan, max_workers: int = 4) -> ExecutionContext:
    """Run a validated plan. Never raises - a failed step is recorded, not thrown.

    Independent steps run concurrently, dependent ones in order. The kernel is
    pure and reads an immutable snapshot, so concurrency here needs no locking;
    results are collected per wave and ordering stays deterministic because a
    wave's members cannot see each other.
    """
    ctx = ExecutionContext(plan)
    t0 = time.perf_counter()

    for wave in execution_levels(plan):
        if len(wave) == 1:
            outcomes = [_run_step(wave[0], ctx)]
        else:
            ctx.parallel_waves += 1
            with ThreadPoolExecutor(max_workers=min(max_workers, len(wave))) as pool:
                outcomes = list(pool.map(lambda s: _run_step(s, ctx), wave))
        # Collected after the wave so ordering does not depend on thread timing.
        for step, args, result, elapsed, failed in outcomes:
            ctx.results[step.id] = result
            ctx.evidence += result.evidence
            ctx.rules_evaluated += result.rules_evaluated
            ctx.steps.append(StepResult(
                id=step.id, operation=step.operation, args=args,
                ok=not failed, error=result.data.get("error") if failed else None,
                duration_ms=round(elapsed, 2), evidence_count=len(result.evidence)))

    ctx.duration_ms = (time.perf_counter() - t0) * 1000
    return ctx
