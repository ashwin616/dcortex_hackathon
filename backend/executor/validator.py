"""Plan validator - the security and correctness boundary.

Everything upstream of this file is model output and is treated as untrusted.
Nothing downstream runs unless it passed through here.

Three jobs, in order of importance:

  1. Refuse anything not in the operation allowlist. The model cannot invent
     `run_sql`, `execute_python`, or a tool we removed last week.
  2. Refuse arguments that do not correspond to real data. A plan asking about
     C-9999 fails here rather than producing a confident answer about a crew
     member who does not exist.
  3. Repair the small stuff. Models reliably write `date` for `on_date` and
     `station` for `dep_station`; rejecting a plan over a synonym wastes a
     round trip that the daily token budget cannot spare.
"""
from __future__ import annotations

import re
from datetime import date

from backend.config import SETTINGS
from backend.domain.data import store
from backend.executor.registry import ALLOWED_OPERATIONS, OPERATIONS
from backend.planner.schemas import ExecutionPlan, PlanStep, ValidationIssue, ValidationResult

# Operations that need at least one of a set of arguments. Per-parameter
# "required" cannot express this: recommend_options is happy with a crew_id OR
# a pairing_id OR flight_ids, and useless with none of them.
REQUIRES_ONE_OF = {
    "recommend_options": ("crew_id", "pairing_id", "flight_ids"),
    "recommend_joint": ("crew_ids", "pairing_ids"),
    "recovery_options": ("pairing_id", "aircraft", "flight_ids"),
    "check_assignment": ("pairing_id", "flight_ids"),
    "delay_impact": ("pairing_id", "aircraft", "flight_no"),
    "get_pairing": ("pairing_id", "flight_id", "crew_id", "on_date", "aircraft"),
    "find_crew": ("crew_id", "rank", "base", "rating", "status", "name"),
}

# Entities the validator may copy into a step that is missing them. The values
# come from deterministic extraction over the question text, never from the
# model, so backfilling cannot introduce a fact the controller did not state.
BACKFILLABLE = ("crew_id", "pairing_id", "flight_ids", "on_date", "from_date",
                "aircraft", "flight_no", "station", "dep_station", "role")

# The dataset spells ranks exactly one way; models spell them five.
CANONICAL_ROLES = {
    "captain": "Captain", "cpt": "Captain", "capt": "Captain", "commander": "Captain",
    "first officer": "First Officer", "fo": "First Officer", "f/o": "First Officer",
    "co-pilot": "First Officer", "copilot": "First Officer",
    "senior cabin crew": "Senior Cabin Crew", "scc": "Senior Cabin Crew",
    "purser": "Senior Cabin Crew",
    "cabin crew": "Cabin Crew", "cc": "Cabin Crew", "flight attendant": "Cabin Crew",
}

CREW_RE = re.compile(r"^C-\d{3,4}$")
PAIRING_RE = re.compile(r"^P-\d{3,4}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T([01]\d|2[0-3]):[0-5]\d(:[0-5]\d)?Z?$")

# Synonyms models actually emit, mapped per operation where the target differs.
ALIASES = {
    "date": ("on_date", "from_date", "as_of"),
    "dates": ("on_date",),
    "day": ("on_date",),
    "station": ("dep_station", "station"),
    "origin": ("dep_station",),
    "from_station": ("dep_station",),
    "destination": ("arr_station",),
    "to_station": ("arr_station",),
    "crew": ("crew_id",),
    "crew_ids": ("crew_id",),
    "pairing": ("pairing_id",),
    "flight": ("flight_no", "flight_ids"),
    "flight_id": ("flight_ids",),
    "tail": ("aircraft",),
    "aircraft_id": ("aircraft",),
    "start": ("start_utc",),
    "end": ("end_utc",),
    "within": ("within_days",),
    "threshold": ("duty_threshold_7d",),
    "delay": ("delay_minutes", "delay_hours"),
}


def _coerce_scalar(op_name: str, key: str, value):
    """Nudge a value into the shape the tool expects, without inventing data."""
    if key == "flight_ids" and isinstance(value, str):
        return [value]
    if key in ("start_utc", "end_utc", "after_utc", "before_utc") and isinstance(value, str):
        if TS_RE.match(value):
            return value[11:16]
        if re.match(r"^\d{1,2}$", value):
            return f"{int(value):02d}:00"
    if key in ("on_date", "from_date", "as_of") and isinstance(value, str) and len(value) > 10:
        if TS_RE.match(value):
            return value[:10]
    if key in ("delay_minutes", "within_days", "delay_hours", "duty_threshold_7d"):
        try:
            return float(value) if "." in str(value) else int(value)
        except (TypeError, ValueError):
            return value
    if key == "role" and isinstance(value, str):
        return CANONICAL_ROLES.get(value.strip().lower(), value)
    if key == "rank" and isinstance(value, str):
        return CANONICAL_ROLES.get(value.strip().lower(), value)
    if isinstance(value, str) and key in ("crew_id", "pairing_id", "aircraft",
                                          "flight_no", "dep_station", "arr_station",
                                          "station", "base", "rating"):
        return value.strip().upper()
    return value


def _declared_types() -> dict:
    """{operation: {param: default}} - the shape each tool actually expects."""
    import inspect

    from backend.tools import TOOLS
    out: dict = {}
    for name, fn in TOOLS.items():
        sig = inspect.signature(fn)
        out[name] = {p: (par.default if par.default is not inspect._empty else None)
                     for p, par in sig.parameters.items()}
    return out


_TYPES = None


def _coerce_to_signature(op_name: str, key: str, value):
    """Nudge a value towards the type the operation declares.

    The planner is a language model: it will send "45" for a float and [4,5,6]
    for an int. Both are unambiguous. Refusing them wastes a round trip and
    sometimes a whole answer.
    """
    global _TYPES
    if _TYPES is None:
        _TYPES = _declared_types()
    default = _TYPES.get(op_name, {}).get(key, "__missing__")
    if default == "__missing__":
        return value

    # the tool wants a list
    if isinstance(default, list) and not isinstance(value, list):
        return [value]
    # the tool wants one scalar but got a list - take the first usable entry
    if isinstance(value, list) and not isinstance(default, list) \
            and default is not None and not isinstance(default, bool):
        value = next((v for v in value if v not in (None, "")), None)
    if value is None:
        return None
    # numbers arriving as strings
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(value)
    if isinstance(default, (int, float)) and isinstance(value, str):
        try:
            return float(value) if isinstance(default, float) or "." in value \
                else int(value)
        except ValueError:
            return value
    if isinstance(default, float) and isinstance(value, int):
        return float(value)
    return value


def _repair_keys(step: PlanStep, repaired: list) -> None:
    """Map synonym argument names onto the operation's real parameters."""
    op = OPERATIONS[step.operation]
    fixed = {}
    for k, v in step.args.items():
        if k in op.params:
            fixed[k] = v
            continue
        target = None
        for cand in ALIASES.get(k, ()):
            if cand in op.params and cand not in step.args:
                target = cand
                break
        if target:
            fixed[target] = v
            repaired.append(f"{step.id}: {k} -> {target}")
        else:
            repaired.append(f"{step.id}: dropped unknown arg {k}")
    step.args = fixed


def _within_days(value: str, start: str, end: str, slack: int) -> bool:
    """Is this date close enough to the dataset to be a real question?"""
    from datetime import date as _date, timedelta as _td
    try:
        d = _date.fromisoformat(value)
    except ValueError:
        return False
    return (_date.fromisoformat(start) - _td(days=slack) <= d
            <= _date.fromisoformat(end) + _td(days=slack))


def _backfill(step: PlanStep, entities: dict, repaired: list) -> None:
    """Fill arguments the planner omitted from entities we extracted ourselves."""
    op = OPERATIONS[step.operation]
    for key in BACKFILLABLE:
        if key in step.args or key not in op.params:
            continue
        val = entities.get(key)
        if val is None and key in ("on_date", "from_date"):
            val = entities.get("date")
        if val is None and key in ("station", "dep_station"):
            val = entities.get("station")
        if val is None:
            continue
        step.args[key] = val
        repaired.append(f"{step.id}: backfilled {key}={val}")


def validate(plan: ExecutionPlan, entities: dict | None = None) -> ValidationResult:
    """Shape, allowlist, identifiers, dates, dependencies, size.

    `entities` are the deterministically extracted facts from the question.
    They are used only to fill gaps the planner left, never to override it.
    """
    s = store()
    issues: list[ValidationIssue] = []
    repaired: list[str] = []

    if plan.abstain_reason and not plan.steps:
        return ValidationResult(ok=True, plan=plan, issues=[], repaired=[])

    if not plan.steps:
        issues.append(ValidationIssue(message="plan has no steps and no abstain_reason"))
        return ValidationResult(ok=False, plan=plan, issues=issues)

    if len(plan.steps) > SETTINGS.max_plan_steps:
        issues.append(ValidationIssue(
            message=f"{len(plan.steps)} steps exceeds the limit of {SETTINGS.max_plan_steps}"))
        return ValidationResult(ok=False, plan=plan, issues=issues)

    seen_ids: set[str] = set()
    for step in plan.steps:
        # --- allowlist: the hard boundary ---------------------------------
        if step.operation not in ALLOWED_OPERATIONS:
            issues.append(ValidationIssue(
                step_id=step.id, field="operation",
                message=f"unknown operation {step.operation!r}; "
                        f"allowed: {', '.join(sorted(ALLOWED_OPERATIONS))}"))
            continue

        if step.id in seen_ids:
            issues.append(ValidationIssue(step_id=step.id, message="duplicate step id"))
            continue
        seen_ids.add(step.id)

        _repair_keys(step, repaired)
        op = OPERATIONS[step.operation]

        needs_any = REQUIRES_ONE_OF.get(step.operation)
        if entities and needs_any and not any(step.args.get(k) for k in needs_any):
            _backfill(step, entities, repaired)

        for k in list(step.args):
            v = _coerce_scalar(step.operation, k, step.args[k])
            step.args[k] = _coerce_to_signature(step.operation, k, v)

        # --- required arguments ------------------------------------------
        supplied = set(step.args) | set(_dependency_supplied(step, plan))
        for req in op.required:
            if req not in supplied:
                issues.append(ValidationIssue(
                    step_id=step.id, field=req,
                    message=f"{step.operation} requires {req}"))

        supplied_truthy = {k for k in supplied if step.args.get(k, True)}
        if needs_any and not (set(needs_any) & supplied_truthy):
            issues.append(ValidationIssue(
                step_id=step.id, field="args",
                message=f"{step.operation} needs at least one of "
                        f"{', '.join(needs_any)} - none supplied and no upstream "
                        f"step provides one"))

        # --- identifiers must exist in the dataset ------------------------
        for k, v in step.args.items():
            if k == "crew_id" and isinstance(v, str):
                if not CREW_RE.match(v):
                    issues.append(ValidationIssue(step_id=step.id, field=k,
                                                  message=f"{v!r} is not a crew id"))
                elif v not in s.crew:
                    issues.append(ValidationIssue(
                        step_id=step.id, field=k,
                        message=f"crew {v} is not in the dataset"))
            elif k == "pairing_id" and isinstance(v, str):
                if not PAIRING_RE.match(v):
                    issues.append(ValidationIssue(step_id=step.id, field=k,
                                                  message=f"{v!r} is not a pairing id"))
                elif v not in s.pairing_by_id:
                    issues.append(ValidationIssue(
                        step_id=step.id, field=k,
                        message=f"pairing {v} is not in the dataset"))
            elif k == "flight_ids" and isinstance(v, list):
                bad = [f for f in v if f not in s.flight_by_id]
                if bad:
                    issues.append(ValidationIssue(
                        step_id=step.id, field=k,
                        message=f"unknown flight ids: {', '.join(map(str, bad[:4]))}"))
            elif k in ("on_date", "from_date", "as_of") and isinstance(v, str):
                if not DATE_RE.match(v):
                    issues.append(ValidationIssue(step_id=step.id, field=k,
                                                  message=f"{v!r} is not YYYY-MM-DD"))
                elif not (s.dates[0] <= v <= s.dates[-1]) and k != "as_of":
                    # A near miss - "yesterday", or the Monday after the week -
                    # is a real question whose answer is "no data for that day".
                    # The tools say so. Only a wild date is a fabrication.
                    near = _within_days(v, s.dates[0], s.dates[-1], 7)
                    issues.append(ValidationIssue(
                        step_id=step.id, field=k, repairable=near,
                        message=(f"{v} is just outside the dataset "
                                 f"({s.dates[0]}..{s.dates[-1]}); the tools will "
                                 f"report no data for it"
                                 if near else
                                 f"{v} is far outside the schedule week "
                                 f"{s.dates[0]}..{s.dates[-1]}")))
            elif k in ("start_utc", "end_utc", "after_utc", "before_utc") and isinstance(v, str):
                if not TIME_RE.match(v):
                    issues.append(ValidationIssue(step_id=step.id, field=k,
                                                  message=f"{v!r} is not HH:MM UTC"))
            elif k in ("station", "dep_station", "arr_station", "base") and isinstance(v, str):
                if v not in s.stations:
                    issues.append(ValidationIssue(
                        step_id=step.id, field=k,
                        message=f"{v} is not a station in this network "
                                f"({', '.join(sorted(s.stations))})"))

        # --- dependency graph --------------------------------------------
        for dep in step.depends_on:
            if dep not in {x.id for x in plan.steps}:
                issues.append(ValidationIssue(step_id=step.id, field="depends_on",
                                              message=f"unknown step {dep!r}"))
            elif dep == step.id:
                issues.append(ValidationIssue(step_id=step.id, field="depends_on",
                                              message="step depends on itself"))

    if not _acyclic(plan):
        issues.append(ValidationIssue(message="dependency cycle in plan"))

    blocking = [i for i in issues if not i.repairable]
    return ValidationResult(ok=not blocking, plan=plan, issues=issues, repaired=repaired)


def _dependency_supplied(step: PlanStep, plan: ExecutionPlan) -> set:
    """Arguments a prior step is expected to fill in at execution time.

    A required parameter may legitimately be absent from a step's own args if
    an upstream step will produce it - "recommend cover for whoever that sick
    call affects" is a two-step plan where step 2 has no crew_id of its own.
    """
    from backend.executor.executor import CARRY_FORWARD          # single source of truth
    inheritable = set(CARRY_FORWARD.get(step.operation, ()))
    return inheritable if step.depends_on else set()


def _acyclic(plan: ExecutionPlan) -> bool:
    graph = {s.id: list(s.depends_on) for s in plan.steps}
    state: dict = {}

    def visit(n: str) -> bool:
        if state.get(n) == 1:
            return False
        if state.get(n) == 2:
            return True
        state[n] = 1
        for m in graph.get(n, []):
            if m in graph and not visit(m):
                return False
        state[n] = 2
        return True

    return all(visit(n) for n in graph)


def repair_prompt(result: ValidationResult) -> str:
    """What to tell the planner so its second attempt is better than its first."""
    s = store()
    lines = ["Your execution plan was rejected by the validator:"]
    for i in result.blocking:
        lines.append(f"  - {i.step_id or 'plan'}: {i.message}")
    lines.append("")
    lines.append(f"Valid dates: {s.dates[0]}..{s.dates[-1]}. "
                 f"Stations: {', '.join(sorted(s.stations))}.")
    lines.append("Return a corrected plan as JSON only. If the question cannot be "
                 "answered with the available operations, return "
                 '{"intent":"abstain","entities":{},"steps":[],'
                 '"abstain_reason":"<what you would need>"}.')
    return "\n".join(lines)
