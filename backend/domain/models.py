"""Typed results that cross the LLM / deterministic boundary.

Every structure here carries the *arithmetic*, not just a verdict. The LLM
renders these into prose; it never recomputes them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Any


def _jsonable(v: Any) -> Any:
    if isinstance(v, (datetime,)):
        return v.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "to_dict"):
        return v.to_dict()
    return v


class Jsonable:
    def to_dict(self) -> dict:
        return {k: _jsonable(v) for k, v in asdict(self).items()}


@dataclass
class RuleResult(Jsonable):
    """One rule evaluated against one concrete situation.

    `limit` / `actual` / `margin` are what make the explanation derivable
    instead of narrated. margin > 0 means headroom, < 0 means breach.
    """
    rule_id: str
    passed: bool
    detail: str
    limit: float | None = None
    actual: float | None = None
    margin: float | None = None
    on_date: str | None = None
    window: str | None = None

    @property
    def short(self) -> str:
        return f"{self.rule_id}: {self.detail}"


@dataclass
class Duty(Jsonable):
    """A single duty period (one day of one pairing) for one crew member."""
    crew_id: str
    date: str
    report_utc: str
    release_utc: str
    duty_hours: float
    flight_hours: float
    pairing_id: str
    flights: list[str] = field(default_factory=list)

    @property
    def sectors(self) -> int:
        return len(self.flights)


@dataclass
class Verdict(Jsonable):
    """Result of checking one crew member against one proposed assignment."""
    crew_id: str
    legal: bool
    rules: list[RuleResult] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def rules_checked(self) -> list[str]:
        seen, out = set(), []
        for r in self.rules:
            if r.rule_id not in seen:
                seen.add(r.rule_id)
                out.append(r.rule_id)
        return out


@dataclass
class Option(Jsonable):
    """One ranked resolution option."""
    action: str
    crew_id: str | None
    legal: bool
    cost_inr: int
    delay_hours: float
    rules_checked: list[str] = field(default_factory=list)
    reasoning: str = ""
    rank: int = 0
    cost_breakdown: dict = field(default_factory=dict)
    risk_score: float | None = None


def _prune(v: Any, max_items: int) -> Any:
    """Cap list lengths recursively, leaving a marker for what was dropped."""
    if isinstance(v, dict):
        return {k: _prune(x, max_items) for k, x in v.items()}
    if isinstance(v, list):
        head = [_prune(x, max_items) for x in v[:max_items]]
        if len(v) > max_items:
            head.append(f"...+{len(v) - max_items} more (full list in the evidence panel)")
        return head
    return v


@dataclass
class ToolResult(Jsonable):
    """The envelope every tool returns. `evidence` is what the grounding
    validator checks the model's answer against."""
    tool: str
    data: Any
    evidence: list[str] = field(default_factory=list)
    rules_evaluated: list[dict] = field(default_factory=list)
    note: str | None = None

    def for_model(self, max_items: int = 60, max_chars: int = 14000) -> dict:
        """A slimmed view for the prompt.

        Three rules, learned the hard way:

        1. Never truncate the list the question is about. A closure result
           holding 13 flights, cut to 4 with a "+9 more" marker, made the
           narrator abstain - correctly, because it will not invent the rest.
           The answer was destroyed by the context trimming, not by the model.
        2. Never stringify. An oversized payload once collapsed the whole
           `data` object into a clipped string, which is unparseable and
           unciteable.
        3. When something must go, drop the largest *secondary* field and say
           which, so the loss is visible rather than silent.

        Rules that passed collapse to a count - a breach is what needs
        explaining, and forty passing checks only spend context.
        """
        import json as _json

        breaches = [r for r in self.rules_evaluated if not r.get("passed")]
        passed = len(self.rules_evaluated) - len(breaches)
        data = _prune(_jsonable(self.data), max_items)
        out = {
            "tool": self.tool,
            "data": data,
            "evidence_count": len(self.evidence),
        }
        if self.rules_evaluated:
            out["rule_breaches"] = _prune(breaches, max_items)
            out["rules_passed_count"] = passed
            # A passing rule still carries the answer to "how much room is
            # left?" - margin IS the headroom. Collapsing every pass to a count
            # made "is this legal?" answerable and "how much headroom does that
            # leave?" an abstention, on the same tool result. The detail string
            # is dropped (it restates the numbers) and only rules that computed
            # something are kept, so this stays cheap.
            # Named in the controller's words, not the engine's. Emitted as
            # `margin` the model read it as an internal field and abstained on
            # "how much headroom is left?" while holding the answer - `margin`
            # IS the headroom, and nothing said so.
            # `detail` is kept, not dropped. Without it the payload said
            # RULE-DUTY-02, 60, 9.5, 50.5 - and the model, not knowing that
            # RULE-DUTY-02 *is* the seven-day rule, quoted 50.5 and then
            # abstained for want of "the 7-day figure". The sentence the
            # engine already wrote says which window it measured.
            headroom = [{"rule_id": r.get("rule_id"), "on_date": r.get("on_date"),
                         "window": r.get("window"), "limit": r.get("limit"),
                         "used": r.get("actual"),
                         "headroom_remaining": r.get("margin"),
                         "detail": r.get("detail")}
                        for r in self.rules_evaluated
                        if r.get("passed") and r.get("limit") is not None]
            if headroom:
                out["rules_passed_margins"] = _prune(headroom, max_items)
        if self.note:
            out["note"] = self.note

        if len(_json.dumps(out, default=str)) <= max_chars:
            return out

        # Too big: shed the heaviest secondary fields one at a time, keeping
        # every identifier list intact, and record what was dropped.
        protected = {"error", "options", "assignments", "affected_flights",
                     "uncrewed_now", "also_at_risk", "legs_needing_cover",
                     "flights_needing_recrew_or_cancel", "covering", "issues",
                     "per_flight_assessment", "crew", "reserves", "flights",
                     "certifications", "legal", "breach"}
        omitted = []
        while isinstance(out["data"], dict) and                 len(_json.dumps(out, default=str)) > max_chars:
            droppable = {k: len(_json.dumps(v, default=str))
                         for k, v in out["data"].items() if k not in protected}
            if not droppable:
                break
            heaviest = max(droppable, key=droppable.get)
            out["data"].pop(heaviest)
            omitted.append(heaviest)
        if omitted:
            out["omitted_fields"] = omitted
            out["note"] = ((out.get("note") or "") +
                           f" Fields omitted for size: {', '.join(omitted)}. "
                           "Everything present is complete.").strip()
        return out
