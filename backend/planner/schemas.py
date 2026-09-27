"""The execution plan - the typed contract the LLM must produce.

This is the only structure that crosses from the model into the deterministic
half of the system, which is why it is validated twice: once by pydantic for
shape, then by `validator.py` for meaning (does this crew id exist, is that
date inside the schedule week, does the dependency graph close).
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

STEP_ID = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,23}$")


class PlanStep(BaseModel):
    id: str = Field(description="unique within the plan")
    operation: str
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not STEP_ID.match(v):
            raise ValueError(f"step id {v!r} must be a short identifier")
        return v

    @field_validator("args", mode="before")
    @classmethod
    def _args_null(cls, v):
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("args must be an object")
        # A model that means "unset" says so in several ways: null, the string
        # "none", or - the one that actually bit us - an empty list. An empty
        # container that counts as "argument supplied" silently starves the
        # operation of the identifier it needed.
        out = {}
        for k, x in v.items():
            if x is None or x == [] or x == {}:
                continue
            if isinstance(x, str) and x.strip().lower() in ("", "null", "none", "n/a"):
                continue
            out[k] = x
        return out


class ExecutionPlan(BaseModel):
    intent: str
    entities: dict[str, Any] = Field(default_factory=dict)
    steps: list[PlanStep] = Field(default_factory=list)
    abstain_reason: str | None = Field(
        default=None,
        description="set instead of steps when no operation can answer the question")

    model_config = {"extra": "ignore"}

    @property
    def operations(self) -> list[str]:
        return [s.operation for s in self.steps]

    def step(self, sid: str) -> PlanStep | None:
        return next((s for s in self.steps if s.id == sid), None)

    def summary(self) -> str:
        if self.abstain_reason:
            return f"abstain: {self.abstain_reason}"
        return " -> ".join(s.operation for s in self.steps) or "(empty plan)"


class ValidationIssue(BaseModel):
    step_id: str | None = None
    field: str | None = None
    message: str
    repairable: bool = False


class ValidationResult(BaseModel):
    ok: bool
    plan: ExecutionPlan | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)
    repaired: list[str] = Field(default_factory=list)

    @property
    def blocking(self) -> list[ValidationIssue]:
        return [i for i in self.issues if not i.repairable]

    def report(self) -> str:
        return "; ".join(
            f"{i.step_id or 'plan'}.{i.field or '-'}: {i.message}" for i in self.issues)


class StepResult(BaseModel):
    """One executed step, kept for the audit trail."""
    id: str
    operation: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error: str | None = None
    duration_ms: float = 0.0
    evidence_count: int = 0
    model_config = {"arbitrary_types_allowed": True}
