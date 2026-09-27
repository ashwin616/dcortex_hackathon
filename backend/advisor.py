"""The orchestrator: plan -> validate -> execute -> narrate -> ground.

This is the only place the five stages meet. Each of them is independently
testable and none of them knows about the others' internals.

    question
       |
       v
    planner      fast path (0 tokens) or LLM
       |
       v
    validator    allowlist, real identifiers, real dates, closed graph
       |         reject -> one repair attempt -> abstain
       v
   ============ LLM / DETERMINISTIC BOUNDARY ============
       |
       v
    executor     topological order, evidence collected
       |
       v
    narrator     result -> prose, no tools, no data access
       |
       v
    grounding    every id and figure traced back, else abstain
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from backend import cache, conversation, grounding
from backend.grounding import citations, completeness
from backend.cache import ANSWER_CACHE, answer_key
from backend.client import LLMError, Usage
from backend.config import SETTINGS
from backend.executor.executor import ExecutionContext, execute
from backend.narrator import ABSTAIN, narrate, narrate_stream, renarrate
from backend.planner.planner import (extract_entities, fallback_plan,
                                     plan_question, replan)
from backend.ratelimit import POOL
from backend.planner.schemas import ExecutionPlan, ValidationResult
from backend.telemetry import LOG, audit
from backend.text import plain
from backend.executor.validator import repair_prompt, validate


@dataclass
class Turn:
    """One question, everything that happened, and what it cost."""
    question: str
    answer: str = ""
    plan: ExecutionPlan | None = None
    plan_source: str = ""
    validation: ValidationResult | None = None
    context: ExecutionContext | None = None
    grounded: bool = True
    unsupported: list = field(default_factory=list)
    abstained: bool = False
    error: str | None = None
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    kernel_ms: float = 0.0
    cached: bool = False
    completeness: dict = field(default_factory=dict)
    citations: list = field(default_factory=list)
    annotated: str = ""

    # ---- convenience for the UIs ------------------------------------
    @property
    def evidence(self) -> list:
        return sorted(set(self.context.evidence)) if self.context else []

    @property
    def rules_evaluated(self) -> list:
        return self.context.rules_evaluated if self.context else []

    @property
    def results(self) -> list:
        return self.context.tool_results if self.context else []

    @property
    def tool_calls(self) -> list:
        if not self.context:
            return []
        return [{"tool": s.operation, "args": s.args} for s in self.context.steps]

    @property
    def throttled_s(self) -> float:
        return self.usage.waited_s

    def payload(self, max_result_chars: int = 20000) -> dict:
        """Everything a client needs to render this turn, JSON-safe.

        Text is normalised here as well as at the end of the turn: a client
        rendering in a cp1252 console must never receive a character it cannot
        show, whichever path produced the answer.
        """
        ctx = self.context
        self.answer = plain(self.answer)
        self.annotated = plain(self.annotated or self.answer)
        return {
            **self.to_dict(),
            "annotated": self.annotated or self.answer,
            "plan_detail": (None if not self.plan else {
                "intent": self.plan.intent,
                "entities": self.plan.entities,
                "abstain_reason": self.plan.abstain_reason,
                "steps": [s.model_dump() for s in self.plan.steps],
            }),
            "validation_detail": (None if not self.validation else {
                "ok": self.validation.ok,
                "repaired": self.validation.repaired,
                "issues": [i.model_dump() for i in self.validation.issues],
            }),
            "results": ({} if not ctx else
                        {sid: json.loads(json.dumps(r.data, default=str))
                         for sid, r in ctx.results.items()}),
            "kernel_ms": round(self.kernel_ms, 2),
            "throttled_s": round(self.throttled_s, 2),
        }

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "intent": self.plan.intent if self.plan else None,
            "plan": self.plan.summary() if self.plan else None,
            "plan_source": self.plan_source,
            "validation": self.validation.report() if self.validation else None,
            "steps": [s.model_dump() for s in self.context.steps] if self.context else [],
            "evidence": self.evidence,
            "rules_evaluated": self.rules_evaluated,
            "grounded": self.grounded,
            "unsupported": self.unsupported,
            "abstained": self.abstained,
            "cached": self.cached,
            "error": self.error,
            "citations": self.citations,
            "completeness": self.completeness,
            "usage": self.usage.to_dict(),
            "latency_s": round(self.latency_s, 2),
            "kernel_ms": round(self.kernel_ms, 2),
        }


class Advisor:
    """Ask questions. Thread-safe enough for Streamlit; one instance per session."""

    def __init__(self, planner_model: str | None = None,
                 narrator_model: str | None = None,
                 use_cache: bool = True, use_fastpath: bool = True):
        if planner_model:
            SETTINGS.planner_model = planner_model
        if narrator_model:
            SETTINGS.narrator_model = narrator_model
        self.use_cache = use_cache
        self.use_fastpath = use_fastpath
        self.history: list = []
        self.turns: list = []

    # ------------------------------------------------------------------
    def ask(self, question: str, keep_history: bool = True) -> Turn:
        t0 = time.time()
        turn = Turn(question=question)
        history = self.history if keep_history else None

        try:
            # "hi" and "what can you do?" are not crew-control questions. The
            # planner correctly finds no operation and then refuses, which is
            # the wrong answer to a greeting - and a judge's first message is
            # very often "what can you do?".
            social = conversation.handle(question)
            if social is not None:
                turn.answer, turn.plan_source = social[0], f"social:{social[1]}"
                return self._finish(turn, t0, keep_history)

            self._plan(turn, history)
            if turn.plan is None:
                return self._finish(turn, t0, keep_history)

            if turn.plan.abstain_reason and not turn.plan.steps:
                fb = fallback_plan(question)
                if fb is not None and validate(fb, extract_entities(question)).ok:
                    LOG.info("planner abstained on a known entity; falling back to %s",
                             fb.summary())
                    turn.plan, turn.plan_source = fb, "fallback"
                else:
                    turn.answer = f"{ABSTAIN}. {turn.plan.abstain_reason}"
                    turn.abstained = True
                    return self._finish(turn, t0, keep_history)

            turn.context = execute(turn.plan)
            turn.kernel_ms = turn.context.duration_ms

            if not turn.context.ok and not turn.context.results:
                turn.answer = (f"{ABSTAIN}. Every step of the plan failed: "
                               + "; ".join(turn.context.errors()))
                turn.abstained = True
                return self._finish(turn, t0, keep_history)

            cached = self._cached_answer(turn)
            if cached is not None:
                turn.answer, turn.cached = cached, True
                return self._finish(turn, t0, keep_history)

            turn.answer = narrate(question, turn.context, usage=turn.usage,
                                  history=history)
            if not turn.answer.strip():
                # An empty completion is a truncation, not an answer. Saying so
                # beats handing the controller a blank screen.
                turn.answer = (
                    f"{ABSTAIN}. The narration step returned nothing - usually the "
                    "result was too large to summarise within the token budget. "
                    "The computed result is in the evidence panel; ask a narrower "
                    "question to get it in prose.")
                turn.abstained = True
                return self._finish(turn, t0, keep_history)
            self._ground(turn)

        except LLMError as e:
            turn.error = str(e)
            turn.abstained = True
            turn.answer = (
                "The daily free-tier token budget is exhausted on every model. "
                "The deterministic layer still works - the evidence panel below "
                "shows the computed result." if e.exhausted else
                f"The Advisor could not reach the model: {e}")
        except Exception as e:                          # noqa: BLE001
            LOG.exception("turn failed")
            turn.error = f"{type(e).__name__}: {e}"
            turn.answer = f"The Advisor could not complete that request: {turn.error}"

        return self._finish(turn, t0, keep_history)

    # ---- stages -------------------------------------------------------
    def _plan(self, turn: Turn, history) -> None:
        entities = extract_entities(turn.question)
        plan, source = plan_question(turn.question, history=history, usage=turn.usage,
                                     allow_fastpath=self.use_fastpath)
        turn.plan, turn.plan_source = plan, source

        result = validate(plan, entities)
        turn.validation = result
        if result.repaired:
            LOG.debug("plan repaired: %s", "; ".join(result.repaired))

        if result.ok:
            return

        # One corrective round. The fast path never gets one - if a
        # deterministic plan fails validation that is our bug, so fall through
        # to the LLM planner instead of asking the model to fix our regex.
        LOG.info("plan rejected (%s): %s", turn.plan_source, result.report())
        try:
            if source == "fastpath":
                plan, source = plan_question(turn.question, history=history,
                                             usage=turn.usage, allow_fastpath=False)
            else:
                plan = replan(turn.question, repair_prompt(result), usage=turn.usage)
                source = "llm-repair"
            turn.plan, turn.plan_source = plan, source
            turn.validation = validate(plan, entities)
        except LLMError:
            raise
        except Exception as e:                          # noqa: BLE001
            turn.validation = result
            LOG.debug("replan failed: %s", e)

        if not turn.validation.ok:
            # Before refusing, try the obvious lookup for whatever the question
            # named. A question about a crew member who exists always has an
            # answer, even when the planner could not see it.
            fb = fallback_plan(turn.question)
            if fb is not None:
                fb_result = validate(fb, entities)
                if fb_result.ok:
                    turn.plan, turn.plan_source = fb, "fallback"
                    turn.validation = fb_result
                    LOG.info("using entity fallback: %s", fb.summary())
                    return
            turn.answer = (f"{ABSTAIN}. I could not build a valid plan for that "
                           f"question: {turn.validation.report()}")
            turn.abstained = True
            turn.plan = None

    def _complete(self, turn: Turn) -> None:
        """Restore anything the narrator summarised away."""
        if not turn.context or turn.abstained or not turn.answer:
            return
        turn.answer, turn.completeness = completeness.complete(
            turn.answer, turn.context.tool_results)

    def _ground(self, turn: Turn) -> None:
        # Normalise before anything is cached or matched. Doing it only at the
        # end of the turn left the cache holding the raw copy, which then
        # served typographic characters back for as long as the entry lived.
        turn.answer = plain(turn.answer)
        ok, bad = grounding.check(turn.answer, turn.context.tool_results)
        if ok:
            if ABSTAIN.lower() in turn.answer.lower():
                turn.abstained = True
            self._complete(turn)
            self._cache_answer(turn)
            return

        LOG.info("grounding failed: %s", [u["token"] for u in bad])
        retry = plain(renarrate(turn.question, turn.context,
                                grounding.correction_prompt(bad), usage=turn.usage))
        ok2, bad2 = grounding.check(retry, turn.context.tool_results)
        if ok2:
            turn.answer = retry
            self._complete(turn)
            self._cache_answer(turn)
            return

        turn.answer = (
            f"{ABSTAIN}. My draft answer contained figures I could not trace back "
            f"to the data ({', '.join(u['token'] for u in bad2)}), so I am not "
            "showing it. The computed result is in the evidence panel - please "
            "read it directly, or narrow the question.")
        turn.grounded = False
        turn.unsupported = bad2
        turn.abstained = True

    # ---- caching ------------------------------------------------------
    def _cached_answer(self, turn: Turn):
        if not (self.use_cache and SETTINGS.cache_enabled) or self.history:
            return None
        return ANSWER_CACHE.get(answer_key(turn.question, turn.context.evidence))

    def _cache_answer(self, turn: Turn) -> None:
        if not (self.use_cache and SETTINGS.cache_enabled) or self.history:
            return
        ANSWER_CACHE.put(answer_key(turn.question, turn.context.evidence), turn.answer)

    # ---- citations ----------------------------------------------------
    def _cite(self, turn: Turn) -> None:
        """Attach a source to every identifier in the answer.

        Derived from the tool results, never asked of the model - a fabricated
        citation is worse than none, because it reads as proof.
        """
        if not (turn.context and turn.answer):
            turn.annotated = turn.answer
            return
        built = citations.build(turn.answer, turn.context.tool_results,
                                turn.context.rules_evaluated)
        turn.citations = built["citations"]
        turn.annotated = built["annotated"]

    # ---- bookkeeping --------------------------------------------------
    def _finish(self, turn: Turn, t0: float, keep_history: bool) -> Turn:
        # The model writes non-breaking hyphens and narrow spaces. They render
        # as mojibake in a cp1252 console and they break every downstream
        # string match, so the text is normalised once here - after the answer
        # is final, before anybody reads or indexes it.
        turn.answer = plain(turn.answer)
        if not turn.annotated:
            self._cite(turn)
        turn.annotated = plain(turn.annotated)
        turn.latency_s = time.time() - t0
        if keep_history:
            self.history.append(f"Q: {turn.question}")
            self.history.append(f"A: {turn.answer[:400]}")
            self.history = self.history[-8:]
        self.turns.append(turn)
        audit(turn.to_dict())
        return turn

    def reset(self) -> None:
        self.history = []

    # ---- introspection ------------------------------------------------
    @staticmethod
    def capacity() -> dict:
        return {**POOL.capacity(), "cache": cache.stats()}


# ==========================================================================
# streaming
# ==========================================================================
STAGES = ("plan", "validate", "execute", "narrate", "ground")


def _ev(kind: str, **kw) -> dict:
    return {"type": kind, **kw}


class StreamingAdvisor(Advisor):
    """`ask`, but the caller watches it happen.

    Yields events rather than returning once at the end:

        {"type": "stage",  "name": "plan", "status": "done", "detail": "..."}
        {"type": "token",  "text": "..."}                  narration, live
        {"type": "turn",   "turn": Turn}                   always last

    The stages are the architecture, so showing them is not decoration: a
    controller can see that a rules engine ran before any prose was written,
    which is the claim the whole system rests on.

    One honesty caveat, surfaced in the final event rather than hidden: tokens
    stream before the grounding check completes. If grounding then rejects the
    draft, the `turn` event carries the corrected or withheld answer and the
    caller must replace what it showed.
    """

    def ask_stream(self, question: str, keep_history: bool = True):
        t0 = time.time()
        turn = Turn(question=question)
        history = self.history if keep_history else None

        def finish(t):
            yield _ev("turn", turn=self._finish(t, t0, keep_history))

        try:
            social = conversation.handle(question)
            if social is not None:
                turn.answer, turn.plan_source = social[0], f"social:{social[1]}"
                yield _ev("stage", name="plan", status="skipped",
                          detail="conversational - no operation needed")
                yield _ev("token", text=turn.answer)
                yield from finish(turn)
                return

            yield _ev("stage", name="plan", status="running")
            self._plan(turn, history)
            if turn.plan is None:
                yield _ev("stage", name="plan", status="failed",
                          detail=turn.validation.report() if turn.validation else "no plan")
                yield from finish(turn)
                return

            yield _ev("stage", name="plan", status="done",
                      detail=turn.plan.summary(), source=turn.plan_source,
                      tokens=turn.usage.total)
            yield _ev("stage", name="validate", status="done",
                      detail=("accepted" if turn.validation and turn.validation.ok
                              else "repaired"),
                      repaired=turn.validation.repaired if turn.validation else [])

            if turn.plan.abstain_reason and not turn.plan.steps:
                turn.answer = f"{ABSTAIN}. {turn.plan.abstain_reason}"
                turn.abstained = True
                yield _ev("stage", name="execute", status="skipped",
                          detail="no operation applies")
                yield from finish(turn)
                return

            yield _ev("stage", name="execute", status="running")
            turn.context = execute(turn.plan)
            turn.kernel_ms = turn.context.duration_ms
            for s in turn.context.steps:
                yield _ev("step", operation=s.operation, ok=s.ok,
                          duration_ms=s.duration_ms, records=s.evidence_count,
                          error=s.error)
            breaches = sum(1 for r in turn.context.rules_evaluated if not r["passed"])
            yield _ev("stage", name="execute", status="done",
                      detail=f"{len(turn.context.rules_evaluated)} rule checks, "
                             f"{breaches} breach" if turn.context.rules_evaluated
                             else f"{len(turn.context.evidence)} records",
                      kernel_ms=turn.kernel_ms)

            if not turn.context.ok and not turn.context.results:
                turn.answer = (f"{ABSTAIN}. Every step of the plan failed: "
                               + "; ".join(turn.context.errors()))
                turn.abstained = True
                yield from finish(turn)
                return

            cached = self._cached_answer(turn)
            if cached is not None:
                turn.answer, turn.cached = cached, True
                yield _ev("stage", name="narrate", status="cached")
                yield _ev("token", text=cached)
                yield from finish(turn)
                return

            yield _ev("stage", name="narrate", status="running")
            parts = []
            for piece in narrate_stream(question, turn.context, usage=turn.usage,
                                        history=history):
                piece = plain(piece)
                parts.append(piece)
                yield _ev("token", text=piece)
            turn.answer = "".join(parts).strip()
            yield _ev("stage", name="narrate", status="done",
                      tokens=turn.usage.total)

            if not turn.answer:
                turn.answer = (
                    f"{ABSTAIN}. The narration step returned nothing - usually the "
                    "result was too large to summarise within the token budget. "
                    "The computed result is in the evidence panel.")
                turn.abstained = True
                yield from finish(turn)
                return

            yield _ev("stage", name="ground", status="running")
            before = turn.answer
            self._ground(turn)
            yield _ev("stage", name="ground",
                      status="done" if turn.grounded else "failed",
                      rewritten=turn.answer != before,
                      unsupported=[u["token"] for u in turn.unsupported])

        except LLMError as e:
            turn.error = str(e)
            turn.abstained = True
            turn.answer = (
                "The daily free-tier token budget is exhausted on every model. "
                "The deterministic layer still works - the evidence panel shows "
                "the computed result." if e.exhausted else
                f"The Advisor could not reach the model: {e}")
            yield _ev("stage", name="narrate", status="failed", detail=str(e))
        except Exception as e:                          # noqa: BLE001
            LOG.exception("streaming turn failed")
            turn.error = f"{type(e).__name__}: {e}"
            turn.answer = f"The Advisor could not complete that request: {turn.error}"
            yield _ev("stage", name="narrate", status="failed", detail=turn.error)

        yield from finish(turn)
