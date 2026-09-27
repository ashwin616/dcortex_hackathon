"""Per-model token budgets, and the pool that spreads work across them.

The free tier gives every model its own 8,000 tokens/min and 1,000 requests/day
allowance. A single-model client therefore idles at 8k TPM while three
identical buckets sit untouched; with four tool-capable models the ceiling is
32k TPM. This module is what turns that fact into throughput.

Two behaviours matter more than the arithmetic:

  * We *predict* the cost of a request and refuse to send it into a bucket that
    cannot pay. Waiting 400 ms for a bucket to refill beats a 429 and a blind
    retry, which spends latency and the daily request budget on nothing.
  * Every response carries authoritative `x-ratelimit-*` headers. Local
    estimates drift; the headers do not, so they overwrite our accounting on
    every call.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

from backend.config import MODEL_SPECS, SETTINGS, ModelSpec

WINDOW_S = 60.0

# Groq bills input + output. Measured against real `usage` on this workload:
# 3.53 chars/token for the planner prompt (mostly prose) and 3.05 for the
# narrator's (a JSON result, which tokenises densely). Take the dense end -
# under-estimating the prompt is what produces a 429, and the prompt is the
# larger half of every request.
CHARS_PER_TOKEN = 3.1


class OutputEstimator:
    """How many completion tokens a call will actually use.

    Reserving `max_output` treats every answer as if it will run to the token
    ceiling. Measured over 127 real calls the mean completion was 301 against a
    1,600 ceiling, so that assumption reserved 4,025 tokens for a call that
    cost 1,780 - which throttled each model to one call a minute and left the
    pool idling at a quarter of its capacity while questions failed.

    So the reservation is learned: an exponential mean of what completions
    really cost, with generous headroom over it, clamped to the caller's
    ceiling. Under-reserving costs a 429; over-reserving costs throughput.
    """

    def __init__(self, seed: float = 380.0, headroom: float = 2.0):
        self.mean = seed
        self.headroom = headroom
        self.samples = 0
        self._lock = threading.Lock()

    def observe(self, completion_tokens: int) -> None:
        if not completion_tokens or completion_tokens <= 0:
            return
        with self._lock:
            alpha = 0.35 if self.samples < 10 else 0.12
            self.mean = (1 - alpha) * self.mean + alpha * completion_tokens
            self.samples += 1

    def reserve_for(self, max_output: int) -> int:
        return int(min(max_output, max(200.0, self.mean * self.headroom)))

    def stats(self) -> dict:
        return {"mean_completion": round(self.mean, 1), "samples": self.samples}


OUTPUT = OutputEstimator()


def estimate_tokens(messages: list, extra_chars: int = 0, max_output: int = 700) -> int:
    """Predicted cost of a request, biased high but not absurdly so.

    Under-estimating causes a 429; over-estimating starves the pool. The prompt
    is measured exactly; only the completion is predicted.
    """
    chars = extra_chars
    for m in messages:
        chars += len(str(m.get("content") or ""))
        for c in m.get("tool_calls") or []:
            chars += len(str(c))
    return int(chars / CHARS_PER_TOKEN) + OUTPUT.reserve_for(max_output) + 64


def parse_reset(value: str | None) -> float:
    """Groq resets look like '547ms', '1m26.4s', '2h19m40.8s'."""
    if not value:
        return 0.0
    total, found = 0.0, False
    for num, unit in re.findall(r"([\d.]+)\s*(ms|s|m|h)", value):
        found = True
        n = float(num)
        total += {"ms": n / 1000, "s": n, "m": n * 60, "h": n * 3600}[unit]
    if found:
        return total
    try:
        return float(value)
    except ValueError:
        return 0.0


@dataclass
class Budget:
    """One model's live allowance."""
    spec: ModelSpec
    spent: list = field(default_factory=list)          # (timestamp, tokens)
    remaining_tokens: int | None = None                # authoritative, from headers
    tokens_reset_at: float = 0.0
    remaining_requests: int | None = None
    requests_reset_at: float = 0.0
    failures: int = 0
    cooldown_until: float = 0.0
    calls: int = 0
    tokens_billed: int = 0

    # ---- accounting ---------------------------------------------------
    def _prune(self, now: float) -> None:
        cutoff = now - WINDOW_S
        self.spent = [(t, n) for t, n in self.spent if t > cutoff]

    def used(self, now: float | None = None) -> int:
        now = now or time.time()
        self._prune(now)
        return sum(n for _, n in self.spent)

    def headroom(self, now: float | None = None) -> int:
        """Tokens we believe are available right now."""
        now = now or time.time()
        reserve = int(self.spec.tpm * SETTINGS.reserve_fraction)
        local = self.spec.tpm - reserve - self.used(now)
        if self.remaining_tokens is not None:
            if now >= self.tokens_reset_at:
                return max(local, self.spec.tpm - reserve)
            return min(local, self.remaining_tokens - reserve)
        return local

    def healthy(self, now: float | None = None) -> bool:
        now = now or time.time()
        if now < self.cooldown_until:
            return False
        if self.remaining_requests is not None and self.remaining_requests <= 1 \
                and now < self.requests_reset_at:
            return False
        return True

    def wait_for(self, need: int, now: float | None = None) -> float:
        """Seconds until this bucket could pay `need` tokens."""
        now = now or time.time()
        if self.headroom(now) >= need:
            return 0.0
        if not self.spent:
            return max(0.0, self.tokens_reset_at - now)
        # tokens free up as the oldest entries age out of the 60s window
        deficit = need - self.headroom(now)
        freed = 0
        for t, n in sorted(self.spent):
            freed += n
            if freed >= deficit:
                return max(0.0, t + WINDOW_S - now)
        return WINDOW_S

    # ---- updates ------------------------------------------------------
    def charge(self, tokens: int) -> None:
        self.spent.append((time.time(), tokens))

    def observe(self, headers, billed: int | None = None,
                completion: int | None = None) -> None:
        """Trust the server over our own estimate."""
        now = time.time()
        g = headers.get
        rt, rr = g("x-ratelimit-remaining-tokens"), g("x-ratelimit-remaining-requests")
        if rt is not None:
            try:
                self.remaining_tokens = int(rt)
                self.tokens_reset_at = now + parse_reset(g("x-ratelimit-reset-tokens"))
            except ValueError:
                pass
        if rr is not None:
            try:
                self.remaining_requests = int(rr)
                self.requests_reset_at = now + parse_reset(g("x-ratelimit-reset-requests"))
            except ValueError:
                pass
        self.calls += 1
        if completion:
            OUTPUT.observe(completion)
        if billed:
            self.tokens_billed += billed
            # replace the estimate with the true figure for this call
            if self.spent:
                t, _ = self.spent[-1]
                self.spent[-1] = (t, billed)

    def penalise(self, seconds: float) -> None:
        self.failures += 1
        self.cooldown_until = max(self.cooldown_until, time.time() + seconds)

    def stats(self) -> dict:
        return {"model": self.spec.id, "calls": self.calls,
                "tokens_billed": self.tokens_billed,
                "used_60s": self.used(), "headroom": self.headroom(),
                "remaining_requests": self.remaining_requests,
                "failures": self.failures}


class ModelPool:
    """Chooses which model serves a request, and when.

    Selection is: models that can do the job, are healthy, and can pay now -
    ordered by the role's preference, then by spare capacity. If nobody can pay,
    we wait for whichever bucket refills soonest rather than hammering one.
    """

    def __init__(self, specs=MODEL_SPECS):
        self.budgets = {s.id: Budget(s) for s in specs}
        self.lock = threading.Lock()
        self.waited_s = 0.0
        self.switches = 0

    def _capable(self, role: str) -> list:
        return [b for b in self.budgets.values()
                if (b.spec.plans if role == "planner" else b.spec.narrates)]

    def acquire(self, role: str, est_tokens: int, preferred: str | None = None,
                exclude: set | None = None):
        """Reserve budget and return (ModelSpec, waited_seconds).

        `exclude` is a soft preference, not a filter: models the caller has
        already failed on this turn are skipped while any other capable model
        remains. Without it a caller can be handed the same busy bucket on
        every retry, exhaust its attempts against one model, and fail while
        three untouched buckets sit idle - which is exactly how a 38-question
        run produced eight "rate limited" errors with the pool three-quarters
        unused.

        Blocks only as long as the cheapest bucket needs; never longer than
        `max_wait_s`, after which it proceeds on the least-loaded model and
        lets the HTTP layer's backoff handle the rest.
        """
        waited = 0.0
        deadline = time.time() + SETTINGS.max_wait_s
        while True:
            with self.lock:
                now = time.time()
                capable = self._capable(role)
                fresh = [b for b in capable if b.spec.id not in (exclude or ())]
                pool = [b for b in (fresh or capable) if b.healthy(now)]
                if not pool:
                    pool = fresh or capable

                ready = [b for b in pool if b.headroom(now) >= est_tokens]
                if ready:
                    if not SETTINGS.spread_across_models and preferred:
                        pick = next((b for b in ready if b.spec.id == preferred), None)
                    else:
                        pick = None
                    if pick is None:
                        pick = max(ready, key=lambda b: (
                            b.spec.weight + (400 if b.spec.id == preferred else 0)
                            + b.headroom(now) / 100.0))
                    if preferred and pick.spec.id != preferred:
                        self.switches += 1
                    pick.charge(est_tokens)
                    return pick.spec, waited

                if time.time() >= deadline:
                    pick = max(pool, key=lambda b: b.headroom(now))
                    pick.charge(est_tokens)
                    return pick.spec, waited

                sleep_for = min(b.wait_for(est_tokens, now) for b in pool)
                sleep_for = max(0.05, min(sleep_for, 2.0))

            time.sleep(sleep_for)
            waited += sleep_for
            self.waited_s += sleep_for

    def observe(self, model_id: str, headers, billed: int | None = None,
                completion: int | None = None) -> None:
        b = self.budgets.get(model_id)
        if b:
            with self.lock:
                b.observe(headers, billed, completion)

    def penalise(self, model_id: str, seconds: float) -> None:
        b = self.budgets.get(model_id)
        if b:
            with self.lock:
                b.penalise(seconds)

    def capacity(self) -> dict:
        now = time.time()
        return {"aggregate_tpm": sum(b.spec.tpm for b in self.budgets.values()),
                "headroom_now": sum(max(0, b.headroom(now)) for b in self.budgets.values()),
                "models": [b.stats() for b in self.budgets.values()],
                "pool_waited_s": round(self.waited_s, 2),
                "model_switches": self.switches,
                "output_estimate": OUTPUT.stats()}


POOL = ModelPool()
