"""Groq HTTP client with pooled, budget-aware model selection.

Every call goes through the rate limiter, which picks whichever capable model
can pay for it right now. That is the whole trick behind getting past 8,000
tokens per minute: the limit is per model, and there are four of them.

Failures are handled by class, not by blanket retry:
  429 - the bucket is empty; mark it and move to another model immediately,
        only sleeping if every bucket is dry.
  5xx - transient; retry the same model with jittered backoff.
  4xx - our bug; surface it rather than burning the daily request budget.
"""
from __future__ import annotations

import json
import random
import re
import time

import requests

from backend.config import SETTINGS
from backend.ratelimit import (CHARS_PER_TOKEN, POOL, estimate_tokens,
                               parse_reset)

TPD_RE = re.compile(r"tokens per day", re.I)


class LLMError(RuntimeError):
    """Anything that stopped us getting a completion."""

    def __init__(self, message: str, status: int | None = None, exhausted: bool = False):
        super().__init__(message)
        self.status = status
        self.exhausted = exhausted


class Usage:
    """Token accounting for one logical turn."""

    def __init__(self):
        self.prompt = 0
        self.completion = 0
        self.calls = 0
        self.waited_s = 0.0
        self.models: list[str] = []

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    def add(self, model: str, usage: dict, waited: float):
        self.calls += 1
        self.models.append(model)
        self.waited_s += waited
        self.prompt += int(usage.get("prompt_tokens") or 0)
        self.completion += int(usage.get("completion_tokens") or 0)

    def to_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt,
                "completion_tokens": self.completion, "total_tokens": self.total,
                "waited_s": round(self.waited_s, 2), "models": self.models}


def complete(messages: list, role: str = "narrator", *, json_mode: bool = False,
             max_tokens: int = 900, temperature: float | None = None,
             usage: Usage | None = None, preferred: str | None = None,
             reasoning_effort: str | None = None) -> str:
    """One completion. Returns the assistant's text.

    `reasoning_effort` matters more than it looks: gpt-oss models spend
    completion tokens on hidden reasoning before emitting anything, and those
    tokens count against both max_tokens and the daily budget. Planning is a
    translation task, not a reasoning one, so it runs at "low".
    """
    if not SETTINGS.api_key:
        raise LLMError("No GROQ_API_KEY. Put it in the environment or a .env file "
                       "at the project root as GROQ_API_KEY=gsk_...")

    want = SETTINGS.resolve(role)
    preferred = preferred or want.id
    est = estimate_tokens(messages, max_output=max_tokens)

    last_error: LLMError | None = None
    tried: set = set()

    # Two bounds, not one. A fixed retry count is wrong for a per-minute limit:
    # every bucket can be legitimately busy for a few seconds, and a limit that
    # refills should be waited out rather than reported as a failure. So the
    # loop runs until either a generous attempt cap or a wall-clock deadline,
    # whichever comes first - and a *daily* exhaustion still fails immediately,
    # because that one will not refill today.
    attempts = max(SETTINGS.max_retries, len(POOL.budgets)) * 3 + 1
    deadline = time.time() + SETTINGS.throttle_deadline_s

    for attempt in range(attempts):
        if attempt and time.time() > deadline:
            break
        spec, waited = POOL.acquire(role, est, preferred=preferred, exclude=tried)
        body = {
            "model": spec.id,
            "messages": messages,
            "temperature": SETTINGS.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if reasoning_effort and spec.id.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = reasoning_effort

        try:
            r = requests.post(SETTINGS.api_url, timeout=SETTINGS.request_timeout,
                              headers={"Authorization": f"Bearer {SETTINGS.api_key}",
                                       "Content-Type": "application/json"},
                              json=body)
        except requests.RequestException as e:
            POOL.penalise(spec.id, 5.0)
            tried.add(spec.id)
            last_error = LLMError(f"network: {e}")
            continue

        if r.status_code == 200:
            data = r.json()
            u = data.get("usage") or {}
            POOL.observe(spec.id, r.headers, billed=u.get("total_tokens"),
                         completion=u.get("completion_tokens"))
            if usage is not None:
                usage.add(spec.id, data.get("usage") or {}, waited)
            return (data["choices"][0]["message"].get("content") or "").strip()

        POOL.observe(spec.id, r.headers)
        tried.add(spec.id)

        if r.status_code == 429:
            body_text = r.text[:400]
            reset = parse_reset(r.headers.get("retry-after")) or _retry_hint(body_text)
            daily = bool(TPD_RE.search(body_text))
            # The per-day allowance is a *rolling* window, not a midnight reset:
            # a bucket reporting 199,997/200,000 used can be answering again
            # four seconds later. So honour the server's own reset hint instead
            # of assuming an hour - assuming made a four-second wait look like
            # an exhausted account and failed questions we could have answered.
            POOL.penalise(spec.id, max(reset, 1.0) if reset else (3600.0 if daily else 5.0))
            preferred = None            # let the pool pick a different bucket
            last_error = LLMError(
                ("daily token budget exhausted on " if daily else "rate limited on ")
                + spec.id + (f"; resets in {reset:.0f}s" if reset else ""),
                status=429, exhausted=daily and reset > SETTINGS.throttle_deadline_s)
            if len(tried) >= len(POOL.budgets):
                nap = min((max(0.0, b.cooldown_until - time.time())
                           for b in POOL.budgets.values()), default=1.0)
                if time.time() + nap > deadline:
                    raise last_error        # nothing will free up in time
                time.sleep(min(max(nap, 0.5), 15.0))
                tried.clear()
            continue

        if 500 <= r.status_code < 600:
            POOL.penalise(spec.id, 2.0)
            time.sleep(min(2 ** attempt + random.random(), 8.0))
            last_error = LLMError(f"Groq {r.status_code}: {r.text[:200]}",
                                  status=r.status_code)
            continue

        raise LLMError(f"Groq {r.status_code}: {r.text[:300]}", status=r.status_code)

    raise last_error or LLMError("no completion after retries")


def _retry_hint(text: str) -> float:
    m = re.search(r"try again in ([\dhms.]+)", text)
    return parse_reset(m.group(1)) + 0.3 if m else 0.0


def complete_json(messages: list, role: str = "planner", **kw) -> dict:
    """A completion that must be a JSON object.

    Groq's strict `response_format: json_object` intermittently rejects its own
    smaller models' output with `json_validate_failed` and an empty generation.
    Falling back to a plain completion plus a tolerant parser recovers the turn
    instead of spending a retry on the same failure.
    """
    kw.setdefault("reasoning_effort", "low")
    try:
        return parse_json_object(complete(messages, role=role, json_mode=True, **kw))
    except LLMError as e:
        if e.status != 400 or "json_validate_failed" not in str(e):
            raise
    nudged = list(messages) + [{
        "role": "user",
        "content": "Reply with the JSON object only. No prose, no code fence."}]
    return parse_json_object(complete(nudged, role=role, json_mode=False, **kw))


def parse_json_object(raw: str) -> dict:
    """Models fence JSON, prepend prose, or emit a bare object. Handle all three."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise LLMError(f"planner returned no JSON object: {text[:200]}")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError as e:
                    raise LLMError(f"planner JSON malformed: {e}") from e
    raise LLMError(f"planner JSON never closed: {text[:200]}")


def complete_stream(messages: list, role: str = "narrator", *, max_tokens: int = 900,
                    temperature: float | None = None, usage: Usage | None = None,
                    preferred: str | None = None, reasoning_effort: str | None = None):
    """Yield the completion token by token.

    Same pooling, budgeting and failover as `complete()` - a streaming call that
    gave up on the first busy bucket would strand the other three, which is the
    exact failure the pool exists to prevent.

    Streaming responses carry no `usage` block until the final chunk, so the
    estimate stands in when the server omits it.
    """
    if not SETTINGS.api_key:
        raise LLMError("No GROQ_API_KEY. Put it in the environment or a .env file "
                       "at the project root as GROQ_API_KEY=gsk_...")

    want = SETTINGS.resolve(role)
    preferred = preferred or want.id
    est = estimate_tokens(messages, max_output=max_tokens)

    last_error: LLMError | None = None
    tried: set = set()
    attempts = max(SETTINGS.max_retries, len(POOL.budgets)) * 3 + 1
    deadline = time.time() + SETTINGS.throttle_deadline_s

    for attempt in range(attempts):
        if attempt and time.time() > deadline:
            break
        spec, waited = POOL.acquire(role, est, preferred=preferred, exclude=tried)
        body = {
            "model": spec.id, "messages": messages, "stream": True,
            "temperature": SETTINGS.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        if reasoning_effort and spec.id.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = reasoning_effort

        try:
            r = requests.post(SETTINGS.api_url, timeout=SETTINGS.request_timeout,
                              stream=True,
                              headers={"Authorization": f"Bearer {SETTINGS.api_key}",
                                       "Content-Type": "application/json"},
                              json=body)
        except requests.RequestException as e:
            POOL.penalise(spec.id, 5.0)
            tried.add(spec.id)
            last_error = LLMError(f"network: {e}")
            continue

        if r.status_code != 200:
            POOL.observe(spec.id, r.headers)
            detail = r.text[:300]
            tried.add(spec.id)
            if r.status_code == 429:
                daily = bool(TPD_RE.search(detail))
                reset = parse_reset(r.headers.get("retry-after")) or _retry_hint(detail)
                POOL.penalise(spec.id,
                              max(reset, 1.0) if reset else (3600.0 if daily else 5.0))
                preferred = None
                last_error = LLMError(
                    ("daily token budget exhausted on " if daily else "rate limited on ")
                    + spec.id + (f"; resets in {reset:.0f}s" if reset else ""),
                    status=429,
                    exhausted=daily and reset > SETTINGS.throttle_deadline_s)
                if len(tried) >= len(POOL.budgets):
                    nap = min((max(0.0, b.cooldown_until - time.time())
                               for b in POOL.budgets.values()), default=1.0)
                    if time.time() + nap > deadline:
                        raise last_error
                    time.sleep(min(max(nap, 0.5), 15.0))
                    tried.clear()
                continue
            if 500 <= r.status_code < 600:
                POOL.penalise(spec.id, 2.0)
                last_error = LLMError(f"Groq {r.status_code}: {detail}",
                                      status=r.status_code)
                time.sleep(min(2 ** attempt + random.random(), 8.0))
                continue
            raise LLMError(f"Groq {r.status_code}: {detail}", status=r.status_code)

        POOL.observe(spec.id, r.headers)
        billed = None
        produced = 0
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            payload = raw[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                billed = chunk["usage"].get("total_tokens")
                POOL.observe(spec.id, r.headers, billed=billed,
                             completion=chunk["usage"].get("completion_tokens"))
                if usage is not None:
                    usage.add(spec.id, chunk["usage"], waited)
            for choice in chunk.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    produced += len(piece)
                    yield piece

        if billed is None and usage is not None:
            usage.add(spec.id, {"prompt_tokens": est - max_tokens,
                                "completion_tokens": int(produced / CHARS_PER_TOKEN)},
                      waited)
        return

    raise last_error or LLMError("no completion after retries")
