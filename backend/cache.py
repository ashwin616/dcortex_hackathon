"""Caches, sized for the thing that is actually scarce.

The daily token budget is 200,000 per model. A demo that asks the same six
questions repeatedly should not spend it twice, and a plan is a pure function
of the question text - so plans cache well and safely.

Two caches, different risk profiles:

  * Plan cache - keyed on the normalised question. Safe: re-planning the same
    sentence produces the same plan, and the plan is validated before use
    either way.
  * Answer cache - keyed on question *and* the evidence the answer was built
    from, so a stale answer cannot outlive the data it cites. Off by default
    for multi-turn, where context changes meaning.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict

from backend.config import SETTINGS


def normalise_question(q: str) -> str:
    """Cache key for a question: case, whitespace and punctuation are noise."""
    q = (q or "").lower().strip()
    q = q.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    q = re.sub(r"[^\w\s:/-]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def _key(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


class TTLCache:
    """Small thread-safe LRU with expiry. Not a library - it is 40 lines."""

    def __init__(self, max_entries: int | None = None, ttl_s: int | None = None):
        self.max = max_entries or SETTINGS.cache_max_entries
        self.ttl = ttl_s or SETTINGS.cache_ttl_s
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        if not SETTINGS.cache_enabled:
            return None
        with self._lock:
            item = self._d.get(key)
            if item is None:
                self.misses += 1
                return None
            expires, value = item
            if time.time() > expires:
                del self._d[key]
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value) -> None:
        if not SETTINGS.cache_enabled:
            return
        with self._lock:
            self._d[key] = (time.time() + self.ttl, value)
            self._d.move_to_end(key)
            while len(self._d) > self.max:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"entries": len(self._d), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0}


PLAN_CACHE = TTLCache()
ANSWER_CACHE = TTLCache()


def plan_key(question: str) -> str:
    return _key("plan", normalise_question(question))


def answer_key(question: str, evidence: list) -> str:
    """Bind the answer to the exact records it was built from."""
    return _key("answer", normalise_question(question), "|".join(sorted(set(evidence))))


def stats() -> dict:
    return {"plan_cache": PLAN_CACHE.stats(), "answer_cache": ANSWER_CACHE.stats(),
            "tool_cache": tool_stats()}


def clear_all() -> None:
    PLAN_CACHE.clear()
    ANSWER_CACHE.clear()
    TOOL_CACHE.clear()


# --------------------------------------------------------------------------
# tool-result memoisation with single-flight
# --------------------------------------------------------------------------
class SingleFlight:
    """One in-flight computation per key, shared by every concurrent caller.

    Without it, a parallel wave asking two threads for the same pairing does
    the work twice; worse, a desk with several controllers watching the same
    disruption multiplies identical kernel work by the number of screens. The
    first caller computes, the rest wait on the same result.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._inflight: dict = {}
        self.shared = 0

    def run(self, key: str, fn):
        with self._lock:
            event = self._inflight.get(key)
            if event is None:
                event = self._inflight[key] = {"event": threading.Event(),
                                               "value": None, "error": None}
                owner = True
            else:
                owner = False
                self.shared += 1
        if not owner:
            event["event"].wait(timeout=30)
            if event["error"]:
                raise event["error"]
            return event["value"]
        try:
            event["value"] = fn()
            return event["value"]
        except BaseException as e:          # noqa: BLE001 - re-raised below
            event["error"] = e
            raise
        finally:
            event["event"].set()
            with self._lock:
                self._inflight.pop(key, None)


TOOL_CACHE = TTLCache(max_entries=1024)
FLIGHT = SingleFlight()


def tool_key(name: str, args: dict) -> str:
    """Tool calls are pure functions of their arguments over an immutable
    snapshot, so the arguments are the whole key."""
    return _key("tool", name, json.dumps(args, sort_keys=True, default=str))


def memoise(name: str, args: dict, fn):
    """Cache a tool call, collapsing concurrent duplicates into one."""
    if not SETTINGS.cache_enabled:
        return fn()
    key = tool_key(name, args)
    hit = TOOL_CACHE.get(key)
    if hit is not None:
        return hit
    value = FLIGHT.run(key, fn)
    TOOL_CACHE.put(key, value)
    return value


def tool_stats() -> dict:
    return {**TOOL_CACHE.stats(), "coalesced": FLIGHT.shared}
