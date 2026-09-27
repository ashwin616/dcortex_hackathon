"""Crew Ops Advisor - HTTP API.

    uvicorn backend.main:app --reload --port 8000
    python -m backend.main                       (same thing, no reload)

One process owns the dataset, the kernel and the model pool; clients hold
nothing but a session id. That matters for more than tidiness: the rate limiter
budgets four token buckets across a whole desk, and it can only do that if
every request goes through one pool. A second copy of the advisor in the UI
process would double-spend a budget neither of them can see.

Endpoints
    GET  /health                snapshot, dataset size, model roster
    GET  /capacity              live token pool and cache statistics
    GET  /operations            the operation catalogue the planner sees
    POST /ask                   one question, one answer
    POST /ask/stream            the same, as server-sent events
    POST /tools/{name}          call one operation directly - no LLM
    POST /session/reset         clear a conversation's memory
    GET  /audit                 recent turns from the audit log
    GET  /metrics               latency, tokens, cache, abstention, budget
"""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Path, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend import tools as T
from backend.advisor import Advisor, StreamingAdvisor
from backend.cache import clear_all as clear_caches
from backend.config import MODEL_SPECS, SETTINGS
from backend.domain.data import store
from backend.executor.registry import OPERATIONS, catalogue
from backend.telemetry import LOG, read_audit

app = FastAPI(
    title="Crew Ops Advisor",
    version="1.0.0",
    summary="Conversational crew control for dCortex Air, over a deterministic rules engine.",
    description=__doc__,
)

import time as _time

_STARTED_EPOCH = _time.time()
_STARTED_AT = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(_STARTED_EPOCH))

MAX_SESSIONS = 64
_sessions: OrderedDict = OrderedDict()


def session(session_id: str) -> StreamingAdvisor:
    """One advisor per conversation. Oldest is evicted rather than grown."""
    adv = _sessions.get(session_id)
    if adv is None:
        adv = _sessions[session_id] = StreamingAdvisor()
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
    _sessions.move_to_end(session_id)
    return adv


# --------------------------------------------------------------------------
# request / response models
# --------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = "default"
    keep_history: bool = True
    planner_model: str | None = None
    narrator_model: str | None = None
    fastpath: bool | None = None
    cache: bool | None = None


class ToolRequest(BaseModel):
    args: dict[str, Any] = Field(default_factory=dict)


def _apply(req: AskRequest) -> StreamingAdvisor:
    """Per-request overrides. Model choice is a desk-wide setting on a shared
    pool, so it is applied to SETTINGS rather than held per session."""
    if req.planner_model:
        SETTINGS.planner_model = req.planner_model
    if req.narrator_model:
        SETTINGS.narrator_model = req.narrator_model
    if req.fastpath is not None:
        SETTINGS.fastpath_enabled = req.fastpath
    if req.cache is not None:
        SETTINGS.cache_enabled = req.cache
    return session(req.session_id)


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------
def _code_fingerprint() -> dict:
    """When the running code was last edited.

    A server started before a fix keeps serving the bug, and the only symptom
    is a user reporting something the code no longer does. This makes the
    staleness visible instead of mysterious.
    """
    import pathlib
    import time
    root = pathlib.Path(__file__).resolve().parent
    files = list(root.rglob("*.py"))
    newest = max((f.stat().st_mtime for f in files), default=0)
    return {"newest_source_mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                 time.gmtime(newest)),
            "started_at": _STARTED_AT,
            "stale": newest > _STARTED_EPOCH,
            "files": len(files)}


@app.get("/health")
def health() -> dict:
    s = store()
    return {
        "status": "ok",
        "code": _code_fingerprint(),
        "snapshot_utc": s.snapshot_utc,
        "week": [s.dates[0], s.dates[-1]],
        "flights": len(s.flights), "crew": len(s.crew), "pairings": len(s.pairings),
        "stations": sorted(s.stations),
        "operations": len(OPERATIONS),
        "models": [{"alias": m.alias, "id": m.id, "tpm": m.tpm, "rpd": m.rpd}
                   for m in MODEL_SPECS],
        "planner_model": SETTINGS.planner_model,
        "narrator_model": SETTINGS.narrator_model,
        "api_key_present": bool(SETTINGS.api_key),
        "fastpath_enabled": SETTINGS.fastpath_enabled,
        "cache_enabled": SETTINGS.cache_enabled,
        "sessions": len(_sessions),
    }


@app.get("/capacity")
def capacity() -> dict:
    return Advisor.capacity()


@app.get("/operations")
def operations() -> dict:
    return {
        "count": len(OPERATIONS),
        "catalogue": catalogue(),
        "operations": [{"name": o.name, "tier": o.tier, "summary": o.summary,
                        "params": list(o.params), "required": list(o.required)}
                       for o in sorted(OPERATIONS.values(), key=lambda o: (o.tier, o.name))],
    }


# --------------------------------------------------------------------------
# asking
# --------------------------------------------------------------------------
@app.post("/ask")
def ask(req: AskRequest) -> dict:
    adv = _apply(req)
    turn = adv.ask(req.question, keep_history=req.keep_history)
    return turn.payload()


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Server-sent events: `stage`, `step`, `token`, then a final `turn`.

    The pipeline stages are streamed as they complete rather than buffered,
    because watching the rules engine finish before any prose is written is
    the point - it is the system's central claim, made visible.
    """
    adv = _apply(req)

    def events():
        try:
            for ev in adv.ask_stream(req.question, keep_history=req.keep_history):
                if ev["type"] == "turn":
                    ev = {"type": "turn", "turn": ev["turn"].payload()}
                yield f"data: {json.dumps(ev, default=str)}\n\n"
        except Exception as e:                      # noqa: BLE001
            LOG.exception("stream failed")
            yield f'data: {json.dumps({"type": "error", "message": str(e)})}\n\n'
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------------
# the deterministic layer, directly
# --------------------------------------------------------------------------
@app.post("/tools/{name}")
def call_tool(name: str = Path(description="operation name"),
              req: ToolRequest = Body(default=ToolRequest())) -> dict:
    """Run one operation with no model involved.

    This is how the UI shows a kernel result without an API key, and how a
    controller checks a figure the chat gave them.
    """
    if name not in OPERATIONS:
        raise HTTPException(404, f"no such operation: {name}")
    result = T.call(name, **req.args)
    return result.to_dict()


# --------------------------------------------------------------------------
# session and audit
# --------------------------------------------------------------------------
@app.post("/session/reset")
def reset(session_id: str = Query("default"), caches: bool = Query(False)) -> dict:
    adv = _sessions.get(session_id)
    if adv:
        adv.reset()
    if caches:
        clear_caches()
    return {"reset": session_id, "caches_cleared": caches}


@app.get("/audit")
def audit(limit: int = Query(25, ge=1, le=500)) -> dict:
    rows = read_audit(limit)
    return {"count": len(rows), "turns": rows}


@app.get("/metrics")
def metrics() -> dict:
    """Everything you would want on a wall during a shift.

    Deliberately one endpoint rather than a Prometheus exporter: the numbers
    that matter here are budget, cache and abstention rates, and a JSON blob is
    what the console already renders.
    """
    from backend.cache import stats as cache_stats
    from backend.ratelimit import OUTPUT, POOL

    turns = read_audit(200)
    answered = [t for t in turns if not t.get("error")]
    abstained = [t for t in turns if t.get("abstained")]
    ungrounded = [t for t in turns if not t.get("grounded", True)]
    latencies = sorted(t.get("latency_s", 0) for t in answered)
    tokens = [t.get("usage", {}).get("total_tokens", 0) for t in answered]
    sources: dict = {}
    for t in turns:
        sources[t.get("plan_source") or "?"] = sources.get(t.get("plan_source") or "?", 0) + 1

    def pct(vals, q):
        return round(vals[min(len(vals) - 1, int(len(vals) * q))], 2) if vals else 0.0

    return {
        "turns_sampled": len(turns),
        "answered": len(answered),
        "abstained": len(abstained),
        "ungrounded": len(ungrounded),
        "abstention_rate": round(len(abstained) / len(turns), 3) if turns else 0.0,
        "latency_s": {"p50": pct(latencies, 0.5), "p90": pct(latencies, 0.9),
                      "max": latencies[-1] if latencies else 0.0},
        "tokens_per_turn": round(sum(tokens) / len(tokens), 1) if tokens else 0.0,
        "plan_source": sources,
        "zero_token_share": round(sources.get("fastpath", 0) / len(turns), 3) if turns else 0.0,
        "pool": POOL.capacity(),
        "cache": cache_stats(),
        "output_estimator": OUTPUT.stats(),
        "sessions": len(_sessions),
    }


@app.get("/")
def index() -> dict:
    return {"service": "Crew Ops Advisor", "docs": "/docs", "health": "/health"}


def main() -> None:
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
