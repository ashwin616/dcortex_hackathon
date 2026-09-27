"""Runtime configuration. Environment first, sane defaults second.

Nothing in the kernel reads this - configuration belongs to the edges. Only the
model pool, the rate limiter and the orchestrator consult it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # repository root


def _load_dotenv() -> None:
    """Minimal .env support - avoids a dependency for two variables."""
    for p in (Path.cwd() / ".env", ROOT / ".env"):
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _env(key: str, default):
    v = os.environ.get(key)
    if v is None:
        return default
    if isinstance(default, bool):
        return v.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v


@dataclass(frozen=True)
class ModelSpec:
    """One Groq model and the role it is good at.

    Every model on the free tier carries its own independent 8,000 tokens/min
    and 1,000 requests/day budget. Treating them as one pool rather than
    picking a favourite is the difference between 8k and 32k TPM.
    """
    id: str
    alias: str
    tpm: int = 8000
    rpd: int = 1000
    plans: bool = True       # reliable enough to emit a valid execution plan
    narrates: bool = True    # writes acceptable controller prose
    weight: int = 100        # preference when several models are free


# Verified against GET /openai/v1/models and a live tool-calling probe on
# 2026-09-05. compound / compound-mini reject tool calls; whisper and
# prompt-guard are not chat models.
MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("openai/gpt-oss-120b", "120b", plans=True, narrates=True, weight=100),
    ModelSpec("openai/gpt-oss-20b", "20b", plans=True, narrates=True, weight=80),
    ModelSpec("qwen/qwen3.8-27b", "qwen38", plans=True, narrates=True, weight=70),
    ModelSpec("qwen/qwen3.6-27b", "qwen36", plans=True, narrates=True, weight=60),
)
BY_ALIAS = {m.alias: m for m in MODEL_SPECS}
BY_ID = {m.id: m for m in MODEL_SPECS}


@dataclass
class Settings:
    api_url: str = _env("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
    api_key: str = _env("GROQ_API_KEY", "")

    # Which model leads each role. The pool falls back to any capable model
    # with budget, so these are preferences, not hard bindings.
    planner_model: str = _env("CREWOPS_PLANNER_MODEL", "20b")
    narrator_model: str = _env("CREWOPS_NARRATOR_MODEL", "120b")

    temperature: float = _env("CREWOPS_TEMPERATURE", 0.0)
    request_timeout: int = _env("CREWOPS_TIMEOUT", 45)
    max_plan_steps: int = _env("CREWOPS_MAX_PLAN_STEPS", 5)
    max_retries: int = _env("CREWOPS_MAX_RETRIES", 3)

    # Rate limiting
    reserve_fraction: float = _env("CREWOPS_RESERVE_FRACTION", 0.10)  # headroom kept free
    max_wait_s: float = _env("CREWOPS_MAX_WAIT", 20.0)
    # How long to keep trying when every bucket is throttled per-minute.
    # A minute limit refills; failing the question instead of waiting it
    # out turns a transient squeeze into a wrong answer on the desk.
    throttle_deadline_s: float = _env("CREWOPS_THROTTLE_DEADLINE", 120.0)
    spread_across_models: bool = _env("CREWOPS_SPREAD", True)

    # Caching
    cache_enabled: bool = _env("CREWOPS_CACHE", True)
    cache_ttl_s: int = _env("CREWOPS_CACHE_TTL", 3600)
    cache_max_entries: int = _env("CREWOPS_CACHE_MAX", 512)

    # Planning shortcuts
    fastpath_enabled: bool = _env("CREWOPS_FASTPATH", True)

    # The database is rebuilt from the JSON at startup by default, so there
    # is nothing to migrate and nothing to go stale. Point this at a file
    # if you want one on disk.
    db_path: str = _env("CREWOPS_DB_PATH", str(ROOT / "data" / "crewops.db"))

    log_level: str = _env("CREWOPS_LOG_LEVEL", "INFO")
    audit_path: str = _env("CREWOPS_AUDIT_LOG", str(ROOT / "logs" / "audit.jsonl"))

    def resolve(self, role: str) -> ModelSpec:
        alias = self.planner_model if role == "planner" else self.narrator_model
        return BY_ALIAS.get(alias) or BY_ID.get(alias) or MODEL_SPECS[0]


SETTINGS = Settings()
