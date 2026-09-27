"""Structured logging and the audit trail.

Every answer writes one JSON line recording the plan, the operations run, the
records touched, the rules evaluated and the tokens spent. In a real crew
system this log is what a regulator asks for after a bad decision; here it is
also how the failure analysis in the README was written.

Crew identifiers are the only personal data present, and they are already
pseudonymous. Nothing else about a person is logged.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from backend.config import SETTINGS

_lock = threading.Lock()
_configured = False


def logger(name: str = "crewops") -> logging.Logger:
    global _configured
    if not _configured:
        logging.basicConfig(
            level=getattr(logging, SETTINGS.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S")
        _configured = True
    return logging.getLogger(name)


LOG = logger()


def audit(record: dict) -> None:
    """Append one turn to the audit log. Never raises - telemetry must not be
    able to take the advisor down."""
    try:
        path = Path(SETTINGS.audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           **record}, default=str)
        with _lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:                      # noqa: BLE001 - deliberate
        LOG.debug("audit write failed: %s", e)


def read_audit(limit: int = 50) -> list:
    path = Path(SETTINGS.audit_path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
