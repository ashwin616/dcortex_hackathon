"""The SQL agent: a question becomes a SELECT, the database answers it.

This hands the model more rope than the rest of the system does, so the rope is
short and inspected. Four guards, in order of how much they matter:

  1. The connection is read-only (`PRAGMA query_only`), so the worst a wrong
     query can do is return the wrong rows.
  2. Exactly one statement, and it must be a SELECT or a WITH. Semicolons,
     PRAGMA, ATTACH, INSERT, UPDATE, DELETE, DROP and friends are refused
     before execution, not caught afterwards.
  3. A LIMIT is imposed if the model did not write one, and the cursor is
     capped, so a cartesian join cannot return the whole database.
  4. The SQL that ran is returned with the answer. A controller - or a judge -
     can read exactly what was asked of the data.

What this is *not* used for: legality. RULE-DUTY-02 asks what would happen if a
crew member took a pairing they are not on, which is a simulation over a
modified world, not a row that exists to be selected. Those questions keep the
rules engine, and the planner is told so.
"""
from __future__ import annotations

import re
import time

from backend.domain.db import connection, schema_text, table_counts
from backend.domain.models import ToolResult

MAX_ROWS = 200
DEFAULT_LIMIT = 50
STATEMENT_TIMEOUT_MS = 2000

# Anything that is not a read. Matched on word boundaries so a column called
# "updated_at" does not trip the UPDATE check.
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|"
    r"detach|pragma|vacuum|reindex|analyze|begin|commit|rollback|grant|"
    r"revoke|load_extension)\b", re.I)
# sqlite_master is schema introspection; harmless to read, but there is a
# dedicated operation for it and letting the agent roam it invites confusion.
DISCOURAGED = re.compile(r"\bsqlite_(master|schema|temp_master)\b", re.I)
LIMIT_RE = re.compile(r"\blimit\s+\d+", re.I)


class SqlRejected(Exception):
    """The statement never ran."""


def _strip_fences(sql: str) -> str:
    s = (sql or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:sql)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    return s.strip().rstrip(";").strip()


def validate_sql(sql: str) -> str:
    """Return the statement if it is safe to run, else raise SqlRejected."""
    s = _strip_fences(sql)
    if not s:
        raise SqlRejected("empty statement")

    # One statement only. A trailing semicolon is already stripped, so any
    # remaining one means a second statement was appended.
    if ";" in s:
        raise SqlRejected("only one statement may be run; found ';'")
    if not re.match(r"^\s*(select|with)\b", s, re.I):
        raise SqlRejected("only SELECT (or WITH ... SELECT) is allowed; "
                          f"got {s.split()[0] if s.split() else '?'}")
    if (bad := FORBIDDEN.search(s)):
        raise SqlRejected(f"{bad.group(0).upper()} is not permitted - this "
                          "connection is read-only")
    if DISCOURAGED.search(s):
        raise SqlRejected("use the sql_schema operation to inspect the schema")
    if not LIMIT_RE.search(s):
        s = f"{s} LIMIT {DEFAULT_LIMIT}"
    return s


def run_sql(sql: str, purpose: str | None = None) -> ToolResult:
    """Execute one validated read-only SELECT against the snapshot database."""
    try:
        safe = validate_sql(sql)
    except SqlRejected as e:
        return ToolResult(tool="run_sql",
                          data={"error": str(e), "sql_rejected": _strip_fences(sql)},
                          evidence=[])

    conn = connection()
    interrupted = {"hit": False}
    deadline = time.perf_counter() + STATEMENT_TIMEOUT_MS / 1000

    def guard():
        # A runaway join should end as a refusal, not as a hung desk.
        if time.perf_counter() > deadline:
            interrupted["hit"] = True
            return 1
        return 0

    conn.set_progress_handler(guard, 10000)
    try:
        t0 = time.perf_counter()
        cur = conn.execute(safe)
        rows = [dict(r) for r in cur.fetchmany(MAX_ROWS)]
        truncated = cur.fetchone() is not None
        elapsed = (time.perf_counter() - t0) * 1000
        columns = [d[0] for d in cur.description] if cur.description else []
    except Exception as e:                      # noqa: BLE001 - reported, not raised
        return ToolResult(tool="run_sql",
                          data={"error": f"{type(e).__name__}: {e}", "sql": safe,
                                "hint": "check column and table names against sql_schema"},
                          evidence=[])
    finally:
        conn.set_progress_handler(None, 0)

    if interrupted["hit"]:
        return ToolResult(tool="run_sql",
                          data={"error": f"query exceeded {STATEMENT_TIMEOUT_MS} ms "
                                         "and was stopped", "sql": safe}, evidence=[])

    return ToolResult(
        tool="run_sql",
        data={"sql": safe, "purpose": purpose, "columns": columns,
              "row_count": len(rows), "truncated": truncated,
              "duration_ms": round(elapsed, 2), "rows": rows},
        evidence=_evidence(rows),
        note=(f"more than {MAX_ROWS} rows matched; showing the first {MAX_ROWS}"
              if truncated else None))


# Which identifier columns map to which source file, so SQL results cite the
# same records every other operation does.
_ID_COLUMNS = {
    "flight_id": "flights.json", "crew_id": "crew.json",
    "pairing_id": "rosters.json", "rule_id": "rules.json",
}


def _evidence(rows: list) -> list:
    out: list = []
    for row in rows[:60]:
        for col, file in _ID_COLUMNS.items():
            val = row.get(col)
            if isinstance(val, str) and val:
                ref = f"{file}#{val}"
                if ref not in out:
                    out.append(ref)
    return out


def sql_schema() -> ToolResult:
    """The tables, columns and row counts available to run_sql."""
    return ToolResult(tool="sql_schema",
                      data={"schema": schema_text(), "row_counts": table_counts(),
                            "dialect": "SQLite", "access": "read-only",
                            "limits": {"max_rows": MAX_ROWS,
                                       "default_limit": DEFAULT_LIMIT,
                                       "timeout_ms": STATEMENT_TIMEOUT_MS}},
                      evidence=["schema#sqlite"])


def ask_sql(question: str, purpose: str | None = None) -> ToolResult:
    """Answer a data question by writing SQL and running it.

    Generate, validate, execute; on a database error, feed the error back once
    and try again. A second failure is reported rather than guessed at.
    """
    from backend.planner.sqlagent import generate_sql

    attempts: list = []
    sql = generate_sql(question)
    for attempt in range(2):
        if sql.strip().upper().startswith("NOT_A_QUERY"):
            return ToolResult(
                tool="ask_sql",
                data={"error": "not answerable by query",
                      "reason": ("this asks whether a rule would be breached by a "
                                 "change to the roster, which is a simulation over "
                                 "modified state - use check_assignment or "
                                 "recommend_options"),
                      "attempts": attempts},
                evidence=[])

        result = run_sql(sql, purpose=purpose or question)
        data = result.data
        attempts.append({"sql": data.get("sql") or data.get("sql_rejected") or sql,
                         "error": data.get("error")})
        if "error" not in data:
            return ToolResult(tool="ask_sql",
                              data={**data, "question": question,
                                    "attempts": len(attempts)},
                              evidence=result.evidence, note=result.note)
        if attempt == 0:
            sql = generate_sql(question, error=data["error"], previous=sql)

    return ToolResult(tool="ask_sql",
                      data={"error": "could not produce a working query",
                            "question": question, "attempts": attempts},
                      evidence=[])
