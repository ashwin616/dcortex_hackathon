"""Question in, SQL out - with one chance to fix its own mistake.

The agent sees the schema and the question, writes one SELECT, and the guards
in `tools/sql.py` decide whether it runs. If SQLite rejects it - a mistyped
column, a bad join - the error goes back once and it tries again. A second
failure is reported as a failure rather than a third guess.

Two rules the prompt insists on, because they are where a generated query
quietly produces a confident wrong number on this dataset:

  * `duty_clocks` already contains the 7-day and 28-day totals. Re-deriving
    them from `duty_history` misses the rostered week and reads about 8 hours
    low per crew member - the figures disagree, and the wrong one looks fine.
  * Legality is not a SELECT. Whether somebody *would* breach if moved onto a
    pairing is a simulation over a modified roster; the rules engine owns it.
"""
from __future__ import annotations

from backend.client import Usage, complete
from backend.domain.data import store
from backend.domain.db import schema_text

MAX_SQL_TOKENS = 700


def sql_prompt() -> str:
    s = store()
    return f"""You write SQLite SELECT statements against an airline crew-operations
snapshot. You output SQL and nothing else - no prose, no code fence, no
explanation, no trailing semicolon.

SNAPSHOT: now = {s.snapshot_utc}. Dates are ISO strings 'YYYY-MM-DD'; times are
UTC 'HH:MM'. The schedule week is {s.dates[0]} to {s.dates[-1]}. Money is INR.

SCHEMA
{schema_text()}

RULES
1. One SELECT (or WITH ... SELECT). Read-only: no INSERT, UPDATE, DELETE, DROP,
   ATTACH, PRAGMA, and no second statement.
2. Always include a LIMIT unless the query is an aggregate returning one row.
3. Return the columns that answer the question, plus the identifier column
   (flight_id, crew_id, pairing_id) so the answer can cite its records.
4. Use duty_clocks.duty_hours_7d and duty_clocks.flight_hours_28d for accrued
   hours. Do NOT re-derive them by summing duty_history: that window also
   includes the rostered week, and a hand-rolled sum reads low.
5. For a superlative, prefer ordering with a LIMIT over a subquery MAX, and
   return every row tied at the extreme when the question asks "which".
6. If the question is about whether a rule WOULD be breached by a change to the
   roster, answer with exactly: NOT_A_QUERY
   Legality is decided by the rules engine, not by selecting existing rows.

QUESTION: """


def generate_sql(question: str, usage: Usage | None = None,
                 error: str | None = None, previous: str | None = None) -> str:
    """Write one SELECT for `question`, optionally repairing a failed attempt."""
    messages = [{"role": "system", "content": sql_prompt()},
                {"role": "user", "content": question}]
    if error and previous:
        messages.append({
            "role": "user",
            "content": (f"That statement failed.\n\nSQL:\n{previous}\n\n"
                        f"Error: {error}\n\nReturn a corrected SELECT, or "
                        "NOT_A_QUERY if the database cannot answer this.")})
    return complete(messages, role="planner", max_tokens=MAX_SQL_TOKENS,
                    reasoning_effort="low", usage=usage).strip()
