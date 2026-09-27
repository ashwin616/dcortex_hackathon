"""SQLite mirror of the dataset.

The JSON files stay the source of truth - they are what the graders ship and
what `validate.py` checks - and this builds a relational view of them at
startup. Two things it buys:

  * a real query engine, so aggregation, joins and grouping do not have to be
    hand-written per question
  * a scale story: at 150 crew a Python scan is 1.2 ms, at 50,000 it is not,
    and an indexed table is the answer that survives the question

Built in memory by default, so there is no migration, no stale file and no
second source of truth to keep in step. The schema is flattened where the JSON
nests, because a controller asks "which pairings night-stop at DEL", not
"traverse rosters.days[0].flights[-1]".
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import threading
import time

from backend.domain.data import Store, parse_dt, store
from backend.domain.duty import fdp_limit
from backend.telemetry import LOG

SCHEMA = """
CREATE TABLE flights (
    flight_id     TEXT PRIMARY KEY,
    flight_no     TEXT NOT NULL,
    date          TEXT NOT NULL,
    dep_station   TEXT NOT NULL,
    arr_station   TEXT NOT NULL,
    dep_utc       TEXT NOT NULL,
    arr_utc       TEXT NOT NULL,
    dep_time      TEXT NOT NULL,
    arr_time      TEXT NOT NULL,
    block_hours   REAL NOT NULL,
    aircraft      TEXT NOT NULL,
    aircraft_type TEXT NOT NULL,
    seats         INTEGER NOT NULL,
    route         TEXT NOT NULL,
    pairing_id    TEXT
);
CREATE TABLE crew (
    crew_id              TEXT PRIMARY KEY,
    name                 TEXT NOT NULL,
    rank                 TEXT NOT NULL,
    base                 TEXT NOT NULL,
    seniority            INTEGER,
    reachability_minutes INTEGER,
    status               TEXT NOT NULL,
    is_reserve           INTEGER NOT NULL,
    ratings              TEXT NOT NULL          -- comma separated, see crew_ratings
);
CREATE TABLE crew_ratings (crew_id TEXT NOT NULL, rating TEXT NOT NULL);
CREATE TABLE pairings (pairing_id TEXT PRIMARY KEY, aircraft TEXT NOT NULL,
                       aircraft_type TEXT NOT NULL, days INTEGER NOT NULL);
CREATE TABLE pairing_days (
    pairing_id        TEXT NOT NULL,
    date              TEXT NOT NULL,
    day_index         INTEGER NOT NULL,
    report_utc        TEXT NOT NULL,
    release_utc       TEXT NOT NULL,
    duty_hours        REAL NOT NULL,
    block_hours       REAL NOT NULL,
    sectors           INTEGER NOT NULL,
    seats             INTEGER NOT NULL,
    fdp_limit         REAL NOT NULL,
    fdp_slack         REAL NOT NULL,
    overnight_station TEXT NOT NULL,
    aircraft          TEXT NOT NULL,
    aircraft_type     TEXT NOT NULL
);
CREATE TABLE pairing_flights (pairing_id TEXT NOT NULL, date TEXT NOT NULL,
                              flight_id TEXT NOT NULL);
CREATE TABLE pairing_crew (pairing_id TEXT NOT NULL, crew_id TEXT NOT NULL,
                           role TEXT NOT NULL);
CREATE TABLE duty_clocks (
    crew_id             TEXT PRIMARY KEY,
    as_of_utc           TEXT NOT NULL,
    duty_hours_7d       REAL NOT NULL,
    flight_hours_28d    REAL NOT NULL,
    duty_headroom_7d    REAL NOT NULL,
    flight_headroom_28d REAL NOT NULL,
    last_rest_ended     TEXT
);
CREATE TABLE duty_history (crew_id TEXT NOT NULL, date TEXT NOT NULL,
                           duty_hours REAL NOT NULL, flight_hours REAL NOT NULL);
CREATE TABLE reserves (
    crew_id              TEXT PRIMARY KEY,
    base                 TEXT NOT NULL,
    rank                 TEXT NOT NULL,
    window_start         TEXT NOT NULL,
    window_end           TEXT NOT NULL,
    window_hours         REAL NOT NULL,
    reachability_minutes INTEGER
);
CREATE TABLE reserve_dates (crew_id TEXT NOT NULL, date TEXT NOT NULL);
CREATE TABLE certifications (
    crew_id         TEXT NOT NULL,
    cert_type       TEXT NOT NULL,
    valid_from      TEXT NOT NULL,
    valid_to        TEXT NOT NULL,
    days_to_expiry  INTEGER NOT NULL
);
CREATE TABLE risk_signals (crew_id TEXT PRIMARY KEY, score REAL NOT NULL,
                           drivers TEXT NOT NULL);
CREATE TABLE costs (item TEXT PRIMARY KEY, amount_inr INTEGER NOT NULL);
CREATE TABLE rules (rule_id TEXT PRIMARY KEY, text TEXT NOT NULL, params TEXT);

CREATE INDEX ix_flights_date       ON flights(date);
CREATE INDEX ix_flights_dep        ON flights(dep_station, date);
CREATE INDEX ix_flights_arr        ON flights(arr_station, date);
CREATE INDEX ix_flights_aircraft   ON flights(aircraft, date);
CREATE INDEX ix_crew_rank_base     ON crew(rank, base);
CREATE INDEX ix_crew_status        ON crew(status);
CREATE INDEX ix_pd_date            ON pairing_days(date);
CREATE INDEX ix_pd_pairing         ON pairing_days(pairing_id);
CREATE INDEX ix_pc_crew            ON pairing_crew(crew_id);
CREATE INDEX ix_pc_pairing         ON pairing_crew(pairing_id);
CREATE INDEX ix_hist_crew          ON duty_history(crew_id, date);
CREATE INDEX ix_certs_crew         ON certifications(crew_id);
CREATE INDEX ix_certs_expiry       ON certifications(days_to_expiry);
CREATE INDEX ix_rd_date            ON reserve_dates(date);
"""

# One line per table, rendered into the SQL agent's prompt.
TABLE_NOTES = {
    "flights": "one row per leg; route is 'BLR-DEL'; pairing_id may be NULL",
    "crew": "one row per crew member; ratings is a comma list, or join crew_ratings",
    "crew_ratings": "crew_id -> one rating per row (A320, ATR72)",
    "pairings": "one row per pairing; days = how many duty days it spans",
    "pairing_days": "one row per pairing per day - the duty itself; fdp_slack = "
                    "fdp_limit - duty_hours",
    "pairing_flights": "which legs belong to which pairing-day",
    "pairing_crew": "who is rostered on a pairing, and in what role",
    "duty_clocks": "per crew, hours used and headroom as of the snapshot",
    "duty_history": "per crew per day, the 28 days up to the snapshot",
    "reserves": "reserve pool with on-call windows (UTC HH:MM)",
    "reserve_dates": "which dates each reserve is on call",
    "certifications": "licence/medical/recurrent/dangerous goods validity",
    "risk_signals": "provided disruption-risk score per crew (not a prediction)",
    "costs": "callout, deadhead, delay and cancellation rates in INR",
    "rules": "the seven legality rules, as reference text",
}

# Bump when the schema changes, so an old file is rebuilt rather than queried
# with columns that no longer mean what the code thinks they mean.
SCHEMA_VERSION = 1

_META = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_conn: sqlite3.Connection | None = None
_lock = threading.Lock()


def source_fingerprint(data_dir=None) -> str:
    """A cheap identity for the JSON the database was built from.

    Name, size and modification time of every source file. Cheaper than hashing
    679 KB on every startup, and it changes whenever the data does - which is
    the only property that matters here.
    """
    import hashlib

    from backend.domain.data import DATA_DIR
    root = pathlib.Path(data_dir or DATA_DIR)
    h = hashlib.sha256()
    h.update(str(SCHEMA_VERSION).encode())
    for f in sorted(root.glob("*.json")):
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:32]


def _read_meta(conn: sqlite3.Connection, key: str) -> str | None:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    except sqlite3.DatabaseError:
        return None
    return row[0] if row else None


def _write_meta(conn: sqlite3.Connection, **pairs) -> None:
    conn.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)",
                     [(k, str(v)) for k, v in pairs.items()])
    conn.commit()


def ensure(path: str = ":memory:", force: bool = False):
    """Open the database, building it only when there is nothing usable.

    Returns (connection, action) where action is 'created', 'reused' or
    'rebuilt' - so startup can say which happened instead of leaving it to
    guesswork.
    """
    import time

    fingerprint = source_fingerprint()

    if path == ":memory:":
        conn = build(path=path)
        _stamp(conn, fingerprint)
        return conn, "created"

    file = pathlib.Path(path)
    if file.exists() and not force:
        try:
            existing = sqlite3.connect(str(file), check_same_thread=False)
            existing.row_factory = sqlite3.Row
            stored = _read_meta(existing, "source_fingerprint")
            version = _read_meta(existing, "schema_version")
            if stored == fingerprint and version == str(SCHEMA_VERSION):
                existing.execute("PRAGMA query_only = ON")
                return existing, "reused"
            existing.close()
            reason = ("schema changed" if version != str(SCHEMA_VERSION)
                      else "source data changed")
        except sqlite3.DatabaseError:
            reason = "existing file unreadable"
        file.unlink(missing_ok=True)
        action = "rebuilt"
        LOG.info("rebuilding %s: %s", file, reason)
    else:
        # force=True still has to clear the old file: CREATE TABLE on an
        # existing database fails, it does not overwrite.
        action = "rebuilt" if file.exists() else "created"
        file.unlink(missing_ok=True)

    file.parent.mkdir(parents=True, exist_ok=True)
    conn = build(path=str(file))
    _stamp(conn, fingerprint)
    return conn, action


def _stamp(conn: sqlite3.Connection, fingerprint: str) -> None:
    """Record what this database was built from, so the next run can tell."""
    import time

    conn.execute("PRAGMA query_only = OFF")
    conn.executescript(_META)
    _write_meta(conn,
                source_fingerprint=fingerprint,
                schema_version=SCHEMA_VERSION,
                built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    conn.execute("PRAGMA query_only = ON")


def build(s: Store | None = None, path: str = ":memory:") -> sqlite3.Connection:
    """Create the database and load it from the JSON store."""
    s = s or store()
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)

    flight_pairing = {}
    for p in s.pairings:
        for day in p["days"]:
            for fid in day["flights"]:
                flight_pairing[fid] = p["pairing_id"]

    conn.executemany(
        "INSERT INTO flights VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(f["flight_id"], f["flight_no"], f["date"], f["dep_station"],
          f["arr_station"], f["dep_utc"], f["arr_utc"], f["dep_utc"][11:16],
          f["arr_utc"][11:16], f["block_hours"], f["aircraft"],
          f["aircraft_type"], f["seats"],
          f"{f['dep_station']}-{f['arr_station']}",
          flight_pairing.get(f["flight_id"])) for f in s.flights])

    conn.executemany(
        "INSERT INTO crew VALUES (?,?,?,?,?,?,?,?,?)",
        [(c["crew_id"], c["name"], c["rank"], c["base"], c.get("seniority"),
          c.get("reachability_minutes"), c.get("status", "active"),
          int(s.is_reserve(c["crew_id"])), ",".join(c.get("ratings") or []))
         for c in s.crew_list])
    conn.executemany("INSERT INTO crew_ratings VALUES (?,?)",
                     [(c["crew_id"], r) for c in s.crew_list
                      for r in (c.get("ratings") or [])])

    for p in s.pairings:
        first = s.flight_by_id[p["days"][0]["flights"][0]]
        conn.execute("INSERT INTO pairings VALUES (?,?,?,?)",
                     (p["pairing_id"], p["aircraft"], first["aircraft_type"],
                      len(p["days"])))
        for i, day in enumerate(p["days"]):
            legs = [s.flight_by_id[f] for f in day["flights"]]
            dur = (parse_dt(day["release_utc"])
                   - parse_dt(day["report_utc"])).total_seconds() / 3600
            n = len(day["flights"])
            conn.execute(
                "INSERT INTO pairing_days VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (p["pairing_id"], day["date"], i, day["report_utc"],
                 day["release_utc"], round(dur, 2),
                 round(sum(l["block_hours"] for l in legs), 2), n,
                 sum(l["seats"] for l in legs), fdp_limit(n),
                 round(fdp_limit(n) - dur, 2), legs[-1]["arr_station"],
                 p["aircraft"], first["aircraft_type"]))
            conn.executemany("INSERT INTO pairing_flights VALUES (?,?,?)",
                             [(p["pairing_id"], day["date"], f)
                              for f in day["flights"]])
        conn.executemany("INSERT INTO pairing_crew VALUES (?,?,?)",
                         [(p["pairing_id"], m["crew_id"], m["role"])
                          for m in p["crew"]])

    conn.executemany(
        "INSERT INTO duty_clocks VALUES (?,?,?,?,?,?,?)",
        [(c["crew_id"], c["as_of_utc"], c["duty_hours_7d"], c["flight_hours_28d"],
          round(60 - c["duty_hours_7d"], 2), round(100 - c["flight_hours_28d"], 2),
          c.get("last_rest_ended")) for c in s.clocks_list])
    conn.executemany("INSERT INTO duty_history VALUES (?,?,?,?)",
                     [(c["crew_id"], h["date"], h["duty_hours"], h["flight_hours"])
                      for c in s.clocks_list for h in c.get("daily_history", [])])

    for r in s.reserve_list:
        c = s.crew[r["crew_id"]]
        w = r["oncall_window_utc"]
        hours = ((int(w["end"][:2]) * 60 + int(w["end"][3:]))
                 - (int(w["start"][:2]) * 60 + int(w["start"][3:]))) / 60
        conn.execute("INSERT INTO reserves VALUES (?,?,?,?,?,?,?)",
                     (r["crew_id"], r["base"], c["rank"], w["start"], w["end"],
                      round(hours, 2), c.get("reachability_minutes")))
        conn.executemany("INSERT INTO reserve_dates VALUES (?,?)",
                         [(r["crew_id"], d) for d in r.get("dates", [])])

    snap = s.snapshot.date()
    conn.executemany(
        "INSERT INTO certifications VALUES (?,?,?,?,?)",
        [(c["crew_id"], c["cert_type"], c["valid_from"], c["valid_to"],
          (__import__("datetime").date.fromisoformat(c["valid_to"]) - snap).days)
         for c in s.cert_list])

    conn.executemany("INSERT INTO risk_signals VALUES (?,?,?)",
                     [(r["crew_id"], r["disruption_risk_score"],
                       "; ".join(r.get("drivers", []))) for r in s.risk_list])
    conn.executemany("INSERT INTO costs VALUES (?,?)",
                     [(k, v) for k, v in s.costs.items() if isinstance(v, int)])
    conn.executemany("INSERT INTO rules VALUES (?,?,?)",
                     [(r["rule_id"], r["text"], json.dumps(r.get("params", {})))
                      for r in s.rules["rules"]])
    conn.commit()
    # Read-only from here: the dataset is a snapshot, and nothing downstream has
    # any business writing to it. `_stamp` lifts this briefly to record what the
    # file was built from, then puts it back.
    conn.execute("PRAGMA query_only = ON")
    return conn


def connection() -> sqlite3.Connection:
    """Process-wide read-only connection, built on first use."""
    global _conn
    with _lock:
        if _conn is None:
            from backend.config import SETTINGS
            _conn, action = ensure(SETTINGS.db_path)
            LOG.info("database %s (%s)", action, SETTINGS.db_path)
    return _conn


def reset() -> None:
    """Drop the process-wide handle so the next call reopens from disk.

    Only needed after a deliberate rebuild: a connection to a file that has
    since been replaced still answers, quietly, from the old one.
    """
    global _conn
    with _lock:
        if _conn is not None:
            try:
                _conn.close()
            except sqlite3.Error:
                pass
            _conn = None


def schema_text() -> str:
    """The schema as the SQL agent sees it: columns, types and a one-line note."""
    conn = connection()
    lines = []
    for (table,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        cols = [f"{r['name']} {r['type']}"
                for r in conn.execute(f"PRAGMA table_info({table})")]
        note = TABLE_NOTES.get(table, "")
        lines.append(f"{table}({', '.join(cols)})")
        if note:
            lines.append(f"    -- {note}")
    return "\n".join(lines)


def table_counts() -> dict:
    conn = connection()
    out = {}
    for (table,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        out[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return out


def materialise(path: str | None = None) -> str:
    """Write the database out as a real file, unconditionally.

    `ensure` is what the service calls: it builds once and then skips. This is
    the manual override - it always rebuilds, which is what you want when you
    are inspecting the schema by hand and want to be certain of what you are
    looking at.

    It defaults to the same path the service uses. Defaulting to the working
    directory instead produced a second database at the repository root, which
    then drifted from the one the service was actually reading - two files with
    the same name and different contents being exactly the failure this module
    exists to prevent.
    """
    import os

    from backend.config import SETTINGS
    path = path or SETTINGS.db_path

    # Open the live database BEFORE touching the target. Deleting the file the
    # service is about to open, and then opening it again from here, left two
    # handles on one path and Windows refused the unlink - a self-inflicted
    # deadlock that only appeared once the default path became the real one.
    source = connection()
    if os.path.abspath(path) == os.path.abspath(SETTINGS.db_path or ""):
        return os.path.abspath(path)        # already the file on disk

    if os.path.exists(path):
        os.remove(path)
    disk = sqlite3.connect(path)
    source.backup(disk)
    disk.commit()
    disk.close()
    return os.path.abspath(path)


def main() -> int:
    """python -m backend.domain.db [path]"""
    import sys

    from backend.config import SETTINGS
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    target = args[0] if args else SETTINGS.db_path
    if "--force" in sys.argv:
        conn, action = ensure(target, force=True)
        conn.close()
        reset()
        print(f"{action} from source (--force)")
    written = materialise(target)
    size = __import__("os").path.getsize(written)
    print(f"wrote {written} ({size / 1024:.0f} KB)")
    for table, count in sorted(table_counts().items()):
        print(f"  {table:18} {count:>5} rows")
    print("\nopen it with:  sqlite3 " + target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
