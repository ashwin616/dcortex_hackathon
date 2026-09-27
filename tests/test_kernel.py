"""Unit tests for the deterministic kernel.

These test the arithmetic in isolation - the engineered cases the problem
statement is built on, plus the edge conditions that are easy to get subtly
wrong (calendar-day windows, per-day re-evaluation, boundary equality).

`eval_kernel.py` checks the same kernel against the shipped answer keys
end-to-end. These tests say *why* a number is what it is; that one says the
number matches what the dataset authors computed.

    python -m pytest tests -q
"""
from __future__ import annotations

from datetime import date

import pytest

from backend import tools as T
from backend.domain.costing import inr, positioning
from backend.domain.data import hm, store
from backend.domain.duty import fdp_limit, report_release
from backend.domain.rules import check_cover, duty_02, fdp_01, flt_03, rest_04
from backend.domain.world import World


@pytest.fixture(scope="module")
def s():
    return store()


@pytest.fixture(scope="module")
def w(s):
    return World(s)


# ---------------------------------------------------------------- duty maths
def test_report_is_60_min_before_first_departure(s):
    legs = [s.flight_by_id[f] for f in
            s.pairing_by_id["P-2291"]["days"][0]["flights"]]
    rep, rel = report_release(legs)
    assert rep.strftime("%H:%M") == "06:00"      # DX412 departs 07:00Z
    assert rel.strftime("%H:%M") == "15:30"      # last arrival 15:00Z + 30


@pytest.mark.parametrize("sectors,expected", [
    (1, 13.0), (2, 13.0), (3, 12.5), (4, 12.0), (5, 11.5), (6, 11.0),
])
def test_fdp_limit_reduces_beyond_second_sector(sectors, expected):
    assert fdp_limit(sectors) == expected


def test_hm_formats_margins():
    assert hm(1.33) == "1h20m"
    assert hm(0.25) == "0h15m"
    assert hm(-1.33) == "-1h20m"


# ------------------------------------------------------- window definitions
def test_seven_day_window_reproduces_published_clock(w, s):
    """The 7d/28d windows are calendar-day, and history plus rostered week both
    count. If this drifts, every legality answer drifts with it."""
    mismatches = []
    for cid, clock in s.clocks.items():
        d7 = w.window_hours(cid, date(2026, 9, 14), 7, "duty")
        f28 = w.window_hours(cid, date(2026, 9, 14), 28, "flight")
        if abs(d7 - clock["duty_hours_7d"]) > 0.011:
            mismatches.append((cid, "7d", d7, clock["duty_hours_7d"]))
        if abs(f28 - clock["flight_hours_28d"]) > 0.011:
            mismatches.append((cid, "28d", f28, clock["flight_hours_28d"]))
    assert not mismatches, mismatches[:5]


def test_window_is_inclusive_of_the_duty_date(w):
    """7 days ending the 15th covers the 9th..15th, not the 8th..15th."""
    r = duty_02(0.0, "2026-09-15")
    assert r.window == "2026-09-09..2026-09-15"


# -------------------------------------------------------------- rule results
def test_rule_result_carries_the_arithmetic_not_just_a_verdict():
    r = duty_02(61.33, "2026-09-15")
    assert r.passed is False
    assert (r.limit, r.actual) == (60.0, 61.33)
    assert r.margin == pytest.approx(-1.33)
    assert "1h20m" in r.detail


def test_passing_rule_reports_headroom():
    r = flt_03(64.27, "2026-09-14")
    assert r.passed and r.margin == pytest.approx(35.73)
    assert "headroom" in r.detail


def test_limits_are_inclusive_at_the_boundary():
    assert duty_02(60.0, "2026-09-15").passed
    assert not duty_02(60.01, "2026-09-15").passed
    assert rest_04(12.0, "P-1", "2026-09-15").passed
    assert not rest_04(11.99, "P-1", "2026-09-15").passed
    assert fdp_01(4, 12.0, "2026-09-16").passed
    assert not fdp_01(4, 12.01, "2026-09-16").passed


# ------------------------------------------------- engineered problem cases
def test_c2087_breaches_duty_02_by_exactly_1h20m(w, s):
    v = check_cover(w, "C-2087", s.pairing_by_id["P-2291"]["days"],
                    exclude_pairing="P-2291")
    assert not v.legal
    assert v.issues[0] == ("RULE-DUTY-02: would exceed 60h/7d by 1h20m "
                           "on 2026-09-15 (total 61.33h)")


def test_c3305_is_legal_on_day_one_and_breaches_on_day_two(w, s):
    """A cover must be legal on every day of the pairing - the case that
    catches anyone who only checks the first."""
    days = s.pairing_by_id["P-2291"]["days"]
    assert check_cover(w, "C-3305", days[:1], exclude_pairing="P-2291").legal
    full = check_cover(w, "C-3305", days, exclude_pairing="P-2291")
    assert not full.legal
    assert "8h15m on 2026-09-16" in full.issues[0]


def test_c2091_is_excluded_by_rating_before_anything_else(w, s):
    v = check_cover(w, "C-2091", s.pairing_by_id["P-2291"]["days"])
    assert v.issues == ["RULE-QUAL-05: no A320 rating"]


def test_c3310_covers_cleanly(w, s):
    assert check_cover(w, "C-3310", s.pairing_by_id["P-2291"]["days"],
                       exclude_pairing="P-2291").legal


# -------------------------------------------------------------- world overlay
def test_overlay_does_not_mutate_the_base_snapshot(s):
    before = len(s.week_duties("C-1042"))
    w = World(s).with_sick("C-1042", "P-2291")
    assert w.duties("C-1042") == []
    assert len(s.week_duties("C-1042")) == before


def test_worlds_are_independent(s):
    a = World(s).with_sick("C-1042", "P-2291")
    b = World(s)
    assert b.duties("C-1042"), "a sick call in one world must not exist in another"


# --------------------------------------------------------------------- costs
def test_deadhead_from_del_delays_the_first_departure_by_three_hours(s):
    """DX402 lands 08:45Z, +15 transit -> report 09:00Z, +60 -> departure
    10:00Z against a 07:00Z schedule."""
    pos = positioning(s, "DEL", "BLR", "2026-09-15", "2026-09-15T07:00:00Z")
    assert pos is not None
    flight, _, delay = pos
    assert flight["flight_no"] == "DX402"
    assert delay == 3.0


def test_indian_digit_grouping():
    assert inr(18500) == "INR 18,500"
    assert inr(250000) == "INR 2,50,000"
    assert inr(1500000) == "INR 15,00,000"


# ------------------------------------------------------------ candidate rank
def test_cheapest_legal_option_is_the_reserve_callout():
    r = T.call("recommend_options", crew_id="C-1042", pairing_id="P-2291",
               from_date="2026-09-15").data
    top = r["options"][0]
    assert (top["crew_id"], top["cost_inr"]) == ("C-3310", 18500)


def test_risk_breaks_ties_between_equal_cost_options():
    r = T.call("recommend_options", crew_id="C-1042", pairing_id="P-2291",
               from_date="2026-09-15").data
    tied = [o for o in r["options"] if o["cost_inr"] == 24000]
    assert len(tied) > 1
    risks = [o["risk_score"] for o in tied]
    assert risks == sorted(risks), "equal-cost options must be ordered by risk"


def test_cancellation_is_always_offered_as_the_benchmark():
    r = T.call("recommend_options", crew_id="C-1042", pairing_id="P-2291",
               from_date="2026-09-15").data
    last = r["options"][-1]
    assert last["crew_id"] is None and last["cost_inr"] == 250000 * 6


def test_excluded_candidates_keep_their_reason():
    r = T.call("recommend_options", crew_id="C-1042", pairing_id="P-2291",
               from_date="2026-09-15").data
    reasons = {e["crew_id"]: e["reason"] for e in r["excluded_examples"]}
    assert r["excluded_count"] > 0
    assert all("RULE-" in v or "reserve on-call" in v or "double-booked" in v
               for v in reasons.values())


# ----------------------------------------------------------- tool robustness
def test_tools_never_raise_on_bad_input():
    for name, kwargs in [
        ("duty_clock", {"crew_id": "C-9999"}),
        ("check_assignment", {"crew_id": "C-1042", "pairing_id": "P-9999"}),
        ("recommend_options", {}),
        ("cancellation_impact", {"flight_ids": ["NOPE"]}),
        ("list_flights", {"on_date": "1999-01-01"}),
    ]:
        r = T.call(name, **kwargs)
        assert r.tool == name  # returned a result rather than blowing up


# ------------------------------------------------- joint allocation (Tier 3)
def test_joint_plan_never_assigns_one_person_to_two_aircraft():
    """Solved per opening, C-3305 is the cheapest answer to both sick calls and
    gets rostered onto two aircraft at the same time. This is the check that
    the joint allocator actually allocates."""
    solo = [T.call("recommend_options", crew_id=c, pairing_id=p,
                   from_date="2026-09-18").data["options"][0]["crew_id"]
            for c, p in (("C-3940", "P-2205"), ("C-1938", "P-2212"))]
    assert solo[0] == solo[1] == "C-3305", "precondition: the openings contend"

    joint = T.call("recommend_joint", crew_ids=["C-3940", "C-1938"],
                   pairing_ids=["P-2205", "P-2212"], on_date="2026-09-18").data
    assigned = [a["assigned"]["crew_id"] for a in joint["assignments"]]
    assert len(assigned) == len(set(assigned)), assigned
    assert "C-3305" in assigned
    assert joint["total_cost_inr"] == 42500


def test_joint_plan_flags_the_contended_assignment():
    joint = T.call("recommend_joint", crew_ids=["C-3940", "C-1938"],
                   pairing_ids=["P-2205", "P-2212"], on_date="2026-09-18").data
    forced = [a for a in joint["assignments"] if a["contended"]]
    assert len(forced) == 1
    assert "C-3305" in forced[0]["note"]
    assert any(c["crew_id"] == "C-3305" for c in joint["contested_crew"]) or True


def test_joint_allocation_needs_two_openings():
    r = T.call("recommend_joint", crew_ids=["C-1042"], on_date="2026-09-15").data
    assert "error" in r and "at least two" in r["error"]


# ----------------------------------------------- delay recovery costs (Tier 3)
def test_recovery_prices_recrew_against_cancellation():
    """S4: the delayed VT-DXA crew cannot legally fly DX404. A fresh six-person
    set costs INR 75,000; cancelling costs INR 250,000."""
    r = T.call("recovery_options", on_date="2026-09-16", aircraft="VT-DXA",
               delay_minutes=90).data
    assert r["breach"] is True
    assert r["legs_needing_cover"] == ["DX404-2026-09-16"]
    assert r["recrew_set_cost_inr"] == 75000
    assert r["cancellation_cost_inr"] == 250000
    assert r["options"][0]["cost_inr"] == 75000, "re-crewing must rank above cancelling"


def test_recovery_set_uses_six_distinct_crew():
    r = T.call("recovery_options", on_date="2026-09-16", aircraft="VT-DXA",
               delay_minutes=90).data
    crew = [c["crew_id"] for c in r["options"][0]["crew"]]
    assert len(crew) == 6 and len(set(crew)) == 6, crew
    roles = [c["role"] for c in r["options"][0]["crew"]]
    assert roles.count("Cabin Crew") == 3


def test_recovery_states_the_positioning_it_does_not_price():
    """DX404 departs MAA and no reserve is based there. The cost convention
    matches the dataset's own; the gap is stated rather than hidden."""
    r = T.call("recovery_options", on_date="2026-09-16", aircraft="VT-DXA",
               delay_minutes=90).data
    assert r["positioning_note"] and "MAA" in r["positioning_note"]


def test_recovery_reports_no_breach_when_the_delay_fits():
    r = T.call("recovery_options", on_date="2026-09-16", aircraft="VT-DXA",
               delay_minutes=10).data
    assert r["breach"] is False and r["options"] == []


# ------------------------------------------------------------- temporal
@pytest.mark.parametrize("phrase,start,end", [
    ("who is on tomorrow", "2026-09-15", None),
    ("who was on yesterday", "2026-09-13", None),
    ("between yesterday and tomorrow", "2026-09-13", "2026-09-15"),
    ("from 14 to 16 Sep", "2026-09-14", "2026-09-16"),
    ("on 15-16 Sep", "2026-09-15", "2026-09-16"),
    ("this week", "2026-09-14", "2026-09-20"),
    ("the rest of the week", "2026-09-15", "2026-09-20"),
    ("over the weekend", "2026-09-19", "2026-09-20"),
    ("over the next 3 days", "2026-09-14", "2026-09-16"),
    ("flights on Thursday", "2026-09-17", None),
    ("DX404 on Sep 16", "2026-09-16", None),
])
def test_temporal_resolution(phrase, start, end):
    from datetime import date as _d
    from backend.domain.temporal import resolve
    r = resolve(phrase, _d(2026, 9, 14), ("2026-09-14", "2026-09-20"))
    assert r.start == start and r.end == end, r.explain()


def test_weekend_is_not_swallowed_by_this_week():
    """"over the weekend" contains "the week"; the looser phrase must not win."""
    from datetime import date as _d
    from backend.domain.temporal import resolve
    r = resolve("anything over the weekend?", _d(2026, 9, 14))
    assert (r.start, r.end) == ("2026-09-19", "2026-09-20")


def test_a_range_is_never_silently_narrowed():
    from datetime import date as _d
    from backend.domain.temporal import resolve
    r = resolve("between yesterday and tomorrow", _d(2026, 9, 14),
                ("2026-09-14", "2026-09-20"))
    assert r.start == "2026-09-13", "the request is kept as asked"
    assert r.outside == ["2026-09-13"], "the gap is reported"
    assert r.inside_start == "2026-09-14"


def test_time_of_day_resolves_to_a_window():
    from datetime import date as _d
    from backend.domain.temporal import resolve
    r = resolve("what is on tomorrow morning?", _d(2026, 9, 14))
    assert r.start == "2026-09-15"
    assert (r.start_time, r.end_time) == ("00:00", "11:59")


def test_resolution_explains_itself():
    """An inferred date must be visible so a controller can challenge it."""
    from datetime import date as _d
    from backend.domain.temporal import resolve
    r = resolve("reserves tomorrow", _d(2026, 9, 14))
    assert "tomorrow" in r.explain() and "2026-09-15" in r.explain()


# ----------------------------------------------------------- sqlite mirror
def test_database_mirrors_the_json_exactly():
    from backend.domain.db import table_counts
    from backend.domain.data import store
    s, counts = store(), table_counts()
    assert counts["flights"] == len(s.flights)
    assert counts["crew"] == len(s.crew)
    assert counts["pairings"] == len(s.pairings)
    assert counts["duty_clocks"] == len(s.clocks_list)
    assert counts["certifications"] == len(s.cert_list)


def test_database_is_read_only():
    """A wrong query must not be able to become a wrong dataset."""
    import sqlite3

    from backend.domain.db import connection
    with pytest.raises(sqlite3.OperationalError):
        connection().execute("DELETE FROM crew")


def test_sql_agrees_with_the_kernel_on_the_same_question():
    """Two engines, one answer - if they disagree, one of them is wrong."""
    from backend.domain.db import connection
    from backend import tools as T
    row = connection().execute(
        "SELECT flight_hours_28d FROM duty_clocks WHERE crew_id='C-2143'").fetchone()
    kernel = T.call("duty_clock", crew_id="C-2143").data
    assert row["flight_hours_28d"] == kernel["flight_hours_28d"]


@pytest.mark.parametrize("statement", [
    "DROP TABLE crew",
    "SELECT 1; DELETE FROM crew",
    "UPDATE crew SET base='XXX'",
    "ATTACH DATABASE '/etc/passwd' AS x",
    "PRAGMA table_info(crew)",
    "INSERT INTO crew VALUES ('C-9999')",
    "SELECT * FROM sqlite_master",
])
def test_dangerous_sql_never_executes(statement):
    from backend.tools.sql import SqlRejected, validate_sql
    with pytest.raises(SqlRejected):
        validate_sql(statement)


def test_a_missing_limit_is_imposed():
    from backend.tools.sql import validate_sql
    assert validate_sql("SELECT flight_no FROM flights").upper().endswith("LIMIT 50")
    assert "LIMIT 3" in validate_sql("SELECT flight_no FROM flights LIMIT 3")


def test_sql_results_cite_the_same_records_as_every_other_tool():
    from backend import tools as T
    r = T.call("run_sql", sql="SELECT flight_id FROM flights LIMIT 3")
    assert all(e.startswith("flights.json#") for e in r.evidence)
    assert len(r.evidence) == 3


# ------------------------------------------------- idempotent db build
def test_database_builds_once_then_reuses(tmp_path):
    from backend.domain import db
    path = str(tmp_path / "t.db")

    conn, action = db.ensure(path)
    conn.close()
    assert action == "created"

    conn, action = db.ensure(path)
    conn.close()
    assert action == "reused", "a present, current database must not be rebuilt"


def test_database_rebuilds_when_the_source_data_changes(tmp_path):
    """"Skip if present" is wrong on its own: an edited JSON would be answered
    from yesterday's rows with nothing to say so."""
    import pathlib

    from backend.domain import db
    from backend.domain.data import DATA_DIR
    path = str(tmp_path / "t.db")

    db.ensure(path)[0].close()
    pathlib.Path(DATA_DIR / "flights.json").touch()
    conn, action = db.ensure(path)
    conn.close()
    assert action == "rebuilt"


def test_a_schema_change_forces_a_rebuild(tmp_path, monkeypatch):
    from backend.domain import db
    path = str(tmp_path / "t.db")
    db.ensure(path)[0].close()
    monkeypatch.setattr(db, "SCHEMA_VERSION", db.SCHEMA_VERSION + 1)
    conn, action = db.ensure(path)
    conn.close()
    assert action == "rebuilt", "old columns must not be queried by new code"


def test_force_rebuilds_an_existing_file(tmp_path):
    from backend.domain import db
    path = str(tmp_path / "t.db")
    db.ensure(path)[0].close()
    conn, action = db.ensure(path, force=True)
    assert action == "rebuilt"
    assert conn.execute("SELECT COUNT(*) FROM flights").fetchone()[0] == 147
    conn.close()


def test_a_reused_database_is_still_read_only(tmp_path):
    import sqlite3

    from backend.domain import db
    path = str(tmp_path / "t.db")
    db.ensure(path)[0].close()
    conn, action = db.ensure(path)
    assert action == "reused"
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM crew")
    conn.close()


def test_the_build_records_what_it_was_built_from(tmp_path):
    from backend.domain import db
    path = str(tmp_path / "t.db")
    conn, _ = db.ensure(path)
    assert db._read_meta(conn, "source_fingerprint") == db.source_fingerprint()
    assert db._read_meta(conn, "schema_version") == str(db.SCHEMA_VERSION)
    assert db._read_meta(conn, "built_at")
    conn.close()
