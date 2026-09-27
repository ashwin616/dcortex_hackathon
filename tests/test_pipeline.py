"""Unit tests for the planner / validator / executor pipeline.

No network. The planner's LLM route is exercised only through its fast path
and through hand-written plans, so these run in under a second and fail for
exactly one reason: the code changed.
"""
from __future__ import annotations

import pytest

from backend.cache import TTLCache, normalise_question
from backend.executor.executor import execute, resolve_args, topological_order
from backend.grounding import check, normalize
from backend.planner.planner import extract_entities, fastpath
from backend.ratelimit import Budget, ModelPool, estimate_tokens, parse_reset
from backend.config import MODEL_SPECS
from backend.planner.schemas import ExecutionPlan, PlanStep
from backend.executor.validator import validate
from backend import tools as T


def plan_of(op, args=None, **kw):
    return ExecutionPlan(intent="test", entities={},
                         steps=[PlanStep(id="s1", operation=op, args=args or {}, **kw)])


# ------------------------------------------------------------ entity parsing
def test_extracts_identifiers_and_dates():
    e = extract_entities("Captain C-1042 is out for pairing P-2291 on 15-16 Sep")
    assert e["crew_id"] == "C-1042"
    assert e["pairing_id"] == "P-2291"
    assert e["role"] == "Captain"
    assert e["date"] == "2026-09-15", "a day range means the day it starts"
    assert e["date_end"] == "2026-09-16"


def test_relative_dates_resolve_against_the_snapshot():
    assert extract_entities("who is on reserve tomorrow")["date"] == "2026-09-15"
    assert extract_entities("what is on today")["date"] == "2026-09-14"


def test_closure_window_and_delay_are_parsed():
    e = extract_entities("BLR is closed 08:00-14:00Z on 17 Sep")
    assert (e["start_utc"], e["end_utc"], e["station"]) == ("08:00", "14:00", "BLR")
    assert extract_entities("VT-DXA is delayed 90 minutes")["delay_minutes"] == 90


# ---------------------------------------------------------------- fast path
@pytest.mark.parametrize("question,operation", [
    ("Who is on reserve at BLR on 2026-09-15?", "list_reserves"),
    ("How many duty hours does C-1042 have left this week?", "duty_clock"),
    ("Which flights depart DEL on 2026-09-15?", "list_flights"),
    ("List all certifications expiring within 30 days of 2026-09-15.",
     "certifications_expiring"),
    ("Which crew are assigned to pairing P-2291, and in what roles?", "get_pairing"),
    ("What is the disruption-risk score for C-1042?", "risk_signal"),
    ("Which stations does the network serve nonstop from BLR?", "schedule_stats"),
])
def test_fastpath_routes_lookups_without_a_model(question, operation):
    p = fastpath(question)
    assert p is not None and p.steps[0].operation == operation


@pytest.mark.parametrize("question", [
    "Captain C-1042 is out for P-2291. What should I do?",
    "Outline the recovery plan for the BLR closure.",
    "If the desk wants a standing morning briefing, what should it surface and why?",
])
def test_fastpath_declines_anything_needing_judgement(question):
    assert fastpath(question) is None


@pytest.mark.parametrize("question,operation", [
    ("If C-2087 covers P-2291, does anything breach?", "check_assignment"),
    ("Captain C-5837 is proposed to cover P-2291. Legal?", "check_assignment"),
    ("BLR is closed 08:00-14:00Z on 17 Sep. Which flights are affected?",
     "station_closure_impact"),
    ("Captain C-1042 calls in sick on 15 Sep for P-2291. Which flights are uncrewed?",
     "sick_call_impact"),
    ("Draft the callout notification to C-3310 for covering P-2291.",
     "notification_facts"),
    ("If DX404 on 16 Sep is cancelled, what is the cancellation cost?",
     "cancellation_impact"),
    ("Which crew have 45 or more duty hours in the 7 days ending 2026-09-15?",
     "crew_near_limits"),
    ("A crew is released at 15:30Z on 16 Sep. When may they report next?",
     "earliest_next_report"),
    ("How many captains are based at DEL, and who are they?", "find_crew"),
])
def test_fastpath_covers_the_rigid_shapes(question, operation):
    """Every one of these skips the planner: ~1s and ~1,500 tokens saved, and
    the routing stops depending on which model is answering."""
    p = fastpath(question)
    assert p is not None, question
    assert p.steps[-1].operation == operation, p.summary()
    assert validate(p, extract_entities(question)).ok


def test_a_named_candidate_is_a_legality_question_not_a_cover_search():
    """"C-5837 is proposed to cover P-2291. Legal?" names who is proposed, so
    it must check that person - not go shopping for a cheaper one."""
    p = fastpath("Captain C-5837 (VT-DXA line) is proposed to cover P-2291 "
                 "on 15 Sep. Legal?")
    assert p.steps[-1].operation == "check_assignment"
    assert p.steps[-1].args.get("crew_id") == "C-5837"


def test_every_fastpath_plan_passes_the_validator():
    for q in ["Who is on reserve at BLR on 2026-09-15?",
              "Which flights fly BLR to BOM on 2026-09-17?",
              "What is C-2210 base and rating?"]:
        p = fastpath(q)
        assert validate(p, extract_entities(q)).ok, q


# ---------------------------------------------------------------- validator
def test_unknown_operation_is_refused():
    """`run_sql` used to be the example here; it is a real operation now, which
    is exactly why the allowlist has to be checked against the live registry
    rather than a hardcoded idea of what exists."""
    v = validate(plan_of("execute_python", {"code": "import os"}))
    assert not v.ok and "unknown operation" in v.report()
    v = validate(plan_of("shell", {"cmd": "rm -rf /"}))
    assert not v.ok and "unknown operation" in v.report()


def test_the_sql_operation_is_allowlisted_but_still_cannot_write():
    """Adding SQL widened what the planner may ask for, not what it may do."""
    from backend.tools.sql import SqlRejected, validate_sql
    assert validate(plan_of("run_sql", {"sql": "SELECT 1 LIMIT 1"})).ok
    with pytest.raises(SqlRejected):
        validate_sql("DROP TABLE crew")


def test_nonexistent_crew_is_refused():
    v = validate(plan_of("duty_clock", {"crew_id": "C-9999"}))
    assert not v.ok and "not in the dataset" in v.report()


def test_date_outside_the_schedule_week_is_refused():
    v = validate(plan_of("list_flights", {"on_date": "2027-01-01"}))
    assert not v.ok and "outside the schedule week" in v.report()


def test_step_limit_is_enforced():
    steps = [PlanStep(id=f"s{i}", operation="find_crew", args={"rank": "Captain"})
             for i in range(9)]
    v = validate(ExecutionPlan(intent="x", steps=steps))
    assert not v.ok and "exceeds the limit" in v.report()


def test_argument_synonyms_are_repaired_not_rejected():
    v = validate(plan_of("list_flights", {"date": "2026-09-15", "origin": "DEL"}))
    assert v.ok
    assert v.plan.steps[0].args == {"on_date": "2026-09-15", "dep_station": "DEL"}


def test_role_casing_is_canonicalised():
    v = validate(plan_of("recommend_options",
                         {"pairing_id": "P-2291", "role": "captain"}))
    assert v.plan.steps[0].args["role"] == "Captain"


def test_empty_list_does_not_count_as_a_supplied_argument():
    """The bug that made Tier 3 fail: flight_ids=[] is present but useless."""
    step = PlanStep(id="s1", operation="recommend_options",
                    args={"flight_ids": [], "role": "captain"})
    assert "flight_ids" not in step.args


def test_missing_identifier_is_backfilled_from_the_question():
    v = validate(plan_of("recommend_options", {"role": "Captain"}),
                 {"crew_id": "C-1042", "pairing_id": "P-2291", "date": "2026-09-15"})
    assert v.ok
    assert v.plan.steps[0].args["crew_id"] == "C-1042"
    assert any("backfilled" in r for r in v.repaired)


def test_operation_with_no_usable_argument_is_refused():
    v = validate(plan_of("recommend_options", {"role": "Captain"}), {})
    assert not v.ok and "at least one of" in v.report()


def test_dependency_cycles_are_refused():
    p = ExecutionPlan(intent="x", steps=[
        PlanStep(id="a", operation="find_crew", args={"rank": "Captain"}, depends_on=["b"]),
        PlanStep(id="b", operation="find_crew", args={"rank": "Captain"}, depends_on=["a"]),
    ])
    assert not validate(p).ok


def test_abstain_plan_is_valid():
    p = ExecutionPlan(intent="abstain", steps=[],
                      abstain_reason="no operation predicts sick calls")
    assert validate(p).ok


# ----------------------------------------------------------------- executor
def test_steps_run_in_dependency_order():
    p = ExecutionPlan(intent="x", steps=[
        PlanStep(id="second", operation="recommend_options", depends_on=["first"]),
        PlanStep(id="first", operation="sick_call_impact",
                 args={"crew_id": "C-1042", "pairing_id": "P-2291",
                       "from_date": "2026-09-15"}),
    ])
    assert [s.id for s in topological_order(p)] == ["first", "second"]


def test_executor_carries_identifiers_between_steps():
    p = ExecutionPlan(intent="x", steps=[
        PlanStep(id="impact", operation="sick_call_impact",
                 args={"crew_id": "C-1042", "pairing_id": "P-2291",
                       "from_date": "2026-09-15"}),
        PlanStep(id="rec", operation="recommend_options", depends_on=["impact"]),
    ])
    ctx = execute(p)
    assert ctx.ok
    assert ctx.steps[1].args.get("crew_id") == "C-1042"
    assert ctx.results["rec"].data["options"][0]["crew_id"] == "C-3310"


def test_a_failed_step_is_recorded_not_raised():
    ctx = execute(plan_of("duty_clock", {"crew_id": "C-1042"}))
    assert ctx.ok
    bad = execute(plan_of("recommend_options", {"role": "Captain"}))
    assert not bad.ok and bad.errors()


def test_execution_collects_evidence_and_rules():
    ctx = execute(plan_of("check_assignment",
                          {"crew_id": "C-2087", "pairing_id": "P-2291",
                           "on_date": "2026-09-15"}))
    assert ctx.evidence and ctx.rules_evaluated
    assert any(not r["passed"] for r in ctx.rules_evaluated)


# ---------------------------------------------------------------- grounding
def test_grounding_accepts_facts_that_came_from_a_tool():
    r = T.call("check_assignment", crew_id="C-2087", pairing_id="P-2291",
               on_date="2026-09-15")
    ok, bad = check("C-2087 on P-2291 exceeds the 60h limit: 61.33h.", [r])
    assert ok, bad


def test_grounding_rejects_an_invented_crew_member():
    r = T.call("duty_clock", crew_id="C-1042")
    ok, bad = check("Assign C-9999 instead.", [r])
    assert not ok and any(u["token"] == "C-9999" for u in bad)


def test_grounding_rejects_an_invented_cost():
    r = T.call("duty_clock", crew_id="C-1042")
    ok, bad = check("It will cost INR 77,777.", [r])
    assert not ok


def test_typographic_dashes_do_not_read_as_fabricated_numbers():
    """Models emit U+2011 inside identifiers; without normalising, the bare
    digits of 'P‑2291' look like an invented figure."""
    r = T.call("get_pairing", pairing_id="P-2291")
    ok, bad = check("Pairing P‑2291 is affected.", [r])
    assert ok, bad
    assert normalize("C‑1042") == "C-1042"


def test_small_counts_are_not_treated_as_claims():
    r = T.call("duty_clock", crew_id="C-1042")
    ok, _ = check("There are 3 legs to cover.", [r])
    assert ok


# --------------------------------------------------------------- rate limits
@pytest.mark.parametrize("text,seconds", [
    ("547ms", 0.547), ("1m26.4s", 86.4), ("2h19m40.8s", 8380.8), ("", 0.0),
])
def test_groq_reset_strings_parse(text, seconds):
    assert parse_reset(text) == pytest.approx(seconds, abs=0.01)


def test_budget_refuses_a_request_it_cannot_pay_for():
    b = Budget(MODEL_SPECS[0])
    assert b.headroom() > 0
    b.charge(7200)
    assert b.headroom() <= 0
    assert b.wait_for(1000) > 0


def test_pool_spreads_across_models_when_one_is_spent():
    pool = ModelPool()
    first, _ = pool.acquire("narrator", 7000)
    second, _ = pool.acquire("narrator", 7000)
    assert first.id != second.id, "a spent bucket must not be chosen again"


def test_aggregate_capacity_is_the_sum_of_the_buckets():
    cap = ModelPool().capacity()
    assert cap["aggregate_tpm"] == 8000 * len(MODEL_SPECS)


def test_token_estimate_is_biased_high():
    msgs = [{"role": "user", "content": "x" * 3600}]
    assert estimate_tokens(msgs, max_output=500) > 1000 + 500


# -------------------------------------------------------------------- cache
def test_question_normalisation_ignores_case_and_punctuation():
    assert normalise_question("Who is on RESERVE at BLR?") == \
           normalise_question("who is on reserve at blr")


def test_ttl_cache_evicts_and_counts():
    c = TTLCache(max_entries=2, ttl_s=60)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    assert c.get("a") is None      # evicted
    assert c.get("c") == 3
    assert c.stats()["hits"] == 1


# --------------------------------------------------------------- citations
def test_citations_are_derived_from_tool_results():
    from backend.grounding import citations
    r = T.call("check_assignment", crew_id="C-2087", pairing_id="P-2291",
               on_date="2026-09-15")
    built = citations.build(
        "C-2087 cannot cover P-2291: RULE-DUTY-02 is exceeded on DX412-2026-09-15.",
        [r], r.rules_evaluated)
    refs = {c["ref"] for c in built["citations"]}
    assert "crew.json#C-2087" in refs
    assert "rosters.json#P-2291" in refs
    assert "rules.json#RULE-DUTY-02" in refs
    assert not built["uncited"]


def test_citation_numbers_follow_reading_order():
    from backend.grounding import citations
    r = T.call("check_assignment", crew_id="C-2087", pairing_id="P-2291",
               on_date="2026-09-15")
    built = citations.build("C-2087 then P-2291 then RULE-DUTY-02.", [r],
                            r.rules_evaluated)
    order = [c["tokens"][0] for c in built["citations"]]
    assert order.index("C-2087") < order.index("P-2291") < order.index("RULE-DUTY-02")


def test_rule_citation_carries_the_arithmetic():
    from backend.grounding import citations
    r = T.call("check_assignment", crew_id="C-2087", pairing_id="P-2291",
               on_date="2026-09-15")
    built = citations.build("RULE-DUTY-02 is breached.", [r], r.rules_evaluated)
    rule = next(c for c in built["citations"] if c["file"] == "rules.json")
    assert "61.33" in rule["detail"] and "margin" in rule["detail"]


def test_annotation_marks_first_mention_only_and_skips_tables():
    from backend.grounding import citations
    r = T.call("get_pairing", pairing_id="P-2291")
    answer = "P-2291 is affected. P-2291 again.\n| P-2291 | row |"
    built = citations.build(answer, [r], [])
    assert built["annotated"].count("<sup") == 1
    assert "| P-2291 | row |" in built["annotated"]


def test_unknown_identifier_is_reported_uncited():
    from backend.grounding import citations
    r = T.call("get_pairing", pairing_id="P-2291")
    built = citations.build("Pairing P-2291 and crew C-9999.", [r], [])
    assert built["uncited"] == ["C-9999"]


# ------------------------------------------------------------- number parsing
def test_a_unit_suffix_does_not_truncate_a_number():
    """"23.5h" once parsed as 23.0, so the validator compared the wrong figure."""
    from backend.grounding.validator import _numbers
    assert _numbers("23.5h") == {23.5}
    assert _numbers("**12.75h**") == {12.75}
    assert _numbers("It cost INR 77,777.") == {77777.0}
    assert _numbers("61.33h vs 60h") == {61.33, 60.0}


def test_a_negative_margin_supports_its_positive_magnitude():
    from backend.grounding.validator import supported_numbers
    assert 1.33 in supported_numbers('{"margin": -1.33}')
    assert -1.33 in supported_numbers('{"margin": -1.33}')


# ------------------------------------------------------------ pool failover
def test_retries_prefer_a_model_that_has_not_failed_yet():
    from backend.ratelimit import ModelPool
    from backend.config import MODEL_SPECS
    pool = ModelPool()
    picked, failed = [], set()
    for _ in range(len(MODEL_SPECS)):
        spec, _w = pool.acquire("narrator", 100, preferred=MODEL_SPECS[0].id,
                                exclude=failed)
        picked.append(spec.id)
        failed.add(spec.id)
    assert len(set(picked)) == len(MODEL_SPECS), picked


# ------------------------------------------------------------- completeness
def test_completeness_restores_items_the_summary_dropped():
    from backend.grounding import completeness
    r = T.call("station_closure_impact", station="BLR", on_date="2026-09-17",
               start_utc="08:00", end_utc="14:00")
    partial = ("Affected: DX402-2026-09-17, DX422-2026-09-17, "
               "DX462-2026-09-17, DX453-2026-09-17.")
    out, report = completeness.complete(partial, [r])
    assert report["appended"]
    for fid in r.data["affected_flights"]:
        assert fid in out


def test_completeness_leaves_a_genuinely_complete_answer_alone():
    """Complete now means the figures too, not just the identifiers: listing
    thirteen flights without their delays is still half an answer."""
    from backend.grounding import completeness
    r = T.call("station_closure_impact", station="BLR", on_date="2026-09-17",
               start_utc="08:00", end_utc="14:00")
    rows = r.data["per_flight_assessment"]
    full = ("All affected: " + ", ".join(r.data["affected_flights"])
            + ". Needing re-crew or cancellation: "
            + ", ".join(r.data["flights_needing_recrew_or_cancel"])
            + ". Delays: " + ", ".join(f"{a['min_delay_hours']}h" for a in rows))
    _out, report = completeness.complete(full, [r])
    assert not report["appended"], report


def test_completeness_restores_the_per_flight_numbers():
    from backend.grounding import completeness
    r = T.call("station_closure_impact", station="BLR", on_date="2026-09-17",
               start_utc="08:00", end_utc="14:00")
    partial = ("Pairings P-2204 and P-2211 need re-crewing after "
               "DX402-2026-09-17 and DX422-2026-09-17 are delayed.")
    out, report = completeness.complete(partial, [r])
    assert report["assessment_rows_added"] == 13
    assert "Min delay" in out and "5.75h" in out


def test_completeness_names_the_near_miss_exclusions():
    from backend.grounding import completeness
    r = T.call("recommend_options", pairing_id="P-2224", from_date="2026-09-16",
               role="Captain")
    out, report = completeness.complete("Captain C-3315 can cover it.", [r])
    assert report["exclusions_added"] > 0
    assert "C-3310" in out, "the reserve whose window just missed comes first"


def test_completeness_ignores_a_set_the_answer_is_not_discussing():
    """A question about one flight must not drag in a table of thirteen."""
    from backend.grounding import completeness
    r = T.call("station_closure_impact", station="BLR", on_date="2026-09-17",
               start_utc="08:00", end_utc="14:00")
    _out, report = completeness.complete("BLR closes for six hours.", [r])
    assert not report["appended"]


def test_completeness_survives_the_cancellation_option_having_no_crew():
    from backend.grounding import completeness
    r = T.call("recommend_options", crew_id="C-1042", pairing_id="P-2291",
               from_date="2026-09-15")
    out, report = completeness.complete("Assign C-3310 at INR 18,500.", [r])
    assert isinstance(out, str) and report["options_omitted"] >= 1


# --------------------------------------------------- modelling completeness
def test_out_of_base_assignment_reports_its_positioning():
    d = T.call("check_assignment", crew_id="C-2210", pairing_id="P-2291",
               on_date="2026-09-15").data
    assert d["legal"] is True
    pos = d["positioning"]
    assert pos["deadhead_flight"] == "DX402"
    assert pos["first_departure_delayed_hours"] == 3.0
    assert pos["cost_inr"] == 41200


def test_same_base_assignment_needs_no_positioning():
    d = T.call("check_assignment", crew_id="C-3310", pairing_id="P-2291",
               on_date="2026-09-15").data
    assert d["positioning"] is None


def test_recovery_reports_what_the_crew_can_still_fly():
    d = T.call("recovery_options", on_date="2026-09-16", aircraft="VT-DXA",
               delay_minutes=90).data
    kept = d["legs_original_crew_can_still_operate"]
    assert kept == ["DX401-2026-09-16", "DX402-2026-09-16", "DX403-2026-09-16"]
    rd = d["retained_duty"]
    assert rd["legal"] and rd["fdp_limit"] == 12.5
    assert "assumption" in rd


def test_reserve_eligibility_respects_the_aircraft_rating():
    """An A320-only captain must never be offered for an ATR72 duty."""
    from backend.planner.planner import fastpath, extract_entities
    from backend.executor.executor import execute
    from backend.executor.validator import validate
    q = ("The VT-DXE captain is sick on 16 Sep (called 01:30Z). Which reserve "
         "captains on-call windows cover the callout, and are they qualified?")
    ctx = execute(validate(fastpath(q), extract_entities(q)).plan)
    d = ctx.results["options"].data
    eligible = [o["crew_id"] for o in d["options"]
                if o["crew_id"] and o["cost_inr"] <= 18500]
    assert eligible == ["C-3315"]
    reasons = {e["crew_id"]: e["reason"] for e in d["excluded_examples"]}
    assert any("ATR72" in r for r in reasons.values())


# ---------------------------------------------------------- text hygiene
def test_typographic_punctuation_is_normalised():
    """The model writes non-breaking hyphens and narrow spaces; a cp1252
    console renders those as `A320,â¯ATR72`."""
    from backend.text import plain
    raw = "A320,\u202fATR72 04:00\u201116:00 45\u202fmin C\u20111042"
    assert plain(raw) == "A320, ATR72 04:00-16:00 45 min C-1042"


def test_normalised_text_survives_a_cp1252_console():
    from backend.text import plain
    raw = "\u201cquoted\u201d \u2192 x\u2265y \u2014 45\u202fmin"
    plain(raw).encode("cp1252")          # raises if anything is unrepresentable


def test_normalisation_leaves_identifiers_matchable():
    """A crew id written with U+2011 is not the string 'C-1042', which is how
    three separate matching bugs got in."""
    from backend.text import plain
    assert "C-1042" in plain("crew C\u20111042 is out")


def test_thin_space_between_digits_becomes_a_thousands_separator():
    """"INR 18<U+202F>500" is one number. Mapping that space to ASCII the way
    every other space is mapped turns it into two, and the figure then matches
    neither the tool result nor anything a controller can paste."""
    from backend.text import plain
    assert plain("INR 18 500") == "INR 18,500"
    assert plain("1 000 000") == "1,000,000"
    assert plain("24 000") == "24,000"


def test_thin_space_normalisation_does_not_invent_separators():
    """Only digit-space-three-digits. Everything else stays a space."""
    from backend.text import plain
    assert plain("45 min") == "45 min"
    assert plain("2026 147 flights") == "2026 147 flights"   # ordinary space
    assert plain("9 5") == "9 5"                        # not three digits


def test_normalisation_is_idempotent_and_keeps_markdown():
    from backend.text import plain
    table = "| A | B |\n|---|---|\n| C-1042 | 9.5h |"
    assert plain(table) == table
    assert plain(plain("a \u2013 b")) == plain("a \u2013 b")


# ------------------------------------------------------- relative dates
@pytest.mark.parametrize("question,start,end", [
    ("Who is on reserve at BLR between yesterday and tomorrow?",
     "2026-09-13", "2026-09-15"),
    ("Who is on reserve at BLR yesterday?", "2026-09-13", None),
    ("reserves at BLR from 14 to 16 Sep", "2026-09-14", "2026-09-16"),
    ("Who is on reserve at BLR tomorrow?", "2026-09-15", None),
    ("Who is on reserve at BLR on 2026-09-15?", "2026-09-15", None),
])
def test_relative_and_ranged_dates(question, start, end):
    """A range must stay a range: answering "between yesterday and tomorrow"
    for tomorrow alone looks complete and is two-thirds missing."""
    e = extract_entities(question)
    assert e.get("date") == start
    assert e.get("date_end") == end


def test_a_range_reports_days_the_dataset_does_not_hold():
    d = T.call("list_reserves", on_date="2026-09-13", through_date="2026-09-15",
               base="BLR").data
    assert d["days_covered"] == ["2026-09-14", "2026-09-15"]
    assert d["dates_outside_dataset"] == ["2026-09-13"]
    assert "outside" in (d["note"] or "")


def test_a_single_out_of_window_date_says_so():
    d = T.call("list_reserves", on_date="2026-09-13", base="BLR").data
    assert d["outside_dataset"] and d["count"] == 0 and d["note"]


# --------------------------------------------------- general query layer
@pytest.mark.parametrize("question,field,direction", [
    ("What is the longest block time in the schedule?", "block_hours", "desc"),
    ("What is the shortest block time in the schedule?", "block_hours", "asc"),
    ("Which flight has the most seats?", "seats", "desc"),
    ("Which duty has the least FDP headroom?", "fdp_slack", "asc"),
    ("Who has the highest 28-day flight hours?", "flight_hours_28d", "desc"),
])
def test_superlatives_work_in_both_directions(question, field, direction):
    """A tool per phrase answers "longest" and refuses "shortest". One rule
    over a general query answers both."""
    p = fastpath(question)
    assert p is not None, question
    args = p.steps[-1].args
    assert p.steps[-1].operation == "query_data"
    assert args["order_by"] == field and args["direction"] == direction


def test_query_returns_every_row_tied_at_the_extreme():
    d = T.call("query_data", entity="flights", order_by="block_hours",
               direction="asc", limit=3).data
    assert d["extreme_value"] == 1.0
    assert all(r["block_hours"] == 1.0 for r in d["rows"])
    assert d["returned"] > 3, "a superlative means everything tied with it"


@pytest.mark.parametrize("kw,expected", [
    (dict(entity="duties", aggregate="avg", field="duty_hours",
          group_by="aircraft_type"), {"A320": 10.62, "ATR72": 6.75}),
    (dict(entity="flights", where=[{"field": "date", "op": "eq",
                                    "value": "2026-09-18"}],
          aggregate="sum", field="block_hours"), 36.25),
    (dict(entity="crew", where=[{"field": "status", "op": "in",
                                 "value": ["leave", "training"]}],
          aggregate="count"), 8),
])
def test_query_aggregations(kw, expected):
    assert T.call("query_data", **kw).data["result"] == expected


def test_query_refuses_an_unknown_entity():
    d = T.call("query_data", entity="payroll").data
    assert "error" in d and "unknown entity" in d["error"]


def test_query_refuses_a_field_outside_the_schema():
    """The allowlist is the security boundary: no attribute the schema does not
    name is reachable."""
    d = T.call("query_data", entity="flights", order_by="__class__").data
    assert "error" in d and "no field" in d["error"]


def test_query_refuses_an_unknown_operator():
    d = T.call("query_data", entity="flights",
               where=[{"field": "seats", "op": "eval", "value": 1}]).data
    assert "error" in d and "unknown operator" in d["error"]


def test_query_returns_the_query_it_ran():
    """Auditability: the controller can see exactly what was asked."""
    d = T.call("query_data", entity="flights", order_by="seats",
               direction="asc").data
    assert d["query"]["entity"] == "flights"
    assert d["query"]["direction"] == "asc"


def test_long_result_sets_are_projected_not_dumped():
    """28 complete flight records to say "1.0h, on these flights" is paying
    tokens for detail the answer never uses."""
    d = T.call("query_data", entity="flights", order_by="block_hours",
               direction="asc").data
    assert d["projected"] is True
    assert set(d["rows"][0]) == {"flight_id", "flight_no", "date", "route",
                                 "block_hours"}
    assert "dep_utc" not in d["rows"][0], "detail belongs in the evidence panel"


def test_short_result_sets_keep_every_column():
    d = T.call("query_data", entity="crew", order_by="flight_hours_28d",
               limit=5).data
    assert d["projected"] is False and d["columns"] is None
    assert "seniority" in d["rows"][0]


def test_projection_keeps_the_field_that_was_ordered_on():
    d = T.call("query_data", entity="duties", order_by="fdp_slack",
               direction="asc", limit=20).data
    assert all("fdp_slack" in r for r in d["rows"])


def test_evidence_is_unaffected_by_projection():
    """The full records stay citable even when the prose gets a summary."""
    r = T.call("query_data", entity="flights", order_by="block_hours",
               direction="asc")
    assert len(r.evidence) >= 20


def test_the_whole_api_payload_is_ascii_safe():
    """A client rendering in a cp1252 console must never receive a character it
    cannot show - and the citation labels we build ourselves count too."""
    import json

    from backend.advisor import Advisor
    turn = Advisor().ask.__self__ if False else None      # keep import local
    from backend.executor.executor import execute
    from backend.executor.validator import validate
    from backend.grounding import citations
    from backend.planner.planner import extract_entities, fastpath

    q = "Who is on reserve at BLR on 2026-09-15?"
    ctx = execute(validate(fastpath(q), extract_entities(q)).plan)
    built = citations.build("Reserves include C-3310 on P-2291.",
                            ctx.tool_results, ctx.rules_evaluated)
    blob = json.dumps(built, ensure_ascii=False)
    blob.encode("cp1252")            # raises if anything is unrepresentable


def test_middle_dots_in_our_own_labels_are_normalised():
    from backend.text import plain
    assert plain("limit 60.0 \u00b7 window 09-15") == "limit 60.0 - window 09-15"
    assert plain("soft\u00adhyphen") == "softhyphen"


# ------------------------------------------------ source integrity guard
def test_no_source_file_contains_control_characters():
    """A shell heredoc twice wrote literal backspace bytes (0x08) where a
    regex needed \b. The regex then matched a control character, matched
    nothing, and the failure was invisible in every editor. This is the guard.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for path in list(root.glob("*.py")) + list(root.glob("backend/**/*.py")) \
            + list(root.glob("evals/*.py")) + list(root.glob("frontend/*.py")) \
            + list(root.glob("tests/*.py")):
        text = path.read_text(encoding="utf-8")
        for n, line in enumerate(text.splitlines(), 1):
            bad = {hex(ord(ch)) for ch in line if ord(ch) < 9 or 11 <= ord(ch) < 32}
            if bad:
                offenders.append(f"{path.relative_to(root)}:{n} {sorted(bad)}")
    assert not offenders, offenders


# ---------------------------------------------------- conversational edges
@pytest.mark.parametrize("question,intent", [
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("thanks", "thanks"),
    ("what can you do?", "capability"),
    ("help", "capability"),
    ("what is today date?", "context"),
    ("what time is it?", "context"),
    ("what day is it", "context"),
    ("which week are we looking at?", "context"),
])
def test_conversational_edges_answer_without_a_model(question, intent):
    """"hi" and "what is today's date?" were both refused. Neither is a crew
    question, and neither is a reason to tell a controller you cannot help."""
    from backend import conversation
    handled = conversation.handle(question)
    assert handled is not None, question
    assert handled[1] == intent
    assert len(handled[0]) > 20


@pytest.mark.parametrize("question", [
    "hi, is C-1042 free on 15 Sep?",
    "Who is on reserve at BLR on 2026-09-15?",
    "what is the FDP limit for 4 sectors?",
])
def test_real_questions_are_never_swallowed_by_smalltalk(question):
    from backend import conversation
    assert conversation.handle(question) is None, question


def test_the_snapshot_is_answerable_not_just_usable():
    d = T.call("operating_snapshot").data
    assert d["today"] == "2026-09-14"
    assert d["today_weekday"] == "Monday"
    assert d["schedule_week"]["first"] == "2026-09-14"
    assert d["tomorrow"] == "2026-09-15"


# ------------------------------------------------- enumerated filter values
def test_unknown_filter_value_is_named_not_silently_empty():
    """`status = "on_leave"` when the data says "leave" returned [], and an
    empty list reads as "nobody is on leave". The worst failure is the silent
    one."""
    from backend.tools.query import query_data
    r = query_data(entity="crew", where=[{"field": "status", "op": "eq",
                                          "value": "on_leave"}])
    assert "error" in r.data
    bad = r.data["unrecognised_filter_values"]["status"]
    assert bad["not_in_data"] == ["on_leave"]
    assert "leave" in bad["valid_values"]


def test_partial_match_still_reports_the_bad_value():
    """in ["on_leave", "training"] matches the second value and returns a
    partial answer that looks complete. That is worse than returning nothing."""
    from backend.tools.query import query_data
    r = query_data(entity="crew", where=[{"field": "status", "op": "in",
                                          "value": ["on_leave", "training"]}])
    assert r.data["matched"] > 0
    assert "on_leave" in (r.note or "")
    assert "incomplete" in (r.note or "")


def test_valid_filter_values_produce_no_warning():
    from backend.tools.query import query_data
    r = query_data(entity="crew", where=[{"field": "status", "op": "in",
                                          "value": ["leave", "training"]}])
    assert r.data["matched"] > 0
    assert "unrecognised_filter_values" not in r.data


def test_planner_prompt_carries_the_enumerated_values():
    """The fix that prevents the guess, rather than reporting it afterwards."""
    from backend.executor.registry import query_schema_block
    block = query_schema_block()
    assert "crew.status" in block
    assert "active | leave | training" in block


def test_every_answer_survives_a_cp1252_console_whatever_the_model_emits():
    """The map names the characters gpt-oss is known to emit. This is the
    backstop for the ones it has not emitted yet - a new symbol in a model
    update should degrade to plainer text, never to mojibake."""
    from backend.text import plain
    for raw in ["approx ≈ 3.0 h", "✅ legal", "₹18,500",
                "\U0001f6eb depart", "− 1.33h margin", "℃"]:
        plain(raw).encode("cp1252")          # raises if anything is unprintable


def test_the_backstop_leaves_accented_names_alone():
    from backend.text import plain
    assert plain("José Iyer") == "José Iyer"


# ------------------------------------------------- arithmetic on the pass path
def test_a_legal_assignment_still_reports_its_margins():
    """"Is this legal?" and "how much headroom does that leave?" are the same
    tool result. Collapsing passing rules to a count answered the first and
    forced an abstention on the second."""
    from backend.tools import TOOLS
    payload = TOOLS["check_assignment"](crew_id="C-3310",
                                        pairing_id="P-2291").for_model()
    assert payload["data"]["legal"] is True
    margins = {(m["rule_id"], m["on_date"]): m for m in payload["rules_passed_margins"]}
    duty = margins[("RULE-DUTY-02", "2026-09-15")]
    assert (duty["limit"], duty["used"], duty["headroom_remaining"]) == (60.0, 9.5, 50.5)


def test_headroom_is_reported_before_and_after_the_duty():
    """Both readings are real, and a controller means different ones on
    different days. One figure under an ambiguous name answered the other
    question half the time."""
    from backend.tools import TOOLS
    rows = {r["crew_id"]: r for r in
            TOOLS["pairing_headroom"](pairing_id="P-2291",
                                      on_date="2026-09-15").data["crew"]}
    cap = rows["C-1042"]
    assert cap["duty_hours_7d_before_this_duty"] == 20.93
    assert cap["duty_headroom_7d_before_this_duty"] == 39.07
    assert cap["duty_headroom_7d_after_this_duty"] == 29.57


def test_a_tie_is_reported_as_a_tie_not_as_a_ranking():
    """Twenty-six crew tie at three duty days. Showing ten of them without
    saying so reads as a ranking, and a controller protects an arbitrary ten."""
    from backend.tools import TOOLS
    r = TOOLS["roster_load"]()
    assert r.data["crew_tied_at_busiest"] > len(r.data["crew"])
    assert "tie" in (r.note or "")
