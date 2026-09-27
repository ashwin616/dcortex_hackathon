"""Natural language in, typed execution plan out.

Two routes to a plan:

  1. A deterministic fast path for questions whose shape is unambiguous
     ("who is on reserve at BLR on 15 Sep"). It costs zero tokens and zero
     latency. On a 200,000-token daily budget, not calling the model is the
     single most effective optimisation available.
  2. The LLM planner for everything else. It sees a compact operation
     catalogue - names and one-line summaries, not JSON schemas - which is
     roughly a third of the prompt cost of tool-calling and, because the
     narrator never receives tool definitions at all, structurally cannot
     produce the "model called a tool when tool_choice is none" failure.

The planner never computes and never answers. Its entire output is a plan.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from backend import cache
from backend.client import Usage, complete_json, parse_json_object
from backend.config import SETTINGS
from backend.domain.data import store
from backend.domain.temporal import resolve as resolve_time
from backend.planner.prompts import planner_prompt
from backend.planner.schemas import ExecutionPlan, PlanStep

# A plan is ~150 tokens of JSON, but gpt-oss emits hidden reasoning first and
# a truncated plan is an unrecoverable turn - so the ceiling is generous.
PLAN_MAX_TOKENS = 1400

# --------------------------------------------------------------------------
# entity extraction - shared by both routes
# --------------------------------------------------------------------------
CREW = re.compile(r"\bC[-‐-―−]?(\d{3,4})\b", re.I)
PAIRING = re.compile(r"\bP[-‐-―−]?(\d{3,4})\b", re.I)
TAIL = re.compile(r"\bVT[-‐-―−]?(DX[A-Z])\b", re.I)
FLIGHTNO = re.compile(r"\b(DX\d{3})\b", re.I)
ISO_DATE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
# "15-16 Sep" is a two-day pairing, not the 16th. The range is matched first
# and the start is taken - a cover question is always asked about when it begins.
DAY_RANGE = re.compile(r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s*[-‐-―−]\s*"
                       r"(\d{1,2})\s*(?:st|nd|rd|th)?\s+"
                       r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I)
DAY_MONTH = re.compile(r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s+"
                       r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I)
# "from 14 to 16 Sep", "between the 15th and the 17th" - a range written in
# words rather than with a dash.
WORD_RANGE = re.compile(
    r"\b(?:from|between)\s+(?:the\s+)?(\d{1,2})\s*(?:st|nd|rd|th)?\s*"
    r"(?:(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)?\s*"
    r"(?:to|and|until|through|till|-)\s+(?:the\s+)?(\d{1,2})\s*(?:st|nd|rd|th)?\s*"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?[a-z]*\b", re.I)
# Relative words, in the order they must be tried.
RELATIVE = (("day before yesterday", -2), ("yesterday", -1), ("tonight", 0),
            ("today", 0), ("tomorrow", 1), ("day after tomorrow", 2))
RANGE_JOIN = re.compile(r"\b(?:between|from)\b.*\b(?:and|to|until|through|till)\b", re.I)
TIME_RANGE = re.compile(r"\b([0-2]?\d):?([0-5]\d)\s*(?:z|utc)?\s*(?:-|to|until|–|—)\s*"
                        r"([0-2]?\d):?([0-5]\d)\s*(?:z|utc)?\b", re.I)
MINUTES = re.compile(r"\b(\d{1,3})\s*(?:-|\s)?\s*(?:min|minute)", re.I)
HOURS_DELAY = re.compile(r"\b(\d{1,2}(?:\.\d)?)\s*(?:-|\s)?\s*(?:h|hr|hour)", re.I)
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def extract_entities(q: str) -> dict:
    """Pull identifiers, dates, stations and times out of the question."""
    s = store()
    e: dict = {}
    ql = q.lower()

    crew = [f"C-{m.group(1)}" for m in CREW.finditer(q)]
    if crew:
        e["crew_id"] = crew[0]
        if len(crew) > 1:
            e["crew_ids"] = crew            # two people out at once is a joint problem
    pairings = [f"P-{m.group(1)}" for m in PAIRING.finditer(q)]
    if pairings:
        e["pairing_id"] = pairings[0]
        if len(pairings) > 1:
            e["pairing_ids"] = pairings
    tails = [f"VT-{m.group(1).upper()}" for m in TAIL.finditer(q)]
    if tails:
        e["aircraft"] = tails[0]
        if len(tails) > 1:
            e["aircraft_list"] = tails
    if m := FLIGHTNO.search(q):
        e["flight_no"] = m.group(1).upper()

    # One temporal module, not a regex per phrasing. It resolves days, ranges,
    # relative words and times of day together, and reports anything outside
    # the dataset instead of dropping it.
    when = resolve_time(q, s.snapshot.date(), (s.dates[0], s.dates[-1]))
    if when.start:
        e["date"] = when.start
        if when.end:
            e["date_end"] = when.end
        if when.outside:
            e["dates_outside_dataset"] = when.outside
        if when.phrase:
            e["date_phrase"] = when.phrase
        e["date_explanation"] = when.explain()
    if when.start_time and not e.get("start_utc"):
        e["at_time_utc"] = when.start_time

    stations = [w for w in re.findall(r"\b([A-Z]{3})\b", q) if w in s.stations]
    if stations:
        e["stations"] = stations
        e["station"] = stations[0]

    if m := TIME_RANGE.search(q):
        e["start_utc"] = f"{int(m.group(1)):02d}:{m.group(2)}"
        e["end_utc"] = f"{int(m.group(3)):02d}:{m.group(4)}"
    if m := MINUTES.search(q):
        e["delay_minutes"] = int(m.group(1))
    elif m := HOURS_DELAY.search(q):
        if any(w in ql for w in ("delay", "late", "tech")):
            e["delay_minutes"] = int(float(m.group(1)) * 60)

    for rank, canon in (("senior cabin crew", "Senior Cabin Crew"),
                        ("cabin crew", "Cabin Crew"),
                        ("first officer", "First Officer"),
                        ("captain", "Captain")):
        if rank in ql:
            e["role"] = canon
            break
    return e


# --------------------------------------------------------------------------
# route 1: deterministic fast path
# --------------------------------------------------------------------------
_HAS_CACHE: dict = {}


def _has(q: str, *words) -> bool:
    """Whole-word phrase match.

    Substring matching looks harmless until "rank" fires on "ranked resolution
    options" and routes a Tier-3 recommendation into a crew lookup. Every
    phrase here is matched on word boundaries.
    """
    for w in words:
        pat = _HAS_CACHE.get(w)
        if pat is None:
            pat = _HAS_CACHE[w] = re.compile(rf"(?<!\w){re.escape(w)}(?!\w)")
        if pat.search(q):
            return True
    return False


# A superlative is a direction plus a measure. Enumerating the pairs is a dozen
# lines; enumerating the questions they express is a tool per phrase.
_SUPERLATIVE_DIR = {
    "desc": ("longest", "largest", "highest", "most", "biggest", "greatest",
             "busiest", "maximum", "max"),
    "asc": ("shortest", "smallest", "lowest", "fewest", "least", "minimum",
            "min", "tightest", "closest"),
}
_MEASURES = (
    (("block time", "block hours", "block", "flight time", "sector length"),
     "flights", "block_hours"),
    (("seats", "passengers", "capacity"), "flights", "seats"),
    # Most specific first: "least FDP headroom" contains the word "duty" in
    # "duty" questions too, and the loosest phrase would otherwise win.
    (("fdp headroom", "fdp slack", "fdp", "slack", "headroom"), "duties", "fdp_slack"),
    (("duty hours", "duty length", "duty time", "duty"), "duties", "duty_hours"),
    (("sectors", "legs"), "duties", "sectors"),
    (("flight hours", "28-day", "28 day"), "crew", "flight_hours_28d"),
    (("seniority",), "crew", "seniority"),
    (("reachability", "reach"), "crew", "reachability_minutes"),
    (("risk", "risk score"), "risk", "score"),
)


def _superlative(q: str):
    """(entity, field, direction) when the question asks for an extreme."""
    direction = None
    for d, words in _SUPERLATIVE_DIR.items():
        if _has(q, *words):
            direction = d
            break
    if direction is None:
        return None
    for phrases, entity, field in _MEASURES:
        if _has(q, *phrases):
            return entity, field, direction
    return None


def fastpath(question: str) -> ExecutionPlan | None:
    """A plan for questions whose shape leaves no room for interpretation.

    Conservative by design: anything asking for a judgement, a comparison, or
    anything the patterns do not cover falls through to the LLM planner. A
    wrong fast-path match is far more expensive than a planner call.
    """
    if not SETTINGS.fastpath_enabled:
        return None
    s = store()
    q = cache.normalise_question(question)
    e = extract_entities(question)
    d = e.get("date")

    def plan(intent, op, **args):
        args = {k: v for k, v in args.items() if v is not None}
        return ExecutionPlan(intent=intent, entities=e,
                             steps=[PlanStep(id="s1", operation=op, args=args)])

    # ---- the rulebook itself ------------------------------------------------
    # "What is the duty limit?" is a question about the rules the system is
    # built on. Refusing it while quoting those limits in every other answer
    # is not a defensible place to draw the line.
    # The guard matters as much as the rule. "Does the rostered crew breach any
    # limit if they fly all four legs?" contains the word "limit" and is not a
    # question about the rulebook at all - it is a question about one delayed
    # aircraft on one day. Anything naming a concrete operational subject is a
    # situation, not a citation, so it falls through to the branches below.
    names_a_situation = any(e.get(k) for k in
                            ("crew_id", "crew_ids", "pairing_id", "pairing_ids",
                             "aircraft", "aircraft_list", "flight_no",
                             "delay_minutes", "station", "stations"))
    # And "what is the maximum flight duty period for four sectors?" names no
    # rule and no limit, so it used to be read as a superlative over rostered
    # duties - answering "which duty is longest" to a question about the
    # regulatory ceiling itself.
    asks_the_ceiling = (
        _has(q, "flight duty period", "fdp", "duty period", "minimum rest",
             "rest before", "flight hours in", "duty hours in")
        and _has(q, "maximum", "max", "limit", "allowed", "ceiling",
                 "permitted", "what is the", "what's the")
        and not _has(q, "rostered", "scheduled", "in the schedule", "which",
                     "who", "closest", "nearest", "headroom", "left"))
    if (asks_the_ceiling
            or _has(q, "rule", "rules", "rulebook", "limit", "limits",
                    "legal maximum", "how long can", "how many hours",
                    "allowed")) \
            and not names_a_situation \
            and not _has(q, "left", "remaining", "headroom", "used", "accrued",
                         "breach", "breaches", "delayed", "sick", "closed",
                         "cover", "if they", "if i", "would",
                         # "Who is closest to the 60-hour limit?" quotes a rule
                         # in order to ask about people. It is a crew question.
                         "who is", "who are", "which crew", "closest",
                         "nearest", "approaching", "anybody", "anyone",
                         "bust", "busts", "come within"):
        sectors = None
        if m := re.search(r"(\d)\s*sectors?", q):
            sectors = int(m.group(1))
        elif _has(q, "four", "five", "six"):
            sectors = [4, 5, 6]
        return plan("rules_reference", "rules_reference", sectors=sectors)

    # ---- station closure ---------------------------------------------------
    # Station, date and window are all extracted deterministically, so a
    # planner call here buys nothing but a second of latency and 1,500 tokens.
    if e.get("station") and d and e.get("start_utc") and e.get("end_utc") \
            and _has(q, "closed", "closes", "closure", "shut"):
        return plan("station_closure_impact", "station_closure_impact",
                    station=e["station"], on_date=d,
                    start_utc=e["start_utc"], end_utc=e["end_utc"])

    # ---- sick call, impact only --------------------------------------------
    if e.get("crew_id") and d and _has(q, "sick", "unavailable") \
            and not _has(q, "what should", "options", "cheapest", "resolve",
                         "joint", "legal", "breach"):
        return plan("sick_call_impact", "sick_call_impact", crew_id=e["crew_id"],
                    pairing_id=e.get("pairing_id"), from_date=d)

    # ---- two aircraft out at once ------------------------------------------
    if len(e.get("aircraft_list") or []) > 1 and d \
            and _has(q, "sick", "out", "both", "joint", "simultaneous"):
        steps = [PlanStep(id=f"p{i}", operation="get_pairing",
                          args={"aircraft": tail, "on_date": d})
                 for i, tail in enumerate(e["aircraft_list"][:3])]
        steps.append(PlanStep(id="joint", operation="recommend_joint",
                              args={"on_date": d},
                              depends_on=[st.id for st in steps]))
        return ExecutionPlan(intent="recommend_joint", entities=e, steps=steps)

    # ---- cheapest cover for a role on a named aircraft ---------------------
    if e.get("aircraft") and d and e.get("role") \
            and _has(q, "cheapest", "cover", "sick", "replace") \
            and not e.get("crew_id") \
            and not _has(q, "reserve", "reserves") \
            and not _has(q, "is it legal", "does any rule", "breach"):
        # A named crew member plus a named pairing is a legality question
        # ("is C-5837 legal on P-2291?"), not a search for cover - so this rule
        # stands down whenever the question already names who is proposed.
        return ExecutionPlan(intent="recommend_options", entities=e, steps=[
            PlanStep(id="pairing", operation="get_pairing",
                     args={"aircraft": e["aircraft"], "on_date": d}),
            PlanStep(id="options", operation="recommend_options",
                     args={"role": e["role"]}, depends_on=["pairing"])])

    # ---- draft a callout ----------------------------------------------------
    if e.get("crew_id") and e.get("pairing_id") \
            and _has(q, "notification", "notify", "draft", "message", "callout"):
        return plan("notification_facts", "notification_facts",
                    crew_id=e["crew_id"], pairing_id=e["pairing_id"], from_date=d)

    # ---- cancellation cost --------------------------------------------------
    if e.get("flight_no") and d and _has(q, "cancelled", "cancel", "cancellation"):
        fid = f"{e['flight_no']}-{d}"
        if fid in s.flight_by_id:
            return plan("cancellation_impact", "cancellation_impact", flight_ids=[fid])

    # ---- who is closest to a ceiling ---------------------------------------
    # "Which five crew are closest to the 100-hour/28-day limit?" reads as a
    # superlative, and the superlative branch below would answer it by sorting
    # a column - which returns the crew with the most hours, not the crew with
    # the least room, and drops the headroom the question actually asked for.
    # crew_near_limits computes the margin against the rule.
    if _has(q, "closest to", "nearest to", "approaching", "close to the",
            "running out of", "least room", "least headroom", "most at risk of",
            "come within", "bust", "busts") \
            and _has(q, "limit", "ceiling", "100-hour", "100 hour", "60-hour",
                     "60 hour", "28-day", "28 day", "7-day", "7 day",
                     "duty", "flight hours"):
        # "100 hours / 28 days" is RULE-FLT-03; anything else is the duty clock.
        by_flight = _has(q, "100-hour", "100 hour", "28-day", "28 day",
                         "flight hours", "flight-hour", "block hours")
        return plan("crew_near_limits", "crew_near_limits",
                    on_date=d or s.dates[0], base=e.get("station"),
                    rank_by="flight" if by_flight else "duty",
                    duty_threshold_7d=0.0 if by_flight else 45.0)

    # ---- who is carrying the week ------------------------------------------
    # Same trap: "who is working the most duty days" is a superlative over a
    # figure nothing in the schema stores - it has to be counted per crew, and
    # the question is really about where concentration risk sits.
    if _has(q, "duty days", "most days", "working the most", "busiest",
            "concentration", "carrying the most", "worked the most"):
        return plan("roster_load", "roster_load", base=e.get("station"))

    # ---- crew at or above a duty threshold ---------------------------------
    if d and _has(q, "duty hours", "duty hour") \
            and _has(q, "or more", "at least", "above", "over", "approaching"):
        m = re.search(r"(\d{2}(?:\.\d)?)\s*(?:or more)?\s*(?:duty )?hours", q)
        return plan("crew_near_limits", "crew_near_limits", on_date=d,
                    duty_threshold_7d=float(m.group(1)) if m else 45.0,
                    base=e.get("station"))

    # ---- roster count by rank and base -------------------------------------
    if e.get("role") and e.get("station") and not e.get("crew_id") \
            and _has(q, "how many", "who are they", "based at"):
        return plan("find_crew", "find_crew", rank=e["role"], base=e["station"])

    # "Which reserves could take the VT-DXE duty" is an eligibility question,
    # not a roster listing: it needs the aircraft rating and the on-call window
    # checked against the actual report time. list_reserves does neither, and
    # answering from it named an A320-only captain for an ATR72 duty - a wrong
    # answer, which is worse than an incomplete one.
    if e.get("aircraft") and d and _has(q, "reserve", "reserves") \
            and _has(q, "cover", "sick", "qualified", "eligible", "could take",
                     "callout"):
        return ExecutionPlan(
            intent="reserve_eligibility", entities=e,
            steps=[PlanStep(id="pairing", operation="get_pairing",
                            args={"aircraft": e["aircraft"], "on_date": d}),
                   PlanStep(id="options", operation="recommend_options",
                            args={"role": e.get("role") or "Captain"},
                            depends_on=["pairing"])])

    # reserves on call
    if _has(q, "on reserve", "reserves", "reserve pool", "standby") and d \
            and not _has(q, "cover", "should i", "recommend", "cheapest", "legal"):
        return plan("list_reserves", "list_reserves", on_date=d,
                    through_date=e.get("date_end"),
                    base=e.get("station"), rank=e.get("role"))

    # duty clock / headroom for a named crew member
    if e.get("crew_id") and _has(q, "duty hour", "duty hours", "hours left", "headroom",
                                 "flight hours", "how many hours", "accrued") \
            and not _has(q, "if", "cover", "assign"):
        return plan("duty_clock", "duty_clock", crew_id=e["crew_id"], on_date=d)

    # certifications expiring
    if _has(q, "expiring", "expires", "expiry", "expired", "expire", "lapsed",
            "licence", "license", "certification", "certifications", "medical",
            "recurrent") \
            and not e.get("pairing_id") and not _has(q, "resolve", "should i", "cover"):
        m = re.search(r"(\d{1,3})\s*days", q)
        return plan("certifications_expiring", "certifications_expiring",
                    as_of=d or s.snapshot.date().isoformat(),
                    within_days=int(m.group(1)) if m else 30,
                    crew_id=e.get("crew_id"))

    # Two or more people out at once is a different problem, not two copies of
    # the same one: solved per opening, the cheapest reserve gets assigned to
    # both aircraft simultaneously.
    if len(e.get("crew_ids") or []) > 1 and _has(
            q, "sick", "out", "unavailable", "both", "simultaneous", "at once"):
        return ExecutionPlan(
            intent="recommend_joint", entities=e,
            steps=[PlanStep(id="s1", operation="recommend_joint",
                            args={"crew_ids": e["crew_ids"], "on_date": d}
                            | ({"pairing_ids": e["pairing_ids"]}
                               if e.get("pairing_ids") else {}))])

    # A delay named against an aircraft and a date. The planner abstains here
    # because no crew or pairing is named - but delay_impact resolves the
    # pairing from the tail itself, so the question is perfectly answerable.
    if e.get("delay_minutes") and (e.get("aircraft") or e.get("pairing_id")) and d             and _has(q, "delay", "delayed", "late"):
        # "does it breach" wants the FDP verdict; "what should we do" wants the
        # priced recovery. Routing both to delay_impact answers the first
        # question when the second was asked.
        wants_action = _has(q, "what should", "what do", "recovery", "recover",
                            "resolve", "options", "plan", "do about", "action")
        op = "recovery_options" if wants_action else "delay_impact"
        return plan(op, op, on_date=d, delay_minutes=e["delay_minutes"],
                    aircraft=e.get("aircraft"), pairing_id=e.get("pairing_id"))

    # "Resolve their 19 Sep assignment" - crew and date, no pairing named.
    # recommend_options finds the pairing operating on that date itself.
    if e.get("crew_id") and d and _has(q, "resolve", "cover their", "replace",
                                       "stand down", "cheapest legal way",
                                       "what should i do", "options"):
        return plan("recommend_options", "recommend_options",
                    crew_id=e["crew_id"], from_date=d, role=e.get("role"))

    # Legality of a named person on a named duty. Deterministic because the
    # question shape is rigid - "can X operate/cover Y", "does X breach on Y" -
    # and because the LLM planner abstains on these more often than it should.
    legality = _has(q, "legally", "legal", "is it legal", "does any rule",
                    "does anyone breach", "breach a", "breaches", "breach",
                    "can they cover", "cover the full", "cover the whole")
    # "If C-3310 takes both days of P-2291, how much headroom is he left with?"
    # is the same simulation with the verdict phrased as a number. Routed to
    # duty_clock it read the CURRENT clock and abstained, because the current
    # clock cannot answer a question about a roster that does not exist yet.
    if e.get("crew_id") and e.get("pairing_id") \
            and _has(q, "if", "were to", "took", "takes", "picks up",
                     "picked up", "operates", "operated", "flies", "flew") \
            and _has(q, "headroom", "left with", "room left", "remaining",
                     "how much", "left"):
        return plan("check_assignment", "check_assignment",
                    crew_id=e["crew_id"], pairing_id=e["pairing_id"],
                    on_date=d if _has(q, "from") else None)
    if e.get("crew_id") and legality:
        if e.get("pairing_id"):
            return plan("check_assignment", "check_assignment",
                        crew_id=e["crew_id"], pairing_id=e["pairing_id"],
                        on_date=d if _has(q, "from") else None)
        if e.get("aircraft"):
            # The pairing is not named, so resolve it from tail + date first and
            # let the executor carry pairing_id into the legality check.
            return ExecutionPlan(
                intent="check_assignment", entities=e,
                steps=[PlanStep(id="pairing", operation="get_pairing",
                                args={"aircraft": e["aircraft"], "on_date": d}),
                       PlanStep(id="legal", operation="check_assignment",
                                args={"crew_id": e["crew_id"]},
                                depends_on=["pairing"])])

    # Superlatives, in whichever direction they were asked. A rule per phrase
    # ("longest block") answers exactly one question and refuses its mirror
    # image; one rule over a general query answers both, and everything between.
    if (sup := _superlative(q)) is not None:
        entity, order_field, direction = sup
        return plan("query", "query_data", entity=entity, order_by=order_field,
                    direction=direction, limit=5)

    if _has(q, "nonstop", "non-stop", "destinations", "stations does", "network serve"):
        return plan("schedule_stats", "schedule_stats", metric="destinations",
                    station=e.get("station") or "BLR")
    if _has(q, "most seats", "largest leg", "seats at risk"):
        return plan("schedule_stats", "schedule_stats", metric="largest_leg")

    # flight schedule lookups
    if _has(q, "which flights", "what flights", "flights depart", "flights operate",
            "how many flights", "flights fly", "departures", "arrivals") \
            and not _has(q, "uncrewed", "affected", "closed", "cancel", "sick"):
        st = e.get("stations") or []
        dep = arr = None
        if len(st) >= 2:
            dep, arr = st[0], st[1]
        elif st:
            dep = st[0] if _has(q, "depart", "from", "out of") else None
            arr = st[0] if (dep is None and _has(q, "arrive", "into", "to")) else None
            dep = dep or (st[0] if arr is None else None)
        if _has(q, "how many"):
            return plan("schedule_stats", "schedule_stats", metric="daily_counts", on_date=d)
        return plan("list_flights", "list_flights", on_date=d, dep_station=dep,
                    arr_station=arr, flight_no=e.get("flight_no"), aircraft=e.get("aircraft"))

    # which aircraft / seats on a specific flight
    if e.get("flight_no") and _has(q, "which aircraft", "what aircraft", "seats",
                                   "aircraft operates"):
        return plan("list_flights", "list_flights", on_date=d, flight_no=e["flight_no"])

    # pairing roster, by pairing id or by tail+date. Naming the tail matters:
    # get_pairing on a bare date returns every pairing that day, and the slim
    # view handed to the narrator caps long lists - so the answer can go
    # missing simply because the query was too wide.
    crewish = _has(q, "who is assigned", "which crew", "crew are assigned", "roles",
                   "complement", "who is the", "senior cabin crew", "captain on",
                   "first officer on", "who is on", "who are on", "who is flying",
                   "who operates", "who is working", "report and release",
                   "report time", "release time")
    if e.get("pairing_id") and crewish and not _has(q, "if", "cover", "legal", "breach"):
        return plan("get_pairing", "get_pairing", pairing_id=e["pairing_id"])
    if e.get("aircraft") and crewish and not _has(q, "if", "cover", "legal", "breach",
                                                  "sick", "delay"):
        return plan("get_pairing", "get_pairing", aircraft=e["aircraft"], on_date=d)

    # risk signal - checked before the generic crew lookup, which also matches
    # "what is ... for C-1042" and would otherwise swallow it
    if _has(q, "risk score", "disruption risk", "risk signal") and e.get("crew_id"):
        return plan("risk_signal", "risk_signal", crew_id=e["crew_id"])

    # single crew record
    if e.get("crew_id") and _has(q, "base and rating", "what is", "who is", "rank",
                                 "reachability", "on-call window", "oncall") \
            and not _has(q, "sick", "cover", "legal", "breach", "should i"):
        return plan("find_crew", "find_crew", crew_id=e["crew_id"])

    # rest rule arithmetic
    if _has(q, "earliest", "may report", "can report", "next report",
            "report next", "when may", "when can") \
            and _has(q, "rest", "released", "release"):
        if m := re.search(r"(20\d{2}-\d{2}-\d{2}t[0-2]\d:[0-5]\d(?::[0-5]\d)?z?)", q):
            raw = m.group(1).upper()
            ts = raw if len(raw) > 17 else raw.rstrip("Z") + ":00Z"
            return plan("earliest_next_report", "earliest_next_report", release_utc=ts)
        # the same fact stated in two halves: "released at 15:30Z on 16 Sep"
        if (tm := re.search(r"\b([0-2]?\d):([0-5]\d)\s*z?\b", q)) and d:
            ts = f"{d}T{int(tm.group(1)):02d}:{tm.group(2)}:00Z"
            return plan("earliest_next_report", "earliest_next_report", release_utc=ts)

    # network shape
    if _has(q, "nonstop", "non-stop", "destinations", "stations does", "network serve"):
        return plan("schedule_stats", "schedule_stats", metric="destinations",
                    station=e.get("station") or "BLR")
    return None


# --------------------------------------------------------------------------
# route 2: LLM planner
# --------------------------------------------------------------------------


def plan_question(question: str, history: list | None = None,
                  usage: Usage | None = None, allow_fastpath: bool = True) -> tuple:
    """Return (plan, source) where source is 'fastpath', 'cache' or 'llm'."""
    if allow_fastpath and not history:
        fp = fastpath(question)
        if fp is not None:
            return fp, "fastpath"

    key = cache.plan_key(question)
    if not history:
        hit = cache.PLAN_CACHE.get(key)
        if hit is not None:
            return ExecutionPlan.model_validate(hit), "cache"

    messages = [{"role": "system", "content": planner_prompt()}]
    if history:
        messages.append({"role": "user", "content":
                         "Earlier in this conversation:\n" + "\n".join(history[-4:])})
    messages.append({"role": "user", "content": question})

    raw = complete_json(messages, role="planner", max_tokens=PLAN_MAX_TOKENS, usage=usage)
    plan = ExecutionPlan.model_validate(raw)
    if not history:
        cache.PLAN_CACHE.put(key, plan.model_dump())
    return plan, "llm"


def replan(question: str, feedback: str, usage: Usage | None = None) -> ExecutionPlan:
    """One corrective attempt after the validator rejected a plan."""
    messages = [
        {"role": "system", "content": planner_prompt()},
        {"role": "user", "content": question},
        {"role": "user", "content": feedback},
    ]
    raw = complete_json(messages, role="planner", max_tokens=PLAN_MAX_TOKENS, usage=usage)
    return ExecutionPlan.model_validate(raw)


def fallback_plan(question: str) -> ExecutionPlan | None:
    """A last-resort plan for a question that names something real.

    The planner sometimes declines a perfectly answerable question - "give me
    the crew list for P-2291" was refused on the grounds that no operation
    returns a crew list, which get_pairing plainly does. Refusing a question
    about an entity that exists is the one abstention that is never correct, so
    when identifiers were extracted we answer the obvious thing about them.

    Not a guess at intent: it returns the broadest safe lookup for whatever the
    question named, and nothing at all when it named nothing.
    """
    e = extract_entities(question)
    d = e.get("date")
    s = store()

    if e.get("pairing_id") and e["pairing_id"] in s.pairing_by_id:
        return ExecutionPlan(intent="lookup_pairing", entities=e, steps=[
            PlanStep(id="s1", operation="get_pairing",
                     args={"pairing_id": e["pairing_id"]})])
    if e.get("crew_id") and e["crew_id"] in s.crew:
        return ExecutionPlan(intent="lookup_crew", entities=e, steps=[
            PlanStep(id="s1", operation="find_crew",
                     args={"crew_id": e["crew_id"]})])
    if e.get("aircraft") and d:
        return ExecutionPlan(intent="lookup_pairing", entities=e, steps=[
            PlanStep(id="s1", operation="get_pairing",
                     args={"aircraft": e["aircraft"], "on_date": d})])
    if e.get("flight_no") and d:
        return ExecutionPlan(intent="lookup_flight", entities=e, steps=[
            PlanStep(id="s1", operation="list_flights",
                     args={"flight_no": e["flight_no"], "on_date": d})])
    return None
