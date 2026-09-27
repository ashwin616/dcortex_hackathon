# Crew Ops Advisor

A conversational advisor for the dCortex Air Crew Control desk. Ask in plain
English; get an answer that is **computed, not guessed**, with the rule
arithmetic attached and every figure traceable to a record.

```
> Captain C-1042 is out for pairing P-2291 on 15-16 Sep. What should I do?

Assign Captain C-3310 (reserve callout) at INR 18,500.

This is the cheapest legal option. C-3310 is BLR-based, A320 rated and on call
06:00-18:00Z, can report at 06:00Z, is reachable in 45 minutes, and carries a
low disruption risk (0.17).

| Rank | Captain | Action           | Cost (INR) | Reachable | Risk |
|------|---------|------------------|-----------:|-----------|------|
| 1    | C-3310  | Reserve callout  |     18,500 | 45 min    | 0.17 |
| 2    | C-5566  | Day-off callout  |     24,000 | 60 min    | 0.08 |
| 3    | C-3983  | Day-off callout  |     24,000 | 45 min    | 0.10 |
| 4    | C-1526  | Day-off callout  |     24,000 | 90 min    | 0.12 |

Why not the others? Of 150 considered, 19 were excluded:
  C-1017  only 11.0h rest before P-2217 on 16 Sep, and double-booked
  C-1600  no A320 rating
  C-2087  would exceed the 60h/7d limit by 1h20m (61.33h vs 60h) on 15 Sep

[🧠 recommend_options | 21 rule checks, all pass | 47 records | kernel 6.2ms | 4.2s | 3,683 tokens]
```

---

## Setup

```bash
pip install -r requirements.txt
echo "GROQ_API_KEY=gsk_..." > .env

python -m pytest tests -q              # 194 unit tests, no network
python eval.py kernel                  # deterministic layer vs the answer keys
python cli.py --kernel                 # kernel demo, no API key needed
python cli.py                          # terminal chat
python eval.py llm                     # end-to-end scoring, 38 questions
```

The web console needs the API up:

```bash
uvicorn backend.main:app --reload         # http://127.0.0.1:8000/docs
streamlit run frontend/streamlit_app.py   # controller console
```

The SQLite mirror builds itself on first use (~600 ms) into `data/crewops.db`
and is reused afterwards in about 5 ms. It rebuilds on its own whenever a file
in `data/` changes, so there is nothing to remember and nothing to invalidate.
To build or inspect it directly:

```bash
python -m backend.domain.db            # build/refresh, then print the tables
```

---

## The one decision everything follows from

> **The LLM decides what needs to be done. Deterministic code decides what is true.**

Legality is exact arithmetic against a seven-rule book. A model that
approximates a duty-hour sum produces answers that are fluent, confident and
wrong — operationally worse than no answer. So the model is confined to the two
ends of the pipeline and everything between them is deterministic Python.

```
question → PLANNER → VALIDATOR ═╪═ EXECUTOR → TOOLS → KERNEL → NARRATOR → GROUNDING → answer
           (LLM)     (code)     ╪   (code)    (code)   (code)   (LLM)      (code)
                                ╪
                 LLM / DETERMINISTIC BOUNDARY
```

| LLM does | Deterministic code does |
|---|---|
| Turn a sentence into a typed `ExecutionPlan` | Decide whether that plan may run |
| Resolve fuzzy references ("tomorrow", "the VT-DXA captain") | Every hour, rupee, date window and rule verdict |
| Render a rule trace into controller English | Candidate enumeration and ranking |
| Say "I can't answer that reliably" | Legality — pass/fail **with the arithmetic attached** |

Model output is untrusted input to the validator. Nothing downstream can be
influenced by the model except through a plan that passed it.

Full diagram, stage by stage: **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Five things worth looking at

### 1. `RuleResult` carries arithmetic, not a boolean

```json
{ "rule_id": "RULE-DUTY-02", "passed": false, "limit": 60.0, "actual": 61.33,
  "margin": -1.33, "window": "2026-09-09..2026-09-15",
  "detail": "61.33h duty in 7 days ending 2026-09-15 vs 60h limit: exceeds by 1h20m" }
```

The explanation a controller reads is *derived* from those numbers. The narrator
renders `detail`; it cannot drift from it because there is nothing left to
compute. A rule that passes still reports its headroom.

### 2. A plan is an artifact, so it can be refused

The planner emits JSON; the validator vets it before anything runs — operation
allowlist, required arguments, "at least one of" arguments, identifiers that
exist, dates inside the schedule week, an acyclic graph, a step cap. A plan
naming `C-9999` or calling `run_sql` never reaches the executor.

It also repairs rather than rejects where repair is safe: `date` → `on_date`,
`"captain"` → `"Captain"`, and backfilling `crew_id` from entities extracted
deterministically from the question. Those values come from our own regex over
the controller's words, never from the model, so backfill cannot introduce a
fact nobody stated.

### 3. The world overlay makes what-ifs real

The base snapshot is immutable; a disruption produces a copy-on-write `World`
and the kernel runs against that. "If C-2087 covers P-2291, does anything
break?" is a simulation over modified state, not a request for the model to
imagine consequences. Worlds do not chain.

### 4. Grounding, then abstention

Every `C-####`, `P-####`, `DX###`, `VT-DX#`, `RULE-*` and every consequential
number is pulled back out of the answer and checked against what the tools
returned. One correction round, then the answer is withheld.

Deliberately asymmetric — identifiers strictly, numbers only when large (money)
or precise (hours). Checking "3 legs" against the corpus produces noise, not
safety.

It earns its place. In testing it caught the narrator writing `INR 5,500` — the
*difference* between two real costs. Both inputs were real; the derived figure
was the model doing arithmetic. The narrator is now explicitly forbidden from
stating differences, sums or percentages.

### 5. Risk as a tiebreaker, not a ranker

`risk_signals.json` is a provided input, not something we predict. It orders
equally priced legal options and is always stated in the reasoning. It never
overrides cost or legality.

---

## Getting past 8,000 tokens per minute

The free tier's limits are **per model**, not per account:

| Limit | Per model | Across the 4 tool-capable models |
|---|---|---|
| Tokens / minute | 8,000 | **32,000** |
| Tokens / day | 200,000 | **800,000** |
| Requests / day | 1,000 | **4,000** |

A single-model client leaves three quarters of that unused. Five mechanisms, in
order of effect:

1. **Pool across models.** `ratelimit.ModelPool` predicts a request's cost and
   picks a model that can pay for it *now*. A 429 is a bug in our accounting,
   not a normal event. Planner and narrator default to different models, so a
   turn usually spans two independent buckets.
2. **Deterministic fast path.** Unambiguous lookups are planned by regex — 0
   tokens, ~0.05 ms, no model involved. Most Tier-1 questions never reach one.
3. **Trust the headers.** Every response carries `x-ratelimit-remaining-tokens`
   and a reset interval; local estimates drift, these do not, so they overwrite
   our accounting on each call.
4. **Cache.** Plans key on the normalised question. Answers key on the question
   *and* the evidence they cite, so a cached answer cannot outlive its data. A
   repeated question costs 0 tokens and ~0 ms.
5. **`reasoning_effort: low` for planning.** gpt-oss spends completion tokens on
   hidden reasoning before emitting anything. Planning is translation, not
   reasoning — untuned, it silently truncated plans at the token cap.

Failures are handled by class, never blanket retry: **429** marks that bucket
and moves to another; **5xx** retries the same model with jittered backoff;
**4xx** is our bug and is surfaced rather than spending the daily request budget
on it. Per-minute throttling is waited out; daily exhaustion is not.

Both CLI and UI report compute latency and rate-limit queueing **separately**,
because conflating them misrepresents the system: the architecture meets the
"usable on a live shift" bar; the free API tier occasionally does not.

---

## Verification

| | |
|---|---|
| `python -m pytest tests -q` | **70/70** — rule arithmetic, boundary conditions, validator refusals, executor ordering, grounding, rate limiter |
| `python eval.py kernel` | **70/70** against the shipped answer keys: all 6 scenarios plus every machine-comparable question |
| `python eval.py llm` | end-to-end fact recall through the full chat stack |

The kernel reproduces every engineered case the problem statement is built on:
C-2087 breaching DUTY-02 by exactly 1h20m, C-3310 clean at ₹18,500, C-2210 legal
via deadhead at ₹41,200 with a 3h delay, C-3305 legal on day 1 and breaching on
day 2, C-2091 excluded by QUAL-05, the S3 closure's per-flight delay/FDP table,
and S4's 12.75h FDP against a 12.0h limit.

Independent check of the window semantics: recomputing `duty_hours_7d` and
`flight_hours_28d` from `daily_history` plus the rostered week reproduces the
published clock for **150/150 crew**.

Every turn appends one JSON line to `logs/audit.jsonl` — plan, operations,
records touched, rules evaluated, tokens spent.

---

## Trade-offs

**Plan-and-execute rather than tool-calling.** The first build used native
tool-calling and was replaced. A plan is auditable where a call sequence is not;
the validator gets somewhere to check *meaning*, not just shape; two LLM calls
replace three, without ~1,900 tokens of JSON schema on each. And because the
narrator receives no tool definitions, it cannot emit a tool call — which
removed the single largest error class in the tool-calling evaluation.

**Exhaustive enumeration, no solver.** 150 crew, so every candidate is checked
against all seven rules — about 6 ms. A search heuristic would be faster and
harder to explain, and the brief explicitly does not ask for an optimiser.

**Rejected candidates are kept.** "Why not C-3305?" is what controllers ask
next, and the answer is the most persuasive evidence that the system checked
properly.

**Tools are coarse, not primitive.** `recommend_options` does a lot in one call
rather than making the model orchestrate ten lookups: fewer round trips, fewer
chances to reason wrongly about intermediate state, and a boundary that matches
how a controller thinks about the problem.

**Carry-forward is declared per operation.** In `recommend_options`, `crew_id`
is the person being replaced; in `check_assignment` it is the candidate being
tested. A global carry-forward table would silently answer a different question
than the one asked, so operations that would misread an inherited value simply
do not accept it.

**Two representations of one dataset, built from one loader.** Objects for the
rules kernel, SQLite for open-ended questions. The kernel walks Python objects
because legality is graph work — a pairing, its legs, the crew on them, their
clocks — and that is clumsy in SQL. But a controller also asks *"which flight
has the shortest block time?"*, and a tool per question shape does not scale;
`ask_sql` answers the long tail without a new tool.

The obvious objection is two sources of truth. Three things stop that. The
database is **built from the same loader the kernel uses**, never parsed
separately. It is **read-only** (`PRAGMA query_only`), so nothing can drift into
it. And it is **fingerprinted against the source JSON**, so an edit to the data
rebuilds it rather than serving a stale answer. Legality never touches SQL:
`ask_sql` returns `NOT_A_QUERY` for "would this breach?", because that is a
simulation over a modified roster, not a row that exists to be selected.

---

## Known limitations

- **Deadhead positioning is same-day and single-leg.** With no direct flight
  from the crew's base that day, the candidate is excluded rather than routed
  via a connection. Correct for this network (only DEL→BLR positioning arises),
  but it would not generalise.
- **Multi-role failures are solved role-by-role.** For two simultaneous sick
  calls the system ranks options per opening; it does not jointly allocate one
  scarce reserve across both. It surfaces the contention and leaves the split to
  the controller. A real desk would want the joint plan.
- **Cascading disruption is one level deep.** A cover is checked against the
  covering crew's own downstream duties, but the system does not then re-plan
  the pairings that cover displaces.
- **The grounding validator checks tokens, not claims.** "C-3310 is the most
  expensive option" would pass — the identifier and figures are real, the
  relation is wrong. Catching that needs a claim-level check we did not build.
- **Nothing audits what the tool chose not to compute.** Grounding proves every
  stated number is real; completeness proves every named set is whole. Neither
  can ask whether the *right* quantity was produced. `pairing_headroom` reported
  headroom under one name when the question had two readings, and every check
  passed on an answer that was correct for the other one. `SAMPLES.md` works
  through that case in full.
- **Tier 2 sits at 0.63 mean fact recall on the controller-voice question set**,
  against 0.94 on the supplied one. The gap is narration, not arithmetic: the
  tool computes the right figures and the answer renders a subset of the
  columns. The structural fix — render tables deterministically from the tool
  result and leave the model only the prose around them — is designed, not
  built.
- **The fast path is regex.** It is deliberately conservative and falls through
  to the LLM planner when unsure, but a mis-fire is a wrong tool with a
  confident answer. Its coverage is asserted in `tests/test_pipeline.py`, which
  is the only thing keeping it honest.
- **Hotel and downstream repositioning costs are not modelled** beyond the rate
  in `costs.json`.
- **The dataset is clean.** Tools return `{"error": ...}` rather than raising,
  so the chat degrades to an abstention instead of a stack trace, but that is
  the extent of malformed-input handling.

---

## Crew PII in production

The dataset is synthetic; a real one has the same shape — names, rosters,
medical certificate expiry, fatigue reports. Three things would change.

**The model would never see identity.** Tools would return opaque crew keys and
the identity join would happen in the UI *after* the answer is composed, so
prompts and provider logs carry `C-1042`, never a person.

**Medical data would reduce to a boolean at the tool boundary** —
`cert_valid_on(date)`, never a diagnosis or a fitness category.

**Every operation would be logged with the controller's id and the records it
touched.** `telemetry.audit()` already writes exactly that as a by-product of
the evidence list. That log is what a regulator asks for after a bad decision,
and what makes a wrong recommendation reviewable after the fact.

---

## Scale

Reasoning is per-disruption and local — one pairing, one role, the affected
crew's own windows. Candidate enumeration is O(crew of that rank), so a
5,000-crew carrier is a rank-and-base-indexed query away, not an algorithm
change. What would actually strain are the parts deliberately kept small: the
world overlay is copy-on-write over an in-memory snapshot, which at real scale
becomes a transaction against the roster store, and duty-clock windows would
move from recomputation to a maintained running total. The LLM layer is
unaffected by carrier size — it sees a plan and a result, never the fleet.

---

## Repository

```
backend/
  main.py              orchestration: plan -> validate -> execute -> narrate -> ground
  config.py            env-driven settings, model specs
  client.py            pooled Groq HTTP
  ratelimit.py         per-model token budgets, ModelPool
  cache.py             plan cache + evidence-bound answer cache
  narrator.py          structured result -> prose
  telemetry.py         structured logging, audit trail
  planner/
    planner.py         deterministic fast path + LLM planner
    prompts.py         both system prompts
    schemas.py         ExecutionPlan, PlanStep (pydantic)
    sqlagent.py        natural language -> one SELECT, with self-repair
  executor/
    validator.py       the security boundary
    executor.py        deterministic workflow engine
    registry.py        operation catalogue - one source for prompt and validator
  conversation.py      greetings, "what can you do", "what is today's date"
  text.py              unicode normalisation at the last mile
  tools/
    lookup.py          Tier 1 - lookups
    query.py           Tier 1 - typed query DSL (the long tail, no new tools)
    sql.py             Tier 1 - SQL agent, four guards, read-only
    impact.py          Tier 2 - consequence and simulation
    recommendation.py  Tier 3 - ranked options, morning brief
    notification.py    Tier 3 - verified callout facts
    common.py          shared evidence helpers
  domain/              the kernel, zero LLM
    data.py world.py duty.py rules.py costing.py candidates.py models.py
    db.py              SQLite mirror, fingerprinted, built once
    temporal.py        "yesterday", "this weekend", "next 3 days" -> dates
  grounding/
    validator.py       post-hoc fact check
    completeness.py    the opposite failure: an answer that omits evidence
    citations.py       computed from evidence, never model-generated
frontend/
  streamlit_app.py     controller console
tests/                 194 unit tests, no network
evals/                 kernel.py (answer keys) · llm.py (fact recall) · custom.py (unseen)
cli.py                 terminal console; --kernel runs without an API key
eval.py                entry point: kernel | llm | all
data/                  the provided starter dataset, unmodified (+ derived crewops.db)
DATASET.md             the starter pack's own README
ARCHITECTURE.md        the full diagram and the boundary argument
SAMPLES.md             worked inputs and outputs, including a failure case
PRESENTATION.md        the deck
samples/               recorded eval runs
logs/audit.jsonl       one line per answered question
```

The dataset stays at the repository root rather than under `backend/` because
the supplied `validate.py` and `generate.py` expect it there, and the brief asks
for the provided data unmodified. `backend/domain/data.py` looks for it in both
places, so the package still works if it is ever vendored.
