# Crew Ops Advisor — Architecture

> **The LLM decides what needs to be done. Deterministic code decides what is true.**

Legality here is exact arithmetic against a seven-rule book. A model that
approximates a duty-hour sum produces answers that are fluent, confident and
wrong — which in operations is worse than no answer. So the model is confined
to two jobs at the two ends of the pipeline, and everything between them is
deterministic Python.

**LLM = Planner + Narrator. Validator + Executor + Domain Kernel = source of truth.**

---

## 1. System

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CREW CONTROLLER                                                         │
│  "Captain C-1042 is out for P-2291 on 15-16 Sep. What should I do?"      │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PRESENTATION   Streamlit (frontend/) · CLI (cli.py) · FastAPI + SSE     │
│  chat │ plan │ rules checked │ citations │ live trace │ token pool       │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  CONVERSATION GATE                                backend/conversation.py│
│  "hi" · "what can you do" · "what is today's date" — 0 tokens, from the  │
│  live registry. Anything naming a crew/flight/date falls straight through.│
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  TEMPORAL RESOLVER                                domain/temporal.py     │
│  yesterday · this weekend · next 3 days · "this afternoon" → real dates  │
│  reports days outside the dataset instead of silently clipping them      │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PLANNER                                          planner/planner.py     │
│    route 1  deterministic fast path — 0 tokens, ~0.05 ms                 │
│    route 2  LLM planner — compact operation catalogue, not JSON schemas  │
│    route 3  SQL agent — NL → one SELECT, self-repairs once (sqlagent.py) │
│  Output: ExecutionPlan (pydantic)                 planner/schemas.py     │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PLAN VALIDATOR                                   executor/validator.py  │
│    operation allowlist · required + "one of" args · identifiers exist    │
│    dates inside the schedule week · acyclic graph · step cap             │
│    repair synonyms · backfill from extracted entities                    │
│    reject → one repair round → abstain                                   │
└────────────────────────────────┬─────────────────────────────────────────┘
     ═══════════════════ LLM / DETERMINISTIC BOUNDARY ════════════════════
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  EXECUTOR                                         executor/executor.py   │
│  topological order · per-operation carry-forward · evidence collected    │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  TOOL LAYER  33 operations                        backend/tools/         │
│  T1 lookup (18) · T2 consequence (10) · T3 recommendation (5)            │
│  every one returns {data, evidence, rules_evaluated}                     │
│                                                                          │
│  Two of them absorb the long tail instead of growing the catalogue:      │
│    query_data  typed DSL — filter/sort/aggregate, allowlisted fields     │
│    ask_sql     generated SELECT, guarded (see below)                     │
└───────────────┬──────────────────────────────────────┬───────────────────┘
                ▼                                      ▼
┌──────────────────────────────────────┐ ┌─────────────────────────────────┐
│  DOMAIN KERNEL — zero LLM            │ │  SQL SURFACE     tools/sql.py   │
│   world.py   snapshot + copy-on-     │ │  read-only connection           │
│              write disruption overlay│ │  one statement, SELECT/WITH only│
│   duty.py    report -60 · release +30│ │  LIMIT imposed · 2 s timeout    │
│   rules.py   7 predicates →          │ │  the SQL that ran is returned   │
│              RuleResult{limit,       │ │                                 │
│              actual, margin}         │ │  legality is NOT selectable:    │
│   costing.py callout · deadhead ·    │ │  "would this breach?" returns   │
│              delay · cancellation    │ │  NOT_A_QUERY → rules engine     │
│   candidates.py enumerate → 7 rules  │ │                                 │
│              → cost/delay/risk → rank│ │                                 │
└───────────────┬──────────────────────┘ └────────────────┬────────────────┘
                ▼                                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  DATA  one loader, two representations            domain/data.py · db.py │
│  objects: crew→pairings · flight→pairing · date→flights · crew→certs     │
│  SQLite: 16 tables, built from the same loader, fingerprinted against    │
│          the source JSON — edit the data and it rebuilds, never stales   │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  NARRATOR  structured result → English            backend/narrator.py    │
│  no tools, no dataset, no arithmetic — including derived figures         │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  GROUNDING                                        backend/grounding/     │
│  validator.py    every id and consequential number traced to a result    │
│  completeness.py the opposite failure — an answer that drops evidence    │
│  citations.py    computed from the evidence list, never model-generated  │
│  fail → regenerate once → abstain                                        │
└──────────────────────────────────────────────────────────────────────────┘
```

## 2. The boundary, explicitly

| LLM does | Deterministic code does |
|---|---|
| Turn a sentence into an `ExecutionPlan` | Decide whether that plan may run |
| Write a `SELECT` for an open-ended data question | Refuse it unless it is one read-only statement |
| Resolve fuzzy references ("tomorrow", "the VT-DXA captain") | Every hour, rupee, date window and rule verdict |
| Render a rule trace into controller English | Candidate enumeration and ranking |
| Say "I can't answer that reliably" | Which records were read (the evidence list) |

Everything the model produces is untrusted input to the validator. Nothing
downstream of the validator can be influenced by the model except through a
plan that passed it.

## 3. Why a plan instead of tool-calling

The first build used native tool-calling: the model called tools directly in a
loop. It worked, and it was replaced, for four reasons.

1. **Auditability.** A plan is an artifact. "Model called three tools" is not
   reviewable; `{"intent": "recommend_crew", "steps": [...]}` is, and it is
   what gets written to the audit log before anything executes.
2. **A real security boundary.** Tool-calling has an allowlist by construction
   but no place to check *meaning*. The validator rejects a plan naming a crew
   member who does not exist, or a date outside the schedule week, before any
   work happens.
3. **Cost.** Tool-calling attaches ~1,900 tokens of JSON schema to every round
   trip, and needs two or three of them. Plan → execute → narrate is exactly
   two calls, and the planner sees a ~600-token catalogue instead of schemas.
   Measured: Tier 3 fell from 5,200 tokens per question to ~3,700.
4. **A whole failure class disappears.** The narrator receives no tool
   definitions, so it cannot emit a tool call, so Groq's
   `tool_choice is none, but model called a tool` 400 — which caused 7 of 10
   errors in the tool-calling evaluation — is structurally impossible.

## 4. `RuleResult` carries arithmetic, not a boolean

```json
{ "rule_id": "RULE-DUTY-02", "passed": false, "limit": 60.0, "actual": 61.33,
  "margin": -1.33, "window": "2026-09-09..2026-09-15",
  "detail": "61.33h duty in 7 days ending 2026-09-15 vs 60h limit: exceeds by 1h20m" }
```

The explanation a controller reads is *derived* from those numbers. The
narrator renders `detail`; it cannot drift from it because there is nothing
left to compute. A rule that passes still reports its headroom, which is what
makes "can I also give them tomorrow?" cheap to answer.

## 5. The world overlay

The base snapshot is immutable. A disruption produces a copy-on-write `World`
and the kernel runs against that, so "if C-2087 covers P-2291, does anything
break?" is a real simulation over modified state rather than a request for the
model to imagine consequences. Worlds do not chain — S2's sick call does not
exist in S6's world.

## 6. Getting past 8,000 tokens per minute

The free tier's limits are **per model**, not per account: 8,000 TPM, 200,000
tokens/day and 1,000 requests/day each. Four models on this account accept tool
work, so the real ceiling is **32,000 TPM and 800,000 tokens/day** — the
single-model client was leaving three quarters of it unused.

```
backend/ratelimit.py       per-model Budget · sliding 60s window · ModelPool
backend/client.py          pooled HTTP · failure handled by class
backend/cache.py           plan cache + evidence-bound answer cache
planner/planner.py       fast path — the cheapest call is the one not made
```

Five mechanisms, in order of effect:

1. **Pool across models.** `ModelPool.acquire()` predicts a request's cost and
   picks a model that can pay for it now. A 429 is a bug in our accounting, not
   a normal event.
2. **Fast path.** Unambiguous lookups are planned by regex — 0 tokens, ~0.05 ms.
   11 of 16 Tier-1 questions never reach a model.
3. **Trust the headers.** Every response carries `x-ratelimit-remaining-tokens`
   and a reset interval. Local estimates drift; these do not, so they overwrite
   our accounting on each call.
4. **Cache.** Plans key on the normalised question. Answers key on the question
   *and* the evidence they cite, so a cached answer cannot outlive its data.
5. **`reasoning_effort: low` for planning.** gpt-oss spends completion tokens on
   hidden reasoning before emitting anything. Planning is translation, not
   reasoning, and untuned it silently truncated plans at the token cap.

Failure handling is by class, never blanket retry: **429** marks the bucket and
moves on; **5xx** retries the same model with jittered backoff; **4xx** is our
bug and is surfaced rather than spending the daily request budget on it. Daily
exhaustion is distinguished from per-minute throttling — the first is not worth
waiting out, the second is.

## 7. Grounding and abstention

After the narrator writes, every `C-####`, `P-####`, `DX###`, `VT-DX#` and
`RULE-*` token and every consequential number is pulled back out of the text and
checked against what the tools returned. One correction round, then the answer
is withheld and replaced by an abstention pointing at the evidence panel.

Deliberately asymmetric: identifiers strictly, numbers only when large (money)
or precise (hours). Validating "3 legs" against the corpus produces noise, not
safety. Text is dash-normalised first — models emit `P-2291` with U+2011, and
without normalising, the bare digits read as a fabricated figure.

Abstention is a designed outcome with its own UI treatment (amber, not red),
not an error. The system abstains when the planner finds no operation that
fits, when the validator cannot be satisfied, when every step fails, and when
grounding fails twice.

## 8. Verification

| | |
|---|---|
| `python -m pytest tests -q` | **194 unit tests** - rule arithmetic, boundary conditions, validator refusals, executor ordering, grounding, rate limiter |
| `python eval.py kernel` | **70/70** against the shipped answer keys: all 6 scenarios plus the machine-comparable questions |
| `python eval.py llm` | end-to-end fact recall through the full chat stack |

The kernel reproduces every engineered case the problem statement is built on:
C-2087 breaching DUTY-02 by exactly 1h20m, C-3310 clean at ₹18,500, C-2210 legal
via deadhead at ₹41,200 with a 3h delay, C-3305 legal on day 1 and breaching on
day 2, C-2091 excluded by QUAL-05, the S3 closure's per-flight delay/FDP table,
and S4's 12.75h FDP against a 12.0h limit.

Independent check of the window semantics: recomputing `duty_hours_7d` and
`flight_hours_28d` from `daily_history` plus the rostered week reproduces the
published clock for **150/150 crew**.

## 9. Performance

```
fast-path plan       ~0.05 ms      0 tokens
kernel execution     0.1 - 10 ms   0 tokens
LLM plan             ~1 s          ~1,200 tokens
LLM narration        ~2 s          ~2,500 tokens
------------------------------------------------
Tier 1 (fast path)   ~2 s          ~2,100 tokens
Tier 3               ~4 s          ~3,700 tokens
cache hit            ~0 s          0 tokens
```

Compute latency and rate-limit queueing are reported separately everywhere,
because conflating them would misrepresent the system: the architecture meets
the "usable on a live shift" bar; the free API tier is what occasionally does
not.

## 10. Repository

```
backend/
  main.py              orchestration: plan -> validate -> execute -> narrate -> ground
  config.py            env-driven settings · model specs
  client.py            pooled Groq HTTP
  ratelimit.py         per-model budgets · ModelPool
  cache.py             plan + answer caches
  narrator.py          structured result -> prose
  telemetry.py         structured logging · audit trail
  planner/
    planner.py         deterministic fast path + LLM planner
    prompts.py         planner and narrator system prompts
    schemas.py         ExecutionPlan · PlanStep (pydantic)
    sqlagent.py        natural language -> one SELECT, one self-repair
  executor/
    validator.py       the security boundary
    executor.py        deterministic workflow engine
    registry.py        operation catalogue — one source for prompt and validator
  conversation.py      greetings · capabilities · "what is today's date"
  text.py              unicode normalisation at the last mile
  tools/
    lookup.py          Tier 1 — record lookups
    query.py           Tier 1 — typed query DSL, the long tail without new tools
    sql.py             Tier 1 — SQL agent: validate, run, cite
    impact.py          Tier 2 — consequence and simulation
    recommendation.py  Tier 3 — ranked options, morning brief
    notification.py    Tier 3 — verified callout facts
    common.py          shared evidence helpers
  domain/              the kernel, zero LLM
    data.py            9 JSON files -> indexed in-memory store
    models.py          RuleResult · Duty · Verdict · Option · ToolResult
    world.py           copy-on-write disruption overlay
    duty.py            report/release · sectors · FDP
    rules.py           7 legality predicates
    costing.py         callout · deadhead · delay · cancellation
    candidates.py      enumerate -> check -> rank
    db.py              SQLite mirror: 16 tables, fingerprinted, built once
    temporal.py        relative dates, ranges and dayparts -> real dates
  grounding/
    validator.py       post-hoc fact check
    completeness.py    the opposite failure: evidence dropped from the answer
    citations.py       deterministic, computed from the evidence list
frontend/
  streamlit_app.py     controller console
tests/                 194 unit tests, no network
evals/                 kernel.py (answer keys) · llm.py (recall) · custom.py (unseen)
cli.py                 terminal console; --kernel runs without an API key
eval.py                entry point: kernel | llm | custom | validate | all
data/                  the provided starter dataset, unmodified
  crewops.db           derived, rebuilt whenever the JSON changes
SAMPLES.md             worked inputs and outputs, including a failure case
PRESENTATION.md        the deck
```

This follows the layout in the architecture document, with one deliberate
deviation: the dataset stays at the repository root rather than moving under
`backend/`, because the supplied `validate.py` and `generate.py` expect it
there and the brief asks for the provided data unmodified.
`backend/domain/data.py` searches both locations.

## 11. Build order followed

1. Data loader → 2. domain kernel → 3. rule engine → 4. unit tests →
5. eval against answer keys → 6. tool layer → 7. executor → 8. plan validator →
9. LLM planner → 10. narrator → 11. grounding validator → 12. Streamlit UI.

The kernel and both evaluations run with **no LLM at all**. That is the safety
net: if the model layer fails during the demo, `python cli.py --kernel` and
`python eval.py kernel` still prove the system computes the right answers.
