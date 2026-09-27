# Crew Ops Advisor
### AI-driven operational superintelligence for airline Crew Control

dCortex hackathon · built on the supplied synthetic dataset, unmodified

---

## 1 · The 6 a.m. problem

A captain calls in sick. Inside ten minutes the controller has to know:

> Which flights just lost their crew — including the ones tomorrow, because the
> pairing night-stops. Who can legally take them. Who can actually be reached.
> What it costs. And what breaks *next*.

Five screens, a dense rulebook, and one senior controller's memory.

**The bottleneck is not detecting the disruption. It is reasoning correctly and
fast about its consequences** — and that reasoning degrades exactly when it
matters most.

---

## 2 · The question we were actually answering

The brief names it: *what should the language model do, what should
deterministic code do, and how do you compose them?*

The obvious build — put the data in the prompt, let the model answer — works
for Tier 1 and fails at Tier 2 and 3. **Legality is exact arithmetic.** A model
that approximates a duty-hour sum produces answers that are fluent, confident
and wrong, which on an ops desk is worse than no answer at all.

So we drew one line and held it everywhere:

> ### The LLM plans and narrates. It never computes.

Not "we try to check the model's arithmetic". The model is never given the
arithmetic to do.

---

## 3 · The boundary, explicitly

| The LLM does | Deterministic code does |
|---|---|
| Turn a sentence into an `ExecutionPlan` | Decide whether that plan may run |
| Resolve fuzzy references — "tomorrow", "the VT-DXA captain" | Every hour, rupee, date window and rule verdict |
| Write a `SELECT` for an open-ended data question | Refuse it unless it is one read-only statement |
| Render a rule trace into controller English | Candidate enumeration, costing and ranking |
| Say "I can't answer that reliably" | Which records were read — the evidence list |

Everything the model produces is **untrusted input to a validator**. Nothing
downstream of that validator can be influenced by the model except through a
plan that passed it.

---

## 4 · The pipeline

```
question
   │
   ├─ conversation gate ──── "hi" · "what can you do" · "what day is it"   0 tokens
   ├─ temporal resolver ──── "this weekend", "next 3 days" → real dates
   │
   ├─ PLAN      fast path (regex, 0 tokens) · LLM planner · SQL agent
   │
   ├─ VALIDATE  operation allowlist · identifiers must exist · dates in week
   │            acyclic · step cap · one repair round · else abstain
   │  ═══════════════ LLM / DETERMINISTIC BOUNDARY ═══════════════
   ├─ EXECUTE   33 operations over the rules kernel — zero LLM
   │
   ├─ NARRATE   structured result → English. No tools. No dataset. No maths.
   │
   └─ GROUND    every id and number traced back · completeness · citations
                fail → regenerate once → abstain
```

Two model calls per question. The narrator is never given tool definitions, so
it structurally cannot call one.

---

## 5 · Five things worth a minute each

**1 · `RuleResult` carries arithmetic, not a boolean.** Every rule returns
`{limit, actual, margin}`, so the explanation is *derived*, not narrated:

> "Exceeds the 60h/7d limit by 1h20m (61.33h vs 60h) on 15 Sep"

The model did not work that out. It could not have.

**2 · A plan is an artifact, so it can be refused.** Tool-calling has an
allowlist by construction but nowhere to check *meaning*. Our validator rejects
a plan naming a crew member who does not exist, or a date outside the schedule
week, **before any work happens** — and writes it to the audit log first.

**3 · The world overlay makes what-ifs real.** "If C-2087 covers P-2291" is
evaluated against a copy-on-write `World` with that assignment applied — a
simulation, not an imagination.

**4 · Grounding, then completeness.** Grounding proves the answer contains
nothing the tools did not produce. Completeness catches the opposite and more
dangerous failure: an answer that is entirely true and quietly omits nine of
thirteen affected flights.

**5 · Rejected candidates are kept.** "Why not C-3305?" is the controller's
next question, and it is the strongest evidence that all 150 crew were checked
rather than three plausible ones surfaced.

---

## 6 · The scalability question, and the answer

Ask *"what is the longest block time?"* and a purpose-built tool answers.
Change one word to *"shortest"* and it refuses. **You cannot write a tool per
question shape** — that is not an architecture, it is a backlog.

Two layers absorb the long tail without growing the catalogue:

- **`query_data`** — a typed DSL with allowlisted entities, fields and
  operators. One rule answers every superlative in both directions.
- **`ask_sql`** — the question becomes one `SELECT` against a SQLite mirror of
  the snapshot, for the joins and aggregates the DSL cannot shape.

The SQL agent gets more rope than anything else here, so the rope is short and
inspected: **read-only connection**, one statement, `SELECT`/`WITH` only,
`LIMIT` imposed, 2-second timeout — and the SQL that ran is returned with the
answer, for a controller or a judge to read.

**And legality is not selectable.** "Would C-2087 breach if moved onto P-2291?"
returns `NOT_A_QUERY`: that is a simulation over a modified roster, not a row
that exists to be selected. It goes to the rules engine. *Widening what the
model may ask for did not widen what it may do.*

One dataset, two representations, **one loader**. The database is built by the
same code the kernel uses, fingerprinted against the source JSON, and rebuilds
itself when the data changes. Two sources of truth was the risk; a derived,
self-invalidating mirror is the answer.

---

## 7 · Engineering under a hard constraint

Groq's free tier gives **8,000 tokens per minute**. A Tier 3 question costs
~3,800. That is two questions a minute before the desk stalls.

- **Pool four models** — gpt-oss-120b, gpt-oss-20b, qwen3.6-27b, qwen3.8-27b —
  each with its own budget: **32,000 TPM aggregate**, planner on the small fast
  model, narrator on the large one.
- **Adaptive output estimation.** We reserved 4,025 tokens for calls that
  billed 1,780, throttling ourselves to one call per model per minute at 75%
  idle capacity. An EWMA over observed completions fixed it.
- **The fast path** answers 37 of 38 core questions with **zero planner
  tokens** — the single most effective optimisation available.
- **Caching** at three levels: plan, tool and evidence-bound answer. A repeated
  question returns 402× faster.

Measured honestly: compute latency and rate-limit queueing are reported
**separately** everywhere, because conflating them would misrepresent the
system. The architecture meets the live-shift bar; the free tier is what
occasionally does not.

---

## 8 · Results

**Deterministic kernel — no model involved**

| | |
|---|---|
| Answer-key checks, all 6 scenarios | **74 / 74** |
| Duty clocks reproduced independently | **150 / 150** crew |
| Unit tests, no network | **194** |
| Kernel compute | median **0.8 ms** |

**End to end — the 38 supplied questions.** Three consecutive runs, caches
cold, so this is the spread rather than the best one.

| Tier | Mean fact recall | Grounded | Abstained |
|---|---|---|---|
| 1 — lookup | **1.00** (all three runs) | 16 / 16 | 0 |
| 2 — consequence | **0.94 - 0.96** | 14 / 14 | 0 |
| 3 — recommendation | **0.95 - 0.98** | 8 / 8 | 0 |

38/38 grounded and zero abstentions in every run. Median compute latency
**1.3 - 2.0 s**, p90 under 4.2 s. **~2,500 tokens** per question, against 5,200
on the tool-calling build we replaced.

**A harder set — 32 questions rewritten in a controller's own voice**, not the
brief's phrasing: Tier 1 **0.91**, Tier 2 **0.63**, 32/32 grounded, 6
abstentions rather than 6 guesses. That gap is the real one, and §10 is about
it.

---

## 9 · One thing we found in the data

Question 45's shipped answer key computes 28-day flight hours by summing
`duty_history` — which **excludes the rostered week**, and so reads about eight
hours low per crew member. Our kernel reproduces the *published*
`duty_clocks.json` for all 150 crew, so we believe the key is wrong, and we did
not bend the engine to match it. The disagreement is documented, not hidden.

---

## 10 · Where it is weak — honestly

**Q. "For P-2291 day 1, how much duty-clock headroom does each crew member
have?"** Six of six crew, correctly ranked, arithmetic exact. **Scored 0.33.**

The key wants the headroom they walk *into* the duty with (39.07 h). We gave
the headroom left *after* flying it (29.57 h). Both true; both what a
controller means, on different days.

**Nothing caught it, and nothing could have.** Grounding checks that stated
numbers are real — they were. Completeness checks that named sets are whole —
it was. **Neither can ask whether the right quantity was computed.** The
failure was in what the tool chose to report, and there is no downstream audit
for a number that was never produced.

Fixed for this tool: it now returns both, explicitly named. Not fixed as a
class. That is where the next real defect will come from, and we would rather
say so than discover it on stage.

Also open:

- Deadhead positioning is same-day and single-leg.
- Cascading disruption is one level deep.
- Tier 2 on the controller-voice set is a **narration** gap, not an arithmetic
  one: the tool computes the right figures and the answer renders a subset of
  the columns. The structural fix — render tables deterministically from the
  tool result, leave the model only the prose around them — is designed, not
  built.

---

## 11 · Would this hold at real airline scale?

**Reasoning is local.** One pairing, one role, the affected crew's own windows.
Candidate enumeration is `O(crew of that rank)`, so a 5,000-crew carrier is a
rank-and-base indexed query away, not an algorithm change.

**What would actually strain** are the parts deliberately kept small: the world
overlay is copy-on-write over an in-memory snapshot, which at real scale becomes
a transaction against the roster store; duty-clock windows move from
recomputation to a maintained running total.

**The LLM layer is unaffected by carrier size.** It sees a plan and a result,
never the fleet.

**Crew PII.** Tools would return opaque crew keys, and the identity join would
happen in the UI *after* the answer is composed — prompts and provider logs
carry `C-1042`, never a person. Medical data reduces to a boolean at the tool
boundary: `cert_valid_on(date)`, never a diagnosis. `telemetry.audit()` already
logs every operation and the records it touched, as a by-product of the
evidence list — which is exactly what a regulator asks for after a bad
decision.

---

## 12 · Business impact

A sick call today costs ten minutes of cross-referencing by the one person on
shift who can do it well. Here it is one question, answered in about two
seconds, with the rule arithmetic attached and the cheapest legal option named
— together with the options that were rejected, and why.

The value is not the seconds saved. It is that **the reasoning is reviewable**:
every figure carries a citation to the record it came from, every plan reaches
the audit log before it runs, and when the data cannot support an answer the
system says so rather than producing a fluent one.

---

## 13 · Demo

```bash
python eval.py kernel                     # 74/74, no API key, no network
python cli.py                             # terminal chat
uvicorn backend.main:app --reload         # API + /docs
streamlit run frontend/streamlit_app.py   # controller console
```

**The five questions to ask, in order**

1. `Who is on reserve at BLR tomorrow?` — Tier 1, fast path, zero planner tokens
2. `How many duty hours does C-1042 have left this week?` — the clock
3. `Captain C-1042 calls in sick on 15 Sep for P-2291. Which flights are uncrewed?` — Tier 2; watch tomorrow's legs surface
4. `If C-2087 covers P-2291, does anything breach?` — the margin, to the minute
5. `Captain C-1042 is out for P-2291. What should I do?` — ranked, priced, with the rejections

Then, for the architecture point: `What is the shortest block time in the
schedule?` — the question a tool-per-shape build refuses.

---

## 14 · What the submission contains

| Deliverable | Where |
|---|---|
| Source code repository | this repository |
| Architecture diagram + LLM/deterministic boundary | `ARCHITECTURE.md` |
| README — setup, approach, trade-offs, limitations, PII | `README.md` |
| Sample inputs and outputs, with a failure case analysed | `SAMPLES.md` |
| Presentation | this document |
| Live demo | §13 |

---

> The best submission will be the one whose Advisor a real crew controller
> would want beside them at 6 a.m. on a bad day — because it is fast, because
> it is right, and because when it isn't sure, it says so.

All three are engineering decisions. We made them in that order.
