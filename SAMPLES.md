# Sample inputs and outputs

Recorded from the shipped system, not retyped. Every run below is in
`samples/` as JSON with the plan, the tools called, the evidence list, the
token count and the latency.

Tables are reflowed to fit this page; every identifier, figure and rule
verdict is verbatim from the recorded run.

Reproduce any of them:

```bash
python cli.py                          # ask them by hand
python eval.py llm --set core          # all 38, scored against the answer keys
python eval.py llm --set user          # the 32 controller-voice questions
```

---

## Tier 1 — lookup

**Q. Who is on reserve at BLR on 2026-09-15, and what are their on-call windows?**

```
Reserves on 15 Sep 2026 at BLR

| Crew ID | Name      | Rank              | Ratings     | On-call window (UTC) |
|---------|-----------|-------------------|-------------|----------------------|
| C-1329  | P. Pillai | Cabin Crew        | A320, ATR72 | 04:00 - 16:00        |
| C-2248  | H. Reddy  | Cabin Crew        | A320, ATR72 | 04:00 - 16:00        |
| C-4809  | M. Das    | Cabin Crew        | A320, ATR72 | 00:00 - 12:00        |
| C-5418  | S. Das    | Cabin Crew        | A320, ATR72 | 04:00 - 16:00        |
| C-3305  | V. Menon  | Captain           | A320        | 00:00 - 05:30        |
| C-3310  | D. Reddy  | Captain           | A320        | 06:00 - 18:00        |
| C-3315  | M. Pillai | Captain           | ATR72       | 03:00 - 15:00        |
| C-3311  | K. Rao    | First Officer     | A320        | 06:00 - 18:00        |
| C-3312  | P. Sharma | First Officer     | A320        | 00:00 - 12:00        |
| C-3316  | N. Verma  | First Officer     | ATR72       | 03:00 - 15:00        |
| C-2111  | A. Gupta  | Senior Cabin Crew | A320, ATR72 | 04:00 - 16:00        |
| C-3677  | M. Iyer   | Senior Cabin Crew | A320, ATR72 | 04:00 - 16:00        |

[list_reserves · 12 records · kernel 0.4 ms · 1 model call · 2,334 tokens]
```

All twelve, not a sample. The narrator is told that a truncated list is a wrong
answer, and the completeness pass re-checks the finished text against the
evidence before it is shown.

---

## Tier 2 — consequence

**Q. If Captain C-2087 is assigned to cover P-2291 from 15 Sep, does any rule
breach? Give the detail.**

```
Rule breach detected

| Rule         | Date       | Limit | Actual | Margin | Detail                                     |
|--------------|------------|-------|--------|--------|--------------------------------------------|
| RULE-DUTY-02 | 2026-09-15 | 60h   | 61.33h | -1h20m | exceeds the 60h/7d limit by 1h20m          |
| RULE-DUTY-02 | 2026-09-16 | 60h   | 61.08h | -1h05m | exceeds the 60h/7d limit by 1h05m          |

C-2087 (Captain, BLR) would be assigned to P-2291 starting 15 Sep. The
assignment is not legal: RULE-DUTY-02 breaches on both days. No other rule is
breached; all other checked rules passed.

[check_assignment · 14 rule checks, 2 breaches · kernel 1.1 ms · 1,889 tokens]
```

The margin `-1h20m` is not narrated. `RuleResult` carries `limit`, `actual` and
`margin`, and the sentence is assembled from them — which is why it agrees with
the answer key to the minute.

---

## Tier 3 — recommendation

**Q. Captain C-1042 is out for pairing P-2291 (15-16 Sep). Produce ranked
resolution options with costs and reasoning.**

```
| Rank | Action (Crew ID)                                   | Cost        | Delay | Reasoning                                                              |
|------|----------------------------------------------------|-------------|-------|------------------------------------------------------------------------|
| 1    | Assign Captain C-3310 (reserve callout)            | INR 18,500  | 0 h   | BLR-based, A320-rated, on call 06:00-18:00Z, report 06:00Z, 45 min, risk 0.17 |
| 2    | Assign Captain C-5566 (day-off callout)            | INR 24,000  | 0 h   | BLR-based, A320-rated, not rostered these days, 60 min, risk 0.08      |
| 3    | Assign Captain C-3983 (day-off callout)            | INR 24,000  | 0 h   | BLR-based, A320-rated, not rostered these days, 45 min, risk 0.10      |
| 4    | Assign Captain C-1526 (day-off callout)            | INR 24,000  | 0 h   | BLR-based, A320-rated, not rostered these days, 90 min, risk 0.12      |
| 5    | Assign Captain C-2210 (reserve + deadhead from DEL) | INR 41,200 | 3.0 h | DEL-based, positions on DX402, first departure delayed ~3.0 h          |
| 6    | Cancel all 6 flights of the pairing                | INR 1,500,000 | 0 h | Legal but strands 972 seats across 6 legs; the benchmark, not a plan   |

Why C-3310: cheapest legal option, no delay, all seven rules pass.

Notable excluded candidates
| C-3305 | reserve on-call window 00:00-05:30Z does not cover the 06:00Z report |
| C-1600 | no A320 rating (RULE-QUAL-05)                                        |
| C-2087 | would exceed the 60h/7d limit by 1h20m (61.33h vs 60h) on 15 Sep     |

[recommend_options · 150 candidates, 21 rule checks · kernel 6.2 ms · 3,805 tokens]
```

The rejected candidates are the point. "Why not C-3305?" is the controller's
next question, and the answer is the strongest evidence that all 150 were
actually checked rather than three plausible ones surfaced.

---

## When it does not know

**Q. If I could add exactly one more person to the reserve pool, what rank,
rating and window should it be, and why?**

```
I can't answer that reliably. No operation supports designing a new reserve
profile; the operations cover assignments, replacements and queries against
the reserves that exist, not the shape of one that does not.

[no tools called - the planner found no operation and said so]
```

The plan validator allowlists operations against the live registry, so a plan
naming something that does not exist never reaches the executor. The narrator
is separately told that "the result does not answer this" is a correct
outcome. Across the 38-question core set the system abstains 0 times and is
grounded 38/38; on the harder controller-voice set it abstains 6 times out of
32 rather than guess.

---

## A case it handles poorly

The brief asks for one, with analysis. This is the honest one.

**Q. For P-2291 day 1, how much duty-clock headroom does each rostered crew
member actually have?**

```
Duty-clock headroom on day 1 (2026-09-15) for each crew member rostered on P-2291

| Crew ID | Role              | Duty headroom, 7 days |
|---------|-------------------|-----------------------|
| C-4273  | Cabin Crew        | 29.29 h               |
| C-1042  | Captain           | 29.57 h               |
| C-1694  | First Officer     | 29.79 h               |
| C-1873  | Cabin Crew        | 31.72 h               |
| C-4395  | Cabin Crew        | 31.80 h               |
| C-3005  | Senior Cabin Crew | 32.80 h               |
```

Six of six crew, correctly ranked, arithmetic exact. Scored **0.33**.

**What went wrong.** The answer key wants the headroom they walk *into* the
duty with (39.07 h for C-1042); the answer gives the headroom left *after*
flying it (29.57 h). Both are true — 39.07 - 9.5 = 29.57, the duty itself —
and both are what a controller means, on different days. "Can they absorb a
delay tomorrow?" wants the second. "Have they got room for this at all?" wants
the first.

**Why the architecture did not catch it.** Nothing here is a hallucination, so
grounding passes: every figure traces to a tool result. The rules engine had
computed both numbers and the tool reported one, under the name `headroom` —
and an ambiguous name is not something a downstream check can detect, because
the answer is *right* for one reading of the question.

**The fix, and its limit.** `pairing_headroom` now returns both, named
explicitly (`duty_headroom_7d_before_this_duty` and `..._after_this_duty`), and
the narrator is forbidden from computing differences, so anything it should be
able to say has to be in the result already. That removes the ambiguity. It
does not remove the class: **the failure was in what the tool chose to report,
and nothing downstream can audit a number that was never produced.** Grounding
checks that stated numbers are real. Completeness checks that named sets are
whole. Neither can ask whether the right quantity was computed. That gap is
open, and it is where the next real defect will come from.

**The same class, unfixed.** On the 32 controller-voice questions the system
sits at Tier 2 **0.63** mean fact recall, and the remaining losses are almost
all this shape: the tool computed the right thing, the answer rendered a
subset of the columns. `roster_load` now reports "26 crew tie at 3 duty days"
rather than showing ten as if they were a ranking; `check_assignment` now
carries the margins of the rules that *passed*, not just the ones that broke.
The structural fix is to render tables deterministically from the tool result
and let the model write only the prose around them. We did not build it.
