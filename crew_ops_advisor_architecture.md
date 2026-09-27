# Crew Ops Advisor — Architecture

## 1. Architecture Principle

> **The LLM decides what needs to be done, but deterministic code decides what is true.**

The Crew Ops Advisor is designed around a strict boundary between **LLM reasoning** and **deterministic airline operations logic**.

The LLM is responsible for:

- Understanding natural-language questions
- Extracting entities and intent
- Resolving references such as "tomorrow" or "the captain on DX412"
- Creating a structured execution plan
- Converting deterministic results into controller-friendly language
- Abstaining when the system cannot answer reliably

The deterministic system is responsible for:

- Retrieving operational data
- Performing arithmetic
- Calculating duty and flight hours
- Applying all seven legality rules
- Simulating disruptions
- Enumerating candidate crew
- Calculating costs and delays
- Ranking legal alternatives
- Producing evidence and reasoning traces

The key safety principle is:

> **LLM = Planner + Narrator. Executor + Domain Kernel = Source of Truth.**

---

# 2. System Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                              CREW CONTROLLER                                  │
│                                                                              │
│   "Captain C-1042 is sick tomorrow. What should I do?"                      │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                                   │
│                            Streamlit UI                                      │
│                                                                              │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────┐ │
│  │   Chat Window   │  │ Evidence / Trace │  │ Rules / Cost / Risk Details │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────────────────┘ │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
                                     │ Natural Language
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           LLM PLANNER                                        │
│                         Tool / Plan Agent                                    │
│                                                                              │
│  1. Understand intent                                                       │
│  2. Extract entities                                                        │
│  3. Resolve dates / references                                              │
│  4. Select operations                                                       │
│  5. Build structured execution plan                                         │
│                                                                              │
│  Output: Pydantic / JSON ExecutionPlan                                       │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           PLAN VALIDATOR                                     │
│                                                                              │
│  • Is the operation allowed?                                                │
│  • Are required parameters present?                                         │
│  • Are identifiers valid?                                                   │
│  • Are dates valid?                                                         │
│  • Are dependencies valid?                                                  │
│  • Is the plan within allowed tool limits?                                  │
│                                                                              │
│                Invalid Plan → Reject / Repair / Abstain                      │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
                                     │ Validated ExecutionPlan
                                     ▼
══════════════════════════════════════════════════════════════════════════════════
                    LLM / DETERMINISTIC BOUNDARY
══════════════════════════════════════════════════════════════════════════════════
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              EXECUTOR                                        │
│                       Deterministic Workflow Engine                          │
│                                                                              │
│  • Executes approved operations                                             │
│  • Resolves step dependencies                                               │
│  • Passes outputs between steps                                             │
│  • Maintains disruption state                                               │
│  • Collects evidence                                                        │
│  • Never invents data                                                       │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
                  ┌──────────────────┼───────────────────┐
                  ▼                  ▼                   ▼
          ┌──────────────┐   ┌──────────────┐   ┌────────────────┐
          │ Tier 1 Tools │   │ Tier 2 Tools │   │  Tier 3 Tools │
          └──────┬───────┘   └──────┬───────┘   └───────┬────────┘
                 │                  │                   │
                 └──────────────────┼───────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         DETERMINISTIC DOMAIN KERNEL                          │
│                              ZERO LLM                                        │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────────┐ │
│  │    world.py    │  │    duty.py     │  │            rules.py             │ │
│  │                │  │                │  │                                │ │
│  │ Base snapshot  │  │ Report time    │  │ 7 legality predicates           │ │
│  │ Disruption     │  │ Release time   │  │ FDP / Duty / Flight / Rest      │ │
│  │ Overlay        │  │ FDP            │  │ Qualification / Certification  │ │
│  │ Copy-on-write  │  │ Sector count   │  │ Base / Deadhead                 │ │
│  └────────────────┘  └────────────────┘  └────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────────┐ │
│  │   costing.py   │  │ candidates.py  │  │          RuleResult             │ │
│  │                │  │                │  │                                │ │
│  │ Callout        │  │ Enumerate      │  │ passed                         │ │
│  │ Deadhead       │  │ Filter legal   │  │ limit                          │ │
│  │ Delay          │  │ Score          │  │ actual                         │ │
│  │ Cancellation   │  │ Rank           │  │ margin                         │ │
│  └────────────────┘  └────────────────┘  │ window / detail                │ │
│                                          └────────────────────────────────┘ │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
│                                                                              │
│  flights.json        crew.json          rosters.json                        │
│  duty_clocks.json    reserve_pool.json   certifications.json                │
│  rules.json          costs.json         risk_signals.json                    │
│  scenarios.json      questions.json                                           │
│                                                                              │
│                       ↓ Startup Load                                        │
│                                                                              │
│                   SQLite / In-Memory Store                                   │
│                                                                              │
│  crew → pairings       flight → pairing      date → flights                  │
│  crew → certifications      crew → daily history                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

# 3. LLM Planner → Executor Architecture

The main architectural improvement is to avoid allowing the LLM to directly control arbitrary tool execution.

Instead:

```text
User Question
      │
      ▼
┌───────────────┐
│  LLM Planner  │
└───────┬───────┘
        │
        │ Structured Plan
        ▼
┌───────────────┐
│ Plan Validator│
└───────┬───────┘
        │
        │ Valid Plan
        ▼
┌───────────────┐
│    Executor   │
└───────┬───────┘
        │
        ▼
┌──────────────────────┐
│ Deterministic Tools  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    Domain Kernel     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Grounded Result      │
└──────────┬───────────┘
           │
           ▼
┌───────────────┐
│ LLM Narrator  │
└───────┬───────┘
        │
        ▼
      User
```

This creates a clean separation between:

- **Planning**
- **Execution**
- **Computation**
- **Explanation**

---

# 4. Responsibilities of Each Layer

## 4.1 LLM Planner

The planner converts natural language into a typed execution plan.

Example:

User:

```text
Captain C-1042 is sick tomorrow. What should I do?
```

LLM output:

```json
{
  "intent": "recommend_crew",
  "entities": {
    "crew_id": "C-1042",
    "date": "2026-09-15"
  },
  "steps": [
    {
      "id": "s1",
      "operation": "get_pairing",
      "args": {
        "crew_id": "C-1042",
        "date": "2026-09-15"
      }
    },
    {
      "id": "s2",
      "operation": "simulate_sick_call",
      "depends_on": ["s1"],
      "args": {
        "crew_id": "C-1042",
        "date": "2026-09-15"
      }
    },
    {
      "id": "s3",
      "operation": "recommend_options",
      "depends_on": ["s2"]
    }
  ]
}
```

The LLM does **not** calculate:

- Duty hours
- FDP
- Cost
- Delay
- Rule margins
- Candidate legality

It only determines:

> "What deterministic operations are needed?"

---

# 5. Execution Plan Schema

Use a strongly typed schema, preferably Pydantic.

```python
class PlanStep(BaseModel):
    id: str
    operation: str
    args: dict = {}
    depends_on: list[str] = []


class ExecutionPlan(BaseModel):
    intent: str
    entities: dict
    steps: list[PlanStep]
```

Example:

```json
{
  "intent": "check_assignment",
  "entities": {
    "crew_id": "C-2087",
    "flight_id": "DX412"
  },
  "steps": [
    {
      "id": "check",
      "operation": "check_assignment",
      "args": {
        "crew_id": "C-2087",
        "flight_id": "DX412"
      }
    }
  ]
}
```

---

# 6. Plan Validator

The validator is a security and correctness boundary.

It verifies:

```text
ExecutionPlan
      │
      ├── Operation exists?
      ├── Operation allowed?
      ├── Required arguments present?
      ├── IDs valid?
      ├── Date valid?
      ├── Dependencies valid?
      ├── Step count <= configured limit?
      └── No arbitrary code / SQL / filesystem operation?
```

Example:

```python
ALLOWED_OPERATIONS = {
    "find_crew",
    "duty_clock",
    "list_reserves",
    "list_flights",
    "expiring_certs",
    "get_pairing",
    "sick_call_impact",
    "check_assignment",
    "station_closure_impact",
    "recommend_options",
    "draft_notification",
}
```

The LLM cannot introduce a new operation such as:

```text
execute_python()
run_sql()
shell_command()
```

---

# 7. Executor

The executor is responsible for running the plan.

Example:

```python
def execute(plan: ExecutionPlan):
    context = {}

    for step in topological_sort(plan.steps):
        args = resolve_dependencies(step.args, context)

        result = TOOL_REGISTRY[
            step.operation
        ](**args)

        context[step.id] = result

    return context
```

The executor should be:

- Deterministic
- Observable
- Auditable
- Testable
- Independent of the LLM

---

# 8. Tool Layer

Every tool exposes a typed interface.

## Tier 1 — Lookup

```text
find_crew
duty_clock
list_reserves
list_flights
expiring_certs
get_pairing
```

Examples:

```python
find_crew(
    crew_id="C-1042"
)
```

```python
duty_clock(
    crew_id="C-1042"
)
```

```python
list_reserves(
    station="BLR",
    date="2026-09-15"
)
```

## Tier 2 — Consequence / Simulation

```text
sick_call_impact
check_assignment
station_closure_impact
```

These tools operate against the deterministic world model.

## Tier 3 — Recommendation

```text
recommend_options
draft_notification
```

`recommend_options` should invoke the deterministic candidate engine rather than ask the LLM to choose a crew.

---

# 9. Standard Tool Result Envelope

Every tool returns the same structure:

```json
{
  "data": {},
  "evidence": [
    "rosters.json#P-2291",
    "duty_clocks.json#C-2087"
  ],
  "rules_evaluated": []
}
```

This gives the system three important properties:

1. Grounding
2. Explainability
3. Auditability

The UI can display the evidence directly.

---

# 10. Domain Kernel

The domain kernel contains all business-critical logic.

It contains **zero LLM calls**.

## world.py

Maintains:

```text
Base World
     │
     ├── flights
     ├── crew
     ├── rosters
     ├── duty clocks
     └── certifications
```

A disruption creates an overlay:

```text
Base World
     │
     ▼
Copy-on-Write Overlay
     │
     ├── C-1042 = sick
     └── 15 Sep
```

This means what-if scenarios do not mutate the original dataset.

---

# 11. Duty Calculation

`duty.py` performs deterministic calculations.

Example:

```text
Report Time = First Departure - 60 minutes

Release Time = Last Arrival + 30 minutes

FDP = Release Time - Report Time
```

It also calculates:

- Sector count
- Duty duration
- Flight duration
- Daily accumulation
- 7-day duty totals
- 28-day flight totals

The LLM never performs these calculations.

---

# 12. Rules Engine

`rules.py` contains the seven legality predicates.

```text
RULE-FDP-01
Maximum FDP of 13 hours, reduced by sectors

RULE-DUTY-02
Maximum 60 duty hours in any 7 consecutive days

RULE-FLT-03
Maximum 100 flight hours in any 28 consecutive days

RULE-REST-04
Minimum 12 hours rest before duty

RULE-QUAL-05
Valid aircraft rating required

RULE-CERT-06
Certifications must be valid

RULE-BASE-07
Reserve callout from base unless deadhead cost is applied
```

Each rule returns a `RuleResult`.

---

# 13. RuleResult

Never return only:

```json
{
  "legal": false
}
```

Return the arithmetic that produced the decision:

```json
{
  "rule_id": "RULE-DUTY-02",
  "passed": false,
  "limit": 60.0,
  "actual": 61.33,
  "margin": -1.33,
  "window": "2026-09-09..2026-09-15",
  "detail": "Adding P-2291 day 1 (9.4h) exceeds the 7-day duty cap by 1h20m"
}
```

This is critical because the explanation is generated from deterministic values.

---

# 14. Candidate Engine

`candidates.py` performs Tier 3 recommendation.

```text
Required Role
     │
     ▼
Enumerate Candidates
     │
     ├── Reserve crew
     ├── Day-off crew
     └── Other-base crew
     │
     ▼
Apply Constraints
     │
     ├── Qualification
     ├── Certification
     ├── Rest
     ├── Duty hours
     ├── Flight hours
     ├── FDP
     └── Base / deadhead
     │
     ▼
Legal Candidates
     │
     ▼
Calculate
     │
     ├── Cost
     ├── Delay
     └── Disruption Risk
     │
     ▼
Rank
```

Recommended ranking:

```text
1. Legal candidates only
2. Lowest operational cost
3. Lowest delay
4. Lowest disruption risk
```

---

# 15. Risk Signals

`risk_signals.json` can be used as a tiebreaker.

Example:

```text
Candidate A
Cost = ₹24,000
Risk = 0.21

Candidate B
Cost = ₹24,000
Risk = 0.78
```

Both are legal and cost the same.

The system chooses Candidate A because it has lower disruption risk.

This demonstrates meaningful use of the provided risk data rather than using it as decoration.

---

# 16. Cost Engine

`costing.py` handles deterministic operational cost.

Examples:

```text
Reserve callout      ₹18,500
Pilot day-off        ₹24,000
Deadhead positioning ₹6,500
Delay                ₹5,400 / duty hour
Cancellation         ₹250,000 / leg
```

The LLM never calculates these numbers.

---

# 17. End-to-End Example

User:

```text
Captain C-1042 is sick tomorrow. What should I do?
```

### Step 1 — Planner

```json
{
  "intent": "recommend_crew",
  "entities": {
    "crew_id": "C-1042",
    "date": "2026-09-15"
  },
  "steps": [
    {
      "id": "impact",
      "operation": "sick_call_impact",
      "args": {
        "crew_id": "C-1042",
        "date": "2026-09-15"
      }
    },
    {
      "id": "recommend",
      "operation": "recommend_options",
      "depends_on": ["impact"],
      "args": {}
    }
  ]
}
```

### Step 2 — Validation

```text
✓ Operation exists
✓ crew_id exists
✓ Date valid
✓ Dependency valid
✓ No unsupported operation
```

### Step 3 — Executor

```text
sick_call_impact()
        │
        ▼
Pairing P-2291
        │
        ├── DX412
        ├── DX413
        └── DX588
        │
        ▼
recommend_options()
```

### Step 4 — Candidate Engine

```text
Enumerate candidates
        │
        ▼
Check 7 rules
        │
        ▼
Calculate cost
        │
        ▼
Calculate delay
        │
        ▼
Check risk
        │
        ▼
Rank legal candidates
```

### Step 5 — Deterministic Result

```json
{
  "options": [
    {
      "rank": 1,
      "crew_id": "C-3310",
      "legal": true,
      "cost_inr": 18500,
      "coverage": [
        "DX412",
        "DX413",
        "DX588"
      ],
      "risk": 0.21
    },
    {
      "rank": 2,
      "crew_id": "C-2210",
      "legal": true,
      "cost_inr": 41200,
      "coverage": [
        "DX412",
        "DX413",
        "DX588"
      ],
      "risk": 0.42
    }
  ]
}
```

### Step 6 — LLM Narrator

The LLM receives only the structured result and produces:

```text
C-1042's absence breaks pairing P-2291, affecting DX412, DX413
and DX588.

Recommended action: assign reserve C-3310.

Why:
• Legally qualified
• Covers all three flights
• BLR based
• Cost: ₹18,500
• Lower disruption risk

Alternative: C-2210 is also legal but costs ₹41,200.
```

The narrator is **not allowed to add facts that aren't present in the result**.

---

# 18. Grounding Validator

After the narrator produces the response:

```text
LLM Draft
    │
    ▼
Grounding Validator
    │
    ├── Crew IDs
    ├── Flight IDs
    ├── Pairing IDs
    ├── Rule IDs
    └── Numeric values
    │
    ▼
Compare against tool-result bundle
```

If the response contains:

```text
C-9999
```

but no tool result contains `C-9999`:

```text
Grounding Failure
       │
       ├── Regenerate once
       │
       └── If still invalid → Abstain
```

This prevents hallucinated operational facts.

---

# 19. Abstention

Abstention is a normal outcome.

Example:

```text
User:
"Which crew member will probably call in sick next week?"
```

If the system is not designed to predict this:

```text
I can't answer that reliably from the available operational data.

I can instead show:
• crew with existing disruption-risk signals
• crew approaching duty limits
• upcoming certification expiries
• reserve availability
```

The system should prefer:

```text
"I don't know"
```

over an unsupported operational answer.

---

# 20. Multi-Turn Conversation

The architecture can support conversation state without allowing the LLM to become the source of truth.

Example:

```text
User:
C-1042 is sick tomorrow.

System:
C-1042 affects DX412, DX413 and DX588.

User:
What about C-2087?

LLM Planner:
intent = check_assignment
crew_id = C-2087
context = previous disruption

Executor:
check_assignment(C-2087, affected flights)

Kernel:
calculate legality

Narrator:
C-2087 would exceed RULE-DUTY-02 by 1h20m.
```

Conversation memory stores references and previous results, but the actual facts are revalidated by deterministic tools.

---

# 21. Performance Architecture

Because the dataset is small, the system should keep the deterministic path fast.

```text
JSON files
    │
    ▼
Startup Load
    │
    ▼
In-Memory Objects
    │
    ▼
Indexes
    │
    ├── crew → pairings
    ├── flight → pairing
    ├── date → flights
    ├── crew → certifications
    └── crew → daily history
```

Target:

```text
Kernel execution: < 50 ms
LLM planning:      ~1–2 sec
LLM narration:     ~1–2 sec
Total:             ~2–4 sec
```

Avoid unnecessary LLM calls.

---

# 22. Recommended Project Structure

```text
crew-ops-advisor/
│
├── backend/
│   ├── main.py
│   │
│   ├── planner/
│   │   ├── planner.py
│   │   ├── schemas.py
│   │   └── prompts.py
│   │
│   ├── executor/
│   │   ├── executor.py
│   │   ├── validator.py
│   │   └── registry.py
│   │
│   ├── tools/
│   │   ├── lookup.py
│   │   ├── impact.py
│   │   ├── recommendation.py
│   │   └── notification.py
│   │
│   ├── domain/
│   │   ├── world.py
│   │   ├── duty.py
│   │   ├── rules.py
│   │   ├── costing.py
│   │   └── candidates.py
│   │
│   ├── grounding/
│   │   └── validator.py
│   │
│   ├── data/
│   │   ├── flights.json
│   │   ├── crew.json
│   │   ├── rosters.json
│   │   ├── duty_clocks.json
│   │   ├── reserve_pool.json
│   │   ├── certifications.json
│   │   ├── rules.json
│   │   ├── costs.json
│   │   ├── risk_signals.json
│   │   ├── scenarios.json
│   │   └── questions.json
│   │
│   └── tests/
│       ├── test_rules.py
│       ├── test_duty.py
│       ├── test_costing.py
│       ├── test_candidates.py
│       └── test_scenarios.py
│
├── frontend/
│   └── streamlit_app.py
│
├── eval.py
├── requirements.txt
└── README.md
```

---

# 23. Build Order

Build in this order:

```text
1. DATA LOADER
      ↓
2. DOMAIN KERNEL
      ↓
3. RULE ENGINE
      ↓
4. UNIT TESTS
      ↓
5. EVAL AGAINST ANSWER KEYS
      ↓
6. TOOL LAYER
      ↓
7. EXECUTOR
      ↓
8. PLAN VALIDATOR
      ↓
9. LLM PLANNER
      ↓
10. LLM NARRATOR
      ↓
11. GROUNDING VALIDATOR
      ↓
12. STREAMLIT UI
      ↓
13. DEMO / PRESENTATION
```

The kernel and evaluation should work **without any LLM**.

That gives you a safety net if the model layer fails during the demo.

---

# 24. Tier Mapping

```text
                         CREW OPS ADVISOR
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
             ▼                  ▼                  ▼
          TIER 1              TIER 2             TIER 3
         LOOKUP            CONSEQUENCE        RECOMMENDATION
             │                  │                  │
             ▼                  ▼                  ▼
       Retrieval tools     Simulation tools    Candidate engine
             │                  │                  │
             ▼                  ▼                  ▼
          Data only         World overlay      Rules + Cost + Risk
             │                  │                  │
             └──────────────────┼──────────────────┘
                                ▼
                        Structured Result
                                │
                                ▼
                           LLM Narrator
```

---

# 25. Why This Architecture Is Strong

## Correctness

The LLM cannot calculate legality.

```text
LLM
 ❌ "61.33 hours looks acceptable"

Kernel
 ✓ actual = 61.33
 ✓ limit = 60
 ✓ margin = -1.33
 ✓ passed = false
```

## Explainability

Every decision contains:

```text
Rule
Limit
Actual
Margin
Window
Evidence
Detail
```

## Auditability

Every answer can be traced:

```text
User Question
      ↓
Execution Plan
      ↓
Tool Calls
      ↓
Dataset Records
      ↓
Rule Results
      ↓
Cost / Risk
      ↓
Final Answer
```

## Safety

The LLM cannot:

```text
❌ Execute arbitrary code
❌ Invent crew IDs
❌ Invent flights
❌ Override legality
❌ Modify source data
❌ Calculate operational values
```

## Agentic Behavior

The system is still agentic because the LLM:

```text
Understand
    ↓
Plan
    ↓
Select operations
    ↓
Use context
    ↓
Interpret results
    ↓
Explain
```

But execution remains controlled.

---

# 26. Key Design Decision for Judges

The architecture can be summarized in one sentence:

> **"Our LLM is not the decision-maker; it is the planner and narrator around a deterministic decision engine."**

Or, for the presentation:

> **"The LLM decides what to ask. The executor decides what to run. The domain kernel decides what is legal. The evidence decides what we can say."**

This is the central architectural story.

---

# 27. Hackathon Demo Flow

Recommended live demo:

### Demo 1 — Tier 1

```text
"Who's on reserve at BLR tomorrow?"
```

Show:

```text
LLM Plan
   ↓
list_reserves()
   ↓
Evidence
   ↓
Answer
```

### Demo 2 — Tier 2

```text
"C-1042 is sick tomorrow. Which flights are affected?"
```

Show:

```text
LLM Plan
   ↓
sick_call_impact()
   ↓
World Overlay
   ↓
Affected Pairing
   ↓
Affected Flights
   ↓
Rule Checks
   ↓
Explanation
```

### Demo 3 — Tier 3

```text
"C-1042 is sick tomorrow. What should I do?"
```

Show:

```text
Plan
 ↓
Impact
 ↓
Candidate Enumeration
 ↓
7 Rule Checks
 ↓
Cost
 ↓
Delay
 ↓
Risk
 ↓
Ranking
 ↓
Recommendation
```

### Demo 4 — Failure / Abstention

Ask something outside the supported domain.

Show:

```text
No reliable deterministic evidence
             ↓
         ABSTAIN
             ↓
"I can't answer that reliably."
```

This demonstrates that the system knows its limits.

---

# 28. Final Architecture

```text
                         ┌──────────────────┐
                         │       USER       │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   STREAMLIT UI   │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │       LLM PLANNER        │
                    │                          │
                    │ Intent + Entities + Plan │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     PLAN VALIDATOR       │
                    └────────────┬─────────────┘
                                 │
═════════════════════════════════╪═════════════════════════════════
              LLM               │       DETERMINISTIC
═════════════════════════════════╪═════════════════════════════════
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │        EXECUTOR          │
                    │   Controlled Operations   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       TOOL LAYER         │
                    │                          │
                    │ T1 │ T2 │ T3             │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      DOMAIN KERNEL       │
                    │                          │
                    │ World │ Duty │ Rules     │
                    │ Cost  │ Candidates       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       DATA LAYER         │
                    │       9 JSON files       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    STRUCTURED RESULT     │
                    │                          │
                    │ Data + Evidence + Rules  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      GROUNDING CHECK     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       LLM NARRATOR       │
                    │   Facts → Human English  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                              USER
```

## Final Principle

```text
              ┌───────────────────────────────┐
              │            LLM                │
              │                               │
              │  UNDERSTAND → PLAN → EXPLAIN  │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │       DETERMINISTIC CORE      │
              │                               │
              │  RETRIEVE → CALCULATE → CHECK │
              │  SIMULATE → COST → RANK       │
              └───────────────────────────────┘
```

**The LLM provides intelligence at the interface.  
The deterministic kernel provides truth at the core.**
