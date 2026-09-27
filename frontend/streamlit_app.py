"""Crew Ops Advisor - controller console.

    uvicorn backend.main:app --port 8000          # start the API first
    streamlit run frontend/streamlit_app.py       # then the console

A thin client: it holds no dataset, no kernel and no model pool, and talks to
the API over HTTP. That split is not ceremony - the rate limiter budgets four
token buckets across the whole desk, and it can only do that if one process
owns them. A second advisor inside the UI would double-spend a budget neither
copy could see.

The answer streams in while the pipeline runs above it, so a controller watches
the rules engine finish before a word of prose is written. Every identifier
carries a superscript citation back to the record it came from - computed by
the backend, never asked of the model.
"""
from __future__ import annotations

import json
import os
import uuid

import requests
import streamlit as st

API = os.environ.get("CREWOPS_API", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 180

st.set_page_config(page_title="Crew Ops Advisor", page_icon="✈", layout="wide")

STARTERS = [
    "Who is on reserve at BLR on 2026-09-15?",
    "Captain C-1042 calls in sick for 15 Sep - which flights are uncrewed?",
    "If Captain C-2087 covers P-2291 from 15 Sep, does anything breach?",
    "Captain C-1042 is out for P-2291 on 15-16 Sep. What should I do?",
    "Both captains C-3940 and C-1938 are sick on 18 Sep. Give the joint plan.",
    "Which crew member will call in sick next week?",
]

STAGE_LABEL = {
    "plan": "Planning — question to typed execution plan",
    "validate": "Validating — allowlist, identifiers, dates, dependencies",
    "execute": "Executing — deterministic rules engine",
    "narrate": "Writing — structured result to English",
    "ground": "Grounding — every figure traced back to a record",
}

CSS = """
<style>
  .stMarkdown sup { font-size: .68em; vertical-align: super; line-height: 0; }
  .stMarkdown sup:hover { cursor: help; }
  .cite-chip { display:inline-block; border:1px solid rgba(128,128,128,.35);
               border-radius:6px; padding:1px 7px; margin:2px 4px 2px 0;
               font-size:.78em; font-family:ui-monospace,monospace; }
</style>
"""


# --------------------------------------------------------------------------
# API client
# --------------------------------------------------------------------------
def api_get(path: str, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict):
    r = requests.post(f"{API}{path}", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_stream(payload: dict):
    """Yield decoded server-sent events from /ask/stream."""
    with requests.post(f"{API}/ask/stream", json=payload, stream=True,
                       timeout=TIMEOUT) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                return
            try:
                yield json.loads(body)
            except json.JSONDecodeError:
                continue


@st.cache_data(ttl=30, show_spinner=False)
def health():
    return api_get("/health")


def request_body(question: str) -> dict:
    return {
        "question": question,
        "session_id": st.session_state.session_id,
        "keep_history": True,
        "planner_model": st.session_state.get("planner"),
        "narrator_model": st.session_state.get("narrator"),
        "fastpath": st.session_state.get("fastpath", True),
        "cache": st.session_state.get("cache", True),
    }


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
if "turns" not in st.session_state:
    st.session_state.turns = []
if "pending" not in st.session_state:
    st.session_state.pending = None

try:
    HEALTH = health()
except Exception as e:                              # noqa: BLE001
    st.error(f"Cannot reach the Crew Ops API at `{API}`.\n\n`{e}`")
    st.info("Start it with:\n\n```bash\nuvicorn backend.main:app --port 8000\n```\n"
            "Point elsewhere with the `CREWOPS_API` environment variable.")
    st.stop()


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Crew Ops Advisor")
    st.caption("dCortex Air · Crew Control desk")
    st.markdown(
        f"**Snapshot** `{HEALTH['snapshot_utc']}`  \n"
        f"**Week** {HEALTH['week'][0]} → {HEALTH['week'][1]}  \n"
        f"{HEALTH['flights']} flights · {HEALTH['crew']} crew · "
        f"{HEALTH['pairings']} pairings  \n"
        f"{HEALTH['operations']} operations · all times UTC · INR")
    st.caption(f"API `{API}` · session `{st.session_state.session_id}`")

    st.divider()
    aliases = [m["alias"] for m in HEALTH["models"]]
    c1, c2 = st.columns(2)
    st.session_state.planner = c1.selectbox(
        "Planner", aliases,
        index=aliases.index(HEALTH["planner_model"]) if HEALTH["planner_model"] in aliases else 0)
    st.session_state.narrator = c2.selectbox(
        "Narrator", aliases,
        index=aliases.index(HEALTH["narrator_model"]) if HEALTH["narrator_model"] in aliases else 0)
    st.session_state.stream = st.toggle("Stream the answer", value=True)
    st.session_state.fastpath = st.toggle(
        "Deterministic fast path", value=HEALTH["fastpath_enabled"],
        help="Plans common questions with zero tokens")
    st.session_state.cache = st.toggle("Cache plans and answers",
                                       value=HEALTH["cache_enabled"])
    if not HEALTH["api_key_present"]:
        st.error("The API has no GROQ_API_KEY. Kernel operations still work; "
                 "chat will not.")

    st.divider()
    try:
        cap = api_get("/capacity")
        st.markdown(f"**Token pool** · {cap['aggregate_tpm']:,} TPM aggregate")
        st.progress(min(1.0, max(0.0, cap["headroom_now"] / max(1, cap["aggregate_tpm"]))),
                    text=f"{cap['headroom_now']:,} available now")
        with st.expander("Per model"):
            st.dataframe([{"model": m["model"].split("/")[-1], "calls": m["calls"],
                           "billed": m["tokens_billed"], "headroom": m["headroom"],
                           "reqs left": m["remaining_requests"]} for m in cap["models"]],
                         hide_index=True, width="stretch")
            st.caption(f"{cap['model_switches']} failovers · "
                       f"{cap['pool_waited_s']}s waiting on budget · "
                       f"cache {cap['cache']['plan_cache']['hit_rate']:.0%} plan hits")
    except Exception as e:                          # noqa: BLE001
        st.caption(f"capacity unavailable: {e}")

    st.divider()
    dates = [HEALTH["week"][0]] if HEALTH["week"][0] == HEALTH["week"][1] else None
    brief_date = st.text_input("Morning brief for", value="2026-09-15")
    if st.button("Run brief", width="stretch"):
        st.session_state.pending = (
            f"Give me the morning briefing for {brief_date}: per aircraft line, "
            "the tightest duty headroom, FDP slack and certification exposure.")
    if st.button("Clear conversation", width="stretch"):
        try:
            requests.post(f"{API}/session/reset",
                          params={"session_id": st.session_state.session_id},
                          timeout=15)
        except Exception:                           # noqa: BLE001
            pass
        st.session_state.turns = []
        st.rerun()


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def body(turn: dict) -> str:
    return turn.get("annotated") or turn.get("answer") or ""


def render_citations(turn: dict):
    cites = turn.get("citations") or []
    if not cites:
        return
    with st.expander(f"Sources · {len(cites)} records cited", expanded=False):
        st.caption("Computed from the tool results, not written by the model. "
                   "Hover a marker in the answer for the same detail.")
        for c in cites:
            line = f"**[{c['index']}]** `{c['ref']}` — {c['label']}"
            if c.get("detail"):
                line += f"  \n&nbsp;&nbsp;&nbsp;{c['detail']}"
            st.markdown(line, unsafe_allow_html=True)


def render_footer(turn: dict):
    flags = []
    if turn.get("plan"):
        icon = {"fastpath": "⚡", "cache": "♻️"}.get(turn.get("plan_source"), "🧠")
        flags.append(f"{icon} `{turn['plan']}`")
    rules = turn.get("rules_evaluated") or []
    breaches = [r for r in rules if not r["passed"]]
    if rules:
        flags.append(f"{len(rules)} rule checks"
                     + (f" · **{len(breaches)} breach**" if breaches else " · all pass"))
    flags.append(f"{len(turn.get('evidence') or [])} records")
    flags.append(f"kernel {turn.get('kernel_ms', 0):.1f}ms")
    thr = turn.get("throttled_s") or 0
    lat = turn.get("latency_s") or 0
    flags.append(f"{lat - thr:.1f}s + {thr:.0f}s throttle" if thr > 0.5 else f"{lat:.1f}s")
    tok = (turn.get("usage") or {}).get("total_tokens", 0)
    flags.append("0 tokens" if not tok else f"{tok} tokens")
    if not turn.get("grounded", True):
        flags.append(":red[grounding failed]")
    st.caption(" · ".join(flags))


def render_turn(turn: dict):
    if turn.get("error") and not turn.get("results"):
        st.error(turn.get("answer"))
    elif turn.get("abstained"):
        st.warning(body(turn))
    else:
        st.markdown(body(turn), unsafe_allow_html=True)
    render_citations(turn)
    render_footer(turn)


def run_streaming(question: str):
    """Drive one turn over SSE, showing each stage as it completes."""
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        status = st.status("Planning…", expanded=True)
        slot = st.empty()
        parts, turn = [], None
        try:
            for ev in api_stream(request_body(question)):
                kind = ev.get("type")
                if kind == "stage":
                    if ev["status"] == "running":
                        status.update(label=STAGE_LABEL.get(ev["name"], ev["name"]))
                    else:
                        mark = {"done": "✅", "cached": "♻️", "skipped": "⏭️",
                                "failed": "❌"}.get(ev["status"], "•")
                        bits = [f"{mark} **{ev['name']}** — {ev['status']}"]
                        if ev.get("detail"):
                            bits.append(f"`{ev['detail']}`")
                        if ev.get("source"):
                            bits.append(f"via {ev['source']}"
                                        + (" (0 tokens)" if ev["source"] == "fastpath" else ""))
                        if ev.get("kernel_ms") is not None:
                            bits.append(f"{ev['kernel_ms']:.1f} ms")
                        status.write(" · ".join(bits))
                        for rep in ev.get("repaired") or []:
                            status.caption(f"repaired — {rep}")
                        if ev.get("rewritten"):
                            status.caption("answer rewritten to drop an unsupported figure")
                        for tok in ev.get("unsupported") or []:
                            status.caption(f":red[unsupported: {tok}]")
                elif kind == "step":
                    icon = "✅" if ev["ok"] else "❌"
                    status.write(f"&nbsp;&nbsp;{icon} `{ev['operation']}` · "
                                 f"{ev['duration_ms']:.2f} ms · {ev['records']} records")
                    if ev.get("error"):
                        status.caption(f":red[{ev['error']}]")
                elif kind == "token":
                    parts.append(ev["text"])
                    slot.markdown("".join(parts) + " ▌")
                elif kind == "turn":
                    turn = ev["turn"]
                elif kind == "error":
                    status.write(f"❌ {ev.get('message')}")
        except Exception as e:                      # noqa: BLE001
            status.update(label="Failed", state="error")
            slot.error(f"Stream failed: {e}")
            return None

        status.update(label="Done", state="complete", expanded=False)
        if turn is not None:
            # The final answer can differ from what streamed: grounding runs
            # after the last token and may correct or withhold it.
            slot.empty()
            with slot.container():
                render_turn(turn)
        return turn


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)

# st.chat_input only docks to the bottom of the viewport when it is a
# TOP-LEVEL element. Declared inside a column it renders inline, so it drifts
# up the page as the conversation grows. Declaring it here - before the
# columns - keeps it pinned; the value is still available to the code below,
# because Streamlit reruns the whole script on submit.
typed = st.chat_input("e.g. how many duty hours does C-1042 have left this week?")

chat_col, panel_col = st.columns([3, 2], gap="large")

with chat_col:
    st.markdown("#### Ask")
    cols = st.columns(3)
    for i, s in enumerate(STARTERS):
        if cols[i % 3].button(s[:34] + "…", key=f"st{i}", width="stretch", help=s):
            st.session_state.pending = s
            st.rerun()

    for t in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(t["question"])
        with st.chat_message("assistant"):
            render_turn(t)

    q = typed or st.session_state.pending
    st.session_state.pending = None
    if q:
        if st.session_state.get("stream", True):
            turn = run_streaming(q)
            if turn is not None:
                st.session_state.turns.append(turn)
        else:
            with st.spinner("planning, checking the rulebook…"):
                st.session_state.turns.append(api_post("/ask", request_body(q)))
            st.rerun()

with panel_col:
    st.markdown("#### Behind the answer")
    if not st.session_state.turns:
        st.info("The model plans and narrates. A deterministic rules engine does "
                "every calculation. The plan, the rules evaluated and the records "
                "consulted appear here — and every identifier in the answer "
                "carries a citation back to its record.")
        with st.expander("Kernel check — no LLM involved", expanded=True):
            try:
                d = api_post("/tools/recommend_options",
                             {"args": {"crew_id": "C-1042", "pairing_id": "P-2291",
                                       "from_date": "2026-09-15"}})["data"]
                st.caption(f"Cover Captain on P-2291 · {d['covering']['seats']} seats · "
                           f"{d['excluded_count']} candidates excluded")
                st.dataframe([{"rank": o["rank"], "action": o["action"][:42],
                               "cost": f"INR {o['cost_inr']:,}",
                               "delay h": o["delay_hours"], "risk": o["risk_score"]}
                              for o in d["options"][:6]],
                             hide_index=True, width="stretch")
            except Exception as e:                  # noqa: BLE001
                st.caption(f"kernel call failed: {e}")
    else:
        t = st.session_state.turns[-1]
        tabs = st.tabs(["Plan", "Rules checked", "Evidence", "Execution"])

        with tabs[0]:
            plan = t.get("plan_detail")
            if not plan:
                st.caption("No plan was produced.")
            else:
                st.markdown(f"**intent** `{plan['intent']}` — planned by "
                            f"`{t.get('plan_source')}`"
                            + (" (zero tokens)" if t.get("plan_source") == "fastpath" else ""))
                if plan.get("entities"):
                    st.json(plan["entities"], expanded=False)
                for s in plan["steps"]:
                    dep = f" ← {', '.join(s['depends_on'])}" if s.get("depends_on") else ""
                    st.code(f"{s['id']}: {s['operation']}"
                            f"({json.dumps(s['args'], default=str)}){dep}", language=None)
                v = t.get("validation_detail")
                if v:
                    st.markdown("**validator** " + ("✅ accepted" if v["ok"] else "❌ rejected"))
                    for r in v["repaired"]:
                        st.caption(f"repaired — {r}")
                    for i in v["issues"]:
                        st.caption(f":red[{i.get('step_id') or 'plan'}: {i['message']}]")
                u = t.get("usage") or {}
                if u.get("calls"):
                    st.caption(f"{u['calls']} model calls · {u['total_tokens']} tokens · "
                               f"{', '.join(m.split('/')[-1] for m in u.get('models', []))}")

        with tabs[1]:
            rules = t.get("rules_evaluated") or []
            if not rules:
                st.caption("No rule evaluation was needed for that answer.")
            else:
                seen, rows = set(), []
                for r in rules:
                    k = (r["rule_id"], r.get("on_date"), r.get("actual"))
                    if k in seen:
                        continue
                    seen.add(k)
                    rows.append({"": "✅" if r["passed"] else "❌", "rule": r["rule_id"],
                                 "date": r.get("on_date") or "", "limit": r.get("limit"),
                                 "actual": r.get("actual"), "margin": r.get("margin"),
                                 "detail": r["detail"]})
                rows.sort(key=lambda x: (x[""] == "✅", x["rule"]))
                st.dataframe(rows, hide_index=True, width="stretch")
                st.caption("margin = headroom remaining; negative means the limit is exceeded")

        with tabs[2]:
            evidence = t.get("evidence") or []
            if not evidence:
                st.caption("No records cited.")
            else:
                cited = {c["ref"] for c in (t.get("citations") or [])}
                by_file: dict = {}
                for e in evidence:
                    f, _, rid = e.partition("#")
                    by_file.setdefault(f, []).append((rid, e in cited))
                for f, ids in sorted(by_file.items()):
                    n = sum(1 for _, c in ids if c)
                    st.markdown(f"**{f}** · {len(ids)} read"
                                + (f", {n} cited in the answer" if n else ""))
                    chips = "".join(
                        f"<span class='cite-chip' style='"
                        f"{'font-weight:600;border-color:#2e7d32' if c else 'opacity:.55'}'>"
                        f"{rid}</span>" for rid, c in ids[:40])
                    st.markdown(chips + (" …" if len(ids) > 40 else ""),
                                unsafe_allow_html=True)
            if t.get("unsupported"):
                st.error("Values the model wrote that no tool returned: "
                         + ", ".join(u["token"] for u in t["unsupported"]))

        with tabs[3]:
            steps = t.get("steps") or []
            if not steps:
                st.caption("Nothing was executed.")
            else:
                st.caption(f"kernel total {t.get('kernel_ms', 0):.2f} ms")
                results = t.get("results") or {}
                for s in steps:
                    icon = "✅" if s["ok"] else "❌"
                    with st.expander(f"{icon} `{s['operation']}` · "
                                     f"{s['duration_ms']:.2f} ms · "
                                     f"{s['evidence_count']} records"):
                        st.json(s["args"], expanded=False)
                        if s.get("error"):
                            st.error(s["error"])
                        elif s["id"] in results:
                            st.json(results[s["id"]], expanded=False)
