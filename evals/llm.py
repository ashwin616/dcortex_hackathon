"""End-to-end evaluation - runs questions through the full chat stack.

The kernel eval proves the arithmetic. This one measures the layer above it:
did the model pick the right tool, and did the facts survive into the prose?

Scoring is deliberately blunt. For each question we pull the identifiers and
the consequential numbers out of the shipped answer key and check they appear
in the Advisor's reply. That under-credits a good answer phrased differently,
which is the right direction to be wrong in.

    python eval.py llm                 all 38 questions
    python eval.py llm --tier 2        one tier
    python eval.py llm --n 6           first 6
    python eval.py llm --planner qwen38 --narrator 120b
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import re
import sys
import time
from pathlib import Path

from backend.domain.data import store
from backend.grounding.validator import ID_PATTERNS, _numbers, normalize
from backend.advisor import Advisor
from backend.config import BY_ALIAS

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = Path(__file__).resolve().parent.parent / "samples"


def key_facts(expected) -> tuple[set, set, set]:
    """What a correct answer should mention, split by how much it matters.

    Operational facts - crew, flights, pairings, costs, hours - are what a
    controller acts on, and omitting one is a real miss. Rule identifiers are
    scored separately: an answer that says "would exceed the 60h/7d limit by
    1h20m" has conveyed RULE-DUTY-02 perfectly well without naming it, and the
    shipped keys list all seven `rules_checked` on every option, which no
    readable answer recites.
    """
    blob = json.dumps(expected, default=str)
    rules = set(ID_PATTERNS["rule_id"].findall(blob))
    ids = set()
    for name, pat in ID_PATTERNS.items():
        if name != "rule_id":
            ids |= set(pat.findall(blob))

    # A timestamp is a fact, but "2026" is not the fact - the time of day is.
    # Left in, the year of every ISO date became a number the answer had to
    # recite, and "earliest report 03:30Z on 17 Sep" scored zero for being
    # written the way a controller would say it.
    for ts in re.findall(r"\d{4}-\d{2}-\d{2}T(\d{2}:\d{2})", blob):
        ids.add(ts)
    blob = re.sub(r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?Z?)?", " ", blob)

    nums = {n for n in _numbers(blob) if n >= 1000 or n != int(n)}
    return ids, nums, rules


def recall(answer: str, ids: set, nums: set) -> tuple[float, list]:
    # models emit typographic dashes inside identifiers; normalise before
    # matching or every "C-1042" reads as a miss
    answer = normalize(answer)
    ans_nums = _numbers(answer)
    missing = []
    hit = 0
    total = len(ids) + len(nums)
    for i in ids:
        if i in answer or re.search(rf"\b{re.escape(i.split('-2026')[0])}\b", answer):
            hit += 1
        else:
            missing.append(i)
    for n in nums:
        if any(abs(n - a) < 0.011 for a in ans_nums):
            hit += 1
        else:
            missing.append(f"{n:g}")
    return (hit / total if total else 1.0), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=None)
    ap.add_argument("--set", dest="qset", default="core",
                    choices=["core", "user", "all"],
                    help="core = questions.json, user = the user-perspective set, all = both")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--ids", default=None,
                    help="comma-separated question ids, e.g. Q31,Q32")
    ap.add_argument("--planner", default="20b")
    ap.add_argument("--narrator", default="120b")
    ap.add_argument("--no-fastpath", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0, help="pause between calls (rate limits)")
    ap.add_argument("--workers", type=int, default=1,
                    help="run questions concurrently; the model pool absorbs the load")
    a = ap.parse_args()

    s = store()
    pool = list(s.questions)
    if a.qset in ("user", "all"):
        extra = json.loads(
            (s.dir / "questions_user_perspective.json").read_text(encoding="utf-8"))
        rows_extra = extra["questions"] if isinstance(extra, dict) else extra
        pool = rows_extra if a.qset == "user" else pool + rows_extra
    qs = [q for q in pool if a.tier is None or q["tier"] == a.tier]
    if a.ids:
        want = {x.strip().upper() for x in a.ids.split(",")}
        qs = [q for q in qs if q["question_id"].upper() in want]
    if a.n:
        qs = qs[:a.n]

    advisor = Advisor(planner_model=a.planner, narrator_model=a.narrator,
                      use_cache=not a.no_cache, use_fastpath=not a.no_fastpath)
    rows, t0 = [], time.time()
    per_tier: dict = {}

    for q in qs:
        turn = advisor.ask(q["prompt"], keep_history=False)
        ids, nums, rules = key_facts(q["expected_answer"])
        score, missing = recall(turn.answer, ids, nums)
        rule_score, rule_missing = recall(turn.answer, rules, set())
        rows.append({**q, "answer": turn.answer, "recall": round(score, 3),
                     "rule_recall": round(rule_score, 3), "rule_missing": rule_missing,
                     "missing": missing, "tools": [c["tool"] for c in turn.tool_calls],
                     "plan_source": turn.plan_source,
                     "usage": turn.usage.to_dict(),
                     "kernel_ms": turn.kernel_ms,
                     "grounded": turn.grounded, "abstained": turn.abstained,
                     "latency_s": round(turn.latency_s, 2),
                     "compute_s": round(turn.latency_s - turn.throttled_s, 2),
                     "throttled_s": round(turn.throttled_s, 2), "error": turn.error,
                     "evidence": turn.evidence[:20],
                     "rules_evaluated": turn.rules_evaluated,
               "open_ended": bool(q.get("grading_note"))})
        if not q.get("grading_note"):
            per_tier.setdefault(q["tier"], []).append(score)

        open_ended = bool(q.get("grading_note"))
        mark = "OK  " if score >= 0.999 else ("PART" if score >= 0.5 else "MISS")
        if open_ended:
            mark = "OPEN"
        if turn.error:
            mark = "ERR "
        print(f"[{mark}] T{q['tier']} {q['question_id']} r={score:.2f} "
              f"rr={rule_score:.2f} "
              f"{turn.latency_s - turn.throttled_s:4.1f}s  "
              f"{','.join(c['tool'] for c in turn.tool_calls) or '-'}", flush=True)
        if missing and not turn.error:
            print(f"        missing from answer: {', '.join(map(str, missing[:8]))}", flush=True)
        if turn.error:
            print(f"        {turn.error}", flush=True)
        if a.sleep:
            time.sleep(a.sleep)

    OUT.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = OUT / f"llm-eval-{a.qset}-{a.planner}+{a.narrator}-{stamp}.json"
    path.write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")

    print("\n" + "-" * 58)
    n_open = sum(1 for r in rows if r.get("open_ended"))
    if n_open:
        print(f"({n_open} open-ended rows scored by grading note, excluded from recall)")
    for tier in sorted(per_tier):
        v = per_tier[tier]
        full = sum(1 for x in v if x >= 0.999)
        print(f"Tier {tier}: mean fact recall {sum(v)/len(v):.2f}  "
              f"({full}/{len(v)} complete)")
    rr = [r["rule_recall"] for r in rows]
    print(f"rule-id recall (secondary): {sum(rr)/len(rr):.2f}")
    grounded = sum(1 for r in rows if r["grounded"])
    abst = sum(1 for r in rows if r["abstained"])
    errs = sum(1 for r in rows if r["error"])
    lat = sorted(r["compute_s"] for r in rows if not r["error"])
    thr = sum(r["throttled_s"] for r in rows)
    print(f"grounded {grounded}/{len(rows)} · abstained {abst} · errors {errs}")
    if lat:
        print(f"compute latency: median {lat[len(lat)//2]:.1f}s, "
              f"p90 {lat[int(len(lat)*0.9)]:.1f}s, max {lat[-1]:.1f}s")
        print(f"free-tier rate-limit wait excluded above: {thr:.0f}s total")
    tok = sum(r["usage"]["total_tokens"] for r in rows)
    src = {}
    for r in rows:
        src[r["plan_source"]] = src.get(r["plan_source"], 0) + 1
    print(f"plan source: " + ", ".join(f"{k}={v}" for k, v in sorted(src.items())))
    print(f"tokens: {tok} total, {tok // max(1, len(rows))} per question")
    cap = Advisor.capacity()
    print(f"pool: {cap['aggregate_tpm']} TPM aggregate across "
          f"{len(cap['models'])} models, {cap['model_switches']} switches")
    print(f"cache: {cap['cache']}")
    print(f"total {time.time()-t0:.0f}s · written to {path}")
    return 0


