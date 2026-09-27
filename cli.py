"""Crew Ops Advisor - terminal interface.

    python cli.py                          interactive
    python cli.py "who is on reserve at BLR tomorrow?"
    python cli.py --kernel                 deterministic layer only, no API key

In-session commands:
    /plan       the execution plan and how it was validated
    /rules      rule arithmetic behind the last answer
    /evidence   records the last answer was built from
    /trace      operations executed, with timings
    /capacity   live token budget across the model pool
    /reset      clear conversation memory
    /quit
"""
from __future__ import annotations

import json
import sys

from backend.advisor import Advisor
from backend.domain.costing import inr
from backend.domain.data import store

try:
    # Windows consoles default to cp1252, which cannot represent what the model
    # writes; without this the answer arrives as mojibake even when the text
    # itself is correct.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BOLD, DIM, GREEN, RED, YELLOW, CYAN, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m")


def banner():
    s = store()
    print(f"{BOLD}Crew Ops Advisor{RESET} - dCortex Air")
    print(f"{DIM}snapshot {s.snapshot_utc} | {len(s.flights)} flights, "
          f"{len(s.crew)} crew, {len(s.pairings)} pairings | all times UTC{RESET}")
    print(f"{DIM}ask a question, or /help{RESET}\n")


def render(turn):
    if turn.error and not turn.context:
        print(f"{RED}{turn.answer}{RESET}\n")
        return
    print(f"{YELLOW if turn.abstained else ''}{turn.answer}{RESET}\n")

    bits = []
    if turn.plan:
        tag = {"fastpath": GREEN, "cache": CYAN}.get(turn.plan_source, "")
        bits.append(f"{tag}{turn.plan_source}{RESET}: {turn.plan.summary()}")
    breaches = [r for r in turn.rules_evaluated if not r["passed"]]
    if turn.rules_evaluated:
        bits.append(f"{len(turn.rules_evaluated)} rule checks"
                    + (f", {RED}{len(breaches)} breach{RESET}" if breaches else ", all pass"))
    if turn.evidence:
        bits.append(f"{len(turn.evidence)} records")
    bits.append(f"kernel {turn.kernel_ms:.1f}ms")
    if turn.throttled_s > 0.5:
        bits.append(f"{turn.latency_s - turn.throttled_s:.1f}s + "
                    f"{turn.throttled_s:.0f}s throttle")
    else:
        bits.append(f"{turn.latency_s:.1f}s")
    u = turn.usage
    bits.append("0 tokens" if not u.calls else f"{u.total} tok / {u.calls} calls")
    if turn.cached:
        bits.append(f"{CYAN}cached{RESET}")
    if not turn.grounded:
        bits.append(f"{RED}ungrounded{RESET}")
    print(f"{DIM}[{' | '.join(bits)}]  /plan /rules /evidence /trace{RESET}\n")


def show_plan(turn):
    if not turn.plan:
        print(f"{DIM}no plan was produced{RESET}\n")
        return
    print(f"{BOLD}intent{RESET} {turn.plan.intent}   {BOLD}source{RESET} {turn.plan_source}")
    if turn.plan.entities:
        print(f"{BOLD}entities{RESET} {json.dumps(turn.plan.entities, default=str)}")
    for s in turn.plan.steps:
        dep = f"  after {', '.join(s.depends_on)}" if s.depends_on else ""
        print(f"  {s.id}: {BOLD}{s.operation}{RESET}"
              f"({json.dumps(s.args, default=str)}){dep}")
    v = turn.validation
    if v:
        print(f"{BOLD}validator{RESET} {GREEN + 'accepted' + RESET if v.ok else RED + 'rejected' + RESET}")
        for r in v.repaired:
            print(f"  {DIM}repaired: {r}{RESET}")
        for i in v.issues:
            print(f"  {RED}{i.step_id or 'plan'}: {i.message}{RESET}")
    print()


def show_rules(turn):
    if not turn.rules_evaluated:
        print(f"{DIM}no rules were evaluated for that answer{RESET}\n")
        return
    print(f"{BOLD}{'rule':<16}{'date':<12}{'limit':>8}{'actual':>9}{'margin':>9}  detail{RESET}")
    seen = set()
    for r in turn.rules_evaluated:
        k = (r["rule_id"], r.get("on_date"), r.get("actual"))
        if k in seen:
            continue
        seen.add(k)
        mark = GREEN + "ok    " + RESET if r["passed"] else RED + "BREACH" + RESET
        lim = "" if r.get("limit") is None else f"{r['limit']:.2f}"
        act = "" if r.get("actual") is None else f"{r['actual']:.2f}"
        mar = "" if r.get("margin") is None else f"{r['margin']:+.2f}"
        print(f"{r['rule_id']:<16}{(r.get('on_date') or ''):<12}{lim:>8}{act:>9}{mar:>9}  "
              f"{mark} {r['detail'][:78]}")
    print()


def show_evidence(turn):
    if not turn.evidence:
        print(f"{DIM}no records cited{RESET}\n")
        return
    by_file: dict = {}
    for e in turn.evidence:
        f, _, rid = e.partition("#")
        by_file.setdefault(f, []).append(rid)
    for f, ids in sorted(by_file.items()):
        shown = ", ".join(ids[:14]) + (f" ... (+{len(ids) - 14})" if len(ids) > 14 else "")
        print(f"  {CYAN}{f}{RESET}  {shown}")
    print()


def show_trace(turn):
    if not turn.context:
        print(f"{DIM}nothing was executed{RESET}\n")
        return
    for i, s in enumerate(turn.context.steps, 1):
        mark = GREEN + "ok" + RESET if s.ok else RED + "fail" + RESET
        args = ", ".join(f"{k}={v!r}" for k, v in s.args.items())
        print(f"  {i}. {BOLD}{s.operation}{RESET}({args})  {mark} "
              f"{s.duration_ms:.2f}ms, {s.evidence_count} records")
        if s.error:
            print(f"     {RED}{s.error}{RESET}")
        else:
            body = json.dumps(turn.context.results[s.id].data, default=str)
            print(f"     {DIM}{body[:180]}{'...' if len(body) > 180 else ''}{RESET}")
    print()


def show_capacity():
    c = Advisor.capacity()
    print(f"{BOLD}pool{RESET} {c['aggregate_tpm']} TPM aggregate, "
          f"{c['headroom_now']} available now, {c['model_switches']} switches, "
          f"{c['pool_waited_s']}s waited")
    for m in c["models"]:
        print(f"  {m['model']:28} calls {m['calls']:>4}  billed {m['tokens_billed']:>7}  "
              f"headroom {m['headroom']:>6}  reqs left {m['remaining_requests']}")
    print(f"{BOLD}cache{RESET} plan {c['cache']['plan_cache']} answer {c['cache']['answer_cache']}\n")


def kernel_demo():
    """Deterministic layer only - proves the numbers without an API key."""
    from backend import tools as T
    print(f"{DIM}kernel mode: no LLM, calling the tool layer directly{RESET}\n")
    r = T.call("recommend_options", crew_id="C-1042", pairing_id="P-2291",
               from_date="2026-09-15")
    d = r.data
    print(f"{BOLD}Cover {d['covering']['role']} on {d['covering']['pairing_id']}"
          f" ({', '.join(d['covering']['dates'])}) - {d['covering']['seats']} seats{RESET}")
    for o in d["options"][:5]:
        tag = GREEN + "legal" + RESET if o["legal"] else RED + "illegal" + RESET
        print(f"  {o['rank']}. {o['action'][:62]:<62} {inr(o['cost_inr']):>14}  {tag}")
        print(f"     {DIM}{o['reasoning'][:100]}{RESET}")
    print(f"\n{DIM}{d['excluded_count']} candidates excluded. First few:{RESET}")
    for e in d["excluded_examples"][:3]:
        print(f"  {e['crew_id']}: {e['reason'][:96]}")
    print()


COMMANDS = {"/plan": show_plan, "/rules": show_rules,
            "/evidence": show_evidence, "/trace": show_trace}


def main():
    args = list(sys.argv[1:])
    if "--kernel" in args:
        kernel_demo()
        return 0

    advisor = Advisor()
    if args:
        turn = advisor.ask(" ".join(args), keep_history=False)
        render(turn)
        return 1 if turn.error else 0

    banner()
    last = None
    while True:
        try:
            q = input(f"{BOLD}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        low = q.lower()
        if low in ("/quit", "/exit", "/q"):
            return 0
        if low == "/help":
            print(__doc__)
            continue
        if low == "/reset":
            advisor.reset()
            print(f"{DIM}conversation cleared{RESET}\n")
            continue
        if low == "/capacity":
            show_capacity()
            continue
        if low in COMMANDS:
            if not last:
                print(f"{DIM}ask something first{RESET}\n")
                continue
            COMMANDS[low](last)
            continue
        last = advisor.ask(q)
        render(last)


if __name__ == "__main__":
    raise SystemExit(main())
