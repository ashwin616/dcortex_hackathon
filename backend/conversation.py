"""The conversational edges: greetings, thanks, and "what can you do?".

These are not crew-control questions, so the planner correctly finds no
operation for them - and then refuses, which is the wrong response to "hi". A
controller opening the tool for the first time gets told the system cannot help,
and a judge's first message is very often "what can you do?".

So they are handled here, deterministically, before the planner runs: no tokens,
no latency, and - importantly - the capability answer is generated from the live
operation registry rather than written by hand, so it cannot drift from what the
system actually does or promise something it does not.

Deliberately narrow. Anything naming a crew member, pairing, flight, station or
date is a real question and falls straight through, so "hi, is C-1042 free?" is
answered rather than waved at.
"""
from __future__ import annotations

import re
from datetime import timedelta

from backend.domain.data import store
from backend.executor.registry import OPERATIONS

GREETING = re.compile(
    r"^\s*(hi|hey|hello|hiya|yo|good\s+(morning|afternoon|evening|day)|"
    r"howdy|namaste|greetings)\b[\s!.,?]*$", re.I)
THANKS = re.compile(r"^\s*(thanks|thank\s+you|thx|ta|cheers|nice|great|perfect|"
                    r"brilliant|awesome|ok|okay|got\s+it)\b[\s!.,?]*$", re.I)
FAREWELL = re.compile(r"^\s*(bye|goodbye|see\s+you|later|good\s*night|cya)\b[\s!.,?]*$", re.I)
CONTEXT = re.compile(
    r"\b(what(?:'s| is)?\s+(?:the\s+)?(?:today'?s?\s+)?(?:date|day|time)|"
    r"what\s+day\s+is\s+it|what\s+time\s+is\s+it|today'?s?\s+date|"
    r"current\s+(?:date|time)|what\s+snapshot|which\s+week|"
    r"what\s+week|when\s+are\s+we)\b", re.I)
CAPABILITY = re.compile(
    r"\b(what\s+can\s+you\s+do|what\s+do\s+you\s+do|how\s+can\s+you\s+help|"
    r"what\s+are\s+you|who\s+are\s+you|help\b|capabilities|commands|"
    r"what\s+can\s+i\s+ask|how\s+does\s+this\s+work|examples?)\b", re.I)

# If any of these appear the message is a real question, whatever else it says.
DOMAIN_SIGNAL = re.compile(
    r"\bC-\d{3,4}\b|\bP-\d{3,4}\b|\bDX\d{3}\b|\bVT-DX[A-Z]\b|"
    r"\b(BLR|DEL|BOM|MAA|HYD|CCU|COK|GOI)\b|\b20\d{2}-\d{2}-\d{2}\b|"
    r"\breserve|duty|roster|pairing|flight|crew|sick|legal|cover|delay|closure",
    re.I)

MAX_SOCIAL_WORDS = 8


def _capabilities() -> str:
    """Built from the registry, so it always matches what is really wired up."""
    s = store()
    by_tier: dict = {}
    for op in OPERATIONS.values():
        by_tier.setdefault(op.tier, []).append(op)
    label = {1: "Look things up", 2: "Work out consequences", 3: "Tell you what to do"}
    example = {
        1: "“Who's on reserve at BLR on 15 Sep?” · “How many duty hours has C-1042 used?”",
        2: "“C-1042 is sick on 15 Sep — which flights are uncrewed?” · "
           "“If C-2087 covers P-2291, does anything breach?”",
        3: "“C-1042 is out for P-2291 — what should I do?” · "
           "“Draft the callout to C-3310.”",
    }
    lines = [
        "I'm the Crew Ops Advisor for dCortex Air — ask me in plain English.",
        "",
        f"I'm working from the snapshot at **{s.snapshot_utc}**: "
        f"{len(s.flights)} flights, {len(s.crew)} crew and {len(s.pairings)} pairings "
        f"across {s.dates[0]} to {s.dates[-1]}. All times UTC, money in INR.",
        "",
    ]
    for tier in sorted(by_tier):
        lines.append(f"**{label.get(tier, tier)}** — {len(by_tier[tier])} operations")
        lines.append(f"{example.get(tier, '')}")
        lines.append("")
    lines += [
        "Every number comes from a deterministic rules engine, never from me — "
        "I plan the query and write the answer, and each figure carries a "
        "citation back to the record it came from.",
        "",
        "If the data can't answer something, I'll say so rather than guess.",
    ]
    return "\n".join(lines)


def handle(question: str) -> tuple[str, str] | None:
    """Return (answer, intent) for a social message, or None to carry on.

    None means "this is a real question" - the default, and the safe one.
    """
    text = (question or "").strip()
    if not text:
        return None

    asks_capability = bool(CAPABILITY.search(text))
    asks_context = bool(CONTEXT.search(text))

    # "What time does the P-2204 crew report on 17 Sep?" matched the context
    # pattern on the words "what time" and was answered with today's date -
    # the most annoying possible failure, because the question was perfectly
    # answerable. A question naming a pairing, a flight, a crew member or a
    # date is about the operation, never about what day it is, so the domain
    # signal vetoes the social route outright rather than only gating it.
    if DOMAIN_SIGNAL.search(text):
        return None

    if not (asks_capability or asks_context):
        # A greeting stops being a greeting the moment it carries a question.
        if len(text.split()) > MAX_SOCIAL_WORDS:
            return None

    if asks_context and not asks_capability:
        # The snapshot was in the planner prompt all along, and the system still
        # refused "what is today's date?" because no operation returned it.
        # Anything the advisor reasons *from* should be answerable *about*.
        s = store()
        today = s.snapshot.date()
        tomorrow = (today + timedelta(days=1)).isoformat()
        return ("\n".join([
            f"**Today is {today.isoformat()} ({today.strftime('%A')})** — the "
            f"operating snapshot is `{s.snapshot_utc}`.",
            "",
            f"The schedule week runs {s.dates[0]} to {s.dates[-1]}. "
            "All times are UTC, money is INR.",
            "",
            "This is a fixed snapshot, not wall-clock time: the dataset is a "
            f"planned week frozen at that instant, so \"tomorrow\" means "
            f"{tomorrow}.",
        ]), "context")
    if asks_capability:
        return _capabilities(), "capability"
    if GREETING.match(text):
        s = store()
        return (f"Hello — Crew Ops Advisor here, on the "
                f"{s.snapshot_utc[:10]} snapshot.\n\n"
                "Ask me about crew, flights, duty limits or a disruption. "
                "For example: *“Who's on reserve at BLR on 15 Sep?”* or "
                "*“C-1042 is out for P-2291 — what should I do?”*\n\n"
                "Say **what can you do** for the full picture."), "greeting"
    if THANKS.match(text):
        return "Any time. What else do you need?", "thanks"
    if FAREWELL.match(text):
        return "Cheers — I'll keep the rulebook warm.", "farewell"
    return None
