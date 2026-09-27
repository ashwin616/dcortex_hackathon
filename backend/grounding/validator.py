"""Grounding validator.

The dataset brief says invented facts are failures, not rounding errors. So
after the model writes its answer we pull every identifier and every
consequential number back out of the text and require each one to appear in
what the tools actually returned.

Deliberately asymmetric: identifiers are checked strictly, numbers only when
they are large enough (money) or precise enough (hours) to matter. Checking
"3 flights" against the corpus produces noise, not safety.
"""
from __future__ import annotations

import json
import re

ID_PATTERNS = {
    "crew_id": re.compile(r"\bC-\d{3,4}\b"),
    "pairing_id": re.compile(r"\bP-\d{3,4}\b"),
    "flight": re.compile(r"\bDX\d{3}(?:-\d{4}-\d{2}-\d{2})?\b"),
    "rule_id": re.compile(r"\bRULE-[A-Z]+-\d{2}\b"),
    "tail": re.compile(r"\bVT-DX[A-Z]\b"),
}
# A trailing unit letter must not end the match: with `(?![\w-])` the string
# "23.5h" matched only "23" - so the validator checked 23.0 against a corpus
# containing 23.5, and a fabricated hours figure written as "61.4h" would
# have been compared as "61". Only a digit, comma or dot may continue a
# number - a digit, a comma before digits, or a decimal point before
# digits. A unit suffix or a full stop ends it cleanly.
NUM = re.compile(r"(?<![\w.-])(\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+\.\d+|\d{4,})(?!\d|,\d|\.\d)")

# Numbers that are constants of the rulebook or the clock, not claims about data.
SAFE_NUMBERS = {12.0, 13.0, 60.0, 100.0, 28.0, 7.0, 30.0, 0.5, 24.0, 2026.0}


# Models routinely emit typographic dashes (U+2010-2015, U+2212) inside
# identifiers - "P‑2291" instead of "P-2291". Without normalising, the id
# pattern misses and the bare digits then look like a fabricated number.
_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―−"})


def normalize(text: str) -> str:
    return (text or "").translate(_DASHES)


def _numbers(text: str) -> set[float]:
    out = set()
    for m in NUM.finditer(text):
        try:
            out.add(float(m.group(1).replace(",", "")))
        except ValueError:
            pass
    return out


# A margin is stored negative (-1.33) and read positive ("exceeds by 1h20m",
# "over by 1.33h"). The magnitude of every negative figure therefore counts as
# support, or every breach the system correctly reports looks fabricated.
NEG = re.compile(r"(?<![\w.])-(\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+\.\d+|\d{4,})"
                 r"(?!\d|,\d|\.\d)")


def supported_numbers(text: str) -> set:
    """Every figure the tool output can vouch for, sign-insensitive."""
    found = _numbers(text)
    for m in NEG.finditer(text):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        found.add(v)
        found.add(-v)
    return found


def corpus(tool_results: list) -> str:
    """Everything the deterministic layer returned this turn, as one string."""
    parts = []
    for r in tool_results:
        d = r.to_dict() if hasattr(r, "to_dict") else r
        parts.append(json.dumps(d, default=str))
    return "\n".join(parts)


def check(answer: str, tool_results: list, number_floor: float = 1000.0):
    """Return (ok, unsupported) for an answer against this turn's tool output."""
    answer = normalize(answer)
    hay = normalize(corpus(tool_results))
    hay_nums = supported_numbers(hay)
    unsupported = []

    for kind, pat in ID_PATTERNS.items():
        for token in set(pat.findall(answer)):
            if token in hay:
                continue
            # a bare flight number is supported by any dated instance of it
            if kind == "flight" and re.search(rf"\b{re.escape(token)}-\d{{4}}", hay):
                continue
            unsupported.append({"token": token, "kind": kind})

    for n in _numbers(answer):
        if n in SAFE_NUMBERS or n in hay_nums:
            continue
        is_money = n >= number_floor
        is_precise = n != int(n)
        if not (is_money or is_precise):
            continue
        # tolerate rounding of an hours figure that is present at more precision
        if any(abs(n - h) < 0.02 for h in hay_nums):
            continue
        unsupported.append({"token": f"{n:g}", "kind": "number"})

    return (not unsupported), unsupported


def correction_prompt(unsupported: list) -> str:
    toks = ", ".join(f"{u['token']} ({u['kind']})" for u in unsupported)
    return (
        "GROUNDING CHECK FAILED. These appear in your answer but not in any tool "
        f"result this turn: {toks}.\n"
        "Rewrite the answer using only identifiers and figures present in the "
        "result. Most often this is a figure you calculated - a difference "
        "between two costs, a total, a percentage. Do not calculate: quote the "
        "figures that are there and drop the derived one. If a fact you wanted "
        "is genuinely absent, say plainly that you cannot answer that part "
        "reliably. Do not restate the unsupported values."
    )
