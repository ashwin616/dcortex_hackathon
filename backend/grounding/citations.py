"""Citations, derived rather than requested.

The obvious way to cite is to ask the model for footnotes. That reintroduces
exactly the failure this system is built to avoid: a citation the model made up
is worse than no citation, because it looks like proof.

So citations are computed. The grounding validator already proves every
identifier in the answer came from a tool result; this walks the same ground
once more and records *which* result, so each `C-3310`, `P-2291`, `DX412` and
`RULE-DUTY-02` in the prose can be traced to the record or the rule evaluation
it came from. Zero extra tokens, and nothing to hallucinate.
"""
from __future__ import annotations

import json
import re

from backend.grounding.validator import ID_PATTERNS, normalize
from backend.text import plain

# Order matters: the longest identifiers are matched first so "DX412-2026-09-15"
# is not shadowed by the bare "DX412" inside it.
_ORDER = ("flight", "crew_id", "pairing_id", "tail", "rule_id")

FILE_LABEL = {
    "crew.json": "crew record",
    "rosters.json": "roster",
    "flights.json": "schedule",
    "duty_clocks.json": "duty clock",
    "reserve_pool.json": "reserve pool",
    "certifications.json": "certifications",
    "risk_signals.json": "risk signal",
    "costs.json": "cost rates",
    "rules.json": "rulebook",
}


class Citation:
    """One source, and the tokens in the answer that rest on it."""

    def __init__(self, index: int, ref: str):
        self.index = index
        self.ref = ref
        self.file, _, self.record = ref.partition("#")
        self.tokens: list = []
        self.detail: str | None = None

    @property
    def label(self) -> str:
        return f"{FILE_LABEL.get(self.file, self.file)} {self.record}".strip()

    def to_dict(self) -> dict:
        return {"index": self.index, "ref": self.ref, "file": self.file,
                "record": self.record, "label": plain(self.label),
                "tokens": sorted(set(self.tokens)),
                "detail": plain(self.detail) if self.detail else None}


def _rule_detail(rules: list, rule_id: str) -> str | None:
    """The arithmetic behind a rule, preferring a breach over a pass."""
    hits = [r for r in rules if r.get("rule_id") == rule_id]
    if not hits:
        return None
    worst = min(hits, key=lambda r: (r.get("passed", True),
                                     r.get("margin") if r.get("margin") is not None else 0))
    parts = [worst.get("detail", "")]
    if worst.get("limit") is not None:
        parts.append(f"limit {worst['limit']}, actual {worst.get('actual')}, "
                     f"margin {worst.get('margin')}")
    if worst.get("window"):
        parts.append(f"window {worst['window']}")
    return plain(" · ".join(p for p in parts if p))


def build(answer: str, results: list, rules_evaluated: list | None = None) -> dict:
    """Map every identifier in `answer` to the tool results that contain it.

    Returns {"citations": [...], "annotated": markdown-with-superscripts,
             "uncited": [tokens we could not source]}.
    """
    rules_evaluated = rules_evaluated or []
    text = normalize(answer or "")

    # ref -> Citation, in first-appearance order
    by_ref: dict = {}
    token_refs: dict = {}

    payloads = []
    for r in results:
        d = r.to_dict() if hasattr(r, "to_dict") else r
        payloads.append((d.get("evidence") or [], json.dumps(d, default=str)))

    # Collect every identifier with its position, longest-pattern first so a
    # dated flight id is not shadowed by the bare flight number inside it, then
    # process in reading order so citation [1] is the first one the eye meets.
    spans: list = []
    claimed: list = []
    for kind in _ORDER:
        for m in ID_PATTERNS[kind].finditer(text):
            if any(a <= m.start() < b for a, b in claimed):
                continue
            claimed.append((m.start(), m.end()))
            spans.append((m.start(), m.group(0), kind))
    spans.sort()

    for _, token, kind in spans:
        if token in token_refs:
            continue

        refs: list = []
        if kind == "rule_id":
            refs = [f"rules.json#{token}"]
        else:
            for evidence, blob in payloads:
                if token not in blob:
                    continue
                exact = [e for e in evidence if e.endswith(f"#{token}")]
                refs.extend(exact or [e for e in evidence
                                      if token.split("-2026")[0] in e][:1])
        uniq = list(dict.fromkeys(refs))          # de-duplicate, keep order
        if not uniq:
            token_refs[token] = []
            continue
        idxs = []
        for ref in uniq[:3]:
            cite = by_ref.get(ref)
            if cite is None:
                cite = by_ref[ref] = Citation(len(by_ref) + 1, ref)
                if cite.file == "rules.json":
                    cite.detail = _rule_detail(rules_evaluated, cite.record)
            cite.tokens.append(token)
            idxs.append(cite.index)
        token_refs[token] = idxs

    annotated = _annotate(answer, token_refs, by_ref)
    return {
        "citations": [c.to_dict() for c in sorted(by_ref.values(), key=lambda c: c.index)],
        "annotated": annotated,
        "uncited": sorted(t for t, refs in token_refs.items() if not refs),
    }


def _annotate(answer: str, token_refs: dict, by_ref: dict) -> str:
    """Insert a superscript marker after the first mention of each identifier.

    Only the first, and never inside a table cell - a `[1]` in every row of a
    six-column table is noise, and noise is what makes people stop reading the
    evidence at all.
    """
    if not token_refs:
        return answer
    tips = {c.index: (c.detail or c.label) for c in by_ref.values()}
    used: set = set()
    out_lines = []
    for line in (answer or "").splitlines():
        if line.lstrip().startswith("|"):
            out_lines.append(line)
            continue
        norm = normalize(line)
        replacements: list = []
        for kind in _ORDER:
            for m in ID_PATTERNS[kind].finditer(norm):
                token = m.group(0)
                idxs = token_refs.get(token)
                if not idxs or token in used:
                    continue
                used.add(token)
                marker = "".join(
                    f'<sup title="{_esc(tips.get(i, ""))}">[{i}]</sup>' for i in idxs)
                replacements.append((m.end(), marker))
        for pos, marker in sorted(replacements, reverse=True):
            line = line[:pos] + marker + line[pos:]
        out_lines.append(line)
    return "\n".join(out_lines)


def _esc(s: str) -> str:
    return (s or "").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
