"""Time, resolved properly.

Dates were being pulled out of questions with a handful of regexes bolted on as
gaps appeared, and it showed: "yesterday" was not recognised at all, "between
yesterday and tomorrow" silently collapsed to tomorrow, and "from 14 to 16 Sep"
answered for the 16th. Each was found by a person asking an ordinary question.

This module does the job once. Everything temporal - a day, a range, a time of
day, a phrase like "the rest of the week" - resolves here, against the snapshot,
with the dataset's own window as the boundary. Two properties matter:

  * a range stays a range. Narrowing a three-day question to one day and
    answering it completely is worse than refusing, because nothing in the
    answer reveals what went missing.
  * anything outside the dataset is reported, not dropped. "Yesterday" is a
    fair question whose honest answer is "the data starts today".

Every result carries the phrase it came from, so an inferred date can be shown
to the controller and challenged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
     "nov", "dec"], 1)}
WEEKDAYS = {"monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
            "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3,
            "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
            "sunday": 6, "sun": 6}

# Time-of-day words, as UTC windows. Coarse on purpose: a controller saying
# "tomorrow morning" means the early bank, not a precise interval.
DAYPARTS = {
    "morning": ("00:00", "11:59"),
    "afternoon": ("12:00", "16:59"),
    "evening": ("17:00", "20:59"),
    "night": ("21:00", "23:59"),
    "overnight": ("21:00", "05:59"),
    "early": ("00:00", "07:59"),
    "late": ("18:00", "23:59"),
}

ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
ISO_TS = re.compile(r"\b(\d{4}-\d{2}-\d{2})[T ]([0-2]\d):([0-5]\d)(?::[0-5]\d)?Z?\b", re.I)
DAY_MONTH = re.compile(
    r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s+"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I)
MONTH_DAY = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+"
    r"(\d{1,2})\s*(?:st|nd|rd|th)?\b", re.I)
DASH_RANGE = re.compile(
    r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s*[-‐-―−]\s*"
    r"(\d{1,2})\s*(?:st|nd|rd|th)?\s+"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I)
WORD_RANGE = re.compile(
    r"\b(?:from|between)\s+(?:the\s+)?(\d{1,2})\s*(?:st|nd|rd|th)?\s*"
    r"(?:(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)?\s*"
    r"(?:to|and|until|through|till|[-‐-―])\s+(?:the\s+)?"
    r"(\d{1,2})\s*(?:st|nd|rd|th)?\s*"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?[a-z]*\b", re.I)
NEXT_N = re.compile(r"\b(?:next|coming|following|over the next)\s+(\d{1,2})\s*days?\b", re.I)
LAST_N = re.compile(r"\b(?:last|past|previous|preceding)\s+(\d{1,2})\s*days?\b", re.I)
WITHIN_N = re.compile(r"\bwithin\s+(?:the\s+)?(?:next\s+)?(\d{1,3})\s*days?\b", re.I)
BARE_DAY = re.compile(r"\bon\s+the\s+(\d{1,2})\s*(?:st|nd|rd|th)\b", re.I)

RELATIVE_DAYS = (
    ("day before yesterday", -2),
    ("day after tomorrow", 2),
    ("yesterday", -1),
    ("tomorrow", 1),
    ("tonight", 0),
    ("today", 0),
)


@dataclass
class Resolved:
    """A temporal expression, pinned to real dates."""
    start: str | None = None
    end: str | None = None                 # inclusive; None means a single day
    start_time: str | None = None          # "HH:MM" UTC
    end_time: str | None = None
    phrase: str | None = None              # what in the question produced this
    outside: list = field(default_factory=list)   # days the dataset lacks
    clipped: bool = False
    days_requested: int = 0
    inside_start: str | None = None        # the part the dataset actually holds
    inside_end: str | None = None

    @property
    def is_range(self) -> bool:
        return bool(self.end and self.end != self.start)

    def dates(self) -> list:
        if not self.start:
            return []
        d0 = date.fromisoformat(self.start)
        d1 = date.fromisoformat(self.end or self.start)
        return [(d0 + timedelta(days=i)).isoformat()
                for i in range((d1 - d0).days + 1)]

    def explain(self) -> str:
        if not self.start:
            return "no date in the question"
        span = self.start if not self.is_range else f"{self.start}..{self.end}"
        bits = [f"'{self.phrase}' -> {span}" if self.phrase else span]
        if self.start_time:
            bits.append(f"{self.start_time}-{self.end_time}Z" if self.end_time
                        else f"at {self.start_time}Z")
        if self.outside:
            bits.append(f"outside the dataset: {', '.join(self.outside)}")
        return "; ".join(bits)

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end,
                "inside_start": self.inside_start, "inside_end": self.inside_end,
                "start_time": self.start_time, "end_time": self.end_time,
                "phrase": self.phrase, "outside_dataset": self.outside,
                "clipped": self.clipped, "explanation": self.explain()}


def _mk(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def resolve(question: str, snapshot: date, window: tuple | None = None) -> Resolved:
    """Pin every temporal expression in `question` to real dates.

    `window` is (first, last) date the dataset holds; days outside it are
    reported rather than silently dropped.
    """
    q = question or ""
    ql = q.lower()
    year = snapshot.year
    out = Resolved()

    # ---- explicit timestamps and ISO dates -----------------------------
    ts = ISO_TS.search(q)
    if ts:
        out.start = ts.group(1)
        out.start_time = f"{ts.group(2)}:{ts.group(3)}"
        out.phrase = ts.group(0)

    isos = ISO_DATE.findall(q)
    if len(isos) >= 2:
        out.start, out.end = min(isos), max(isos)
        out.phrase = f"{out.start} to {out.end}"
    elif isos and not out.start:
        out.start, out.phrase = isos[0], isos[0]

    # ---- written ranges -------------------------------------------------
    if not out.end:
        if m := DASH_RANGE.search(q):
            mon = MONTHS[m.group(3).lower()[:3]]
            out.start = _mk(year, mon, int(m.group(1)))
            out.end = _mk(year, mon, int(m.group(2)))
            out.phrase = m.group(0)
        elif m := WORD_RANGE.search(q):
            m_start = MONTHS[(m.group(2) or m.group(4) or "sep").lower()[:3]]
            m_end = MONTHS[(m.group(4) or m.group(2) or "sep").lower()[:3]]
            out.start = _mk(year, m_start, int(m.group(1)))
            out.end = _mk(year, m_end, int(m.group(3)))
            out.phrase = m.group(0)

    # ---- relative windows ----------------------------------------------
    if not out.start:
        if m := NEXT_N.search(ql):
            n = int(m.group(1))
            out.start = snapshot.isoformat()
            out.end = (snapshot + timedelta(days=n - 1)).isoformat()
            out.phrase = m.group(0)
        elif m := LAST_N.search(ql):
            n = int(m.group(1))
            out.start = (snapshot - timedelta(days=n - 1)).isoformat()
            out.end = snapshot.isoformat()
            out.phrase = m.group(0)
        elif m := WITHIN_N.search(ql):
            n = int(m.group(1))
            out.start = snapshot.isoformat()
            out.end = (snapshot + timedelta(days=n)).isoformat()
            out.phrase = m.group(0)
        elif "weekend" in ql:
            # Checked before "this week": "over the weekend" contains "the week"
            # and the looser phrase would swallow it.
            ahead = (5 - snapshot.weekday()) % 7
            sat = snapshot + timedelta(days=ahead)
            out.start, out.end = sat.isoformat(), (sat + timedelta(days=1)).isoformat()
            out.phrase = "the weekend"
        elif "rest of the week" in ql or "remainder of the week" in ql:
            out.start = (snapshot + timedelta(days=1)).isoformat()
            out.end = (snapshot + timedelta(days=6 - snapshot.weekday())).isoformat()
            out.phrase = "rest of the week"
        elif "this week" in ql or "the week" in ql:
            out.start = snapshot.isoformat()
            out.end = (snapshot + timedelta(days=6)).isoformat()
            out.phrase = "this week"
        elif "next week" in ql:
            out.start = (snapshot + timedelta(days=7)).isoformat()
            out.end = (snapshot + timedelta(days=13)).isoformat()
            out.phrase = "next week"

    # ---- relative days, possibly two of them ----------------------------
    if not out.start:
        hits, seen = [], []
        for word, offset in RELATIVE_DAYS:
            if word in ql and not any(word in s for s in seen):
                seen.append(word)
                hits.append(((snapshot + timedelta(days=offset)).isoformat(), word))
        if hits:
            hits.sort()
            out.start, out.phrase = hits[0][0], hits[0][1]
            if len(hits) > 1:
                out.end, out.phrase = hits[-1][0], f"{hits[0][1]} to {hits[-1][1]}"

    # ---- named weekday --------------------------------------------------
    if not out.start:
        for name, idx in WEEKDAYS.items():
            if re.search(rf"\b{name}\b", ql):
                ahead = (idx - snapshot.weekday()) % 7
                out.start = (snapshot + timedelta(days=ahead)).isoformat()
                out.phrase = name
                break

    # ---- "15 Sep" / "Sep 15" / "on the 15th" ----------------------------
    if not out.start:
        if m := DAY_MONTH.search(q):
            out.start = _mk(year, MONTHS[m.group(2).lower()[:3]], int(m.group(1)))
            out.phrase = m.group(0)
        elif m := MONTH_DAY.search(q):
            out.start = _mk(year, MONTHS[m.group(1).lower()[:3]], int(m.group(2)))
            out.phrase = m.group(0)
        elif m := BARE_DAY.search(q):
            out.start = _mk(year, snapshot.month, int(m.group(1)))
            out.phrase = m.group(0)

    # ---- time of day ----------------------------------------------------
    if not out.start_time:
        for word, (a, b) in DAYPARTS.items():
            if re.search(rf"\b{word}\b", ql):
                out.start_time, out.end_time = a, b
                out.phrase = f"{out.phrase} {word}".strip() if out.phrase else word
                break

    # ---- normalise and clamp -------------------------------------------
    if out.start and out.end and out.end < out.start:
        out.start, out.end = out.end, out.start
    if out.end == out.start:
        out.end = None
    out.days_requested = len(out.dates())

    if window and out.start:
        first, last = window
        requested = out.dates()
        inside = [d for d in requested if first <= d <= last]
        out.outside = [d for d in requested if d not in inside]
        # The requested range is kept as asked. Clipping it here would hide the
        # gap: the tools report which days they hold nothing for, and that
        # sentence is the honest part of the answer.
        out.inside_start = inside[0] if inside else None
        out.inside_end = inside[-1] if len(inside) > 1 else None
        out.clipped = bool(out.outside)
    return out
