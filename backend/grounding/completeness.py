"""Completeness, enforced rather than requested.

Grounding proves the answer contains nothing the tools did not produce. It says
nothing about the opposite failure: an answer that is entirely true and quietly
leaves out nine of thirteen affected flights. On a crew desk that omission is
the more dangerous of the two - a controller acts on what they were shown.

The narrator's prompt already forbids it and the narrator does it anyway, so
this checks in code. When the answer clearly *is* discussing an enumerable set
from the result - it names some of the members - and omits others, the missing
rows are appended as an authoritative table built from the tool output.

Deliberately conservative. It only fires when the answer has already engaged
with that set, so a question about one flight never drags in a table of
thirteen.
"""
from __future__ import annotations

from backend.grounding.validator import normalize

# Result fields that are answers in their own right, with the label to use if
# they have to be rendered.
PRIMARY_SETS = {
    "affected_flights": "Affected flights",
    "uncrewed_now": "Uncrewed now",
    "also_at_risk": "Also at risk",
    "legs_needing_cover": "Legs needing cover",
    "flights_needing_recrew_or_cancel": "Needing re-crew or cancellation",
    "crew_affected": "Crew affected",
}

# Bottom-line figures. Each is the number a controller signs off on, and each
# belongs to a result whose entire purpose is to produce it - so if it is
# absent from the answer, the answer is unfinished rather than merely brief.
HEADLINE = {
    "total_cost_inr": "Total cost",
    "cancellation_cost_inr": "Cancellation cost",
    "recrew_set_cost_inr": "Re-crew set cost",
    "passengers_affected": "Passengers affected",
}

MIN_SET_SIZE = 4          # below this, an omission is not a completeness problem
ENGAGED_FRACTION = 0.15   # the answer must already mention this much of the set

# Sets that must be reported whenever their partner set is, however short.
# A sick call answered with "these three legs are uncrewed" is true and
# incomplete: the pairing overnights, and tomorrow's three legs are at risk
# too. The controller who is not told that finds out in the morning.
PAIRED_WITH = {"also_at_risk": "uncrewed_now"}
PAIRED_LABEL = {"also_at_risk": "Also at risk (same pairing, later days)"}


def _walk(data, out: dict) -> None:
    if isinstance(data, dict):
        for k, v in data.items():
            if k in PRIMARY_SETS and isinstance(v, list) and v:
                flat = [x for x in v if isinstance(x, str)]
                if flat:
                    out.setdefault(k, [])
                    out[k].extend(x for x in flat if x not in out[k])
            else:
                _walk(v, out)
    elif isinstance(data, list):
        for x in data:
            _walk(x, out)


def find_gaps(answer: str, results: list) -> dict:
    """{field: [items present in the result but absent from the answer]}."""
    text = normalize(answer or "")
    sets: dict = {}
    for r in results:
        d = r.to_dict() if hasattr(r, "to_dict") else r
        _walk(d.get("data"), sets)

    gaps: dict = {}
    for field, items in sets.items():
        partner = PAIRED_WITH.get(field)
        if partner and sets.get(partner):
            # Report it if the answer covered its partner and skipped this.
            covered = [i for i in sets[partner] if i in text]
            if covered and not any(i in text for i in items):
                gaps[field] = list(items)
            continue
        if len(items) < MIN_SET_SIZE:
            continue
        present = [i for i in items if i in text]
        missing = [i for i in items if i not in text]
        if not missing:
            continue
        if len(present) < max(1, int(len(items) * ENGAGED_FRACTION)):
            continue                      # the answer is not about this set
        gaps[field] = missing
    return gaps


def option_gaps(answer: str, results: list) -> list:
    """Ranked options the answer left out, once it started listing them."""
    text = normalize(answer or "")
    options: list = []
    for r in results:
        d = r.to_dict() if hasattr(r, "to_dict") else r
        data = d.get("data")
        if isinstance(data, dict) and isinstance(data.get("options"), list):
            options = [o for o in data["options"] if isinstance(o, dict)]
    if len(options) < 3:
        return []
    # The cancellation fallback carries no crew_id, so identify options by
    # whatever names them: the crew if there is one, else the action text.
    def shown(o) -> bool:
        cid = o.get("crew_id")
        if cid:
            return cid in text
        return "cancel" in text.lower()

    if not any(o.get("crew_id") and o["crew_id"] in text for o in options):
        return []
    return [o for o in options if not shown(o)]


def headline_gaps(answer: str, results: list) -> list:
    """Bottom-line figures the result produced and the answer never stated.

    Gated on engagement, like every other rule here: the answer must already be
    discussing this result - it names an identifier the result returned - before
    a figure is added. Otherwise a one-line reply about something else acquires
    a cost total it never asked for.
    """
    raw = normalize(answer or "")
    text = raw.replace(",", "").replace("\u202f", "").replace("\u00a0", "")
    missing: list = []
    for r in results:
        d = r.to_dict() if hasattr(r, "to_dict") else r
        data = d.get("data")
        if not isinstance(data, dict):
            continue
        ids = [e.split("#", 1)[1] for e in (d.get("evidence") or []) if "#" in e]
        if not any(i and i in raw for i in ids):
            continue                  # the answer is not about this result
        for field, label in HEADLINE.items():
            value = data.get(field)
            if not isinstance(value, (int, float)) or value <= 0:
                continue
            if str(int(value)) in text:
                continue
            missing.append({"label": label, "value": int(value)})
    return missing


def exclusion_gaps(answer: str, results: list) -> list:
    """Notable rejected candidates the answer never mentioned.

    "Why not C-3305?" is the controller's next question, and an answer listing
    who *can* fly without saying who cannot has done half the job. The engine
    already orders exclusions by how near-miss they were, so the first few are
    the ones worth naming.
    """
    text = normalize(answer or "")
    excluded: list = []
    for r in results:
        d = r.to_dict() if hasattr(r, "to_dict") else r
        data = d.get("data")
        if isinstance(data, dict) and isinstance(data.get("excluded_examples"), list):
            excluded = [e for e in data["excluded_examples"] if isinstance(e, dict)]
    if not excluded:
        return []
    if any(e.get("crew_id") and e["crew_id"] in text for e in excluded):
        return []                     # the answer already explains exclusions
    return excluded[:4]


def assessment_gap(answer: str, results: list) -> list:
    """Per-flight delay/FDP rows the answer discussed but did not quantify.

    "P-2204 needs re-crewing" is an observation. "DX402 delays 5.75h, pushing
    the duty to 17.0h against a 12.0h limit" is the reason - and it is the part
    a controller has to defend afterwards.
    """
    text = normalize(answer or "")
    rows: list = []
    for r in results:
        d = r.to_dict() if hasattr(r, "to_dict") else r
        data = d.get("data")
        if isinstance(data, dict) and isinstance(data.get("per_flight_assessment"), list):
            rows = [x for x in data["per_flight_assessment"] if isinstance(x, dict)]
    if not rows:
        return []
    named = [x for x in rows if str(x.get("flight_id", "")) in text
             or str(x.get("flight_no", "")) in text]
    if not named:
        return []
    quantified = sum(1 for x in rows
                     if str(x.get("min_delay_hours")) in text)
    if quantified >= max(2, len(rows) // 2):
        return []                     # the answer already carries the numbers
    return rows


def appendix(gaps: dict, missing_options: list, exclusions: list | None = None,
             assessments: list | None = None, headlines: list | None = None) -> str:
    """The rows the answer omitted, rendered from the tool output verbatim."""
    exclusions = exclusions or []
    assessments = assessments or []
    headlines = headlines or []
    if not (gaps or missing_options or exclusions or assessments or headlines):
        return ""
    lines = ["", "---", "*Completed from the rules engine - "
                   "items the summary above left out:*", ""]
    for field, items in gaps.items():
        label = PAIRED_LABEL.get(field) or PRIMARY_SETS.get(field, field)
        lines.append(f"**{label}** ({len(items)} not listed above): "
                     + ", ".join(f"`{i}`" for i in items))
        lines.append("")
    if missing_options:
        lines.append("**Remaining options**")
        lines.append("")
        lines.append("| Rank | Action | Cost (INR) | Legal |")
        lines.append("|---|---|---:|---|")
        for o in missing_options:
            action = str(o.get("action", ""))[:58]
            cost = o.get("cost_inr")
            lines.append(f"| {o.get('rank', '')} | {action} | "
                         f"{cost:,} | {'yes' if o.get('legal') else 'no'} |")
        lines.append("")
    if headlines:
        for h in headlines:
            lines.append(f"**{h['label']}:** INR {h['value']:,}"
                         if "cost" in h["label"].lower()
                         else f"**{h['label']}:** {h['value']:,}")
        lines.append("")
    if assessments:
        lines.append("**Per-flight assessment**")
        lines.append("")
        lines.append("| Flight | Pairing | Min delay | Crew FDP after | Limit | Action |")
        lines.append("|---|---|---:|---:|---:|---|")
        for a in assessments:
            lines.append(
                f"| {a.get('flight_no') or a.get('flight_id')} | {a.get('pairing_id', '')} | "
                f"{a.get('min_delay_hours')}h | {a.get('crew_fdp_after_delay')}h | "
                f"{a.get('fdp_limit')}h | "
                f"{'legal' if a.get('crew_legal_after_delay') else 're-crew or cancel'} |")
        lines.append("")
    if exclusions:
        lines.append("**Not eligible**")
        lines.append("")
        for e in exclusions:
            lines.append(f"- `{e.get('crew_id')}` — {str(e.get('reason', ''))[:110]}")
        lines.append("")
    return "\n".join(lines)


def complete(answer: str, results: list) -> tuple[str, dict]:
    """Return (answer, report). The answer gains an appendix only if needed."""
    gaps = find_gaps(answer, results)
    missing_options = option_gaps(answer, results)
    missing_exclusions = exclusion_gaps(answer, results)
    missing_rows = assessment_gap(answer, results)
    missing_headlines = headline_gaps(answer, results)
    extra = appendix(gaps, missing_options, missing_exclusions, missing_rows,
                     missing_headlines)
    report = {"fields": {k: len(v) for k, v in gaps.items()},
              "options_omitted": len(missing_options),
              "exclusions_added": len(missing_exclusions),
              "assessment_rows_added": len(missing_rows),
              "headlines_added": len(missing_headlines),
              "appended": bool(extra)}
    return (answer + "\n" + extra if extra else answer), report
