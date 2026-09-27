"""Plain-text hygiene for anything the model writes.

Left alone, gpt-oss renders "A320, ATR72" with a narrow no-break space (U+202F)
and "04:00-16:00" with a non-breaking hyphen (U+2011) - 750-odd of them in a
single evaluation run. They are valid Unicode and they look fine in a browser,
but a Windows console decoding UTF-8 as cp1252 shows `A320,â¯ATR72` and
`04:00â16:00`, which is what a controller would actually see in the terminal.

They cause quieter damage too: an identifier written `C-1042` with U+2011 is not
the string `C-1042`, so every downstream match - grounding, citations, the
evaluation harness - has to normalise defensively. Three separate bugs in this
codebase came from exactly that. Cleaning the text once, at the point the answer
is finished, removes the whole class.

Only punctuation and spacing are touched. Real content - a rupee sign, an
accented name - is left exactly as written.
"""
from __future__ import annotations

import re

# A thin space between digits is a thousands separator, not a space. The model
# writes INR 18,500 as "18<U+202F>500"; mapping that space to ASCII the way
# everything else here does turns it into "18 500", which reads as two numbers
# and matches neither the figure in the tool result nor anything a controller
# would paste into a system. Caught before the space map runs, while the narrow
# space is still there to identify it.
_GROUPING = re.compile(r"(?<=\d)[     ⁠](?=\d{3}(?!\d))")

# Everything that renders as a hyphen, dash or minus.
_DASHES = "‐‑‒–—―−"
# Everything that renders as a space.
_SPACES = "       　"

_MAP = {ord(c): "-" for c in _DASHES}
_MAP.update({ord(c): " " for c in _SPACES})
_MAP.update({
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"',
    0x2026: "...",
    0x2192: "->", 0x2190: "<-", 0x21D2: "=>",
    0x00D7: "x",
    0x2264: "<=", 0x2265: ">=", 0x2260: "!=",
    0x200B: "", 0x200C: "", 0x200D: "", 0xFEFF: "",   # zero-width, pure noise
    0x2022: "-", 0x00B7: "-", 0x2027: "-",   # bullets and middle dots
    0x00A0: " ",                              # plain nbsp
    0x2044: "/", 0x2215: "/",
    0x2248: "~", 0x2243: "~", 0x223C: "~",
    0x2705: "[ok]", 0x274C: "[no]", 0x26A0: "[!]", 0x2713: "[ok]", 0x2717: "[no]",
    0x20B9: "INR ",                           # the money format is "INR 18,500"
    0x00AD: "",                               # soft hyphen: invisible, breaks matching
})


def plain(text: str) -> str:
    """Normalise typographic punctuation to ASCII. Idempotent and safe to
    apply to markdown - it never touches table pipes or code fences."""
    if not text:
        return text
    return console_safe(_GROUPING.sub(",", text).translate(_MAP))


def _foldable(ch: str) -> str:
    """The closest thing a cp1252 console can print, or nothing at all."""
    import unicodedata
    decomposed = unicodedata.normalize("NFKD", ch)
    kept = "".join(c for c in decomposed if _fits(c))
    return kept


def _fits(ch: str) -> bool:
    try:
        ch.encode("cp1252")
        return True
    except UnicodeEncodeError:
        return False


def console_safe(text: str) -> str:
    """The backstop.

    The map above names the characters the model is known to emit; this catches
    the ones it has not emitted yet. Anything a cp1252 console cannot print is
    folded to its nearest printable form and otherwise dropped, so a new symbol
    in a model update degrades to slightly plainer text rather than to
    `A320,â¯` on a controller's screen.

    Accented names survive untouched - cp1252 covers Latin-1, so this only ever
    touches characters that genuinely cannot be shown.
    """
    if not text:
        return text
    try:
        text.encode("cp1252")
        return text
    except UnicodeEncodeError:
        return "".join(ch if _fits(ch) else _foldable(ch) for ch in text)


def looks_mojibake(text: str) -> bool:
    """Has this text already been decoded with the wrong codec somewhere?

    Used only in diagnostics: the sequences below cannot arise from correctly
    decoded English text, so seeing them means an encoding boundary was crossed
    with the wrong assumption.
    """
    return any(marker in (text or "") for marker in ("â€", "Ã©", "Â ", "ï»¿"))
