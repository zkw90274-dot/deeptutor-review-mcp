"""Turn a free-form symbol note into a scannable, one-per-line list.

The notes model is asked for ``符号：含义，单位`` entries, but it sometimes
answers with a paragraph that buries the definitions between summary
sentences and repeats the same symbol for each concept. A paragraph in the
pre-answer callout is slow to scan, so the conversion is deterministic:

* split on the separators the model actually uses (``；`` and newlines);
* keep only chunks that *start* a definition — either a ``$...$`` token or a
  bare ASCII symbol — and drop the prose preamble before the first token;
* keep the first definition of each symbol (later repeats are usually the
  same symbol restated with less context).

Nothing here executes or rewrites the LaTeX: the ``$...$`` delimiters are
kept verbatim so the export still typesets the entry.
"""

from __future__ import annotations

import re

#: Heading used by the generated callout. Export recognises the same string
#: so a pack generated before this normaliser existed is still rendered as a
#: list instead of a paragraph.
SYMBOL_HEADING = "符号与单位"

#: One section callout, not a second document. Keep it scannable.
MAX_SYMBOL_LINES = 12
MAX_SYMBOL_CHARS = 700

# ``$f_s$：...`` or ``$\omega_c$：...``. The model's own math delimiters are
# the most reliable marker that a definition starts here.
_MATH = re.compile(r"\$([^$]+)\$")
# A bare symbol such as ``n：倍频次数`` or ``f_s — 射频``. Restricting the
# left side to an ASCII identifier is what keeps Chinese prose out: a
# sentence has no colon directly after a short identifier.
_PLAIN = re.compile(r"^([A-Za-z][A-Za-z0-9_]{0,15})\s*(?:[：:]|—|–)\s*\S")
_SEPARATORS = re.compile(r"[；;\n]")
_TRAILING = "。；; \t"
#: ``、$b$`` / ``，$b$`` / ``和 $b$`` inside a multi-symbol definition.
_MATH_JOINER = re.compile(r"^\s*[、,，和与]\s*\$[^$]+\$")

#: The notes model sometimes answers with a list literal instead of prose.
#: By the time it reaches the callout the opening ``['`` is already gone, so
#: the fragments surface as ``a', 'b', 'c']`` and the definition splitter —
#: which only knows ``；`` and newlines — keeps the quotes and the bracket.
_LIST_SEP = re.compile(r"['\"]\s*,\s*['\"]")
#: A closing ``']`` is never legitimate content. It was anchored to the end of
#: the string, which missed the case where the literal was spliced mid-note:
#: ``... rad/s'] $s_k$：模拟极点 `` kept the bracket inside a definition.
_LIST_TAIL = re.compile(r"['\"]\]")

#: The notes answer sometimes arrives double-escaped. MathJax reads ``\\Omega``
#: as a line break plus the word "Omega", so the learner sees italics instead
#: of Ω. A definition is a single line, so collapsing is safe.
_DOUBLED_COMMAND = re.compile(r"\\\\(?=[A-Za-z])")


def _unpack_list_repr(text: str) -> str:
    """Turn ``a', 'b', 'c']`` back into one entry per element.

    Only quote-comma-quote boundaries split, so a definition such as
    ``$a,b$：内、外导体半径`` is untouched.
    """

    if not _LIST_SEP.search(text):
        return _LIST_TAIL.sub("", text).strip()
    parts: list[str] = []
    for chunk in _LIST_SEP.split(text):
        chunk = _LIST_TAIL.sub("", chunk).strip().strip("[]").strip().strip("'\"")
        if chunk:
            parts.append(chunk)
    return "；".join(parts)


def _definition(chunk: str) -> tuple[str, str] | None:
    """Return ``(dedupe_key, line)`` for a definition, or ``None`` for prose."""
    chunk = chunk.strip().strip(_TRAILING)
    if not chunk:
        return None
    match = _MATH.search(chunk)
    if match:
        tail = chunk[match.end() :]
        # ``$a$、$b$：同一条含义`` shares one definition between symbols; the
        # colon sits after the last token, not the first.
        while True:
            joiner = _MATH_JOINER.match(tail)
            if joiner is None:
                break
            tail = tail[joiner.end() :]
        tail = tail.lstrip()
        if not tail.startswith(("：", ":", "—", "–")):
            # A formula that is not introducing a definition; the chunk may
            # still be a prose sentence that happens to contain math.
            return None
        # Drop the model's preamble (``...的理由：$n$：倍频次数``) and keep the
        # definition itself, including a trailing relation such as
        # ``$f_I$：中频频率，Hz，$f_I=|f_0-f_s|$``.
        line = chunk[match.start() :].strip().strip(_TRAILING)
        return _key(match.group(1)), line
    plain = _PLAIN.match(chunk)
    if plain:
        return _key(plain.group(1)), chunk
    return None


def _key(symbol: str) -> str:
    """Dedupe on the symbol itself; spacing inside ``$...$`` is not content."""
    return re.sub(r"\s+", "", symbol)


def normalize_symbol_callout(text: str) -> str:
    """Return ``符号与单位`` plus one definition per line.

    If no chunk looks like a definition the original text is returned
    unchanged: dropping a learner's only pre-answer note is worse than
    leaving a paragraph in place.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    body = raw
    if body.startswith(SYMBOL_HEADING):
        body = body[len(SYMBOL_HEADING) :].lstrip("：: \t\n")
    # A list literal is one "line" per source sentence; unpack each line so
    # the newline structure the model did produce is preserved.
    body = "\n".join(_unpack_list_repr(line) for line in body.splitlines())
    body = _DOUBLED_COMMAND.sub(r"\\", body)

    entries: list[str] = []
    seen: set[str] = set()
    for chunk in _SEPARATORS.split(body):
        parsed = _definition(chunk)
        if parsed is None:
            continue
        key, line = parsed
        if key in seen:
            continue
        seen.add(key)
        entries.append(line)
        if len(entries) >= MAX_SYMBOL_LINES:
            break
    if not entries:
        return raw

    lines = [SYMBOL_HEADING]
    for line in entries:
        if len("\n".join([*lines, line])) > MAX_SYMBOL_CHARS:
            break
        lines.append(line)
    if len(lines) == 1:
        # Not even the first definition fits the budget. Returning a bare
        # heading would silently delete the note, so keep the original text;
        # clipping mid-LaTeX could leave an unclosed ``$``.
        return raw
    return "\n".join(lines)
