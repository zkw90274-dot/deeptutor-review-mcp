from __future__ import annotations

import re
import unicodedata
from decimal import Decimal
from typing import Iterator

MATH = re.compile(r"\$\$[\s\S]+?\$\$|\$[^$\n]+\$|\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\]")
_BREAK = re.compile(r"[。！？!?；;\r\n]+|(?<!\d)\.|\.(?!\d)")
_SKIP = re.compile(r"(?ms)^[ \t]*(`{3,}|~{3,})[^\r\n]*\r?\n.*?(?:^[ \t]*\1[ \t]*(?=\r?$)|\Z)|<!--.*?-->")
_INLINE_MARKUP = re.compile(r"!?\[([^\]\n]*)\]\([^\n)]*\)|</?[A-Za-z][^>\n]*>")
_HEADING = re.compile(r"^#{1,6}(?:\s|$)")
_PREFIX = re.compile(r"^(?:>\s*|[-*+]\s+|\d+[.)、]\s+)+")
_HAN = re.compile(r"[\u3400-\u9fff]+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]+")
_FRAC = re.compile(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}")
_WRAPPER = re.compile(r"\\(?:text|mathrm|operatorname|textnormal|mathbf|mathit)\s*\{([^{}]*)\}")
_POWER = re.compile(r"([_^])\{(\w|[+-]?\d+)\}")
_QUANTITY = re.compile(
    r"(?<![A-Za-z0-9_.+-])(?P<number>[+-]?(?:[0-9]{1,16}(?:\.[0-9]{1,12})?|\.[0-9]{1,12})(?:[eE][+-]?[0-9]{1,2})?)"
    r"\s*(?P<prefix>[pnumμkMG]?)(?P<unit>Hz|Ω|ohm|V|A|F|H|W|s)(?![A-Za-z0-9_])"
)
_SCALES = {"p": -12, "n": -9, "u": -6, "μ": -6, "m": -3, "": 0, "k": 3, "M": 6, "G": 9}
_RANGE = re.compile(r"\d\s*(?:[A-Za-zΩμ]+\s*)?(?:~|\\sim|–|—|-|至|到)\s*[+-]?\d")
_TEX = {
    "Omega": "Ω", "omega": "ω", "pi": "π", "rho": "ρ", "eta": "η", "mu": "μ",
    "Delta": "Δ", "theta": "θ", "alpha": "α", "beta": "β", "times": "×",
    "cdot": "·", "pm": "±", "approx": "≈", "lt": "<", "gt": ">", "le": "≤", "ge": "≥",
}
_EDGE = " \t*`_\"'“”‘’「」『』:：,，。！？!?；;·、."


def _tex_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("−", "-")
    text = _WRAPPER.sub(r"\1", text)
    text = re.sub(r"\\(?:left|right|displaystyle|textstyle)\b", "", text)
    text = re.sub(r"\\(?:[,;:! ]|quad\b|qquad\b|enspace\b)", " ", text)
    return re.sub(r"\\([A-Za-z]+)(?:\{\})?", lambda match: _TEX.get(match[1], match[0]), text)


def _formula(body: str) -> str:
    text = re.sub(r"\s+", "", _tex_text(body))
    text = _POWER.sub(r"\1\2", text)
    text = re.sub(r"\\(?:dfrac|tfrac)\b", r"\\frac", text)
    text = _FRAC.sub(r"(\1)/(\2)", text)
    return text if len(text) <= 96 and re.search(r"[=≈≤≥<>]", text) and re.search(r"[A-Za-zα-ωΑ-Ω]", text) else ""


def _visible_text(text: str) -> str:
    text = _INLINE_MARKUP.sub(lambda match: (match[1] or "") if not match[0].startswith("!") else "", text)
    text = MATH.sub(lambda match: _tex_text(match[0].strip("$") if match[0].startswith("$") else match[0][2:-2]), text)
    text = _tex_text(text)
    return _PREFIX.sub("", text.strip()).replace("**", "").replace("__", "").replace("`", "").strip(_EDGE)


def quantities(text: str) -> dict[str, str]:
    visible = _visible_text(text)
    if _RANGE.search(visible):
        return {}
    values = {}
    for match in _QUANTITY.finditer(visible):
        number = Decimal(match["number"]).scaleb(_SCALES[match["prefix"]])
        unit = "Ω" if match["unit"] == "ohm" else match["unit"]
        label = f"{format(number.normalize(), 'f')} {unit}"
        values[label] = match[0]
    return values


def formulas(text: str) -> set[str]:
    return {
        formula for match in MATH.finditer(text)
        if (formula := _formula(match[0].strip("$") if match[0].startswith("$") else match[0][2:-2]))
    }


def statement(text: str) -> str:
    def replace_math(match: re.Match[str]) -> str:
        formula = formulas(match[0])
        return next(iter(formula)) if formula else _visible_text(match[0])

    normalized = _visible_text(MATH.sub(replace_math, text))
    normalized = re.sub(r"[，,、:：\"'“”‘’「」『』]", " ", normalized)
    normalized = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", normalized)
    return " ".join(normalized.split())


def prose(text: str) -> str:
    return _QUANTITY.sub(" ", _visible_text(MATH.sub(" ", text))).casefold()


def words(text: str) -> set[str]:
    return set(_WORD.findall(text))


def grams(text: str, width: int) -> set[str]:
    runs = [*_HAN.findall(text), *re.findall(r"[a-z]+", text)]
    return {run[index:index + width] for run in runs for index in range(len(run) - width + 1)}


def passages(raw: str) -> Iterator[tuple[int, int, str]]:
    excluded = sorted([*_SKIP.finditer(raw), *MATH.finditer(raw)], key=lambda match: match.start())
    start = cursor = 0
    for span in excluded:
        if span.start() < cursor:
            continue
        for split in _BREAK.finditer(raw, cursor, span.start()):
            yield from _passage(raw, start, split.start())
            start = split.end()
        if span.re is _SKIP:
            yield from _passage(raw, start, span.start())
            start = span.end()
        cursor = span.end()
    for split in _BREAK.finditer(raw, cursor):
        yield from _passage(raw, start, split.start())
        start = split.end()
    yield from _passage(raw, start, len(raw))


def _passage(raw: str, start: int, end: int) -> Iterator[tuple[int, int, str]]:
    text = raw[start:end]
    if text.strip(_EDGE) and not _HEADING.match(text.lstrip()):
        yield start, end, text
