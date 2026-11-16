"""Shared text width estimation for server-rendered SVG labels.

Every label in this package is placed by our own code at a known coordinate
and a known font size, so its bounding box can be derived analytically — no
browser, no font-file parsing, no external process. The only approximation is
the glyph advance:

* a full-width glyph (CJK, full-width punctuation) counts as ``1.0em``;
* every other glyph counts as ``0.6em``, the average advance of Latin text.

That rule was validated on the concept graph, where a character-count budget
put full-width labels 5-8% past their node edge. It is deliberately shared:
one copy of the rule, two call sites, instead of a third divergent one.

All functions are pure and deterministic.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Literal

#: Advance of a full-width glyph, in em.
FULL_WIDTH_EM = 1.0

#: Average advance of a half-width glyph, in em.
HALF_WIDTH_EM = 0.6

#: Budget reserved for the ellipsis when a label is cut, in em. The ellipsis
#: glyph itself measures :data:`HALF_WIDTH_EM`; one full em is kept because
#: that is the budget the concept graph shipped with and the extra slack is
#: harmless (it can only make a truncated label shorter).
ELLIPSIS_RESERVE_EM = 1.0

#: Fraction of the font size above the baseline occupied by the glyph body.
#: This is an estimate: no font is loaded, so a real face may reach a little
#: higher (accented capitals) or lower (descenders). It is conservative for
#: collision checks because the collision test treats *more* area as worse.
ASCENT_EM = 0.8

#: Fraction of the font size below the baseline occupied by descenders.
DESCENT_EM = 0.2

Anchor = Literal["start", "middle", "end"]

_ANCHORS = ("start", "middle", "end")


def label_units(text: str) -> float:
    """Approximate label width in em, counting CJK as one full em.

    A half-width glyph averages roughly ``0.6em`` at these sizes; a CJK glyph
    is one full em. Counting characters alone put a full-width label about
    5-8% past the node edge.
    """
    units = 0.0
    for character in text:
        units += (
            FULL_WIDTH_EM
            if unicodedata.east_asian_width(character) in ("W", "F")
            else HALF_WIDTH_EM
        )
    return units


def truncate_to_units(
    text: str,
    *,
    max_units: float,
    ellipsis: str = "…",
) -> str:
    """Collapse whitespace and clamp *text* to a width budget, measured in em.

    The cap is expressed in em units rather than characters because one CJK
    glyph is worth nearly two Latin ones. No wrapping is attempted; the cut is
    marked with *ellipsis*.

    Callers that escape their labels must truncate **before** escaping — an
    HTML entity is markup, not text, and counting it would cut the visible
    label short.

    Args:
        text: Raw label.
        max_units: Positive width budget in em. Must be finite.
        ellipsis: Suffix marking a cut. ``""`` is allowed but then the result
            may be exactly ``max_units`` wide with no visual cue.

    Raises:
        ValueError: *max_units* is not a positive finite number.
    """
    budget = _positive_finite(max_units, "max_units")
    normalized = " ".join(text.split())
    if label_units(normalized) <= budget:
        return normalized
    kept: list[str] = []
    used = 0.0
    for character in normalized:
        width = label_units(character)
        if used + width > budget - ELLIPSIS_RESERVE_EM:
            break
        kept.append(character)
        used += width
    return "".join(kept) + ellipsis


def text_width_px(text: str, *, font_size: float) -> float:
    """Estimated advance width of *text* in SVG user units (pixels)."""
    size = _positive_finite(font_size, "font_size")
    return label_units(text) * size


def text_box(
    text: str,
    *,
    x: float,
    y: float,
    anchor: Anchor = "middle",
    font_size: float = 12.0,
) -> tuple[float, float, float, float]:
    """Estimated ``(left, top, right, bottom)`` box of an SVG ``<text>`` node.

    ``x`` / ``y`` are the SVG text anchor point, with ``y`` the **baseline**
    and the coordinate system growing downwards. Horizontally the box follows
    ``text-anchor``: ``start`` puts *left* at ``x``, ``middle`` centres it and
    ``end`` puts *right* at ``x``.

    Vertically the box spans ``y - 0.8 * font_size`` to
    ``y + 0.2 * font_size``: the glyph body sits above the baseline and only
    descenders drop below it. **These ratios are an estimate** — no font file
    is read, so the real ink box can differ by a few percent. They are chosen
    slightly generous so a false negative (a genuinely clipped label reported
    as fine) is less likely than a false positive.

    Raises:
        ValueError: *anchor* is unknown, or *x* / *y* / *font_size* is not a
            finite number.
    """
    if anchor not in _ANCHORS:
        raise ValueError(f"unknown text anchor {anchor!r}")
    origin_x = _finite(x, "x")
    origin_y = _finite(y, "y")
    size = _positive_finite(font_size, "font_size")
    width = label_units(text) * size
    if anchor == "start":
        left, right = origin_x, origin_x + width
    elif anchor == "end":
        left, right = origin_x - width, origin_x
    else:
        left, right = origin_x - width / 2, origin_x + width / 2
    return (left, origin_y - size * ASCENT_EM, right, origin_y + size * DESCENT_EM)


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _positive_finite(value: float, name: str) -> float:
    number = _finite(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number
