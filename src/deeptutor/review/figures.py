"""Inline a pack's living-book diagrams, the way the book itself shows them.

A living book keeps its drawings in one ``window.FIGURES`` map and injects
each one into ``<figure data-fig="name">`` at load time. A review pack was only
ever given a link into the book, so a learner reading the pack offline saw no
diagram at all. This module reads the same map, prepares each SVG for another
page, and hands the exporter static markup, so the pack still needs no script.

Preparation is what makes a verbatim copy unsafe, and each step answers a
concrete failure:

* the ``<?xml …?>`` declaration is fatal as raw HTML, so it is dropped;
* every figure names its glyphs ``glyph-0-3``, so a second figure on the same
  page would borrow the first one's glyphs; ids are therefore namespaced per
  figure and every ``href="#…"``/``url(#…)`` reference follows;
* coordinates carry machine precision (``403.649845``); rounding geometry
  attributes only is invisible at print size and saves about a third of the
  bytes, while never touching a number that is printed as text;
* a drawing carrying a script, an event handler or an external link is refused
  outright rather than sanitised, because a pack must stay a fully offline
  document.

Nothing here writes: callers pass paths and get strings back.
"""

from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from deeptutor.review import materials

#: A drawing larger than this is almost certainly a mistake (a rasterised
#: page); inlining it would push a pack over the export size budget.
MAX_FIGURE_BYTES = 600_000

_XML_DECLARATION = re.compile(r"^\s*<\?xml[^>]*\?>\s*")
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_BETWEEN_TAGS = re.compile(r">\s+<")
_DECLARATION = re.compile(r"<!(?:DOCTYPE|ENTITY)\b", re.I)
#: Only geometry attributes carry the machine-precision numbers; a decimal
#: inside aria-label or a title is text and must survive as written.
_GEOMETRY = re.compile(
    r'(\s(?:d|x|y|x1|x2|y1|y2|cx|cy|r|rx|ry|width|height|points|transform|'
    r'viewBox|stroke-width|font-size|dx|dy)=")([^"]*)(")'
)
_LONG_DECIMAL = re.compile(r"(?<![\w.])(-?\d+\.\d{5,})(?![\w.])")
_ID = re.compile(r'\bid="([^"]+)"')
_ID_REF = re.compile(r'((?:xlink:)?href="#|url\(#)([^")]+)')
_HAZARD = re.compile(r"<script\b|<foreignObject\b|\son[a-z]+\s*=", re.I)
_EXTERNAL = re.compile(r'(?:xlink:)?(?:href|src)="(?!#|data:)', re.I)
_OPEN_SVG = re.compile(r"<svg\b[^>]*>", re.I)
_FIGURE_MARK = re.compile(r'<figure data-fig="([^"]*)"')
_ANCHOR_SECTION = re.compile(r'<section class="section"[^>]*\bid="([^"]+)"')
_NUMERIC = re.compile(r"(\d+)")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


@lru_cache(maxsize=8)
def _book_figures(path: Path, stamp: tuple[int, int]) -> dict[str, str]:
    """Name -> SVG as the built book records it.

    ``stamp`` keeps the cache honest when a book is rebuilt in place; it is
    part of the key rather than read inside, because a stale drawing is worse
    than a cache miss.
    """
    page = _read(path)
    marker = page.find("FIGURES = ")
    if marker < 0:
        return {}
    start = page.find("{", marker)
    if start < 0:
        return {}
    try:
        payload, _ = json.JSONDecoder().raw_decode(page[start:])
    except ValueError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): str(v) for k, v in payload.items() if isinstance(v, str)}


def book_figures(path: Path) -> dict[str, str]:
    """Name -> SVG for a built living book (empty when the map is absent)."""
    try:
        info = path.stat()
    except OSError:
        return {}
    return _book_figures(path, (info.st_mtime_ns, info.st_size))


@lru_cache(maxsize=8)
def _section_names(path: Path, stamp: tuple[int, int]) -> dict[str, list[str]]:
    """Anchor -> figure names in document order, for one built book."""
    page = _read(path)
    marks = [(m.start(), m.group(1)) for m in _ANCHOR_SECTION.finditer(page)]
    out: dict[str, list[str]] = {}
    for index, (position, anchor) in enumerate(marks):
        stop = marks[index + 1][0] if index + 1 < len(marks) else len(page)
        out[anchor] = _FIGURE_MARK.findall(page[position:stop])
    return out


def section_names(path: Path, anchor: str) -> list[str]:
    """The figures one section prints, in order (empty when unmapped)."""
    try:
        info = path.stat()
    except OSError:
        return []
    return list(_section_names(path, (info.st_mtime_ns, info.st_size)).get(anchor, ()))


def natural_key(value: str) -> list[tuple[int, object]]:
    """Sort ``2.1`` before ``2.10`` and both before ``2.11``.

    Each token is tagged so a digit run never compares against a text run.
    """
    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in _NUMERIC.split(value)
    ]


def _round_geometry(match: re.Match[str]) -> str:
    body = _LONG_DECIMAL.sub(
        lambda number: f"{float(number.group(1)):.2f}".rstrip("0").rstrip("."),
        match.group(2),
    )
    return match.group(1) + body + match.group(3)


def _namespace_ids(svg: str, prefix: str) -> str:
    ids = _ID.findall(svg)
    if not ids:
        return svg
    rename = {name: prefix + name for name in ids}
    svg = _ID.sub(lambda m: f'id="{rename[m.group(1)]}"', svg)
    return _ID_REF.sub(lambda m: m.group(1) + rename.get(m.group(2), m.group(2)), svg)


def _strip_tex(text: str) -> str:
    """A caption without math delimiters, for the ARIA label only."""
    return re.sub(r"\$\$|\$|\\[()\[\]]", "", text).strip()


def prepare_svg(svg: str, prefix: str, caption: str) -> str | None:
    """One drawing as safe inline HTML, or None when it is refused.

    ``prefix`` must be unique per figure on the page; it is what keeps the
    glyph ids of two drawings apart. A refused drawing is reported to the
    caller so the exporter can say why instead of shipping a silent gap.
    """
    if not svg or len(svg.encode("utf-8")) > MAX_FIGURE_BYTES:
        return None
    if _HAZARD.search(svg) or _EXTERNAL.search(svg) or _DECLARATION.search(svg):
        return None
    cleaned = _XML_DECLARATION.sub("", svg)
    cleaned = _COMMENT.sub("", cleaned)
    cleaned = _BETWEEN_TAGS.sub("><", cleaned).strip()
    cleaned = _GEOMETRY.sub(_round_geometry, cleaned)
    cleaned = _namespace_ids(cleaned, prefix)
    if not _OPEN_SVG.search(cleaned):
        return None
    try:
        # A parse failure would surface as a blank box in the learner's
        # browser; refuse the drawing while the reason is still visible.
        ElementTree.fromstring(cleaned)
    except ElementTree.ParseError:
        return None
    if 'role="img"' not in cleaned:
        label = html.escape(_strip_tex(caption), quote=True)
        cleaned = _OPEN_SVG.sub(
            lambda m: m.group(0)[:-1] + f' role="img" aria-label="{label}">',
            cleaned,
            count=1,
        )
    return cleaned


def _chapter_sections(
    chapter: str, sections: dict[str, Any], catalog: dict[str, Any]
) -> list[str]:
    """Section numbers the plan maps to ``chapter``, in first-seen order.

    ``book-sections.json`` names the chapter's notes; each note carries the
    book section it is read in. A number only counts when the book's own title
    matches the plan's, so the same number in a different subject's book can
    never claim the chapter.
    """
    numbers: list[str] = []
    for entries in (sections.get(chapter) or {}).values():
        for entry in entries if isinstance(entries, list) else [entries]:
            if not isinstance(entry, dict):
                continue
            number = str(entry.get("no") or "").strip()
            hit = catalog.get(number) if number else None
            if (
                isinstance(hit, dict)
                and str(hit.get("title") or "") == str(entry.get("title") or "")
                and number not in numbers
            ):
                numbers.append(number)
    return numbers


def pack_figures(
    subject: str,
    chapter: str,
    *,
    index: dict[str, Any] | None = None,
    books: dict[str, Any] | None = None,
    sections: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Everything a pack page needs to print its chapter's diagrams.

    The pack title gives the subject and chapter, ``book-sections.json`` maps
    the chapter onto book sections, and the built book supplies the drawings.
    Sections are resolved against the *subject's* catalog, so a chapter name
    that exists in two books cannot borrow the wrong figures. A missing book
    yields None: the page then renders no block rather than an empty promise.
    """
    table = materials.load_book_figures() if index is None else index
    recorded = (materials.load_books() if books is None else books).get(subject) or ""
    if isinstance(recorded, dict):
        recorded = str(recorded.get("html") or "")
    book = materials.resolve_book(str(recorded or ""))
    if book is None:
        return None
    catalog = (table.get("books") or {}).get(subject) or {}
    plan = materials.load_book_sections() if sections is None else sections
    numbers = _chapter_sections(chapter, plan, catalog)
    mapping = materials.chapter_figures(chapter, table) if not numbers else {}
    if numbers:
        rows = [
            (number, catalog[number], list((catalog[number] or {}).get("figures") or []))
            for number in numbers
        ]
    elif mapping:
        # Deployment without book-sections.json: keep the precomputed join.
        rows = [
            (number, catalog.get(number) or {}, list(mapping[number]))
            for number in sorted(mapping, key=natural_key)
        ]
    else:
        return None
    svgs = book_figures(book)
    produced_sections: list[dict[str, Any]] = []
    missing: list[str] = []
    for number, entry, figures in rows:
        anchor = str(entry.get("anchor") or "") if isinstance(entry, dict) else ""
        names = section_names(book, anchor) if anchor else []
        produced: list[dict[str, str]] = []
        for position, row in enumerate(figures):
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "") or (
                names[position] if position < len(names) else ""
            )
            caption = str(row.get("caption") or "")
            label = str(row.get("label") or "")
            prefix = f"f{len(produced_sections)}-{position}-"
            svg = prepare_svg(svgs.get(name, ""), prefix, caption)
            if svg is None:
                missing.append(label or name or number)
                # Keep the row: the export tree can still point at the book,
                # and a caller with no book drops it while rendering.
                produced.append({"label": label, "caption": caption})
                continue
            produced.append({"label": label, "caption": caption, "svg": svg})
        if produced:
            produced_sections.append(
                {
                    "no": number,
                    "title": str(entry.get("title") or "") if isinstance(entry, dict) else "",
                    "anchor": anchor,
                    "figures": produced,
                }
            )
    if not produced_sections:
        return None
    return {"book": book.stem, "sections": produced_sections, "missing": missing}


def pack_for_title(title: str) -> dict[str, Any] | None:
    """The figure block for a pack title (``"<subject> · <chapter> 复习包"``)."""
    from deeptutor.review.export_tree import split_pack_title

    subject, chapter = split_pack_title(title)
    return pack_figures(subject, chapter)
