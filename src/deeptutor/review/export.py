"""Single-file offline HTML export (RS-401/403/404/405/507).

The template is static and contains no network references. ``build_html`` is a
pure function of ``(spec, states, now, include_snapshot)``; given the same
arguments it produces byte-identical output. That property is tested and is
what lets the exporter live without the Next.js frontend.
"""

from __future__ import annotations

import html
import re
import time
import uuid
from functools import lru_cache
from importlib.resources import files
from typing import Any, Callable

from deeptutor.learning.models import RepetitionState

from deeptutor.review import concept_graph as concept_graph_provider
from deeptutor.review.correction import build_correction
from deeptutor.review.models import (
    Item,
    QualityReport,
    ReviewSpec,
    canonical_json,
)
from deeptutor.review.render import FigureRender, render_figure, render_widget
from deeptutor.review.symbols import SYMBOL_HEADING, normalize_symbol_callout
from deeptutor.review.timeutil import iso, tomorrow_window, zone

MAX_BYTES = 8_000_000


class ExportTooLarge(ValueError):
    """The rendered HTML exceeds the export byte budget."""


#: Wall-clock cap for all TikZ subprocess renders in one export. Each figure
#: can spend up to 30s in xelatex plus 30s in pdftocairo; without a shared
#: deadline a pack with many figures blocks the request for minutes. Figure
#: kinds that do not shell out are unaffected, and cached TikZ renders are
#: cheap enough that the deadline is rarely reached.
MAX_TIKZ_RENDER_SECONDS = 120.0
FOOTER_NOTE = (
    "离线复习页：无外部依赖，断网可打开。"
    "公式由内嵌 MathJax 渲染；"
    "自评仅保存在本机浏览器，不会同步回服务器。"
)
#: Math-free pages ship without the renderer, so they must not claim one.
FOOTER_NOTE_PLAIN = (
    "离线复习页：无外部依赖，断网可打开。"
    "自评仅保存在本机浏览器，不会同步回服务器。"
)
MATHJAX_CONFIG = (
    r"<script>window.MathJax={"
    r"loader:{load:[]},"
    r"tex:{inlineMath:[['$','$'],['\\(','\\)']],"
    r"displayMath:[['$$','$$'],['\\[','\\]']],packages:{'[-]':['autoload','require','noundefined']},"
    # ``\oiint``/``\oiiint`` live in the esint extension, which the offline
    # bundle cannot autoload (no network). Define them locally as a double
    # integral with a circle above, the conventional closed-surface sign.
    r"macros:{oiint:'\\overset{\\bigcirc}{\\iint}',oiiint:'\\overset{\\bigcirc}{\\iiint}'}},"
    r"svg:{fontCache:'global'},options:{enableMenu:false},"
    r"startup:{typeset:false}};</script>"
)


#: The delimiters the template's ``mathText`` splits on. A payload that matches
#: any of them (or carries formulas/widget formulas, which are typeset without
#: delimiters) needs the renderer; a page without math must not pay 2.1 MB for
#: it. Keeping the pattern identical to the template is what makes the skip
#: safe: anything the template would wrap in ``.math`` is detected here.
_MATH_PATTERN = re.compile(
    r"\$\$[\s\S]+?\$\$|\$[^$]+?\$|\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\]"
)
#: ``re.split`` drops unmatched text when the pattern has no group; the caption
#: renderer needs the math runs kept *in* the output, so wrap the same pattern.
_CAPTURED_MATH = re.compile(f"({_MATH_PATTERN.pattern})")


def _needs_mathjax(payload: dict[str, Any]) -> bool:
    if any(concept["formulas"] for concept in payload["concepts"].values()):
        return True
    if any(widget.get("formula_latex") for widget in payload["widgets"].values()):
        return True
    return bool(_MATH_PATTERN.search(canonical_json(payload)))


@lru_cache(maxsize=1)
def _mathjax_block() -> str:
    """Return the vendored MathJax config + tex-svg bundle.

    The bundle is trusted local code, not learner content. It is injected only
    after the rendered page has passed the external-URL check, because the
    bundle contains W3C namespace strings (and an SRE CDN fallback that CSP
    blocks) which are not network references.
    """
    bundle = (
        files("deeptutor.review")
        .joinpath("templates/tex-svg.js")
        .read_text(encoding="utf-8")
    )
    if "</script" in bundle.lower():
        raise ValueError("vendored MathJax bundle contains </script>")
    return MATHJAX_CONFIG + "\n<script>" + bundle + "</script>"


@lru_cache(maxsize=1)
def _template() -> str:
    return (
        files("deeptutor.review")
        .joinpath("templates/review.html")
        .read_text(encoding="utf-8")
    )


def _answer_display(item: Item, *, answer_text: str | None = None) -> str:
    if item.kind == "single_choice" and item.answer.choice_index is not None:
        option = item.options[item.answer.choice_index]
        return f"{option.label}. {option.text}"
    if item.kind == "numeric" and item.answer.value is not None:
        if item.answer.tolerance:
            return f"{item.answer.value:g}（容差 ±{item.answer.tolerance:g}）"
        return f"{item.answer.value:g}"
    return item.answer.text if answer_text is None else answer_text


#: Lines that open or close a fenced code block. Math inside such a block is
#: literal text in the export, so normalisation must leave it untouched.
_FENCE_LINE = re.compile(r"^[ \t]*(?:```+|~~~+)")
#: Two or more underscores in a row are a fill-in blank, not TeX subscripts.
_BLANK_RUN = re.compile(r"_{2,}")
#: A backslash command is the cheapest reliable signal of bare TeX.
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
#: ``C:\Users`` and friends are paths, not formulas.
_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:\\")
_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def _outside_fenced_code(text: str, transform: Callable[[str], str]) -> str:
    """Apply ``transform`` to the text between fenced code blocks only."""
    out: list[str] = []
    segment: list[str] = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        if _FENCE_LINE.match(line):
            chunk = "".join(segment)
            out.append(chunk if in_fence else transform(chunk))
            segment = []
            out.append(line)
            in_fence = not in_fence
            continue
        segment.append(line)
    chunk = "".join(segment)
    out.append(chunk if in_fence else transform(chunk))
    return "".join(out)


def normalize_math_placeholders(text: str) -> str:
    """Turn a run of underscores inside math into a visible blank.

    MathJax reads ``____`` as chained subscripts and raises ``Missing open
    brace for subscript``; the template then restores the raw source and shows
    the page-wide warning. The learner meant a fill-in blank, so replace the
    run with ``\\underline{\\hspace{2em}}`` inside math segments only. Single
    underscores (``x_1``) and fenced code are left exactly as written, and the
    replacement is idempotent because it contains no underscores.
    """
    if not text or "____" not in text:
        return text

    def replace(match: re.Match[str]) -> str:
        return _BLANK_RUN.sub(r"\\underline{\\hspace{2em}}", match.group(0))

    return _outside_fenced_code(text, lambda chunk: _MATH_PATTERN.sub(replace, chunk))


def normalize_prose_math(text: str, *, pure_math_only: bool = False) -> str:
    """Wrap a bare TeX fragment in ``$...$`` where the page would show source.

    Formula variable names, a free-text answer and correction choices are
    stored by some models without delimiters, while the template only typesets
    text that carries them. The rule stays narrow: a dollar sign means the
    author already chose a form, a Windows path is not math, and with
    ``pure_math_only`` a CJK answer is prose and must not be wrapped.
    """
    if not text or "$" in text:
        return text
    value = text.strip()
    if not value or _WINDOWS_PATH.search(value) or not _LATEX_COMMAND.search(value):
        return text
    if pure_math_only and _CJK.search(value):
        return text
    return f"${value}$"


def _item_payload(spec: ReviewSpec, item: Item) -> dict[str, Any]:
    answer_text = normalize_prose_math(item.answer.text, pure_math_only=True)
    correction = build_correction(spec, item)
    # The correction buttons go through the same rich-text renderer as an
    # answer, so a bare-TeX choice would still show its source there.
    for option in correction["options"]:
        option["text"] = normalize_prose_math(option["text"], pure_math_only=True)
    return {
        "id": item.id,
        "kind": item.kind,
        "stem": normalize_math_placeholders(item.stem),
        "difficulty": item.difficulty,
        "options": [
            {"label": option.label, "text": option.text} for option in item.options
        ],
        "answer_display": normalize_math_placeholders(
            _answer_display(item, answer_text=answer_text)
        ),
        "answer_index": item.answer.choice_index,
        "explanation": normalize_math_placeholders(item.explanation),
        "study_aids": {
            "assumptions": list(item.assumptions),
            "self_check": item.self_check,
            "pitfalls": list(item.pitfalls),
            "table": item.study_table,
        },
        "correction": correction,
        "source_ids": item.source_ids,
        "concept_ids": item.concept_ids,
    }


#: Styles for the end-of-page figure block. They travel inside the block
#: rather than the shared <style>, so a page without figures carries no rules
#: for a section it does not have.
BOOK_FIGURES_CSS = (
    # Drawing palette copied from the living book, scoped to this block: the
    # book's SVGs paint with style="stroke:var(--fig-accent)", which resolves
    # through inheritance, and a page without the block never sees the tokens.
    ".book-figures{--fig-ink:#14161a;--fig-muted:#5f6673;--fig-grid:#dfe2e8;"
    "--fig-accent:#0d6e64;--fig-blue:#2f5fd0;--fig-warm:#a55a06;--fig-red:#c33b3b;"
    "--fig-green:#1d7a4f;margin:28px 0 0;padding:22px 24px;background:var(--card);"
    "border:1px solid var(--hairline);border-radius:var(--radius);"
    "box-shadow:var(--shadow),var(--card-highlight)}"
    'html[data-theme="dark"] .book-figures{--fig-ink:#eef0f3;--fig-muted:#9aa1ad;'
    "--fig-grid:#23262c;--fig-accent:#4fd1c0;--fig-blue:#8fb0ff;--fig-warm:#e8b063;"
    "--fig-red:#f08a84;--fig-green:#6fdaa6}"
    "@media (prefers-color-scheme: dark){html:not([data-theme=\"light\"]) .book-figures{"
    "--fig-ink:#eef0f3;--fig-muted:#9aa1ad;--fig-grid:#23262c;--fig-accent:#4fd1c0;"
    "--fig-blue:#8fb0ff;--fig-warm:#e8b063;--fig-red:#f08a84;--fig-green:#6fdaa6}}"
    ".book-figures h2{font-size:1.05rem;line-height:1.4;margin:0 0 6px}"
    ".book-figures-lede{margin:0 0 12px;color:var(--muted);font-size:.86rem;line-height:1.7}"
    ".book-figures ol,.book-figures ul{margin:0;padding:0;list-style:none;display:grid;"
    "grid-template-columns:minmax(0,1fr);gap:10px}"
    ".book-figures ol,.book-figures .fig-list{gap:18px}"
    ".book-figures ol>li{padding-top:10px;border-top:1px solid var(--hairline);min-width:0}"
    ".book-figures ol>li:first-child{border-top:0;padding-top:0}"
    ".book-figures .section-no{display:inline-block;min-width:3.2em;"
    "font-variant-numeric:tabular-nums;font-weight:650;color:var(--accent)}"
    ".book-figures .section-title{font-weight:600}"
    ".book-figures h3.section-head{font-size:.93rem;line-height:1.5;margin:0 0 10px;"
    "letter-spacing:0;color:var(--muted);font-weight:600}"
    ".book-figures ul{gap:2px;margin-top:6px;font-size:.88rem;line-height:1.6}"
    ".book-figures ul a{display:inline-block;padding:3px 0;color:var(--muted);text-decoration:none}"
    ".book-figures ul a:hover,.book-figures ul a:focus-visible{color:var(--accent);text-decoration:underline}"
    ".book-figures ul a b{color:var(--ink);margin-right:4px}"
    # One drawing per figure, sized the way the book sizes it: the SVG scales
    # to the column and the caption sits under it, centered.
    ".book-figure{margin:0;padding:14px 14px 10px;border:1px solid var(--hairline);"
    "border-radius:var(--radius-inner);background:var(--card-soft);text-align:center;"
    "min-width:0}"
    ".book-figure>svg{display:block;width:100%;height:auto;margin:0 auto}"
    ".book-figure figcaption{margin:10px 0 0;font-size:.84rem;line-height:1.65;"
    "color:var(--muted);text-align:center}"
    ".book-figure figcaption b{color:var(--ink);font-weight:650;margin-right:6px}"
    "@media (max-width:640px){.book-figure{overflow-x:auto;padding:10px 10px 8px}"
    ".book-figure>svg{min-width:520px}}"
    "@media print{"
    # The block's palette is lighter ink on paper, matching the template's
    # print reset; a dark-theme page must not print dark drawings.
    ".book-figures{--fig-ink:#14161a;--fig-muted:#5f6673;--fig-grid:#dfe2e8;"
    "--fig-accent:#0d6e64;--fig-blue:#2f5fd0;--fig-warm:#a55a06;--fig-red:#c33b3b;"
    "--fig-green:#1d7a4f;break-before:page;page-break-before:always;box-shadow:none;"
    "border-radius:0;padding:0;border:0;background:#fff}"
    ".book-figure{break-inside:avoid;page-break-inside:avoid;border-radius:0;background:#fff}"
    ".book-figure>svg{min-width:0;max-height:215mm}"
    "}"
)


def _without_label(caption: str, label: str) -> str:
    """The caption without the printed figure number it already starts with."""
    if label and caption.startswith(label):
        rest = caption[len(label):].lstrip(" \u3000:：·")
        return rest or caption
    return caption


def _caption_html(label: str, caption: str) -> str:
    """A caption with its math runs wrapped for the page's typesetter.

    The book prints captions as text and lets MathJax find ``$…$``/``\\(…\\)``;
    a pack does the same, but only the wrapped runs are registered, so the
    typesetter never walks the drawings. Without JavaScript the delimiters stay
    readable as text.
    """
    parts: list[str] = []
    for part in _CAPTURED_MATH.split(_without_label(caption, label)):
        if not part:
            continue
        if _CAPTURED_MATH.fullmatch(part):
            parts.append(f'<span class="math">{html.escape(part)}</span>')
        else:
            parts.append(html.escape(part))
    head = f"<b>{html.escape(label)}</b>" if label else ""
    return head + (" " if head and parts else "") + "".join(parts)


def _book_figures_html(book_figures: dict[str, Any] | None) -> str:
    """This chapter's living-book diagrams, inlined at the end of the page.

    A pack that owns the drawings prints them the way the book does: static
    ``<figure>``/``<figcaption>`` pairs with the prepared SVG inlined, so the
    page needs no script and no second file. Figures the caller could only
    name (no SVG available) fall back to the relative link block, because a
    link to the book beats a silent gap. An empty block renders nothing.
    """
    if not book_figures:
        return ""
    sections = [
        section
        for section in (book_figures.get("sections") or [])
        if isinstance(section, dict) and section.get("figures")
    ]
    if not sections:
        return ""
    href = html.escape(str(book_figures.get("href") or ""), quote=True)
    rows: list[str] = []
    drawn = 0
    for section in sections:
        number = html.escape(str(section.get("no") or ""), quote=True)
        title = html.escape(str(section.get("title") or ""), quote=True)
        anchor = html.escape(str(section.get("anchor") or ""), quote=True)
        figures: list[str] = []
        for figure in section["figures"]:
            if not isinstance(figure, dict):
                continue
            label = str(figure.get("label") or "")
            caption = str(figure.get("caption") or "")
            svg = figure.get("svg")
            if isinstance(svg, str) and svg.lstrip().startswith("<svg"):
                drawn += 1
                figures.append(
                    '<figure class="book-figure">'
                    + svg
                    + "<figcaption>"
                    + _caption_html(label, caption)
                    + "</figcaption></figure>"
                )
            elif href:
                figures.append(
                    '<li><a href="{href}{hash}"><b>{label}</b> {caption}</a></li>'.format(
                        href=href,
                        hash="#" + anchor if anchor else "",
                        label=html.escape(label or "插图"),
                        # The caption the book prints already opens with its
                        # number; repeating it after the label is a stutter.
                        caption=html.escape(_without_label(caption, label)),
                    )
                )
        if not figures:
            continue
        if not drawn:
            rows.append(
                "<li><span class='section-no'>{number}</span>"
                "<span class='section-title'>{title}</span>"
                "<ul>{figures}</ul></li>".format(
                    number=number, title=title, figures="".join(figures)
                )
            )
        else:
            rows.append(
                "<li><h3 class='section-head'><span class='section-no'>{number}</span>"
                "{title}</h3><div class='fig-list'>{figures}</div></li>".format(
                    number=number, title=title, figures="".join(figures)
                )
            )
    if not rows:
        return ""
    book = html.escape(str(book_figures.get("book") or ""), quote=True)
    lede = (
        f"以下 {drawn} 幅插图取自活书《{book}》，已随本页内嵌，断网也能看。"
        if drawn
        else f"插图在活书《{book}》里，点开跳到对应小节；复习包本身只收题目。"
    )
    return (
        f"<style>{BOOK_FIGURES_CSS}</style>"
        '<section class="book-figures" id="book-figures" aria-labelledby="book-figures-title">'
        '<h2 id="book-figures-title">本章相关插图</h2>'
        f'<p class="book-figures-lede">{lede}</p>'
        f"<ol>{''.join(rows)}</ol>"
        "</section>"
    )


def _safe_json(payload: Any) -> str:
    raw = canonical_json(payload)
    # The payload is embedded in a <script> element. Escaping the first three
    # characters keeps a literal </script> in learner content from closing the
    # element; JSON.parse restores the original string exactly.
    return (
        raw.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _render_concept_graph(spec: ReviewSpec) -> dict[str, Any] | None:
    """Render the prerequisite graph, or ``None`` when it carries no edges.

    A graph whose concepts have no prerequisites is a column of disconnected
    boxes: it states nothing the concept list does not already state, so it is
    skipped rather than shipped as decoration. Every failure mode still yields
    a payload with a warning, because a missing graph must be explainable.
    """
    if not any(concept.prerequisites for concept in spec.concepts):
        return None
    try:
        # No in-SVG title: the chapter heading already names the graph, and a
        # second title both duplicates the text and reserves vertical space.
        rendered = concept_graph_provider.render_concept_graph(spec.concepts)
    except Exception as exc:  # noqa: BLE001 - a bad graph must stay visible
        return {
            "svg": "",
            "warning": f"concept graph render failed: {type(exc).__name__}",
        }
    if not rendered.svg:
        return {"svg": "", "warning": rendered.warning or "概念图未生成"}
    return {"svg": rendered.svg, "warning": rendered.warning}


def _render_assets(
    spec: ReviewSpec,
    *,
    read_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Render trusted figures/widgets and collect visible-degradation warnings."""
    figures: dict[str, Any] = {}
    widgets: dict[str, Any] = {}
    warnings: list[str] = []
    tikz_deadline: float | None = None
    for figure in spec.figures:
        rendered: FigureRender | None = None
        if figure.kind == "tikz" and read_only:
            rendered = render_figure(figure, allow_tikz=False)
        elif figure.kind == "tikz":
            now = time.monotonic()
            if tikz_deadline is None:
                tikz_deadline = now + MAX_TIKZ_RENDER_SECONDS
            if now >= tikz_deadline:
                rendered = FigureRender(
                    "", warning="tikz render budget exhausted"
                )
        if rendered is None:
            try:
                rendered = render_figure(figure)
            except Exception as exc:  # noqa: BLE001 - a bad asset must stay visible
                rendered = FigureRender(
                    "", warning=f"figure render failed: {type(exc).__name__}"
                )
        figures[figure.id] = {
            # The raw source (SVG / TikZ) never reaches the reader: only the
            # rendered, sanitized SVG does. Shipping it would duplicate bytes
            # and leak LaTeX into the page.
            **figure.model_dump(mode="json", exclude={"svg", "tikz"}),
            "render": rendered.as_dict(),
        }
        if rendered.warning:
            warnings.append(f"figure {figure.id}: {rendered.warning}")
    for widget in spec.widgets:
        try:
            rendered_widget = render_widget(widget)
        except Exception as exc:  # noqa: BLE001
            from deeptutor.review.render import WidgetRender

            rendered_widget = WidgetRender(warning=f"widget render failed: {type(exc).__name__}")
        widgets[widget.id] = {
            **widget.model_dump(mode="json"),
            "render": rendered_widget.as_dict(),
        }
        if rendered_widget.warning:
            warnings.append(f"widget {widget.id}: {rendered_widget.warning}")
    return figures, widgets, warnings


def _payload(
    spec: ReviewSpec,
    states: dict[str, RepetitionState],
    *,
    now: float,
    include_snapshot: bool,
    quality: QualityReport,
    rendered_figures: dict[str, Any] | None = None,
    rendered_widgets: dict[str, Any] | None = None,
    concept_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tz = zone(spec.plan.timezone)
    figure_index = {figure.id: figure for figure in spec.figures}
    widget_index = {widget.id: widget for widget in spec.widgets}
    item_index = {item.id: item for item in spec.items}

    sections = []
    referenced_items: set[str] = set()
    for section in sorted(spec.sections, key=lambda row: (row.order, row.id)):
        blocks: list[dict[str, Any]] = []
        section_payload: dict[str, Any] = {
            "id": section.id,
            "title": section.title,
            "order": section.order,
            "concept_ids": list(section.concept_ids),
            "blocks": blocks,
        }
        for block in section.blocks:
            if block.kind in ("prose", "summary"):
                blocks.append(block.model_dump(mode="json"))
            elif block.kind == "callout":
                row = block.model_dump(mode="json")
                # Packs generated before the symbol normaliser stored the
                # notes as a paragraph. Normalise at export time so an old
                # pack renders as a list without rewriting its content hash.
                if row["text"].lstrip().startswith(SYMBOL_HEADING):
                    row["text"] = normalize_symbol_callout(row["text"])
                    row["format"] = "symbols"
                blocks.append(row)
            elif block.kind == "item":
                referenced_items.add(block.item_id)
                blocks.append(
                    {
                        "kind": "item",
                        "item": _item_payload(spec, item_index[block.item_id]),
                    }
                )
            elif block.kind == "figure":
                figure = figure_index.get(block.figure_id)
                blocks.append(
                    {
                        "kind": "figure",
                        "figure": {
                            "id": block.figure_id,
                            "title": figure.title if figure else "",
                        },
                    }
                )
            elif block.kind == "widget":
                widget = widget_index.get(block.widget_id)
                blocks.append(
                    {
                        "kind": "widget",
                        "widget": {
                            "id": block.widget_id,
                            "title": widget.title if widget else "",
                        },
                    }
                )
        sections.append(section_payload)

    # A section can omit an item (hand-edited spec or a generator bug). Never
    # silently drop reviewable content from the offline page; surface it in a
    # clearly named trailing section instead.
    orphaned = [item for item in spec.items if item.id not in referenced_items]
    if orphaned:
        sections.append(
            {
                "id": "__ungrouped__",
                "title": "其他题目",
                "blocks": [
                    {"kind": "item", "item": _item_payload(spec, item)}
                    for item in orphaned
                ],
            }
        )

    tomorrow_date, start, end = tomorrow_window(now, spec.plan.timezone)
    due_tomorrow = 0
    for concept in spec.concepts:
        state = states[concept.id]
        if start <= state.next_review_at < end:
            due_tomorrow += len(spec.items_for_concept(concept.id))

    snapshot = None
    if include_snapshot:
        snapshot = {
            "generated_at": iso(now, tz),
            "concepts": [
                {
                    "concept_id": concept.id,
                    "concept_key": concept.key,
                    "name": concept.name,
                    "next_due": iso(states[concept.id].next_review_at, tz),
                    "next_due_ts": states[concept.id].next_review_at,
                    "retrievability": round(states[concept.id].retrievability, 4),
                    "review_count": states[concept.id].review_count,
                    "lapse_count": states[concept.id].lapse_count,
                }
                for concept in spec.concepts
            ],
        }

    # Long-term progress for the hero line. ``strong`` means "reviewed at
    # least three times and never forgotten", not a retrievability value:
    # retrievability decays with time and would hint at the forgetting curve
    # before the learner has answered.
    progress = {
        "total": len(spec.concepts),
        "reviewed": sum(
            1 for concept in spec.concepts if states[concept.id].review_count > 0
        ),
        "strong": sum(
            1
            for concept in spec.concepts
            if states[concept.id].review_count >= 3
            and states[concept.id].lapse_count == 0
        ),
    }

    return {
        "schema_version": spec.schema_version,
        "spec_id": spec.id,
        "title": spec.title,
        "language": spec.language,
        "plan": {
            "daily_minutes": spec.plan.daily_minutes,
            "exam_date": spec.plan.exam_date.isoformat() if spec.plan.exam_date else None,
            "timezone": spec.plan.timezone,
            "target_retention": spec.plan.target_retention,
        },
        "sections": sections,
        "figures": rendered_figures or {},
        "widgets": rendered_widgets or {},
        # ``None`` (rather than an empty object) tells the template there is
        # nothing worth drawing; the runtime skips the block entirely.
        "concept_graph": concept_graph,
        "concepts": {
            concept.id: {
                "id": concept.id,
                "name": concept.name,
                "definition": concept.definition,
                "formulas": [
                    {
                        "latex": normalize_math_placeholders(formula.latex),
                        "conditions": normalize_math_placeholders(formula.conditions),
                        "variables": {
                            normalize_math_placeholders(
                                normalize_prose_math(name)
                            ): normalize_math_placeholders(description)
                            for name, description in formula.variables.items()
                        },
                    }
                    for formula in concept.formulas
                ],
            }
            for concept in spec.concepts
        },
        "sources": {
            source_id: {
                "title": source.title,
                "ref": source.ref,
                "snippet": source.snippet,
                "kind": source.kind,
            }
            for source_id, source in sorted(spec.sources.items())
        },
        "progress": progress,
        "tomorrow": {"date": tomorrow_date, "due_items": due_tomorrow},
        "snapshot": snapshot,
        "quality": {
            "passed": quality.passed,
            # G11 is generator guidance with internal field paths, not a
            # caveat about the learner's material. Keep it in the stored
            # report for the tool layer, but off the learner-facing page.
            "warnings": [
                warning
                for warning in quality.warnings
                if not warning.startswith("G11-bare-tex:")
            ],
        },
        # Concepts the QA gates allow through without a source anchor (only
        # non-kb coverage reaches export). Surfaced so the reader knows which
        # definitions are paraphrased rather than cited.
        "source_gaps": [
            {"id": concept.id, "name": concept.name, "coverage": concept.coverage}
            for concept in spec.concepts
            if not concept.source_ids
        ],
        "footer_note": FOOTER_NOTE,
    }


def build_html(
    spec: ReviewSpec,
    states: dict[str, RepetitionState],
    *,
    now: float,
    include_snapshot: bool = False,
    quality: QualityReport,
    quality_state: str | None = None,
    anchors: dict[str, Any] | None = None,
    read_only: bool = False,
    book_figures: dict[str, Any] | None = None,
) -> str:
    if not quality.passed:
        raise ValueError("refusing to export a spec that failed QA gates")
    rendered_figures, rendered_widgets, render_warnings = _render_assets(spec, read_only=read_only)
    concept_graph = _render_concept_graph(spec)
    if concept_graph is not None and concept_graph.get("warning"):
        render_warnings.append(f"concept graph: {concept_graph['warning']}")
    if render_warnings:
        quality = quality.model_copy(
            update={"warnings": [*quality.warnings, *render_warnings]}
        )
    payload = _payload(
        spec,
        states,
        now=now,
        include_snapshot=include_snapshot,
        quality=quality,
        rendered_figures=rendered_figures,
        rendered_widgets=rendered_widgets,
        concept_graph=concept_graph,
    )
    needs_mathjax = _needs_mathjax(payload)
    # A figure caption can carry math the pack itself never mentions; the page
    # typesets those runs too, so the renderer has to ship for them.
    figures_html = _book_figures_html(book_figures)
    needs_mathjax = needs_mathjax or 'class="math"' in figures_html
    payload["quality_watermark"] = "未人工复核" if quality_state in {"draft", "legacy"} else None
    payload["anchors"] = anchors or {}
    if not needs_mathjax:
        payload["footer_note"] = FOOTER_NOTE_PLAIN
    template = _template()
    # Escape braces in the title as entities so the title cannot introduce a
    # second replacement token.
    title_html = (
        html.escape(spec.title, quote=True)
        .replace("{", "&#123;")
        .replace("}", "&#125;")
        .replace("\x00", "")
    )
    if (
        template.count("{{TITLE}}") != 1
        or template.count("{{PAYLOAD}}") != 1
        or template.count("{{MATHJAX}}") != 1
        or template.count("{{BOOK_FIGURES}}") != 1
    ):
        raise ValueError("template token drift in export")
    # Replace the template's MathJax token with a per-render sentinel *before*
    # the payload is inserted. A fixed NUL sentinel would be forgeable by a
    # title containing those exact bytes; the random nonce cannot be predicted
    # from learner content, and ``json.dumps`` escapes NUL as ``\\u0000`` so a
    # stem containing the literal ``{{MATHJAX}}`` is never re-scanned.
    mathjax_sentinel = f"\x00MATHJAX-{uuid.uuid4().hex}\x00"
    payload_json = _safe_json(payload)
    if mathjax_sentinel in title_html or mathjax_sentinel in payload_json:
        raise ValueError("unexpected sentinel collision in export")
    rendered = (
        template.replace("{{TITLE}}", title_html)
        .replace("{{MATHJAX}}", mathjax_sentinel)
        .replace("{{BOOK_FIGURES}}", figures_html)
        .replace("{{PAYLOAD}}", payload_json)
    )
    # Do not scan the rendered page for braces: learner content is allowed to
    # contain the literal text ``{{PAYLOAD}}``; only the template is a contract.
    # URL check runs before MathJax injection: the vendored bundle contains
    # XML namespace strings and an SRE CDN fallback that CSP blocks. Generated
    # SVG carries the W3C namespace, which is not a network dependency.
    external = [
        match.group(0)
        for match in re.finditer(r"https?://[^\s\"'<>]+", rendered)
        if not match.group(0)
        .lower()
        .startswith(("http://www.w3.org/", "https://www.w3.org/"))
    ]
    if external:
        raise ValueError(f"offline export contains an external URL: {external[0]}")
    # A math-free page renders every formula site as plain text, so shipping the
    # 2.1 MB bundle would be pure dead weight. The template marks
    # ``mathjax-ready`` and skips its warning when there is nothing to typeset.
    rendered = rendered.replace(
        mathjax_sentinel, _mathjax_block() if needs_mathjax else ""
    )
    encoded = rendered.encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise ExportTooLarge(
            f"export is {len(encoded)} bytes, over the {MAX_BYTES}-byte W2 budget"
        )
    return rendered
