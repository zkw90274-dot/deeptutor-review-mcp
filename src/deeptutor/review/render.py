"""Deterministic figure and slider rendering (Phase 1 W4 slice).

No LLM-authored SVG and no browser-side eval:

* figures are drawn from explicit points / bar values into a fixed SVG;
* sliders use a closed arithmetic AST evaluated on the server, then ship the
  precomputed values to the browser.

Unsupported input returns a visible placeholder plus a warning; it never
silently drops a figure or widget.
"""

from __future__ import annotations

import ast
import html
import math
from dataclasses import dataclass, field
from typing import Any

from deeptutor.review.label_gate import LabelBox, check_labels
from deeptutor.review.models import FigureSpec, WidgetSpec
from deeptutor.review.text_metrics import (
    ELLIPSIS_RESERVE_EM,
    FULL_WIDTH_EM,
    text_box,
    text_width_px,
    truncate_to_units,
)

SVG_WIDTH = 640
SVG_HEIGHT = 360
MARGIN = 48

#: Engineering panel palette. Positive blue / negative orange follows the
#: textbook convention; every value is also labelled so colour is never the
#: only channel.
PANEL_POS = "#1c60be"
PANEL_NEG = "#c4710c"
PANEL_AXIS = "#2b3440"
PANEL_ZERO = "#9aa4b2"
PANEL_GRID = "#eef1f5"
PANEL_TOP = 34
PANEL_BOTTOM = 58
PANEL_GAP = 44
PANEL_HEIGHT = 156
PANEL_LEFT = 62
PANEL_RIGHT = 20

#: Bar label font size, in pixels, that the slot budget is measured against.
BAR_LABEL_FONT_SIZE = 10

#: Horizontal gap kept between neighbouring bar labels, in pixels.
BAR_LABEL_GAP_PX = 4.0

#: Floor for a clipped bar label: one full-width glyph plus the ellipsis.
#: A slot narrower than this still gets a readable label; the label gate then
#: reports the overlap instead of the renderer dropping the text.
_MIN_BAR_LABEL_EM = FULL_WIDTH_EM + ELLIPSIS_RESERVE_EM

#: Plot tick label width, in pixels, beyond which the layout adapts. The y
#: tick labels are right-anchored at ``MARGIN - 8``, so this leaves 8px of
#: slack to the canvas edge.
_PLOT_TICK_LABEL_LIMIT_PX = MARGIN - 16

#: Tick counts tried in order: five, then three when the labels stay too wide.
_PLOT_TICK_COUNTS = (5, 3)

#: Tick font sizes tried in order: shrink to 8px before thinning the ticks.
_PLOT_TICK_FONT_SIZES = (10, 8)

#: Room the placeholder's first line has before it crosses the canvas edge.
_PLACEHOLDER_REASON_CHARS = 40
MAX_EXPRESSION_CHARS = 500
MAX_AST_NODES = 200

_ALLOWED_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "abs": abs,
}
_ALLOWED_CONSTANTS = {"pi": math.pi, "e": math.e}


@dataclass(frozen=True)
class FigureRender:
    svg: str
    warning: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"svg": self.svg, "warning": self.warning}


@dataclass(frozen=True)
class WidgetRender:
    values: list[float] = field(default_factory=list)
    warning: str = ""
    expression: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "warning": self.warning,
            "expression": self.expression,
        }


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _svg(title: str, body: str, *, height: int = SVG_HEIGHT) -> str:
    safe_title = html.escape(title or "figure")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_WIDTH} {height}" '
        f'role="img" aria-label="{safe_title}">'
        f"<title>{safe_title}</title>{body}</svg>"
    )


def _text(x: float, y: float, value: str, *, anchor: str = "middle", size: int = 12) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-size="{size}" fill="#5b6675">{html.escape(value)}</text>'
    )


def _label(
    boxes: list[LabelBox],
    value: str,
    *,
    x: float,
    y: float,
    anchor: str = "middle",
    size: int = 12,
) -> str:
    """Emit a ``<text>`` node and record its estimated box for the label gate.

    The SVG string is exactly what :func:`_text` produced before the gate
    existed; only the geometry bookkeeping is new.
    """
    left, top, right, bottom = text_box(
        value, x=x, y=y, anchor=anchor, font_size=size
    )
    boxes.append(LabelBox(value, left, top, right, bottom))
    return _text(x, y, value, anchor=anchor, size=size)


def _label_warning(
    boxes: list[LabelBox],
    *,
    max_overlap_details: int = 3,
    width: int = SVG_WIDTH,
    height: int = SVG_HEIGHT,
) -> str:
    """Summarise label-gate findings as one ``"; "``-joined warning.

    ``FigureRender.warning`` is a single string, so each issue keeps its code
    as a prefix (``label_overlap: ...``); that is the only place the code can
    survive, unlike :class:`LabelReport` which carries it as a field.

    Every error is listed. Overlap is only a warning — slight overlap can be
    readable — but a figure whose labels pile up is still worth one line in
    the export report, so a bounded sample is included instead of all pairs.
    """
    if not boxes:
        return ""
    report = check_labels(boxes, width=width, height=height)
    errors = [
        f"{issue.code}: {issue.detail}"
        for issue in report.issues
        if issue.severity == "error"
    ]
    overlaps = [
        f"{issue.code}: {issue.detail}"
        for issue in report.issues
        if issue.code == "label_overlap"
    ]
    details = errors + overlaps[:max_overlap_details]
    if len(overlaps) > max_overlap_details:
        details.append(
            f"{len(overlaps) - max_overlap_details} more label_overlap issues"
        )
    return "; ".join(details)


def _placeholder(title: str, reason: str) -> FigureRender:
    # The reason comes from exception text and tool messages, so it is bounded
    # here: an unbounded string would overflow the fixed 640x360 canvas.
    shown_reason = reason if len(reason) <= _PLACEHOLDER_REASON_CHARS else (
        reason[: _PLACEHOLDER_REASON_CHARS - 1] + "…"
    )
    safe = html.escape(title or "figure")
    body = (
        f'<rect x="1" y="1" width="{SVG_WIDTH - 2}" height="{SVG_HEIGHT - 2}" '
        'fill="#fafbfc" stroke="#d8dee7" stroke-dasharray="6 4"/>'
        + _text(SVG_WIDTH / 2, SVG_HEIGHT / 2, f"图暂不可用：{shown_reason}")
        + _text(SVG_WIDTH / 2, SVG_HEIGHT / 2 + 24, safe, size=11)
    )
    return FigureRender(_svg(title, body), warning=reason)


def _render_tikz(figure: FigureSpec) -> FigureRender:
    """Render a TikZ figure through the external LaTeX toolchain.

    Every failure mode (missing toolchain, LaTeX error, timeout, unsafe source)
    becomes a visible placeholder plus a warning, never a dropped figure.
    """
    if not figure.tikz.strip():
        return _placeholder(figure.title or figure.id, "tikz source is empty")
    try:
        from deeptutor.review.tikz import render_tikz_document
    except ImportError:
        return _placeholder(figure.title or figure.id, "tikz provider is not installed")
    try:
        rendered = render_tikz_document(
            figure.tikz, title=figure.title or figure.id
        )
    except Exception as exc:  # noqa: BLE001 - a bad figure must stay visible
        reason = getattr(exc, "reason", "") or f"tikz render failed: {type(exc).__name__}"
        return _placeholder(figure.title or figure.id, reason)
    if not rendered.svg:
        return _placeholder(figure.title or figure.id, "tikz produced no SVG")
    warnings = [rendered.layout_error, *rendered.layout_warnings]
    warning = "; ".join(part for part in warnings if part)
    return FigureRender(rendered.svg, warning=warning)


def render_figure(figure: FigureSpec) -> FigureRender:
    if figure.kind == "plot":
        return _render_plot(figure)
    if figure.kind == "bar":
        return _render_bar(figure)
    if figure.kind == "panels":
        return _render_panels(figure)
    if figure.kind == "tikz":
        return _render_tikz(figure)
    return _placeholder(figure.title or figure.id, f"unsupported figure kind {figure.kind!r}")


def _tick_value(low: float, high: float, index: int, count: int) -> float:
    """Value at tick *index* of *count* evenly spaced ticks in ``[low, high]``."""
    return low + index / (count - 1) * (high - low)


def _plot_tick_layout(low: float, high: float) -> tuple[int, int]:
    """Pick ``(tick_count, font_size)`` so the tick labels fit the margin.

    Progression, both axes alike: keep five ticks at 10px while the widest
    label fits ``MARGIN - 16``; otherwise shrink the font to 8px; only if the
    labels still do not fit, thin the ticks to three. The label text is never
    truncated — a shortened numeral would misreport the data — so a figure
    whose labels remain too wide keeps them and lets the label gate report
    the overflow.
    """
    def widest(count: int, size: int) -> float:
        return max(
            text_width_px(
                f"{_tick_value(low, high, index, count):.3g}", font_size=size
            )
            for index in range(count)
        )

    full_count = _PLOT_TICK_COUNTS[0]
    size = _PLOT_TICK_FONT_SIZES[0]
    if widest(full_count, size) > _PLOT_TICK_LABEL_LIMIT_PX:
        size = _PLOT_TICK_FONT_SIZES[1]
    count = full_count
    if widest(full_count, size) > _PLOT_TICK_LABEL_LIMIT_PX:
        count = _PLOT_TICK_COUNTS[1]
    return count, size


def _bar_label_budget_em(slot: float) -> float:
    """Width budget, in em, for a bar label centred in a *slot* wide slot."""
    return max(
        _MIN_BAR_LABEL_EM, (slot - BAR_LABEL_GAP_PX) / BAR_LABEL_FONT_SIZE
    )


def _render_plot(figure: FigureSpec) -> FigureRender:
    points = [
        (float(x), float(y))
        for x, y in figure.points
        if _finite(x) and _finite(y)
    ]
    if len(points) < 2:
        return _placeholder(figure.title or figure.id, "plot needs at least 2 finite points")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    def sx(x: float) -> float:
        return MARGIN + (x - x_min) / (x_max - x_min) * (SVG_WIDTH - 2 * MARGIN)

    def sy(y: float) -> float:
        return SVG_HEIGHT - MARGIN - (y - y_min) / (y_max - y_min) * (SVG_HEIGHT - 2 * MARGIN)

    body = [
        f'<rect x="0" y="0" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="#ffffff"/>',
        f'<line x1="{MARGIN}" y1="{SVG_HEIGHT - MARGIN}" x2="{SVG_WIDTH - MARGIN}" y2="{SVG_HEIGHT - MARGIN}" stroke="#5b6675"/>',
        f'<line x1="{MARGIN}" y1="{MARGIN}" x2="{MARGIN}" y2="{SVG_HEIGHT - MARGIN}" stroke="#5b6675"/>',
    ]
    boxes: list[LabelBox] = []
    x_count, x_size = _plot_tick_layout(x_min, x_max)
    y_count, y_size = _plot_tick_layout(y_min, y_max)
    for index in range(max(x_count, y_count)):
        # Emit both axes' tick marks before both labels so the element order
        # of the original fixed-budget renderer is preserved byte for byte.
        x: float | None = None
        y: float | None = None
        if index < x_count:
            ratio = index / (x_count - 1)
            x = MARGIN + ratio * (SVG_WIDTH - 2 * MARGIN)
            body.append(f'<line x1="{x:.2f}" y1="{SVG_HEIGHT - MARGIN}" x2="{x:.2f}" y2="{SVG_HEIGHT - MARGIN + 5}" stroke="#9aa4b2"/>')
        if index < y_count:
            ratio = index / (y_count - 1)
            y = SVG_HEIGHT - MARGIN - ratio * (SVG_HEIGHT - 2 * MARGIN)
            body.append(f'<line x1="{MARGIN - 5}" y1="{y:.2f}" x2="{MARGIN}" y2="{y:.2f}" stroke="#9aa4b2"/>')
        if x is not None:
            body.append(_label(boxes, f"{_tick_value(x_min, x_max, index, x_count):.3g}", x=x, y=SVG_HEIGHT - MARGIN + 18, size=x_size))
        if y is not None:
            body.append(_label(boxes, f"{_tick_value(y_min, y_max, index, y_count):.3g}", x=MARGIN - 8, y=y + 4, anchor="end", size=y_size))
    polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
    body.append(
        f'<polyline points="{polyline}" fill="none" stroke="#2457a7" stroke-width="2.5" '
        'stroke-linejoin="round" stroke-linecap="round"/>'
    )
    if figure.x_label:
        body.append(_label(boxes, figure.x_label, x=SVG_WIDTH / 2, y=SVG_HEIGHT - 10))
    if figure.y_label:
        body.append(_label(boxes, figure.y_label, x=14, y=MARGIN / 2, anchor="start"))
    return FigureRender(_svg(figure.title or figure.id, "".join(body)), warning=_label_warning(boxes))


def _render_bar(figure: FigureSpec) -> FigureRender:
    labels = list(figure.bar_labels)
    values = [float(value) for value in figure.bar_values if _finite(value)]
    count = min(len(labels), len(values))
    if count == 0:
        return _placeholder(figure.title or figure.id, "bar needs labels and finite values")
    labels = labels[:count]
    values = values[:count]
    maximum = max(abs(value) for value in values) or 1.0
    plot_width = SVG_WIDTH - 2 * MARGIN
    plot_height = SVG_HEIGHT - 2 * MARGIN
    slot = plot_width / count
    bar_width = min(48.0, slot * 0.65)
    baseline = SVG_HEIGHT - MARGIN
    body = [
        f'<rect x="0" y="0" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="#ffffff"/>',
        f'<line x1="{MARGIN}" y1="{baseline}" x2="{SVG_WIDTH - MARGIN}" y2="{baseline}" stroke="#5b6675"/>',
    ]
    boxes: list[LabelBox] = []
    label_budget = _bar_label_budget_em(slot)
    for index, (label, value) in enumerate(zip(labels, values)):
        height = abs(value) / maximum * (plot_height - 20)
        x = MARGIN + slot * index + (slot - bar_width) / 2
        y = baseline - height if value >= 0 else baseline
        body.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{height:.2f}" '
            'fill="#2457a7" opacity="0.85"/>'
        )
        body.append(
            _label(
                boxes,
                truncate_to_units(str(label), max_units=label_budget),
                x=x + bar_width / 2,
                y=baseline + 18,
                size=BAR_LABEL_FONT_SIZE,
            )
        )
        body.append(_label(boxes, f"{value:.3g}", x=x + bar_width / 2, y=y - 6, size=10))
    if figure.x_label:
        body.append(_label(boxes, figure.x_label, x=SVG_WIDTH / 2, y=SVG_HEIGHT - 8))
    if figure.y_label:
        body.append(_label(boxes, figure.y_label, x=14, y=MARGIN / 2, anchor="start"))
    return FigureRender(_svg(figure.title or figure.id, "".join(body)), warning=_label_warning(boxes))


def _panel_format(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    return f"{float(value):.4g}"


def _panel_values(panel: Any) -> list[float]:
    if panel.kind == "shear" and panel.segments:
        return [float(value) for _, _, value in panel.segments if _finite(value)]
    return [float(y) for _, y in panel.points if _finite(y)]


def _render_panels(figure: FigureSpec) -> FigureRender:
    """Stack 1-4 panels on a shared x-axis (Q/M, Bode, etc.).

    The renderer is deliberately data-driven: no SVG is supplied by the model.
    Positive and negative values use the engineering blue/orange convention and
    are always labelled with their numeric value, so the figure remains
    readable in monochrome and in print.
    """
    panels = list(figure.panels)
    if not panels:
        return _placeholder(
            figure.title or figure.id, "panels figure needs at least one panel"
        )

    xs: list[float] = []
    for panel in panels:
        for x0, x1, _ in panel.segments:
            if _finite(x0) and _finite(x1):
                xs.extend([float(x0), float(x1)])
        for x, _ in panel.points:
            if _finite(x):
                xs.append(float(x))
    if not xs:
        return _placeholder(
            figure.title or figure.id, "panels figure has no finite x data"
        )

    x_min, x_max = min(xs), max(xs)
    if x_max == x_min:
        x_max = x_min + 1.0
    count = len(panels)
    height = (
        PANEL_TOP
        + count * PANEL_HEIGHT
        + (count - 1) * PANEL_GAP
        + PANEL_BOTTOM
    )
    left, right = PANEL_LEFT, SVG_WIDTH - PANEL_RIGHT

    def sx(value: float) -> float:
        return left + (float(value) - x_min) / (x_max - x_min) * (right - left)

    x_count = 5
    x_ticks = [_tick_value(x_min, x_max, index, x_count) for index in range(x_count)]
    body = [
        f'<rect x="0" y="0" width="{SVG_WIDTH}" height="{height}" fill="#ffffff"/>'
    ]
    for tick in x_ticks:
        x = sx(tick)
        body.append(
            f'<line x1="{x:.2f}" y1="{PANEL_TOP - 6}" x2="{x:.2f}" '
            f'y2="{height - PANEL_BOTTOM + 6}" stroke="{PANEL_GRID}"/>'
        )

    boxes: list[LabelBox] = []
    warnings: list[str] = []
    for index, panel in enumerate(panels):
        top = PANEL_TOP + index * (PANEL_HEIGHT + PANEL_GAP)
        bottom = top + PANEL_HEIGHT
        plot_top, plot_bottom = top + 18, bottom - 12
        values = _panel_values(panel)
        title = panel.title or {
            "shear": "剪力图",
            "moment": "弯矩图",
            "line": "曲线",
        }.get(panel.kind, "面板")
        if not values:
            warnings.append(f"panel {index + 1} has no finite values")
            body.append(
                _label(
                    boxes,
                    f"{title}：无数据",
                    x=left,
                    y=top + 18,
                    anchor="start",
                    size=11,
                )
            )
            continue

        data_min, data_max = min(0.0, min(values)), max(0.0, max(values))
        if data_min == data_max == 0.0:
            data_min, data_max = -1.0, 1.0
        elif data_max == data_min:
            data_max = data_min + 1.0
        pad = (data_max - data_min) * 0.08
        y_min, y_max = data_min - pad, data_max + pad

        def sy(value: float) -> float:
            return plot_bottom - (float(value) - y_min) / (y_max - y_min) * (
                plot_bottom - plot_top
            )

        if panel.unit:
            title = f"{title} / {panel.unit}"
        body.append(_label(boxes, title, x=left, y=top + 6, anchor="start", size=12))
        body.append(
            f'<line x1="{left}" y1="{plot_top}" x2="{left}" y2="{plot_bottom}" '
            f'stroke="{PANEL_AXIS}"/>'
        )
        zero = sy(0.0)
        body.append(
            f'<line x1="{left}" y1="{zero:.2f}" x2="{right}" y2="{zero:.2f}" '
            f'stroke="{PANEL_ZERO}" stroke-dasharray="4 4"/>'
        )
        y_ticks = {
            _tick_value(data_min, data_max, index, 3) for index in range(3)
        }
        y_ticks.add(0.0)
        for tick in sorted(y_ticks):
            y = sy(tick)
            body.append(
                f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" '
                f'stroke="{PANEL_ZERO}"/>'
            )
            body.append(
                _label(
                    boxes,
                    _panel_format(tick),
                    x=left - 8,
                    y=y + 4,
                    anchor="end",
                    size=10,
                )
            )

        if panel.kind == "shear" and panel.segments:
            for x0, x1, value in panel.segments:
                if not (_finite(x0) and _finite(x1) and _finite(value)):
                    continue
                x0, x1, value = float(x0), float(x1), float(value)
                if x1 <= x0:
                    continue
                xa, xb = sx(x0), sx(x1)
                y = sy(value)
                colour = PANEL_POS if value >= 0 else PANEL_NEG
                body.append(
                    f'<rect x="{xa:.2f}" y="{min(y, zero):.2f}" '
                    f'width="{max(0.0, xb - xa):.2f}" height="{abs(y - zero):.2f}" '
                    f'fill="{colour}" opacity="0.82"/>'
                )
                body.append(
                    _label(
                        boxes,
                        _panel_format(value),
                        x=(xa + xb) / 2,
                        y=y - 6 if value >= 0 else y + 14,
                        size=10,
                    )
                )
        else:
            points = sorted(
                (float(x), float(y))
                for x, y in panel.points
                if _finite(x) and _finite(y)
            )
            if len(points) == 1:
                x, y = points[0]
                colour = PANEL_POS if y >= 0 else PANEL_NEG
                body.append(
                    f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{colour}"/>'
                )
            for (x0, y0), (x1, y1) in zip(points, points[1:]):
                colour = PANEL_POS if (y0 + y1) >= 0 else PANEL_NEG
                body.append(
                    f'<polygon points="{sx(x0):.2f},{sy(y0):.2f} '
                    f'{sx(x1):.2f},{sy(y1):.2f} {sx(x1):.2f},{zero:.2f} '
                    f'{sx(x0):.2f},{zero:.2f}" fill="{colour}" opacity="0.12"/>'
                )
                body.append(
                    f'<line x1="{sx(x0):.2f}" y1="{sy(y0):.2f}" '
                    f'x2="{sx(x1):.2f}" y2="{sy(y1):.2f}" stroke="{colour}" '
                    'stroke-width="2.4" stroke-linecap="round"/>'
                )

            marks = [
                {
                    "x": float(mark.x),
                    "value": float(mark.value),
                    "text": str(mark.text or ""),
                }
                for mark in panel.marks
                if _finite(mark.x) and _finite(mark.value)
            ]
            if not marks and points:
                candidates = [
                    max(points, key=lambda point: point[1]),
                    min(points, key=lambda point: point[1]),
                ]
                seen: set[tuple[float, float]] = set()
                for x, y in candidates:
                    key = (round(x, 10), round(y, 10))
                    if key in seen or abs(y) < 1e-12:
                        continue
                    seen.add(key)
                    marks.append(
                        {"x": x, "value": y, "text": _panel_format(y)}
                    )
            for mark in marks:
                colour = PANEL_POS if mark["value"] >= 0 else PANEL_NEG
                body.append(
                    f'<circle cx="{sx(mark["x"]):.2f}" cy="{sy(mark["value"]):.2f}" '
                    f'r="3" fill="{colour}"/>'
                )
                body.append(
                    _label(
                        boxes,
                        mark["text"] or _panel_format(mark["value"]),
                        x=sx(mark["x"]),
                        y=(
                            sy(mark["value"]) - 8
                            if mark["value"] >= 0
                            else sy(mark["value"]) + 16
                        ),
                        size=10,
                    )
                )

        if index == count - 1:
            for tick in x_ticks:
                x = sx(tick)
                body.append(
                    f'<line x1="{x:.2f}" y1="{plot_bottom}" x2="{x:.2f}" '
                    f'y2="{plot_bottom + 4}" stroke="{PANEL_ZERO}"/>'
                )
                body.append(
                    _label(
                        boxes,
                        _panel_format(tick),
                        x=x,
                        y=bottom + 16,
                        size=10,
                    )
                )
            if figure.x_label:
                body.append(
                    _label(
                        boxes,
                        figure.x_label,
                        x=(left + right) / 2,
                        y=height - 10,
                        size=11,
                    )
                )

    label_warning = _label_warning(boxes, width=SVG_WIDTH, height=height)
    if label_warning:
        warnings.append(label_warning)
    return FigureRender(
        _svg(figure.title or figure.id, "".join(body), height=height),
        warning="; ".join(warnings),
    )


def _parse_expression(expression: str, parameter: str) -> Any:
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError("expression too long")
    tree = ast.parse(expression.replace("^", "**"), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError("expression too complex")
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.UAdd,
        ast.USub,
        ast.Load,
    )
    call_targets: set[ast.expr] = set()
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"unsupported expression node {type(node).__name__}")
        # The callee check must run before the name check: ``sqrt(x)`` also
        # reaches this loop with ``sqrt`` as an ``ast.Name``, and rejecting
        # bare names first would make the function whitelist unreachable.
        if isinstance(node, ast.Call) and not (
            isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCTIONS
        ):
            raise ValueError("unsupported function call")
        if isinstance(node, ast.Call):
            call_targets.add(node.func)
        if (
            isinstance(node, ast.Name)
            and node not in call_targets
            and node.id not in {parameter, *_ALLOWED_CONSTANTS}
        ):
            raise ValueError(f"unsupported name {node.id!r}")
        if isinstance(node, ast.Constant) and not isinstance(
            node.value, (int, float)
        ):
            raise ValueError("only numeric constants are allowed")
        if isinstance(node, ast.Constant) and abs(float(node.value)) > 1_000_000:
            raise ValueError("numeric constant is too large")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if not (
                isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, (int, float))
                and abs(float(node.right.value)) <= 6
            ):
                raise ValueError("power exponent must be a small constant")
    return tree


def _eval_expression(tree: Any, parameter: str, value: float) -> float:
    scope = {parameter: float(value), **_ALLOWED_CONSTANTS, **_ALLOWED_FUNCTIONS}
    result = eval(compile(tree, "<review-slider>", "eval"), {"__builtins__": {}}, scope)
    number = float(result)
    if not math.isfinite(number):
        raise ValueError("expression produced a non-finite value")
    return number


def render_widget(widget: WidgetSpec) -> WidgetRender:
    expression = (widget.expression or "").strip()
    if not expression:
        return WidgetRender(warning="slider has no expression")
    # NaN compares false against everything, so the ``<=`` guard below would
    # let a NaN endpoint through and the failure would surface as a misleading
    # "expression produced a non-finite value" instead of "bad range".
    if not (_finite(widget.minimum) and _finite(widget.maximum)):
        return WidgetRender(warning="slider range must be finite")
    if widget.maximum <= widget.minimum:
        return WidgetRender(warning="slider maximum must be greater than minimum")
    try:
        tree = _parse_expression(expression, widget.parameter)
    except (SyntaxError, ValueError) as exc:
        return WidgetRender(warning=f"unsafe slider expression: {exc}")
    samples = max(5, min(101, int(widget.samples)))
    values: list[float] = []
    try:
        for index in range(samples):
            ratio = index / (samples - 1)
            x = widget.minimum + ratio * (widget.maximum - widget.minimum)
            values.append(round(_eval_expression(tree, widget.parameter, x), 9))
    except (ArithmeticError, ValueError) as exc:
        return WidgetRender(warning=f"slider expression failed: {exc}")
    return WidgetRender(values=values, expression=expression)


__all__ = ["FigureRender", "WidgetRender", "render_figure", "render_widget"]
