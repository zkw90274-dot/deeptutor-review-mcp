"""Deterministic concept dependency graph rendering (Phase 1 W4 slice).

The graph is drawn from ``Concept.id`` / ``Concept.name`` /
``Concept.prerequisites`` only. No mermaid, no browser, no external process:

* Kahn topological layering, stable ordering inside every layer;
* a fixed 720px canvas width, height driven by the busiest layer and capped;
* invalid input (unknown prerequisite, self loop, cycle, oversize) returns a
  visible placeholder plus a warning, never an exception and never a silently
  dropped edge.

The output is a pure function of the input: two calls with the same concepts
produce byte-identical SVG.
"""

from __future__ import annotations

import heapq
import html
from dataclasses import dataclass
from typing import Sequence

from deeptutor.review.models import Concept
from deeptutor.review.text_metrics import truncate_to_units

SVG_WIDTH = 720
MARGIN = 48
MAX_NODES = 60
MAX_CANVAS_HEIGHT = 900
MIN_CANVAS_HEIGHT = 120
PLACEHOLDER_HEIGHT = 360

NODE_HEIGHT = 34
NODE_MIN_WIDTH = 40
NODE_MAX_WIDTH = 160
NODE_RADIUS = 7
ROW_GAP = 18
COLUMN_GAP = 10
MIN_ROW_NODE_HEIGHT = 16
LAYER_MIN_HEIGHT = 12

TITLE_HEIGHT = 36
BOTTOM_PAD = 28

#: Upper bound for a label, in em units (a CJK glyph counts as one, a Latin
#: glyph as 0.6). Kept as an alias so callers that reason in characters still
#: have a stable name; the actual budget comes from the column width.
MAX_LABEL_EM = 15.0
MAX_LABEL_CHARS = 18

_NODE_FILL = "#ffffff"
_ROOT_FILL = "#f5f7fa"
_NODE_STROKE = "#5b6675"
_TEXT_FILL = "#2f3a48"
_EDGE_STROKE = "#8a95a5"
_PLACEHOLDER_FILL = "#fafbfc"
_PLACEHOLDER_STROKE = "#d8dee7"
_ARROW_ID = "cg-arrow"


@dataclass(frozen=True)
class ConceptGraphRender:
    svg: str
    warning: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"svg": self.svg, "warning": self.warning}


@dataclass(frozen=True)
class _Placed:
    concept: Concept
    x: float
    y: float


def _text(x: float, y: float, value: str, *, anchor: str = "middle", size: int = 12) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-size="{size}" fill="{_TEXT_FILL}">{html.escape(value)}</text>'
    )


def _placeholder(title: str, reason: str) -> ConceptGraphRender:
    safe_title = html.escape(title or "concept graph")
    body = (
        f'<rect x="1" y="1" width="{SVG_WIDTH - 2}" height="{PLACEHOLDER_HEIGHT - 2}" '
        f'fill="{_PLACEHOLDER_FILL}" stroke="{_PLACEHOLDER_STROKE}" stroke-dasharray="6 4"/>'
        + _text(SVG_WIDTH / 2, PLACEHOLDER_HEIGHT / 2, f"图暂不可用：{reason}")
        + _text(SVG_WIDTH / 2, PLACEHOLDER_HEIGHT / 2 + 24, title or "concept graph", size=11)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_WIDTH} {PLACEHOLDER_HEIGHT}" '
        f'role="img" aria-label="{safe_title}"><title>{safe_title}</title>{body}</svg>'
    )
    return ConceptGraphRender(svg=svg, warning=reason)


def _clean_label(name: str, *, max_units: float = 0.0) -> str:
    """Collapse whitespace and clamp the label to the node's width budget.

    The glyph-width rule itself lives in
    :func:`deeptutor.review.text_metrics.truncate_to_units` so the concept
    graph and the built-in plot/bar label gate cannot drift apart. This
    wrapper only applies the graph's global ceiling and keeps the historical
    ``max_units <= 0`` convention (meaning "use the ceiling").
    """
    budget = MAX_LABEL_EM if max_units <= 0 else min(MAX_LABEL_EM, max_units)
    return truncate_to_units(name, max_units=budget)


def _duplicate_id(concepts: Sequence[Concept]) -> str:
    seen: set[str] = set()
    for concept in sorted(concepts, key=lambda item: (item.id, item.name)):
        if concept.id in seen:
            return concept.id
        seen.add(concept.id)
    return ""


def _prerequisite_map(concepts: Sequence[Concept]) -> dict[str, list[str]]:
    """Sorted, de-duplicated prerequisite ids per concept id."""
    return {
        concept.id: sorted(set(concept.prerequisites))
        for concept in sorted(concepts, key=lambda item: (item.id, item.name))
    }


def _find_cycle(prereqs: dict[str, list[str]]) -> list[str]:
    """Return one cycle as ``a -> b -> a``; empty when the graph is a DAG."""
    white, gray, black = 0, 1, 2
    color = {node: white for node in prereqs}
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        color[node] = gray
        stack.append(node)
        for parent in prereqs[node]:
            if color[parent] == gray:
                return stack[stack.index(parent) :] + [parent]
            if color[parent] == white:
                found = visit(parent)
                if found:
                    return found
        stack.pop()
        color[node] = black
        return None

    for node in sorted(prereqs):
        if color[node] == white:
            found = visit(node)
            if found:
                return found
    return []


def _layer_concepts(prereqs: dict[str, list[str]]) -> dict[str, int]:
    """Longest-path layering over a DAG, deterministic for a given input."""
    children: dict[str, list[str]] = {node: [] for node in prereqs}
    indegree: dict[str, int] = {node: len(parents) for node, parents in prereqs.items()}
    for node in sorted(prereqs):
        for parent in prereqs[node]:
            children[parent].append(node)
    for node in children:
        children[node].sort()

    layer = {node: 0 for node in prereqs}
    ready = [node for node in sorted(prereqs) if indegree[node] == 0]
    heapq.heapify(ready)
    while ready:
        node = heapq.heappop(ready)
        for child in children[node]:
            layer[child] = max(layer[child], layer[node] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    # Validation already excluded cycles, so every node was emitted.
    return layer


def _layout(
    groups: dict[int, list[Concept]],
    *,
    has_title: bool,
) -> tuple[int, float, float, float, float] | str:
    """Return ``(height, slot, node_w, node_h, title_pad)`` or a warning string."""
    layer_count = max(groups) + 1
    widest = max(len(group) for group in groups.values())
    slot = (SVG_WIDTH - 2 * MARGIN) / layer_count
    node_w = min(NODE_MAX_WIDTH, slot - COLUMN_GAP)
    if node_w < NODE_MIN_WIDTH:
        return (
            f"graph needs {layer_count} layers; columns would be narrower than "
            f"{NODE_MIN_WIDTH}px and unreadable"
        )

    title_pad = TITLE_HEIGHT if has_title else 12
    natural = widest * (NODE_HEIGHT + ROW_GAP) - ROW_GAP
    layer_floor = title_pad + layer_count * LAYER_MIN_HEIGHT + BOTTOM_PAD
    height = max(MIN_CANVAS_HEIGHT, title_pad + natural + BOTTOM_PAD, layer_floor)

    node_h = float(NODE_HEIGHT)
    if height > MAX_CANVAS_HEIGHT:
        height = float(MAX_CANVAS_HEIGHT)
        span = height - title_pad - BOTTOM_PAD
        # Rows shrink to fit the cap instead of overflowing it. Below the legible
        # floor we degrade explicitly: a 5px-tall box with overlapping neighbours
        # is not a graph, and silently drawing one is worse than a placeholder.
        node_h = min(NODE_HEIGHT, (span / widest) * 0.72)
        if node_h < MIN_ROW_NODE_HEIGHT:
            return (
                f"graph is too dense to render legibly "
                f"({widest} concepts in one layer)"
            )
    return int(round(height)), slot, node_w, node_h, title_pad


def _edge(source: _Placed, target: _Placed, node_w: float) -> str:
    x1 = source.x + node_w / 2
    x2 = target.x - node_w / 2
    bend = max(8.0, (x2 - x1) * 0.5)
    return (
        f'<path d="M {x1:.2f} {source.y:.2f} C {x1 + bend:.2f} {source.y:.2f} '
        f'{x2 - bend:.2f} {target.y:.2f} {x2:.2f} {target.y:.2f}" '
        f'fill="none" stroke="{_EDGE_STROKE}" stroke-width="1.2" '
        f'marker-end="url(#{_ARROW_ID})"/>'
    )


def _node(placed: _Placed, node_w: float, node_h: float, *, root: bool) -> str:
    x = placed.x - node_w / 2
    y = placed.y - node_h / 2
    fill = _ROOT_FILL if root else _NODE_FILL
    dash = ' stroke-dasharray="5 3"' if root else ""
    font_size = 11 if node_w >= 70 and node_h >= 22 else 9
    if node_h < 22:
        font_size = max(7, min(font_size, int(round(node_h * 0.62))))
    # Width budget for the label, in ems, minus the node's horizontal padding.
    max_em = max(1.0, (node_w - 10) * 1.0 / max(font_size, 1))
    # A short node cannot carry a full baseline offset; keep it proportional.
    baseline = placed.y + min(font_size * 0.36, node_h * 0.35)
    return (
        "<g>"
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{node_w:.2f}" height="{node_h:.2f}" '
        f'rx="{NODE_RADIUS}" fill="{fill}" stroke="{_NODE_STROKE}" '
        f'stroke-width="1.2"{dash}/>'
        + _text(
            placed.x,
            baseline,
            _clean_label(placed.concept.name, max_units=max_em),
            size=font_size,
        )
        + "</g>"
    )


def render_concept_graph(
    concepts: Sequence[Concept],
    *,
    title: str = "",
) -> ConceptGraphRender:
    """Render a concept prerequisite DAG to a self-contained SVG string."""
    items = list(concepts)
    if not items:
        return _placeholder(title, "concept graph needs at least one concept")
    if len(items) > MAX_NODES:
        return _placeholder(
            title,
            f"too many concepts for a readable graph ({len(items)} > {MAX_NODES})",
        )
    duplicate = _duplicate_id(items)
    if duplicate:
        return _placeholder(title, f"duplicate concept id '{duplicate}'")

    prereqs = _prerequisite_map(items)
    for concept_id in sorted(prereqs):
        if concept_id in prereqs[concept_id]:
            return _placeholder(title, f"self prerequisite on concept '{concept_id}'")
    for concept_id in sorted(prereqs):
        for parent in prereqs[concept_id]:
            if parent not in prereqs:
                return _placeholder(
                    title,
                    f"unknown prerequisite '{parent}' on concept '{concept_id}'",
                )

    cycle = _find_cycle(prereqs)
    if cycle:
        return _placeholder(title, "cycle detected: " + " -> ".join(cycle))

    layer = _layer_concepts(prereqs)
    groups: dict[int, list[Concept]] = {}
    for concept in sorted(items, key=lambda item: (item.id, item.name)):
        groups.setdefault(layer[concept.id], []).append(concept)

    laid_out = _layout(groups, has_title=bool(title))
    if isinstance(laid_out, str):
        return _placeholder(title, laid_out)
    height, slot, node_w, node_h, title_pad = laid_out

    span = height - title_pad - BOTTOM_PAD
    placed: dict[str, _Placed] = {}
    for layer_index in sorted(groups):
        group = groups[layer_index]
        center_x = MARGIN + (layer_index + 0.5) * slot
        for index, concept in enumerate(group):
            center_y = title_pad + (index + 0.5) * span / len(group)
            placed[concept.id] = _Placed(concept, center_x, center_y)

    edges: list[str] = []
    for concept_id in sorted(prereqs):
        for parent in prereqs[concept_id]:
            edges.append(_edge(placed[parent], placed[concept_id], node_w))

    nodes: list[str] = []
    for concept_id in sorted(placed):
        is_root = not prereqs[concept_id]
        nodes.append(_node(placed[concept_id], node_w, node_h, root=is_root))

    safe_title = html.escape(title or "concept graph")
    title_text = (
        _text(MARGIN, 26, title, anchor="start", size=13) if title else ""
    )
    marker = (
        f'<defs><marker id="{_ARROW_ID}" viewBox="0 0 8 8" refX="7" refY="4" '
        'markerWidth="8" markerHeight="8" orient="auto">'
        f'<path d="M0,0 L8,4 L0,8 z" fill="{_EDGE_STROKE}"/></marker></defs>'
    )
    body = (
        marker
        + title_text
        + f'<g class="cg-edges">{"".join(edges)}</g>'
        + f'<g class="cg-nodes">{"".join(nodes)}</g>'
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_WIDTH} {height}" '
        f'role="img" aria-label="{safe_title}"><title>{safe_title}</title>{body}</svg>'
    )
    return ConceptGraphRender(svg=svg)
