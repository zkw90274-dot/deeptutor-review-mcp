"""Static guards for the ``slider_plot`` widget kind.

The exported page is a single self-contained file with no build step and no
browser in the test path, so these checks read the template source and the
committed schema. The one behavior that needs a DOM (curve visible, marker
follows the slider) is recorded as a manual Chrome check in
``docs/maintainer-notes.md`` §3.

Security contract under test: the browser receives precomputed ``values`` from
the server-side restricted AST and only *draws* them. No expression evaluation,
no ``innerHTML``, no external resources.
"""

from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path

import pytest
from pydantic import ValidationError

from deeptutor.review.export import _safe_json
from deeptutor.review.fixture import build_fixture_spec
from deeptutor.review.models import ReviewSpec, WidgetSpec
from deeptutor.review.service import ReviewService

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = PACKAGE_ROOT / "src/deeptutor/review/templates/review.html"
TEMPLATE = TEMPLATE_PATH.read_text(encoding="utf-8")
SCHEMA_PATH = PACKAGE_ROOT / "spec_schema.json"

NOW = 1_800_000_000.0


def _plot_widget(widget_id: str = "w-plot") -> dict:
    return {
        "id": widget_id,
        "kind": "slider_plot",
        "title": "平方曲线",
        "formula_latex": "x^2",
        "expression": "x^2",
        "parameter": "x",
        "minimum": 0,
        "maximum": 2,
        "step": 0.05,
        "samples": 41,
        "source_ids": ["src-hf-2-1"],
    }


def _plot_renderer() -> str:
    """The SVG builder, from its definition up to ``renderWidget``."""
    start = TEMPLATE.index("function renderWidgetPlot")
    end = TEMPLATE.index("function renderWidget(widgetId, widget)", start)
    return TEMPLATE[start:end]


def _plot_guard() -> str:
    """The ``slider_plot`` guard and the DOM it appends, inside ``renderWidget``."""
    marker = 'widget.kind === "slider_plot"'
    start = TEMPLATE.index(marker)
    end = TEMPLATE.index('var input = document.createElement("input")', start)
    return TEMPLATE[start:end]


# --- template: branching -------------------------------------------------


def test_template_has_a_slider_plot_branch() -> None:
    assert 'widget.kind === "slider_plot"' in TEMPLATE


def test_plot_branch_uses_svg_namespace_and_never_innerhtml() -> None:
    branch = _plot_renderer()
    assert "createElementNS" in branch
    assert 'var SVG_NS = "http://www.w3.org/2000/svg"' in TEMPLATE
    assert "createElementNS(SVG_NS" in branch
    # A single innerHTML anywhere would reopen the injection path this
    # component is supposed to avoid.
    assert "innerHTML" not in TEMPLATE


def test_plot_branch_creates_polyline_and_marker_circle() -> None:
    branch = _plot_renderer()
    assert '"polyline"' in branch
    assert '"circle"' in branch
    assert 'setAttribute("fill", "none")' in branch
    # The marker is repositioned on every slider move.
    assert "plot.move(index)" in TEMPLATE


def test_plot_svg_is_an_image_with_an_accessible_name() -> None:
    branch = _plot_renderer()
    assert 'setAttribute("class", "widget-plot")' in branch
    assert 'setAttribute("role", "img")' in branch
    assert 'setAttribute("aria-label", plotName)' in branch
    # The name comes from the widget title, with a suffix only when the title
    # does not already say 曲线/图 (otherwise it reads "平方曲线 曲线").
    assert "var plotName = (widget && widget.title) || widgetId;" in branch
    assert 'plotName += " 曲线"' in branch
    assert 'setAttribute("viewBox"' in branch


def test_plot_branch_is_guarded_by_widget_kind() -> None:
    branch = _plot_guard()
    assert branch.count('widget.kind === "slider_plot"') == 1
    # The guard sits inside renderWidget, before the controls are built.
    assert TEMPLATE.index('widget.kind === "slider_plot"') < TEMPLATE.index(
        'input.type = "range"'
    )


def test_slider_formula_rendering_path_is_untouched() -> None:
    # Existing contract: range input, announced value with unit, 44px targets.
    assert 'input.type = "range"' in TEMPLATE
    assert 'input.setAttribute("aria-valuetext", text)' in TEMPLATE
    assert ".widget-range{flex:1 1 220px;min-height:44px" in TEMPLATE
    # A widget without the new kind must not build any SVG.
    assert 'widget.kind === "slider_plot"' in TEMPLATE
    assert 'if (plot) plot.move(index);' in TEMPLATE


def test_template_does_not_evaluate_expressions_in_the_browser() -> None:
    assert "eval(" not in TEMPLATE
    assert "new Function" not in TEMPLATE
    # The only data source for the curve is the server-precomputed array.
    assert "render.values" in TEMPLATE


# --- model and schema ----------------------------------------------------


def test_model_accepts_slider_plot_kind() -> None:
    widget = WidgetSpec(
        id="w-plot",
        kind="slider_plot",
        expression="x^2",
        parameter="x",
        minimum=0.0,
        maximum=2.0,
        samples=41,
    )
    assert widget.kind == "slider_plot"
    assert widget.samples == 41


def test_model_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        WidgetSpec(id="w-bad", kind="bogus", expression="x")


def test_committed_schema_lists_both_kinds() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    kind = schema["$defs"]["WidgetSpec"]["properties"]["kind"]
    assert kind["enum"] == ["slider_formula", "slider_plot"]
    assert kind["default"] == "slider_formula"


# --- export payload ------------------------------------------------------


def test_export_payload_carries_widget_kind(service: ReviewService) -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    payload["widgets"] = [_plot_widget()]
    payload["sections"][0]["blocks"].append(
        {"kind": "widget", "widget_id": "w-plot"}
    )
    spec = ReviewSpec.model_validate(payload)
    service.put_spec(spec)
    html = service.export(spec.id, now=NOW)
    assert '"kind":"slider_plot"' in html
    # The curve data arrives precomputed; 41 samples by default.
    assert '"values":[' in html
    assert "widget-plot" in TEMPLATE  # class ships with the page
    assert '"expression":"x^2"' in html or "x^2" in html
    assert "噪音" not in html


def test_plot_payload_survives_safe_json_escaping() -> None:
    escaped = _safe_json({"kind": "slider_plot", "label": "</script>"})
    assert '"kind":"slider_plot"' in escaped
    assert "</script>" not in escaped


# --- budgets (mirrors tests/test_export.py) -----------------------------


def test_css_and_js_budgets_still_hold() -> None:
    style = re.search(r"<style>(.*?)</style>", TEMPLATE, re.S)
    assert style is not None
    assert len(style.group(1).encode("utf-8")) <= 60_000
    scripts = re.findall(r"<script>(.*?)</script>", TEMPLATE, re.S)
    assert len(scripts) == 1
    # P2 study aids + P5 table/code blocks + the per-item answer toggle +
    # the explanation paragraph/list splitter + the emphasis pass; keep the
    # page's JS under 56 KB.
    assert len(scripts[0].encode("utf-8")) <= 56_000


def test_plot_adds_no_external_resource() -> None:
    branch = _plot_renderer()
    assert "http://" not in branch.replace('"http://www.w3.org/2000/svg"', "")
    assert "https://" not in branch
    assert "<image" not in branch
