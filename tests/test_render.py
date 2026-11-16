from __future__ import annotations

import hashlib

from deeptutor.review.fixture import build_fixture_spec
from deeptutor.review.models import FigureSpec, ReviewSpec, WidgetSpec
from deeptutor.review.render import render_figure, render_widget
from deeptutor.review.service import ReviewService


def test_plot_is_deterministic_svg() -> None:
    figure = FigureSpec(
        id="fig-plot",
        kind="plot",
        title="谐振曲线",
        points=[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)],
        x_label="f / kHz",
        y_label="I / A",
    )
    first = render_figure(figure)
    second = render_figure(figure)
    assert first.svg == second.svg
    assert "<polyline" in first.svg
    assert "谐振曲线" in first.svg
    assert "<script" not in first.svg
    assert first.warning == ""


def test_bar_and_placeholder_are_visible() -> None:
    bar = render_figure(
        FigureSpec(
            id="fig-bar",
            kind="bar",
            title="频谱",
            bar_labels=["f1", "f2", "f3"],
            bar_values=[1.0, 3.0, 2.0],
        )
    )
    assert "<rect" in bar.svg
    assert "f2" in bar.svg
    placeholder = render_figure(FigureSpec(id="fig-x", kind="unknown", title="未知图"))
    assert placeholder.svg
    assert "unsupported figure kind" in placeholder.warning
    assert "未知图" in placeholder.svg


def test_slider_precomputes_41_values() -> None:
    widget = WidgetSpec(
        id="w1",
        title="平方",
        formula_latex="x^2",
        expression="x^2",
        parameter="x",
        minimum=0.0,
        maximum=2.0,
        step=0.1,
    )
    rendered = render_widget(widget)
    assert rendered.warning == ""
    assert len(rendered.values) == 41
    assert rendered.values[0] == 0.0
    assert rendered.values[-1] == 4.0


def test_slider_rejects_code_execution() -> None:
    for expression in (
        '__import__("os")',
        "x.__class__",
        "open('x')",
        "x ** 100",
        "x if x else 1",
    ):
        rendered = render_widget(
            WidgetSpec(id="w-bad", expression=expression, parameter="x")
        )
        assert rendered.values == []
        assert rendered.warning


def test_slider_allows_whitelisted_functions() -> None:
    """``sqrt`` must survive the name check when used as a call target.

    The callee check has to run before the bare-name check, otherwise the
    function whitelist is unreachable and every ``sqrt(x)`` slider dies with
    ``unsupported name 'sqrt'`` while only ``x ** 0.5`` keeps working.
    """
    rendered = render_widget(
        WidgetSpec(
            id="w-sqrt", title="根号", expression="sqrt(x)",
            parameter="x", minimum=0.0, maximum=4.0, step=0.1,
        )
    )
    assert rendered.warning == ""
    assert len(rendered.values) == 41
    assert rendered.values[-1] == 2.0


def test_slider_rejects_a_bare_function_name() -> None:
    """A whitelisted name outside a call is still a bare name."""
    rendered = render_widget(
        WidgetSpec(id="w-bare", expression="sqrt", parameter="x")
    )
    assert rendered.values == []
    assert "unsupported name 'sqrt'" in rendered.warning


def test_slider_rejects_shadowed_parameter_in_call() -> None:
    """The parameter name may not be smuggled through a call target."""
    rendered = render_widget(
        WidgetSpec(id="w-shadow", expression="x(x)", parameter="x")
    )
    assert rendered.values == []
    assert "unsupported function call" in rendered.warning


def test_export_renders_figure_and_widget(service: ReviewService) -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    payload["figures"] = [
        {
            "id": "fig-1",
            "kind": "plot",
            "title": "测试曲线",
            "points": [[0, 0], [1, 1], [2, 0]],
            "source_ids": ["src-hf-2-1"],
        }
    ]
    payload["widgets"] = [
        {
            "id": "w-1",
            "title": "平方滑块",
            "formula_latex": "x^2",
            "expression": "x^2",
            "parameter": "x",
            "minimum": 0,
            "maximum": 2,
            "step": 0.1,
            "source_ids": ["src-hf-2-1"],
        }
    ]
    payload["sections"][0]["blocks"].extend(
        [
            {"kind": "figure", "figure_id": "fig-1"},
            {"kind": "widget", "widget_id": "w-1"},
        ]
    )
    spec = ReviewSpec.model_validate(payload)
    service.put_spec(spec)
    html = service.export(spec.id, now=1_800_000_000.0)
    assert "fig-1" in html
    # SVG is embedded in the escaped JSON payload, so tag delimiters are
    # <; the element name and the generated SVG namespace remain.
    assert "polyline" in html
    assert "widget-range" in html
    assert "x^2" in html
    assert "http://www.w3.org/2000/svg" in html


def test_tikz_figure_uses_the_provider(monkeypatch) -> None:
    from deeptutor.review import tikz

    calls = []

    def fake_render(source: str, *, title: str = "", check_layout: bool = True):
        calls.append((source, title))
        return tikz.TikzRender(
            svg='<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>'
        )

    monkeypatch.setattr(tikz, "render_tikz_document", fake_render)
    figure = FigureSpec(
        id="fig-tikz",
        kind="tikz",
        title="谐振电路",
        tikz=r"\draw (0,0) -- (1,1);",
    )
    rendered = render_figure(figure)
    assert calls == [(r"\draw (0,0) -- (1,1);", "谐振电路")]
    assert rendered.warning == ""
    assert "<path" in rendered.svg


def test_tikz_failure_stays_visible(monkeypatch) -> None:
    from deeptutor.review import tikz

    def fail(source: str, *, title: str = "", check_layout: bool = True):
        raise tikz.TikzRenderError("xelatex unavailable: missing")

    monkeypatch.setattr(tikz, "render_tikz_document", fail)
    rendered = render_figure(
        FigureSpec(id="fig-tikz-bad", kind="tikz", title="坏图", tikz=r"\draw (0,0);")
    )
    assert "xelatex unavailable" in rendered.warning
    assert "图暂不可用" in rendered.svg


def test_bar_truncates_long_labels_to_the_slot_width() -> None:
    """20 long CJK labels are clipped per slot, so the gate stays quiet (V1).

    Before the fix a fixed 12-character cut left 120px labels in 27px slots,
    which the gate reported as overlapping. The budget is now ``(slot - 4) /
    font_size``, so neighbouring labels keep a 4px gap.
    """
    figure = FigureSpec(
        id="fig-bar-crowded",
        kind="bar",
        title="二十项长标签",
        bar_labels=[f"第{index}个很长的类别名称" for index in range(20)],
        bar_values=[float(index + 1) for index in range(20)],
    )
    rendered = render_figure(figure)
    assert rendered.warning == ""
    assert "<rect" in rendered.svg
    # Budgeted, marked with an ellipsis, and no longer the 12-character stub.
    assert "第…" in rendered.svg
    assert "第0个很长的类别名称" not in rendered.svg


def test_bar_with_three_short_labels_is_warning_free() -> None:
    figure = FigureSpec(
        id="fig-bar-clean",
        kind="bar",
        title="三项",
        bar_labels=["甲", "乙", "丙"],
        bar_values=[1.0, 2.0, 3.0],
    )
    assert render_figure(figure).warning == ""


def test_plot_with_huge_axis_values_stays_inside_the_canvas() -> None:
    """V2: shrink the tick font, then thin the ticks; never cut the numbers.

    ``2.25e+12`` is 48px at 10px and 38.4px at 8px, both wider than the
    ``MARGIN - 16`` budget, so the y axis drops to three ticks at 8px. The x
    axis (0..2) keeps five ticks at 10px. Neither now leaves the canvas.
    """
    figure = FigureSpec(
        id="fig-plot-huge",
        kind="plot",
        title="大数",
        points=[(0.0, 0.0), (1.0, 1e12), (2.0, 3e12)],
    )
    rendered = render_figure(figure)
    assert rendered.warning == ""
    # Numerals are never truncated: the widest tick value survives verbatim.
    assert "3e+12" in rendered.svg
    # Only the overflowing y axis adapted: three end-anchored y labels at 8px,
    # five middle-anchored x labels at the default 10px.
    assert rendered.svg.count('text-anchor="end"') == 3
    assert rendered.svg.count('font-size="8"') == 3
    assert rendered.svg.count('text-anchor="middle"') == 5


def test_overlong_axis_title_still_reports_out_of_canvas() -> None:
    """Axis titles are user text, not numbers, so they are never truncated.

    The gate is the fallback for anything the tick layout cannot fix.
    """
    figure = FigureSpec(
        id="fig-plot-long-title",
        kind="plot",
        title="超长轴标题",
        points=[(0.0, 0.0), (1.0, 1.0)],
        x_label="很长的轴标题" * 20,
    )
    rendered = render_figure(figure)
    assert "label_out_of_canvas" in rendered.warning
    assert "很长的轴标题" * 20 in rendered.svg


def test_medium_inputs_keep_the_pre_adaptation_bytes() -> None:
    """Regression pin: comfortable inputs render exactly as before WS-B.

    The hashes were recorded from the fixed-budget renderer. A change to tick
    order, coordinate formatting or label text on ordinary figures breaks
    them.
    """
    bar = FigureSpec(
        id="fig-bar-base",
        kind="bar",
        title="三项",
        bar_labels=["甲", "乙", "丙"],
        bar_values=[1.0, 2.0, 3.0],
    )
    plot = FigureSpec(
        id="fig-plot-base",
        kind="plot",
        title="谐振曲线",
        points=[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)],
        x_label="f / kHz",
        y_label="I / A",
    )
    assert hashlib.sha256(render_figure(bar).svg.encode()).hexdigest() == (
        "59399ebdccbce241da2a50dc305eca6ee669ce3518df5c92bd838bce5b4ccca2"
    )
    assert hashlib.sha256(render_figure(plot).svg.encode()).hexdigest() == (
        "093c6989dab7a68492141950a3f09c8b6c1b94b78a2590f72a114cb18d879907"
    )


def test_label_warnings_do_not_change_the_svg() -> None:
    """The gate is diagnostic only: a noisy figure renders exactly as before.

    200 labels leave a 2.7px slot, below the truncation floor (one full-width
    glyph plus the ellipsis). The renderer keeps the readable label and the
    gate reports the overlap — the fallback the adaptive pass must not hide.
    """
    crowded = FigureSpec(
        id="fig-bar-extreme",
        kind="bar",
        title="两百项",
        bar_labels=[f"第{index}个很长的类别名称" for index in range(200)],
        bar_values=[float(index + 1) for index in range(200)],
    )
    rendered = render_figure(crowded)
    assert "label_overlap" in rendered.warning
    assert "<rect" in rendered.svg
    # No warning text leaks into the drawing itself.
    assert "label_overlap" not in rendered.svg
    assert "label_out_of_canvas" not in rendered.svg
    # Deterministic: the warning does not perturb the bytes.
    assert render_figure(crowded).svg == rendered.svg


def test_nan_slider_range_is_reported_as_a_range_problem() -> None:
    """NaN compares false against everything, so the ``<=`` guard passes it.

    The failure must be attributed to the range, not to the expression —
    otherwise the author is sent hunting through a formula that is fine.
    """
    widget = WidgetSpec(
        id="w-nan", title="坏区间", expression="x^2", parameter="x",
        minimum=float("nan"), maximum=1.0,
    )
    rendered = render_widget(widget)
    assert rendered.values == []
    assert "range" in rendered.warning.lower()


def test_placeholder_text_is_bounded() -> None:
    """A long tool message must not push the placeholder text off-canvas."""
    from deeptutor.review.render import _placeholder

    reason = "xelatex exploded: " + "x" * 500
    rendered = _placeholder("占位", reason)

    assert rendered.warning == reason, "the warning keeps the full detail"
    assert "x" * 40 + "x" not in rendered.svg, "the drawn text is truncated"
    assert "…" in rendered.svg
