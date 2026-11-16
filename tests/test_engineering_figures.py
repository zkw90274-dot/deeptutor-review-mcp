"""Engineering multi-panel figures: shared x-axis, sign colours, value labels.

The provider is deterministic: it draws from segments/points, never from
LLM-authored SVG. Positive/negative values use the textbook blue/orange
convention, but every value is also labelled so colour is not the only
information channel.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deeptutor.review.fixture import build_fixture_spec
from deeptutor.review.models import (
    FigureMark,
    FigurePanel,
    FigureSpec,
    ReviewSpec,
)
from deeptutor.review.render import render_figure
from deeptutor.review.service import ReviewService

CAPTION = (
    "正蓝负橙；剪力以微段顺时针转为正；弯矩以下侧受拉为正；"
    "单位 kN / kN·m。"
)


def _panels_figure() -> FigureSpec:
    return FigureSpec(
        id="fig-qm",
        kind="panels",
        title="简支梁剪力图与弯矩图",
        caption=CAPTION,
        x_label="x / m",
        source_ids=["src-hf-2-1"],
        panels=[
            FigurePanel(
                kind="shear",
                title="剪力图 Q",
                unit="kN",
                segments=[(0.0, 2.0, 5.0), (2.0, 4.0, -5.0)],
            ),
            FigurePanel(
                kind="moment",
                title="弯矩图 M",
                unit="kN·m",
                points=[
                    (0.0, 0.0),
                    (0.5, 2.5),
                    (1.0, 5.0),
                    (1.5, 7.5),
                    (2.0, 10.0),
                    (2.5, 7.5),
                    (3.0, 5.0),
                    (3.5, 2.5),
                    (4.0, 0.0),
                ],
                marks=[FigureMark(x=2.0, value=10.0, text="Mmax = 10 kN·m")],
            ),
        ],
    )


def test_panels_render_with_sign_colours_and_value_labels() -> None:
    rendered = render_figure(_panels_figure())

    assert rendered.warning == ""
    for needle in (
        "#1c60be",  # positive blue
        "#c4710c",  # negative orange
        "剪力图 Q",
        "弯矩图 M",
        "Mmax = 10 kN·m",
        "x / m",
        "kN·m",
    ):
        assert needle in rendered.svg, needle
    # One shared x-axis label, not one per panel.
    assert rendered.svg.count("x / m") == 1
    # Self-contained: no external image/link references.
    assert "<image" not in rendered.svg
    assert "https://" not in rendered.svg


def test_empty_panels_are_a_visible_placeholder() -> None:
    rendered = render_figure(FigureSpec(id="fig-empty", kind="panels"))

    assert rendered.warning
    assert "at least one panel" in rendered.warning


def test_panels_are_optional_and_keep_legacy_content_hash() -> None:
    original = build_fixture_spec()
    payload = original.model_dump(mode="json")
    payload["content_hash"] = ""

    rebuilt = ReviewSpec.model_validate(payload)

    assert rebuilt.content_hash == original.content_hash


def test_panels_figure_exports_into_the_review_page(service: ReviewService) -> None:
    original = build_fixture_spec()
    payload = original.model_dump(mode="json")
    payload["content_hash"] = ""
    payload["figures"].append(_panels_figure().model_dump(mode="json"))
    payload["sections"][0]["blocks"].insert(
        2, {"kind": "figure", "figure_id": "fig-qm"}
    )

    spec = ReviewSpec.model_validate(payload)
    service.put_spec(spec)
    html = service.export(spec.id)

    assert "剪力图 Q" in html
    assert "弯矩图 M" in html
    assert "Mmax = 10 kN·m" in html
    assert CAPTION in html


def test_panels_list_is_capped_at_four() -> None:
    with pytest.raises(ValidationError):
        FigureSpec(
            id="fig-too-many",
            kind="panels",
            panels=[FigurePanel(kind="line") for _ in range(5)],
        )
