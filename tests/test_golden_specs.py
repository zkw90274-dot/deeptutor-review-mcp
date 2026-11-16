"""RS-1301: the golden spec set as a regression baseline.

The three JSON files under ``tests/golden/`` are the deterministic inputs the
export pipeline is measured against (phase-28 §3.1). These tests prove three
things and no more:

1. each file is a valid :class:`ReviewSpec` that passes every QA gate;
2. each spec exports through ``ReviewService.export`` without copying export
   logic into the test;
3. two exports of the same spec at the same ``now`` are byte-identical — the
   golden-set version of acceptance matrix #7.

The coverage assertions pin the contract itself: if a future edit shrinks the
20-label bar or drops the huge-axis plot, the failure names the missing case
instead of silently weakening the baseline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from deeptutor.review.models import ReviewSpec
from deeptutor.review.qa import run_gates
from deeptutor.review.service import ReviewService

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
GOLDEN_IDS = ("golden-formula", "golden-figures", "golden-widgets")

#: ``window.__REVIEW__`` carries the exported payload on one compact line.
_PAYLOAD = re.compile(r"window\.__REVIEW__ = (\{.*?\});\n", re.DOTALL)

#: Matches ``tests/test_export.py`` and ``scripts/build_golden_specs.py``.
NOW = 1_800_000_000.0

#: A "long Chinese label" for the V1/V5 stress cases. The old renderer cut
#: labels at 12 characters, so a label must be longer than that to exercise
#: the budget-based truncation path.
LONG_LABEL_CHARS = 13


def load_spec(spec_id: str) -> ReviewSpec:
    payload = json.loads(
        (GOLDEN_DIR / f"{spec_id}.json").read_text(encoding="utf-8")
    )
    return ReviewSpec.model_validate(payload)


def export_payload(service: ReviewService, spec: ReviewSpec) -> dict:
    """Export through the service and return the page payload.

    The payload is the product the browser consumes; parsing it here does not
    replicate any export logic, it only reads the finished document.
    """
    service.put_spec(spec)
    document = service.export(spec.id, now=NOW)
    match = _PAYLOAD.search(document)
    assert match is not None, "export payload is missing from the document"
    return json.loads(match.group(1))


@pytest.mark.parametrize("spec_id", GOLDEN_IDS)
def test_golden_spec_is_valid_and_passes_gates(spec_id: str) -> None:
    spec = load_spec(spec_id)
    assert spec.id == spec_id
    assert spec.content_hash, "content_hash must be persisted in the JSON"
    report = run_gates(spec)
    assert report.passed, report.model_dump()


@pytest.mark.parametrize("spec_id", GOLDEN_IDS)
def test_golden_export_is_byte_identical(
    spec_id: str, service: ReviewService
) -> None:
    spec = load_spec(spec_id)
    service.put_spec(spec)
    first = service.export(spec.id, now=NOW)
    second = service.export(spec.id, now=NOW)
    assert first == second
    assert first.encode("utf-8") == second.encode("utf-8")
    assert spec.title in first
    if spec.figures or any(concept.prerequisites for concept in spec.concepts):
        # The SVG lives inside the JSON payload, where ``_safe_json`` rewrites
        # every less-than sign as the six-character unicode escape so learner
        # content cannot close a tag.
        assert "\\u003csvg" in first, "figures and the concept graph must be inlined"


def test_formula_golden_covers_three_item_kinds_and_both_math_levels() -> None:
    spec = load_spec("golden-formula")
    assert {item.kind for item in spec.items} == {
        "single_choice",
        "numeric",
        "flashcard",
    }
    assert all(item.source_ids for item in spec.items)

    texts = [item.stem for item in spec.items]
    texts.extend(
        block.text
        for section in spec.sections
        for block in section.blocks
        if hasattr(block, "text")
    )
    assert any("$$" in text for text in texts), "block-level LaTeX is missing"
    assert any(
        "$" in text and "$$" not in text for text in texts
    ), "inline LaTeX is missing"
    assert any(
        len(item.explanation) > 200 for item in spec.items
    ), "the long-explanation case is missing"


def test_figures_golden_covers_v1_v2_and_the_concept_graph() -> None:
    spec = load_spec("golden-figures")
    figures = {figure.id: figure for figure in spec.figures}

    plot = figures["fig-gf2-large-ticks"]
    assert plot.kind == "plot"
    assert max(abs(value) for _, value in plot.points) >= 1e11, (
        "V2 needs axis values large enough to widen the tick labels"
    )

    bar = figures["fig-gf2-long-labels"]
    assert bar.kind == "bar"
    assert len(bar.bar_labels) == 20
    assert len(bar.bar_values) == len(bar.bar_labels)
    assert all(
        len(label) >= LONG_LABEL_CHARS for label in bar.bar_labels
    ), "every bar label must cross the old 12-character truncation boundary"

    assert any(
        concept.prerequisites for concept in spec.concepts
    ), "the concept graph needs at least one prerequisite edge"
    assert all(figure.kind != "tikz" for figure in spec.figures)


def test_widgets_golden_covers_both_sliders_and_narrow_screen_stress() -> None:
    spec = load_spec("golden-widgets")
    assert {widget.kind for widget in spec.widgets} == {
        "slider_formula",
        "slider_plot",
    }
    assert all(widget.expression.strip() for widget in spec.widgets)
    assert all(widget.samples >= 5 for widget in spec.widgets)

    bars = [figure for figure in spec.figures if figure.kind == "bar"]
    assert bars, "the narrow-screen bar case is missing"
    assert any(
        len(bar.bar_labels) >= 12
        and all(len(label) >= LONG_LABEL_CHARS for label in bar.bar_labels)
        for bar in bars
    ), "narrow-screen stress needs at least 12 long Chinese labels"
    assert any(
        len(figure.caption) >= 20 for figure in spec.figures
    ), "the narrow-screen caption case is missing"
    assert all(figure.kind != "tikz" for figure in spec.figures)


def test_golden_assets_render_instead_of_falling_back(
    service: ReviewService,
) -> None:
    """The baseline is only useful if the renderers accept its inputs."""
    figures_spec = load_spec("golden-figures")
    figures_payload = export_payload(service, figures_spec)
    for figure in figures_spec.figures:
        rendered = figures_payload["figures"][figure.id]["render"]
        assert rendered["warning"] == ""
        assert rendered["svg"].startswith("<svg")
        assert "图暂不可用" not in rendered["svg"]
    assert figures_payload["concept_graph"]["svg"].startswith("<svg")

    widgets_spec = load_spec("golden-widgets")
    widgets_payload = export_payload(service, widgets_spec)
    for widget in widgets_spec.widgets:
        rendered = widgets_payload["widgets"][widget.id]["render"]
        assert rendered["warning"] == "", rendered["warning"]
        assert len(rendered["values"]) == widget.samples
