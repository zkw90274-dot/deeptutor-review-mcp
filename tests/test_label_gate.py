"""Pure-geometry tests for the label bounding-box gate (RS-505).

These tests never touch ``render.py``: they build :class:`LabelBox` values by
hand so a failure points at the geometry rule, not at a renderer layout.
"""

from __future__ import annotations

import json
import math

import pytest

from deeptutor.review.label_gate import (
    LabelBox,
    LabelReport,
    check_labels,
    DEFAULT_OVERLAP_THRESHOLD,
)
from deeptutor.review.text_metrics import label_units, text_box, truncate_to_units


def box(
    text: str,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> LabelBox:
    return LabelBox(text=text, left=left, top=top, right=right, bottom=bottom)


def codes(report: LabelReport) -> list[str]:
    return [issue.code for issue in report.issues]


# --------------------------------------------------------------------------
# text_metrics (shared estimate the gate is built on)
# --------------------------------------------------------------------------


def test_label_units_counts_full_width_as_one_em() -> None:
    assert abs(label_units("ab") - 1.2) < 1e-9
    assert abs(label_units("品质") - 2.0) < 1e-9
    assert abs(label_units("A品") - 1.6) < 1e-9
    assert label_units("") == 0.0


def test_text_box_follows_the_anchor() -> None:
    start = text_box("ab", x=100.0, y=50.0, anchor="start", font_size=10.0)
    middle = text_box("ab", x=100.0, y=50.0, anchor="middle", font_size=10.0)
    end = text_box("ab", x=100.0, y=50.0, anchor="end", font_size=10.0)
    # 2 half-width glyphs at 10px = 12px wide.
    assert start == (100.0, 42.0, 112.0, 52.0)
    assert middle == (94.0, 42.0, 106.0, 52.0)
    assert end == (88.0, 42.0, 100.0, 52.0)


def test_text_box_vertical_estimate_is_baseline_relative() -> None:
    _, top, _, bottom = text_box("x", x=0.0, y=100.0, font_size=20.0)
    assert top == 100.0 - 20.0 * 0.8
    assert bottom == 100.0 + 20.0 * 0.2


def test_text_box_rejects_unknown_anchor_and_bad_geometry() -> None:
    with pytest.raises(ValueError):
        text_box("x", x=0.0, y=0.0, anchor="centre")
    with pytest.raises(ValueError):
        text_box("x", x=math.nan, y=0.0)
    with pytest.raises(ValueError):
        text_box("x", x=0.0, y=0.0, font_size=0.0)


def test_truncate_to_units_keeps_short_labels_verbatim() -> None:
    assert truncate_to_units("  a  b ", max_units=10.0) == "a b"
    assert truncate_to_units("品质", max_units=2.0) == "品质"


def test_truncate_to_units_reserves_one_em_for_the_ellipsis() -> None:
    text = "品质因数与通频带"
    cut = truncate_to_units(text, max_units=3.0)
    assert cut.endswith("…")
    assert label_units(cut) <= 3.0
    assert cut.startswith("品质")


# --------------------------------------------------------------------------
# check_labels: empty / trivial input
# --------------------------------------------------------------------------


def test_no_labels_is_ok() -> None:
    report = check_labels([], width=640, height=360)
    assert report.ok is True
    assert report.issues == ()
    assert report.boxes == 0


def test_single_label_inside_the_canvas_is_ok() -> None:
    report = check_labels([box("a", 10, 10, 20, 30)], width=640, height=360)
    assert report.ok is True
    assert report.issues == ()
    assert report.boxes == 1


# --------------------------------------------------------------------------
# overlaps
# --------------------------------------------------------------------------


def test_identical_boxes_overlap_ratio_is_one() -> None:
    first = box("甲", 10, 10, 110, 30)
    second = box("乙", 10, 10, 110, 30)
    assert first.overlaps(second) == 1.0
    report = check_labels([first, second], width=640, height=360)
    assert codes(report) == ["label_overlap"]
    assert report.ok is True  # warning, not error
    assert report.issues[0].severity == "warning"
    assert "ratio 1.00" in report.issues[0].detail


def test_partial_intersection_ratio_is_intersection_over_smaller() -> None:
    small = box("小", 0, 0, 10, 10)  # area 100
    large = box("大", 5, 0, 105, 50)  # area 5000
    # Intersection: 5 x 10 = 50 -> 50 / min(100, 5000) = 0.5
    assert small.overlaps(large) == 0.5
    assert large.overlaps(small) == 0.5


def test_fully_contained_box_ratio_is_one() -> None:
    outer = box("外", 0, 0, 100, 100)
    inner = box("内", 40, 40, 60, 60)
    assert outer.overlaps(inner) == 1.0


def test_disjoint_boxes_overlap_is_zero() -> None:
    assert box("a", 0, 0, 10, 10).overlaps(box("b", 20, 20, 30, 30)) == 0.0
    # Touching edges are not an overlap.
    assert box("a", 0, 0, 10, 10).overlaps(box("b", 10, 0, 20, 10)) == 0.0


def test_threshold_boundary_is_strictly_greater_than() -> None:
    # Two boxes of equal area overlapping by exactly 15% of the smaller box.
    first = box("甲", 0, 0, 100, 10)  # area 1000
    second = box("乙", 85, 0, 185, 10)  # intersection 15 x 10 = 150 -> 0.15
    assert second.overlaps(first) == 0.15
    at_threshold = check_labels(
        [first, second], width=640, height=360, overlap_threshold=0.15
    )
    assert codes(at_threshold) == []
    just_below = check_labels(
        [first, second], width=640, height=360, overlap_threshold=0.1500001
    )
    assert codes(just_below) == []
    just_above = check_labels(
        [first, second], width=640, height=360, overlap_threshold=0.1499999
    )
    assert codes(just_above) == ["label_overlap"]


def test_default_threshold_matches_the_pdf_gate() -> None:
    assert DEFAULT_OVERLAP_THRESHOLD == 0.15


def test_float_rounding_at_the_threshold_does_not_report_zero_slack() -> None:
    """A pair one ULP above the threshold must not print "0.15 > 0.15".

    Renderer coordinates come from divisions, so exact-threshold pairs can
    land fractionally above it. The epsilon keeps the detail line honest.
    """
    first = box("甲", 0.0, 0.0, 100.0, 10.0)
    second = box("乙", 84.99999999999999, 0.0, 184.99999999999999, 10.0)
    ratio = second.overlaps(first)
    assert ratio > 0.15  # 0.15000000000000013, one ULP over
    report = check_labels(
        [first, second], width=640, height=360, overlap_threshold=0.15
    )
    assert report.issues == ()


def test_each_overlapping_pair_is_reported_once() -> None:
    boxes = [box(name, 0, 0, 100, 20) for name in ("甲", "乙", "丙")]
    report = check_labels(boxes, width=640, height=360)
    # Three mutually overlapping boxes -> C(3, 2) = 3 pairs, not 6 directions.
    assert codes(report) == ["label_overlap", "label_overlap", "label_overlap"]
    assert report.boxes == 3


# --------------------------------------------------------------------------
# canvas bounds
# --------------------------------------------------------------------------


def test_box_past_the_left_edge_is_an_error() -> None:
    report = check_labels([box("左", -5, 10, 20, 30)], width=640, height=360)
    assert codes(report) == ["label_out_of_canvas"]
    assert report.issues[0].severity == "error"
    assert report.ok is False


def test_box_past_the_top_edge_is_an_error() -> None:
    report = check_labels([box("上", 10, -5, 20, 30)], width=640, height=360)
    assert codes(report) == ["label_out_of_canvas"]
    assert report.ok is False


def test_box_past_the_right_edge_is_an_error() -> None:
    report = check_labels([box("右", 620, 10, 660, 30)], width=640, height=360)
    assert codes(report) == ["label_out_of_canvas"]
    assert report.ok is False


def test_box_past_the_bottom_edge_is_an_error() -> None:
    report = check_labels([box("下", 10, 340, 20, 370)], width=640, height=360)
    assert codes(report) == ["label_out_of_canvas"]
    assert report.ok is False


def test_margin_tolerance_absorbs_small_overshoot() -> None:
    touching = box("边", -1.0, 0.0, 100.0, 20.0)
    assert codes(check_labels([touching], width=640, height=360)) == []
    # Tolerance is exclusive: one pixel of slack covers exactly -1.0.
    assert codes(
        check_labels([touching], width=640, height=360, margin_tolerance=0.5)
    ) == ["label_out_of_canvas"]
    assert codes(
        check_labels([touching], width=640, height=360, margin_tolerance=1.0)
    ) == []


# --------------------------------------------------------------------------
# invalid geometry
# --------------------------------------------------------------------------


def test_non_finite_coordinates_are_a_geometry_error() -> None:
    for value in (math.nan, math.inf, -math.inf):
        report = check_labels(
            [box("坏", value, 0, 10, 10)], width=640, height=360
        )
        assert codes(report) == ["label_geometry_invalid"]
        assert report.issues[0].severity == "error"
        assert report.ok is False


def test_inverted_box_is_a_geometry_error() -> None:
    inverted_x = check_labels(
        [box("反", 20, 0, 10, 10)], width=640, height=360
    )
    inverted_y = check_labels(
        [box("反", 0, 20, 10, 10)], width=640, height=360
    )
    assert codes(inverted_x) == ["label_geometry_invalid"]
    assert codes(inverted_y) == ["label_geometry_invalid"]


def test_geometry_errors_are_not_also_tested_for_bounds() -> None:
    report = check_labels(
        [box("坏", math.nan, math.nan, math.nan, math.nan)], width=640, height=360
    )
    assert codes(report) == ["label_geometry_invalid"]


# --------------------------------------------------------------------------
# report shape, truncation, determinism
# --------------------------------------------------------------------------


def test_issue_list_is_capped_and_flagged() -> None:
    boxes = [box(f"标{index}", index * 2, 0, index * 2 + 1, 10) for index in range(30)]
    report = check_labels(
        boxes, width=20, height=10, overlap_threshold=1.0, max_issues=5
    )
    assert len(report.issues) == 6  # 5 kept + the truncation notice
    assert report.issues[-1].code == "issues_truncated"
    assert report.issues[-1].severity == "error"  # errors were omitted
    assert "errors" in report.issues[-1].detail


def test_truncation_notice_is_a_warning_when_only_warnings_were_cut() -> None:
    boxes = [box(f"标{index}", 0, 0, 100, 20) for index in range(6)]
    report = check_labels(boxes, width=640, height=360, max_issues=2)
    assert report.issues[-1].code == "issues_truncated"
    assert report.issues[-1].severity == "warning"
    assert "13 more issues omitted" in report.issues[-1].detail
    assert report.ok is True


def test_overlap_sweep_reports_the_same_pairs_as_brute_force() -> None:
    boxes = [
        box("甲", 0, 0, 10, 10),
        box("乙", 5, 0, 15, 10),
        box("丙", 12, 0, 22, 10),
        box("丁", 9, 5, 30, 8),
        box("戊", 100, 0, 110, 10),
    ]
    expected = [
        (first, second)
        for index, first in enumerate(boxes)
        for second in boxes[index + 1 :]
        if first.overlaps(second) > DEFAULT_OVERLAP_THRESHOLD
    ]
    report = check_labels(boxes, width=200, height=50, max_issues=100)
    assert codes(report) == ["label_overlap"] * len(expected)
    details = [issue.detail for issue in report.issues]
    for first, second in expected:
        assert any(
            f"'{first.text}'" in detail and f"'{second.text}'" in detail
            for detail in details
        )


def test_as_dict_is_json_serialisable() -> None:
    report = check_labels(
        [box("甲", 0, 0, 100, 20), box("乙", 0, 0, 100, 20)],
        width=640,
        height=360,
    )
    payload = report.as_dict()
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
    assert payload["ok"] is True
    assert payload["boxes"] == 2
    assert payload["issues"][0]["code"] == "label_overlap"
    assert payload["issues"][0]["severity"] == "warning"


def test_same_input_produces_the_same_report() -> None:
    boxes = [
        box("甲", 0, 0, 100, 20),
        box("乙", 90, 0, 190, 20),
        box("丙", 500, 350, 700, 380),
    ]
    first = check_labels(boxes, width=640, height=360)
    second = check_labels(boxes, width=640, height=360)
    assert first == second
    assert first.as_dict() == second.as_dict()


def test_invalid_arguments_fail_visible_instead_of_raising() -> None:
    bad_width = check_labels([box("a", 0, 0, 1, 1)], width=0, height=360)
    bad_threshold = check_labels(
        [box("a", 0, 0, 1, 1)], width=640, height=360, overlap_threshold=1.5
    )
    bad_tolerance = check_labels(
        [box("a", 0, 0, 1, 1)], width=640, height=360, margin_tolerance=-1
    )
    for report in (bad_width, bad_threshold, bad_tolerance):
        assert codes(report) == ["invalid_argument"]
        assert report.ok is False


def test_detail_text_is_truncated_for_long_labels() -> None:
    long_name = "非" * 80
    report = check_labels(
        [box(long_name, -10, 0, 10, 20)], width=640, height=360
    )
    assert long_name not in report.issues[0].detail
    assert "…" in report.issues[0].detail
