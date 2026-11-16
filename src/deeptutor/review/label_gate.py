"""Pure-geometry label gate for server-rendered SVG figures (RS-505).

The built-in plot and bar renderers place every label themselves at a known
coordinate and font size, so their bounding boxes can be computed
analytically — no rasterisation, no PDF probe, no browser. This module owns
only the geometry check; :mod:`deeptutor.review.text_metrics` owns the width
estimate the boxes are built from.

Checks
------
* ``label_geometry_invalid`` (error): non-finite coordinates or an inverted
  box (``right < left`` / ``bottom < top``). Such a box cannot be tested for
  bounds or overlap, so it is reported once and skipped.
* ``label_out_of_canvas`` (error): a box lies outside the canvas by more than
  ``margin_tolerance``.
* ``label_overlap`` (warning): the intersection covers more than
  ``overlap_threshold`` of the smaller box. Slight overlap can still be
  readable, so it warns rather than blocks; the default ``0.15`` is the same
  threshold the PDF layout gate uses.
* ``issues_truncated`` (warning): the issue list hit ``max_issues``.

A visual readability judgement (is the label *legible* on top of a data line?)
is explicitly out of scope: it needs eyes or a vision model, and guessing it
would produce false confidence.

Overlap checking sorts by left edge and stops the inner loop once a box starts
past the current box's right edge, so the work is O(n log n) plus the actual
overlapping pairs. Issue details stop at ``max_issues``; the truncation notice
still reports the true omitted count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

#: Default overlap ratio above which two boxes are flagged.
DEFAULT_OVERLAP_THRESHOLD = 0.15

#: Canvas slack, in pixels, before a box counts as outside.
DEFAULT_MARGIN_TOLERANCE = 1.0

#: Upper bound on reported issues so a pathological figure cannot produce an
#: unbounded JSON payload.
MAX_ISSUES = 100

#: Slack added to ``overlap_threshold`` before a pair counts as overlapping.
#: Renderer coordinates are computed from divisions, so a pair that is exactly
#: at the threshold arithmetically can land one ULP above it; without this
#: guard the detail line reads "ratio 0.15 > 0.15", which looks like a bug.
_OVERLAP_EPSILON = 1e-9

_DETAIL_TEXT_LIMIT = 40


@dataclass(frozen=True)
class LabelBox:
    """One label's bounding box in SVG user units (y grows downwards)."""

    text: str
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return self.width * self.height

    def overlaps(self, other: LabelBox) -> float:
        """Intersection area divided by the smaller box's area, in ``0..1``.

        Returns ``0.0`` when the boxes do not intersect or when either box has
        zero area (a degenerate box cannot meaningfully "overlap").
        """
        intersection_width = min(self.right, other.right) - max(
            self.left, other.left
        )
        if intersection_width <= 0:
            return 0.0
        intersection_height = min(self.bottom, other.bottom) - max(
            self.top, other.top
        )
        if intersection_height <= 0:
            return 0.0
        smaller = min(self.area, other.area)
        if smaller <= 0:
            return 0.0
        return (intersection_width * intersection_height) / smaller


@dataclass(frozen=True)
class LabelIssue:
    code: str
    detail: str
    severity: Literal["error", "warning"] = "warning"

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "detail": self.detail,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class LabelReport:
    """Immutable gate result.

    ``ok`` is ``False`` only when at least one issue is an ``error``.
    """

    ok: bool
    issues: tuple[LabelIssue, ...] = ()
    boxes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "boxes": self.boxes,
            "issues": [issue.as_dict() for issue in self.issues],
        }


def _short(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _DETAIL_TEXT_LIMIT:
        return collapsed
    return collapsed[: _DETAIL_TEXT_LIMIT - 1] + "…"


def _is_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _geometry_error(box: LabelBox) -> str:
    """Return a human-readable reason when *box* cannot be tested, else ``""``."""
    values = (box.left, box.top, box.right, box.bottom)
    if not all(_is_finite(value) for value in values):
        return "non-finite coordinate"
    if box.right < box.left:
        return f"right ({box.right:.2f}) < left ({box.left:.2f})"
    if box.bottom < box.top:
        return f"bottom ({box.bottom:.2f}) < top ({box.top:.2f})"
    return ""


def _bounds_issue(
    box: LabelBox, *, width: float, height: float, tolerance: float
) -> LabelIssue | None:
    outside = (
        box.left < -tolerance
        or box.top < -tolerance
        or box.right > width + tolerance
        or box.bottom > height + tolerance
    )
    if not outside:
        return None
    return LabelIssue(
        code="label_out_of_canvas",
        detail=(
            f"label {_short(box.text)!r} box "
            f"({box.left:.2f},{box.top:.2f})-({box.right:.2f},{box.bottom:.2f}) "
            f"exceeds canvas {width:g}x{height:g}"
        ),
        severity="error",
    )


def _overlap_issues(
    boxes: Sequence[LabelBox], threshold: float, *, limit: int
) -> tuple[list[LabelIssue], int]:
    """Sweep overlapping pairs, one issue per unordered pair (``i < j``).

    Sorted by left edge, so every later box that starts at or beyond the
    current box's right edge cannot overlap it and ends the inner loop. Only
    the first ``limit`` details are built; the returned count covers every
    overlapping pair so the truncation notice can report the real total.
    """
    issues: list[LabelIssue] = []
    ordered = sorted(enumerate(boxes), key=lambda item: (item[1].left, item[0]))
    total = 0
    for position, (_, first) in enumerate(ordered):
        for _, second in ordered[position + 1 :]:
            if second.left >= first.right:
                break
            ratio = first.overlaps(second)
            if ratio <= threshold + _OVERLAP_EPSILON:
                continue
            total += 1
            if len(issues) >= limit:
                continue
            issues.append(
                LabelIssue(
                    code="label_overlap",
                    detail=(
                        f"labels {_short(first.text)!r} and "
                        f"{_short(second.text)!r} overlap "
                        f"(ratio {ratio:.2f} > {threshold:g})"
                    ),
                    severity="warning",
                )
            )
    return issues, total - len(issues)


def _build_report(
    issues: Iterable[LabelIssue],
    *,
    boxes: int,
    max_issues: int,
    skipped: int = 0,
) -> LabelReport:
    all_issues = list(issues)
    ok = not any(issue.severity == "error" for issue in all_issues)
    kept = all_issues[:max_issues]
    omitted = all_issues[max_issues:]
    if omitted or skipped:
        # ``ok`` is computed from the full list; if a blocking issue was cut,
        # the truncation notice must block too so the visible issues never
        # contradict ``ok=False``. ``skipped`` counts overlap warnings whose
        # detail was never built, so they cannot contain errors.
        omitted_errors = sum(
            1 for issue in omitted if issue.severity == "error"
        )
        detail = f"{len(omitted) + skipped} more issues omitted"
        if omitted_errors:
            detail += f" ({omitted_errors} errors)"
        kept.append(
            LabelIssue(
                code="issues_truncated",
                detail=detail,
                severity="error" if omitted_errors else "warning",
            )
        )
    return LabelReport(ok=ok, issues=tuple(kept), boxes=boxes)


def _valid_threshold(value: object) -> bool:
    return _is_finite(value) and 0.0 <= float(value) <= 1.0


def _invalid_argument(detail: str, *, boxes: int) -> LabelReport:
    return _build_report(
        [LabelIssue("invalid_argument", detail, "error")],
        boxes=boxes,
        max_issues=MAX_ISSUES,
    )


def check_labels(
    boxes: Sequence[LabelBox],
    *,
    width: float,
    height: float,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    margin_tolerance: float = DEFAULT_MARGIN_TOLERANCE,
    max_issues: int = MAX_ISSUES,
) -> LabelReport:
    """Check *boxes* against a ``width`` x ``height`` canvas.

    Args:
        boxes: Label boxes in the canvas coordinate system.
        width: Canvas width in SVG user units.
        height: Canvas height in SVG user units.
        overlap_threshold: Ratio in ``0..1``; a pair above it is reported.
        margin_tolerance: Slack in pixels outside the canvas that is accepted.
        max_issues: Hard cap on returned issues (a truncation warning is added
            when the cap is hit).

    Returns:
        A :class:`LabelReport`. Invalid *width* / *height* / threshold /
        *max_issues* values produce a single ``invalid_argument`` error rather
        than raising, matching the fail-visible contract of the PDF layout
        gate.
    """
    items = tuple(boxes)
    if not _is_finite(width) or float(width) <= 0:
        return _invalid_argument(
            "width must be a positive finite number", boxes=len(items)
        )
    if not _is_finite(height) or float(height) <= 0:
        return _invalid_argument(
            "height must be a positive finite number", boxes=len(items)
        )
    if not _valid_threshold(overlap_threshold):
        return _invalid_argument(
            "overlap_threshold must be a finite number in 0..1",
            boxes=len(items),
        )
    if not _is_finite(margin_tolerance) or float(margin_tolerance) < 0:
        return _invalid_argument(
            "margin_tolerance must be a finite non-negative number",
            boxes=len(items),
        )
    if (
        not isinstance(max_issues, int)
        or isinstance(max_issues, bool)
        or max_issues < 1
    ):
        max_issues = MAX_ISSUES

    issues: list[LabelIssue] = []
    testable: list[LabelBox] = []
    for box in items:
        reason = _geometry_error(box)
        if reason:
            issues.append(
                LabelIssue(
                    code="label_geometry_invalid",
                    detail=f"label {_short(box.text)!r} has {reason}",
                    severity="error",
                )
            )
            continue
        testable.append(box)
        issue = _bounds_issue(
            box,
            width=float(width),
            height=float(height),
            tolerance=float(margin_tolerance),
        )
        if issue is not None:
            issues.append(issue)

    overlaps, skipped = _overlap_issues(
        testable, float(overlap_threshold), limit=max_issues
    )
    issues.extend(overlaps)
    return _build_report(
        issues, boxes=len(items), max_issues=max_issues, skipped=skipped
    )
