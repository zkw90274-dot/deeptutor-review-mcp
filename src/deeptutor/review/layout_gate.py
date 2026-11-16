"""Deterministic layout gate for rendered figures (RS-505).

``pdftotext -bbox`` reports one box per word in PDF points, which is enough to
prove three classes of layout defects without looking at pixels:

* a word box outside the page (tolerance :data:`BOUNDS_TOLERANCE`);
* two word boxes whose intersection covers more than a ratio of the smaller
  box (default ``0.15``);
* a word box with non-positive width or height.

The gate is fail-visible: a missing tool, a timeout, a non-zero exit or
unparseable XHTML produces an ``error`` issue on the report instead of raising
into the render request. The report object is immutable so it can be cached
alongside the rendered asset.

``pdftotext -bbox`` does **not** expose font sizes. When the caller asks for a
``min_font_size`` the gate emits a ``font_size_unsupported`` warning rather
than inventing a measurement; the caller must use a different probe (for
example ``pdffonts`` or a box-height heuristic) if that check is required.

Only the PDF path is read. The tool runs through an argument list with
``shell=False`` and a hard timeout.
"""

from __future__ import annotations

import math
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from deeptutor.review.render_tools import PDFTOTEXT, probe_toolchain

#: PDF points of slack allowed before a box counts as outside the page.
BOUNDS_TOLERANCE = 1.0

#: Upper bound on reported issues; a pathological page must not produce an
#: unbounded JSON payload.
MAX_ISSUES = 200

_DETAIL_TEXT_LIMIT = 40
_DEFAULT_THRESHOLD = 0.15
_DEFAULT_FONT_SIZE = 6.0
_DEFAULT_TIMEOUT_SECONDS = 10.0

#: U+FEFF written as an escape so this file never contains an invisible BOM
#: character of its own.
_BOM = chr(0xFEFF)


@dataclass(frozen=True)
class LayoutIssue:
    """One finding from the layout gate.

    ``page`` is 1-based; ``0`` means the issue is not tied to a page (missing
    tool, bad argument, unparseable output).
    """

    code: str
    detail: str
    page: int = 0
    severity: Literal["error", "warning"] = "error"

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "page": self.page,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class LayoutReport:
    """Immutable result of :func:`check_layout`.

    ``ok`` is ``False`` only when at least one issue has ``severity ==
    "error"``. Warnings stay visible in ``issues`` but do not block.
    """

    ok: bool
    issues: tuple[LayoutIssue, ...] = ()
    pages: int = 0
    words: int = 0
    threshold: float = _DEFAULT_THRESHOLD

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "pages": self.pages,
            "words": self.words,
            "threshold": self.threshold,
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class _Word:
    text: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class _Page:
    number: int
    width: float | None
    height: float | None
    words: tuple[_Word, ...]


def _local_name(tag: object) -> str:
    """Return the local part of a possibly namespaced element tag."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _attribute(element: ET.Element, name: str) -> str | None:
    value = element.get(name)
    if value is not None:
        return value
    for key, candidate in element.attrib.items():
        if _local_name(key).lower() == name.lower():
            return candidate
    return None


def _number(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw.strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _short(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _DETAIL_TEXT_LIMIT:
        return collapsed
    return collapsed[:_DETAIL_TEXT_LIMIT] + "..."


def _decode(raw: bytes | str | None) -> str:
    """Decode tool output; a leading UTF-8 BOM on either input kind is dropped."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8-sig", errors="replace")
    if raw.startswith(_BOM):
        return raw[1:]
    return raw


def _parse_bbox(
    xhtml: str,
) -> tuple[list[_Page], list[LayoutIssue]]:
    """Parse ``pdftotext -bbox`` output.

    Missing or non-numeric geometry is reported as an ``error`` and the
    affected word is skipped: an unreadable box cannot be proven to be
    correct, and RS-505 blocks on unproven layout rather than passing it.
    """
    issues: list[LayoutIssue] = []
    try:
        root = ET.fromstring(xhtml)
    except ET.ParseError as exc:
        return [], [
            LayoutIssue(
                code="bbox_parse_error",
                detail=f"pdftotext XHTML could not be parsed: {exc}",
            )
        ]

    pages: list[_Page] = []
    for index, page_element in enumerate(
        element for element in root.iter() if _local_name(element.tag) == "page"
    ):
        number = index + 1
        width = _number(_attribute(page_element, "width"))
        height = _number(_attribute(page_element, "height"))
        if width is None or height is None or width <= 0 or height <= 0:
            issues.append(
                LayoutIssue(
                    code="page_geometry_invalid",
                    detail=f"page {number} has no usable width/height",
                    page=number,
                )
            )
            width = height = None

        words: list[_Word] = []
        for word_element in page_element.iter():
            if _local_name(word_element.tag) != "word":
                continue
            text = (word_element.text or "").strip()
            if not text:
                continue
            coordinates = [
                _number(_attribute(word_element, name))
                for name in ("xMin", "yMin", "xMax", "yMax")
            ]
            if any(value is None for value in coordinates):
                issues.append(
                    LayoutIssue(
                        code="word_geometry_invalid",
                        detail=f"word {_short(text)!r} has incomplete bbox",
                        page=number,
                    )
                )
                continue
            x_min, y_min, x_max, y_max = (float(value) for value in coordinates)
            words.append(_Word(text, x_min, y_min, x_max, y_max))

        pages.append(_Page(number, width, height, tuple(words)))

    return pages, issues


def _overlap_issues(page: _Page, threshold: float) -> list[LayoutIssue]:
    issues: list[LayoutIssue] = []
    candidates = [
        word
        for word in page.words
        if word.width > 0 and word.height > 0 and word.area > 0
    ]
    ordered = sorted(
        enumerate(candidates), key=lambda item: (item[1].x_min, item[0])
    )
    for position, (_, first) in enumerate(ordered):
        for _, second in ordered[position + 1 :]:
            # Sorted by xMin: no later box can overlap horizontally.
            if second.x_min >= first.x_max:
                break
            intersection_height = min(first.y_max, second.y_max) - max(
                first.y_min, second.y_min
            )
            if intersection_height <= 0:
                continue
            intersection_width = min(first.x_max, second.x_max) - max(
                first.x_min, second.x_min
            )
            if intersection_width <= 0:
                continue
            ratio = (
                intersection_width * intersection_height
            ) / min(first.area, second.area)
            if ratio > threshold:
                issues.append(
                    LayoutIssue(
                        code="word_overlap",
                        detail=(
                            f"words {_short(first.text)!r} and "
                            f"{_short(second.text)!r} overlap "
                            f"(ratio {ratio:.2f} > {threshold:g})"
                        ),
                        page=page.number,
                    )
                )
    return issues


def _check_page(page: _Page, threshold: float) -> list[LayoutIssue]:
    issues: list[LayoutIssue] = []
    for word in page.words:
        if word.width <= 0 or word.height <= 0:
            issues.append(
                LayoutIssue(
                    code="word_box_degenerate",
                    detail=(
                        f"word {_short(word.text)!r} has a non-positive box "
                        f"({word.width:.2f}x{word.height:.2f})"
                    ),
                    page=page.number,
                )
            )
            continue

        if page.width is None or page.height is None:
            continue
        outside = (
            word.x_min < -BOUNDS_TOLERANCE
            or word.y_min < -BOUNDS_TOLERANCE
            or word.x_max > page.width + BOUNDS_TOLERANCE
            or word.y_max > page.height + BOUNDS_TOLERANCE
        )
        if outside:
            issues.append(
                LayoutIssue(
                    code="word_out_of_bounds",
                    detail=(
                        f"word {_short(word.text)!r} box "
                        f"({word.x_min:.2f},{word.y_min:.2f})-"
                        f"({word.x_max:.2f},{word.y_max:.2f}) exceeds "
                        f"page {page.width:g}x{page.height:g}"
                    ),
                    page=page.number,
                )
            )

    issues.extend(_overlap_issues(page, threshold))
    return issues


def _build_report(
    issues: Sequence[LayoutIssue],
    *,
    pages: int = 0,
    words: int = 0,
    threshold: float = _DEFAULT_THRESHOLD,
) -> LayoutReport:
    ok = not any(issue.severity == "error" for issue in issues)
    kept = list(issues)
    if len(kept) > MAX_ISSUES:
        omitted = kept[MAX_ISSUES:]
        kept = kept[:MAX_ISSUES]
        # ``ok`` is computed from the full list. If the omitted tail contains
        # a blocking issue, the truncation notice must itself be blocking so
        # the visible issues never contradict ``ok=False``.
        omitted_errors = sum(
            1 for issue in omitted if issue.severity == "error"
        )
        detail = f"{len(omitted)} more issues omitted"
        if omitted_errors:
            detail += f" ({omitted_errors} errors)"
        kept.append(
            LayoutIssue(
                code="issues_truncated",
                detail=detail,
                severity="error" if omitted_errors else "warning",
            )
        )
    return LayoutReport(
        ok=ok,
        issues=tuple(kept),
        pages=pages,
        words=words,
        threshold=threshold,
    )


def _invalid_argument(detail: str, threshold: float) -> LayoutReport:
    return _build_report(
        [LayoutIssue(code="invalid_argument", detail=detail)],
        threshold=threshold,
    )


def _valid_threshold(value: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def check_layout(
    pdf_path: str | Path,
    *,
    overlap_threshold: float = _DEFAULT_THRESHOLD,
    min_font_size: float | None = _DEFAULT_FONT_SIZE,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> LayoutReport:
    """Check a rendered figure PDF for text collisions and clipped words.

    Args:
        pdf_path: Path to the PDF produced from the figure source.
        overlap_threshold: Intersection-over-smaller-area ratio above which two
            word boxes are reported. Must be within ``[0, 1]``.
        min_font_size: Requested minimum font size in points. ``pdftotext
            -bbox`` cannot measure it, so any non-``None`` value produces a
            ``font_size_unsupported`` warning. Pass ``None`` to opt out.
        timeout_seconds: Hard limit for the ``pdftotext`` call.

    Returns:
        A :class:`LayoutReport`. Tool failures and bad arguments are reported
        as ``error`` issues; this function does not raise for them.
    """
    if not _valid_threshold(overlap_threshold):
        return _invalid_argument(
            f"overlap_threshold must be a number in [0, 1], got "
            f"{overlap_threshold!r}",
            _DEFAULT_THRESHOLD,
        )
    threshold = float(overlap_threshold)
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        return _invalid_argument(
            f"timeout_seconds must be a positive finite number, got "
            f"{timeout_seconds!r}",
            threshold,
        )
    if min_font_size is not None and (
        not isinstance(min_font_size, (int, float))
        or isinstance(min_font_size, bool)
        or not math.isfinite(min_font_size)
        or min_font_size <= 0
    ):
        return _invalid_argument(
            f"min_font_size must be None or a positive finite number, got "
            f"{min_font_size!r}",
            threshold,
        )

    path = Path(pdf_path)
    if not path.is_file():
        return _build_report(
            [
                LayoutIssue(
                    code="pdf_missing",
                    detail=f"PDF not found: {path}",
                )
            ],
            threshold=threshold,
        )

    capability = probe_toolchain().capability(PDFTOTEXT)
    if capability is None or not capability.available:
        reason = capability.reason if capability is not None else "not probed"
        return _build_report(
            [
                LayoutIssue(
                    code="pdftotext_unavailable",
                    detail=f"pdftotext unavailable: {reason}",
                )
            ],
            threshold=threshold,
        )

    executable = capability.executable or PDFTOTEXT
    command = [executable, "-bbox", "-enc", "UTF-8", str(path), "-"]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=float(timeout_seconds),
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return _build_report(
            [
                LayoutIssue(
                    code="pdftotext_timeout",
                    detail=f"pdftotext exceeded {timeout_seconds:g}s",
                )
            ],
            threshold=threshold,
        )
    except OSError as exc:
        return _build_report(
            [
                LayoutIssue(
                    code="pdftotext_failed",
                    detail=f"pdftotext could not run: {exc}",
                )
            ],
            threshold=threshold,
        )

    if completed.returncode != 0:
        stderr = _short(_decode(completed.stderr)) or "no stderr"
        return _build_report(
            [
                LayoutIssue(
                    code="pdftotext_failed",
                    detail=(
                        f"pdftotext exited with status "
                        f"{completed.returncode}: {stderr}"
                    ),
                )
            ],
            threshold=threshold,
        )

    xhtml = _decode(completed.stdout)
    if not xhtml.strip():
        return _build_report(
            [
                LayoutIssue(
                    code="bbox_parse_error",
                    detail="pdftotext produced no XHTML output",
                )
            ],
            threshold=threshold,
        )

    pages, issues = _parse_bbox(xhtml)
    if not pages and not issues:
        issues.append(
            LayoutIssue(
                code="no_pages",
                detail="no <page> element in bbox output",
            )
        )

    word_count = sum(len(page.words) for page in pages)
    for page in pages:
        issues.extend(_check_page(page, overlap_threshold))

    if min_font_size is not None:
        issues.append(
            LayoutIssue(
                code="font_size_unsupported",
                detail=(
                    "pdftotext -bbox does not report font sizes; "
                    f"min_font_size={min_font_size:g} was not checked"
                ),
                severity="warning",
            )
        )

    return _build_report(
        issues,
        pages=len(pages),
        words=word_count,
        threshold=threshold,
    )


__all__ = (
    "BOUNDS_TOLERANCE",
    "MAX_ISSUES",
    "LayoutIssue",
    "LayoutReport",
    "check_layout",
)
