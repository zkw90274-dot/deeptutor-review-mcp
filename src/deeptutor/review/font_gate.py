"""Deterministic minimum-font-size gate for rendered figures (RS-505).

``pdftotext -bbox`` (used by :mod:`deeptutor.review.layout_gate`) reports word
boxes but no font sizes. poppler's ``pdftohtml -xml`` does: every page carries
a ``<fontspec id="..." size="..." family="..." color="..."/>`` table and every
``<text>`` element references one of those ids.

Two details are load-bearing and were measured, not inferred:

* ``-zoom 1.0`` **must** be passed explicitly. The default zoom is ``1.5`` and
  the reported ``size`` is multiplied by it (a 4 pt label is reported as 6).
* ``size`` is a **rounded integer point value**. Measurements: 5.5 -> 5,
  5.9 -> 6, 6.4 -> 6, 6.6 -> 7. The gate therefore has ±0.5 pt resolution and
  a hit means "at least half a point below the threshold", not "0.01 pt below".

The gate is fail-visible but never raises for tool problems: a missing tool, a
timeout, a non-zero exit or unparseable XML becomes an ``error`` issue on the
report. A malformed ``size`` attribute is skipped so one broken font entry
cannot invalidate the whole document. Only the PDF path is read; the tool runs
through an argument list with ``shell=False`` and a hard timeout.

Memory is bounded on **both** sides. The input PDF is rejected above
:data:`MAX_PDF_BYTES` before the tool starts — a cheap check, but not the real
bound, because a compressed PDF can expand far beyond its byte size. The real
bound is on the output: the XML is streamed to a temporary file, and its size
is checked before anything is read into memory, so peak RAM is
:data:`MAX_OUTPUT_BYTES` regardless of how much the tool wrote to disk.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from deeptutor.review.render_tools import PDFTOHTML, probe_toolchain

#: Upper bound on reported size issues; a pathological PDF must not produce an
#: unbounded JSON payload.
MAX_SIZE_ISSUES = 100

#: Hard ceiling for the XML pdftohtml writes, enforced on the temp file's
#: size *before* it is read into memory. A figure PDF is a few KB; 4 MB admits
#: an unusually text-dense page while keeping peak memory bounded even when
#: the input PDF was compressed text (compression beats the input-side check).
MAX_OUTPUT_BYTES = 4 * 1024 * 1024

#: Input ceiling. A figure PDF is a few KB; 16 MB admits an unusually detailed
#: one. This is a cheap pre-check, not the memory bound — compressed content
#: streams can expand far beyond their byte size, which is why the output side
#: is capped separately.
MAX_PDF_BYTES = 16 * 1024 * 1024

DEFAULT_MIN_FONT_SIZE = 6.0
DEFAULT_TIMEOUT_SECONDS = 10.0

_DETAIL_LIMIT = 200

#: U+FEFF written as an escape so this file never contains an invisible BOM
#: character of its own.
_BOM = chr(0xFEFF)


@dataclass(frozen=True)
class FontSizeIssue:
    """One finding from the font-size gate.

    ``page`` is 1-based; ``0`` means the issue is not tied to a page (missing
    tool, timeout, unparseable output). ``size`` is set only for
    ``font_size_below_minimum``.
    """

    code: str
    detail: str
    page: int = 0
    size: float | None = None
    severity: Literal["error", "warning"] = "error"

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "page": self.page,
            "size": self.size,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class FontSizeReport:
    """Immutable result of :func:`check_font_sizes`.

    ``ok`` is ``False`` only when at least one issue has ``severity ==
    "error"``. ``smallest_size`` is ``None`` when the PDF contains no text
    (a pure drawing has no ``<fontspec>`` elements); that is a normal result,
    not a failure.
    """

    ok: bool
    issues: tuple[FontSizeIssue, ...] = ()
    pages: int = 0
    fonts_seen: int = 0
    smallest_size: float | None = None
    min_font_size: float = DEFAULT_MIN_FONT_SIZE

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "pages": self.pages,
            "fonts_seen": self.fonts_seen,
            "smallest_size": self.smallest_size,
            "min_font_size": self.min_font_size,
            "issues": [issue.as_dict() for issue in self.issues],
        }


def _local_name(tag: object) -> str:
    """Return the local part of a possibly namespaced element tag."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _decode(raw: bytes | str | None) -> str:
    """Decode tool output; a leading UTF-8 BOM on either input kind is dropped."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8-sig", errors="replace")
    if raw.startswith(_BOM):
        return raw[1:]
    return raw


def _short(text: str) -> str:
    """Collapse whitespace and bound the result to :data:`_DETAIL_LIMIT` chars."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _DETAIL_LIMIT:
        return collapsed
    return collapsed[: _DETAIL_LIMIT - 3] + "..."


def _size(raw: str | None) -> float | None:
    """Parse a ``size`` attribute; anything non-numeric is skipped."""
    if raw is None:
        return None
    try:
        value = float(raw.strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _page_number(element: ET.Element, fallback: int) -> int:
    raw = element.get("number")
    if raw is None:
        return fallback
    try:
        number = int(raw.strip())
    except (TypeError, ValueError):
        return fallback
    return number if number > 0 else fallback


def _truncate(issues: Sequence[FontSizeIssue]) -> list[FontSizeIssue]:
    """Cap the issue list, keeping ``ok`` consistent with the visible issues."""
    kept = list(issues)
    if len(kept) <= MAX_SIZE_ISSUES:
        return kept
    omitted = kept[MAX_SIZE_ISSUES:]
    kept = kept[:MAX_SIZE_ISSUES]
    omitted_errors = sum(1 for issue in omitted if issue.severity == "error")
    detail = f"{len(omitted)} more issues omitted"
    if omitted_errors:
        detail += f" ({omitted_errors} errors)"
    kept.append(
        FontSizeIssue(
            code="issues_truncated",
            detail=detail,
            severity="error" if omitted_errors else "warning",
        )
    )
    return kept


def _build_report(
    issues: Sequence[FontSizeIssue],
    *,
    pages: int = 0,
    fonts_seen: int = 0,
    smallest_size: float | None = None,
    min_font_size: float = DEFAULT_MIN_FONT_SIZE,
) -> FontSizeReport:
    # ``ok`` is computed from the full list, before truncation, so a blocking
    # issue hidden past MAX_SIZE_ISSUES still blocks.
    ok = not any(issue.severity == "error" for issue in issues)
    kept = _truncate(issues)
    return FontSizeReport(
        ok=ok,
        issues=tuple(kept),
        pages=pages,
        fonts_seen=fonts_seen,
        smallest_size=smallest_size,
        min_font_size=min_font_size,
    )


def _failure(
    code: str,
    detail: str,
    *,
    min_font_size: float,
    pages: int = 0,
    fonts_seen: int = 0,
    smallest_size: float | None = None,
) -> FontSizeReport:
    return _build_report(
        [FontSizeIssue(code=code, detail=detail)],
        pages=pages,
        fonts_seen=fonts_seen,
        smallest_size=smallest_size,
        min_font_size=min_font_size,
    )


def _parse_font_sizes(
    xml_text: str,
    *,
    min_font_size: float,
) -> FontSizeReport:
    """Turn ``pdftohtml -xml`` output into a report.

    Duplicate ``(page, size)`` pairs are reported once: a paragraph in 5 pt
    should produce one finding, not one per line. Pages and font specs are
    visited in document order so the output is deterministic.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return _failure(
            "xml_parse_error",
            f"pdftohtml output could not be parsed: {exc}",
            min_font_size=min_font_size,
        )

    if _local_name(root.tag) != "pdf2xml":
        return _failure(
            "xml_parse_error",
            f"unexpected root element <{_local_name(root.tag) or '?'}>",
            min_font_size=min_font_size,
        )

    issues: list[FontSizeIssue] = []
    reported: set[tuple[int, float]] = set()
    pages = 0
    fonts_seen = 0
    smallest: float | None = None

    page_index = 0
    for page in root.iter():
        if _local_name(page.tag) != "page":
            continue
        page_index += 1
        pages += 1
        number = _page_number(page, page_index)
        for element in page:
            if _local_name(element.tag) != "fontspec":
                continue
            size = _size(element.get("size"))
            if size is None:
                # One malformed entry must not invalidate the whole document.
                continue
            fonts_seen += 1
            if smallest is None or size < smallest:
                smallest = size
            if size >= min_font_size:
                continue
            if (number, size) in reported:
                continue
            reported.add((number, size))
            issues.append(
                FontSizeIssue(
                    code="font_size_below_minimum",
                    detail=(
                        f"font size {size:g}pt is below the {min_font_size:g}pt "
                        f"minimum (reported size is rounded to whole points)"
                    ),
                    page=number,
                    size=size,
                )
            )

    return _build_report(
        issues,
        pages=pages,
        fonts_seen=fonts_seen,
        smallest_size=smallest,
        min_font_size=min_font_size,
    )


def _valid_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def check_font_sizes(
    pdf_path: str | Path,
    *,
    min_font_size: float = DEFAULT_MIN_FONT_SIZE,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> FontSizeReport:
    """Check a rendered figure PDF for fonts smaller than ``min_font_size``.

    Args:
        pdf_path: Path to the PDF produced from the figure source.
        min_font_size: Threshold in points. Must be a positive finite number.
        timeout_seconds: Hard limit for the ``pdftohtml`` call.

    Returns:
        A :class:`FontSizeReport`. Tool failures are reported as ``error``
        issues; this function does not raise for them.

    Raises:
        ValueError: ``min_font_size`` is not a positive finite number. That is
            a caller bug, not a property of the figure.
    """
    if not _valid_number(min_font_size):
        raise ValueError(
            f"min_font_size must be a positive finite number, got {min_font_size!r}"
        )
    threshold = float(min_font_size)

    if not _valid_number(timeout_seconds):
        return _failure(
            "invalid_argument",
            f"timeout_seconds must be a positive finite number, got "
            f"{timeout_seconds!r}",
            min_font_size=threshold,
        )

    path = Path(pdf_path)
    if not path.is_file():
        return _failure(
            "pdf_missing",
            f"PDF not found: {path}",
            min_font_size=threshold,
        )

    # Reject an oversized input *before* the tool runs: this is the bound that
    # actually limits how much stdout can be buffered later.
    try:
        pdf_size = path.stat().st_size
    except OSError as exc:
        return _failure(
            "pdf_missing",
            _short(f"PDF could not be read: {exc}"),
            min_font_size=threshold,
        )
    if pdf_size > MAX_PDF_BYTES:
        return _failure(
            "pdf_too_large",
            f"PDF is {pdf_size} bytes, above the {MAX_PDF_BYTES}-byte limit",
            min_font_size=threshold,
        )

    # Deployment choice: no probe means no gate, and definitely no subprocess.
    capability = probe_toolchain().capability(PDFTOHTML)
    if capability is None or not capability.available:
        reason = capability.reason if capability is not None else "not probed"
        return _failure(
            "pdftohtml_unavailable",
            f"pdftohtml unavailable: {reason}",
            min_font_size=threshold,
        )

    executable = capability.executable or PDFTOHTML
    command = [
        executable,
        "-xml",
        "-i",
        "-zoom",
        "1.0",
        "-stdout",
        str(path),
    ]
    # The XML is streamed to a temp file instead of a pipe. A compressed PDF
    # decompresses to far more text than its own byte size (a 100 KB file can
    # yield tens of MB of XML), so a pipe plus a post-read length check would
    # still have buffered everything. Here peak memory is the size we read
    # back, which is capped below; the disk copy is temporary and unbounded
    # only in the pathological case the timeout already bounds in time.
    with tempfile.TemporaryFile() as sink:
        try:
            completed = subprocess.run(
                command,
                stdout=sink,
                stderr=subprocess.PIPE,
                timeout=float(timeout_seconds),
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return _failure(
                "pdftohtml_timeout",
                f"pdftohtml exceeded {timeout_seconds:g}s",
                min_font_size=threshold,
            )
        except OSError as exc:
            return _failure(
                "pdftohtml_failed",
                _short(f"pdftohtml could not run: {exc}"),
                min_font_size=threshold,
            )

        if completed.returncode != 0:
            stderr = _short(_decode(completed.stderr)) or "no stderr"
            return _failure(
                "pdftohtml_failed",
                _short(
                    f"pdftohtml exited with status {completed.returncode}: {stderr}"
                ),
                min_font_size=threshold,
            )

        sink.seek(0, os.SEEK_END)
        output_size = sink.tell()
        if output_size > MAX_OUTPUT_BYTES:
            return _failure(
                "pdftohtml_failed",
                f"pdftohtml output exceeded {MAX_OUTPUT_BYTES} bytes",
                min_font_size=threshold,
            )
        sink.seek(0)
        output_bytes = sink.read(MAX_OUTPUT_BYTES)

    if not output_bytes:
        return _failure(
            "xml_parse_error",
            "pdftohtml produced no XML output",
            min_font_size=threshold,
        )

    xml_text = _decode(output_bytes)
    if not xml_text.strip():
        return _failure(
            "xml_parse_error",
            "pdftohtml produced no XML output",
            min_font_size=threshold,
        )

    return _parse_font_sizes(xml_text, min_font_size=threshold)


__all__ = (
    "DEFAULT_MIN_FONT_SIZE",
    "DEFAULT_TIMEOUT_SECONDS",
    "FontSizeIssue",
    "FontSizeReport",
    "MAX_OUTPUT_BYTES",
    "MAX_SIZE_ISSUES",
    "check_font_sizes",
)
