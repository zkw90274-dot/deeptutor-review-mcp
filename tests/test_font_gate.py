"""RS-505 font-size gate: ``pdftohtml -xml`` fontspec parsing.

Every test fakes both the toolchain probe and ``subprocess.run``; the suite
never needs a real ``pdftohtml`` or a real PDF binary.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from deeptutor.review.font_gate import (
    MAX_OUTPUT_BYTES,
    MAX_SIZE_ISSUES,
    FontSizeIssue,
    FontSizeReport,
    check_font_sizes,
)
from deeptutor.review.render_tools import PDFTOHTML, ToolCapability, Toolchain


@pytest.fixture(autouse=True)
def _reset_toolchain_cache() -> Any:
    from deeptutor.review.render_tools import clear_toolchain_cache

    clear_toolchain_cache()
    yield
    clear_toolchain_cache()


@pytest.fixture()
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "figure.pdf"
    path.write_bytes(b"%PDF-1.7 fake")
    return path


def _available_toolchain(executable: str = "/usr/bin/pdftohtml") -> Toolchain:
    return Toolchain(
        capabilities=(
            ToolCapability(
                name=PDFTOHTML,
                executable=executable,
                available=True,
                version="pdftohtml version 26.02.0",
                features=("-xml",),
            ),
        )
    )


def _missing_toolchain(reason: str = "not found on PATH") -> Toolchain:
    return Toolchain(
        capabilities=(
            ToolCapability(
                name=PDFTOHTML,
                executable=None,
                available=False,
                reason=reason,
            ),
        )
    )


def _patch_toolchain(
    monkeypatch: pytest.MonkeyPatch, toolchain: Toolchain
) -> None:
    monkeypatch.setattr(
        "deeptutor.review.font_gate.probe_toolchain", lambda: toolchain
    )


def _patch_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
    raises: BaseException | None = None,
    output_bytes: int | None = None,
) -> list[dict[str, Any]]:
    """Replace ``subprocess.run`` the way the gate now calls it.

    The real gate redirects stdout to a temp file and reads it back through a
    size cap, so the fake must write to whatever file object it is handed.
    ``output_bytes`` simulates a tool that produced more than the cap allows.
    """
    calls: list[dict[str, Any]] = []

    def fake_run(args: list[str], **kwargs: Any) -> Any:
        calls.append({"args": list(args), **kwargs})
        if raises is not None:
            raise raises
        sink = kwargs.get("stdout")
        if sink is not None and hasattr(sink, "write"):
            payload = stdout if output_bytes is None else b"x" * output_bytes
            sink.write(payload)
        return subprocess.CompletedProcess(
            args=args, returncode=returncode, stdout=None, stderr=stderr
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _xml(*pages: str) -> bytes:
    body = "".join(pages)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE pdf2xml SYSTEM "pdf2xml.dtd">\n'
        f'<pdf2xml producer="poppler" version="26.02.0">{body}</pdf2xml>'
    ).encode("utf-8")


def _page(number: int, *fontspecs: tuple[int, float]) -> str:
    specs = "".join(
        f'<fontspec id="{index}" size="{size:g}" '
        f'family="LMRoman{index}" color="#000000"/>'
        for index, (_ident, size) in enumerate(fontspecs)
    )
    texts = "".join(
        f'<text top="1" left="1" width="10" height="10" font="{index}">x</text>'
        for index, _spec in enumerate(fontspecs)
    )
    return (
        f'<page number="{number}" position="absolute" top="0" left="0" '
        f'height="100" width="100">{specs}{texts}</page>'
    )


def _set_up(
    monkeypatch: pytest.MonkeyPatch, pdf: Path, *, stdout: bytes
) -> list[dict[str, Any]]:
    _patch_toolchain(monkeypatch, _available_toolchain())
    return _patch_run(monkeypatch, stdout=stdout)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_readable_fonts_pass(monkeypatch: pytest.MonkeyPatch, pdf: Path) -> None:
    _set_up(monkeypatch, pdf, stdout=_xml(_page(1, (0, 11), (1, 14))))

    report = check_font_sizes(pdf)

    assert report.ok is True
    assert report.issues == ()
    assert report.pages == 1
    assert report.fonts_seen == 2
    assert report.smallest_size == 11
    assert report.min_font_size == 6.0
    assert report.as_dict()["ok"] is True


def test_four_point_font_is_reported(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _set_up(monkeypatch, pdf, stdout=_xml(_page(1, (0, 4), (1, 14))))

    report = check_font_sizes(pdf)

    assert report.ok is False
    assert len(report.issues) == 1
    issue = report.issues[0]
    assert issue.code == "font_size_below_minimum"
    assert issue.severity == "error"
    assert issue.page == 1
    assert issue.size == 4
    assert "4pt" in issue.detail
    assert report.smallest_size == 4


def test_same_page_same_size_reported_once(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    """Two font specs that round to the same size are one finding, not two."""
    _set_up(
        monkeypatch,
        pdf,
        stdout=_xml(_page(1, (0, 5.4), (1, 5.4), (2, 12))),
    )

    report = check_font_sizes(pdf)

    assert [issue.code for issue in report.issues] == ["font_size_below_minimum"]
    assert report.fonts_seen == 3
    assert report.smallest_size == 5.4


def test_same_size_on_different_pages_reported_per_page(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _set_up(
        monkeypatch,
        pdf,
        stdout=_xml(_page(1, (0, 4)), _page(2, (0, 4))),
    )

    report = check_font_sizes(pdf)

    assert report.pages == 2
    assert [issue.page for issue in report.issues] == [1, 2]
    assert all(issue.size == 4 for issue in report.issues)


def test_pdf_without_text_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _set_up(monkeypatch, pdf, stdout=_xml(_page(1)))

    report = check_font_sizes(pdf)

    assert report.ok is True
    assert report.issues == ()
    assert report.pages == 1
    assert report.fonts_seen == 0
    assert report.smallest_size is None


def test_threshold_is_configurable(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _set_up(monkeypatch, pdf, stdout=_xml(_page(1, (0, 8))))

    assert check_font_sizes(pdf, min_font_size=6).ok is True
    report = check_font_sizes(pdf, min_font_size=10)
    assert report.ok is False
    assert report.issues[0].size == 8
    assert report.as_dict()["min_font_size"] == 10.0


def test_malformed_size_attribute_is_skipped(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    """One broken fontspec must not invalidate the rest of the document."""
    broken = _page(1, (0, 9)).replace('size="9"', 'size="huge"')
    good = _page(2, (0, 4))
    _set_up(monkeypatch, pdf, stdout=_xml(broken, good))

    report = check_font_sizes(pdf)

    assert report.fonts_seen == 1
    assert report.smallest_size == 4
    assert report.ok is False


# ---------------------------------------------------------------------------
# Command line: the two flags that are silently load-bearing
# ---------------------------------------------------------------------------


def test_command_forces_zoom_and_ignores_images(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    calls = _set_up(monkeypatch, pdf, stdout=_xml(_page(1, (0, 11))))

    check_font_sizes(pdf)

    assert len(calls) == 1
    args = calls[0]["args"]
    assert args[0] == "/usr/bin/pdftohtml"
    assert "-xml" in args
    assert "-i" in args
    # Default zoom is 1.5 and would scale every reported size by 1.5.
    assert args[args.index("-zoom") + 1] == "1.0"
    assert "-stdout" in args
    assert args[-1] == str(pdf)
    assert calls[0]["shell"] is False
    assert calls[0]["timeout"] == 10.0
    assert calls[0]["check"] is False


# ---------------------------------------------------------------------------
# Failures: always an issue, never an exception
# ---------------------------------------------------------------------------


def test_unavailable_tool_does_not_start_a_subprocess(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _missing_toolchain())
    calls = _patch_run(
        monkeypatch, raises=AssertionError("must not run when unavailable")
    )

    report = check_font_sizes(pdf)

    assert calls == []
    assert report.ok is False
    assert report.issues[0].code == "pdftohtml_unavailable"
    assert "not found on PATH" in report.issues[0].detail


def test_unprobed_tool_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, Toolchain(capabilities=()))
    calls = _patch_run(
        monkeypatch, raises=AssertionError("must not run when unprobed")
    )

    report = check_font_sizes(pdf)

    assert calls == []
    assert report.issues[0].code == "pdftohtml_unavailable"
    assert "not probed" in report.issues[0].detail


def test_timeout_is_reported(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(monkeypatch, raises=subprocess.TimeoutExpired(["pdftohtml"], 10.0))

    report = check_font_sizes(pdf)

    assert report.ok is False
    assert report.issues[0].code == "pdftohtml_timeout"


def test_non_zero_exit_is_reported_and_truncated(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(monkeypatch, returncode=2, stderr=b"x" * 5000)

    report = check_font_sizes(pdf)

    assert report.ok is False
    issue = report.issues[0]
    assert issue.code == "pdftohtml_failed"
    assert "status 2" in issue.detail
    assert len(issue.detail) <= 200


def test_os_error_is_reported(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(monkeypatch, raises=OSError("permission denied"))

    report = check_font_sizes(pdf)

    assert report.issues[0].code == "pdftohtml_failed"
    assert "permission denied" in report.issues[0].detail


def test_unparseable_xml_is_reported(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _set_up(monkeypatch, pdf, stdout=b"<pdf2xml><page>")

    report = check_font_sizes(pdf)

    assert report.ok is False
    assert report.issues[0].code == "xml_parse_error"


def test_unexpected_root_element_is_reported(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _set_up(monkeypatch, pdf, stdout=b"<html><body/></html>")

    report = check_font_sizes(pdf)

    assert report.issues[0].code == "xml_parse_error"
    assert "html" in report.issues[0].detail


def test_empty_output_is_reported(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _set_up(monkeypatch, pdf, stdout=b"   \n")

    report = check_font_sizes(pdf)

    assert report.issues[0].code == "xml_parse_error"


def test_missing_pdf_is_reported_without_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    calls = _patch_run(
        monkeypatch, raises=AssertionError("must not run for a missing PDF")
    )

    report = check_font_sizes(tmp_path / "absent.pdf")

    assert calls == []
    assert report.issues[0].code == "pdf_missing"


def test_oversized_output_fails_instead_of_buffering(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(monkeypatch, stdout=b"<" + b"x" * (MAX_OUTPUT_BYTES + 1))

    report = check_font_sizes(pdf)

    assert report.ok is False
    assert report.issues[0].code == "pdftohtml_failed"
    assert str(MAX_OUTPUT_BYTES) in report.issues[0].detail


# ---------------------------------------------------------------------------
# Arguments and payload shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, -1.0, float("inf"), float("nan")])
def test_invalid_min_font_size_raises(
    monkeypatch: pytest.MonkeyPatch, pdf: Path, value: float
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(monkeypatch, raises=AssertionError("must not run"))

    with pytest.raises(ValueError):
        check_font_sizes(pdf, min_font_size=value)


def test_invalid_timeout_is_an_issue_not_an_exception(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(monkeypatch, raises=AssertionError("must not run"))

    report = check_font_sizes(pdf, timeout_seconds=0)

    assert report.issues[0].code == "invalid_argument"


def test_issue_cap_adds_a_truncation_notice(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    pages = "".join(_page(number, (0, 4)) for number in range(1, MAX_SIZE_ISSUES + 6))
    _set_up(monkeypatch, pdf, stdout=_xml(pages))

    report = check_font_sizes(pdf)

    assert report.pages == MAX_SIZE_ISSUES + 5
    assert len(report.issues) == MAX_SIZE_ISSUES + 1
    notice = report.issues[-1]
    assert notice.code == "issues_truncated"
    assert notice.severity == "error"  # omitted tail contained errors
    assert "5 more issues omitted" in notice.detail
    assert report.ok is False


def test_report_is_json_serializable(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _set_up(monkeypatch, pdf, stdout=_xml(_page(1, (0, 4), (1, 12))))

    payload = check_font_sizes(pdf).as_dict()
    encoded = json.dumps(payload)

    assert json.loads(encoded) == payload
    assert payload["issues"][0]["size"] == 4
    assert payload["issues"][0]["page"] == 1


def test_report_and_issue_are_immutable(pdf: Path) -> None:
    issue = FontSizeIssue(code="x", detail="y")
    report = FontSizeReport(ok=False, issues=(issue,))

    with pytest.raises(Exception):
        issue.code = "z"  # type: ignore[misc]
    with pytest.raises(Exception):
        report.ok = True  # type: ignore[misc]


def test_oversized_pdf_is_rejected_before_the_tool_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The input bound is the real memory bound; stdout is already buffered."""
    from deeptutor.review import font_gate

    huge = tmp_path / "huge.pdf"
    huge.write_bytes(b"%PDF-1.5\n")
    monkeypatch.setattr(font_gate, "MAX_PDF_BYTES", 4)
    _patch_toolchain(monkeypatch, _available_toolchain())

    def explode(*_args, **_kwargs):
        raise AssertionError("the tool must not run for an oversized PDF")

    monkeypatch.setattr(font_gate.subprocess, "run", explode)

    report = font_gate.check_font_sizes(huge)

    assert report.ok is False
    assert [issue.code for issue in report.issues] == ["pdf_too_large"]
