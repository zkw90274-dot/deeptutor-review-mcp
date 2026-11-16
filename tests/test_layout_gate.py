"""RS-505 layout gate: word-box bounds, collisions and degenerate boxes.

Every test fakes both the toolchain probe and ``subprocess.run``; the suite
never needs a real ``pdftotext`` or a real PDF binary.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Sequence

import pytest

from deeptutor.review.layout_gate import (
    BOUNDS_TOLERANCE,
    LayoutIssue,
    LayoutReport,
    check_layout,
)
from deeptutor.review.render_tools import PDFTOTEXT, ToolCapability, Toolchain


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


def _available_toolchain(executable: str = "/usr/bin/pdftotext") -> Toolchain:
    return Toolchain(
        capabilities=(
            ToolCapability(
                name=PDFTOTEXT,
                executable=executable,
                available=True,
                version="pdftotext version 24.02.0",
            ),
        )
    )


def _missing_toolchain(reason: str = "not found on PATH") -> Toolchain:
    return Toolchain(
        capabilities=(
            ToolCapability(
                name=PDFTOTEXT,
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
        "deeptutor.review.layout_gate.probe_toolchain", lambda: toolchain
    )


def _patch_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
    raises: BaseException | None = None,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_run(
        args: Sequence[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append({"args": list(args), **kwargs})
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(
            args=list(args),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _bbox(pages: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" '
        '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><doc>'
        f"{pages}"
        "</doc></body></html>"
    ).encode("utf-8")


def _page(body: str, width: float = 200, height: float = 100) -> str:
    return f'<page width="{width}" height="{height}">{body}</page>'


def _word(
    text: str,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> str:
    return (
        f'<word xMin="{x_min}" yMin="{y_min}" '
        f'xMax="{x_max}" yMax="{y_max}">{text}</word>'
    )


def _codes(report: LayoutReport) -> list[str]:
    return [issue.code for issue in report.issues]


def _severities(report: LayoutReport, code: str) -> list[str]:
    return [issue.severity for issue in report.issues if issue.code == code]


def test_clean_layout_passes_and_reports_counts(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(
        monkeypatch,
        stdout=_bbox(
            _page(
                _word("hello", 10, 10, 50, 20)
                + _word("world", 60, 10, 95, 20)
            )
        ),
    )

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is True
    assert report.issues == ()
    assert report.pages == 1
    assert report.words == 2
    assert report.threshold == pytest.approx(0.15)
    assert report.as_dict()["issues"] == []
    assert report.as_dict()["ok"] is True


def test_word_outside_page_is_blocking(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(
        monkeypatch,
        stdout=_bbox(_page(_word("clipped", 10, 10, 240, 20), width=200)),
    )

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is False
    out_of_bounds = [i for i in report.issues if i.code == "word_out_of_bounds"]
    assert len(out_of_bounds) == 1
    assert out_of_bounds[0].page == 1
    assert "clipped" in out_of_bounds[0].detail


def test_word_within_bounds_tolerance_is_not_reported(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    inside = BOUNDS_TOLERANCE / 2
    _patch_run(
        monkeypatch,
        stdout=_bbox(
            _page(
                _word("edge", -inside, -inside, 200 + inside, 100 + inside),
                width=200,
            )
        ),
    )

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is True
    assert "word_out_of_bounds" not in _codes(report)


def test_negative_origin_beyond_tolerance_is_blocking(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(
        monkeypatch,
        stdout=_bbox(
            _page(_word("left", -BOUNDS_TOLERANCE - 0.5, 10, 20, 20))
        ),
    )

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is False
    assert _codes(report).count("word_out_of_bounds") == 1


def test_overlap_above_threshold_is_blocking(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    # 10x10 boxes offset by 1pt: intersection 9x10 = 90, smallest area 100,
    # ratio 0.9.
    _patch_run(
        monkeypatch,
        stdout=_bbox(
            _page(
                _word("alpha", 10, 10, 20, 20)
                + _word("beta", 11, 10, 21, 20)
            )
        ),
    )

    report = check_layout(pdf, overlap_threshold=0.15, min_font_size=None)

    assert report.ok is False
    overlaps = [i for i in report.issues if i.code == "word_overlap"]
    assert len(overlaps) == 1
    assert "alpha" in overlaps[0].detail
    assert "beta" in overlaps[0].detail
    assert "0.90" in overlaps[0].detail


def test_overlap_below_threshold_passes(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    # 10x10 boxes offset by 9pt: intersection 1x10 = 10, ratio 0.10.
    _patch_run(
        monkeypatch,
        stdout=_bbox(
            _page(
                _word("alpha", 10, 10, 20, 20)
                + _word("beta", 19, 10, 29, 20)
            )
        ),
    )

    report = check_layout(pdf, overlap_threshold=0.15, min_font_size=None)

    assert report.ok is True
    assert "word_overlap" not in _codes(report)


def test_overlap_threshold_is_used_verbatim(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(
        monkeypatch,
        stdout=_bbox(
            _page(
                _word("alpha", 10, 10, 20, 20)
                + _word("beta", 11, 10, 21, 20)
            )
        ),
    )

    report = check_layout(pdf, overlap_threshold=0.95, min_font_size=None)

    assert report.ok is True
    assert report.threshold == pytest.approx(0.95)


def test_zero_area_box_is_blocking(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(
        monkeypatch,
        stdout=_bbox(
            _page(
                _word("empty", 30, 30, 30, 30)
                + _word("flat", 50, 50, 60, 50)
                + _word("ok", 10, 10, 20, 20)
            )
        ),
    )

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is False
    degenerate = [i for i in report.issues if i.code == "word_box_degenerate"]
    assert len(degenerate) == 2
    assert "empty" in degenerate[0].detail
    assert "flat" in degenerate[1].detail


def test_missing_tool_reports_error_without_running(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _missing_toolchain("no pdftotext on PATH"))
    calls = _patch_run(monkeypatch, stdout=_bbox(_page("")))

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is False
    assert _codes(report) == ["pdftotext_unavailable"]
    assert "no pdftotext on PATH" in report.issues[0].detail
    assert report.issues[0].severity == "error"
    assert calls == []


def test_unprobed_tool_is_treated_as_unavailable(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, Toolchain(capabilities=()))
    _patch_run(monkeypatch, stdout=_bbox(_page("")))

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is False
    assert _codes(report) == ["pdftotext_unavailable"]


def test_timeout_reports_error(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(
        monkeypatch,
        raises=subprocess.TimeoutExpired(cmd=["pdftotext"], timeout=10.0),
    )

    report = check_layout(pdf, timeout_seconds=3.5, min_font_size=None)

    assert report.ok is False
    assert _codes(report) == ["pdftotext_timeout"]
    assert "3.5" in report.issues[0].detail


def test_non_zero_exit_reports_stderr(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(monkeypatch, returncode=2, stderr=b"Syntax Error: bad PDF\n")

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is False
    assert _codes(report) == ["pdftotext_failed"]
    assert "status 2" in report.issues[0].detail
    assert "bad PDF" in report.issues[0].detail


def test_run_uses_argument_list_shell_false_and_timeout(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain("/opt/bin/pdftotext"))
    calls = _patch_run(monkeypatch, stdout=_bbox(_page("")))

    check_layout(pdf, timeout_seconds=4.0, min_font_size=None)

    assert len(calls) == 1
    call = calls[0]
    assert call["args"] == [
        "/opt/bin/pdftotext",
        "-bbox",
        "-enc",
        "UTF-8",
        str(pdf),
        "-",
    ]
    assert call["shell"] is False
    assert call["timeout"] == pytest.approx(4.0)
    assert call["capture_output"] is True


def test_empty_page_is_valid_layout(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(monkeypatch, stdout=_bbox(_page("")))

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is True
    assert report.pages == 1
    assert report.words == 0
    assert report.issues == ()


def test_multiple_pages_are_counted(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(
        monkeypatch,
        stdout=_bbox(
            _page(_word("one", 10, 10, 20, 20))
            + _page(_word("two", 10, 10, 20, 20) + _word("three", 30, 10, 40, 20))
        ),
    )

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is True
    assert report.pages == 2
    assert report.words == 3


def test_page_numbers_are_one_based(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(
        monkeypatch,
        stdout=_bbox(
            _page("")
            + _page(_word("bad", 10, 10, 240, 20), width=200)
        ),
    )

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is False
    assert report.issues[0].code == "word_out_of_bounds"
    assert report.issues[0].page == 2


def test_xml_entities_are_decoded(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    # ``&amp;`` -> ``&``, ``&#8722;`` -> U+2212, ``&lt;x&gt;`` -> ``<x>``.
    _patch_run(
        monkeypatch,
        stdout=_bbox(_page(_word("A&amp;B&#8722;c&lt;x&gt;", 10, 10, 240, 20))),
    )

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is False
    assert report.issues[0].code == "word_out_of_bounds"
    # Python repr escapes the non-ASCII minus, so match the ASCII parts.
    detail = report.issues[0].detail.encode("ascii", "backslashreplace").decode()
    assert "A&B" in detail
    assert "c<x>" in detail


def test_utf8_bom_in_output_is_tolerated(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(
        monkeypatch,
        stdout=b"\xef\xbb\xbf" + _bbox(_page(_word("hello", 10, 10, 20, 20))),
    )

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is True
    assert report.words == 1


def test_malformed_xhtml_reports_parse_error(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(monkeypatch, stdout=b"<html><page width='200'><word></html>")

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is False
    assert _codes(report) == ["bbox_parse_error"]


def test_empty_output_reports_parse_error(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(monkeypatch, stdout=b"   \n")

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is False
    assert _codes(report) == ["bbox_parse_error"]


def test_missing_page_element_reports_no_pages(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(monkeypatch, stdout=_bbox(""))

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is False
    assert _codes(report) == ["no_pages"]


def test_missing_word_attribute_is_blocking(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    broken = '<word xMin="10" yMin="10" xMax="20">oops</word>'
    _patch_run(monkeypatch, stdout=_bbox(_page(broken)))

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is False
    assert _codes(report) == ["word_geometry_invalid"]
    assert report.issues[0].page == 1


def test_non_numeric_word_attribute_is_blocking(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    broken = '<word xMin="NaN" yMin="10" xMax="20" yMax="20">nan</word>'
    _patch_run(monkeypatch, stdout=_bbox(_page(broken)))

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is False
    assert "word_geometry_invalid" in _codes(report)


def test_page_without_geometry_is_blocking(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(
        monkeypatch,
        stdout=_bbox(
            '<page><word xMin="10" yMin="10" xMax="20" yMax="20">t</word></page>'
        ),
    )

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is False
    assert "page_geometry_invalid" in _codes(report)


def test_missing_pdf_is_reported_before_tool_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    calls = _patch_run(monkeypatch, stdout=_bbox(_page("")))

    report = check_layout(tmp_path / "absent.pdf", min_font_size=None)

    assert report.ok is False
    assert _codes(report) == ["pdf_missing"]
    assert "absent.pdf" in report.issues[0].detail
    assert calls == []


def test_directory_is_not_accepted_as_pdf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(monkeypatch, stdout=_bbox(_page("")))

    report = check_layout(tmp_path, min_font_size=None)

    assert report.ok is False
    assert _codes(report) == ["pdf_missing"]


def test_min_font_size_only_warns(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(monkeypatch, stdout=_bbox(_page(_word("hello", 10, 10, 20, 20))))

    report = check_layout(pdf, min_font_size=6.0)

    assert report.ok is True
    assert _codes(report) == ["font_size_unsupported"]
    assert _severities(report, "font_size_unsupported") == ["warning"]
    assert "6" in report.issues[0].detail


def test_min_font_size_disabled_emits_no_warning(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(monkeypatch, stdout=_bbox(_page(_word("hello", 10, 10, 20, 20))))

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is True
    assert report.issues == ()


def test_min_font_size_warning_does_not_mask_errors(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    _patch_run(
        monkeypatch,
        stdout=_bbox(_page(_word("clipped", 10, 10, 240, 20))),
    )

    report = check_layout(pdf, min_font_size=6.0)

    assert report.ok is False
    assert "font_size_unsupported" in _codes(report)
    assert "word_out_of_bounds" in _codes(report)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), "0.2", None])
def test_invalid_threshold_is_an_error(
    monkeypatch: pytest.MonkeyPatch, pdf: Path, threshold: Any
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    calls = _patch_run(monkeypatch, stdout=_bbox(_page("")))

    report = check_layout(pdf, overlap_threshold=threshold, min_font_size=None)

    assert report.ok is False
    assert _codes(report) == ["invalid_argument"]
    assert calls == []


@pytest.mark.parametrize("timeout", [0, -1.0, float("inf"), "10"])
def test_invalid_timeout_is_an_error(
    monkeypatch: pytest.MonkeyPatch, pdf: Path, timeout: Any
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    calls = _patch_run(monkeypatch, stdout=_bbox(_page("")))

    report = check_layout(pdf, timeout_seconds=timeout, min_font_size=None)

    assert report.ok is False
    assert _codes(report) == ["invalid_argument"]
    assert calls == []


@pytest.mark.parametrize("font_size", [0, -6.0, float("nan"), "6"])
def test_invalid_min_font_size_is_an_error(
    monkeypatch: pytest.MonkeyPatch, pdf: Path, font_size: Any
) -> None:
    _patch_toolchain(monkeypatch, _available_toolchain())
    calls = _patch_run(monkeypatch, stdout=_bbox(_page("")))

    report = check_layout(pdf, min_font_size=font_size)

    assert report.ok is False
    assert _codes(report) == ["invalid_argument"]
    assert calls == []


def test_issue_list_is_capped(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    from deeptutor.review.layout_gate import MAX_ISSUES

    _patch_toolchain(monkeypatch, _available_toolchain())
    # Every word sits past the right edge but none overlaps another, so the
    # only issue class is ``word_out_of_bounds`` and the count is exact.
    words = "".join(
        _word(f"w{index}", 200 + index * 20, 10, 210 + index * 20, 20)
        for index in range(MAX_ISSUES + 10)
    )
    _patch_run(monkeypatch, stdout=_bbox(_page(words, width=200)))

    report = check_layout(pdf, min_font_size=None)

    assert report.ok is False
    assert len(report.issues) == MAX_ISSUES + 1
    assert set(_codes(report)[:-1]) == {"word_out_of_bounds"}
    assert report.issues[-1].code == "issues_truncated"
    # The omitted tail is entirely errors, so the truncation notice must also
    # block; a visible warning would contradict ``ok=False``.
    assert report.issues[-1].severity == "error"
    assert "10 more" in report.issues[-1].detail
    assert "10 errors" in report.issues[-1].detail


def test_report_is_immutable() -> None:
    report = LayoutReport(ok=True, threshold=0.15)
    with pytest.raises(Exception):
        report.ok = False  # type: ignore[misc]

    issue = LayoutIssue(code="x", detail="y")
    with pytest.raises(Exception):
        issue.code = "z"  # type: ignore[misc]


def test_report_as_dict_is_json_ready() -> None:
    report = LayoutReport(
        ok=False,
        issues=(
            LayoutIssue(code="word_overlap", detail="a/b", page=2),
            LayoutIssue(
                code="font_size_unsupported", detail="n/a", severity="warning"
            ),
        ),
        pages=2,
        words=7,
        threshold=0.15,
    )

    payload = report.as_dict()

    assert payload == {
        "ok": False,
        "pages": 2,
        "words": 7,
        "threshold": 0.15,
        "issues": [
            {
                "code": "word_overlap",
                "detail": "a/b",
                "page": 2,
                "severity": "error",
            },
            {
                "code": "font_size_unsupported",
                "detail": "n/a",
                "page": 0,
                "severity": "warning",
            },
        ],
    }
