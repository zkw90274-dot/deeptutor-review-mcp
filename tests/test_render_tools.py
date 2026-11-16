from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from deeptutor.review.render_tools import (
    PDFTOCAIRO,
    PDFTOHTML,
    PDFTOTEXT,
    VERSION_TIMEOUT_SECONDS,
    XELATEX,
    ToolCapability,
    Toolchain,
    capability_summary,
    clear_toolchain_cache,
    probe_toolchain,
)

_ENV_VARS = (
    "DEEPTUTOR_XELATEX",
    "DEEPTUTOR_PDFTOCAIRO",
    "DEEPTUTOR_PDFTOTEXT",
    "DEEPTUTOR_PDFTOHTML",
)


@pytest.fixture(autouse=True)
def _isolate_probe(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Every test starts from an empty cache and a clean override environment."""
    for variable in _ENV_VARS:
        monkeypatch.delenv(variable, raising=False)
    clear_toolchain_cache()
    yield
    clear_toolchain_cache()


def _patch_which(
    monkeypatch: pytest.MonkeyPatch, mapping: dict[str, str | None]
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: mapping.get(name))


def _completed(
    args: list[str],
    returncode: int = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        args=args, returncode=returncode, stdout=stdout, stderr=stderr
    )


def _tool_name(args: list[str]) -> str:
    return Path(args[0]).stem


def _capability(toolchain: Toolchain, name: str) -> ToolCapability:
    capability = toolchain.capability(name)
    assert capability is not None, f"{name} missing from probe result"
    return capability


def test_all_tools_present_versions_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_which(
        monkeypatch,
        {
            XELATEX: "/opt/tex/xelatex",
            PDFTOCAIRO: "/usr/bin/pdftocairo",
            PDFTOTEXT: "/usr/bin/pdftotext",
            PDFTOHTML: "/usr/bin/pdftohtml",
        },
    )
    calls: list[list[str]] = []
    versions = {
        XELATEX: (b"XeTeX 3.141592653-2.6-0.999995 (TeX Live 2023)\n", b""),
        PDFTOCAIRO: (b"", b"pdftocairo version 22.02.0\n"),
        PDFTOTEXT: (b"pdftotext version 22.02.0\n", b""),
        PDFTOHTML: (b"pdftohtml version 22.02.0\n", b""),
    }

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(args)
        if _tool_name(args) == PDFTOTEXT:
            if "-bbox" in args:
                return _completed(
                    args, stderr=b"  -bbox : output bounding box for each word\n"
                )
            return _completed(args, stdout=b"pdftotext version 22.02.0\n")
        if _tool_name(args) == PDFTOHTML:
            if "-xml" in args:
                return _completed(
                    args, stderr=b"  -xml : output for XML post-processing\n"
                )
            return _completed(args, stdout=b"pdftohtml version 22.02.0\n")
        stdout, stderr = versions[_tool_name(args)]
        return _completed(args, stdout=stdout, stderr=stderr)

    monkeypatch.setattr("subprocess.run", fake_run)

    first = probe_toolchain()
    second = probe_toolchain()

    assert first is second
    assert len(calls) == 6
    assert first.available(XELATEX)
    assert _capability(first, XELATEX).version.startswith("XeTeX")
    assert _capability(first, PDFTOCAIRO).version == "pdftocairo version 22.02.0"
    assert _capability(first, PDFTOTEXT).version == "pdftotext version 22.02.0"
    assert _capability(first, PDFTOHTML).version == "pdftohtml version 22.02.0"
    assert first.missing() == ()
    assert first.warnings() == ()

    summary = capability_summary(first)
    assert summary["tikz_available"] is True
    assert summary["layout_check_available"] is True
    assert summary["font_check_available"] is True
    assert summary["missing"] == []


def test_partial_toolchain_degrades_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_which(monkeypatch, {XELATEX: "/opt/tex/xelatex"})
    monkeypatch.setattr(
        "subprocess.run",
        lambda args, **kwargs: _completed(args, stdout=b"XeTeX 3.14\n"),
    )

    toolchain = probe_toolchain()

    assert toolchain.available(XELATEX)
    assert not toolchain.available(PDFTOCAIRO)
    assert not toolchain.available(PDFTOTEXT)
    assert not toolchain.available(PDFTOHTML)
    assert toolchain.missing() == (PDFTOCAIRO, PDFTOTEXT, PDFTOHTML)

    summary = capability_summary(toolchain)
    assert summary["tikz_available"] is False
    assert summary["layout_check_available"] is False
    assert summary["font_check_available"] is False
    warnings = summary["warnings"]
    assert any("pdftocairo unavailable: not found on PATH" in item for item in warnings)
    assert any("pdftotext unavailable: not found on PATH" in item for item in warnings)
    assert any("pdftohtml unavailable: not found on PATH" in item for item in warnings)


def test_version_timeout_marks_tool_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_which(
        monkeypatch,
        {
            XELATEX: "/opt/tex/xelatex",
            PDFTOCAIRO: "/usr/bin/pdftocairo",
            PDFTOTEXT: "/usr/bin/pdftotext",
        },
    )

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if _tool_name(args) == PDFTOCAIRO:
            raise subprocess.TimeoutExpired(args, VERSION_TIMEOUT_SECONDS)
        if _tool_name(args) == PDFTOTEXT and "-bbox" in args:
            return _completed(args, stderr=b"  -bbox : bounding box\n")
        return _completed(args, stdout=b"version 1.0\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    toolchain = probe_toolchain()
    capability = _capability(toolchain, PDFTOCAIRO)

    assert not capability.available
    assert "timed out" in capability.reason
    assert toolchain.available(XELATEX)
    assert toolchain.available(PDFTOTEXT)
    assert any("timed out" in warning for warning in toolchain.warnings())


def test_version_failure_marks_tool_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_which(
        monkeypatch,
        {
            XELATEX: "/opt/tex/xelatex",
            PDFTOCAIRO: "/usr/bin/pdftocairo",
            PDFTOTEXT: "/usr/bin/pdftotext",
        },
    )

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if _tool_name(args) == PDFTOTEXT:
            return _completed(args, returncode=3)
        return _completed(args, stdout=b"version 1.0\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    capability = _capability(probe_toolchain(), PDFTOTEXT)

    assert not capability.available
    assert capability.executable == "/usr/bin/pdftotext"
    assert "status 3" in capability.reason


def test_xpdf_version_exit_99_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Xpdf tools print the version banner and then exit with 99."""
    _patch_which(
        monkeypatch,
        {
            XELATEX: "/opt/tex/xelatex",
            PDFTOCAIRO: "/usr/bin/pdftocairo",
            PDFTOTEXT: "/usr/bin/pdftotext",
        },
    )

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if _tool_name(args) == PDFTOTEXT:
            if "-bbox" in args:
                return _completed(
                    args, returncode=99, stderr=b"  -bbox : bounding box\n"
                )
            return _completed(args, returncode=99, stderr=b"pdftotext version 4.00\n")
        return _completed(args, stdout=b"version 1.0\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    capability = _capability(probe_toolchain(), PDFTOTEXT)

    assert capability.available
    assert capability.version == "pdftotext version 4.00"


@pytest.mark.parametrize(
    "override",
    [
        "tools/xelatex",
        "xelatex",
    ],
)
def test_relative_env_override_is_rejected(
    monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    monkeypatch.setenv("DEEPTUTOR_XELATEX", override)
    _patch_which(monkeypatch, {})
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: pytest.fail("invalid override must not be executed"),
    )

    capability = _capability(probe_toolchain(), XELATEX)

    assert not capability.available
    assert capability.executable is None
    assert "DEEPTUTOR_XELATEX" in capability.reason


def test_missing_absolute_env_override_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override = tmp_path / "missing-xelatex"
    monkeypatch.setenv("DEEPTUTOR_XELATEX", str(override))
    _patch_which(monkeypatch, {})
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: pytest.fail("invalid override must not be executed"),
    )

    capability = _capability(probe_toolchain(), XELATEX)

    assert not capability.available
    assert "DEEPTUTOR_XELATEX" in capability.reason


def test_valid_env_override_wins_over_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override = tmp_path / "custom-xelatex"
    override.write_text("#!/bin/sh\n", encoding="utf-8")
    override.chmod(0o755)
    monkeypatch.setenv("DEEPTUTOR_XELATEX", str(override))
    _patch_which(monkeypatch, {XELATEX: "/usr/bin/xelatex"})
    monkeypatch.setattr(
        "subprocess.run",
        lambda args, **kwargs: _completed(args, stdout=b"XeTeX custom\n"),
    )

    capability = _capability(probe_toolchain(), XELATEX)

    assert capability.available
    assert capability.executable == str(override)
    assert capability.version == "XeTeX custom"


def test_xpdf_exit_99_is_only_accepted_for_pdftotext(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_which(
        monkeypatch,
        {
            XELATEX: "/opt/tex/xelatex",
            PDFTOCAIRO: "/usr/bin/pdftocairo",
            PDFTOTEXT: "/usr/bin/pdftotext",
        },
    )

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if _tool_name(args) == XELATEX:
            return _completed(args, returncode=99, stdout=b"XeTeX 3.14\n")
        return _completed(args, stdout=b"version 1.0\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    capability = _capability(probe_toolchain(), XELATEX)
    assert not capability.available
    assert "status 99" in capability.reason


def test_pdftotext_without_bbox_is_reported_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A binary that cannot do -bbox must not be advertised as usable."""
    _patch_which(
        monkeypatch,
        {
            XELATEX: "/opt/tex/xelatex",
            PDFTOCAIRO: "/usr/bin/pdftocairo",
            PDFTOTEXT: "/usr/bin/pdftotext",
        },
    )
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(args)
        if _tool_name(args) == PDFTOTEXT:
            if "-bbox" in args:
                return _completed(
                    args,
                    returncode=99,
                    stderr=b"Usage: pdftotext [options] <PDF-file> [<text-file>]\n  -f\n",
                )
            return _completed(args, returncode=99, stderr=b"pdftotext version 4.00\n")
        return _completed(args, stdout=b"version 1.0\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    toolchain = probe_toolchain()
    capability = _capability(toolchain, PDFTOTEXT)

    assert not capability.available
    assert capability.version == "pdftotext version 4.00"
    assert "-bbox" in capability.reason
    assert toolchain.as_dict()["layout_check_available"] is False
    assert any("-bbox" in call for call in calls)


def test_pdftotext_with_bbox_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_which(
        monkeypatch,
        {
            XELATEX: "/opt/tex/xelatex",
            PDFTOCAIRO: "/usr/bin/pdftocairo",
            PDFTOTEXT: "/usr/bin/pdftotext",
        },
    )

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if _tool_name(args) == PDFTOTEXT:
            if "-bbox" in args:
                return _completed(
                    args,
                    stderr=b"  -bbox : output bounding box for each word\n",
                )
            return _completed(args, stdout=b"pdftotext version 26.02.0\n")
        return _completed(args, stdout=b"version 1.0\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    toolchain = probe_toolchain()
    capability = _capability(toolchain, PDFTOTEXT)

    assert capability.available
    assert capability.version == "pdftotext version 26.02.0"
    assert toolchain.as_dict()["layout_check_available"] is True


def test_pdftohtml_with_xml_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_which(monkeypatch, {PDFTOHTML: "/usr/bin/pdftohtml"})

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if "-xml" in args:
            return _completed(
                args, returncode=1, stderr=b"  -xml : output for XML post-processing\n"
            )
        return _completed(args, stdout=b"pdftohtml version 26.02.0\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    toolchain = probe_toolchain()
    capability = _capability(toolchain, PDFTOHTML)

    assert capability.available
    assert capability.version == "pdftohtml version 26.02.0"
    assert capability.supports("-xml")
    assert toolchain.as_dict()["font_check_available"] is True


def test_pdftohtml_without_xml_is_reported_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A build that cannot emit XML must not be advertised as usable."""
    _patch_which(monkeypatch, {PDFTOHTML: "/usr/bin/pdftohtml"})

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if "-xml" in args:
            return _completed(
                args,
                returncode=1,
                stderr=b"Usage: pdftohtml [options] <PDF-file> [<html-file>]\n",
            )
        return _completed(args, stdout=b"pdftohtml version 0.40\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    toolchain = probe_toolchain()
    capability = _capability(toolchain, PDFTOHTML)

    assert not capability.available
    assert capability.version == "pdftohtml version 0.40"
    assert "-xml" in capability.reason
    assert "does not support" in capability.reason
    assert toolchain.as_dict()["font_check_available"] is False


def test_xpdf_exit_99_is_not_accepted_for_pdftohtml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Xpdf 99 exception belongs to pdftotext only."""
    _patch_which(monkeypatch, {PDFTOHTML: "/usr/bin/pdftohtml"})

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return _completed(args, returncode=99, stdout=b"pdftohtml version 26.02.0\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    capability = _capability(probe_toolchain(), PDFTOHTML)

    assert not capability.available
    assert "status 99" in capability.reason
