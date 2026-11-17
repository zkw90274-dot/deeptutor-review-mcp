"""The pre-install guard must refuse a wheel built for a different core."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install-review.py"


def _load():
    spec = importlib.util.spec_from_file_location("install_review", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wheel(tmp_path: Path, core: str) -> Path:
    wheel = tmp_path / "deeptutor_review-9.9.9-py3-none-any.whl"
    metadata = (
        "Metadata-Version: 2.1\n"
        "Name: deeptutor-review\n"
        "Version: 9.9.9\n"
        f"Requires-Dist: deeptutor=={core}\n"
        "Requires-Dist: deeptutor-mastery==9.9.9\n"
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("deeptutor_review-9.9.9.dist-info/METADATA", metadata)
    return wheel


def test_required_core_reads_the_wheel_pin(tmp_path: Path) -> None:
    module = _load()
    assert module._required_core(_wheel(tmp_path, "9.9.9")) == "9.9.9"


def test_guard_refuses_a_mismatched_core(tmp_path: Path, capsys) -> None:
    module = _load()
    wheel = _wheel(tmp_path, "0.0.0")
    assert module.main([str(wheel), "--dry-run"]) == 1
    assert "refusing to install" in capsys.readouterr().err


def test_guard_allows_a_matching_core(tmp_path: Path, capsys) -> None:
    from importlib.metadata import version

    module = _load()
    wheel = _wheel(tmp_path, version("deeptutor"))
    assert module.main([str(wheel), "--dry-run"]) == 0
    assert "running:" in capsys.readouterr().out
