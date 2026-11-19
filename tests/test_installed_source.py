"""Guard against shipping a stale wheel.

``conftest.py`` prepends the checkout to ``deeptutor.__path__``, so the suite
exercises the working tree rather than the installed wheel. That is deliberate
— otherwise a source edit is invisible until someone rebuilds — but it means a
green suite no longer proves the wheel is current. This test closes that gap:
the installed distribution must match the checkout byte for byte, and it fails
with a rebuild instruction when the two have drifted.

The installed file is located through the distribution record inside the
virtualenv, not through ``metadata.distribution("deeptutor-review")``: that
lookup walks ``sys.path``, where ``conftest.py`` has just placed the checkout,
and the ``src/deeptutor_review.egg-info`` left by wheel builds makes it resolve
to the checkout itself — comparing a tree against itself always passes.
"""

from __future__ import annotations

import hashlib
import sysconfig
from importlib import metadata
from importlib.resources import files
from pathlib import Path


def _digest(path: Path) -> str:
    """Hash file *content*, ignoring the checkout's line endings.

    The repository is checked out with ``core.autocrlf=true``, so a working-tree
    file is CRLF while the wheel built from ``git archive``-like input holds LF.
    Hashing raw bytes would turn a re-checkout into a bogus "rebuild the wheel"
    failure, so the comparison normalises newlines first.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _installed_distribution() -> metadata.Distribution:
    purelib = sysconfig.get_paths()["purelib"]
    for distribution in metadata.distributions(path=[purelib]):
        name = (distribution.metadata["Name"] or "").lower().replace("_", "-")
        if name == "deeptutor-review":
            return distribution
    raise AssertionError(f"deeptutor-review is not installed under {purelib}")


def _installed_path(relative: str) -> Path:
    path = Path(_installed_distribution().locate_file(relative))
    assert path.is_file(), f"{relative} is missing from the installed wheel"
    return path


def test_installed_review_package_matches_source_tree() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "deeptutor" / "review"
    files = sorted(
        path.relative_to(source_root)
        for path in source_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    assert files
    installed_root = _installed_path("deeptutor/review/__init__.py").parent
    installed_files = {
        path.relative_to(installed_root)
        for path in installed_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
    }
    assert installed_files == set(files), (
        f"installed-only files: {sorted(str(path) for path in installed_files - set(files))}; "
        f"missing installed files: {sorted(str(path) for path in set(files) - installed_files)}"
    )
    drifted = [
        str(relative)
        for relative in files
        if _digest(source_root / relative)
        != _digest(_installed_path(f"deeptutor/review/{relative.as_posix()}"))
    ]
    assert not drifted, (
        "these files differ between the checkout and the installed wheel; "
        f"rebuild/install before running the acceptance script: {drifted}"
    )


def test_installed_manifest_matches_source_tree() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "deeptutor"
        / "modules"
        / "manifests"
        / "review.json"
    )
    installed = _installed_path("deeptutor/modules/manifests/review.json")
    assert _digest(source) == _digest(installed), "manifest drift; rebuild/install"


def test_wheel_ships_skill_md() -> None:
    name = "deeptutor/review/skill/SKILL.md"
    records = _installed_distribution().files or ()
    assert name in {record.as_posix() for record in records}
    assert _installed_path(name).read_bytes() == files("deeptutor.review").joinpath("skill", "SKILL.md").read_bytes()


def test_module_catalog_still_resolves_to_the_installed_distribution() -> None:
    """The checkout must not shadow the manifest catalog.

    ``conftest.py`` prepends the checkout to ``deeptutor.__path__``. The single
    deliberate exception is ``deeptutor.modules``: the checkout has no
    ``__init__.py`` for it, so the regular package in site-packages wins — which
    is what the runtime loads. If someone ever adds that ``__init__.py``, the
    catalog would silently switch to an untested copy; this guard turns that
    into a loud failure.
    """
    import deeptutor.modules.catalog as catalog

    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    resolved = Path(catalog.__file__).resolve()
    assert purelib in resolved.parents, (
        f"deeptutor.modules resolved to {resolved} instead of the installed "
        f"distribution under {purelib}"
    )
