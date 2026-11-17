#!/usr/bin/env python3
"""Install a deeptutor-review wheel without letting pip touch the core.

The core module gate requires ``deeptutor-review``'s version to equal
``deeptutor.__version__``. A plain ``pip install .`` lets pip resolve
``Requires-Dist: deeptutor==X`` and silently downgrade a newer core, after
which the review module disappears (the route is not mounted and the UI 404s).

This script reads the wheel metadata first, refuses on a mismatch, and only
then installs with ``--no-deps``. It uses the standard library only, so it runs
the same on Windows and Linux.
"""

from __future__ import annotations

import argparse
import email.parser
import subprocess
import sys
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _core_version() -> str:
    try:
        return version("deeptutor")
    except PackageNotFoundError:
        raise SystemExit(
            "deeptutor is not installed; install the core version this wheel "
            "targets before installing deeptutor-review"
        )


def _required_core(wheel: Path) -> str:
    """Read the core pin from the wheel's METADATA, without installing it."""
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            (name for name in archive.namelist() if name.endswith(".dist-info/METADATA")),
            None,
        )
        if metadata_name is None:
            raise SystemExit(f"{wheel} has no .dist-info/METADATA; is it a wheel?")
        metadata = email.parser.Parser().parsestr(
            archive.read(metadata_name).decode("utf-8")
        )

    for requirement in metadata.get_all("Requires-Dist") or []:
        # Markers are not interesting for the core pin; strip them before the
        # exact-match check. deeptutor-mastery travels on the same release
        # train but is not the gate.
        bare = requirement.split(";", 1)[0].replace(" ", "")
        if bare.startswith("deeptutor=="):
            return bare.split("==", 1)[1]

    raise SystemExit(f"{wheel} does not pin deeptutor==<version>")


def _default_wheel() -> Path:
    package_root = Path(__file__).resolve().parents[1]
    wheels = sorted((package_root / "dist").glob("deeptutor_review-*.whl"))
    if not wheels:
        raise SystemExit("no dist/deeptutor_review-*.whl found; build the wheel first")
    return wheels[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="install-review.py",
        description="Install a deeptutor-review wheel without downgrading the core.",
    )
    parser.add_argument("wheel", nargs="?", type=Path, help="Wheel to install; defaults to the newest in dist/.")
    parser.add_argument("--dry-run", action="store_true", help="Check versions and print the command; install nothing.")
    parser.add_argument("--python", default=sys.executable, help="Python used for pip (default: this interpreter).")
    args = parser.parse_args(argv)

    wheel = args.wheel or _default_wheel()
    if not wheel.is_file():
        raise SystemExit(f"wheel not found: {wheel}")

    core = _core_version()
    required = _required_core(wheel)
    print(f"installed deeptutor: {core}")
    print(f"wheel requires     : deeptutor=={required}")

    if core != required:
        print(
            "refusing to install: versions differ. A plain pip install would "
            "downgrade the core and the review module would stop mounting.",
            file=sys.stderr,
        )
        print(
            f"install deeptutor=={required} first, or build a wheel for deeptutor=={core}.",
            file=sys.stderr,
        )
        return 1

    command = [
        args.python,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        str(wheel),
    ]
    print("running:", " ".join(command))
    if args.dry_run:
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
