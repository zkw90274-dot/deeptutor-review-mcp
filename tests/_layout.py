"""Locate repo-level assets in either checkout layout.

The package also lives inside a monorepo at <repo>/packages/deeptutor-review,
so tests cannot hard-code a depth. Search upward for marker files instead.
"""

from __future__ import annotations

from pathlib import Path

MARKERS = (
    Path("scripts") / "build-book-figures.py",
    Path("src") / "deeptutor" / "review" / "mcp_server.py",
)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if all((parent / marker).exists() for marker in MARKERS):
            return parent
    # Monorepo fallback: tests -> packages/deeptutor-review -> packages -> repo
    return here.parents[3]


def skill_dir() -> Path:
    root = repo_root()
    for candidate in (root / "docs" / "review-studio" / "skill", root / "docs" / "skill"):
        if (candidate / "SKILL.md").is_file():
            return candidate
    return root / "docs" / "skill"
