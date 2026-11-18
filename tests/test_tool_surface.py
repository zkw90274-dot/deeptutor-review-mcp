"""The MCP tool surface and the in-app manifest must not drift.

The MCP server exposes every agent tool; the in-app manifest exposes the subset
a chat loop can call. ``review_export_tree`` is deliberately MCP-only: the web
UI already has an export page, and putting a write-to-disk tool in the in-app
agent would widen its write surface for no benefit. That exclusion is the only
allowed difference, so it is written down here rather than left implicit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deeptutor.review.mcp_server import TOOL_CLASSES

MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "deeptutor"
    / "modules"
    / "manifests"
    / "review.json"
)

#: MCP-only tools. Keep this set small and keep the reason explicit.
MCP_ONLY = {"review_export_tree"}


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_covers_every_tool_except_the_documented_mcp_only_one() -> None:
    manifest_tools = set(_manifest()["tools"])
    assert set(TOOL_CLASSES) == manifest_tools | MCP_ONLY
    assert manifest_tools.isdisjoint(MCP_ONLY)


def test_manifest_class_paths_match_the_tool_classes() -> None:
    tools = _manifest()["tools"]
    for name, entry in tools.items():
        expected = f"deeptutor.review.agent_tools:{TOOL_CLASSES[name].__name__}"
        assert entry["class_path"] == expected, name


def test_manifest_orders_are_unique() -> None:
    orders = [entry["order"] for entry in _manifest()["tools"].values()]
    assert len(orders) == len(set(orders))


def test_experimental_catalog_matches_external_tools() -> None:
    package_root = MANIFEST.parents[4]
    if package_root.parent.name != "packages":
        pytest.skip("experimental catalog lives only in the development checkout")
    catalog = package_root.parents[1] / "experimental" / "mcp_catalog" / "deeptutor-review.json"
    listed = json.loads(catalog.read_text(encoding="utf-8"))["server"]["enabled_tools"]
    assert len(listed) == len(set(listed))
    assert set(listed) == set(TOOL_CLASSES)


def test_check_reports_ready(capsys) -> None:
    from deeptutor.review.mcp_server import _check

    assert _check() == 0
    out = capsys.readouterr().out
    assert "module gate       : ok" in out
    assert out.count("mcp tools         : ") == 1
    assert f"mcp tools         : {len(TOOL_CLASSES)}" in out
    assert len(TOOL_CLASSES) == 11
