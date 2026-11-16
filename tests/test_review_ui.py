"""WS-B: read-only overview page (GET /ui) and manifest navigation/tools."""

from __future__ import annotations

import importlib
import json
import os
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deeptutor.review import api as review_api
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.storage import ReviewStore

MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "deeptutor"
    / "modules"
    / "manifests"
    / "review.json"
)
AGENT_TOOLS_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "deeptutor"
    / "review"
    / "agent_tools.py"
)
EXPECTED_TOOLS = {
    "review_specs": "ReviewSpecsTool",
    "review_today": "ReviewTodayTool",
    "review_attempt": "ReviewAttemptTool",
    "review_weakness": "ReviewWeaknessTool",
    "review_export_link": "ReviewExportLinkTool",
    "review_generate": "ReviewGenerateTool",
    "review_hint": "ReviewHintTool",
    "review_plan": "ReviewPlanTool",
}


def _put(client: TestClient, spec=build_fixture_spec()):
    payload = spec.model_dump(mode="json")
    return client.put(f"/api/review/specs/{spec.id}", json=payload)


def test_manifest_is_valid_json_with_navigation_and_tools() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["navigation"] == ["/review", "/api/review/ui"]
    tools = manifest["tools"]
    assert set(tools) == set(EXPECTED_TOOLS)
    for name, class_name in EXPECTED_TOOLS.items():
        entry = tools[name]
        # Declaration-only check: the string must be a well-formed import path
        # pointing at the module WS-A owns. Importing it is conditional (see
        # test below) because the module may not be delivered yet.
        assert entry["class_path"] == f"deeptutor.review.agent_tools:{class_name}"
        assert entry["class_path"].split(":")[0] == "deeptutor.review.agent_tools"
        assert isinstance(entry["order"], int)


def test_manifest_tool_orders_are_unique_and_in_range() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    orders = [entry["order"] for entry in manifest["tools"].values()]
    assert len(orders) == len(set(orders))
    assert all(60 <= order <= 67 for order in orders)


def test_agent_tools_classes_exist_once_delivered() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert AGENT_TOOLS_PATH.is_file()
    module = importlib.import_module("deeptutor.review.agent_tools")
    for name, class_name in EXPECTED_TOOLS.items():
        assert manifest["tools"][name]["class_path"].endswith(class_name)
        assert hasattr(module, class_name)


def test_ui_lists_spec_card_with_export_link(client: TestClient) -> None:
    response = _put(client)
    assert response.status_code == 201
    page = client.get("/api/review/ui")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    body = page.text
    # Card identity: title and spec id, plus the export link the contract fixes.
    assert "示例课程 · 串联谐振复习" in body
    assert SPEC_ID in body
    assert f'href="/api/review/{SPEC_ID}/export"' in body
    # Card facts: item/concept counts, QA summary, today's due count.
    assert "条目" in body
    assert "QA：" in body
    assert "今日待复习" in body


def test_ui_route_is_registered(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/review/ui" in paths


def test_ui_is_read_only(client: TestClient) -> None:
    spec = build_fixture_spec()
    assert _put(client, spec).status_code == 201
    store = ReviewStore(Path(os.environ["DEEPTUTOR_REVIEW_HOME"]))
    before = _row_counts(store.db_path)
    for _ in range(2):
        assert client.get("/api/review/ui").status_code == 200
    assert _row_counts(store.db_path) == before


def _row_counts(db_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    conn = sqlite3.connect(db_path)
    try:
        for table in ("review_events", "review_views", "review_sessions"):
            counts[table] = conn.execute(
                f"SELECT COUNT(*) FROM {table}"  # noqa: S608 - fixed table names
            ).fetchone()[0]
    finally:
        conn.close()
    return counts


def test_ui_empty_library_shows_explaining_copy(client: TestClient) -> None:
    page = client.get("/api/review/ui")
    assert page.status_code == 200
    assert "还没有任何复习包" in page.text


def test_ui_escapes_learner_content(client: TestClient) -> None:
    # The QA injection gate rejects literal <script> at PUT time, so use markup
    # characters that survive storage and prove the page escapes them.
    spec = build_fixture_spec()
    payload = spec.model_dump(mode="json")
    payload["title"] = "R&D <beta>"
    # The title participates in the content hash; clear it so the spec is
    # accepted after the edit (same convention as the linked_spec fixture).
    payload["content_hash"] = ""
    put = client.put(f"/api/review/specs/{spec.id}", json=payload)
    assert put.status_code == 201
    page = client.get("/api/review/ui")
    assert page.status_code == 200
    assert "R&amp;D &lt;beta&gt;" in page.text
    assert "<beta>" not in page.text


def test_ui_service_error_maps_through_handle(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenService:
        def list_specs(self):
            raise ValueError("boom")

    monkeypatch.setattr(review_api, "_service", lambda: _BrokenService())
    page = client.get("/api/review/ui")
    assert page.status_code == 400
    assert page.json()["detail"]["code"] == "invalid_request"


def test_overview_template_has_single_tokens() -> None:
    template = (
        Path(review_api.__file__).parent / "templates" / "overview.html"
    ).read_text(encoding="utf-8")
    assert template.count("{{TITLE}}") == 1
    assert template.count("{{SUMMARY}}") == 1
    assert template.count("{{CARDS}}") == 1


def test_export_route_accepts_get_for_browser_links(client: TestClient) -> None:
    """The overview page and ``review_export_link`` hand out plain links, and
    a browser can only follow GET — a POST-only route turned every one of
    those links into a 405."""
    _put(client)
    response = client.get(f"/api/review/{SPEC_ID}/export")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "attachment" in response.headers["content-disposition"]
    assert "<!doctype html>" in response.text.lower()
