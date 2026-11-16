"""The personalization MCP store entry for a self-hosted Review Studio.

The entry lives in the core catalog after the deployment patch. These tests
pin the two things the store must get right: the installer supplies their own
public HTTPS endpoint, and the bearer token is stored as a secret reference
rather than plaintext.
"""

from __future__ import annotations

import pytest

from deeptutor.multi_user.paths import current_owner_id
from deeptutor.services.mcp.catalog import (
    build_server_config,
    get_entry,
    search_catalog,
)
from deeptutor.services.mcp.secrets import configured_fields
from deeptutor.services.mcp.user_config import load_user_mcp_config

ENTRY_ID = "deeptutor-review"
PUBLIC_URL = "https://review.example.com/mcp"
TOOL_NAMES = {
    "review_specs",
    "review_today",
    "review_attempt",
    "review_hint",
    "review_weakness",
    "review_export_link",
    "review_export_tree",
    "review_book_figures",
    "review_generate",
    "review_plan",
}


@pytest.fixture(scope="module", autouse=True)
def _require_catalog_entry():
    """Skip this module until the packaged store entry is registered.

    Registering is the post-install step (deeptutor-review-catalog). That command
    rewrites files inside the installed deeptutor package, so doing it from inside
    a test session would leave already-imported modules stale: CI registers the
    entry before pytest, and a local run without it skips with this message.
    """
    from deeptutor.services.mcp.catalog import get_entry

    if get_entry(ENTRY_ID) is None:
        pytest.skip("run deeptutor-review-catalog before these tests")


@pytest.fixture(autouse=True)
def _isolated_user_store(tmp_path, monkeypatch):
    """Never let a store test read or write the deployment's real user config."""
    import deeptutor.multi_user.paths as paths

    monkeypatch.setattr(paths, "SYSTEM_ROOT", tmp_path / "system")


class _Manager:
    async def reload_scope(self, owner):
        return None

    async def ensure_started(self):
        return None

    async def ensure_scope(self, owner):
        return None

    def status(self, owner):
        return {"servers": {}, "deployment": {}}


def test_entry_is_self_service_remote() -> None:
    entry = get_entry(ENTRY_ID)
    assert entry is not None, "run deeptutor-review-catalog first"
    assert entry.self_service is True
    assert entry.transport == "streamableHttp"
    assert {field.key for field in entry.fields} == {"url", "token"}

    url_field = next(field for field in entry.fields if field.key == "url")
    token_field = next(field for field in entry.fields if field.key == "token")
    assert url_field.secret is False and url_field.required is True
    assert token_field.secret is True and token_field.required is True
    assert token_field.value_template == "Bearer {value}"


def test_build_server_config_applies_url_and_token() -> None:
    entry = get_entry(ENTRY_ID)
    built = build_server_config(entry, {"url": PUBLIC_URL, "token": "secret"})

    assert built.config.url == PUBLIC_URL
    assert built.config.resolved_type() == "streamableHttp"
    assert built.config.catalog_entry == ENTRY_ID
    assert built.config.headers["Authorization"] == f"${{secret:{ENTRY_ID}/token}}"
    assert built.secret_values["token"] == "Bearer secret"
    assert set(built.config.enabled_tools) == TOOL_NAMES


def test_missing_url_is_rejected() -> None:
    entry = get_entry(ENTRY_ID)
    with pytest.raises(ValueError, match="url"):
        build_server_config(entry, {"token": "secret"})


def test_plain_http_url_is_rejected() -> None:
    entry = get_entry(ENTRY_ID)
    with pytest.raises(ValueError, match="https"):
        build_server_config(
            entry, {"url": "http://review.example.com/mcp", "token": "secret"}
        )


def test_catalog_search_includes_entry() -> None:
    page = search_catalog(q="review", self_service_only=True)
    assert ENTRY_ID in [entry.id for entry in page.entries]


def test_api_catalog_lists_entry(client) -> None:
    response = client.get("/api/space/mcp/catalog", params={"q": "review"})
    assert response.status_code == 200, response.text
    row = next(
        entry for entry in response.json()["entries"] if entry["id"] == ENTRY_ID
    )
    assert row["self_service"] is True
    assert row["transport"] == "streamableHttp"
    assert {field["key"] for field in row["fields"]} == {"url", "token"}


def test_api_install_persists_to_the_isolated_store(client, monkeypatch) -> None:
    # The real manager would warm the newly installed endpoint; the install
    # path and its persistence are the SUT, not a live TLS handshake.
    monkeypatch.setattr(
        "deeptutor.api.routers.space_mcp.get_mcp_manager", lambda: _Manager()
    )

    response = client.post(
        f"/api/space/mcp/catalog/{ENTRY_ID}/install",
        json={
            "name": "review-store-test",
            "secrets": {"url": "https://1.1.1.1/mcp", "token": "test-token"},
        },
    )
    assert response.status_code == 200, response.text

    owner = current_owner_id()
    config, _ = load_user_mcp_config(owner)
    cfg = config.servers["review-store-test"]
    assert cfg.url == "https://1.1.1.1/mcp"
    assert cfg.catalog_entry == ENTRY_ID
    assert cfg.headers["Authorization"] == f"${{secret:{ENTRY_ID}/token}}"
    assert "token" in configured_fields(owner, "review-store-test")


def test_api_install_rejects_loopback(client) -> None:
    response = client.post(
        f"/api/space/mcp/catalog/{ENTRY_ID}/install",
        json={
            "name": "review-loopback",
            "secrets": {"url": "https://127.0.0.1:8767/mcp", "token": "t"},
        },
    )
    assert response.status_code == 400, response.text
