"""Capability flags, read-only Today, and export path containment."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3

import pytest

from deeptutor.review import capabilities
from deeptutor.review.agent_tools import (
    ReviewAttemptTool,
    ReviewGenerateTool,
    _resolve_export_dir,
)
from deeptutor.review.fixture import build_fixture_spec
from deeptutor.review.mcp_server import (
    AUTH_TOKEN_ENV,
    _configure_capabilities,
    _is_loopback,
    main,
)
from deeptutor.review.service import ReviewService

_FLAGS = (
    capabilities.READ_ONLY,
    capabilities.NO_ATTEMPTS,
    capabilities.NO_PLAN,
    capabilities.NO_GENERATE,
    capabilities.NO_EXPORT_WRITE,
    capabilities.ALLOW_OUT_DIRS,
)


@pytest.fixture(autouse=True)
def _clean_flags():
    # ``_configure_capabilities`` writes ``os.environ`` directly, which
    # monkeypatch does not track. Clear on both sides so a flag cannot leak
    # into the generation tests that run later in the same process.
    for name in _FLAGS:
        os.environ.pop(name, None)
    yield
    for name in _FLAGS:
        os.environ.pop(name, None)


def _args(**overrides) -> argparse.Namespace:
    values = dict(
        read_only=False,
        no_attempts=False,
        no_plan=False,
        no_generate=False,
        no_export_write=False,
        allow_generate=False,
        allow_out_dir=[],
        transport="stdio",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_read_only_disables_every_write_tool(monkeypatch) -> None:
    monkeypatch.setenv(capabilities.READ_ONLY, "1")
    caps = capabilities.current()
    assert caps.read_only is True
    for tool in ("review_attempt", "review_plan", "review_generate", "review_export_tree"):
        assert capabilities.disabled(tool), tool
    # The URL-only export and the read tools stay available.
    for tool in (
        "review_specs",
        "review_today",
        "review_hint",
        "review_weakness",
        "review_export_link",
        "review_book_figures",
    ):
        assert capabilities.disabled(tool) is None, tool


def test_individual_flag_only_disables_its_tool(monkeypatch) -> None:
    monkeypatch.setenv(capabilities.NO_GENERATE, "1")
    assert capabilities.disabled("review_generate")
    assert capabilities.disabled("review_attempt") is None
    assert capabilities.disabled("review_plan") is None
    assert capabilities.disabled("review_export_tree") is None


def test_http_transport_disables_generate_by_default(monkeypatch) -> None:
    _configure_capabilities(_args(transport="streamable-http"))
    assert capabilities.current().generate is False

    monkeypatch.delenv(capabilities.NO_GENERATE, raising=False)
    _configure_capabilities(_args(transport="streamable-http", allow_generate=True))
    assert capabilities.current().generate is True

    monkeypatch.delenv(capabilities.NO_GENERATE, raising=False)
    _configure_capabilities(_args(transport="stdio"))
    assert capabilities.current().generate is True


def test_loopback_detection() -> None:
    assert _is_loopback("127.0.0.1")
    assert _is_loopback("::1")
    assert _is_loopback("localhost")
    assert not _is_loopback("0.0.0.0")
    assert not _is_loopback("192.168.1.10")


def test_public_http_without_token_refuses_to_start(monkeypatch, capsys) -> None:
    monkeypatch.delenv(AUTH_TOKEN_ENV, raising=False)
    assert main(["--transport", "streamable-http", "--host", "0.0.0.0"]) == 2
    assert "refusing to serve" in capsys.readouterr().err


def test_read_only_today_does_not_create_a_session_or_views(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(tmp_path))
    service = ReviewService()
    spec_id = service.put_spec(build_fixture_spec())["spec_id"]

    def counts() -> tuple[int, int]:
        with sqlite3.connect(service.store.db_path) as conn:
            sessions = conn.execute("SELECT COUNT(*) FROM review_sessions").fetchone()[0]
            views = conn.execute("SELECT COUNT(*) FROM review_views").fetchone()[0]
        return int(sessions), int(views)

    assert counts() == (0, 0)
    result = service.today(spec_id, read_only=True)
    assert result["session_id"] == ""
    assert counts() == (0, 0)

    service.today(spec_id)
    sessions, views = counts()
    assert sessions == 1
    assert views > 0


def test_export_dir_must_stay_under_review_home(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(tmp_path))
    allowed, error = _resolve_export_dir(str(tmp_path / "exports" / "nested"))
    assert error is None and allowed is not None

    refused, error = _resolve_export_dir(str(tmp_path / "outside"))
    assert refused is None
    assert error is not None and "--allow-out-dir" in error


def test_disabled_tool_returns_a_soft_failure(monkeypatch) -> None:
    monkeypatch.setenv(capabilities.READ_ONLY, "1")

    attempt = asyncio.run(ReviewAttemptTool().execute(item_id="x", rating="good"))
    payload = json.loads(attempt.content)
    assert attempt.success is True
    assert "disabled" in payload["error"]
    assert "review_attempt is disabled" in payload["instruction"]

    generate = asyncio.run(ReviewGenerateTool().execute(kb="x", chapter="y"))
    payload = json.loads(generate.content)
    assert generate.success is True
    assert "disabled" in payload["error"]
