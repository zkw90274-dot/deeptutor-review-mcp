"""Review quality and shared-deployment contracts.

These pin the fixes made after the 2026-11-09 acceptance round:

* level 1 names the concept instead of reading a summary that may restate the
  answer; summaries are only used at level 2 when no definition exists;
* the correction payload never reuses another item's answer as a distractor;
* ``review_export_link`` adds an absolute URL when the deployment configures a
  public base URL, and stays relative otherwise;
* the HTTP transport can require a bearer token without changing the default
  local behavior.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from starlette.testclient import TestClient

from deeptutor.review.agent_tools import (
    PUBLIC_BASE_URL_ENV,
    PUBLIC_BASE_URL_FALLBACK_ENV,
    ReviewExportLinkTool,
    ReviewHintTool,
)
from deeptutor.review.correction import (
    CORRECTION_LABEL_CHARS,
    answer_text,
    build_correction,
)
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.mcp_server import AUTH_TOKEN_ENV, build_server, create_http_app
from deeptutor.review.models import ReviewSpec
from deeptutor.review.service import ReviewService

HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Host": "127.0.0.1:8767",
}
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": {"name": "shareable-surface", "version": "1"},
    },
}


def _hint(level: int) -> dict:
    result = asyncio.run(
        ReviewHintTool().execute(spec_id=SPEC_ID, item_id="item-resonance-choice", level=level)
    )
    assert result.success is True, result.content
    return json.loads(result.content)


def _spec_with_summary(summary: str) -> ReviewSpec:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    concept = payload["concepts"][0]
    concept["summary_one_line"] = summary
    concept["definition"] = ""
    concept["formulas"] = []
    return ReviewSpec.model_validate(payload)


# -- hint ladder ------------------------------------------------------------


def test_level_one_names_the_concept_not_the_summary(service: ReviewService) -> None:
    service.put_spec(build_fixture_spec())

    payload = _hint(1)

    assert payload["fallback"] is False
    assert payload["hint"] == "串联谐振"
    assert "谐振时 XL = XC" not in payload["hint"]


def test_level_two_still_uses_a_safe_summary_when_no_definition_exists(
    service: ReviewService,
) -> None:
    service.put_spec(_spec_with_summary("该概念用于描述回路的选频能力。"))

    payload = _hint(2)

    assert payload["fallback"] is False
    assert "该概念用于描述回路的选频能力。" in payload["hint"]


def test_level_two_still_blocks_a_summary_that_states_the_answer(
    service: ReviewService,
) -> None:
    service.put_spec(_spec_with_summary("这一问的结论正确的是B。"))

    payload = _hint(2)

    assert payload["fallback"] is True
    assert "正确的是B" not in payload["hint"]


# -- correction distractors -------------------------------------------------


def test_correction_never_reuses_another_items_answer(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)

    for item in spec.items:
        correction = build_correction(spec, item)
        texts = [option["text"] for option in correction["options"]]
        own = answer_text(item).strip()
        assert 2 <= len(texts) <= 3
        assert 0 <= correction["correct_index"] < len(texts)
        for other in spec.items:
            if other.id == item.id:
                continue
            other_answer = answer_text(other).strip()
            if other_answer and other_answer != own:
                assert other_answer not in texts


def test_correction_stays_deterministic(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    item = spec.items[0]

    first = build_correction(spec, item)
    second = build_correction(spec, item)

    assert first == second


def test_correction_skips_a_concept_name_that_contains_another_answer() -> None:
    spec = build_fixture_spec()
    other_answer = answer_text(spec.items[1]).strip()
    assert other_answer
    spec.concepts[1].name = f"{other_answer}（补充说明）"

    correction = build_correction(spec, spec.items[0])

    texts = [option["text"] for option in correction["options"]]
    assert all(other_answer not in text for text in texts)


def test_correction_clips_long_concept_names() -> None:
    spec = build_fixture_spec()
    spec.concepts[1].name = "很长的概念名" * 40

    correction = build_correction(spec, spec.items[0])

    for index, option in enumerate(correction["options"]):
        if index != correction["correct_index"]:
            assert len(option["text"]) <= CORRECTION_LABEL_CHARS


# -- export link ------------------------------------------------------------


def _export_link() -> dict:
    result = asyncio.run(ReviewExportLinkTool().execute(spec_id=SPEC_ID))
    assert result.success is True, result.content
    return json.loads(result.content)


def test_export_link_adds_absolute_url_from_the_environment(
    service: ReviewService, monkeypatch: pytest.MonkeyPatch
) -> None:
    service.put_spec(build_fixture_spec())
    monkeypatch.setenv(PUBLIC_BASE_URL_ENV, "https://study.example.com/")

    payload = _export_link()

    assert payload["url"].startswith("/api/review/")
    assert payload["absolute_url"] == f"https://study.example.com{payload['url']}"


def test_export_link_accepts_the_fallback_base_url_variable(
    service: ReviewService, monkeypatch: pytest.MonkeyPatch
) -> None:
    service.put_spec(build_fixture_spec())
    monkeypatch.delenv(PUBLIC_BASE_URL_ENV, raising=False)
    monkeypatch.setenv(PUBLIC_BASE_URL_FALLBACK_ENV, "http://127.0.0.1:3782")

    payload = _export_link()

    assert payload["absolute_url"].startswith("http://127.0.0.1:3782/api/review/")


def test_export_link_ignores_an_invalid_base_url(
    service: ReviewService, monkeypatch: pytest.MonkeyPatch
) -> None:
    service.put_spec(build_fixture_spec())
    monkeypatch.setenv(PUBLIC_BASE_URL_ENV, "ftp://study.example.com")

    payload = _export_link()

    assert "absolute_url" not in payload


def test_export_link_stays_relative_when_unconfigured(
    service: ReviewService, monkeypatch: pytest.MonkeyPatch
) -> None:
    service.put_spec(build_fixture_spec())
    monkeypatch.delenv(PUBLIC_BASE_URL_ENV, raising=False)
    monkeypatch.delenv(PUBLIC_BASE_URL_FALLBACK_ENV, raising=False)

    payload = _export_link()

    assert "absolute_url" not in payload


# -- HTTP bearer token ------------------------------------------------------


def test_http_auth_rejects_missing_and_wrong_tokens() -> None:
    app = create_http_app(build_server(), auth_token="s3cret")
    with TestClient(app) as client:
        missing = client.post("/mcp", json=INITIALIZE, headers=HEADERS)
        assert missing.status_code == 401
        assert missing.headers.get("www-authenticate") == "Bearer"

        wrong = client.post(
            "/mcp",
            json=INITIALIZE,
            headers={**HEADERS, "Authorization": "Bearer wrong"},
        )
        assert wrong.status_code == 401


def test_http_auth_accepts_the_configured_token() -> None:
    app = create_http_app(build_server(), auth_token="s3cret")
    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            json=INITIALIZE,
            headers={**HEADERS, "Authorization": "Bearer s3cret"},
        )
    assert response.status_code == 200, response.text


def test_http_auth_reads_the_token_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AUTH_TOKEN_ENV, "env-secret")
    app = create_http_app(build_server())
    with TestClient(app) as client:
        missing = client.post("/mcp", json=INITIALIZE, headers=HEADERS)
        assert missing.status_code == 401
        accepted = client.post(
            "/mcp",
            json=INITIALIZE,
            headers={**HEADERS, "Authorization": "Bearer env-secret"},
        )
    assert accepted.status_code == 200, accepted.text


def test_an_explicit_empty_token_disables_environment_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AUTH_TOKEN_ENV, "env-secret")
    app = create_http_app(build_server(), auth_token="")
    with TestClient(app) as client:
        response = client.post("/mcp", json=INITIALIZE, headers=HEADERS)
    assert response.status_code == 200, response.text
