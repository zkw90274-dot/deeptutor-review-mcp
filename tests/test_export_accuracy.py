"""Fact-accuracy checks for the two learner-facing surfaces of Review Studio.

The export page does not server-render items: ``templates/review.html`` carries
``window.__REVIEW__ = {{PAYLOAD}};`` and the browser builds every node from that
embedded JSON with ``textContent``. ``export._safe_json`` escapes ``<``, ``>``,
``&`` and U+2028/U+2029 as JSON escapes (``\\u003c`` ...), so parsing the
payload restores learner text byte-for-byte. These tests therefore assert
against the payload (the real transport) and, where the fixture is ASCII-safe,
against the raw page source as well.

The overview page (``/api/review/ui``) is server-rendered by
``api._overview_card``; it must agree with ``ReviewService`` facts for the
stored spec (counts, QA status, title, export link) and with the Today queue
for its due/new/selected numbers.

These tests are hermetic: stdlib + pytest only, no network, no LLM.
"""

from __future__ import annotations

import html as html_lib
import json
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from conftest import make_wide_spec
from fastapi.testclient import TestClient

from deeptutor.review import api as review_api
from deeptutor.review import service as service_module
from deeptutor.review.adapter import initial_state
from deeptutor.review.export import build_html, normalize_prose_math
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.models import QualityReport, ReviewSpec
from deeptutor.review.service import AttemptInput, ReviewService

NOW = 1_800_000_000.0
DAY = 86_400.0
TZ = ZoneInfo("Asia/Shanghai")
PAYLOAD_PREFIX = "window.__REVIEW__ = "


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _embedded_payload(page: str) -> dict:
    """Parse the JSON the template hands to the browser.

    The template embeds compact canonical JSON on a single line:
    ``window.__REVIEW__ = {...};``. Reading this instead of grepping the HTML
    is the only comparison that is exact for content containing ``<``/``&``
    (those arrive as ``\\u003c``/``\\u0026`` on the wire).
    """
    start = page.index(PAYLOAD_PREFIX) + len(PAYLOAD_PREFIX)
    end = page.index("\n", start)
    raw = page[start:end].strip()
    assert raw.endswith(";"), "embedded payload is not terminated on its line"
    return json.loads(raw[:-1])


def _payload_items(payload: dict) -> dict[str, dict]:
    """Flatten item blocks out of the section payload."""
    found: dict[str, dict] = {}
    for section in payload["sections"]:
        for block in section["blocks"]:
            if block.get("kind") == "item":
                item = block["item"]
                assert item["id"] not in found, f"duplicate item block {item['id']}"
                found[item["id"]] = item
    return found


def _expected_answer_display(item) -> str:
    """The learner-visible answer text promised by the export contract."""
    if item.kind == "single_choice" and item.answer.choice_index is not None:
        option = item.options[item.answer.choice_index]
        return f"{option.label}. {option.text}"
    if item.kind == "numeric" and item.answer.value is not None:
        if item.answer.tolerance:
            return f"{item.answer.value:g}（容差 ±{item.answer.tolerance:g}）"
        return f"{item.answer.value:g}"
    return item.answer.text


def _expected_qa_summary(quality: dict) -> str:
    warnings = len(quality.get("warnings") or [])
    if quality.get("passed"):
        return f"通过，{warnings} 条警告" if warnings else "通过，无警告"
    return f"未通过，{warnings} 条警告"


def _put(client: TestClient, spec) -> None:
    response = client.put(
        f"/api/review/specs/{spec.id}", json=spec.model_dump(mode="json")
    )
    assert response.status_code == 201, response.text


# ---------------------------------------------------------------------------
# 1. export payload carries every stored item verbatim
# ---------------------------------------------------------------------------


def test_export_payload_matches_stored_items_byte_for_byte(
    service: ReviewService,
) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)

    page = service.export(spec.id, now=NOW)
    payload = _embedded_payload(page)
    items = _payload_items(payload)

    assert set(items) == {item.id for item in spec.items}

    for item in spec.items:
        got = items[item.id]

        # Exact transport values: JSON decoding undoes the \uXXXX escaping, so
        # these must equal the stored strings with no whitespace/normalisation.
        assert got["stem"] == item.stem
        assert got["options"] == [
            {"label": option.label, "text": option.text} for option in item.options
        ]
        # The export wraps a bare-TeX answer in $...$ at render time; that is
        # the one documented exception to verbatim transport (storage is
        # untouched). Stems, options and explanations stay byte-for-byte.
        assert got["answer_display"] == normalize_prose_math(
            _expected_answer_display(item), pure_math_only=True
        )
        assert got["answer_index"] == item.answer.choice_index
        assert got["explanation"] == item.explanation
        assert got["kind"] == item.kind
        assert got["source_ids"] == item.source_ids
        assert got["concept_ids"] == item.concept_ids

        # The fixture is free of the five characters _safe_json rewrites, so
        # the raw page source must also contain the learner text unchanged.
        assert item.stem in page
        assert item.explanation in page
        for option in item.options:
            assert option.text in page

    # Byte identity of the transport itself: payload is compact canonical JSON.
    assert '"spec_id":"' + spec.id + '"' in page


# ---------------------------------------------------------------------------
# 2. overview cards agree with the service, including today's queue
# ---------------------------------------------------------------------------


def test_overview_cards_match_service_facts(
    client: TestClient,
    service: ReviewService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = build_fixture_spec()
    _put(client, spec)

    # Make one item genuinely due: a failed recall two days before NOW.
    service.attempt(
        spec.id,
        AttemptInput(item_id="item-resonance-choice", rating="again"),
        now=NOW - 2 * DAY,
    )

    # The page computes its plan at ``time.time()``. Pin the service clock so
    # the page and the direct Today call describe the same instant.
    monkeypatch.setattr(
        service_module, "time", SimpleNamespace(time=lambda: NOW)
    )

    response = client.get("/api/review/ui")
    assert response.status_code == 200
    body = response.text

    meta = service.spec_meta(spec.id)
    today = service.today(spec.id, now=NOW)

    # Exactly one card per stored spec (list_specs is the card source).
    assert len(service.list_specs()) == 1
    assert body.count('<article class="card">') == len(service.list_specs())

    # Identity, counts and QA status.
    assert meta["spec_id"] == SPEC_ID == spec.id
    assert f"<h2>{html_lib.escape(spec.title, quote=True)}</h2>" in body
    assert f"ID <b>{SPEC_ID}</b>" in body
    assert f"条目 <b>{meta['counts']['items']}</b>" in body
    assert f"概念 <b>{meta['counts']['concepts']}</b>" in body
    assert f"QA：<b class=\"{'ok' if meta['quality']['passed'] else 'bad'}\">" in body
    assert _expected_qa_summary(meta["quality"]) in body
    assert f'href="/api/review/{SPEC_ID}/export"' in body
    assert "导出离线复习页" in body

    # Today's numbers must equal what Today() actually returns for this spec.
    due_from_today = len([row for row in today["items"] if row["status"] == "due"])
    new_from_today = len([row for row in today["items"] if row["status"] == "new"])
    selected_from_today = len(today["items"])

    # Non-trivial: the failed attempt really produced a due card.
    assert due_from_today >= 1
    assert today["counts"]["due"] == due_from_today
    assert today["counts"]["new"] == new_from_today

    assert f"今日待复习：<b>{due_from_today}</b> 项" in body
    assert f"今日新内容：<b>{new_from_today}</b> 项" in body
    assert (
        f"今日队列：<b>{selected_from_today}</b> 项 · 预算 {today['budget']}" in body
    )


# ---------------------------------------------------------------------------
# 2b. capped backlog: the card reports the selected queue, not the whole pool
# ---------------------------------------------------------------------------


def test_overview_due_matches_selected_queue_when_backlog_exceeds_budget(
    client: TestClient,
    service: ReviewService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = make_wide_spec(due=20, new=0)
    _put(client, spec)
    for item in spec.items:
        service.attempt(
            spec.id,
            AttemptInput(item_id=item.id, rating="again"),
            now=NOW - 2 * DAY,
        )

    monkeypatch.setattr(
        service_module, "time", SimpleNamespace(time=lambda: NOW)
    )

    body = client.get("/api/review/ui").text
    today = service.today(spec.id, now=NOW)

    due_in_queue = len([row for row in today["items"] if row["status"] == "due"])
    # The backlog is wider than one session, so the two numbers legitimately
    # differ: counts.due is the pool, items are the bounded queue.
    assert today["counts"]["due"] == len(spec.items) == 20
    assert due_in_queue == len(today["items"]) == min(12, len(spec.items))
    assert due_in_queue < today["counts"]["due"]

    # The overview card promises today's queue, so it must match the items.
    assert f"今日待复习：<b>{due_in_queue}</b> 项" in body


# ---------------------------------------------------------------------------
# 3. export route: GET and POST are the same document
# ---------------------------------------------------------------------------


def test_export_get_and_post_produce_identical_bytes(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)

    export_path = f"/api/review/{spec.id}/export"
    get_response = client.get(export_path)
    post_response = client.post(export_path)

    assert get_response.status_code == 200
    assert post_response.status_code == 200
    assert get_response.headers["content-type"].startswith("text/html")
    assert post_response.headers["content-type"].startswith("text/html")
    assert (
        get_response.headers["content-disposition"]
        == post_response.headers["content-disposition"]
        == f'attachment; filename="review-{spec.id}.html"'
    )
    assert get_response.content == post_response.content


# ---------------------------------------------------------------------------
# 4. learner-controlled title cannot inject markup
# ---------------------------------------------------------------------------


def test_script_title_is_escaped_on_export_and_overview(
    service: ReviewService,
) -> None:
    hostile = "<script>alert(1)</script>"
    payload = build_fixture_spec().model_dump(mode="json")
    payload["title"] = hostile
    payload["content_hash"] = ""
    spec = ReviewSpec.model_validate(payload)

    # ---- export page ----
    # build_html is used directly: G6 correctly refuses to store or export a
    # spec with <script> in any string, so ingestion is not the subject here;
    # the renderer must still escape a title that reaches it.
    states = {concept.id: initial_state(concept, now=NOW) for concept in spec.concepts}
    page = build_html(spec, states, now=NOW, quality=QualityReport())

    assert f"<title>&lt;script&gt;alert(1)&lt;/script&gt;</title>" in page
    assert "<script>alert(1)" not in page
    # Inside the embedded JSON the same text is \u-escaped, not raw markup.
    assert '"title":"\\u003cscript\\u003ealert(1)\\u003c/script\\u003e"' in page
    assert _embedded_payload(page)["title"] == hostile

    # ---- overview page ----
    class _StubService:
        def list_specs(self):
            return [{"spec_id": SPEC_ID}]

        def spec_meta(self, spec_id):
            return {
                "spec_id": spec_id,
                "title": hostile,
                "quality": {"passed": True, "warnings": []},
                "counts": {"items": 1, "concepts": 1},
            }

        def session_plan(self, spec_id, *, mode="mixed"):
            return {"counts": {"due": 0, "new": 1}, "selected": 1, "budget": 12}

    overview = review_api._overview_page(_StubService())

    assert f"<h2>&lt;script&gt;alert(1)&lt;/script&gt;</h2>" in overview
    assert "<script>alert(1)" not in overview


# ---------------------------------------------------------------------------
# 5. export metadata identifies the stored spec
# ---------------------------------------------------------------------------


def test_export_metadata_matches_stored_spec(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)

    page = service.export(spec.id, include_snapshot=True, now=NOW)
    payload = _embedded_payload(page)
    stored = service.spec_meta(spec.id)

    assert payload["spec_id"] == stored["spec_id"] == spec.id
    assert payload["title"] == spec.title
    assert payload["schema_version"] == spec.schema_version
    assert payload["plan"]["timezone"] == spec.plan.timezone
    assert payload["quality"]["passed"] is True

    # Snapshot is the only place the export timestamp appears, and it must be
    # the instant the caller asked for, rendered in the plan timezone.
    assert payload["snapshot"] is not None
    assert payload["snapshot"]["generated_at"] == datetime.fromtimestamp(
        NOW, tz=TZ
    ).isoformat()

    # With the route's default (no snapshot) the document carries no clock.
    plain = _embedded_payload(service.export(spec.id, now=NOW))
    assert plain["snapshot"] is None
