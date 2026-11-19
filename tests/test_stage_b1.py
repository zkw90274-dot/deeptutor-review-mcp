from __future__ import annotations

import asyncio
import json
import sqlite3

from deeptutor.learning.assessment import AssessmentOutcome
from deeptutor.review import projection
from deeptutor.review.mcp_server import TOOL_CLASSES
from deeptutor.review.service import AttemptInput
from deeptutor.review.storage import ReviewStore


def _tool(name, **kwargs):
    response = asyncio.run(TOOL_CLASSES[name]().execute(**kwargs))
    assert response.success, response.content
    return json.loads(response.content)


def _bind(spec, concept, **overrides):
    params = dict(
        action="bind", spec_id=spec.id, concept_id=concept.id,
        mastery_path_id="overlay-path", knowledge_point_id="overlay-kp",
    )
    return _tool("review_link", **{**params, **overrides})


def _allow_targets(monkeypatch):
    monkeypatch.setattr(
        projection, "_linkage_preflight", lambda *args, **kwargs: (True, "", "concept", 0.9, 1.0, True)
    )


def test_link_overlay_and_spec_fallback(service, linked_spec, monkeypatch):
    _allow_targets(monkeypatch)
    service.put_spec(linked_spec)
    concept = linked_spec.concepts[0]
    assert _bind(linked_spec, concept)["links"][0]["mastery_path_id"] == "overlay-path"
    listing = _tool("review_link", action="list", spec_id=linked_spec.id)
    assert listing["links"][0]["knowledge_point_id"] == "overlay-kp"
    assert listing["links"][1]["mastery_path_id"] == linked_spec.mastery_path_id
    specs = _tool("review_specs")["specs"]
    assert specs[0]["linked_concepts"] == specs[0]["total_concepts"] == 2
    unbound = _tool("review_link", action="unbind", spec_id=linked_spec.id, concept_id=concept.id)
    assert unbound["links"][0]["knowledge_point_id"] == concept.knowledge_point_id
    assert service.store.get_spec(linked_spec.id).content_hash == linked_spec.content_hash


def test_failed_projection_retries_with_stable_event_and_target(service, fixture_spec, monkeypatch):
    _allow_targets(monkeypatch)
    service.put_spec(fixture_spec)
    item = fixture_spec.items[0]
    concept = fixture_spec.concept_by_id(item.concept_ids[0])
    _bind(fixture_spec, concept)
    event = service.attempt(fixture_spec.id, AttemptInput(item_id=item.id, rating="good"))
    calls = []

    async def record(payload):
        calls.append(payload)
        if len(calls) == 1:
            raise RuntimeError("transient failure")
        return AssessmentOutcome(attempt_recorded=True)

    monkeypatch.setattr(projection, "record_assessment", record)
    result = asyncio.run(projection.project_attempt(service.store, fixture_spec.id, event))
    assert result["status"] == "failed"
    assert service.store.projection_job(event["event_id"])["status"] == "pending"
    summary = _tool("review_link", action="backfill", spec_id=fixture_spec.id)
    assert summary["results"][0]["status"] == "applied"
    assert len(calls) == 2
    assert calls[0].attempt_id == calls[1].attempt_id == f"review:{event['event_id']}"
    assert all(call.mastery_path_id == "overlay-path" for call in calls)
    assert service.store.event_count(fixture_spec.id) == 1


def test_invalid_mapping_leaves_overlay_empty(service, fixture_spec, monkeypatch):
    service.put_spec(fixture_spec)
    concept = fixture_spec.concepts[0]
    monkeypatch.setenv("DEEPTUTOR_REVIEW_ALLOWED_MASTERY_PATHS", "some-other-path")
    assert _bind(fixture_spec, concept)["error"] == "linkage_not_allowed"
    assert _bind(fixture_spec, concept, confidence=True)["error"] == "invalid_link_request"
    assert _bind(fixture_spec, concept, concept_id="missing")["error"] == "invalid_link_request"
    assert service.store.links_for_spec(fixture_spec.id) == {}


def test_retry_limit_and_remapping(service, fixture_spec, monkeypatch):
    _allow_targets(monkeypatch)
    service.put_spec(fixture_spec)
    item = fixture_spec.items[0]
    concept = fixture_spec.concept_by_id(item.concept_ids[0])
    _bind(fixture_spec, concept)
    call_count = 0

    async def unavailable(record):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("temporary core failure")

    monkeypatch.setattr(projection, "record_assessment", unavailable)
    event = service.attempt(fixture_spec.id, AttemptInput(item_id=item.id, rating="good"))
    asyncio.run(projection.project_attempt(service.store, fixture_spec.id, event))
    for _ in range(3):
        assert _tool("review_link", action="backfill", spec_id=fixture_spec.id)["results"][0]["status"] == "failed"
    assert _tool("review_link", action="backfill", spec_id=fixture_spec.id)["results"] == []
    assert service.store.projection_job(event["event_id"])["attempts"] == 3
    assert call_count == 4

    second = service.attempt(fixture_spec.id, AttemptInput(item_id=item.id, rating="good"))
    asyncio.run(projection.project_attempt(service.store, fixture_spec.id, second))
    _bind(fixture_spec, concept, knowledge_point_id="new-kp")
    retry = _tool("review_link", action="backfill", spec_id=fixture_spec.id)["results"][0]
    assert retry["diagnostics"] == ["queued_link_changed"]
    assert call_count == 5
    assert _tool("review_link", action="backfill", spec_id=fixture_spec.id)["results"] == []


def test_concurrent_backfill_claims_once(service, fixture_spec, monkeypatch):
    _allow_targets(monkeypatch)
    service.put_spec(fixture_spec)
    item = fixture_spec.items[0]
    _bind(fixture_spec, fixture_spec.concept_by_id(item.concept_ids[0]))
    event = service.attempt(fixture_spec.id, AttemptInput(item_id=item.id, rating="good"))
    calls = []

    async def record(payload):
        calls.append(payload.attempt_id)
        if len(calls) == 1:
            raise RuntimeError("initial failure")
        return AssessmentOutcome(attempt_recorded=True)

    async def scenario():
        await projection.project_attempt(service.store, fixture_spec.id, event)
        return await asyncio.gather(*[
            projection.backfill_projections(service.store, fixture_spec.id) for _ in range(6)
        ])

    monkeypatch.setattr(projection, "record_assessment", record)
    results = asyncio.run(scenario())
    assert sum(row["status"] == "applied" for batch in results for row in batch) == 1
    assert len(calls) == 2


def test_v6_to_v7_preserves_spec_and_event(service, fixture_spec):
    service.put_spec(fixture_spec)
    service.attempt(fixture_spec.id, AttemptInput(item_id=fixture_spec.items[0].id, rating="good"))
    with sqlite3.connect(service.store.db_path) as conn:
        conn.executescript("DROP TABLE review_links; DROP TABLE review_projection_queue; DROP TABLE review_artifacts; "
                           "UPDATE review_schema_migrations SET version = 6 WHERE version = 9;")
    migrated = ReviewStore(service.store.home)
    assert migrated.links_for_spec(fixture_spec.id) == {}
    assert migrated.pending_projections(fixture_spec.id, limit=20) == []
    assert migrated.get_spec(fixture_spec.id).content_hash == fixture_spec.content_hash
    assert migrated.event_count(fixture_spec.id) == 1
    before = migrated.db_path.read_bytes()
    ReviewStore(service.store.home)
    assert migrated.db_path.read_bytes() == before


def test_read_only_list_and_write_guard(service, fixture_spec, monkeypatch):
    service.put_spec(fixture_spec)
    monkeypatch.setenv("DEEPTUTOR_REVIEW_READ_ONLY", "1")
    before = service.store.db_path.read_bytes()
    assert _tool("review_link", action="list", spec_id=fixture_spec.id)["links"] == []
    assert _tool("review_link", action="backfill", spec_id=fixture_spec.id)["error"] == "link_write_disabled"
    assert _bind(fixture_spec, fixture_spec.concepts[0])["error"] == "link_write_disabled"
    assert service.store.db_path.read_bytes() == before
