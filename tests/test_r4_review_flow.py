from __future__ import annotations

import asyncio
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from deeptutor.review.mcp_server import TOOL_CLASSES
from deeptutor.review.models import ReviewSpec
from deeptutor.review.service import AttemptInput, QualityGateFailed
from deeptutor.review.storage import SCHEMA_VERSION, ReviewStore, ReviewStoreError


def test_review_correct_preserves_history(service, linked_spec) -> None:
    payload = linked_spec.model_dump(mode="json")
    payload["concepts"][1]["prerequisites"] = [payload["concepts"][0]["id"]]
    payload["content_hash"] = ""
    linked_spec = ReviewSpec.model_validate(payload)
    service.put_spec(linked_spec)
    item = linked_spec.items[0]
    attempt = service.attempt(linked_spec.id, AttemptInput(item_id=item.id, rating="good"))
    result = service.correct_item(linked_spec.id, item.id, {"stem": item.stem + " 请说明理由。"})
    assert result["item"]["id"] == item.id
    assert result["revision"] == 1
    assert result["quality"]["content_hash"] != linked_spec.content_hash
    assert service.store.event_by_id(linked_spec.id, attempt["event_id"])
    assert service.store.event_count(linked_spec.id) == 1
    assert service.store.get_spec(linked_spec.id).concepts == linked_spec.concepts


def test_review_correct_rejects_identity_change(service, fixture_spec) -> None:
    service.put_spec(fixture_spec)
    item = fixture_spec.items[0]
    with pytest.raises(ValueError):
        service.correct_item(fixture_spec.id, item.id, {"id": "replacement"})
    with pytest.raises(QualityGateFailed):
        service.correct_item(fixture_spec.id, item.id, {"stem": "<script>alert(1)</script>"})
    assert service.store.get_spec(fixture_spec.id).content_hash == fixture_spec.content_hash


def test_attempt_persists_correct_basis(service, fixture_spec) -> None:
    service.put_spec(fixture_spec)
    item = fixture_spec.items[0]
    request = AttemptInput(item_id=item.id, rating="again", answer=item.answer.choice_index, elapsed_ms=42, submission_id="graded")
    response = service.attempt(fixture_spec.id, request)
    persisted = service.store.event_by_id(fixture_spec.id, response["event_id"])
    assert persisted["correct"] == 1 and persisted["grader_basis"] == "choice_exact"
    assert persisted["elapsed_ms"] == 42 and response["rating"] == "again"
    assert service.attempt(fixture_spec.id, request)["correct"] is True


def test_metrics_objective_accuracy(service, fixture_spec) -> None:
    service.put_spec(fixture_spec)
    item = fixture_spec.items[0]
    for rating, answer in [("again", item.answer.choice_index), ("good", "wrong"), ("good", None)]:
        service.attempt(fixture_spec.id, AttemptInput(item_id=item.id, rating=rating, answer=answer))
    metrics = service.metrics(fixture_spec.id)
    assert metrics["objective_accuracy"] == {"correct": 1, "judged": 2, "accuracy": 0.5}
    assert metrics["learner_calibration"]["basis"] == "self_rating"


def _call(name, **kwargs):
    response = asyncio.run(TOOL_CLASSES[name]().execute(**kwargs))
    assert response.success, response.content
    return json.loads(response.content)


def test_metrics_tool(service, fixture_spec) -> None:
    service.put_spec(fixture_spec)
    before = service.store.db_path.read_bytes()
    result = _call("review_metrics", spec_id=fixture_spec.id)
    assert result["window_days"] == 30
    assert result["objective_accuracy"]["judged"] == 0
    assert service.store.db_path.read_bytes() == before


def test_session_end_tool(service, fixture_spec) -> None:
    service.put_spec(fixture_spec)
    spec_id = fixture_spec.id
    session_id = service.today(spec_id, include_draft=True)["session_id"]
    service.attempt(spec_id, AttemptInput(item_id=fixture_spec.items[0].id, rating="good", session_id=session_id))
    args = dict(spec_id=spec_id, session_id=session_id)
    assert _call("review_session", action="status", **args)["attempts"] == 1
    assert service.store.get_session(session_id)["ended_at"] is None
    assert _call("review_session", action="end", **args)["completed"] == 1
    assert _call("review_session", action="end", **args)["error"] == "session_already_ended"
    assert _call("review_metrics", spec_id=spec_id)["completion"]["completed"] == 1


def test_reveal_audit_and_limit(service, fixture_spec, monkeypatch) -> None:
    monkeypatch.setenv("DEEPTUTOR_REVIEW_MAX_REVEALS_PER_SESSION", "3")
    service.put_spec(fixture_spec)
    spec_id = fixture_spec.id
    queue = _call("review_today", spec_id=spec_id, include_draft=True)
    event = _call("review_attempt", spec_id=spec_id, session_id=queue["session_id"], item_id=queue["items"][0]["item_id"], rating="again")
    def reveal(_index):
        return _call("review_reveal", spec_id=spec_id, event_id=event["event_id"])

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(reveal, range(6)))
    assert sum("answer" in result for result in results) == 3
    assert sum(result.get("error") == "reveal_limit" for result in results) == 3
    with sqlite3.connect(service.store.db_path) as conn:
        audit = conn.execute("SELECT reveal_id, event_id FROM review_reveals ORDER BY reveal_id").fetchall()
    assert audit == [(number, event["event_id"]) for number in range(1, 4)]


def test_http_attempt_shape_unchanged(client, fixture_spec) -> None:
    spec_id = fixture_spec.id
    assert client.put(f"/api/review/specs/{spec_id}", json=fixture_spec.model_dump(mode="json")).status_code == 201
    assert client.post(f"/api/review/{spec_id}/today").json()["items"] == []
    queue = client.post(f"/api/review/{spec_id}/today?include_draft=true").json()
    item = fixture_spec.item_by_id(queue["items"][0]["item_id"])
    response = client.post(f"/api/review/{spec_id}/attempt", json={
        "item_id": item.id, "session_id": queue["session_id"], "rating": "again",
        "answer": item.answer.choice_index, "elapsed_ms": 12,
    })
    assert response.status_code == 200
    body = response.json()
    assert {"answer", "explanation", "study_aids", "correction", "correct"} <= body.keys()
    assert "grader_basis" not in body and "reveal_token" not in body


def test_schema_v6_migration_idempotent(service, fixture_spec) -> None:
    service.store.put_spec(fixture_spec)
    with sqlite3.connect(service.store.db_path) as conn:
        conn.execute("DELETE FROM review_schema_migrations")
        conn.execute("INSERT INTO review_schema_migrations VALUES (5, 0)")
        conn.execute("DROP TABLE review_quality")
    migrated = ReviewStore(service.store.home)
    assert migrated.quality_record(fixture_spec.id)["state"] == "legacy"
    before = migrated.db_path.read_bytes()
    ReviewStore(service.store.home)
    assert migrated.db_path.read_bytes() == before


def test_future_schema_rejected(store) -> None:
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("INSERT INTO review_schema_migrations VALUES (?, 0)", (SCHEMA_VERSION + 1,))
    before = store.db_path.read_bytes()
    with pytest.raises(ReviewStoreError, match="refusing to downgrade"):
        ReviewStore(store.home)
    assert store.db_path.read_bytes() == before
