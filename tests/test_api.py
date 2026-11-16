from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from deeptutor.learning.assessment import AssessmentOutcome
from deeptutor.review import api as review_api
from deeptutor.review import projection
from deeptutor.review.fixture import build_fixture_spec
from deeptutor.review.storage import ReviewStore


def _put(client: TestClient, spec):
    payload = spec.model_dump(mode="json")
    return client.put(f"/api/review/specs/{spec.id}", json=payload)


def test_manifest_mounts_review_routes(client: TestClient) -> None:
    from deeptutor.modules.catalog import module_enabled

    assert module_enabled("review") is True
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/review/specs/{spec_id}" in paths
    assert "/api/review/{spec_id}/today" in paths
    assert "/api/review/{spec_id}/weakness" in paths
    assert "/api/review/{spec_id}/session" in paths
    assert "/api/review/{spec_id}/metrics" in paths
    assert "/api/review/{spec_id}/attempt" in paths
    assert "/api/review/{spec_id}/export" in paths


def test_oversized_request_is_rejected_before_parsing(client: TestClient) -> None:
    spec = build_fixture_spec()
    response = client.put(
        f"/api/review/specs/{spec.id}",
        content=spec.model_dump_json(),
        headers={
            "content-type": "application/json",
            "content-length": "2000001",
        },
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "payload_too_large"


def test_chunked_oversized_request_is_rejected(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(review_api, "MAX_REQUEST_BYTES", 100)

    def chunks():
        yield b'{"id":"chunked","padding":"' + b"a" * 200 + b'"}'

    response = client.put(
        "/api/review/specs/chunked",
        content=chunks(),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "payload_too_large"


def test_put_is_idempotent(client: TestClient) -> None:
    spec = build_fixture_spec()
    first = _put(client, spec)
    second = _put(client, spec)
    assert first.status_code == 201
    assert first.json()["created"] is True
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert first.json()["content_hash"] == second.json()["content_hash"]


def test_put_rejects_content_rewrite_after_history(client: TestClient) -> None:
    spec = build_fixture_spec()
    assert _put(client, spec).status_code == 201
    today = client.get(f"/api/review/{spec.id}/today").json()
    attempt = client.post(
        f"/api/review/{spec.id}/attempt",
        json={
            "item_id": "item-resonance-choice",
            "rating": "good",
            "session_id": today["session_id"],
            "submission_id": "rewrite-sub-1",
            "answer": 1,
        },
    )
    assert attempt.status_code == 200

    changed = spec.model_dump(mode="json")
    changed["title"] = "改标题"
    changed["content_hash"] = ""
    rejected = client.put(f"/api/review/specs/{spec.id}", json=changed)
    assert rejected.status_code == 409
    detail = rejected.json()["detail"]
    assert detail["code"] == "spec_history_conflict"
    assert detail["counts"]["events"] == 1

    forced = client.put(
        f"/api/review/specs/{spec.id}?allow_history_rewrite=true", json=changed
    )
    assert forced.status_code == 200
    assert forced.json()["created"] is False


def test_today_returns_queue(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    response = client.get(f"/api/review/{spec.id}/today")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    assert len(body["items"]) == 3
    assert body["tomorrow"]["date"]
    assert body["counts"]["new"] == 3


def test_attempt_good_and_duplicate(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    today = client.get(f"/api/review/{spec.id}/today").json()
    session_id = today["session_id"]
    payload = {
        "item_id": "item-resonance-choice",
        "rating": "good",
        "session_id": session_id,
        "submission_id": "api-sub-1",
        "answer": 1,
    }
    first = client.post(f"/api/review/{spec.id}/attempt", json=payload)
    assert first.status_code == 200
    assert first.json()["created"] is True
    assert first.json()["correct"] is True
    assert first.json()["scheduled_interval_days"] == 6.4164
    assert first.json()["event_id"]
    assert first.json()["mastery_projection"]["status"] == "skipped"
    second = client.post(f"/api/review/{spec.id}/attempt", json=payload)
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert second.json()["next_due_ts"] == first.json()["next_due_ts"]


def test_retry_without_session_is_idempotent(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    body = {
        "item_id": "item-resonance-choice",
        "rating": "good",
        "submission_id": "no-session-retry",
        "answer": 1,
    }
    first = client.post(f"/api/review/{spec.id}/attempt", json=body)
    second = client.post(f"/api/review/{spec.id}/attempt", json=body)
    assert first.status_code == 200 and first.json()["created"] is True
    assert second.status_code == 200 and second.json()["created"] is False
    assert second.json()["session_id"] == first.json()["session_id"]
    home = Path(os.environ["DEEPTUTOR_REVIEW_HOME"])
    store = ReviewStore(home)
    assert store.event_count(spec.id) == 1
    conn = sqlite3.connect(store.db_path)
    try:
        sessions = conn.execute("SELECT COUNT(*) FROM review_sessions").fetchone()[0]
    finally:
        conn.close()
    assert sessions == 1


def test_submission_conflict_returns_409(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    today = client.get(f"/api/review/{spec.id}/today").json()
    session_id = today["session_id"]
    base = {
        "item_id": "item-resonance-choice",
        "session_id": session_id,
        "submission_id": "api-conflict",
    }
    assert client.post(
        f"/api/review/{spec.id}/attempt", json={**base, "rating": "good"}
    ).status_code == 200
    conflict = client.post(
        f"/api/review/{spec.id}/attempt", json={**base, "rating": "again"}
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "submission_conflict"


def test_attempt_projects_linked_mastery(
    client: TestClient,
    linked_spec,
    monkeypatch,
) -> None:
    captured = []

    async def fake_record(record):
        captured.append(record)
        return AssessmentOutcome(attempt_recorded=True)

    monkeypatch.setattr(projection, "record_assessment", fake_record)
    monkeypatch.setattr(
        projection,
        "_linkage_preflight",
        lambda *_, **__: (True, "", "concept", 0.9, 1.0, False),
    )
    monkeypatch.setattr(projection, "_read_core_due", lambda *_: (None, None))

    _put(client, linked_spec)
    today = client.get(f"/api/review/{linked_spec.id}/today").json()
    response = client.post(
        f"/api/review/{linked_spec.id}/attempt",
        json={
            "item_id": "item-resonance-choice",
            "rating": "good",
            "session_id": today["session_id"],
            "submission_id": "api-linked-1",
            "answer": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mastery_projection"]["status"] == "applied"
    assert [record.knowledge_point_id for record in captured] == [
        "kp-series-resonance"
    ]
    assert captured[0].attempt_id == f"review:{body['event_id']}"


def test_attempt_survives_projection_boundary_failure(
    client: TestClient,
    linked_spec,
    monkeypatch,
) -> None:
    async def boom(*args, **kwargs):
        raise RuntimeError("projection boundary")

    monkeypatch.setattr(projection, "_project_attempt", boom)

    _put(client, linked_spec)
    response = client.post(
        f"/api/review/{linked_spec.id}/attempt",
        json={
            "item_id": "item-resonance-choice",
            "rating": "good",
            "submission_id": "api-boundary-failure",
            "answer": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["created"] is True
    assert response.json()["mastery_projection"]["status"] == "failed"


def test_complete_session_uses_events(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    today = client.get(f"/api/review/{spec.id}/today").json()
    session_id = today["session_id"]
    client.post(
        f"/api/review/{spec.id}/attempt",
        json={
            "item_id": "item-resonance-choice",
            "rating": "good",
            "session_id": session_id,
            "submission_id": "api-complete-1",
            "answer": 1,
        },
    )
    response = client.post(
        f"/api/review/{spec.id}/session/complete",
        json={"session_id": session_id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["completed"] == 1
    assert body["correct"] == 1
    assert body["accuracy"] == 1.0
    assert body["tomorrow"]["date"]


def test_complete_missing_session_is_404(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    response = client.post(
        f"/api/review/{spec.id}/session/complete", json={"session_id": "missing"}
    )
    assert response.status_code == 404


def test_export_is_offline_single_file(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    response = client.post(f"/api/review/{spec.id}/export")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.content.decode("utf-8")
    assert "<!doctype html>" in body.lower()
    assert "串联谐振发生时" in body
    assert "还是不会" in body
    assert len(response.content) <= 8_000_000
    assert "MathJax" in body and "mjx-container" in body
    # Scope: only this file. No external resource loads; MathJax attribution
    # links live in its disabled help menu and are not dependencies.
    assert 'src="http' not in body and "url(http" not in body
    assert "<link" not in body
    assert 'src="//' not in body


def test_unknown_spec_is_404(client: TestClient) -> None:
    response = client.get("/api/review/does-not-exist/today")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "not_found"


def test_unknown_item_is_404(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    today = client.get(f"/api/review/{spec.id}/today").json()
    response = client.post(
        f"/api/review/{spec.id}/attempt",
        json={
            "item_id": "missing",
            "rating": "good",
            "session_id": today["session_id"],
        },
    )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "item_not_found"


def test_invalid_rating_is_422(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    today = client.get(f"/api/review/{spec.id}/today").json()
    response = client.post(
        f"/api/review/{spec.id}/attempt",
        json={
            "item_id": "item-resonance-choice",
            "rating": "nope",
            "session_id": today["session_id"],
        },
    )
    assert response.status_code == 422


def test_extra_attempt_field_is_422(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    response = client.post(
        f"/api/review/{spec.id}/attempt",
        json={
            "item_id": "item-resonance-choice",
            "rating": "good",
            "session_id": "x",
            "unexpected": True,
        },
    )
    assert response.status_code == 422


def test_events_survive_a_new_store_instance(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    today = client.get(f"/api/review/{spec.id}/today").json()
    client.post(
        f"/api/review/{spec.id}/attempt",
        json={
            "item_id": "item-resonance-choice",
            "rating": "good",
            "session_id": today["session_id"],
            "submission_id": "api-persist",
        },
    )
    home = Path(os.environ["DEEPTUTOR_REVIEW_HOME"])
    reopened = ReviewStore(home)
    assert reopened.event_count(spec.id) == 1
    # A fresh process would import the same package; this proves the data is
    # on disk, not in a process-local cache.
    assert reopened.get_spec(spec.id).content_hash == spec.content_hash


def test_weakness_endpoint_returns_ranked_rows(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    response = client.get(f"/api/review/{spec.id}/weakness?limit=2")
    assert response.status_code == 200
    body = response.json()
    assert body["spec_id"] == spec.id
    assert body["limit"] == 2
    assert len(body["weak_concepts"]) <= 2
    scores = [row["weakness"] for row in body["weak_concepts"]]
    assert scores == sorted(scores, reverse=True)
    assert all(row["item_ids"] for row in body["weak_concepts"])


def test_session_endpoint_is_read_only(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    response = client.get(f"/api/review/{spec.id}/session?mode=mixed&limit=2")
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "mixed"
    assert body["selected"] <= 2
    assert "items" in body


def test_metrics_endpoint_is_read_only_and_windowed(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    today = client.get(f"/api/review/{spec.id}/today").json()
    client.post(
        f"/api/review/{spec.id}/attempt",
        json={
            "item_id": "item-resonance-choice",
            "rating": "good",
            "session_id": today["session_id"],
            "submission_id": "api-metrics",
        },
    )

    response = client.get(f"/api/review/{spec.id}/metrics?days=7")

    assert response.status_code == 200
    body = response.json()
    assert body["spec_id"] == spec.id
    assert body["window_days"] == 7
    assert len(body["retention"]) == 10
    assert body["completion"]["sessions"] >= 1
    assert body["completion"]["attempts"] >= 1


def test_metrics_rejects_a_non_positive_window(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    assert client.get(f"/api/review/{spec.id}/metrics?days=0").status_code == 422


def test_metrics_scopes_completion_to_the_window(service: ReviewService) -> None:
    """An attempt from outside the window must not inflate the numerator.

    The session started before the window and is still open, but its only
    attempt is outside the window. It must not count as a sitting in that
    window: a browsing-only session would otherwise depress the completion
    rate even though the learner never answered there.
    """
    from deeptutor.review import service as review_service

    start = 1_800_000_000.0 - 10 * 86400.0
    spec = build_fixture_spec()
    service.put_spec(spec)
    session = service.today(spec.id, session_id="long-session", now=start)
    service.attempt(
        spec.id,
        review_service.AttemptInput(
            item_id="item-resonance-choice",
            rating="good",
            session_id=session["session_id"],
            submission_id="outside-window",
            elapsed_ms=0,
        ),
        now=start + 60.0,
    )

    # A one-day window that contains the attempt sees it.
    inside = service.metrics(spec.id, days=1, now=start + 3600.0)
    assert inside["completion"]["attempts"] >= 1

    # Same still-open session, but the window begins after the attempt: it has
    # no answered work here, so it is not in the denominator either.
    later = service.metrics(spec.id, days=1, now=start + 5 * 86400.0)
    assert later["completion"]["attempts"] == 0
    assert later["completion"]["sessions"] == 0
    assert later["completion"]["rate"] is None


def test_metrics_reports_truncation(client: TestClient, monkeypatch) -> None:
    from deeptutor.review import service as review_service

    spec = build_fixture_spec()
    _put(client, spec)
    today = client.get(f"/api/review/{spec.id}/today").json()
    for index, item_id in enumerate(
        ("item-resonance-choice", "item-resonance-numeric")
    ):
        client.post(
            f"/api/review/{spec.id}/attempt",
            json={
                "item_id": item_id,
                "rating": "good",
                "session_id": today["session_id"],
                "submission_id": f"api-truncation-{index}",
            },
        )
    # Two events exist; a cap of one must set the flag rather than silently
    # reporting a one-row sample as if it were the whole window.
    monkeypatch.setattr(review_service, "MAX_METRIC_EVENTS", 1)

    body = client.get(f"/api/review/{spec.id}/metrics?days=7").json()

    assert body["truncated"] is True
    assert body["sampled_events"] == 1


def test_attempt_again_returns_correction_payload(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    today = client.get(f"/api/review/{spec.id}/today").json()
    response = client.post(
        f"/api/review/{spec.id}/attempt",
        json={
            "item_id": "item-resonance-choice",
            "rating": "again",
            "session_id": today["session_id"],
            "submission_id": "api-correction",
        },
    )
    assert response.status_code == 200
    correction = response.json()["correction"]
    assert 2 <= len(correction["options"]) <= 3
    assert correction["retry_after_items"] == 5


def test_core_overlay_failure_keeps_attempt_200(
    client: TestClient, monkeypatch
) -> None:
    """A display read of core state must never turn a saved answer into a 500."""
    from deeptutor.review import service as review_service

    spec = build_fixture_spec()
    _put(client, spec)
    today = client.get(f"/api/review/{spec.id}/today").json()

    def boom(*_args, **_kwargs):
        raise RuntimeError("core snapshot corrupted")

    monkeypatch.setattr(review_service, "states_for_spec", boom)
    response = client.post(
        f"/api/review/{spec.id}/attempt",
        json={
            "item_id": "item-resonance-choice",
            "rating": "good",
            "session_id": today["session_id"],
            "submission_id": "core-overlay-failure",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["state_source"] == "review"
    assert body["created"] is True


def test_replay_preserves_retry_flag(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    today = client.get(f"/api/review/{spec.id}/today").json()
    session_id = today["session_id"]
    first = client.post(
        f"/api/review/{spec.id}/attempt",
        json={
            "item_id": "item-resonance-choice",
            "rating": "again",
            "session_id": session_id,
            "submission_id": "replay-again",
        },
    )
    assert first.status_code == 200
    retry = client.post(
        f"/api/review/{spec.id}/attempt",
        json={
            "item_id": "item-resonance-choice",
            "rating": "good",
            "session_id": session_id,
            "submission_id": "replay-good",
        },
    ).json()
    assert retry["retry"] is True
    replay = client.post(
        f"/api/review/{spec.id}/attempt",
        json={
            "item_id": "item-resonance-choice",
            "rating": "good",
            "session_id": session_id,
            "submission_id": "replay-good",
        },
    ).json()
    assert replay["created"] is False
    assert replay["retry"] is True
    assert replay["rating"] == retry["rating"]


def test_completed_session_rejects_new_answers(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    today = client.get(f"/api/review/{spec.id}/today").json()
    session_id = today["session_id"]
    client.post(
        f"/api/review/{spec.id}/attempt",
        json={
            "item_id": "item-resonance-choice",
            "rating": "good",
            "session_id": session_id,
            "submission_id": "finished-1",
        },
    )
    assert (
        client.post(
            f"/api/review/{spec.id}/session/complete",
            json={"session_id": session_id},
        ).status_code
        == 200
    )
    late = client.post(
        f"/api/review/{spec.id}/attempt",
        json={
            "item_id": "item-resonance-numeric",
            "rating": "good",
            "session_id": session_id,
            "submission_id": "finished-2",
        },
    )
    assert late.status_code == 409


def test_complete_counts_items_not_attempts(client: TestClient) -> None:
    spec = build_fixture_spec()
    _put(client, spec)
    today = client.get(f"/api/review/{spec.id}/today").json()
    session_id = today["session_id"]
    client.post(
        f"/api/review/{spec.id}/attempt",
        json={
            "item_id": "item-resonance-choice",
            "rating": "again",
            "session_id": session_id,
            "submission_id": "count-items-1",
        },
    )
    client.post(
        f"/api/review/{spec.id}/attempt",
        json={
            "item_id": "item-resonance-choice",
            "rating": "good",
            "session_id": session_id,
            "submission_id": "count-items-2",
        },
    )
    body = client.post(
        f"/api/review/{spec.id}/session/complete",
        json={"session_id": session_id},
    ).json()
    assert body["completed"] == 1
    assert body["attempts"] == 2
    assert body["retests"] == 1
    assert body["correct"] == 1
