from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from deeptutor.learning.assessment import AssessmentOutcome
from deeptutor.review import projection
from deeptutor.review.models import ReviewSpec
from deeptutor.review.projection import build_assessment_record, project_attempt
from deeptutor.review.service import AttemptInput, ReviewService


@pytest.fixture(autouse=True)
def _fake_linkage_preflight(monkeypatch: pytest.MonkeyPatch):
    """Most tests exercise behavior after a valid target was found."""
    monkeypatch.setattr(
        projection,
        "_linkage_preflight",
        lambda *_, **__: (True, "", "concept", 0.9, 1.0, False),
    )


def _attempt(
    service: ReviewService,
    spec: ReviewSpec,
    *,
    item_id: str = "item-resonance-choice",
    rating: str = "good",
    submission_id: str = "sub-1",
    answer: Any = 1,
    elapsed_ms: int = 1500,
) -> dict[str, Any]:
    service.put_spec(spec)
    return service.attempt(
        spec.id,
        AttemptInput(
            item_id=item_id,
            rating=rating,
            session_id="review-session",
            submission_id=submission_id,
            answer=answer,
            elapsed_ms=elapsed_ms,
        ),
    )


def _event(**overrides: Any) -> dict[str, Any]:
    event = {
        "event_id": "evt-1",
        "submission_id": "sub-1",
        "session_id": "review-session",
        "rating": "good",
        "quality": 1.0,
        "elapsed_ms": 1500,
        "ts": 1_800_000_000.0,
        "hints_used": 0,
        "confidence": None,
    }
    event.update(overrides)
    return event


def test_build_record_carries_hints_and_confidence(
    linked_spec: ReviewSpec,
) -> None:
    """The event row is authoritative: hinted evidence must reach core."""
    item = linked_spec.item_by_id("item-resonance-choice")
    concept = linked_spec.concept_by_id(item.concept_ids[0])
    record = build_assessment_record(
        linked_spec,
        item,
        concept,
        _event(rating="hard", hints_used=2, confidence=90.0),
        user_answer=1,
    )

    assert record is not None
    assert record.hints_used == 2
    assert record.confidence == 90.0


def test_build_record_maps_good_choice(linked_spec: ReviewSpec) -> None:
    item = linked_spec.item_by_id("item-resonance-choice")
    concept = linked_spec.concept_by_id(item.concept_ids[0])
    record = build_assessment_record(
        linked_spec, item, concept, _event(), user_answer=1
    )

    assert record is not None
    assert record.origin_type == "external_import"
    assert record.origin_ref == linked_spec.id
    assert record.assessment_type == "review"
    assert record.source == "import"
    assert record.result == "correct"
    assert record.is_correct is True
    assert record.quality == 1.0
    assert record.attempt_id == "review:evt-1"
    assert record.mastery_path_id == "fixture-mastery"
    assert record.knowledge_point_id == "kp-series-resonance"
    assert record.question_id == "item-resonance-choice"
    assert record.user_answer.startswith("B.")
    assert record.correct_answer.startswith("B.")
    assert record.response_time == 1.5
    assert record.session_id == ""


def test_build_record_maps_hard_numeric_to_correct(linked_spec: ReviewSpec) -> None:
    item = linked_spec.item_by_id("item-resonance-numeric")
    concept = linked_spec.concept_by_id(item.concept_ids[0])
    record = build_assessment_record(
        linked_spec,
        item,
        concept,
        _event(rating="hard", quality=0.55, event_id="evt-hard"),
        user_answer=1.5,
    )

    assert record is not None
    # "hard" is remembered-with-effort, not wrong. Core's Question Bank turns
    # partial into is_correct=False and a declined trend.
    assert record.result == "correct"
    assert record.is_correct is True
    assert record.quality == 0.55
    assert record.user_answer == "1.5"
    assert record.correct_answer == str(item.answer.value)


def test_build_record_maps_again_flashcard_to_incorrect(
    linked_spec: ReviewSpec,
) -> None:
    item = linked_spec.item_by_id("item-quality-flashcard")
    concept = linked_spec.concept_by_id(item.concept_ids[0])
    record = build_assessment_record(
        linked_spec,
        item,
        concept,
        _event(rating="again", quality=0.0, event_id="evt-again"),
        user_answer="不会",
    )

    assert record is not None
    assert record.result == "incorrect"
    assert record.is_correct is False
    assert record.quality == 0.0
    assert record.user_answer == "不会"
    assert record.correct_answer == item.answer.text


def test_build_record_is_none_without_linkage(fixture_spec: ReviewSpec) -> None:
    item = fixture_spec.item_by_id("item-resonance-choice")
    concept = fixture_spec.concept_by_id(item.concept_ids[0])
    assert build_assessment_record(fixture_spec, item, concept, _event()) is None


def test_projection_applied_and_due_date_consistent(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _attempt(service, linked_spec)
    captured: list[Any] = []

    async def fake_record(record):
        captured.append(record)
        return AssessmentOutcome(attempt_recorded=True)

    monkeypatch.setattr(projection, "record_assessment", fake_record)
    monkeypatch.setattr(
        projection,
        "_read_core_due",
        lambda *_: (result["next_due_ts"], 0.9),
    )

    projected = asyncio.run(
        project_attempt(
            service.store, linked_spec.id, result, user_answer=1
        )
    )

    assert [record.attempt_id for record in captured] == [
        f"review:{result['event_id']}"
    ]
    assert projected["status"] == "applied"
    assert projected["due_date_consistent"] is True
    assert projected["diagnostics"] == []


def test_projection_duplicate_does_not_add_evidence(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _attempt(service, linked_spec)

    async def fake_record(record):
        return AssessmentOutcome(attempt_recorded=False)

    monkeypatch.setattr(projection, "record_assessment", fake_record)
    monkeypatch.setattr(projection, "_read_core_due", lambda *_: (None, None))

    projected = asyncio.run(
        project_attempt(service.store, linked_spec.id, result, user_answer=1)
    )
    assert projected["status"] == "duplicate"
    assert projected["due_date_consistent"] is None


def test_projection_retry_heals_linked_evidence(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _attempt(service, linked_spec)

    async def fake_record(record):
        return AssessmentOutcome(
            attempt_recorded=False,
            diagnostics=["linked_retention_updated"],
        )

    monkeypatch.setattr(projection, "record_assessment", fake_record)
    monkeypatch.setattr(projection, "_read_core_due", lambda *_: (None, None))

    projected = asyncio.run(
        project_attempt(service.store, linked_spec.id, result, user_answer=1)
    )
    assert projected["status"] == "applied"


def test_projection_reports_invalid_linkage(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _attempt(service, linked_spec)

    async def fake_record(record):
        return AssessmentOutcome(
            attempt_recorded=True,
            diagnostics=["dropped_invalid_mastery_linkage"],
        )

    monkeypatch.setattr(projection, "record_assessment", fake_record)
    monkeypatch.setattr(projection, "_read_core_due", lambda *_: (None, None))

    projected = asyncio.run(
        project_attempt(service.store, linked_spec.id, result, user_answer=1)
    )
    assert projected["status"] == "invalid_linkage"
    assert "dropped_invalid_mastery_linkage" in projected["diagnostics"]


def test_projection_failure_does_not_raise(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _attempt(service, linked_spec)

    async def fake_record(record):
        raise RuntimeError("core unavailable")

    monkeypatch.setattr(projection, "record_assessment", fake_record)
    monkeypatch.setattr(projection, "_read_core_due", lambda *_: (None, None))

    projected = asyncio.run(
        project_attempt(service.store, linked_spec.id, result, user_answer=1)
    )
    assert projected["status"] == "failed"
    assert projected["error"] == "projection_error"


def test_projection_preflight_blocks_core_write(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _attempt(service, linked_spec)
    called = False

    async def fake_record(record):
        nonlocal called
        called = True
        return AssessmentOutcome(attempt_recorded=True)

    monkeypatch.setattr(projection, "record_assessment", fake_record)
    preflights = [
        (False, "path_not_found", "", None, None, False),
        (True, "", "concept", 0.9, 1.0, False),
    ]
    monkeypatch.setattr(
        projection,
        "_linkage_preflight",
        lambda *_, **__: preflights.pop(0),
    )
    monkeypatch.setattr(projection, "_read_core_due", lambda *_: (None, None))

    first = asyncio.run(
        project_attempt(service.store, linked_spec.id, result, user_answer=1)
    )
    assert first["status"] == "invalid_linkage"
    assert first["diagnostics"] == ["path_not_found"]
    assert called is False

    # Creating the path and retrying the same submission must heal the link.
    second = asyncio.run(
        project_attempt(service.store, linked_spec.id, result, user_answer=1)
    )
    assert second["status"] == "applied"
    assert called is True


def test_projection_diagnostic_timeout_is_reported(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _attempt(service, linked_spec)

    async def fake_record(record):
        return AssessmentOutcome(attempt_recorded=True)

    def slow_read(*args):
        time.sleep(0.2)
        return (1.0, 0.9)

    monkeypatch.setattr(projection, "record_assessment", fake_record)
    monkeypatch.setattr(projection, "_read_core_due", slow_read)
    monkeypatch.setattr(projection, "DIAGNOSTIC_TIMEOUT_SECONDS", 0.01)

    projected = asyncio.run(
        project_attempt(service.store, linked_spec.id, result, user_answer=1)
    )
    assert projected["status"] == "applied"
    assert projected["due_date_consistent"] is None
    assert "due_read_timeout" in projected["diagnostics"]


def test_projection_skipped_without_linkage(
    service: ReviewService,
    fixture_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _attempt(service, fixture_spec)
    called = False

    async def fake_record(record):
        nonlocal called
        called = True
        return AssessmentOutcome(attempt_recorded=True)

    monkeypatch.setattr(projection, "record_assessment", fake_record)
    projected = asyncio.run(
        project_attempt(service.store, fixture_spec.id, result, user_answer=1)
    )

    assert projected["status"] == "skipped"
    assert projected["diagnostics"] == ["no_mastery_linkage"]
    assert called is False


def test_projection_reports_retention_and_due_date_mismatch(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _attempt(service, linked_spec)

    async def fake_record(record):
        return AssessmentOutcome(attempt_recorded=True)

    monkeypatch.setattr(projection, "record_assessment", fake_record)
    monkeypatch.setattr(
        projection,
        "_linkage_preflight",
        lambda *_, **__: (True, "", "concept", 0.7, 1.0, False),
    )
    monkeypatch.setattr(
        projection,
        "_read_core_due",
        lambda *_: (result["next_due_ts"] + 120.0, 0.7),
    )

    projected = asyncio.run(
        project_attempt(service.store, linked_spec.id, result, user_answer=1)
    )
    assert projected["status"] == "applied"
    assert projected["due_date_consistent"] is False
    assert "due_date_mismatch" in projected["diagnostics"]
    assert "retention_mismatch" in projected["diagnostics"]
