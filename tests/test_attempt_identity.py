"""Attempt identity across the schema-v3 boundary (Phase 34).

Events written before ``hints_used``/``confidence`` existed hashed the
payload without those keys. Replaying such a submission must stay idempotent
instead of reporting a conflict, which is why the new keys enter the hash
only when they carry a non-default value.
"""

from __future__ import annotations

import time

import pytest

from deeptutor.review.adapter import (
    RATING_MAP,
    apply_rating,
    initial_state,
    scheduler_for,
)
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.service import AttemptInput, ReviewService, _payload_hash
from deeptutor.review.storage import StateTransition, SubmissionConflict

ITEM_ID = "item-resonance-choice"
SESSION_ID = "sess-legacy-identity"


def _append_legacy_good_event(service: ReviewService, submission_id: str) -> None:
    """Write an event exactly as schema v2 did: hash without the new keys."""
    spec = service.store.get_spec(SPEC_ID)
    item = spec.item_by_id(ITEM_ID)
    concept = spec.concept_by_id(item.concept_ids[0])
    scheduler = scheduler_for(spec)
    moment = time.time()
    legacy_hash = _payload_hash(
        {
            "spec_id": spec.id,
            "item_id": item.id,
            "rating": "good",
            "answer": None,
        }
    )

    def transition(_state_json: str | None) -> StateTransition:
        state = initial_state(concept, now=moment, scheduler=scheduler)
        state, predicted, interval_days = apply_rating(
            state,
            concept,
            "good",
            now=moment,
            item_id=item.id,
            session_id=SESSION_ID,
            scheduler=scheduler,
        )
        return StateTransition(state, predicted, interval_days)

    service.store.append_event(
        submission_id=submission_id,
        spec_id=spec.id,
        item_id=item.id,
        concept_key=concept.key,
        session_id=SESSION_ID,
        ts=moment,
        rating="good",
        result=RATING_MAP["good"][0],
        quality=RATING_MAP["good"][1],
        elapsed_ms=0,
        payload_hash=legacy_hash,
        app_version="0.0.0-legacy",
        transition=transition,
    )


def test_legacy_submission_replays_instead_of_conflicting(
    service: ReviewService,
) -> None:
    service.put_spec(build_fixture_spec())
    _append_legacy_good_event(service, "legacy-identity-1")

    result = service.attempt(
        SPEC_ID,
        AttemptInput(
            item_id=ITEM_ID,
            rating="good",
            session_id=SESSION_ID,
            submission_id="legacy-identity-1",
        ),
    )

    assert result["created"] is False
    assert result["event_id"]
    assert result["rating"] == "good"
    # Migrated rows surface the new columns with their defaults.
    assert result["hints_used"] == 0
    assert result["confidence"] is None


def test_new_keys_are_identity_only_when_present(service: ReviewService) -> None:
    service.put_spec(build_fixture_spec())
    first = service.attempt(
        SPEC_ID,
        AttemptInput(
            item_id=ITEM_ID,
            rating="good",
            session_id=SESSION_ID,
            submission_id="v3-identity-1",
        ),
    )
    assert first["created"] is True
    assert first["hints_used"] == 0
    assert first["confidence"] is None

    # Same submission id with a confidence value is a different payload.
    with pytest.raises(SubmissionConflict):
        service.attempt(
            SPEC_ID,
            AttemptInput(
                item_id=ITEM_ID,
                rating="good",
                session_id=SESSION_ID,
                submission_id="v3-identity-1",
                confidence=80,
            ),
        )


def test_zero_confidence_is_a_real_value_not_a_default(service: ReviewService) -> None:
    """0% confidence must enter the identity hash (`is not None`, not truthiness)."""
    service.put_spec(build_fixture_spec())
    first = service.attempt(
        SPEC_ID,
        AttemptInput(
            item_id=ITEM_ID,
            rating="good",
            session_id=SESSION_ID,
            submission_id="zero-confidence-1",
            confidence=0.0,
        ),
    )
    assert first["created"] is True
    assert first["confidence"] == 0.0

    replay = service.attempt(
        SPEC_ID,
        AttemptInput(
            item_id=ITEM_ID,
            rating="good",
            session_id=SESSION_ID,
            submission_id="zero-confidence-1",
            confidence=0.0,
        ),
    )
    assert replay["created"] is False

    with pytest.raises(SubmissionConflict):
        service.attempt(
            SPEC_ID,
            AttemptInput(
                item_id=ITEM_ID,
                rating="good",
                session_id=SESSION_ID,
                submission_id="zero-confidence-1",
                confidence=None,
            ),
        )
