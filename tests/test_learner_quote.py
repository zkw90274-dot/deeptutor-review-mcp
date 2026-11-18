"""``learner_quote``: make "the learner rated it, not the model" auditable.

The rule was prompt-only: ``review_attempt`` told the model to transcribe the
learner's own verdict, and nothing in the record proved it did. The optional
quote closes that gap — the learner's own words ride on the event row and come
back in the response, so a later reader can compare the rating against what was
actually said.

What must NOT change: the quote is evidence, not identity. It never alters the
rating, the hint correction, the scheduler, or the idempotency hash (retries
that differ only by quote must stay the same submission).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from deeptutor.review.agent_tools import ReviewAttemptTool
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.service import (
    MAX_LEARNER_QUOTE_CHARS,
    AttemptInput,
    ReviewService,
)

ITEM_ID = "item-resonance-choice"


@pytest.fixture()
def service() -> ReviewService:
    svc = ReviewService()
    svc.put_spec(build_fixture_spec())
    return svc


def test_quote_rides_on_the_event_and_the_response(service: ReviewService) -> None:
    result = service.attempt(
        SPEC_ID,
        AttemptInput(
            item_id=ITEM_ID,
            rating="hard",
            session_id="sess-quote",
            submission_id="quote-1",
            learner_quote="会，但是想了很久",
        ),
    )
    assert result["learner_quote"] == "会，但是想了很久"
    stored = service.store.event_by_submission("quote-1")
    assert stored["learner_quote"] == "会，但是想了很久"


def test_absent_quote_stays_none(service: ReviewService) -> None:
    result = service.attempt(
        SPEC_ID,
        AttemptInput(item_id=ITEM_ID, rating="good", session_id="sess-quote"),
    )
    assert result["learner_quote"] is None


def test_quote_does_not_change_the_rating_or_the_identity(service: ReviewService) -> None:
    first = service.attempt(
        SPEC_ID,
        AttemptInput(
            item_id=ITEM_ID,
            rating="good",
            session_id="sess-quote",
            submission_id="quote-2",
            learner_quote="挺顺的",
        ),
    )
    # Same submission, same rating, different wording: still the same event.
    second = service.attempt(
        SPEC_ID,
        AttemptInput(
            item_id=ITEM_ID,
            rating="good",
            session_id="sess-quote",
            submission_id="quote-2",
            learner_quote="其实也还好",
        ),
    )
    assert second["created"] is False
    assert second["event_id"] == first["event_id"]
    assert second["rating"] == first["rating"] == "good"


def test_quote_is_clipped_not_rejected(service: ReviewService) -> None:
    long_quote = "很" * (MAX_LEARNER_QUOTE_CHARS * 2)
    result = service.attempt(
        SPEC_ID,
        AttemptInput(
            item_id=ITEM_ID,
            rating="hard",
            session_id="sess-quote",
            learner_quote=long_quote,
        ),
    )
    assert result["learner_quote"] is not None
    assert len(result["learner_quote"]) == MAX_LEARNER_QUOTE_CHARS


def test_blank_quote_is_treated_as_absent(service: ReviewService) -> None:
    result = service.attempt(
        SPEC_ID,
        AttemptInput(
            item_id=ITEM_ID,
            rating="good",
            session_id="sess-quote",
            learner_quote="   ",
        ),
    )
    assert result["learner_quote"] is None


def _call(**kwargs):
    result = asyncio.run(ReviewAttemptTool().execute(**kwargs))
    assert result.success is True, result.content
    return json.loads(result.content)


def test_tool_passes_the_quote_through(service: ReviewService) -> None:
    payload = _call(
        spec_id=SPEC_ID,
        item_id=ITEM_ID,
        rating="again",
        session_id="sess-tool-quote",
        learner_quote="完全想不起来",
    )
    assert payload["learner_quote"] == "完全想不起来"


def test_tool_rejects_a_non_string_quote_as_a_soft_failure(service: ReviewService) -> None:
    result = asyncio.run(
        ReviewAttemptTool().execute(
            spec_id=SPEC_ID,
            item_id=ITEM_ID,
            rating="good",
            session_id="sess-tool-quote",
            learner_quote=["会"],
        )
    )
    assert result.success is True
    payload = json.loads(result.content)
    assert "learner_quote must be a string" in payload["error"]
    assert payload["instruction"]
