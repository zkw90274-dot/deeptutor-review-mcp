"""Reply-accuracy suite: does every field the model receives match the store?

The other agent-tool tests check the contract (shape, instruction, soft
failures). This suite asks a narrower and more suspicious question: after a
tool returns, is each fact in its payload *equal to the authoritative value*
in the spec / event log / sqlite file, or merely self-consistent?

Truth is read independently of the tool:

* a fresh ``ReviewService`` instance over the same ``DEEPTUTOR_REVIEW_HOME``;
* ``ReviewStore.get_spec`` for the stored spec (what was persisted, not what
  the test object happens to hold);
* raw ``sqlite3`` over ``ReviewStore.db_path`` for sessions, views and the
  append-only event log.

Only ``service`` / ``seeded`` and ``build_fixture_spec`` are used; no network,
no LLM, no async plugin (``asyncio.run`` mirrors the rest of the package).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from deeptutor.core.tool_protocol import BaseTool, ToolResult
from deeptutor.review.agent_tools import (
    ReviewAttemptTool,
    ReviewSpecsTool,
    ReviewTodayTool,
    ReviewWeaknessTool,
)
from deeptutor.review.correction import answer_text
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.models import ReviewSpec
from deeptutor.review.qa import run_gates
from deeptutor.review.service import AttemptInput, ReviewService


# -- helpers ----------------------------------------------------------------


def _run(tool: BaseTool, **kwargs: Any) -> ToolResult:
    return asyncio.run(tool.execute(**kwargs))


def _payload(result: ToolResult, *, tool: str) -> dict[str, Any]:
    assert result.success is True, result.content
    payload = json.loads(result.content)
    assert isinstance(payload, dict)
    assert result.metadata[tool] == payload
    return payload


def _sql(service: ReviewService, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Read the same database the tools write, without going through the store."""
    connection = sqlite3.connect(str(service.store.db_path))
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(query, params).fetchall()]
    finally:
        connection.close()


@pytest.fixture()
def seeded(service: ReviewService) -> ReviewService:
    service.put_spec(build_fixture_spec())
    return service


def _spec_with_stems(service: ReviewService) -> ReviewSpec:
    """Store the fixture with stems that stress the tool's text handling.

    The choice item carries LaTeX, the numeric item exceeds the old 200-char
    clip, and the flashcard is a genuine multi-line question. The QA gates run
    on store, so a spec that fails them would fail this fixture too.
    """
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    long_stem = (
        "在串联谐振回路的分析中，需要先判断感抗与容抗是否相等，再比较回路总阻抗与电流的关系，"
        "并说明为什么谐振时电流达到最大值；请把判断依据、阻抗表达式以及电流随频率变化的趋势"
        "分别写清楚，最后给出一个生活中的类比来帮助记忆这一结论。"
    ) * 3
    assert len(long_stem) > 200
    stems = {
        "item-resonance-choice": (
            "串联谐振发生时，回路阻抗为什么最小？提示：$Z = R$ 且 $X_L = X_C$。"
        ),
        "item-resonance-numeric": long_stem,
        "item-quality-flashcard": (
            "第一问：写出串联谐振的条件。\n"
            "第二问：写出品质因数 Q 与带宽的关系，并说明二者的联系。"
        ),
    }
    for item in payload["items"]:
        item["stem"] = stems[item["id"]]
    spec = ReviewSpec.model_validate(payload)
    service.put_spec(spec)
    return service.store.get_spec(SPEC_ID)


# -- 1. review_specs --------------------------------------------------------


def test_review_specs_rows_match_stored_spec_and_qa_gate(seeded: ReviewService) -> None:
    stored = seeded.store.get_spec(SPEC_ID)
    # The stored quality report is what put_spec recorded from the gates; the
    # gate result is deterministic, so a re-run must agree with it.
    gate = run_gates(stored)
    assert gate.passed == stored.quality.passed
    assert gate.warnings == stored.quality.warnings

    payload = _payload(_run(ReviewSpecsTool()), tool="review_specs")
    listed = seeded.list_specs()
    assert payload["count"] == len(listed) == 1
    row = payload["specs"][0]

    assert row["id"] == stored.id
    assert row["title"] == stored.title
    assert row["items"] == len(stored.items)
    assert row["concepts"] == len(stored.concepts)
    assert row["qa_passed"] is stored.quality.passed
    assert row["qa_warnings"] == len(stored.quality.warnings)
    assert row["updated_at"] == listed[0]["updated_at"]


# -- 2. review_today --------------------------------------------------------


def test_review_today_budget_minutes_and_item_count_match_service(
    seeded: ReviewService,
) -> None:
    stored = seeded.store.get_spec(SPEC_ID)
    payload = _payload(_run(ReviewTodayTool(), spec_id=SPEC_ID), tool="review_today")

    # Re-read the same session through the service: this must not create a new
    # queue, and the numbers must be the ones the service itself reports.
    truth = seeded.today(SPEC_ID, session_id=payload["session_id"])

    assert payload["budget"] == truth["budget"]
    assert payload["planned_minutes"] == truth["planned_minutes"]
    assert payload["planned_minutes"] == stored.plan.daily_minutes
    assert payload["estimated_minutes"] == truth["estimated_minutes"]
    assert len(payload["items"]) == len(truth["items"])
    assert {item["item_id"] for item in payload["items"]} == {
        item["item_id"] for item in truth["items"]
    }
    assert len(payload["items"]) <= payload["budget"]


def test_review_today_items_are_byte_identical_for_latex_and_long_stems(
    service: ReviewService,
) -> None:
    stored = _spec_with_stems(service)
    payload = _payload(_run(ReviewTodayTool(), spec_id=SPEC_ID), tool="review_today")

    served = {item["item_id"]: item for item in payload["items"]}
    assert set(served) == {item.id for item in stored.items}

    # Single-line items only: the multi-line flashcard is the subject of
    # ``test_review_today_preserves_multiline_stem_bytes`` below, where the
    # whitespace normalisation is isolated as its own failure.
    for item_id in ("item-resonance-choice", "item-resonance-numeric"):
        row = served[item_id]
        item = stored.item_by_id(item_id)
        concept = stored.concept_by_id(item.concept_ids[0])
        assert row["kind"] == item.kind
        assert row["difficulty"] == item.difficulty
        assert row["concept_id"] == concept.id
        # Concept names are excluded from today's payload: generated names can
        # state the answer outright ("中和法与失配法"), and the tool promises
        # the model has no hint. The weakness tool serves concept names after
        # the fact instead.
        assert "concept_name" not in row
        assert row["stem"] == item.stem
        assert row.get("options", []) == [
            {"label": option.label, "text": option.text} for option in item.options
        ]

    # The two properties the old 200-char clip used to break: LaTeX survives,
    # and a stem longer than the metadata clip survives in full.
    assert "$X_L = X_C$" in served["item-resonance-choice"]["stem"]
    assert len(served["item-resonance-numeric"]["stem"]) > 200
    assert served["item-resonance-numeric"]["stem"] == next(
        item.stem for item in stored.items if item.id == "item-resonance-numeric"
    )


def test_review_today_preserves_multiline_stem_bytes(service: ReviewService) -> None:
    """A multi-line stem must reach the model exactly as stored.

    The tool's own instruction tells the model to read the stem "exactly as
    given". ``agent_tools._clip`` normalises it through ``_text``
    (``" ".join(str(value).split())``), which turns every newline into a
    space, so the newline in this stem is silently lost on the way out.
    """
    stored = _spec_with_stems(service)
    original = next(
        item.stem for item in stored.items if item.id == "item-quality-flashcard"
    )
    assert "\n" in original

    payload = _payload(_run(ReviewTodayTool(), spec_id=SPEC_ID), tool="review_today")
    row = next(
        item for item in payload["items"] if item["item_id"] == "item-quality-flashcard"
    )
    assert row["stem"] == original


def test_review_today_withholds_answers_and_explanations(seeded: ReviewService) -> None:
    stored = seeded.store.get_spec(SPEC_ID)
    result = _run(ReviewTodayTool(), spec_id=SPEC_ID)
    payload = _payload(result, tool="review_today")

    assert "answer" not in payload
    assert "explanation" not in payload
    for item in payload["items"]:
        assert "answer" not in item
        assert "explanation" not in item
        assert "correction" not in item
    for item in stored.items:
        if item.explanation:
            assert item.explanation not in result.content


# -- 3. session reuse -------------------------------------------------------


def test_review_today_session_reuse_keeps_a_single_session_and_view_rows(
    seeded: ReviewService,
) -> None:
    stored = seeded.store.get_spec(SPEC_ID)
    payload = _payload(_run(ReviewTodayTool(), spec_id=SPEC_ID), tool="review_today")
    session_id = payload["session_id"]
    assert session_id

    sessions = _sql(seeded, "SELECT * FROM review_sessions")
    assert len(sessions) == 1
    session = sessions[0]
    assert session["session_id"] == session_id
    assert session["spec_id"] == SPEC_ID
    assert session["ended_at"] is None
    assert session["planned_minutes"] == payload["planned_minutes"]
    assert session["planned_minutes"] == stored.plan.daily_minutes

    # Passing the returned session id back must reuse it, not fork a second one.
    truth = seeded.today(SPEC_ID, session_id=session_id)
    assert truth["session_id"] == session_id
    assert len(_sql(seeded, "SELECT * FROM review_sessions")) == 1

    views = _sql(
        seeded,
        "SELECT item_id FROM review_views WHERE session_id = ?",
        (session_id,),
    )
    assert len(views) == len(payload["items"])
    assert len(views) == len(truth["items"])
    assert {row["item_id"] for row in views} == {
        item["item_id"] for item in payload["items"]
    }


# -- 4. review_attempt ------------------------------------------------------


def _attempt_with_session(
    service: ReviewService,
    *,
    item_id: str = "item-quality-flashcard",
    rating: str = "good",
    submission_id: str = "reply-accuracy-attempt-1",
) -> tuple[dict[str, Any], str]:
    today = _payload(_run(ReviewTodayTool(), spec_id=SPEC_ID), tool="review_today")
    session_id = today["session_id"]
    payload = _payload(
        _run(
            ReviewAttemptTool(),
            spec_id=SPEC_ID,
            item_id=item_id,
            rating=rating,
            session_id=session_id,
            submission_id=submission_id,
        ),
        tool="review_attempt",
    )
    return payload, session_id


def test_review_attempt_returns_stored_answer_and_explanation_bytes(
    seeded: ReviewService,
) -> None:
    payload, _ = _attempt_with_session(seeded)
    stored = seeded.store.get_spec(SPEC_ID)
    item = stored.item_by_id(payload["item_id"])

    # Byte-for-byte, not "same gist": the flashcard answer is raw LaTeX.
    assert payload["answer"] == answer_text(item)
    assert payload["answer"] == item.answer.text
    assert payload["explanation"] == item.explanation
    assert payload["concept_id"] == item.concept_ids[0]

    # The event on disk is the authority for the identifier the model was given.
    events = _sql(
        seeded,
        "SELECT * FROM review_events WHERE submission_id = ?",
        (payload["submission_id"],),
    )
    assert len(events) == 1
    assert payload["event_id"] == events[0]["event_id"]


def test_review_attempt_rating_and_projection_numbers_match_stored_event(
    seeded: ReviewService,
) -> None:
    payload, _ = _attempt_with_session(seeded, rating="good")

    events = _sql(
        seeded,
        "SELECT * FROM review_events WHERE submission_id = ?",
        (payload["submission_id"],),
    )
    assert len(events) == 1
    event = events[0]

    assert payload["created"] is True
    assert payload["rating"] == event["rating"] == "good"
    assert payload["requested_rating"] == "good"
    assert payload["predicted_retrievability"] == round(
        event["predicted_retrievability"], 4
    )
    assert payload["next"]["interval_days"] == round(
        event["scheduled_interval_days"], 6
    )

    # Unlinked pack: the projection has no numbers to report, and the tool must
    # say so instead of inventing a core due date.
    stored = seeded.store.get_spec(SPEC_ID)
    assert not stored.mastery_path_id
    projection = payload["mastery_projection"]
    assert projection["status"] == "skipped"
    assert projection["diagnostics"] == ["no_mastery_linkage"]
    assert projection["core_next_due_ts"] is None
    assert projection["due_date_consistent"] is None


def test_review_attempt_next_date_matches_service_state_for_the_event(
    seeded: ReviewService,
) -> None:
    payload, _ = _attempt_with_session(seeded)

    # The service-level state query replays the event log; its due date is the
    # scheduler's answer for this concept after the recorded rating.
    report = seeded.weakness(SPEC_ID, limit=5)
    row = next(
        concept
        for concept in report["weak_concepts"]
        if concept["concept_id"] == payload["concept_id"]
    )
    assert payload["next"]["due_at_ts"] == row["due_at_ts"]
    assert payload["next"]["due_at"] == row["due_at"]


def test_review_attempt_replay_same_submission_records_one_event(
    seeded: ReviewService,
) -> None:
    first, session_id = _attempt_with_session(seeded)
    second = _payload(
        _run(
            ReviewAttemptTool(),
            spec_id=SPEC_ID,
            item_id=first["item_id"],
            rating=first["requested_rating"],
            submission_id=first["submission_id"],
        ),
        tool="review_attempt",
    )

    assert second["created"] is False
    assert second["event_id"] == first["event_id"]
    assert second["submission_id"] == first["submission_id"]
    assert second["answer"] == first["answer"]
    assert second["explanation"] == first["explanation"]
    assert second["next"] == first["next"]

    events = _sql(
        seeded,
        "SELECT * FROM review_events WHERE submission_id = ?",
        (first["submission_id"],),
    )
    assert len(events) == 1
    assert len(_sql(seeded, "SELECT * FROM review_events WHERE spec_id = ?", (SPEC_ID,))) == 1
    # The replay did not silently end or fork the session it was recorded in.
    assert len(_sql(seeded, "SELECT * FROM review_sessions")) <= 1
    assert all(row["session_id"] == session_id for row in events)


def test_review_attempt_next_due_is_iso_in_plan_timezone(seeded: ReviewService) -> None:
    payload, _ = _attempt_with_session(seeded)
    stored = seeded.store.get_spec(SPEC_ID)
    tz = ZoneInfo(stored.plan.timezone)

    event = _sql(
        seeded,
        "SELECT ts FROM review_events WHERE submission_id = ?",
        (payload["submission_id"],),
    )[0]

    due_at = payload["next"]["due_at"]
    assert isinstance(due_at, str)
    parsed = datetime.fromisoformat(due_at)
    assert parsed.tzinfo is not None

    # The string must denote the same instant as the scheduler's timestamp.
    assert parsed.timestamp() == pytest.approx(
        payload["next"]["due_at_ts"], abs=1e-6
    )
    # Its offset must be the pack's timezone at that instant (not UTC, not the
    # machine's local zone), and the date must be today or later there.
    expected_offset = datetime.fromtimestamp(parsed.timestamp(), tz).utcoffset()
    assert parsed.utcoffset() == expected_offset
    event_local_date = datetime.fromtimestamp(event["ts"], tz).date()
    assert parsed.date() >= event_local_date


# -- 5. review_weakness -----------------------------------------------------


def test_review_weakness_rows_match_service_rows_and_evidence_counts(
    seeded: ReviewService,
) -> None:
    # Two recorded events on two different concepts, so evidence counts differ
    # from the item counts and a copy-paste of item_count cannot pass.
    seeded.attempt(
        SPEC_ID,
        AttemptInput(item_id="item-resonance-choice", rating="good", submission_id="wa-1"),
    )
    seeded.attempt(
        SPEC_ID,
        AttemptInput(item_id="item-quality-flashcard", rating="again", submission_id="wa-2"),
    )

    payload = _payload(_run(ReviewWeaknessTool(), spec_id=SPEC_ID), tool="review_weakness")
    truth = seeded.weakness(SPEC_ID, limit=5)

    assert payload["spec_id"] == truth["spec_id"]
    assert payload["limit"] == truth["limit"] == 5
    assert [row["concept_id"] for row in payload["weak_concepts"]] == [
        row["concept_id"] for row in truth["weak_concepts"]
    ]

    expected = {row["concept_id"]: row for row in truth["weak_concepts"]}
    for row in payload["weak_concepts"]:
        source = expected[row["concept_id"]]
        assert row["weakness"] == source["weakness"]
        assert row["predicted_retrievability"] == source["predicted_retrievability"]
        assert row["item_ids"] == source["item_ids"]
        assert row["item_count"] == source["item_count"]
        assert row["state_source"] == source["state_source"]
        assert row["due_at"] == source["due_at"]

        events = _sql(
            seeded,
            "SELECT COUNT(*) AS n FROM review_events "
            "WHERE spec_id = ? AND concept_key = ?",
            (SPEC_ID, row["concept_key"]),
        )
        assert row["evidence_count"] == events[0]["n"]
        assert row["evidence_count"] == len(
            seeded.store.events_for_concept(SPEC_ID, row["concept_key"])
        )
