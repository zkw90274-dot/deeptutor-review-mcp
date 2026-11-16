"""Phase 34 adversarial edges: migration, hint boundaries, identity, windows.

Seven contracts are pinned here, each at a boundary the happy-path suites do not
reach:

* a schema-v2 database upgrades in place and keeps its history readable;
* the generated-pack hint brief truncates at exactly 80 characters and the
  leak gate still runs *after* truncation;
* a numeric answer is formatted before it is compared, and one-character
  answers stay below the leak threshold;
* ``submission_id`` identity includes ``confidence`` and ``hints_used``;
* the calibration window is a lower bound and the confidence buckets are
  right-closed;
* ``days_left`` is counted in the spec timezone, not UTC;
* exam mode never adds answers to the Today payload.

All tool calls run under ``asyncio.run`` because no async pytest plugin is
installed in this package, matching ``tests/test_hint_tool.py``.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from deeptutor.core.tool_protocol import BaseTool, ToolResult
from deeptutor.review.agent_tools import ReviewHintTool, ReviewTodayTool
from deeptutor.review.fixture import build_fixture_spec
from deeptutor.review.models import ReviewSpec
from deeptutor.review.queue import _exam_plan
from deeptutor.review.service import AttemptInput, ReviewService
from deeptutor.review.storage import SCHEMA_VERSION, ReviewStore, SubmissionConflict
from deeptutor.review.timeutil import local_date, zone

from conftest import NOW

DAY = 86_400.0

CHOICE_ITEM = "item-resonance-choice"
NUMERIC_ITEM = "item-resonance-numeric"
FLASHCARD_ITEM = "item-quality-flashcard"
GENERATED_CONCEPT = 0  # c-series-resonance
FLASHCARD_CONCEPT = 1  # c-quality-factor
CHOICE_OPTION_TEXT = "因为感抗与容抗相互抵消"


# -- shared helpers ---------------------------------------------------------


def _run(tool: BaseTool, **kwargs: Any) -> ToolResult:
    return asyncio.run(tool.execute(**kwargs))


def _payload(result: ToolResult, *, tool: str) -> dict[str, Any]:
    assert result.success is True, result.content
    payload = json.loads(result.content)
    assert isinstance(payload, dict)
    assert "instruction" in payload and payload["instruction"]
    assert result.metadata[tool] == payload
    return payload


def _mutated_spec(service: ReviewService, mutate: Any) -> ReviewSpec:
    """Store a fixture-derived spec after ``mutate`` edits the JSON payload.

    Editing a validated model leaves ``content_hash`` stale, so it is cleared
    before revalidation; otherwise ``put_spec`` rejects the mismatch.
    """
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    mutate(payload)
    spec = ReviewSpec.model_validate(payload)
    service.put_spec(spec)
    return spec


def _hint(spec_id: str, item_id: str, *, level: int = 1) -> dict[str, Any]:
    return _payload(
        _run(ReviewHintTool(), spec_id=spec_id, item_id=item_id, level=level),
        tool="review_hint",
    )


def _attempt(
    service: ReviewService,
    spec: ReviewSpec,
    *,
    now: float,
    index: str,
    rating: str = "good",
    item_id: str = CHOICE_ITEM,
    hints_used: int = 0,
    confidence: float | None = None,
    submission_id: str | None = None,
) -> dict[str, Any]:
    return service.attempt(
        spec.id,
        AttemptInput(
            item_id=item_id,
            rating=rating,
            session_id=f"sess-{index}",
            submission_id=submission_id or f"sub-{index}",
            hints_used=hints_used,
            confidence=confidence,
        ),
        now=now,
    )


# -- 1. schema v2 -> v3 migration ------------------------------------------


_LEGACY_EVENT_ROW = (
    "legacy-event",
    "legacy-submission",
    "legacy-spec",
    "legacy-item",
    "legacy-concept-key",
    "legacy-session",
    NOW - 3_600.0,
    "good",
    "correct",
    1.0,
    500,
    3.0,
    0.8,
    "review",
    "0.0.1",
    "1.0",
    "legacy-payload-hash",
)


def _write_legacy_v2_database(home: Path) -> None:
    """Create the v2 schema by hand, without the two v3 columns."""
    home.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(home / "review.sqlite3")
    try:
        conn.executescript(
            """
            CREATE TABLE review_schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at REAL NOT NULL
            );

            CREATE TABLE review_specs (
                spec_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                spec_json TEXT NOT NULL
            );

            CREATE TABLE review_events (
                event_id TEXT PRIMARY KEY,
                submission_id TEXT NOT NULL UNIQUE,
                spec_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                concept_key TEXT NOT NULL,
                session_id TEXT NOT NULL,
                ts REAL NOT NULL,
                rating TEXT NOT NULL CHECK (rating IN ('again', 'hard', 'good')),
                result TEXT NOT NULL,
                quality REAL NOT NULL,
                elapsed_ms INTEGER NOT NULL DEFAULT 0,
                scheduled_interval_days REAL NOT NULL,
                predicted_retrievability REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'review',
                app_version TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                payload_hash TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO review_schema_migrations (version, applied_at) VALUES (?, ?)",
            (2, NOW),
        )
        conn.execute(
            "INSERT INTO review_events ("
            "event_id, submission_id, spec_id, item_id, concept_key, session_id, ts, "
            "rating, result, quality, elapsed_ms, scheduled_interval_days, "
            "predicted_retrievability, source, app_version, schema_version, payload_hash"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _LEGACY_EVENT_ROW,
        )
        conn.commit()
    finally:
        conn.close()


def _event_columns(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {str(row[1]) for row in conn.execute("PRAGMA table_info(review_events)")}
    finally:
        conn.close()


def _migration_version(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT MAX(version) FROM review_schema_migrations"
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


def test_v2_database_gains_v3_columns_without_losing_history(tmp_path: Path) -> None:
    home = tmp_path / "legacy"
    _write_legacy_v2_database(home)

    store = ReviewStore(home)

    columns = _event_columns(store.db_path)
    assert "hints_used" in columns
    assert "confidence" in columns
    assert _migration_version(store.db_path) == 4
    assert SCHEMA_VERSION == 4

    events = store.events_for_spec("legacy-spec")
    assert len(events) == 1
    event = events[0]
    # The old row survives the ALTER TABLE and picks up the column defaults.
    assert event["event_id"] == "legacy-event"
    assert event["rating"] == "good"
    assert event["hints_used"] == 0
    assert event["confidence"] is None


def test_reopening_the_migrated_database_is_idempotent(tmp_path: Path) -> None:
    home = tmp_path / "legacy"
    _write_legacy_v2_database(home)

    ReviewStore(home)
    reopened = ReviewStore(home)

    assert _migration_version(reopened.db_path) == 4
    assert _event_columns(reopened.db_path) >= {"hints_used", "confidence"}
    assert len(reopened.events_for_spec("legacy-spec")) == 1


# -- 2. generated-pack hint brief boundaries -------------------------------


def _generated_name_spec(
    service: ReviewService, *, name: str, concept_index: int = GENERATED_CONCEPT
) -> ReviewSpec:
    """A pack shaped like the generator output: a long name and nothing else."""

    def mutate(payload: dict[str, Any]) -> None:
        concept = payload["concepts"][concept_index]
        concept["name"] = name
        concept["definition"] = ""
        concept["summary_one_line"] = ""
        concept["formulas"] = []

    return _mutated_spec(service, mutate)


def test_hint_brief_keeps_an_eighty_character_name_unchanged(
    service: ReviewService,
) -> None:
    name = "A" * 80
    spec = _generated_name_spec(service, name=name)

    payload = _hint(spec.id, CHOICE_ITEM, level=1)

    assert payload["fallback"] is False
    assert payload["hint"] == name


def test_hint_brief_truncates_an_eighty_one_character_name(
    service: ReviewService,
) -> None:
    name = "B" * 81
    spec = _generated_name_spec(service, name=name)

    payload = _hint(spec.id, CHOICE_ITEM, level=1)

    assert payload["fallback"] is False
    assert payload["hint"] == "B" * 79 + "…"
    assert len(payload["hint"]) == 80


def test_hint_brief_leaves_a_short_name_unchanged(service: ReviewService) -> None:
    spec = _generated_name_spec(service, name="谐振要点")

    payload = _hint(spec.id, CHOICE_ITEM, level=1)

    assert payload["fallback"] is False
    assert payload["hint"] == "谐振要点"


def test_leak_gate_still_runs_after_the_brief_is_truncated(
    service: ReviewService,
) -> None:
    # The answer sits at the start of the name, so it survives the 79-char cut.
    answer_text = r"Q = \frac{1}{R}\sqrt{\frac{L}{C}}"
    name = answer_text + "Z" * (81 - len(answer_text))
    spec = _generated_name_spec(service, name=name, concept_index=FLASHCARD_CONCEPT)

    payload = _hint(spec.id, FLASHCARD_ITEM, level=1)

    assert payload["fallback"] is True
    assert answer_text not in payload["hint"]
    assert payload["source"] is not None


# -- 3. numeric answer leak threshold --------------------------------------


def _numeric_answer_spec(
    service: ReviewService, *, value: float, definition: str
) -> ReviewSpec:
    def mutate(payload: dict[str, Any]) -> None:
        concept = payload["concepts"][GENERATED_CONCEPT]
        concept["definition"] = definition
        concept["summary_one_line"] = ""
        concept["formulas"] = []
        for item in payload["items"]:
            if item["id"] == NUMERIC_ITEM:
                item["answer"] = {"value": value, "tolerance": 0.0, "text": ""}
                # G4 requires the explanation to contain the answer.
                item["explanation"] = f"谐振频率为 {value:g} 兆赫。"

    return _mutated_spec(service, mutate)


def test_numeric_answer_with_four_characters_still_counts_as_a_leak(
    service: ReviewService,
) -> None:
    spec = _numeric_answer_spec(
        service, value=9.95, definition="回路在 9.95 兆赫时发生谐振。"
    )

    payload = _hint(spec.id, NUMERIC_ITEM, level=2)

    assert payload["fallback"] is True
    assert "9.95" not in payload["hint"]


def test_one_character_numeric_result_is_not_exempt_from_the_leak_gate(
    service: ReviewService,
) -> None:
    definition = "回路在频率 7 时发生谐振。"
    spec = _numeric_answer_spec(service, value=7, definition=definition)

    payload = _hint(spec.id, NUMERIC_ITEM, level=2)

    assert payload["fallback"] is True
    assert "7" not in payload["hint"]


# -- 4. attempt identity ----------------------------------------------------


def test_same_submission_with_a_different_confidence_is_a_conflict(
    service: ReviewService,
) -> None:
    spec = _mutated_spec(service, lambda payload: None)
    _attempt(
        service,
        spec,
        now=NOW,
        index="identity",
        hints_used=1,
        confidence=40,
        submission_id="same-submission",
    )

    with pytest.raises(SubmissionConflict):
        _attempt(
            service,
            spec,
            now=NOW,
            index="identity",
            hints_used=1,
            confidence=60,
            submission_id="same-submission",
        )


def test_replay_with_the_same_confidence_keeps_hints_and_confidence(
    service: ReviewService,
) -> None:
    spec = _mutated_spec(service, lambda payload: None)
    first = _attempt(
        service,
        spec,
        now=NOW,
        index="replay",
        hints_used=1,
        confidence=40,
        submission_id="replay-submission",
    )
    assert first["created"] is True

    replay = _attempt(
        service,
        spec,
        now=NOW + 60.0,
        index="replay",
        hints_used=1,
        confidence=40,
        submission_id="replay-submission",
    )

    assert replay["created"] is False
    assert replay["event_id"] == first["event_id"]
    assert replay["hints_used"] == 1
    assert replay["confidence"] == 40


def test_service_clamps_hints_used_to_two_and_treats_it_as_helped(
    service: ReviewService,
) -> None:
    spec = _mutated_spec(service, lambda payload: None)

    # The tool clamps first; this drives the service goal-line directly.
    result = _attempt(service, spec, now=NOW, index="clamp", hints_used=5)

    assert result["hints_used"] == 2
    # Answering after a hint is not unaided recall, so "good" is stored "hard".
    assert result["rating"] == "hard"
    assert result["requested_rating"] == "good"


# -- 5. calibration window and bucket edges --------------------------------


def _calibration(
    service: ReviewService, spec: ReviewSpec, *, days: int = 7
) -> dict[str, Any]:
    return service.metrics(spec.id, days=days, now=NOW)["learner_calibration"]


def _bucket(calibration: dict[str, Any], bucket_range: str) -> dict[str, Any]:
    return next(
        bucket for bucket in calibration["buckets"] if bucket["range"] == bucket_range
    )


def test_calibration_excludes_events_older_than_the_window(
    service: ReviewService,
) -> None:
    spec = _mutated_spec(service, lambda payload: None)
    # Eight days old: outside a 7-day window, so it must not be sampled.
    _attempt(
        service,
        spec,
        now=NOW - 8 * DAY,
        index="stale",
        confidence=90,
        submission_id="stale-submission",
    )
    for index, confidence in enumerate((0, 20, 100)):
        _attempt(
            service,
            spec,
            now=NOW,
            index=f"fresh-{index}",
            confidence=confidence,
        )

    calibration = _calibration(service, spec)

    assert calibration["samples"] == 3
    assert _bucket(calibration, "0-20")["samples"] == 2
    assert _bucket(calibration, "80-100")["samples"] == 1


def test_confidence_bucket_edges_are_right_closed(service: ReviewService) -> None:
    spec = _mutated_spec(service, lambda payload: None)
    # 0 folds into the first bucket; 20 sits on its upper edge; 100 on the last.
    for index, confidence in enumerate((0, 20, 100)):
        _attempt(
            service,
            spec,
            now=NOW,
            index=f"edge-{index}",
            confidence=confidence,
        )

    calibration = _calibration(service, spec)

    assert _bucket(calibration, "0-20")["samples"] == 2
    assert _bucket(calibration, "80-100")["samples"] == 1
    assert _bucket(calibration, "20-40")["samples"] == 0


def test_a_confident_again_rating_still_counts_as_a_miss(
    service: ReviewService,
) -> None:
    spec = _mutated_spec(service, lambda payload: None)
    _attempt(
        service,
        spec,
        now=NOW,
        index="miss",
        rating="again",
        confidence=50,
    )

    calibration = _calibration(service, spec)

    bucket = _bucket(calibration, "40-60")
    assert bucket["samples"] == 1
    assert bucket["actual_accuracy"] == 0
    assert bucket["mean_confidence"] == 50


# -- 6. exam countdown uses the spec timezone ------------------------------


def test_exam_days_left_is_counted_in_the_spec_timezone(
    service: ReviewService,
) -> None:
    # 20:00 UTC is already the next calendar day in Shanghai (04:00).
    moment = datetime(2026, 11, 6, 20, 0, tzinfo=timezone.utc).timestamp()
    utc_day = local_date(moment, ZoneInfo("UTC"))
    shanghai_day = local_date(moment, ZoneInfo("Asia/Shanghai"))
    assert utc_day != shanghai_day

    exam_date = shanghai_day + timedelta(days=4)

    def mutate(payload: dict[str, Any]) -> None:
        payload["plan"]["timezone"] = "Asia/Shanghai"
        payload["plan"]["exam_date"] = exam_date.isoformat()

    spec = _mutated_spec(service, mutate)

    today = service.today(spec.id, session_id="timezone-edge", now=moment)

    assert today["exam"]["exam_date"] == exam_date.isoformat()
    # Four Shanghai days; a UTC-based count would say five.
    assert today["exam"]["days_left"] == 4
    assert today["exam"]["mode"] == "sprint"
    assert (exam_date - utc_day).days == 5

    exam_block, _planned = _exam_plan(spec, moment, zone("Asia/Shanghai"))
    assert exam_block["days_left"] == 4


# -- 7. exam mode never leaks answers --------------------------------------


def test_final_week_today_tool_keeps_the_exam_block_and_withholds_answers(
    service: ReviewService,
) -> None:
    # The tool reads the real clock, so anchor the exam to today in Shanghai.
    shanghai_today = local_date(time.time(), ZoneInfo("Asia/Shanghai"))
    exam_date = shanghai_today + timedelta(days=2)

    def mutate(payload: dict[str, Any]) -> None:
        payload["plan"]["exam_date"] = exam_date.isoformat()

    spec = _mutated_spec(service, mutate)

    today = _payload(_run(ReviewTodayTool(), spec_id=spec.id), tool="review_today")

    assert today["exam"]["mode"] == "final_week"
    assert today["exam"]["days_left"] in (1, 2)
    assert today["items"]
    for item in today["items"]:
        assert "answer" not in item
        assert "explanation" not in item

    first_item = today["items"][0]
    hint = _hint(spec.id, first_item["item_id"], level=1)
    assert hint["item_id"] == first_item["item_id"]
    assert hint["hint"]
    # The hint ladder is subject-anchored prose, never the stored answer key.
    assert hint["hint"] != CHOICE_OPTION_TEXT
