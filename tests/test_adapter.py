from __future__ import annotations

import pytest

from deeptutor.learning.scheduler import SpacedRepetitionScheduler
from deeptutor.review.adapter import (
    RATING_MAP,
    apply_rating,
    initial_state,
    replay_state,
)
from deeptutor.review.fixture import build_fixture_spec

NOW = 1_800_000_000.0
DAY = 86400.0


@pytest.mark.parametrize(
    ("rating", "result", "quality"),
    [("again", "incorrect", 0.0), ("hard", "correct", 0.55), ("good", "correct", 1.0)],
)
def test_rating_mapping_is_locked(rating: str, result: str, quality: float) -> None:
    assert RATING_MAP[rating] == (result, quality)


def _concept(knowledge_type: str):
    spec = build_fixture_spec()
    return spec.concepts[0].model_copy(update={"knowledge_type": knowledge_type})


@pytest.mark.parametrize(
    ("knowledge_type", "expected"),
    [
        ("memory", {"good": 0.115, "hard": 0.059, "again": 0.014}),
        ("concept", {"good": 6.416, "hard": 3.325, "again": 0.388}),
    ],
)
def test_documented_single_rating_intervals(knowledge_type: str, expected: dict[str, float]) -> None:
    concept = _concept(knowledge_type)
    engine = SpacedRepetitionScheduler(desired_retention=0.9)
    for rating, interval in expected.items():
        state = initial_state(concept, now=NOW, scheduler=engine)
        _, predicted, scheduled = apply_rating(
            state, concept, rating, now=NOW, scheduler=engine
        )
        assert round(scheduled, 3) == interval
        assert 0.0 <= predicted <= 1.0


@pytest.mark.parametrize(
    ("knowledge_type", "expected"),
    [("memory", 0.799), ("concept", 17.361)],
)
def test_two_consecutive_good_ratings(knowledge_type: str, expected: float) -> None:
    concept = _concept(knowledge_type)
    engine = SpacedRepetitionScheduler(desired_retention=0.9)
    state = initial_state(concept, now=NOW, scheduler=engine)
    apply_rating(state, concept, "good", now=NOW, scheduler=engine)
    _, _, scheduled = apply_rating(
        state, concept, "good", now=NOW + 6 * DAY, scheduler=engine
    )
    assert round(scheduled, 3) == expected


def test_again_enters_failure_branch() -> None:
    concept = _concept("concept")
    engine = SpacedRepetitionScheduler(desired_retention=0.9)
    state = initial_state(concept, now=NOW, scheduler=engine)
    apply_rating(state, concept, "again", now=NOW, scheduler=engine)
    assert state.lapse_count == 1
    assert state.scheduled_after_failure is True


def test_hard_is_not_a_failure() -> None:
    concept = _concept("concept")
    engine = SpacedRepetitionScheduler(desired_retention=0.9)
    state = initial_state(concept, now=NOW, scheduler=engine)
    apply_rating(state, concept, "hard", now=NOW, scheduler=engine)
    assert state.lapse_count == 0
    assert state.scheduled_after_failure is False


def test_replay_reconstructs_state() -> None:
    concept = _concept("concept")
    engine = SpacedRepetitionScheduler(desired_retention=0.9)
    state = initial_state(concept, now=NOW, scheduler=engine)
    apply_rating(state, concept, "good", now=NOW, scheduler=engine)

    class Row(dict):
        pass

    events = [
        Row(
            item_id="i",
            ts=NOW,
            result="correct",
            quality=1.0,
            session_id="s",
        )
    ]
    replayed = replay_state(concept, events, now=NOW, scheduler=engine)
    assert replayed.next_review_at == pytest.approx(state.next_review_at)
    assert replayed.stability == pytest.approx(state.stability)
