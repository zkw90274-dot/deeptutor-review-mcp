"""Thin adapter between the review UI's three ratings and the core scheduler.

There is no interval arithmetic in this file. ``deeptutor.learning.scheduler``
owns every transition; this module only answers three questions:

* what does "again / hard / good" mean as ``LearningEvidence``;
* how do persisted events replay into a ``RepetitionState``;
* what is the retrievability of a concept at a given instant.
"""

from __future__ import annotations

from typing import Iterable, Literal

from deeptutor.learning.models import (
    KnowledgeType,
    LearningEvidence,
    RepetitionState,
)
from deeptutor.learning.scheduler import SpacedRepetitionScheduler

from deeptutor.review.models import Concept, ReviewSpec
from deeptutor.review.timeutil import iso as _iso
from deeptutor.review.timeutil import zone as _zone

Rating = Literal["again", "hard", "good"]

#: UI rating -> (core result, core quality).
#:
#: ``hard`` deliberately maps to ``correct/0.55`` rather than ``partial/0.4``:
#: the core treats ``quality < 0.5`` as a failure and would increase review
#: frequency for a learner who said "I remembered, with effort".
RATING_MAP: dict[str, tuple[str, float]] = {
    "again": ("incorrect", 0.0),
    "hard": ("correct", 0.55),
    "good": ("correct", 1.0),
}

SECONDS_PER_DAY = 86400.0


def scheduler_for(spec: ReviewSpec) -> SpacedRepetitionScheduler:
    return SpacedRepetitionScheduler(desired_retention=spec.plan.target_retention)


def knowledge_type_of(concept: Concept) -> KnowledgeType:
    return KnowledgeType(concept.knowledge_type)


def initial_state(
    concept: Concept,
    *,
    now: float,
    scheduler: SpacedRepetitionScheduler | None = None,
) -> RepetitionState:
    engine = scheduler or SpacedRepetitionScheduler()
    return engine.get_initial_state(knowledge_type_of(concept), now=now)


def evidence_for(
    concept: Concept,
    rating: str,
    *,
    timestamp: float,
    item_id: str = "",
    session_id: str = "",
) -> LearningEvidence:
    try:
        result, quality = RATING_MAP[rating]
    except KeyError as exc:
        raise ValueError(f"unknown rating: {rating}") from exc
    return LearningEvidence(
        question_id=item_id,
        knowledge_point_id=concept.key,
        timestamp=timestamp,
        source="review",
        assessment_type="review",
        result=result,
        quality=quality,
        session_id=session_id,
    )


def apply_rating(
    state: RepetitionState,
    concept: Concept,
    rating: str,
    *,
    now: float,
    item_id: str = "",
    session_id: str = "",
    scheduler: SpacedRepetitionScheduler | None = None,
) -> tuple[RepetitionState, float, float]:
    """Apply one rating in place.

    Returns ``(state, predicted_retrievability, scheduled_interval_days)``.
    ``predicted_retrievability`` is measured before the transition: it is the
    quantity the learner's answer is evidence about (roadmap §3.2).
    """
    engine = scheduler or SpacedRepetitionScheduler()
    predicted = engine.retrievability(state, now=now)
    evidence = evidence_for(
        concept, rating, timestamp=now, item_id=item_id, session_id=session_id
    )
    engine.schedule_review(state, knowledge_type_of(concept), evidence, now=now)
    interval_days = max(0.0, (state.next_review_at - now) / SECONDS_PER_DAY)
    return state, predicted, interval_days


def replay_state(
    concept: Concept,
    events: Iterable[Any],
    *,
    now: float,
    scheduler: SpacedRepetitionScheduler,
) -> RepetitionState:
    """Recompute a concept's state from its append-only event log.

    This is the read path. The materialized ``review_state`` row is a cache;
    a disagreement is a bug in the writer, not a reason to trust the cache.
    ``scheduler`` is required: falling back to the default 0.9 retention while
    the write path used ``plan.target_retention`` would show two due dates for
    the same concept.
    """
    engine = scheduler
    rows = list(events)
    if not rows:
        return engine.get_initial_state(knowledge_type_of(concept), now=now)
    evidence = [
        LearningEvidence(
            question_id=str(row["item_id"] or ""),
            knowledge_point_id=concept.key,
            timestamp=float(row["ts"]),
            source="review",
            assessment_type="review",
            result=str(row["result"]),
            quality=float(row["quality"]),
            session_id=str(row["session_id"] or ""),
        )
        for row in rows
    ]
    return engine.replay(
        knowledge_type_of(concept),
        evidence,
        initial_state=engine.get_initial_state(
            knowledge_type_of(concept), now=evidence[0].timestamp
        ),
    )


def retrievability(
    concept: Concept,
    state: RepetitionState,
    *,
    now: float,
    scheduler: SpacedRepetitionScheduler,
) -> float:
    return scheduler.retrievability(state, now=now)


def iso_from_timestamp(timestamp: float, timezone: str = "Asia/Shanghai") -> str:
    return _iso(timestamp, _zone(timezone))
