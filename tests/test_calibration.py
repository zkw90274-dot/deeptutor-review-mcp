"""Learner calibration aggregate (Phase 34).

Events are built through the real ``ReviewService.attempt`` path rather than by
hand so the test exercises the persisted ``rating``/``confidence`` columns the
aggregate actually reads. Every attempt uses its own session so a repeated item
cannot be downgraded to ``hard`` by the same-session retry rule.
"""

from __future__ import annotations

from typing import Any

import pytest

from deeptutor.review.metrics import learner_calibration
from deeptutor.review.models import ReviewSpec
from deeptutor.review.service import AttemptInput, ReviewService

NOW = 1_800_000_000.0
BUCKET_RANGES = ["0-20", "20-40", "40-60", "60-80", "80-100"]

RESONANCE_ITEM = "item-resonance-choice"
QUALITY_ITEM = "item-quality-flashcard"


def _attempt(
    service: ReviewService,
    spec: ReviewSpec,
    *,
    index: int,
    confidence: float | None,
    rating: str = "good",
    item_id: str = RESONANCE_ITEM,
) -> None:
    service.attempt(
        spec.id,
        AttemptInput(
            item_id=item_id,
            rating=rating,
            session_id=f"sess-{index}",
            submission_id=f"sub-{index}",
            confidence=confidence,
        ),
        now=NOW,
    )


def _calibration(service: ReviewService, spec: ReviewSpec) -> dict[str, Any]:
    services = service.metrics(spec.id, days=7, now=NOW)
    # Calibration is additive: the pre-existing keys must still be there.
    assert {"retention", "completion", "sampled_events"} <= set(services)
    return services["learner_calibration"]


def _bucket(calibration: dict[str, Any], bucket_range: str) -> dict[str, Any]:
    return next(
        bucket
        for bucket in calibration["buckets"]
        if bucket["range"] == bucket_range
    )


def test_no_calibrated_events_reports_empty_result(
    service: ReviewService, fixture_spec: ReviewSpec
) -> None:
    service.put_spec(fixture_spec)

    calibration = _calibration(service, fixture_spec)

    assert calibration["basis"] == "self_rating"
    assert calibration["samples"] == 0
    assert [bucket["range"] for bucket in calibration["buckets"]] == BUCKET_RANGES
    assert all(bucket["samples"] == 0 for bucket in calibration["buckets"])
    assert all(
        bucket["mean_confidence"] is None
        and bucket["actual_accuracy"] is None
        and bucket["bias"] is None
        and bucket["verdict"] == "insufficient"
        for bucket in calibration["buckets"]
    )
    assert calibration["by_concept"] == []


def test_all_five_buckets_are_always_present_in_order(
    service: ReviewService, fixture_spec: ReviewSpec
) -> None:
    service.put_spec(fixture_spec)
    # 0 and 20 share the lowest range; 40/60/80/100 sit on the upper edges.
    for index, confidence in enumerate((0, 20, 40, 60, 80, 100)):
        _attempt(service, fixture_spec, index=index, confidence=confidence)

    calibration = _calibration(service, fixture_spec)

    assert [bucket["range"] for bucket in calibration["buckets"]] == BUCKET_RANGES
    assert [bucket["samples"] for bucket in calibration["buckets"]] == [2, 1, 1, 1, 1]
    # Every bucket below the sample threshold keeps its measured numbers but is
    # still not allowed to publish a verdict.
    assert all(
        bucket["verdict"] == "insufficient" for bucket in calibration["buckets"]
    )


def test_bucket_stats_are_hand_computed(
    service: ReviewService, fixture_spec: ReviewSpec
) -> None:
    service.put_spec(fixture_spec)
    # Wrong-but-confident: all three rated "again", so self-rated accuracy is 0.
    for index, confidence in enumerate((10, 15, 20)):
        _attempt(service, fixture_spec, index=index, confidence=confidence, rating="again")
    # Right-but-unconfident: all three rated "good".
    for offset, confidence in enumerate((70, 75, 80), start=10):
        _attempt(service, fixture_spec, index=offset, confidence=confidence, rating="good")

    calibration = _calibration(service, fixture_spec)

    over = _bucket(calibration, "0-20")
    assert over["samples"] == 3
    assert over["mean_confidence"] == 15.0
    assert over["actual_accuracy"] == 0.0
    assert over["bias"] == 0.15
    assert over["verdict"] == "overconfident"

    under = _bucket(calibration, "60-80")
    assert under["samples"] == 3
    assert under["mean_confidence"] == 75.0
    assert under["actual_accuracy"] == 1.0
    assert under["bias"] == -0.25
    assert under["verdict"] == "underconfident"

    assert calibration["samples"] == 6
    # One concept, three calibrated events: it is reported with the same stats
    # as the bucket they happen to share.
    (concept,) = calibration["by_concept"]
    assert concept["concept_key"] == fixture_spec.concept_by_id(
        "c-series-resonance"
    ).key
    # The stored key is a hash; the service resolves it to the spec's ids so
    # a human reading /metrics sees something readable.
    assert concept["concept_id"] == "c-series-resonance"
    assert concept["concept_name"] == fixture_spec.concept_by_id(
        "c-series-resonance"
    ).name
    assert concept["samples"] == 6
    assert concept["mean_confidence"] == 45.0
    assert concept["actual_accuracy"] == 0.5
    assert concept["bias"] == -0.05
    assert concept["verdict"] == "calibrated"


def test_sample_threshold_boundary(
    service: ReviewService, fixture_spec: ReviewSpec
) -> None:
    service.put_spec(fixture_spec)
    _attempt(service, fixture_spec, index=0, confidence=50, rating="again")
    _attempt(service, fixture_spec, index=1, confidence=50, rating="again")

    two = _bucket(_calibration(service, fixture_spec), "40-60")
    assert two["samples"] == 2
    assert two["mean_confidence"] == 50.0
    assert two["actual_accuracy"] == 0.0
    assert two["bias"] == 0.5
    assert two["verdict"] == "insufficient"

    _attempt(service, fixture_spec, index=2, confidence=50, rating="again")

    three = _bucket(_calibration(service, fixture_spec), "40-60")
    assert three["samples"] == 3
    assert three["verdict"] == "overconfident"


def test_by_concept_requires_at_least_three_samples(
    service: ReviewService, fixture_spec: ReviewSpec
) -> None:
    service.put_spec(fixture_spec)
    # Two events on resonance: not enough for a per-concept row.
    for index in range(2):
        _attempt(service, fixture_spec, index=index, confidence=90, rating="again")
    # Three on quality: eligible.
    for offset, confidence in enumerate((10, 15, 20), start=10):
        _attempt(
            service,
            fixture_spec,
            index=offset,
            confidence=confidence,
            rating="good",
            item_id=QUALITY_ITEM,
        )

    calibration = _calibration(service, fixture_spec)

    assert calibration["samples"] == 5
    assert [row["concept_key"] for row in calibration["by_concept"]] == [
        fixture_spec.concept_by_id("c-quality-factor").key
    ]
    (quality,) = calibration["by_concept"]
    assert quality["samples"] == 3
    assert quality["mean_confidence"] == 15.0
    assert quality["actual_accuracy"] == 1.0
    assert quality["bias"] == -0.85
    assert quality["verdict"] == "underconfident"


def test_by_concept_is_ordered_by_absolute_bias(
    service: ReviewService, fixture_spec: ReviewSpec
) -> None:
    service.put_spec(fixture_spec)
    # Resonance: +0.8 bias (80% confidence, everything rated again).
    for index, confidence in enumerate((80, 80, 80)):
        _attempt(service, fixture_spec, index=index, confidence=confidence, rating="again")
    # Quality: -0.9 bias (10% confidence, everything rated good) -> larger |bias|.
    for offset, confidence in enumerate((10, 10, 10), start=10):
        _attempt(
            service,
            fixture_spec,
            index=offset,
            confidence=confidence,
            rating="good",
            item_id=QUALITY_ITEM,
        )

    calibration = _calibration(service, fixture_spec)

    assert [row["concept_key"] for row in calibration["by_concept"]] == [
        fixture_spec.concept_by_id("c-quality-factor").key,
        fixture_spec.concept_by_id("c-series-resonance").key,
    ]
    assert [row["bias"] for row in calibration["by_concept"]] == [-0.9, 0.8]
    assert [row["verdict"] for row in calibration["by_concept"]] == [
        "underconfident",
        "overconfident",
    ]


def test_events_without_confidence_are_ignored(
    service: ReviewService, fixture_spec: ReviewSpec
) -> None:
    service.put_spec(fixture_spec)
    for index, confidence in enumerate((25, 30, 35)):
        _attempt(service, fixture_spec, index=index, confidence=confidence, rating="good")
    # Two answered events with no confidence must not enter any denominator.
    for offset in (10, 11):
        _attempt(
            service,
            fixture_spec,
            index=offset,
            confidence=None,
            rating="again",
            item_id=QUALITY_ITEM,
        )

    calibration = _calibration(service, fixture_spec)

    assert calibration["samples"] == 3
    assert sum(bucket["samples"] for bucket in calibration["buckets"]) == 3
    assert _bucket(calibration, "20-40")["samples"] == 3
    # Quality only has un-calibrated events, so it has no row.
    assert [row["concept_key"] for row in calibration["by_concept"]] == [
        fixture_spec.concept_by_id("c-series-resonance").key
    ]


@pytest.mark.parametrize("outside", [-0.1, 100.1])
def test_out_of_range_confidence_is_treated_as_absent(outside: float) -> None:
    """The service rejects these at write time; the aggregate still must not
    invent a bucket for a corrupt row."""
    result = learner_calibration(
        [{"confidence": outside, "rating": "good", "concept_key": "c-x"}]
    )

    assert result["samples"] == 0
    assert all(bucket["samples"] == 0 for bucket in result["buckets"])
    assert result["by_concept"] == []
