"""Read-only aggregates over Review Studio events (RS-1101/RS-1102).

Everything here is a pure function over already-loaded rows. There is no store
handle, no clock read, no environment lookup: the caller supplies ``now`` so a
report can be replayed and a test can pin the window boundary.

Two deliberate choices:

* A bucket below ``MIN_BUCKET_SAMPLES`` reports ``rate=None`` instead of a
  percentage. Four lucky recalls out of four is not "100% retention"; showing
  that number invites a planning decision based on noise.
* The bucketed value is ``predicted_retrievability``, i.e. the scheduler's own
  forecast at review time. It measures calibration, not real-world retention.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

__all__ = [
    "MIN_BUCKET_SAMPLES",
    "retention_buckets",
    "completion_rate",
    "learner_calibration",
]

# Below this a rate is noise, not a measurement. Matches the threshold used by
# the session-plan spec so the API and the docs cannot drift apart.
MIN_BUCKET_SAMPLES = 5

# Calibration needs at least this many events before a verdict is meaningful.
MIN_CALIBRATION_SAMPLES = 3
# A deviation smaller than this from the self-rating is reported as calibrated.
CALIBRATION_BIAS_THRESHOLD = 0.15
# Fixed confidence buckets in percent. The upper edge belongs to the bucket a
# person would read it in (20 is in "0-20", 100 is in "80-100").
_CONFIDENCE_BUCKETS = ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100))
# Only the most miscalibrated concepts are surfaced; a long tail of concepts
# with one or two answers would otherwise flood the report.
MAX_CALIBRATION_CONCEPTS = 3

_BUCKET_COUNT = 10
_SECONDS_PER_DAY = 86_400.0


def _number(value: Any) -> float | None:
    """Return a finite float for a numeric cell, else ``None``.

    ``bool`` is rejected explicitly: ``True`` is an ``int`` in Python and would
    otherwise land in the 0.1 bucket as if it were a forecast.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _bucket_index(value: float) -> int:
    """Map ``value`` in [0, 1] to a bucket index in [0, 9].

    Boundaries are left-closed and right-open, except the last bucket which is
    right-closed so that exactly 1.0 is counted instead of falling off the end.
    Comparisons are done against the same boundary doubles that are reported in
    ``lo``/``hi``, avoiding the ``int(value * 10)`` rounding trap.
    """
    index = 0
    while index < _BUCKET_COUNT - 1 and value >= (index + 1) / _BUCKET_COUNT:
        index += 1
    return index


def _in_window(event: Mapping[str, Any], cutoff: float) -> bool:
    """Window test; rows with a missing/non-numeric ``ts`` are dropped.

    A future timestamp is kept: the contract only defines a lower bound, and
    clock skew must not silently hide events.
    """
    timestamp = _number(event.get("ts"))
    return timestamp is not None and timestamp >= cutoff


def _retrievability(event: Mapping[str, Any]) -> float | None:
    """Return the forecast if it is a finite number inside [0, 1]."""
    value = _number(event.get("predicted_retrievability"))
    if value is None or value < 0.0 or value > 1.0:
        return None
    return value


def retention_buckets(
    events: Sequence[Mapping[str, Any]],
    *,
    now: float,
    days: int = 7,
) -> list[dict[str, Any]]:
    """Bucket predicted retrievability and compute the observed recall rate.

    Args:
        events: Event rows (``review_events`` or equivalent mappings).
        now: Reference timestamp; the window is ``[now - days*86400, +inf)``.
        days: Window length in days. Must be positive.

    Returns:
        Ten buckets ordered by ``lo`` ascending. Each bucket is
        ``{"lo", "hi", "samples", "recalled", "rate", "status"}``. ``rate`` is
        ``None`` and ``status`` is ``"insufficient"`` when ``samples`` is below
        :data:`MIN_BUCKET_SAMPLES`; otherwise ``status`` is ``"ok"`` and
        ``rate`` is ``recalled / samples`` rounded to 4 decimals.

    Raises:
        ValueError: If ``days <= 0``.

    Events outside the window, with a missing/unparseable ``ts``, or with a
    ``predicted_retrievability`` that is not a finite number in [0, 1] are
    dropped before counting. Dropped rows are intentionally not reported in the
    return value: this is a list contract, and a per-bucket counter would make
    the shape unstable for downstream consumers.
    """
    if days <= 0:
        raise ValueError("days must be positive")

    cutoff = float(now) - days * _SECONDS_PER_DAY
    buckets: list[dict[str, Any]] = [
        {
            "lo": round(index / _BUCKET_COUNT, 1),
            "hi": round((index + 1) / _BUCKET_COUNT, 1),
            "samples": 0,
            "recalled": 0,
            "rate": None,
            "status": "insufficient",
        }
        for index in range(_BUCKET_COUNT)
    ]

    for event in events:
        if not _in_window(event, cutoff):
            continue
        value = _retrievability(event)
        if value is None:
            continue
        bucket = buckets[_bucket_index(value)]
        bucket["samples"] += 1
        # Anything other than the exact token "correct" counts as a miss. The
        # schema also carries a three-level rating, but mixing the two scales
        # here would make the denominator ambiguous.
        if event.get("result") == "correct":
            bucket["recalled"] += 1

    for bucket in buckets:
        if bucket["samples"] >= MIN_BUCKET_SAMPLES:
            bucket["status"] = "ok"
            bucket["rate"] = round(bucket["recalled"] / bucket["samples"], 4)
    return buckets


def completion_rate(
    sessions: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarise session completion and how much of the work was first attempts.

    Args:
        sessions: Session rows with ``session_id`` and ``ended_at``.
        events: Answer events belonging to those sessions. The caller is
            responsible for scoping this sequence to the same sessions/window;
            this function does not join by ``session_id``.

    Returns:
        ``{"sessions", "completed", "answered_items", "attempts", "retests",
        "rate"}``. ``completed`` counts sessions whose ``ended_at`` is not
        ``None``. ``answered_items`` counts distinct non-empty ``item_id``
        values, so re-answering the same item does not inflate coverage.
        ``retests`` is ``attempts - answered_items`` floored at zero. ``rate``
        is ``completed / sessions`` rounded to 4 decimals, or ``None`` when
        there are no sessions.
    """
    total_sessions = len(sessions)
    completed = sum(1 for session in sessions if session.get("ended_at") is not None)
    attempts = len(events)
    answered_items = {
        str(event["item_id"])
        for event in events
        if event.get("item_id") not in (None, "")
    }
    answered = len(answered_items)
    return {
        "sessions": total_sessions,
        "completed": completed,
        "answered_items": answered,
        "attempts": attempts,
        "retests": max(0, attempts - answered),
        "rate": round(completed / total_sessions, 4) if total_sessions else None,
    }


def _confidence(event: Mapping[str, Any]) -> float | None:
    """Return a usable confidence in [0, 100], else ``None``.

    A missing column, a non-number, or a value outside the scale is treated as
    "no confidence recorded" rather than being clamped: clamping would invent a
    measurement the learner never gave.
    """
    value = _number(event.get("confidence"))
    if value is None or value < 0.0 or value > 100.0:
        return None
    return value


def _confidence_bucket_index(value: float) -> int:
    """Map a confidence in [0, 100] to its fixed bucket index in [0, 4].

    The printed ranges are read the way a person reads them: ``0-20`` contains
    20, ``20-40`` contains 40, and ``80-100`` contains 100. Buckets are
    therefore right-closed and left-open, with 0 folded into the first bucket.
    This differs from the retention buckets on purpose: those label a [0, 1]
    forecast, where the conventional half-open split avoids a special case at
    the top end, while these label whole percentages and must keep 20 with the
    ``0-20`` range a learner sees in the report.
    """
    if value <= 0.0:
        return 0
    index = int(math.ceil(value / 20.0)) - 1
    return min(index, len(_CONFIDENCE_BUCKETS) - 1)


def _calibration_verdict(bias: float) -> str:
    """Classify a bias (positive = overconfident) into the three verdicts."""
    if bias >= CALIBRATION_BIAS_THRESHOLD:
        return "overconfident"
    if bias <= -CALIBRATION_BIAS_THRESHOLD:
        return "underconfident"
    return "calibrated"


def _calibration_stats(samples: Sequence[tuple[float, str]]) -> dict[str, Any]:
    """Mean confidence, self-rated accuracy, and bias for one group.

    ``samples`` holds ``(confidence, rating)`` pairs where rating is one of
    ``again``/``hard``/``good``. Anything other than ``again`` counts as the
    learner judging their own recall a success. The returned numbers are always
    real (``n > 0``); the caller decides whether ``n`` is large enough for a
    verdict.
    """
    count = len(samples)
    mean_confidence = round(sum(value for value, _ in samples) / count, 4)
    actual_accuracy = round(
        sum(1 for _, rating in samples if rating != "again") / count, 4
    )
    bias = round(mean_confidence / 100.0 - actual_accuracy, 4)
    return {
        "samples": count,
        "mean_confidence": mean_confidence,
        "actual_accuracy": actual_accuracy,
        "bias": bias,
    }


def objective_accuracy(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    judged = [event["correct"] for event in events if event.get("correct") is not None]
    correct = sum(value == 1 for value in judged)
    return {
        "correct": correct,
        "judged": len(judged),
        "accuracy": round(correct / len(judged), 4) if judged else None,
    }


def learner_calibration(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare the learner's confidence with their own self-rating.

    Basis is ``self_rating``, and that caveat is load-bearing: the event stores
    the learner's chosen ``again``/``hard``/``good`` rating, not whether an
    answer was objectively graded correct. ``actual_accuracy`` is therefore the
    share of events the learner did *not* rate ``again`` — it measures whether
    confidence predicts the learner's own verdict, not real answer correctness.
    Answer correctness is computed at attempt time but is not persisted.

    Args:
        events: Event rows (``review_events`` or equivalent mappings). Rows
            without a usable ``confidence`` in [0, 100] are ignored.

    Returns:
        ``{"basis", "samples", "buckets", "by_concept"}``. ``samples`` counts
        the events that carried a confidence. ``buckets`` always contains all
        five ranges in ascending order; a bucket with no samples reports
        ``mean_confidence``/``actual_accuracy``/``bias`` as ``None`` and verdict
        ``"insufficient"``. A bucket with 1-2 samples reports the measured
        values but still ``"insufficient"``. Otherwise the verdict is
        ``"overconfident"`` when ``bias >= 0.15``, ``"underconfident"`` when
        ``bias <= -0.15``, and ``"calibrated"`` in between; ``bias`` is
        ``mean_confidence/100 - actual_accuracy``. ``by_concept`` lists at most
        three concepts with at least :data:`MIN_CALIBRATION_SAMPLES` calibrated
        events, ordered by ``|bias|`` descending (ties by concept key for a
        stable report).
    """
    bucket_samples: list[list[tuple[float, str]]] = [
        [] for _ in _CONFIDENCE_BUCKETS
    ]
    concept_samples: dict[str, list[tuple[float, str]]] = {}
    calibrated_events = 0

    for event in events:
        confidence = _confidence(event)
        if confidence is None:
            continue
        calibrated_events += 1
        # The stored rating is already the effective one: the service may have
        # downgraded a same-session "good" retest to "hard". Using the stored
        # value keeps this aggregate consistent with the scheduler's history.
        rating = str(event.get("rating") or "")
        pair = (confidence, rating)
        bucket_samples[_confidence_bucket_index(confidence)].append(pair)
        concept_key = str(event.get("concept_key") or "")
        if concept_key:
            concept_samples.setdefault(concept_key, []).append(pair)

    buckets: list[dict[str, Any]] = []
    for (low, high), samples in zip(_CONFIDENCE_BUCKETS, bucket_samples):
        bucket: dict[str, Any] = {
            "range": f"{low}-{high}",
            "samples": len(samples),
            "mean_confidence": None,
            "actual_accuracy": None,
            "bias": None,
            "verdict": "insufficient",
        }
        if samples:
            bucket.update(_calibration_stats(samples))
            if len(samples) >= MIN_CALIBRATION_SAMPLES:
                bucket["verdict"] = _calibration_verdict(bucket["bias"])
        buckets.append(bucket)

    by_concept: list[dict[str, Any]] = []
    for concept_key, samples in concept_samples.items():
        if len(samples) < MIN_CALIBRATION_SAMPLES:
            continue
        row: dict[str, Any] = {"concept_key": concept_key}
        row.update(_calibration_stats(samples))
        row["verdict"] = _calibration_verdict(row["bias"])
        by_concept.append(row)
    by_concept.sort(key=lambda row: (-abs(row["bias"]), row["concept_key"]))

    return {
        "basis": "self_rating",
        "samples": calibrated_events,
        "buckets": buckets,
        "by_concept": by_concept[:MAX_CALIBRATION_CONCEPTS],
    }
