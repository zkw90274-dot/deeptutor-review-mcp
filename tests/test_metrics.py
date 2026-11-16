from __future__ import annotations

import copy

import pytest

from deeptutor.review.metrics import (
    MIN_BUCKET_SAMPLES,
    completion_rate,
    retention_buckets,
)

NOW = 1_800_000_000.0
DAY = 86_400.0


def _event(
    retrievability: object,
    *,
    ts: object = NOW,
    result: object = "correct",
    item_id: object = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "predicted_retrievability": retrievability,
        "ts": ts,
        "result": result,
    }
    if item_id is not None:
        event["item_id"] = item_id
    return event


def _bucket(buckets: list[dict], lo: float) -> dict:
    return next(bucket for bucket in buckets if bucket["lo"] == lo)


# -- retention_buckets -------------------------------------------------------


def test_empty_events_return_ten_insufficient_buckets() -> None:
    buckets = retention_buckets([], now=NOW)

    assert len(buckets) == 10
    assert [bucket["lo"] for bucket in buckets] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert [bucket["hi"] for bucket in buckets] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert all(bucket["samples"] == 0 for bucket in buckets)
    assert all(bucket["recalled"] == 0 for bucket in buckets)
    assert all(bucket["rate"] is None for bucket in buckets)
    assert all(bucket["status"] == "insufficient" for bucket in buckets)


def test_bucket_shape_and_order_are_fixed() -> None:
    buckets = retention_buckets([_event(0.35)], now=NOW)

    assert [bucket["lo"] for bucket in buckets] == sorted(bucket["lo"] for bucket in buckets)
    assert set(buckets[0]) == {"lo", "hi", "samples", "recalled", "rate", "status"}
    assert all(isinstance(bucket["lo"], float) for bucket in buckets)
    assert all(isinstance(bucket["hi"], float) for bucket in buckets)


def test_zero_and_one_land_in_edge_buckets() -> None:
    buckets = retention_buckets(
        [_event(0.0), _event(1.0)],
        now=NOW,
    )

    assert _bucket(buckets, 0.0)["samples"] == 1
    assert _bucket(buckets, 0.9)["samples"] == 1
    assert sum(bucket["samples"] for bucket in buckets) == 2


def test_upper_boundaries_are_left_closed() -> None:
    values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    buckets = retention_buckets([_event(value) for value in values], now=NOW)

    for value in values:
        assert _bucket(buckets, round(value, 1))["samples"] == 1


def test_single_sample_is_insufficient_not_a_percentage() -> None:
    buckets = retention_buckets([_event(0.45)], now=NOW)

    bucket = _bucket(buckets, 0.4)
    assert bucket["samples"] == 1
    assert bucket["recalled"] == 1
    assert bucket["rate"] is None
    assert bucket["status"] == "insufficient"


def test_out_of_range_and_non_numeric_values_are_dropped() -> None:
    events = [
        _event(-0.1),
        _event(1.0001),
        _event(None),
        _event("0.5"),
        _event(float("nan")),
        _event(float("inf")),
        _event(True),
        _event([]),
        {"ts": NOW, "result": "correct"},
    ]

    buckets = retention_buckets(events, now=NOW)

    assert sum(bucket["samples"] for bucket in buckets) == 0


def test_window_excludes_events_older_than_cutoff() -> None:
    events = [
        _event(0.5, ts=NOW - 7 * DAY - 1),
        _event(0.5, ts=NOW - 7 * DAY + 1),
    ]

    bucket = _bucket(retention_buckets(events, now=NOW, days=7), 0.5)

    assert bucket["samples"] == 1


def test_window_cutoff_boundary_is_inclusive() -> None:
    events = [_event(0.5, ts=NOW - 7 * DAY)]

    bucket = _bucket(retention_buckets(events, now=NOW, days=7), 0.5)

    assert bucket["samples"] == 1


def test_days_parameter_controls_window() -> None:
    events = [_event(0.5, ts=NOW - 2 * DAY)]

    assert _bucket(retention_buckets(events, now=NOW, days=3), 0.5)["samples"] == 1
    assert _bucket(retention_buckets(events, now=NOW, days=1), 0.5)["samples"] == 0


def test_future_events_are_kept() -> None:
    events = [_event(0.5, ts=NOW + DAY)]

    assert _bucket(retention_buckets(events, now=NOW), 0.5)["samples"] == 1


def test_five_samples_turn_bucket_ok() -> None:
    events = [_event(0.35)] * (MIN_BUCKET_SAMPLES - 1)
    assert _bucket(retention_buckets(events, now=NOW), 0.3)["status"] == "insufficient"

    events.append(_event(0.35))
    bucket = _bucket(retention_buckets(events, now=NOW), 0.3)

    assert bucket["samples"] == MIN_BUCKET_SAMPLES
    assert bucket["recalled"] == MIN_BUCKET_SAMPLES
    assert bucket["rate"] == 1.0
    assert bucket["status"] == "ok"


def test_only_exact_correct_result_counts_as_recalled() -> None:
    events = [
        _event(0.5, result="correct"),
        _event(0.5, result="again"),
        _event(0.5, result="hard"),
        _event(0.5, result="incorrect"),
        _event(0.5, result=None),
        _event(0.5),
    ]
    events[-1].pop("result")

    bucket = _bucket(retention_buckets(events, now=NOW), 0.5)

    assert bucket["samples"] == 6
    assert bucket["recalled"] == 1
    assert bucket["rate"] == round(1 / 6, 4)


def test_rate_is_rounded_to_four_decimals() -> None:
    events = [_event(0.5)] + [_event(0.5, result="again")] * 6

    bucket = _bucket(retention_buckets(events, now=NOW), 0.5)

    assert bucket["samples"] == 7
    assert bucket["recalled"] == 1
    assert bucket["rate"] == 0.1429


def test_order_independent_and_repeatable() -> None:
    events = [
        _event(0.05),
        _event(0.95, result="again"),
        _event(0.55),
        _event(0.15, result="hard"),
    ]

    forward = retention_buckets(events, now=NOW)
    backward = retention_buckets(list(reversed(events)), now=NOW)
    again = retention_buckets(events, now=NOW)

    assert forward == backward == again


def test_inputs_are_not_mutated() -> None:
    events = [_event(0.5, item_id="a"), _event(0.1, result="again", item_id="a")]
    snapshot = copy.deepcopy(events)

    retention_buckets(events, now=NOW)

    assert events == snapshot


@pytest.mark.parametrize("days", [0, -1])
def test_non_positive_days_raise(days: int) -> None:
    with pytest.raises(ValueError):
        retention_buckets([], now=NOW, days=days)


# -- completion_rate ---------------------------------------------------------


def test_completion_with_no_sessions() -> None:
    assert completion_rate([], []) == {
        "sessions": 0,
        "completed": 0,
        "answered_items": 0,
        "attempts": 0,
        "retests": 0,
        "rate": None,
    }


def test_completion_all_sessions_finished() -> None:
    sessions = [
        {"session_id": "s1", "ended_at": 10.0},
        {"session_id": "s2", "ended_at": 20.0},
    ]

    summary = completion_rate(sessions, [])

    assert summary["sessions"] == 2
    assert summary["completed"] == 2
    assert summary["rate"] == 1.0


def test_completion_partial_is_rounded() -> None:
    sessions = [
        {"session_id": "s1", "ended_at": 10.0},
        {"session_id": "s2", "ended_at": 20.0},
        {"session_id": "s3", "ended_at": None},
    ]

    summary = completion_rate(sessions, [])

    assert summary["sessions"] == 3
    assert summary["completed"] == 2
    assert summary["rate"] == 0.6667


def test_completion_counts_retests_from_duplicate_items() -> None:
    events = [
        {"session_id": "s1", "item_id": "a"},
        {"session_id": "s1", "item_id": "a"},
        {"session_id": "s1", "item_id": "b"},
    ]

    summary = completion_rate([], events)

    assert summary["attempts"] == 3
    assert summary["answered_items"] == 2
    assert summary["retests"] == 1


def test_completion_retests_never_negative() -> None:
    events = [{"session_id": "s1", "item_id": "a"}]

    summary = completion_rate([], events)

    assert summary["attempts"] == 1
    assert summary["answered_items"] == 1
    assert summary["retests"] == 0


def test_completion_ignores_events_without_item_id() -> None:
    events = [
        {"session_id": "s1", "item_id": "a"},
        {"session_id": "s1", "item_id": ""},
        {"session_id": "s1"},
    ]

    summary = completion_rate([], events)

    assert summary["attempts"] == 3
    assert summary["answered_items"] == 1
    assert summary["retests"] == 2


def test_completion_inputs_are_not_mutated() -> None:
    sessions = [{"session_id": "s1", "ended_at": None}]
    events = [{"session_id": "s1", "item_id": "a"}]
    session_snapshot = copy.deepcopy(sessions)
    event_snapshot = copy.deepcopy(events)

    completion_rate(sessions, events)

    assert sessions == session_snapshot
    assert events == event_snapshot
