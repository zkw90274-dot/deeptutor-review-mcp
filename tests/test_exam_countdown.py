"""Exam countdown: the daily queue reacts to ``plan.exam_date``.

Budget scaling is deliberately minute-only. The 12-item hard cap is a queue
invariant, so exam mode changes ``planned_minutes`` / ``estimated_minutes``
and, in the final week, a tiebreak in the review ordering.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from deeptutor.review.fixture import build_fixture_spec
from deeptutor.review.models import ReviewSpec
from deeptutor.review.service import AttemptInput, ReviewService
from deeptutor.review.timeutil import local_date, zone

from conftest import NOW, make_wide_spec

DAY = 86400.0


def _with_exam_days(
    spec: ReviewSpec,
    days_from_now: int | None,
    *,
    daily_minutes: int | None = None,
) -> ReviewSpec:
    """Return a spec copy whose exam date is ``days_from_now`` local days out."""
    payload = spec.model_dump(mode="json")
    # The plan is part of the content hash; clear the stale one so the
    # validator recomputes it instead of rejecting the modified spec.
    payload["content_hash"] = ""
    if daily_minutes is not None:
        payload["plan"]["daily_minutes"] = daily_minutes
    if days_from_now is None:
        payload["plan"]["exam_date"] = None
    else:
        exam = local_date(NOW, zone(payload["plan"]["timezone"])) + timedelta(
            days=days_from_now
        )
        payload["plan"]["exam_date"] = exam.isoformat()
    return ReviewSpec.model_validate(payload)


def test_no_exam_date_is_normal_and_keeps_the_base_budget(
    service: ReviewService,
) -> None:
    spec = _with_exam_days(build_fixture_spec(), None)
    service.put_spec(spec)
    today = service.today(spec.id, session_id="no-exam", now=NOW)
    assert today["exam"] == {"exam_date": None, "days_left": None, "mode": "normal"}
    assert today["planned_minutes"] == 10
    assert today["estimated_minutes"] == 10
    assert today["budget"] == 3


def test_fourteen_days_out_is_sprint_with_scaled_budget(
    service: ReviewService,
) -> None:
    spec = _with_exam_days(build_fixture_spec(), 14)
    service.put_spec(spec)
    today = service.today(spec.id, session_id="sprint", now=NOW)
    assert today["exam"]["mode"] == "sprint"
    assert today["exam"]["days_left"] == 14
    # min(30, round(10 * 1.5)) == 15
    assert today["planned_minutes"] == 15
    assert today["estimated_minutes"] == 15


def test_three_days_out_is_final_week_with_scaled_budget(
    service: ReviewService,
) -> None:
    spec = _with_exam_days(build_fixture_spec(), 3)
    service.put_spec(spec)
    today = service.today(spec.id, session_id="final-week", now=NOW)
    assert today["exam"]["mode"] == "final_week"
    assert today["exam"]["days_left"] == 3
    # min(45, round(10 * 2)) == 20
    assert today["planned_minutes"] == 20
    assert today["estimated_minutes"] == 20


def test_exam_yesterday_is_normal_with_negative_days_left(
    service: ReviewService,
) -> None:
    spec = _with_exam_days(build_fixture_spec(), -1)
    service.put_spec(spec)
    today = service.today(spec.id, session_id="past-exam", now=NOW)
    assert today["exam"]["mode"] == "normal"
    assert today["exam"]["days_left"] == -1
    assert today["planned_minutes"] == 10


@pytest.mark.parametrize(
    ("days_from_now", "mode", "planned"),
    [
        (0, "final_week", 20),
        (3, "final_week", 20),
        (4, "sprint", 15),
        (14, "sprint", 15),
        (15, "normal", 10),
    ],
)
def test_mode_boundaries_and_days_left_are_exact(
    service: ReviewService, days_from_now: int, mode: str, planned: int
) -> None:
    spec = _with_exam_days(build_fixture_spec(), days_from_now)
    service.put_spec(spec)
    today = service.today(
        spec.id, session_id=f"boundary-{days_from_now}", now=NOW
    )
    assert today["exam"]["days_left"] == days_from_now
    assert today["exam"]["mode"] == mode
    assert today["planned_minutes"] == planned


def test_days_left_is_counted_in_the_plan_timezone(
    service: ReviewService,
) -> None:
    # 2026-11-06 23:30 UTC is already 2026-11-07 in Asia/Shanghai, so an exam
    # on 2026-11-10 is three local days away even though UTC says four.
    late_utc = datetime(2026, 11, 6, 23, 30, tzinfo=timezone.utc).timestamp()
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    payload["plan"]["exam_date"] = "2026-11-10"
    spec = ReviewSpec.model_validate(payload)
    service.put_spec(spec)
    today = service.today(spec.id, session_id="tz", now=late_utc)
    assert local_date(late_utc, ZoneInfo("Asia/Shanghai")).isoformat() == "2026-11-07"
    assert today["exam"]["days_left"] == 3
    assert today["exam"]["mode"] == "final_week"


def test_budget_formula_caps_at_the_ceiling(service: ReviewService) -> None:
    sprint = _with_exam_days(build_fixture_spec(), 10, daily_minutes=40)
    service.put_spec(sprint)
    # min(30, round(40 * 1.5) == 60) == 30
    assert service.today(sprint.id, session_id="cap-sprint", now=NOW)[
        "planned_minutes"
    ] == 30

    final = _with_exam_days(build_fixture_spec(), 2, daily_minutes=40)
    service.put_spec(final)
    # min(45, round(40 * 2) == 80) == 45
    assert service.today(final.id, session_id="cap-final", now=NOW)[
        "planned_minutes"
    ] == 45


def test_final_week_orders_weaker_concept_first_when_tied(
    service: ReviewService,
) -> None:
    """item-1 belongs to c-due-1 (lexicographically later) but is weaker.

    Both items are seeded at the same instant, so ``last_views`` ties. Without
    the final-week weakness tiebreak, ``concept_id`` would put item-0 first.
    """
    spec = _with_exam_days(make_wide_spec(due=2, new=0), 2)
    service.put_spec(spec)
    seed_time = NOW - 100 * DAY
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-0",
            rating="good",
            session_id="seed",
            submission_id="seed-0",
        ),
        now=seed_time,
    )
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-1",
            rating="again",
            session_id="seed",
            submission_id="seed-1",
        ),
        now=seed_time,
    )
    today = service.today(
        spec.id, session_id="final-order", now=NOW, allow_recovery=False
    )
    weaknesses = {
        row["concept_id"]: row["weakness"] for row in today["weak_concepts"]
    }
    assert weaknesses["c-due-1"] > weaknesses["c-due-0"]
    assert [item["item_id"] for item in today["items"]] == ["item-1", "item-0"]
