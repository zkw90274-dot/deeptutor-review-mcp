from __future__ import annotations

import pytest

from deeptutor.review.fixture import build_fixture_spec
from deeptutor.review.models import ReviewPlan, ReviewSpec
from deeptutor.review.queue import concept_state_map
from deeptutor.review.service import AttemptInput, ReviewService

from conftest import EVENING, NOW, make_multi_item_spec, make_wide_spec

DAY = 86400.0


def test_today_returns_three_items_with_due_metadata(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    today = service.today(spec.id, session_id="today", now=NOW)
    assert len(today["items"]) == 3
    for item in today["items"]:
        assert item["due_at"]
        assert isinstance(item["due_at_ts"], float)
        assert 0.0 <= item["predicted_retrievability"] <= 1.0
        assert item["stem"]
    assert today["budget"] == 3
    assert today["tomorrow"]["date"]


def test_order_does_not_repeat_a_concept_adjacently(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    today = service.today(spec.id, session_id="order", now=NOW)
    concepts = [item["concept_id"] for item in today["items"]]
    assert all(left != right for left, right in zip(concepts, concepts[1:]))


def test_again_is_repeated_once_in_the_same_session(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    service.today(spec.id, session_id="repeat", now=NOW)
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="again",
            session_id="repeat",
            submission_id="repeat-1",
        ),
        now=NOW + 1,
    )
    today = service.today(spec.id, session_id="repeat", now=NOW + 2)
    repeats = [item for item in today["items"] if item["status"] == "repeat"]
    assert len(repeats) == 1
    assert repeats[0]["item_id"] == "item-resonance-choice"
    # A second "again" must not add a second repeat for the same item.
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="again",
            session_id="repeat",
            submission_id="repeat-2",
        ),
        now=NOW + 3,
    )
    again = service.today(spec.id, session_id="repeat", now=NOW + 4)
    assert not [item for item in again["items"] if item["status"] == "repeat"]


def test_budget_is_capped_at_twelve(service: ReviewService) -> None:
    spec = make_wide_spec(due=0, new=30)
    service.put_spec(spec)
    today = service.today(spec.id, session_id="budget", now=NOW)
    assert len(today["items"]) == 12
    assert today["budget"] == 12


def test_backlog_suspends_new_material(service: ReviewService) -> None:
    spec = make_wide_spec(due=20, new=5)
    service.put_spec(spec)
    seed_time = NOW - 10 * DAY
    for index in range(20):
        service.attempt(
            spec.id,
            AttemptInput(
                item_id=f"item-{index}",
                rating="good",
                session_id="seed",
                submission_id=f"seed-{index}",
            ),
            now=seed_time,
        )
    today = service.today(
        spec.id, session_id="backlog", now=NOW, allow_recovery=False
    )
    assert today["backlog"] is True
    assert len(today["items"]) == 12
    assert {item["status"] for item in today["items"]} == {"due"}


def test_tomorrow_forecast_includes_a_failed_item(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    service.today(spec.id, session_id="tomorrow", now=EVENING)
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="again",
            session_id="tomorrow",
            submission_id="tomorrow-1",
        ),
        now=EVENING + 1,
    )
    today = service.today(spec.id, session_id="tomorrow", now=EVENING + 2)
    assert today["tomorrow"]["date"] == "2026-11-07"
    assert today["tomorrow"]["due_items"] >= 1


def test_read_paths_use_the_plan_target_retention(service: ReviewService) -> None:
    spec = build_fixture_spec()
    payload = spec.model_dump(mode="json")
    payload["plan"]["target_retention"] = 0.7
    payload["content_hash"] = ""
    altered = ReviewSpec.model_validate(payload)
    service.put_spec(altered)
    response = service.attempt(
        altered.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="good",
            session_id="retention",
            submission_id="retention-1",
        ),
        now=NOW,
    )
    states = concept_state_map(altered, service.store, now=NOW)
    concept = altered.concept_by_id("c-series-resonance")
    assert states[concept.id].next_review_at == pytest.approx(response["next_due_ts"])


def test_multi_concept_item_is_queued_once(service: ReviewService) -> None:
    spec = build_fixture_spec()
    payload = spec.model_dump(mode="json")
    payload["items"][0]["concept_ids"] = [
        spec.concepts[0].id,
        spec.concepts[1].id,
    ]
    payload["content_hash"] = ""
    altered = ReviewSpec.model_validate(payload)
    service.put_spec(altered)
    today = service.today(altered.id, session_id="multi", now=NOW)
    item_ids = [item["item_id"] for item in today["items"]]
    assert len(item_ids) == len(set(item_ids)) == 3


def test_multi_item_concept_rotates_instead_of_starving(service: ReviewService) -> None:
    spec = make_multi_item_spec(16)
    service.put_spec(spec)
    day0 = service.today(spec.id, session_id="rotate-0", now=NOW)
    first_ids = [item["item_id"] for item in day0["items"]]
    assert len(first_ids) == 12
    for item_id in first_ids:
        service.attempt(
            spec.id,
            AttemptInput(
                item_id=item_id,
                rating="good",
                session_id="rotate-0",
                submission_id=f"rotate-seed-{item_id}",
            ),
            now=NOW,
        )
    # Day 7: the concept is due; the first 8 still appear because all have the
    # same last-shown time, but their views are recorded.
    service.today(
        spec.id, session_id="rotate-7", now=NOW + 7 * DAY, allow_recovery=False
    )
    # Day 14: the tail shown on day 0 but not day 7 must now be selected.
    day14 = service.today(
        spec.id, session_id="rotate-14", now=NOW + 14 * DAY, allow_recovery=False
    )
    selected = {item["item_id"] for item in day14["items"]}
    assert selected & {"item-08", "item-09", "item-10", "item-11"}


def test_empty_spec_explains_itself(service: ReviewService) -> None:
    spec = ReviewSpec(
        id="empty-spec",
        title="空复习",
        plan=ReviewPlan(daily_minutes=10),
        concepts=[],
        items=[],
        sources={},
    )
    service.put_spec(spec)
    today = service.today(spec.id, session_id="empty", now=NOW)
    assert today["items"] == []
    assert today["empty_message"]


def test_weakness_score_uses_the_documented_formula() -> None:
    from types import SimpleNamespace

    from deeptutor.review.queue import weakness_score

    state = SimpleNamespace(
        review_count=3,
        lapse_count=1,
        next_review_at=NOW - 2 * DAY,
        last_scheduled_at=NOW - 3 * DAY,
        last_review_at=NOW - 3 * DAY,
    )
    # mastery 0.5, lapse_rate 0.25, overdue clamped to 1.0
    assert weakness_score(state, 0.5, None, now=NOW) == 0.538


def test_today_exposes_weak_concepts_and_item_weakness(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    today = service.today(spec.id, session_id="weakness", now=NOW)
    assert 1 <= len(today["weak_concepts"]) <= 5
    assert all("weakness" in row for row in today["weak_concepts"])
    assert all(item["weakness"] >= 0.0 for item in today["items"])


def test_weakness_read_does_not_create_session_or_views(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    result = service.weakness(spec.id, now=NOW)
    assert result["spec_id"] == spec.id
    assert result["limit"] == 5
    assert len(result["weak_concepts"]) <= 5
    assert service.store.last_view_map(spec.id) == {}


def test_session_plan_mixed_matches_today_without_recording_views(
    service: ReviewService,
) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    plan = service.session_plan(spec.id, now=NOW)
    assert service.store.last_view_map(spec.id) == {}
    today = service.today(spec.id, session_id="plan-compare", now=NOW)
    assert [row["item_id"] for row in plan["items"]] == [
        row["item_id"] for row in today["items"]
    ]
    assert plan["mode"] == "mixed"
    assert plan["budget"] == today["budget"]


def test_session_plan_review_and_new_modes_filter_pools(
    service: ReviewService,
) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    review = service.session_plan(spec.id, mode="review", now=NOW)
    new = service.session_plan(spec.id, mode="new", now=NOW)
    assert all(row["status"] == "due" for row in review["items"])
    assert all(row["status"] == "new" for row in new["items"])
    assert review["counts"]["new"] == 0
    assert new["counts"]["due"] == 0


def test_session_plan_weak_mode_uses_weakest_concepts(
    service: ReviewService,
) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    service.today(spec.id, session_id="weak-seed", now=NOW)
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="again",
            session_id="weak-seed",
            submission_id="weak-seed-again",
        ),
        now=NOW,
    )
    plan = service.session_plan(spec.id, mode="weak", now=NOW)
    assert plan["items"]
    weak_ids = {
        row["concept_id"]
        for row in plan["weak_concepts"]
        if row["weakness"] > 0.0
    }
    assert all(row["concept_id"] in weak_ids for row in plan["items"])
    scores = [
        next(
            row["weakness"]
            for row in plan["weak_concepts"]
            if row["concept_id"] == item["concept_id"]
        )
        for item in plan["items"]
    ]
    assert scores == sorted(scores, reverse=True)


def test_session_plan_limit_and_minutes_only_shrink(
    service: ReviewService,
) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    limited = service.session_plan(spec.id, limit=1, now=NOW)
    timed = service.session_plan(spec.id, minutes=1, now=NOW)
    assert limited["selected"] <= 1
    assert timed["selected"] <= limited["budget"]
    assert timed["estimated_minutes"] <= spec.plan.daily_minutes
    with pytest.raises(ValueError):
        service.session_plan(spec.id, mode="nope", now=NOW)


def test_again_returns_low_cost_correction(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    service.today(spec.id, session_id="correction", now=NOW)
    response = service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="again",
            session_id="correction",
            submission_id="correction-1",
        ),
        now=NOW + 1,
    )
    correction = response["correction"]
    assert correction["retry_after_items"] == 5
    assert 2 <= len(correction["options"]) <= 3
    assert 0 <= correction["correct_index"] < len(correction["options"])
    assert (
        correction["options"][correction["correct_index"]]["label"]
        == correction["correct_label"]
    )


def test_retry_good_after_again_is_recorded_as_hard(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    service.today(spec.id, session_id="retry-hard", now=NOW)
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="again",
            session_id="retry-hard",
            submission_id="retry-hard-1",
        ),
        now=NOW + 1,
    )
    response = service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="good",
            session_id="retry-hard",
            submission_id="retry-hard-2",
        ),
        now=NOW + 2,
    )
    assert response["retry"] is True
    assert response["requested_rating"] == "good"
    assert response["rating"] == "hard"
    assert response["result"] == "correct"
    assert response["quality"] == 0.55


def test_retry_again_stays_again(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    service.today(spec.id, session_id="retry-again", now=NOW)
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="again",
            session_id="retry-again",
            submission_id="retry-again-1",
        ),
        now=NOW + 1,
    )
    response = service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="again",
            session_id="retry-again",
            submission_id="retry-again-2",
        ),
        now=NOW + 2,
    )
    assert response["retry"] is True
    assert response["rating"] == "again"


def test_repeat_row_reports_actual_retry_gap(service: ReviewService) -> None:
    spec = make_wide_spec(due=8, new=0)
    service.put_spec(spec)
    service.today(spec.id, session_id="gap", now=NOW)
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-0",
            rating="again",
            session_id="gap",
            submission_id="gap-1",
        ),
        now=NOW + 1,
    )
    today = service.today(spec.id, session_id="gap", now=NOW + 2)
    repeats = [item for item in today["items"] if item["status"] == "repeat"]
    assert len(repeats) == 1
    assert repeats[0]["retry_gap"] >= 5


def test_three_day_gap_activates_recovery_cap(service: ReviewService) -> None:
    spec = make_wide_spec(due=20, new=0)
    service.put_spec(spec)
    for index in range(20):
        service.attempt(
            spec.id,
            AttemptInput(
                item_id=f"item-{index}",
                rating="again",
                session_id="recovery",
                submission_id=f"recovery-{index}",
            ),
            now=NOW + index,
        )
    later = NOW + 4 * DAY
    today = service.today(spec.id, session_id="recovery-2", now=later)
    assert today["recovery"]["active"] is True
    assert today["recovery"]["limit"] == 8
    assert today["recovery"]["days_since_last_review"] >= 3.9
    assert len(today["items"]) == 8
    assert not any(item["status"] == "new" for item in today["items"])
    # RS-1005: no backlog number. ``counts.due`` must describe the selection,
    # not the 20-card pool behind it.
    assert today["counts"]["due"] <= 8
    assert "落后" not in str(today)
    assert "streak" not in today


def test_recovery_is_inactive_before_three_days(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    service.today(spec.id, session_id="near", now=NOW)
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="again",
            session_id="near",
            submission_id="near-1",
        ),
        now=NOW + 1,
    )
    today = service.today(spec.id, session_id="near", now=NOW + 2 * DAY)
    assert today["recovery"]["active"] is False
    assert today["recovery"]["limit"] is None


def test_new_spec_has_no_recovery_metadata(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    today = service.today(spec.id, session_id="fresh", now=NOW)
    assert today["recovery"]["active"] is False
    assert today["recovery"]["days_since_last_review"] is None


def test_session_plan_inherits_recovery(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    service.today(spec.id, session_id="recovery-plan", now=NOW)
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="again",
            session_id="recovery-plan",
            submission_id="recovery-plan-1",
        ),
        now=NOW + 1,
    )
    plan = service.session_plan(spec.id, now=NOW + 4 * DAY)
    assert plan["recovery"]["active"] is True
    # 8 is a cap, not a promise: a 3-item spec cannot exceed its own size.
    assert plan["budget"] == min(8, len(spec.items))
    assert plan["selected"] <= 8
    assert plan["counts"]["new"] == 0


def test_session_plan_new_mode_overrides_recovery_pause(service: ReviewService) -> None:
    spec = make_wide_spec(due=3, new=5)
    service.put_spec(spec)
    for index in range(3):
        service.attempt(
            spec.id,
            AttemptInput(
                item_id=f"item-{index}",
                rating="again",
                session_id="recovery-new",
                submission_id=f"recovery-new-{index}",
            ),
            now=NOW + index,
        )
    plan = service.session_plan(spec.id, mode="new", now=NOW + 4 * DAY)
    assert plan["recovery"]["active"] is False
    assert plan["selected"] > 0
    assert plan["counts"]["new"] > 0
