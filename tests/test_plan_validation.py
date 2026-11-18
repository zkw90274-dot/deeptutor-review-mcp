"""``review_plan`` argument validation: a bad argument is a correctable miss.

``review_plan`` merges seven actions and twelve optional parameters into one
schema whose only required field is ``action``, so a client can send a value
the schema cannot reject per action. Before this file every such value reached
``int(...)`` / ``date.fromisoformat(...)`` and came back as ``success=False``
plus "the store may be unavailable" — a dead end, because the real problem was
a string instead of an integer.

The contract these tests pin: invalid arguments return the *soft* shape
(``success=True`` + ``error`` + ``instruction``) and never touch the store, so
the model can fix the argument and retry inside the same turn.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from deeptutor.review.agent_tools import ReviewPlanTool


def _call(**kwargs):
    result = asyncio.run(ReviewPlanTool().execute(**kwargs))
    return result, json.loads(result.content)


@pytest.mark.parametrize(
    ("label", "kwargs", "fragment"),
    [
        (
            "create/daily_minutes is a string",
            {
                "action": "create",
                "subject": "模拟电路",
                "chapters": ["第1章_绪论"],
                "end_date": "2026-12-31",
                "daily_minutes": "abc",
            },
            "daily_minutes must be an integer",
        ),
        (
            "create/daily_minutes below the advertised minimum",
            {
                "action": "create",
                "subject": "模拟电路",
                "chapters": ["第1章_绪论"],
                "end_date": "2026-12-31",
                "daily_minutes": 5,
            },
            "daily_minutes must be between 15 and 240",
        ),
        (
            "create/end_date is not a date",
            {
                "action": "create",
                "subject": "模拟电路",
                "chapters": ["第1章_绪论"],
                "end_date": "2026-13-40",
            },
            "end_date must be a date",
        ),
        (
            "create/start_date is not a date",
            {
                "action": "create",
                "subject": "模拟电路",
                "chapters": ["第1章_绪论"],
                "start_date": "tomorrow",
                "end_date": "2026-12-31",
            },
            "start_date must be a date",
        ),
        (
            "complete/seq is a string",
            {
                "action": "complete",
                "plan_id": "no-such-plan",
                "date": "2026-11-18",
                "seq": "abc",
            },
            "seq must be an integer",
        ),
        (
            "complete/date is not a date",
            {
                "action": "complete",
                "plan_id": "no-such-plan",
                "date": "2026-99-99",
                "seq": 1,
            },
            "date must be a date",
        ),
        (
            "replan/end_date is not a date",
            {"action": "replan", "plan_id": "no-such-plan", "end_date": "2026-99-99"},
            "end_date must be a date",
        ),
        (
            "replan/daily_minutes is a string",
            {"action": "replan", "plan_id": "no-such-plan", "daily_minutes": "abc"},
            "daily_minutes must be an integer",
        ),
        (
            "today/date is not a date",
            {"action": "today", "date": "2026-99-99"},
            "date must be a date",
        ),
    ],
)
def test_invalid_arguments_soft_fail(label, kwargs, fragment) -> None:
    result, payload = _call(**kwargs)
    assert result.success is True, f"{label}: an argument typo must stay recoverable"
    assert fragment in payload["error"], f"{label}: {payload['error']!r}"
    assert payload["instruction"], f"{label}: the model needs a next move"


def test_invalid_arguments_do_not_create_a_plan() -> None:
    """A rejected argument must not have reached ``create_plan``."""
    from deeptutor.review.storage import ReviewStore

    _, payload = _call(
        action="create",
        subject="模拟电路",
        chapters=["第1章_绪论"],
        end_date="2026-12-31",
        daily_minutes="60",
    )
    assert "error" in payload
    assert ReviewStore().list_plans() == []


def test_bool_is_not_an_integer() -> None:
    """``True`` is a JSON boolean, not the 1 the schema declares."""
    _, payload = _call(
        action="create",
        subject="模拟电路",
        chapters=["第1章_绪论"],
        end_date="2026-12-31",
        daily_minutes=True,
    )
    assert "daily_minutes must be an integer" in payload["error"]


def test_unknown_action_still_soft_fails() -> None:
    result, payload = _call(action="explode")
    assert result.success is True
    assert "unknown action" in payload["error"]
    assert "create" in payload["instruction"]


@pytest.mark.parametrize("text", ["20260118", "2026-W03-7", "2026-1-8", "2026-02-29"])
@pytest.mark.parametrize(
    ("action", "field", "other"),
    [
        ("today", "date", {}),
        ("complete", "date", {"plan_id": "missing", "seq": 1}),
        ("replan", "end_date", {"plan_id": "missing"}),
        ("create", "start_date", {"subject": "Demo", "chapters": ["Chapter"], "end_date": "2026-12-31"}),
    ],
)
def test_plan_dates_require_calendar_format(text, action, field, other) -> None:
    result, payload = _call(action=action, **{field: text}, **other)
    assert result.success is True
    assert "YYYY-MM-DD" in payload["error"]


def test_service_rejects_noncanonical_lookup_dates(store) -> None:
    from deeptutor.review.plan_service import PlanError, plan_today

    with pytest.raises(PlanError, match="YYYY-MM-DD"):
        plan_today(store, on_date="20260118")


@pytest.mark.parametrize("chapters", [5, [5], "第1章"])
def test_create_rejects_non_string_chapter_list(chapters) -> None:
    result, payload = _call(action="create", subject="Demo", chapters=chapters, end_date="2026-12-31")
    assert result.success is True
    assert "chapters" in payload["error"]
