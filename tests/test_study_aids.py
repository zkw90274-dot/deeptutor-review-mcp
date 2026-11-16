"""Structured study aids: assumptions, self-check and pitfalls.

They belong to the answer panel only. ``review_today`` must never expose them,
because a pitfall or assumption can reveal the answer; ``review_attempt`` and
the offline export release them after the attempt.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import ValidationError

from deeptutor.review.agent_tools import ReviewAttemptTool, ReviewTodayTool
from deeptutor.review.fixture import build_fixture_spec
from deeptutor.review.models import ReviewSpec
from deeptutor.review.service import ReviewService

AIDS_INPUT = {
    "self_check": "核对单位与量纲。",
    "pitfalls": ["单位混用", "漏掉 1/(2π)"],
    "assumptions": ["忽略寄生参数"],
}
AIDS = {**AIDS_INPUT, "table": ""}

TABLE = "| 项目 | 串联 | 并联 |\n|---|---|---|\n| 阻抗 | 最小 | 最大 |"


def _spec_with_aids() -> ReviewSpec:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    for item in payload["items"]:
        item.update(AIDS_INPUT)
    return ReviewSpec.model_validate(payload)


def test_export_payload_carries_study_aids(service: ReviewService) -> None:
    spec = _spec_with_aids()
    service.put_spec(spec)

    html = service.export(spec.id)

    assert '"study_aids"' in html
    for text in ("核对单位与量纲。", "单位混用", "忽略寄生参数"):
        assert text in html


def test_today_withholds_study_aids(service: ReviewService) -> None:
    spec = _spec_with_aids()
    service.put_spec(spec)

    payload = json.loads(
        asyncio.run(ReviewTodayTool().execute(spec_id=spec.id)).content
    )

    assert payload["items"]
    for row in payload["items"]:
        for key in (
            "self_check",
            "pitfalls",
            "assumptions",
            "study_table",
            "study_aids",
        ):
            assert key not in row, key


def test_attempt_releases_study_aids(service: ReviewService) -> None:
    spec = _spec_with_aids()
    service.put_spec(spec)
    today = json.loads(
        asyncio.run(ReviewTodayTool().execute(spec_id=spec.id)).content
    )
    item = today["items"][0]

    attempt = json.loads(
        asyncio.run(
            ReviewAttemptTool().execute(
                spec_id=spec.id,
                item_id=item["item_id"],
                session_id=today["session_id"],
                rating="again",
                submission_id="study-aids-1",
                hints_used=0,
            )
        ).content
    )

    assert attempt["study_aids"] == AIDS


def _spec_with_table() -> ReviewSpec:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    payload["items"][0]["study_table"] = TABLE
    return ReviewSpec.model_validate(payload)


def test_export_payload_carries_a_comparison_table(service: ReviewService) -> None:
    spec = _spec_with_table()
    service.put_spec(spec)

    html = service.export(spec.id)

    assert '"table":"| 项目 | 串联 | 并联 |' in html
    assert "| 阻抗 | 最小 | 最大 |" in html


def test_attempt_releases_a_comparison_table(service: ReviewService) -> None:
    spec = _spec_with_table()
    service.put_spec(spec)
    today = json.loads(
        asyncio.run(ReviewTodayTool().execute(spec_id=spec.id)).content
    )
    item = today["items"][0]
    assert "study_table" not in item

    attempt = json.loads(
        asyncio.run(
            ReviewAttemptTool().execute(
                spec_id=spec.id,
                item_id=item["item_id"],
                session_id=today["session_id"],
                rating="good",
                submission_id="study-table-1",
                hints_used=0,
            )
        ).content
    )

    assert attempt["study_aids"]["table"] == TABLE


def test_study_table_is_bounded() -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    payload["items"][0]["study_table"] = "x" * 4001
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_empty_study_aids_keep_the_legacy_content_hash() -> None:
    original = build_fixture_spec()
    payload = original.model_dump(mode="json")
    payload["content_hash"] = ""

    rebuilt = ReviewSpec.model_validate(payload)

    assert rebuilt.content_hash == original.content_hash


def test_study_aids_are_bounded() -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    payload["items"][0]["self_check"] = "x" * 601
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)

    payload["items"][0]["self_check"] = ""
    payload["items"][0]["pitfalls"] = ["a", "b", "c", "d"]
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)
