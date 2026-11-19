"""``review_due``: the read-only cross-pack view that closes the P0-1 gap.

The tool exists because the most common question — "what should I review
today?" — otherwise costs one ``review_today`` per pack, and in normal mode
every one of those calls opens or reuses a session and records views. These
tests pin both halves of the contract:

* the answer is correct and agrees with what ``review_today`` would present;
* the call is a true read (session / view / event counts unchanged).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from deeptutor.review import service as service_module
from deeptutor.review.agent_tools import ReviewDueTool, ReviewSpecsTool
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.service import ReviewService


def _call(**kwargs):
    result = asyncio.run(ReviewDueTool().execute(**kwargs))
    assert result.success is True, result.content
    payload = json.loads(result.content)
    assert result.metadata["review_due"] == payload
    return payload


def _counts() -> dict[str, int]:
    from deeptutor.review.storage import ReviewStore

    store = ReviewStore()
    with sqlite3.connect(store.db_path) as conn:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("review_sessions", "review_views", "review_events")
        }


@pytest.fixture()
def seeded() -> ReviewService:
    service = ReviewService()
    service.put_spec(build_fixture_spec())
    return service


def test_lists_a_pack_with_queued_items(seeded: ReviewService) -> None:
    payload = _call()
    assert payload["count"] == 1
    row = payload["packs"][0]
    assert row["spec_id"] == SPEC_ID
    assert row["queued"] > 0
    assert row["queued"] == row["due"] + row["new"] + row["repeat"]
    assert payload["totals"]["queued"] == row["queued"]
    assert payload["instruction"]


def test_counts_agree_with_review_today(seeded: ReviewService) -> None:
    """``queued`` must be exactly what ``review_today`` would hand the learner."""
    overview = _call()
    today = seeded.today(SPEC_ID, read_only=True)
    assert overview["packs"][0]["queued"] == len(today["items"])


def test_call_writes_nothing(seeded: ReviewService) -> None:
    """The whole point: no session, no view, no event."""
    before = _counts()
    _call()
    _call()
    assert _counts() == before


def test_subject_filter_excludes_other_packs(seeded: ReviewService) -> None:
    assert _call(subject="no such subject")["count"] == 0
    assert _call(subject="示例")["count"] == 1


def test_empty_library_returns_an_honest_zero() -> None:
    payload = _call()
    assert payload["count"] == 0
    assert payload["packs"] == []
    assert "Nothing is queued" in payload["instruction"]


def test_paginated_tools_cover_a_large_library(seeded: ReviewService) -> None:
    source = build_fixture_spec()
    with seeded.store._transaction() as conn:
        for index in range(500):
            key = f"sample-{index:03d}"
            pack = source.model_copy(update={"id": key, "content_hash": ""})
            conn.execute("INSERT INTO review_specs VALUES (?, ?, ?, ?, ?)", (key, "", 0, index, pack.model_dump_json()))

    for tool, field, identifier in ((ReviewSpecsTool(), "specs", "id"), (ReviewDueTool(), "packs", "spec_id")):
        options = {"include_draft": True} if field == "packs" else {}
        first = json.loads(asyncio.run(tool.execute(**options)).content)
        assert first["count"] == 501
        assert len(first[field]) == 50
        assert first["next_offset"] == 50
        ids = []
        for offset in range(0, 501, 100):
            response = json.loads(asyncio.run(tool.execute(offset=offset, limit=100, **options)).content)
            assert response["count"] == 501
            assert len(response[field]) == min(100, 501 - offset)
            ids.extend(row[identifier] for row in response[field])
            if field == "packs":
                assert response["totals"]["queued"] == 501 * 3
        assert len(set(ids)) == 501
        assert response["next_offset"] is None
        assert _counts() == {"review_sessions": 0, "review_views": 0, "review_events": 0}


def test_damaged_pack_does_not_hide_the_rest(seeded: ReviewService) -> None:
    with seeded.store._transaction() as conn:
        conn.execute("UPDATE review_specs SET spec_json = '{}' WHERE spec_id = ?", (SPEC_ID,))
    payload = _call()
    assert payload["count"] == 0
    assert payload["omitted"] == 1
    assert payload["instruction"]
