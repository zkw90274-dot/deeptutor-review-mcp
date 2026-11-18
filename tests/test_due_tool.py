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
from deeptutor.review.agent_tools import ReviewDueTool
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


def test_damaged_pack_does_not_hide_the_rest(
    seeded: ReviewService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One unreadable row must be skipped, not turned into a failure.

    The tool resolves its own service per call, so the patch has to land on the
    store class rather than on the fixture's instance.
    """
    from deeptutor.review.storage import ReviewStore, SpecIntegrityError

    real_get_spec = ReviewStore.get_spec

    def flaky(self, spec_id: str):
        if spec_id == SPEC_ID:
            raise SpecIntegrityError("damaged")
        return real_get_spec(self, spec_id)

    monkeypatch.setattr(ReviewStore, "get_spec", flaky)
    payload = _call()
    assert payload["count"] == 0
    assert payload["instruction"]
