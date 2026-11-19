from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from deeptutor.review.mcp_server import TOOL_CLASSES
from deeptutor.review.models import ReviewSpec
from deeptutor.review.service import AttemptInput
from deeptutor.review.storage import ReviewStore


def _plan(**kwargs):
    result = asyncio.run(TOOL_CLASSES["review_session_plan"]().execute(**kwargs))
    assert result.success, result.content
    return json.loads(result.content)


def _seed(service, fixture_spec, number, *, state="legacy"):
    payload = fixture_spec.model_dump(mode="json")
    payload.update(id=f"pack-{number:02d}", title=f"科目 {number % 3} · 第{number}章", content_hash="")
    spec = ReviewSpec.model_validate(payload)
    service.store.put_spec(spec, assessment=({"state": state}, {}))
    return spec


def test_mixed_session_42_packs_due_first_and_interleaved(service, fixture_spec, monkeypatch):
    specs = [_seed(service, fixture_spec, number) for number in range(42)]
    for spec in specs[:21]:
        service.attempt(spec.id, AttemptInput(item_id=spec.items[0].id, rating="good"), now=1_600_000_000)
    before = service.store.db_path.read_bytes()

    def deny_writes(*args, **kwargs):
        pytest.fail("A planning call attempted a Review write transaction")

    monkeypatch.setattr(ReviewStore, "_transaction", deny_writes)
    result = _plan()
    assert len(result["items"]) == 12
    assert result["selected"] == {"due": 8, "new": 4, "total": 12, "estimated_minutes": 48}
    assert [item["status"] for item in result["items"]] == ["due"] * 8 + ["new"] * 4
    assert result["library_backlog"] == {"due": 21, "new": 105, "total": 126}
    assert len(result["packs"]) == 42
    for offset in (0, 3, 8):
        assert len({item["subject"] for item in result["items"][offset:offset + 3]}) == 3
    assert len({(item["spec_id"], item["item_id"]) for item in result["items"]}) == 12
    assert all({"answer", "explanation", "concept_name", "correction", "session_id"}.isdisjoint(item) for item in result["items"])
    assert service.store.db_path.read_bytes() == before


def test_mixed_session_quality_and_selection_filters(service, fixture_spec):
    legacy, verified, draft, rejected = [
        _seed(service, fixture_spec, number, state=state)
        for number, state in enumerate(("legacy", "verified", "draft", "rejected"))
    ]
    default = _plan()
    assert {row["spec_id"] for row in default["packs"]} == {legacy.id, verified.id}
    result = _plan(spec_ids=[draft.id, rejected.id, "missing", draft.id], include_draft=True)
    assert {item["spec_id"] for item in result["items"]} == {draft.id}
    assert result["library_backlog"]["total"] == 9
    assert result["filtered_backlog"]["total"] == 3
    assert result["unavailable_spec_ids"] == ["missing"]
    assert _plan(spec_ids=[])["items"] == []


@pytest.mark.parametrize("minutes, expected", [(1, 0), (8, 2)])
def test_mixed_session_time_budget(service, fixture_spec, minutes, expected):
    _seed(service, fixture_spec, 0)
    result = _plan(daily_minutes=minutes)
    assert result["selected"]["total"] == expected
    assert result["selected"]["estimated_minutes"] <= minutes
    assert result["library_backlog"]["total"] == 3


def test_mixed_session_empty_home_stays_empty(tmp_path, monkeypatch):
    target = tmp_path / "missing-review"
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(target))
    result = _plan()
    assert result["error"] == "review_store_unavailable"
    assert not target.exists()


def test_mixed_session_reads_mastery_snapshot_without_opening_core_store(
    service, linked_spec, tmp_path, monkeypatch,
):
    from deeptutor.learning.migration import mastery_v2_root
    from deeptutor.learning.models import KnowledgePoint, LearningModule, LearningProgress, RepetitionState
    from deeptutor.learning.storage import LearningStore
    from deeptutor.services import path_service

    spec = _seed(service, linked_spec, 0)
    workspace = tmp_path / "workspace"
    root = mastery_v2_root(workspace / "learning")
    monkeypatch.setattr(path_service, "get_path_service", lambda: SimpleNamespace(get_workspace_dir=lambda: workspace))
    assert _plan()["library_backlog"]["due"] == 0
    assert not root.exists()

    core = LearningStore(root=root)
    first_concept = spec.concepts[0]
    point_id = "preview-point"
    path_id = "preview-path"
    core.save(LearningProgress(
        book_id=path_id,
        modules=[LearningModule(id="m", name="Module", order=1, knowledge_points=[
            KnowledgePoint(id=point_id, name=first_concept.name, type="concept", module_id="m"),
        ])],
        repetition_states={point_id: RepetitionState(next_review_at=1_600_000_000, review_count=1)},
    ))
    service.store.change_link(spec, first_concept.id, (path_id, point_id))
    before = core.db_path.read_bytes()

    def deny_core_init(*args, **kwargs):
        pytest.fail("Planning must not initialize or migrate the Mastery store")

    monkeypatch.setattr(LearningStore, "__init__", deny_core_init)
    result = _plan()
    assert result["library_backlog"]["due"] == len(spec.items_for_concept(first_concept.id))
    assert result["items"][0]["status"] == "due"
    assert core.db_path.read_bytes() == before


def test_mixed_session_reports_backlog_before_pack_cap(service):
    from conftest import make_multi_item_spec

    spec = make_multi_item_spec(20)
    service.store.put_spec(spec, assessment=({"state": "verified"}, {}))
    result = _plan(limit=6)
    assert result["selected"]["total"] == 6
    assert result["packs"][0]["selected"]["total"] == 6
    assert result["library_backlog"] == {"due": 0, "new": 20, "total": 20}


@pytest.mark.parametrize("argument,error", [
    ({"limit": 13}, "invalid_limit"),
    ({"daily_minutes": True}, "invalid_daily_minutes"),
    ({"spec_ids": [""]}, "invalid_spec_ids"),
    ({"include_draft": "true"}, "invalid_include_draft"),
])
def test_mixed_session_validates_arguments(argument, error):
    assert _plan(**argument)["error"] == error
