from __future__ import annotations

import asyncio
import json
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from deeptutor.review import core_sync, materials, plan_service, study_plan
from deeptutor.review.agent_tools import ReviewPlanTool
from deeptutor.review.storage import ReviewStore
from deeptutor.review.study_plan import ChapterWork, Part, PlanError, build_schedule


def _work() -> list[ChapterWork]:
    return [ChapterWork("第1章_绪论", 120), ChapterWork("第2章_选频网络", 180)]


def test_schedule_packs_sessions_and_milestones() -> None:
    result = build_schedule(_work(), date(2026, 11, 11), date(2026, 11, 20), 60)
    assert result["feasible"] is True
    tasks = result["tasks"]
    assert any(task["kind"] == "milestone" and task["title"].startswith("开始：") for task in tasks)
    assert any(task["kind"] == "milestone" and task["title"].startswith("完成：") for task in tasks)
    study = [task for task in tasks if task["kind"] == "study"]
    assert study
    assert all(task["minutes"] > 0 for task in study)
    assert sum(task["minutes"] for task in study) == 300
    # No day gets more new material than the 60% study share of 60 minutes.
    per_day: dict[str, int] = {}
    for task in study:
        per_day[task["local_date"]] = per_day.get(task["local_date"], 0) + task["minutes"]
    assert max(per_day.values()) <= result["study_minutes_per_day"]


def test_infeasible_schedule_reports_the_gap() -> None:
    result = build_schedule([ChapterWork("第1章", 600)], date(2026, 11, 11), date(2026, 11, 12), 60)
    assert result["feasible"] is False
    assert result["required_minutes"] > result["capacity_minutes"]
    assert result["suggested_end_date"] > "2026-11-12"
    assert result["tasks"] == []


def test_schedule_rejects_bad_ranges() -> None:
    with pytest.raises(PlanError):
        build_schedule(_work(), date(2026, 11, 12), date(2026, 11, 11), 60)
    with pytest.raises(PlanError):
        build_schedule(_work(), date(2026, 11, 11), date(2026, 11, 20), 5)
    assert study_plan.estimate_minutes(10) == study_plan.MIN_SESSION_MINUTES


def test_storage_round_trips_plans_tasks_and_notes(tmp_path) -> None:
    store = ReviewStore(tmp_path / "review")
    plan = store.create_plan(
        {"plan_id": "plan_test", "subject": "高频", "status": "active", "start_date": "2026-11-11"},
        now=1.0,
    )
    assert plan["plan_id"] == "plan_test"
    store.replace_tasks(
        "plan_test",
        [
            {"local_date": "2026-11-11", "seq": 1, "kind": "study", "title": "第1章", "minutes": 40},
        ],
    )
    tasks = store.list_tasks("plan_test", local_date="2026-11-11")
    assert [task["title"] for task in tasks] == ["第1章"]
    updated = store.update_task("plan_test", "2026-11-11", 1, {"status": "done", "self_note": "x"})
    assert updated is not None and updated["status"] == "done" and updated["self_note"] == "x"

    note = store.create_note(
        {"note_id": "note_test", "plan_id": "plan_test", "chapter": "第1章", "title": "t", "body": "b"},
        now=2.0,
    )
    assert note["status"] == "draft"
    confirmed = store.update_note("note_test", {"status": "confirmed", "document_id": "doc1"}, now=3.0)
    assert confirmed is not None and confirmed["document_id"] == "doc1"


def test_create_plan_and_complete_draft_a_note(tmp_path, monkeypatch) -> None:
    store = ReviewStore(tmp_path / "review")
    monkeypatch.setattr(
        plan_service,
        "chapter_work",
        lambda subject, chapters: [ChapterWork("第1章_绪论", 30)],
    )
    monkeypatch.setattr(
        plan_service, "attach_parts", lambda subject, work, per_day: work
    )
    result = plan_service.create_plan(
        store,
        subject="高频",
        chapters=["第1章_绪论"],
        start_date="2026-11-11",
        end_date="2026-11-12",
        daily_minutes=60,
        sync=False,
    )
    assert result["ok"] is True
    plan_id = result["plan"]["plan_id"]
    study = [task for task in result["tasks"] if task["kind"] == "study"]
    assert study and study[0]["minutes"] == 30

    done = plan_service.complete_task(
        store,
        plan_id=plan_id,
        local_date=study[0]["local_date"],
        seq=study[0]["seq"],
        self_note="记住了",
        sync=False,
    )
    assert done["ok"] is True
    note = done["note"]
    assert note is not None and note["status"] == "draft"
    assert "第1章_绪论" in note["body"]


def test_confirm_note_writes_through_the_notebook_adapter(tmp_path, monkeypatch) -> None:
    store = ReviewStore(tmp_path / "review")
    store.create_plan(
        {"plan_id": "p1", "subject": "高频", "status": "active", "start_date": "2026-11-11"},
        now=1.0,
    )
    store.create_note(
        {"note_id": "n1", "plan_id": "p1", "chapter": "第1章", "title": "t", "body": "b"},
        now=2.0,
    )
    monkeypatch.setattr(core_sync, "ensure_subject_doc", lambda plan: {"ok": True, "id": "parent"})
    monkeypatch.setattr(
        core_sync,
        "create_note_doc",
        lambda plan, chapter, title, body, parent_id: {"ok": True, "id": "doc9"},
    )
    result = plan_service.confirm_note(store, note_id="n1")
    assert result["ok"] is True
    assert store.get_note("n1")["document_id"] == "doc9"


def test_chapter_sources_list_the_raw_files(tmp_path, monkeypatch) -> None:
    # Deterministic fixture: the real KB is not part of the test contract and
    # the suite may run with a temporary workspace.
    from deeptutor.review import generate

    raw = tmp_path / "高频" / "raw"
    raw.mkdir(parents=True)
    for name in ("001_第1章_绪论.md", "002_第1章_补充.md", "003_第2章_选频网络.md"):
        (raw / name).write_text("x", encoding="utf-8")
    monkeypatch.setattr(generate, "_knowledge_bases_root", lambda: tmp_path)

    names = generate.chapter_sources("高频", "第1章_绪论")
    assert names == ["001_第1章_绪论.md", "002_第1章_补充.md"]
    assert generate.chapter_sources("高频", "不存在的章") == []


def test_plan_today_returns_the_day_tasks(tmp_path, monkeypatch) -> None:
    # Regression: ``plan_today`` compared a date object with the stored ISO
    # strings and raised TypeError, which the MCP client saw as a non-JSON
    # failure (empty task list) rather than a clear error.
    store = ReviewStore(tmp_path / "review")
    monkeypatch.setattr(
        plan_service, "chapter_work", lambda subject, chapters: [ChapterWork("第1章", 30)]
    )
    created = plan_service.create_plan(
        store,
        subject="高频",
        chapters=["第1章"],
        start_date="2026-11-11",
        end_date="2026-11-12",
        sync=False,
    )
    today = plan_service.plan_today(
        store, plan_id=created["plan"]["plan_id"], on_date="2026-11-11"
    )
    day = today["days"][0]
    assert day["date"] == "2026-11-11"
    assert any(task["kind"] == "study" for task in day["tasks"])
    # Review due is computed read-only; with no matching packs it is an empty
    # but well-formed block (never a missing key the model has to guess about).
    assert day["review_due"] == {"items": 0, "specs": []}
    assert day["review_minutes"] >= 0


def test_status_and_sync_calendar_are_exposed() -> None:
    from deeptutor.review.agent_tools import ReviewPlanTool

    actions = ReviewPlanTool().get_definition().parameters[0].enum
    assert actions == list(ReviewPlanTool.ACTIONS)
    assert "sync_calendar" in actions


def test_sync_calendar_moves_a_dragged_day(tmp_path, monkeypatch) -> None:
    # A learner drags the day's block on the calendar; the plan has to follow,
    # and the task board card has to move with it.
    monkeypatch.setattr(plan_service, "local_today", lambda tz: date(2026, 11, 11))
    store = ReviewStore(tmp_path / "review")
    monkeypatch.setattr(
        plan_service, "chapter_work", lambda subject, chapters: [ChapterWork("第1章", 30)]
    )
    created = plan_service.create_plan(
        store,
        subject="高频",
        chapters=["第1章"],
        start_date="2026-11-11",
        end_date="2026-11-20",
        sync=False,
    )
    plan_id = created["plan"]["plan_id"]
    first = store.list_tasks(plan_id, local_date="2026-11-11")
    store.update_task(plan_id, "2026-11-11", first[0]["seq"], {"calendar_event_id": "ev-1"})

    monkeypatch.setattr(
        plan_service.core_sync,
        "calendar_starts",
        lambda event_ids, start, end, tzid="UTC": {"ev-1": "2026-11-15"},
    )
    moved_cards: list[tuple[str, str]] = []
    monkeypatch.setattr(
        plan_service.core_sync,
        "move_card",
        lambda card_id, due: moved_cards.append((card_id, due)) or {"ok": True},
    )
    store.update_task(plan_id, "2026-11-11", first[0]["seq"], {"kanban_card_id": "card-1"})

    result = plan_service.sync_from_calendar(store, plan_id=plan_id)
    assert result["ok"] is True
    assert result["moved"] == [
        {"from": "2026-11-11", "to": "2026-11-15", "tasks": len(first)}
    ]
    assert store.list_tasks(plan_id, local_date="2026-11-11") == []
    assert store.list_tasks(plan_id, local_date="2026-11-15")
    assert moved_cards == [("card-1", "2026-11-15")]


def test_sync_calendar_refuses_to_merge_two_days(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(plan_service, "local_today", lambda tz: date(2026, 11, 11))
    store = ReviewStore(tmp_path / "review")
    monkeypatch.setattr(
        plan_service, "chapter_work", lambda subject, chapters: [ChapterWork("第1章", 120)]
    )
    created = plan_service.create_plan(
        store,
        subject="高频",
        chapters=["第1章"],
        start_date="2026-11-11",
        end_date="2026-11-20",
        sync=False,
    )
    plan_id = created["plan"]["plan_id"]
    day_a = store.list_tasks(plan_id, local_date="2026-11-11")
    day_b = store.list_tasks(plan_id, local_date="2026-11-12")
    assert day_a and day_b
    store.update_task(plan_id, "2026-11-11", day_a[0]["seq"], {"calendar_event_id": "ev-a"})
    monkeypatch.setattr(
        plan_service.core_sync,
        "calendar_starts",
        lambda event_ids, start, end, tzid="UTC": {"ev-a": "2026-11-12"},
    )
    result = plan_service.sync_from_calendar(store, plan_id=plan_id)
    assert result["moved"] == []
    assert result["skipped"] and "already has tasks" in result["skipped"][0]["reason"]
    assert "跳过 1 天" in result["message"]
    # The source day is untouched: same tasks, same seq order.
    assert [task["seq"] for task in store.list_tasks(plan_id, local_date="2026-11-11")] == [
        task["seq"] for task in day_a
    ]


def test_sync_calendar_reads_a_real_whole_day_move(tmp_path, monkeypatch) -> None:
    # No mock: the day block is moved in a real CalendarStore and the plan has
    # to read that move back through the same public occurrences() API.
    from deeptutor.services.calendar import store as calendar_store
    from deeptutor.services.calendar.models import EventInput, EventPatch, EventPatchRequest

    calendar = calendar_store.CalendarStore(tmp_path / "calendar.sqlite")
    monkeypatch.setattr(calendar_store, "get_calendar_store", lambda: calendar)
    monkeypatch.setattr(plan_service, "local_today", lambda tz: date(2026, 11, 11))
    monkeypatch.setattr(
        plan_service, "chapter_work", lambda subject, chapters: [ChapterWork("第1章", 30)]
    )
    event = calendar.create(
        EventInput(
            title="学习：高频",
            all_day=True,
            start="2026-11-11",
            end="2026-11-12",
        )
    )["event"]

    store = ReviewStore(tmp_path / "review")
    created = plan_service.create_plan(
        store,
        subject="高频",
        chapters=["第1章"],
        start_date="2026-11-11",
        end_date="2026-11-12",
        sync=False,
    )
    plan_id = created["plan"]["plan_id"]
    first = store.list_tasks(plan_id, local_date="2026-11-11")
    store.update_task(plan_id, "2026-11-11", first[0]["seq"], {"calendar_event_id": event.id})
    calendar.update(
        event.id,
        EventPatchRequest(changes=EventPatch(start="2026-11-12", end="2026-11-13")),
    )

    result = plan_service.sync_from_calendar(store, plan_id=plan_id)
    assert result["ok"] is True
    assert result["moved"] == [
        {"from": "2026-11-11", "to": "2026-11-12", "tasks": len(first)}
    ]
    assert store.list_tasks(plan_id, local_date="2026-11-11") == []
    assert store.list_tasks(plan_id, local_date="2026-11-12")


def test_sync_calendar_reports_a_calendar_read_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(plan_service, "local_today", lambda tz: date(2026, 11, 11))
    monkeypatch.setattr(
        plan_service, "chapter_work", lambda subject, chapters: [ChapterWork("第1章", 30)]
    )
    store = ReviewStore(tmp_path / "review")
    created = plan_service.create_plan(
        store,
        subject="高频",
        chapters=["第1章"],
        start_date="2026-11-11",
        end_date="2026-11-12",
        sync=False,
    )
    plan_id = created["plan"]["plan_id"]
    first = store.list_tasks(plan_id, local_date="2026-11-11")
    store.update_task(plan_id, "2026-11-11", first[0]["seq"], {"calendar_event_id": "ev-1"})
    first = store.list_tasks(plan_id, local_date="2026-11-11")

    def broken(*args: Any, **kwargs: Any) -> dict[str, str]:
        raise core_sync.CalendarReadError("calendar is down")

    monkeypatch.setattr(plan_service.core_sync, "calendar_starts", broken)
    result = plan_service.sync_from_calendar(store, plan_id=plan_id)
    assert result["ok"] is False
    assert "calendar is down" in result["error"]
    assert "计划未改动" in result["message"]
    assert store.list_tasks(plan_id, local_date="2026-11-11") == first


def test_replan_keeps_completed_work_and_drops_superseded_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(plan_service, "local_today", lambda tz: date(2026, 11, 11))
    monkeypatch.setattr(
        plan_service, "chapter_work", lambda subject, chapters: [ChapterWork("第1章", 120)]
    )
    store = ReviewStore(tmp_path / "review")
    created = plan_service.create_plan(
        store,
        subject="高频",
        chapters=["第1章"],
        start_date="2026-11-11",
        end_date="2026-11-20",
        sync=False,
    )
    plan_id = created["plan"]["plan_id"]
    stale = {
        "local_date": "2026-11-10",
        "seq": 1,
        "kind": "study",
        "title": "旧计划：第1章",
        "detail": "",
        "chapter": "第1章",
        "minutes": 15,
        "status": "todo",
        "calendar_event_id": "stale-event",
        "kanban_card_id": "stale-card",
    }
    store.replace_tasks(plan_id, store.list_tasks(plan_id) + [stale])
    dropped: list[tuple[str | None, str | None]] = []
    monkeypatch.setattr(
        plan_service.core_sync,
        "drop_day",
        lambda event, card: dropped.append((event, card))
        or {"calendar": {"ok": True}, "card": {"ok": True}},
    )

    done = [
        task
        for task in store.list_tasks(plan_id, local_date="2026-11-11")
        if task["kind"] == "study"
    ][0]
    plan_service.complete_task(
        store,
        plan_id=plan_id,
        local_date="2026-11-11",
        seq=done["seq"],
        self_note="今天记住了",
        sync=False,
    )

    result = plan_service.replan(
        store, plan_id=plan_id, end_date="2026-11-25", sync=False
    )
    assert result["ok"] is True
    tasks = store.list_tasks(plan_id)
    assert all(task["local_date"] != "2026-11-10" for task in tasks)
    assert ("stale-event", "stale-card") in dropped
    kept = [task for task in tasks if task["status"] == "done"]
    assert len(kept) == 1
    assert kept[0]["self_note"] == "今天记住了"
    assert kept[0]["completed_at"] is not None
    # The completed minutes are neither lost nor scheduled twice.
    assert sum(task["minutes"] for task in tasks if task["kind"] == "study") == 120
    keys = [(task["local_date"], task["seq"]) for task in tasks]
    assert len(keys) == len(set(keys))


def test_sync_day_links_the_new_card_not_an_archived_one(tmp_path, monkeypatch) -> None:
    from deeptutor.services import task_board

    board = task_board.TaskBoardStore(tmp_path / "tasks.sqlite")
    monkeypatch.setattr(task_board, "get_task_board_store", lambda: board)
    title = "2026-11-11 学习：高频（30 分钟）"
    old = board.create(task_board.CreateCard(title=title, due_date="2026-11-11"), "")
    old_id = old.cards[-1].id
    board.update(old_id, task_board.UpdateCard(archived=True), workspace_id="")

    result = core_sync.sync_day(
        {"subject": "高频"},
        "2026-11-11",
        [{"title": "第1章", "kind": "study", "minutes": 30, "status": "todo"}],
        event_id="ev-1",
    )
    new_id = result["card"]["id"]
    assert new_id and new_id != old_id
    live = {card.id: card for card in board.read().cards}
    assert live[new_id].archived is False


def test_subject_doc_is_reused_and_stale_links_are_repaired(monkeypatch) -> None:
    from deeptutor.co_writer import storage as co_writer_storage

    class FakeStorage:
        def __init__(self) -> None:
            self.documents: dict[str, Any] = {}
            self.created = 0

        def create_document(self, *, title, content="", blocks=None, parent_id=None):
            self.created += 1
            document = SimpleNamespace(
                id=f"doc{self.created}", title=title, parent_id=parent_id
            )
            self.documents[document.id] = document
            return document

        def load_document(self, doc_id):
            return self.documents.get(doc_id)

        def list_documents(self):
            return list(self.documents.values())

    fake = FakeStorage()
    monkeypatch.setattr(co_writer_storage, "get_co_writer_storage", lambda: fake)
    plan = {"subject": "高频", "start_date": "2026-11-11", "end_date": "2026-11-20"}

    first = core_sync.ensure_subject_doc(plan)
    second = core_sync.ensure_subject_doc({**plan, "notes_parent_doc_id": "missing"})
    third = core_sync.ensure_subject_doc({**plan, "notes_parent_doc_id": first["id"]})
    assert first["id"] == second["id"] == third["id"] == "doc1"
    assert fake.created == 1


def test_buffer_day_splits_a_session_instead_of_overflowing() -> None:
    work = [ChapterWork(f"第{index}章", 30) for index in range(1, 8)]
    result = build_schedule(work, date(2026, 11, 11), date(2026, 11, 17), 60)
    assert result["feasible"] is True
    study = [task for task in result["tasks"] if task["kind"] == "study"]
    assert sum(task["minutes"] for task in study) == 210
    per_day: dict[str, int] = {}
    for task in study:
        per_day[task["local_date"]] = per_day.get(task["local_date"], 0) + task["minutes"]
    assert max(per_day.values()) <= result["study_minutes_per_day"]


def test_chapter_coverage_reports_missing_packs(tmp_path, monkeypatch) -> None:
    # "chapter → knowledge points" is answered by the chapter's review pack;
    # the plan must say which chapters have none instead of promising review.
    from deeptutor.review.fixture import build_fixture_spec

    store = ReviewStore(tmp_path / "review")
    spec = build_fixture_spec()
    store.put_spec(spec)
    rows = plan_service.chapter_coverage(store, "高频", ["第1章_绪论", "第9章_不存在"])
    by_chapter = {row["chapter"]: row for row in rows}
    assert by_chapter["第9章_不存在"]["has_pack"] is False
    assert by_chapter["第9章_不存在"]["concepts"] == 0
    # The fixture pack is titled 高频 · …; whichever chapter it names is found.
    found = [row for row in rows if row["has_pack"]]
    assert len(found) <= 1


def test_plan_today_lists_concepts_of_the_day(tmp_path, monkeypatch) -> None:
    from deeptutor.review.fixture import build_fixture_spec

    store = ReviewStore(tmp_path / "review")
    store.put_spec(build_fixture_spec())
    monkeypatch.setattr(
        plan_service, "chapter_work", lambda subject, chapters: [ChapterWork("第1章", 20)]
    )
    created = plan_service.create_plan(
        store,
        subject="高频",
        chapters=["第1章"],
        start_date="2026-11-11",
        end_date="2026-11-15",
        sync=False,
    )
    plan_id = created["plan"]["plan_id"]
    today = plan_service.plan_today(store, plan_id=plan_id, on_date="2026-11-11")
    day = today["days"][0]
    # No pack matches 第1章 in the fixture title, so the list is empty but the
    # key exists: the model never has to guess whether it is missing.
    assert day["chapter_concepts"] == []
    assert created["coverage"][0]["chapter"] == "第1章"


def test_plan_tool_soft_fails_without_arguments() -> None:
    result = asyncio.run(ReviewPlanTool().execute(action="create"))
    assert result.success is True
    assert "create needs subject" in result.content


def test_plan_tool_status_returns_json() -> None:
    result = asyncio.run(ReviewPlanTool().execute(action="status"))
    assert result.success is True
    payload = result.metadata["review_plan"]
    assert "plans" in payload and payload["instruction"]


def test_day_note_folds_chapter_milestones_into_the_study_line() -> None:
    tasks = [
        {"title": "开始：第1章_绪论", "kind": "milestone", "minutes": 0, "status": "todo"},
        {"title": "第1章_绪论", "kind": "study", "minutes": 48, "status": "todo"},
        {"title": "完成：第1章_绪论", "kind": "milestone", "minutes": 0, "status": "todo"},
        {"title": "开始：第2章_选频网络", "kind": "milestone", "minutes": 0, "status": "todo"},
        {"title": "第2章_选频网络 · 第 1/5 段", "kind": "study", "minutes": 24, "status": "todo"},
    ]
    note = core_sync._day_note(tasks)
    assert "开始：" not in note and "完成：" not in note
    assert note.count("第1章_绪论") == 1
    assert "（本章读完）" in note
    assert "第2章_选频网络 · 第 1/5 段" in note


def test_day_note_folds_milestones_for_named_parts() -> None:
    tasks = [
        {"title": "开始：第1章_绪论", "kind": "milestone", "minutes": 0},
        {
            "title": "第1章_绪论 · G1高频第一章_绪论",
            "chapter": "第1章_绪论",
            "kind": "study",
            "minutes": 20,
        },
        {"title": "完成：第1章_绪论", "kind": "milestone", "minutes": 0},
    ]
    note = core_sync._day_note(tasks)
    assert note == "· 第1章_绪论 · G1高频第一章_绪论（本章读完）"


def test_practice_hint_names_the_pack_every_day() -> None:
    """A card always says which pack goes with the chapter and when to sit it:
    the closing day says today, a chapter still in progress says after it."""
    packs = {"第1章": ("kb-1", 6)}
    mid = [
        {
            "title": "第1章 · §一、核心推导",
            "chapter": "第1章",
            "kind": "study",
            "minutes": 30,
        }
    ]
    last = [
        {
            "title": "第1章 · §三、例题",
            "chapter": "第1章",
            "kind": "study",
            "minutes": 30,
        },
        {"title": "完成：第1章", "kind": "milestone", "minutes": 0},
    ]
    assert plan_service._practice_hints(packs, mid) == [
        "复习包：第1章 · 今天不做，学完本章后一次做完 6 题"
    ]
    assert plan_service._practice_hints(packs, last) == [
        "复习包：第1章 · 今天做 6 题（一次做完）"
    ]
    assert plan_service._practice_hints({}, mid) == ["复习包：第1章 · 暂无复习包"]


def test_practice_hint_splits_a_pack_over_the_session_cap() -> None:
    tasks = [{"title": "第1章 · §一", "chapter": "第1章", "kind": "study", "minutes": 30},
             {"title": "完成：第1章", "kind": "milestone", "minutes": 0}]
    hints = plan_service._practice_hints({"第1章": ("kb-1", 20)}, tasks)
    assert hints == ["复习包：第1章 · 今天做 12 题（共 20 题，余下按到期继续）"]


def test_finish_milestone_carries_the_pack_pointer() -> None:
    tasks = [
        {"title": "第1章 · §一", "chapter": "第1章", "kind": "study", "minutes": 30},
        {"title": "完成：第1章", "chapter": "第1章", "kind": "milestone", "minutes": 0},
        {"title": "完成：第9章", "kind": "milestone", "minutes": 0},
    ]
    plan_service._attach_practice(tasks, {"第1章": ("kb-1", 6)})
    assert tasks[1]["detail"] == "复习包：第1章 · 学完本章后一次做完 6 题"
    assert tasks[2]["detail"] == "复习包：第9章 · 暂无复习包"


def test_short_title_keeps_a_lecture_prefix_only_when_it_adds_information() -> None:
    assert (
        materials.short_title("016_第2章__第1讲_课程重点.md", "第2章_选频网络")
        == "第1讲_课程重点"
    )
    assert (
        materials.short_title("004_微机原理及应用_第2讲_课程重点.md", "第2讲")
        == "课程重点"
    )


def test_source_tag_names_the_revision_not_a_count() -> None:
    assert materials.source_tag("002_课程PPT要点_v1_备份__01_第一章_x.md") == "v1"
    assert materials.source_tag("030_课程PPT要点_v2_备份__01_第一章_x.md") == "v2"
    assert materials.source_tag("033_第1章_绪论.md") == ""


def test_two_revisions_of_one_deck_are_tagged_v1_v2(tmp_path, monkeypatch) -> None:
    (tmp_path / "002_课件_v1_备份__01_引言.md").write_text(
        "# 引言\n## 一、a\n内容\n", encoding="utf-8"
    )
    (tmp_path / "030_课件_v2_备份__01_引言.md").write_text(
        "# 引言\n## 一、a\n补充内容\n", encoding="utf-8"
    )
    from deeptutor.review import generate

    monkeypatch.setattr(
        generate,
        "chapter_sources",
        lambda kb, chapter, limit=500: [
            "002_课件_v1_备份__01_引言.md",
            "030_课件_v2_备份__01_引言.md",
        ],
    )
    monkeypatch.setattr(materials, "_raw_root", lambda kb: tmp_path)
    parts = materials.chapter_parts("kb", "第1章", 20, 60)
    assert [part.label for part in parts] == ["01_引言（v1）", "01_引言（v2）"]


def test_a_document_without_sections_stays_whole(tmp_path, monkeypatch) -> None:
    body = "\n".join(f"第 {i} 页\n内容" for i in range(1, 40))
    (tmp_path / "020_课件__07_第二章_传输线.md").write_text(body, encoding="utf-8")
    from deeptutor.review import generate

    monkeypatch.setattr(
        generate,
        "chapter_sources",
        lambda kb, chapter, limit=500: ["020_课件__07_第二章_传输线.md"],
    )
    monkeypatch.setattr(materials, "_raw_root", lambda kb: tmp_path)
    parts = materials.chapter_parts("kb", "第2章", 120, 60)
    assert [part.label for part in parts] == ["07_第二章_传输线"]
    assert parts[0].minutes == 120


def test_md_sections_ignores_slide_page_markers() -> None:
    text = "## 第 1 页\n正文\n## 一、核心推导\n内容\n## 第 2 页\n更多\n"
    assert [name for name, _ in materials.md_sections(text)] == ["一、核心推导"]


def test_pack_reads_named_parts_instead_of_minute_slices() -> None:
    work = [
        ChapterWork(
            name="第1章_绪论",
            minutes=60,
            parts=(Part("MC1-1-1 引入", 20), Part("MC1-1-2 定义", 20)),
        )
    ]
    schedule = build_schedule(work, date(2026, 1, 1), date(2026, 1, 3), 60)
    titles = [task["title"] for task in schedule["tasks"] if task["kind"] == "study"]
    assert titles == [
        "第1章_绪论 · MC1-1-1 引入",
        "第1章_绪论 · MC1-1-2 定义",
    ]


def test_pack_keeps_a_part_whole_when_only_a_sliver_of_the_day_is_left() -> None:
    work = [ChapterWork(name="第1章", minutes=38, parts=(Part("A", 20), Part("B", 18)))]
    schedule = build_schedule(work, date(2026, 1, 1), date(2026, 1, 3), 60)
    study = [task for task in schedule["tasks"] if task["kind"] == "study"]
    assert [task["title"] for task in study] == ["第1章 · A", "第1章 · B"]
    assert [task["local_date"] for task in study] == ["2026-01-01", "2026-01-01"]


def test_pack_never_runs_out_of_days_when_the_total_fits() -> None:
    work = [
        ChapterWork(name="第1章", minutes=60, parts=(Part("A", 20), Part("B", 20), Part("C", 20)))
    ]
    schedule = build_schedule(work, date(2026, 1, 1), date(2026, 1, 2), 60)
    assert schedule["feasible"] is True
    study = [task for task in schedule["tasks"] if task["kind"] == "study"]
    assert [task["title"] for task in study] == ["第1章 · A", "第1章 · B", "第1章 · C"]


def test_pack_splits_a_part_that_overruns_the_day() -> None:
    work = [ChapterWork(name="第1章", minutes=50, parts=(Part("§一 长小节", 50),))]
    schedule = build_schedule(work, date(2026, 1, 1), date(2026, 1, 2), 60)
    titles = [task["title"] for task in schedule["tasks"] if task["kind"] == "study"]
    assert titles == ["第1章 · §一 长小节（一）", "第1章 · §一 长小节（二）"]


def test_day_note_appends_practice_hints() -> None:
    tasks = [{"title": "第1章_绪论", "kind": "study", "minutes": 40, "status": "todo"}]
    note = core_sync._day_note(tasks, "复习包：第1章_绪论 · 今天做 6 题（一次做完）")
    assert note.splitlines()[-1] == "复习包：第1章_绪论 · 今天做 6 题（一次做完）"

def test_day_note_names_the_living_book_chapter(tmp_path, monkeypatch) -> None:
    """The heading is the book's chapter name; nothing else says 活书."""
    home = tmp_path / "review"
    home.mkdir()
    (home / "book-map.json").write_text(
        '{"第2章_选频网络": [{"label": "第 2 章 · 选频网络", "anchor": "ch2"}],'
        ' "第4章_微波元件与微波网络": [{"label": "4 · 微波元件与网络（上）", "anchor": "ch4"},'
        ' {"label": "5 · 微波元件与网络（下）", "anchor": "ch5"}]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(home))
    tasks = [{"title": "第2章_选频网络 · 第1讲_课程重点", "kind": "study", "minutes": 40}]
    note = core_sync._day_note(
        tasks, "复习包：第2章_选频网络 · 今天不做，学完本章后一次做完 6 题"
    )
    assert note.splitlines() == [
        "· 第 2 章 · 选频网络 · 第1讲_课程重点",
        "复习包：第2章_选频网络 · 今天不做，学完本章后一次做完 6 题",
    ]
    two = core_sync._day_note(
        [{"title": "第4章_微波元件与微波网络 · 课件_16", "kind": "study", "minutes": 30}]
    )
    # Two book chapters would read as a double numbering: keep the plan name.
    assert two.splitlines() == ["· 第4章_微波元件与微波网络 · 课件_16"]


def test_day_note_names_book_sections_when_the_chapter_maps(tmp_path, monkeypatch) -> None:
    """Every note of the chapter is a book section: list sections, once each."""
    home = tmp_path / "review"
    home.mkdir()
    (home / "book-map.json").write_text(
        '{"第1章": [{"label": "01 · 离散时间信号与系统", "anchor": "ch01"}]}',
        encoding="utf-8",
    )
    (home / "book-sections.json").write_text(
        json.dumps(
            {
                "version": 1,
                "chapters": {
                    "第1章": {
                        "MC1-1-3-A": {"no": "1.4", "title": "序列的基本运算", "anchor": "s01-4"},
                        "MC1-2-1-B": {
                            "no": "1.7",
                            "title": "离散时间系统：线性与移不变",
                            "anchor": "s01-7",
                        },
                        "MC1-2-2-C": {
                            "no": "1.7",
                            "title": "离散时间系统：线性与移不变",
                            "anchor": "s01-7",
                        },
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(home))
    tasks = [
        {"title": "第1章 · MC1-1-3-A", "chapter": "第1章", "kind": "study", "minutes": 20},
        {"title": "第1章 · MC1-2-1-B", "chapter": "第1章", "kind": "study", "minutes": 20},
        {"title": "第1章 · MC1-2-2-C", "chapter": "第1章", "kind": "study", "minutes": 20},
        {"title": "完成：第1章", "chapter": "第1章", "kind": "milestone", "minutes": 0},
    ]
    assert core_sync._day_note(tasks).splitlines() == [
        "· 01 · 离散时间信号与系统（2 节）",
        "　· 1.4 序列的基本运算",
        "　· 1.7 离散时间系统：线性与移不变（本章读完）",
    ]


def test_day_note_keeps_an_unmapped_note_as_its_own_row(
    tmp_path, monkeypatch
) -> None:
    """A note that covers half a chapter keeps its name; the rest stay sections."""
    home = tmp_path / "review"
    home.mkdir()
    (home / "book-sections.json").write_text(
        json.dumps(
            {
                "version": 1,
                "chapters": {"第2章": {"MC2-1-1": {"no": "2.3", "title": "Z 变换", "anchor": "s02-3"}}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(home))
    tasks = [
        {"title": "第2章 · MC2-1-1", "chapter": "第2章", "kind": "study", "minutes": 20},
        {"title": "第2章 · 第5讲_课程重点", "chapter": "第2章", "kind": "study", "minutes": 20},
    ]
    assert core_sync._day_note(tasks).splitlines() == [
        "· 第2章（2 节）",
        "　· 2.3 Z 变换",
        "　· 第5讲_课程重点",
    ]


def test_day_note_drops_an_admin_note_from_a_sectioned_chapter(
    tmp_path, monkeypatch
) -> None:
    """A 复核报告 task is checking work, not reading: it earns no card line."""
    home = tmp_path / "review"
    home.mkdir()
    (home / "book-sections.json").write_text(
        json.dumps(
            {
                "version": 1,
                "chapters": {
                    "第8讲": {
                        "课程重点": [
                            {"no": "8.3", "title": "指令格式", "anchor": "s8-2"},
                            {"no": "8.4", "title": "寻址方式", "anchor": "s8-3"},
                        ]
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(home))
    tasks = [
        {"title": "第8讲 · 课程重点", "chapter": "第8讲", "kind": "study", "minutes": 30},
        {"title": "第8讲 · 复核报告", "chapter": "第8讲", "kind": "study", "minutes": 12},
    ]
    assert core_sync._day_note(tasks).splitlines() == [
        "· 第8讲（2 节）",
        "　· 8.3 指令格式",
        "　· 8.4 寻址方式",
    ]


def test_day_note_lists_every_section_a_note_covers(tmp_path, monkeypatch) -> None:
    """One lecture note spans several sections: the card names each of them."""
    home = tmp_path / "review"
    home.mkdir()
    (home / "book-sections.json").write_text(
        json.dumps(
            {
                "version": 1,
                "chapters": {
                    "第1讲": {
                        "课程重点": [
                            {"no": "1.1", "title": "微型计算机的发展", "anchor": "s1-0"},
                            {"no": "1.2", "title": "寄存器 `FLAGS` 与芯片组成", "anchor": "s1-1"},
                        ]
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(home))
    tasks = [{"title": "第1讲 · 课程重点", "chapter": "第1讲", "kind": "study", "minutes": 47}]
    assert core_sync._day_note(tasks).splitlines() == [
        "· 第1讲（2 节）",
        "　· 1.1 微型计算机的发展",
        "　· 1.2 寄存器 FLAGS 与芯片组成",
    ]


def test_day_note_keeps_the_plan_chapter_without_a_map() -> None:
    tasks = [{"title": "第2章_选频网络 · 第1讲_课程重点", "kind": "study", "minutes": 40}]
    assert core_sync._day_note(tasks).splitlines() == ["· 第2章_选频网络 · 第1讲_课程重点"]


def test_start_milestone_waits_for_the_first_reading_task() -> None:
    """「开始：」must land on the day the chapter's first document is read."""
    first = ChapterWork(name="甲", minutes=60)
    second = ChapterWork(name="乙", minutes=30, parts=(Part(label="乙a", minutes=30),))
    days = [date(2026, 11, 14), date(2026, 11, 15)]
    tasks = study_plan._pack_tasks([first, second], days, [60, 60], 60, split=True)
    for day in days:
        begins = {
            t["title"].split("：", 1)[1]
            for t in tasks
            if t["local_date"] == day.isoformat()
            and t["kind"] == "milestone"
            and t["title"].startswith("开始：")
        }
        studied = {
            t["chapter"]
            for t in tasks
            if t["local_date"] == day.isoformat() and t["kind"] == "study"
        }
        assert begins <= studied
    begins = [t for t in tasks if t["title"] == "开始：乙"]
    assert len(begins) == 1
    assert begins[0]["local_date"] == "2026-11-15"

def test_day_note_marks_the_last_document_of_a_finished_chapter() -> None:
    tasks = [
        {"title": "开始：第1章_绪论", "kind": "milestone", "minutes": 0},
        {"title": "第1章_绪论 · 第1讲_课程重点", "kind": "study", "minutes": 20},
        {"title": "第1章_绪论 · 绪论", "kind": "study", "minutes": 15},
        {"title": "完成：第1章_绪论", "kind": "milestone", "minutes": 0},
    ]
    lines = core_sync._day_note(tasks).splitlines()
    assert lines[0] == "· 第1章_绪论（2 篇）"
    assert "（本章读完）" not in lines[1]
    assert lines[2].endswith("（本章读完）")


def test_day_note_merges_parens_on_a_continuation_line() -> None:
    tasks = [
        {"title": "第1章_离散时间信号的由来 · MC1-1-2（下）", "kind": "study", "minutes": 30},
        {"title": "完成：第1章_离散时间信号的由来", "kind": "milestone", "minutes": 0},
    ]
    note = core_sync._day_note(tasks)
    assert note.strip().endswith("（下，本章读完）")
    assert "）（" not in note

def test_a_part_cut_three_ways_numbers_its_days() -> None:
    work = [ChapterWork(name="第1章", minutes=150, parts=(Part("§一 长小节", 150),))]
    schedule = build_schedule(work, date(2026, 1, 1), date(2026, 1, 3), 100)
    titles = [task["title"] for task in schedule["tasks"] if task["kind"] == "study"]
    assert titles == [
        "第1章 · §一 长小节（一）",
        "第1章 · §一 长小节（二）",
        "第1章 · §一 长小节（三）",
    ]


def test_slicing_keeps_a_version_tag_inside_the_parenthesis() -> None:
    work = [ChapterWork(name="第4章", minutes=90, parts=(Part("17_第四章_微波网络（v2）", 90),))]
    schedule = build_schedule(work, date(2026, 1, 1), date(2026, 1, 2), 100)
    titles = [task["title"] for task in schedule["tasks"] if task["kind"] == "study"]
    assert titles == [
        "第4章 · 17_第四章_微波网络（v2，一）",
        "第4章 · 17_第四章_微波网络（v2，二）",
    ]

def test_day_note_groups_documents_of_one_chapter() -> None:
    tasks = [
        {"title": "第1章_绪论 · 第1讲_课程重点", "kind": "study", "minutes": 20},
        {"title": "第1章_绪论 · 绪论", "kind": "study", "minutes": 20},
        {"title": "完成：第1章_绪论", "kind": "milestone", "minutes": 0},
    ]
    lines = core_sync._day_note(tasks).splitlines()
    assert lines[0] == "· 第1章_绪论（2 篇）"
    assert lines[1] == "　· 第1讲_课程重点"
    assert lines[2] == "　· 绪论（本章读完）"
    assert len(lines) == 3
