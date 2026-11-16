"""Use-cases for study plans: create, today, complete, replan, note draft.

The tool layer stays thin; everything that touches storage or the workbench
lives here so it can be tested without an MCP client.
"""

from __future__ import annotations

import time
import uuid
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from deeptutor.review import core_sync
from deeptutor.review.materials import attach_parts
from deeptutor.review.queue import MAX_ITEMS as MAX_SESSION_ITEMS
from deeptutor.review.storage import ReviewStore
from deeptutor.review.study_plan import (
    DEFAULT_DAILY_MINUTES,
    REVIEW_RATIO,
    ChapterWork,
    PlanError,
    build_schedule,
    capacities_from_budget,
    chapter_work,
    new_plan_id,
    note_body,
    study_minutes_per_day,
    weekly_end_date,
)

DEFAULT_TIMEZONE = "Asia/Shanghai"


def local_today(timezone: str) -> date:
    try:
        zone = ZoneInfo(timezone)
    except Exception:  # noqa: BLE001 - unknown zone falls back to the default
        zone = ZoneInfo(DEFAULT_TIMEZONE)
    return datetime.now(zone).date()


def _day_ids(tasks: list[dict[str, Any]]) -> dict[str, str | None]:
    event_id = next((t["calendar_event_id"] for t in tasks if t.get("calendar_event_id")), None)
    card_id = next((t["kanban_card_id"] for t in tasks if t.get("kanban_card_id")), None)
    return {"event": event_id, "card": card_id}


def _subject_packs(store: ReviewStore, subject: str) -> dict[str, tuple[str, int]]:
    """chapter -> (spec_id, item count) for one subject's review packs."""
    prefix = f"{subject} · "
    suffix = " 复习包"
    packs: dict[str, tuple[str, int]] = {}
    for row in store.list_specs():
        try:
            spec = store.get_spec(row["spec_id"])
        except Exception:  # noqa: BLE001 - one bad pack must not hide the rest
            continue
        title = spec.title or ""
        if not (title.startswith(prefix) and title.endswith(suffix)):
            continue
        chapter = title[len(prefix) : -len(suffix)]
        packs[chapter] = (spec.id, len(spec.items))
    return packs


def _practice_hints(
    packs: dict[str, tuple[str, int]], tasks: list[dict[str, Any]]
) -> list[str]:
    """Name the practice pack that goes with the chapter read today.

    Every card names the pack for each chapter it touches, so "what do I read"
    and "what do I practise" sit on the same card. A pack only makes sense once
    its chapter is read, so a chapter still in progress says "今天不做，学完本章
    后一次做完 N 题" and the closing day says "今天做 N 题（一次做完）": the
    learner must be able to see the day's scope, not just the pack's size - a
    bare "共 6 题" read as "do all six today" while the chapter was still being
    read. The whole pack fits one sitting (packs hold a handful of items, the
    queue's hard cap is 12), so only an oversized pack is split, and then the
    card says how many to take today and that the rest returns when due. A
    chapter with no pack says so instead of staying silent. The prefix is
    "复习包：", not "练习：", because the pack is the spaced-repetition set for
    that chapter - the living book's own in-section quizzes are not scheduled
    here and must not be read as the same thing.
    """
    finished = {
        str(task["title"]).split("：", 1)[-1]
        for task in tasks
        if task.get("kind") == "milestone"
        and str(task.get("title", "")).startswith("完成：")
    }
    hints: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        if task.get("kind") != "study":
            continue
        chapter = str(task.get("chapter") or task["title"].split(" · ", 1)[0]).strip()
        if not chapter or chapter in seen:
            continue
        seen.add(chapter)
        pack = packs.get(chapter)
        if pack is None:
            hints.append(f"复习包：{chapter} · 暂无复习包")
            continue
        _, count = pack
        if chapter not in finished:
            hints.append(
                f"复习包：{chapter} · 今天不做，学完本章后一次做完 {count} 题"
            )
        elif count > MAX_SESSION_ITEMS:
            hints.append(
                f"复习包：{chapter} · 今天做 {MAX_SESSION_ITEMS} 题"
                f"（共 {count} 题，余下按到期继续）"
            )
        else:
            hints.append(f"复习包：{chapter} · 今天做 {count} 题（一次做完）")
    return hints


def _practice_detail(
    packs: dict[str, tuple[str, int]], chapter: str
) -> str:
    """Detail line for a chapter's 完成 milestone: what to practise next."""
    pack = packs.get(chapter)
    if pack is None:
        return f"复习包：{chapter} · 暂无复习包"
    _, count = pack
    if count > MAX_SESSION_ITEMS:
        return (
            f"复习包：{chapter} · 学完本章后做 {MAX_SESSION_ITEMS} 题"
            f"（共 {count} 题）"
        )
    return f"复习包：{chapter} · 学完本章后一次做完 {count} 题"


def _attach_practice(
    tasks: list[dict[str, Any]], packs: dict[str, tuple[str, int]]
) -> None:
    """Put the pack pointer on each 完成 milestone so the task itself carries it."""
    for task in tasks:
        if task.get("kind") != "milestone":
            continue
        title = str(task.get("title", ""))
        chapter = str(task.get("chapter") or "")
        if not chapter and title.startswith("完成："):
            chapter = title.split("：", 1)[-1].strip()
        if not chapter:
            continue
        task["detail"] = _practice_detail(packs, chapter)


def sync_plan(
    store: ReviewStore, plan: dict[str, Any], *, from_date: str | None = None
) -> dict[str, Any]:
    """Mirror each planned day onto the calendar and the task board."""
    tasks = store.list_tasks(plan["plan_id"], since=from_date)
    by_date: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        by_date.setdefault(task["local_date"], []).append(task)

    results: dict[str, Any] = {}
    packs = _subject_packs(store, plan["subject"])
    for local_date, day_tasks in sorted(by_date.items()):
        ids = _day_ids(day_tasks)
        synced = core_sync.sync_day(
            plan,
            local_date,
            day_tasks,
            event_id=ids["event"],
            card_id=ids["card"],
            extra_note="\n".join(_practice_hints(packs, day_tasks)),
        )
        patch: dict[str, Any] = {}
        if synced["calendar"]["id"] and not ids["event"]:
            patch["calendar_event_id"] = synced["calendar"]["id"]
        if synced["card"]["id"] and not ids["card"]:
            patch["kanban_card_id"] = synced["card"]["id"]
        if patch:
            store.update_task(plan["plan_id"], local_date, day_tasks[0]["seq"], patch)
        results[local_date] = synced
    return results


def drop_mirrors(store: ReviewStore, plan_id: str, *, since: str) -> None:
    for task in store.list_tasks(plan_id, since=since):
        if task.get("calendar_event_id") or task.get("kanban_card_id"):
            core_sync.drop_day(task.get("calendar_event_id"), task.get("kanban_card_id"))
            store.update_task(
                plan_id,
                task["local_date"],
                task["seq"],
                {"calendar_event_id": None, "kanban_card_id": None},
            )


def _chapter_concepts(
    store: ReviewStore, plan: dict[str, Any], tasks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Concepts the day's chapters cover, so study and review share one list.

    Derived from the chapter's review pack (one pack per chapter). A chapter
    without a pack contributes nothing here and says so through
    ``chapter_coverage`` at plan time.
    """
    chapters = list(dict.fromkeys(task["chapter"] for task in tasks if task["chapter"]))
    if not chapters:
        return []
    rows: list[dict[str, Any]] = []
    for chapter in chapters:
        pack = next(
            (
                row
                for row in chapter_coverage(store, plan["subject"], [chapter])
                if row["has_pack"]
            ),
            None,
        )
        if pack is None:
            continue
        try:
            spec = store.get_spec(pack["spec_id"])
        except Exception:  # noqa: BLE001
            continue
        rows.append(
            {
                "chapter": chapter,
                "spec_id": spec.id,
                "concepts": [concept.name for concept in spec.concepts][:12],
            }
        )
    return rows


def _free_day(store: ReviewStore, plan_id: str, target: str) -> bool:
    """A day can be moved onto only when it has no tasks yet."""
    return not store.list_tasks(plan_id, local_date=target)


def _merge_preserved(
    tasks: list[dict[str, Any]], preserved: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep completed rows and slot the rebuilt schedule after them.

    Both sides number their rows from 1 per date, so a preserved task and a new
    task on the same date would share ``(plan_id, local_date, seq)``.
    """
    offsets: dict[str, int] = {}
    for task in preserved:
        day = task["local_date"]
        offsets[day] = max(offsets.get(day, 0), int(task["seq"]))
    merged = list(preserved)
    for task in tasks:
        offset = offsets.get(task["local_date"], 0)
        merged.append({**task, "seq": int(task["seq"]) + offset} if offset else task)
    return merged


def _drop_superseded(store: ReviewStore, plan_id: str, before: str) -> None:
    """Remove mirrors of past days whose work is about to be discarded.

    A day that still has completed work keeps its calendar block and card; a
    day with only superseded incomplete rows loses both, so replanning does not
    leave orphan mirrors behind.
    """
    by_date: dict[str, list[dict[str, Any]]] = {}
    for task in store.list_tasks(plan_id):
        if task["local_date"] < before:
            by_date.setdefault(task["local_date"], []).append(task)
    for day_tasks in by_date.values():
        if any(task["status"] == "done" for task in day_tasks):
            continue
        ids = _day_ids(day_tasks)
        if ids["event"] or ids["card"]:
            core_sync.drop_day(ids["event"], ids["card"])


def sync_from_calendar(
    store: ReviewStore, *, plan_id: str, horizon_days: int = 60
) -> dict[str, Any]:
    """Pull calendar moves back into the plan.

    A learner drags the day block on the calendar; the plan is still the truth,
    so the drag has to be read back. Only *whole-day* moves are applied: if two
    plan days landed on one date, or one date holds tasks from another day, the
    move is reported and skipped rather than merging workloads.
    """
    plan = store.get_plan(plan_id)
    if plan is None:
        raise PlanError(f"unknown plan: {plan_id}")
    today = local_today(plan.get("timezone", DEFAULT_TIMEZONE))
    start = today.isoformat()
    end = (today + timedelta(days=horizon_days)).isoformat()

    tasks = store.list_tasks(plan_id, since=start)
    days: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        days.setdefault(task["local_date"], []).append(task)
    event_ids = [
        day_tasks[0]["calendar_event_id"]
        for day_tasks in days.values()
        if day_tasks[0].get("calendar_event_id")
    ]
    if not event_ids:
        return {
            "ok": True,
            "moved": [],
            "skipped": [],
            "warnings": [],
            "message": "没有可对账的日历事件。",
        }

    try:
        starts = core_sync.calendar_starts(
            event_ids, start, end, plan.get("timezone", DEFAULT_TIMEZONE)
        )
    except core_sync.CalendarReadError as exc:
        return {
            "ok": False,
            "moved": [],
            "skipped": [],
            "warnings": [],
            "error": str(exc),
            "message": "日历暂时读不到，计划未改动；请稍后重试。",
        }
    by_event = {day_tasks[0]["calendar_event_id"]: date_key
                for date_key, day_tasks in days.items()
                if day_tasks[0].get("calendar_event_id")}

    moved: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for event_id, new_date in starts.items():
        old_date = by_event.get(event_id)
        if old_date is None:
            continue
        if new_date == old_date:
            continue
        if not _free_day(store, plan_id, new_date):
            skipped.append(
                {"from": old_date, "to": new_date, "reason": "target day already has tasks"}
            )
            continue
        moved_day = days[old_date]
        store.move_day(plan_id, old_date, new_date)
        card_id = moved_day[0].get("kanban_card_id")
        if card_id:
            card = core_sync.move_card(card_id, new_date)
            if not card["ok"]:
                warnings.append(
                    {"date": new_date, "card_id": card_id, "error": card["error"]}
                )
        moved.append({"from": old_date, "to": new_date, "tasks": len(moved_day)})

    parts = []
    if moved:
        parts.append(f"已同步 {len(moved)} 天")
    if skipped:
        parts.append(f"跳过 {len(skipped)} 天（目标日已有任务）")
    if warnings:
        parts.append(f"{len(warnings)} 张卡片未移动，稍后可重试")
    return {
        "ok": True,
        "moved": moved,
        "skipped": skipped,
        "warnings": warnings,
        "message": "；".join(parts) + "。" if parts else "日历与计划一致，无需调整。",
    }


def chapter_coverage(store: ReviewStore, subject: str, chapters: list[str]) -> list[dict[str, Any]]:
    """Which planned chapters already have a review pack, and what it holds.

    A pack is generated per chapter and carries that chapter's concepts, so
    "chapter → knowledge points" needs no separate mapping table: the pack *is*
    the mapping. What the learner cannot see today is which chapters are still
    missing a pack, so this reports it at plan time.
    """
    packs: list[dict[str, Any]] = []
    for meta in store.list_specs():
        try:
            spec = store.get_spec(meta["spec_id"])
        except Exception:  # noqa: BLE001 - one corrupt pack must not hide the rest
            continue
        if subject not in (spec.title or ""):
            continue
        packs.append(
            {
                "spec_id": spec.id,
                "title": spec.title or "",
                "concepts": len(spec.concepts),
                "items": len(spec.items),
            }
        )

    rows: list[dict[str, Any]] = []
    for chapter in chapters:
        needle = chapter.split("_", 1)[-1] if "_" in chapter else chapter
        match = next(
            (
                pack
                for pack in packs
                if chapter in pack["title"] or (needle and needle in pack["title"])
            ),
            None,
        )
        rows.append(
            {
                "chapter": chapter,
                "has_pack": match is not None,
                "spec_id": (match or {}).get("spec_id", ""),
                "concepts": (match or {}).get("concepts", 0),
                "items": (match or {}).get("items", 0),
            }
        )
    return rows


def create_plan(
    store: ReviewStore,
    *,
    subject: str,
    chapters: list[str],
    start_date: str,
    end_date: str,
    daily_minutes: int = DEFAULT_DAILY_MINUTES,
    timezone: str = DEFAULT_TIMEZONE,
    sync: bool = True,
) -> dict[str, Any]:
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError as exc:
        raise PlanError(f"dates must be YYYY-MM-DD: {exc}") from exc
    work = attach_parts(
        subject, chapter_work(subject, chapters), study_minutes_per_day(daily_minutes)
    )
    schedule = build_schedule(work, start, end, daily_minutes)
    if not schedule["feasible"]:
        return {
            "ok": False,
            "reason": "infeasible",
            "required_minutes": schedule["required_minutes"],
            "capacity_minutes": schedule["capacity_minutes"],
            "suggested_end_date": schedule["suggested_end_date"],
            "message": (
                "按这个结束日期排不下：需要 "
                f"{schedule['required_minutes']} 分钟，可用 "
                f"{schedule['capacity_minutes']} 分钟。"
                "请延长结束日期或减少章节。"
            ),
        }

    now = time.time()
    plan = {
        "plan_id": new_plan_id(),
        "subject": subject,
        "status": "active",
        "chapters": list(chapters),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily_minutes": daily_minutes,
        "timezone": timezone,
        "study_minutes_per_day": schedule["study_minutes_per_day"],
        "required_minutes": schedule["required_minutes"],
        "notes_parent_doc_id": None,
    }
    from deeptutor.review.generate import chapter_sources

    sources: dict[str, list[str]] = {}
    for task in schedule["tasks"]:
        if task["kind"] != "study":
            continue
        chapter = task["chapter"]
        if chapter not in sources:
            try:
                sources[chapter] = chapter_sources(subject, chapter)
            except Exception:  # noqa: BLE001 - a missing KB must not block the plan
                sources[chapter] = []
        files = sources[chapter]
        if files and not task.get("detail"):
            listed = "、".join(files[:4]) + ("…" if len(files) > 4 else "")
            task["detail"] = f"约 {task['minutes']} 分钟；资料：{listed}"

    _attach_practice(schedule["tasks"], _subject_packs(store, subject))
    store.create_plan(plan, now=now)
    store.replace_tasks(plan["plan_id"], schedule["tasks"])
    synced = sync_plan(store, plan) if sync else {}
    return {
        "ok": True,
        "plan": store.get_plan(plan["plan_id"]),
        "tasks": store.list_tasks(plan["plan_id"]),
        "sync": synced,
        "coverage": chapter_coverage(store, subject, chapters),
    }


def _review_due(
    store: ReviewStore, subject: str, minutes: int, *, now: float | None = None
) -> dict[str, Any]:
    """Due review items across the subject's packs, read-only.

    ``session_plan`` never creates a session and never records views, so a
    calendar card can ask this every time it is refreshed without polluting
    the review history.
    """
    from deeptutor.review.service import ReviewService

    service = ReviewService(store)
    moment = time.time() if now is None else now
    specs: list[dict[str, Any]] = []
    total = 0
    for row in store.list_specs():
        try:
            spec = store.get_spec(row["spec_id"])
            if subject not in (spec.title or ""):
                continue
            plan = service.session_plan(
                spec.id, mode="review", minutes=max(5, minutes), now=moment
            )
        except Exception:  # noqa: BLE001 - one bad pack must not hide the rest
            continue
        items = plan.get("items") or []
        if not items:
            continue
        specs.append({"spec_id": spec.id, "title": spec.title, "items": len(items)})
        total += len(items)
    specs.sort(key=lambda entry: entry["items"], reverse=True)
    return {"items": total, "specs": specs[:8]}


def plan_today(
    store: ReviewStore,
    *,
    plan_id: str | None = None,
    on_date: str | None = None,
    refresh_mirror: bool = False,
) -> dict[str, Any]:
    if plan_id:
        plan = store.get_plan(plan_id)
        if plan is None:
            raise PlanError(f"unknown plan: {plan_id}")
        plans = [plan]
    else:
        plans = store.list_plans(status="active")

    days = []
    for plan in plans:
        # Keep the day as an ISO string end to end: task rows store strings,
        # and comparing a date object with them raises TypeError (a failure the
        # MCP client only sees as a non-JSON payload).
        local = str(on_date) if on_date else local_today(
            plan.get("timezone", DEFAULT_TIMEZONE)
        ).isoformat()
        tasks = store.list_tasks(plan["plan_id"], local_date=local)
        overdue = [
            task
            for task in store.list_tasks(plan["plan_id"], since=None)
            if task["local_date"] < local  # both are ISO strings
            and task["kind"] == "study"
            and task["status"] == "todo"
        ]
        study = [task for task in tasks if task["kind"] == "study"]
        review_minutes = max(
            0,
            int(plan.get("daily_minutes", DEFAULT_DAILY_MINUTES))
            - sum(int(task["minutes"]) for task in study),
        )
        review = _review_due(store, plan["subject"], review_minutes)
        packs = _subject_packs(store, plan["subject"])
        hints = _practice_hints(packs, tasks)
        review_lines: list[str] = []
        if review["items"]:
            review_lines.append(f"复习：{review['items']} 题到期")
            for entry in review["specs"][:3]:
                chapter = str(entry["title"]).split(" · ", 1)[-1]
                chapter = chapter.rsplit(" 复习包", 1)[0].strip()
                review_lines.append(
                    f"复习：{chapter} · {entry['items']} 题到期"
                )
        day_note = "\n".join([*hints, *review_lines])
        if refresh_mirror:
            # The card is a mirror; refresh it with today's due count so the
            # board shows study and review together. The plan stays the truth.
            ids = _day_ids(tasks)
            if ids["card"] and tasks:
                core_sync.update_day_card(
                    ids["card"],
                    tasks,
                    done=bool(study) and all(task["status"] == "done" for task in study),
                    extra_note=day_note,
                )
        days.append(
            {
                "plan_id": plan["plan_id"],
                "subject": plan["subject"],
                "date": local,
                "status": plan["status"],
                "tasks": tasks,
                "overdue": overdue,
                "review_minutes": review_minutes,
                "review_due": review,
                "chapter_concepts": _chapter_concepts(store, plan, tasks),
            }
        )
    return {"ok": True, "days": days}


def _find_spec(store: ReviewStore, subject: str, chapter: str) -> Any | None:
    needle = chapter.split("_", 1)[-1] if "_" in chapter else chapter
    for row in store.list_specs():
        try:
            spec = store.get_spec(row["spec_id"])
        except Exception:  # noqa: BLE001 - a corrupt row must not block planning
            continue
        title = spec.title or ""
        if subject in title and (chapter in title or needle in title):
            return spec
    return None


def _note_for_chapter(
    store: ReviewStore, plan: dict[str, Any], chapter: str
) -> dict[str, Any] | None:
    existing = [note for note in store.list_notes(plan["plan_id"]) if note["chapter"] == chapter]
    if existing:
        return existing[0]
    chapter_tasks = [
        task for task in store.list_tasks(plan["plan_id"]) if task["chapter"] == chapter
    ]
    study = [task for task in chapter_tasks if task["kind"] == "study"]
    if not study or any(task["status"] != "done" for task in study):
        return None
    spec = _find_spec(store, plan["subject"], chapter)
    body = note_body(plan, chapter, spec, chapter_tasks)
    note = store.create_note(
        {
            "note_id": "note_" + uuid.uuid4().hex[:12],
            "plan_id": plan["plan_id"],
            "chapter": chapter,
            "title": f"{plan['subject']} · {chapter} 学习笔记",
            "body": body,
            "status": "draft",
        },
        now=time.time(),
    )
    return note


def complete_task(
    store: ReviewStore,
    *,
    plan_id: str,
    local_date: str,
    seq: int,
    self_note: str = "",
    sync: bool = True,
) -> dict[str, Any]:
    plan = store.get_plan(plan_id)
    if plan is None:
        raise PlanError(f"unknown plan: {plan_id}")
    task = store.update_task(
        plan_id,
        local_date,
        seq,
        {"status": "done", "completed_at": time.time(), "self_note": self_note},
    )
    if task is None:
        raise PlanError(f"unknown task: {plan_id}/{local_date}/{seq}")

    day_tasks = store.list_tasks(plan_id, local_date=local_date)
    study = [item for item in day_tasks if item["kind"] == "study"]
    day_done = bool(study) and all(item["status"] == "done" for item in study)
    ids = _day_ids(day_tasks)
    if sync and ids["card"]:
        core_sync.update_day_card(ids["card"], day_tasks, done=day_done)

    note = None
    if task["kind"] == "study" and task["chapter"]:
        note = _note_for_chapter(store, plan, task["chapter"])
    return {"ok": True, "task": task, "day_done": day_done, "note": note}


def replan(
    store: ReviewStore,
    *,
    plan_id: str,
    end_date: str | None = None,
    daily_minutes: int | None = None,
    day_capacities: dict[str, int] | None = None,
    weekly_budget: dict[int, int] | None = None,
    review_ratio: float = REVIEW_RATIO,
    sync: bool = True,
) -> dict[str, Any]:
    """Rebuild the remaining schedule from today.

    Two ways to describe an uneven week: ``day_capacities`` (ISO date -> study
    minutes, the exact window) or ``weekly_budget`` (weekday -> minutes, repeat
    until the work fits). Either one overrides ``daily_minutes`` for capacity;
    the plan keeps whatever was passed for display.
    """
    plan = store.get_plan(plan_id)
    if plan is None:
        raise PlanError(f"unknown plan: {plan_id}")
    today = local_today(plan.get("timezone", DEFAULT_TIMEZONE))
    today_key = today.isoformat()
    new_end = date.fromisoformat(end_date or plan["end_date"])
    new_daily = int(daily_minutes or plan["daily_minutes"])
    capacities: list[int] | None = None
    if day_capacities:
        keys = sorted(day_capacities)
        if date.fromisoformat(keys[0]) < today:
            raise PlanError("day_capacities must start today or later")
        new_end = date.fromisoformat(keys[-1])
        capacities = [
            int(day_capacities.get((today + timedelta(days=offset)).isoformat(), 0))
            for offset in range((new_end - today).days + 1)
        ]

    work = chapter_work(plan["subject"], plan["chapters"])
    done_minutes: dict[str, int] = {}
    for task in store.list_tasks(plan_id):
        if task["kind"] == "study" and task["status"] == "done":
            done_minutes[task["chapter"]] = done_minutes.get(task["chapter"], 0) + int(
                task["minutes"]
            )
    remaining = [
        ChapterWork(
            name=item.name,
            minutes=max(0, item.minutes - done_minutes.get(item.name, 0)),
        )
        for item in work
    ]
    remaining = [item for item in remaining if item.minutes > 0]
    if remaining and capacities is None and weekly_budget:
        new_end = weekly_end_date(
            today,
            sum(item.minutes for item in remaining),
            weekly_budget,
            review_ratio,
        )
        capacities = capacities_from_budget(
            [
                today + timedelta(days=offset)
                for offset in range((new_end - today).days + 1)
            ],
            weekly_budget,
            review_ratio,
        )
    slice_minutes = max(capacities) if capacities else study_minutes_per_day(new_daily)
    remaining = attach_parts(plan["subject"], remaining, slice_minutes)
    if not remaining:
        return {"ok": False, "reason": "nothing_left", "message": "所有章节都已学完。"}

    used_today = sum(
        int(task["minutes"])
        for task in store.list_tasks(plan_id, local_date=today_key)
        if task["kind"] == "study" and task["status"] == "done"
    )
    schedule = build_schedule(
        remaining,
        today,
        new_end,
        new_daily,
        review_ratio=review_ratio,
        first_day_used=used_today,
        day_capacities=capacities,
    )
    if not schedule["feasible"]:
        return {
            "ok": False,
            "reason": "infeasible",
            "required_minutes": schedule["required_minutes"],
            "capacity_minutes": schedule["capacity_minutes"],
            "suggested_end_date": schedule["suggested_end_date"],
            "message": "新的结束日期排不下剩余内容，请延长日期。",
        }

    _attach_practice(schedule["tasks"], _subject_packs(store, plan["subject"]))
    _drop_superseded(store, plan_id, today_key)
    store.delete_incomplete_before(plan_id, today_key)
    drop_mirrors(store, plan_id, since=today_key)
    preserved = [
        task
        for task in store.list_tasks(plan_id, since=today_key)
        if task["status"] == "done"
    ]
    store.replace_tasks(
        plan_id, _merge_preserved(schedule["tasks"], preserved), from_date=today_key
    )
    updated = store.update_plan(
        plan_id,
        {
            "end_date": new_end.isoformat(),
            "daily_minutes": max(capacities) if capacities else new_daily,
            "study_minutes_per_day": schedule["study_minutes_per_day"],
            "required_minutes": schedule["required_minutes"],
            "day_capacities": day_capacities or None,
            "weekly_budget": (
                {str(day): minutes for day, minutes in weekly_budget.items()}
                if weekly_budget
                else None
            ),
        },
        now=time.time(),
    )
    synced = sync_plan(store, updated, from_date=today_key) if sync else {}
    return {
        "ok": True,
        "plan": updated,
        "tasks": store.list_tasks(plan_id, since=today_key),
        "sync": synced,
    }


def confirm_note(store: ReviewStore, *, note_id: str, sync: bool = True) -> dict[str, Any]:
    note = store.get_note(note_id)
    if note is None:
        raise PlanError(f"unknown note: {note_id}")
    if note["status"] == "confirmed" and note.get("document_id"):
        return {"ok": True, "note": note, "document_id": note["document_id"]}

    plan = store.get_plan(note["plan_id"])
    if plan is None:
        raise PlanError(f"unknown plan: {note['plan_id']}")

    parent = core_sync.ensure_subject_doc(plan) if sync else {"ok": True, "id": None}
    if parent["id"] and parent["id"] != plan.get("notes_parent_doc_id"):
        plan = store.update_plan(
            plan["plan_id"], {"notes_parent_doc_id": parent["id"]}, now=time.time()
        )
    document = core_sync.create_note_doc(
        plan, note["chapter"], note["title"], note["body"], parent.get("id")
    )
    if not document["ok"]:
        return {
            "ok": False,
            "reason": "sync_failed",
            "error": document["error"],
            "note": note,
            "message": "笔记本不可用，笔记仍是草稿；核心恢复后重试确认。",
        }
    updated = store.update_note(
        note_id,
        {"status": "confirmed", "document_id": document["id"]},
        now=time.time(),
    )
    return {"ok": True, "note": updated, "document_id": document["id"]}


def plan_status(store: ReviewStore, *, plan_id: str | None = None) -> dict[str, Any]:
    plans = [store.get_plan(plan_id)] if plan_id else store.list_plans()
    if plan_id and plans[0] is None:
        raise PlanError(f"unknown plan: {plan_id}")
    rows = []
    for plan in plans:
        assert plan is not None
        tasks = store.list_tasks(plan["plan_id"])
        study = [task for task in tasks if task["kind"] == "study"]
        done = [task for task in study if task["status"] == "done"]
        notes = store.list_notes(plan["plan_id"])
        rows.append(
            {
                "plan_id": plan["plan_id"],
                "subject": plan["subject"],
                "status": plan["status"],
                "start_date": plan["start_date"],
                "end_date": plan["end_date"],
                "daily_minutes": plan["daily_minutes"],
                "chapters": plan["chapters"],
                "study_tasks": len(study),
                "study_done": len(done),
                "notes_draft": sum(1 for note in notes if note["status"] == "draft"),
                "notes_confirmed": sum(1 for note in notes if note["status"] == "confirmed"),
            }
        )
    return {"ok": True, "plans": rows}
