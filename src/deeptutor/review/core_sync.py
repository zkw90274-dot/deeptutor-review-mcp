"""Best-effort mirrors of a study plan onto the DeepTutor workbench.

The plan is the source of truth (``review_plans``); the calendar, task board and
Co-Writer notebook are derived surfaces. Every call here is allowed to fail —
the learner can keep studying while the workbench is unavailable — so each
function returns a status dict instead of raising.

Calls are made in-process against the installed core services. In a stdio MCP
process there is no HTTP request, and the core falls back to the local admin
user (``multi_user.context.get_current_user``), which is exactly the
single-tenant mode this sidecar runs in.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from deeptutor.review.materials import load_book_map, load_book_sections, section_lines
from deeptutor.review.study_plan import PlanError

MAX_NOTE_CHARS = 40_000


class CalendarReadError(RuntimeError):
    """The calendar could not be read; the plan was left untouched."""


def _status(ok: bool, id: str | None = None, error: str | None = None) -> dict[str, Any]:
    return {"ok": ok, "id": id, "error": error}


def _day_title(plan: dict[str, Any], local_date: str, minutes: int) -> str:
    return f"学习：{plan['subject']} · {minutes} 分钟（{local_date}）"


def _study_chapter(task: dict[str, Any]) -> str:
    """The chapter a study task belongs to (X · any part -> X)."""
    chapter = str(task.get("chapter") or "").strip()
    if chapter:
        return chapter
    return str(task.get("title", "")).split(" · ")[0].strip()


def _note_doc(task: dict[str, Any]) -> str:
    """The document part of a study task title (chapter · document)."""
    return str(task.get("title", "")).split(" · ", 1)[-1]


ADMIN_NOTE_WORDS = ("复核报告", "核对报告", "README")


def _is_admin_note(doc: str) -> bool:
    """A note that checks material instead of teaching a section of the book."""
    return any(word in str(doc or "") for word in ADMIN_NOTE_WORDS)


def _row_key(row: str, index: int) -> tuple[int, ...]:
    """Sort sections by their number (1.10 after 1.9); notes keep task order."""
    match = re.match(r"(\d+)(?:\.(\d+))?", row)
    if not match:
        return (0, index)
    return (int(match.group(1)), int(match.group(2) or 0))


def _book_name(chapter: str, mapping: dict[str, list[tuple[str, str]]]) -> str:
    """The living-book chapter name for a plan chapter (the plan name if none)."""
    entries = mapping.get(str(chapter or "").strip()) or []
    # One book chapter gets its own name; two (元件上/下) would read as a
    # double numbering, so keep the plan's chapter name instead.
    if len(entries) != 1:
        return chapter
    return entries[0][0]


def _day_note(tasks: list[dict[str, Any]], extra_note: str | None = None) -> str:
    """One line per study task; chapter markers fold into those lines.

    A day used to print 开始：第1章 directly above 第1章 · 第 1/5 段; both named
    the same chapter, so the board repeated itself. Milestones now only show
    when no study task covers that chapter on that day.

    A note is planned from its knowledge-base file, which numbers things its
    own way. The card speaks the living book instead: the chapter heading is
    the book's chapter name, and when every note of the chapter maps to a book
    section the rows are those sections (1.7 离散时间系统：线性与移不变) - no
    extra 活书 line, no second numbering to decode.
    """
    studies = [task for task in tasks if task.get("kind") == "study"]
    sections = load_book_sections()
    book = load_book_map()
    section_of: dict[int, list[str]] = {}
    keys: dict[str, set[str]] = {}
    for index, task in enumerate(tasks):
        if task.get("kind") != "study":
            continue
        chapter = _study_chapter(task)
        rows = section_lines(chapter, _note_doc(task), sections)
        section_of[index] = rows
        # 复核报告这类任务是核对工作，不是读书任务：整章按小节列的时候它不占
        # 一行。落不回小节的长笔记（一讲概括半章）仍要占一行，用笔记自己的名字，
        # 否则学习者会漏掉真正要读的材料。
        if not rows and _is_admin_note(_note_doc(task)):
            continue
        keys.setdefault(chapter, set()).update(rows or ["note:" + _note_doc(task)])
    counts = {chapter: len(values) for chapter, values in keys.items()}
    # 有过小节映射的章按节列；一篇笔记都没对上的章仍按篇目列。
    whole = {
        chapter
        for chapter, values in keys.items()
        if values and any(not value.startswith("note:") for value in values)
    }
    studied = {_study_chapter(task) for task in studies}
    finished = {
        str(task["title"]).split("：", 1)[-1]
        for task in tasks
        if task.get("kind") == "milestone"
        and str(task.get("title", "")).startswith("完成：")
    }
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    # A chapter may fold several notes into one row, so「本章读完」goes on the
    # chapter's last emitted line, not on the last task.
    row_of: dict[str, int] = {}
    last_of: dict[str, int] = {}
    for index, task in enumerate(tasks):
        if task.get("kind") == "study":
            last_of[_study_chapter(task)] = index
    # chapter -> [(sort key, row text, row is the whole task)]; flushed once its
    # last note is reached so the rows can be ordered by section number.
    pending: dict[str, list[tuple[tuple[int, ...], str, bool]]] = {}
    marker_of: dict[str, str] = {}

    def flush(chapter: str) -> None:
        rows = pending.pop(chapter, [])
        if not rows:
            return
        named = chapter in whole
        if named:
            # 小节按编号排；落不到小节的笔记排在后面，保住原来的先后。
            rows.sort(key=lambda row: (row[0][0] or 10**6, row[0][1]))
        marker = marker_of.pop(chapter, "·")
        title = _book_name(chapter, book)
        if counts.get(chapter, len(rows)) > 1:
            unit = "节" if named else "篇"
            lines.append(f"{marker} {title}（{counts.get(chapter, len(rows))} {unit}）")
            lines.extend(f"　· {row[1]}" for row in rows)
        else:
            row = rows[0]
            section_title = re.sub(r"^\d+(?:\.\d+)?\s*", "", row[1])
            label_title = title.split(" · ", 1)[-1]
            # 「第1章_绪论」 is the whole task, and「0.1 引言与场论基础」 repeats
            # its chapter name: neither adds a document part worth printing.
            redundant = row[2] or (
                not row[2] and named and section_title == label_title
            )
            lines.append(f"{marker} {title}" if redundant else f"{marker} {title} · {row[1]}")
        row_of[chapter] = len(lines) - 1

    for index, task in enumerate(tasks):
        kind = task.get("kind")
        marker = "✓" if task.get("status") == "done" else "·"
        if kind == "study":
            chapter = _study_chapter(task)
            # 一篇笔记可能横跨好几个活书小节（一讲讲 8 节），这时每一节各列一行。
            rows = section_of.get(index, [])
            if not rows:
                if chapter in whole and _is_admin_note(_note_doc(task)):
                    continue
                rows = [_note_doc(task)]
            marker_of.setdefault(chapter, marker)
            for row in rows:
                # 两篇笔记讲同一节时，这一节只列一次。
                if (chapter, row) in seen:
                    continue
                seen.add((chapter, row))
                pending.setdefault(chapter, []).append(
                    (_row_key(row, index), row, row == str(task.get("title", "")))
                )
            if last_of.get(chapter) == index:
                flush(chapter)
            continue
        if kind == "milestone":
            chapter = str(task["title"]).split("：", 1)[-1]
            if chapter in studied:
                continue
        lines.append(f"{marker} {task['title']}")
    for chapter in list(pending):
        flush(chapter)
    for chapter in finished:
        index = row_of.get(chapter)
        if index is None:
            continue
        line = lines[index]
        # 最后一篇以「）」结尾时并成「（…，本章读完）」。
        lines[index] = (
            line[:-1] + "，本章读完）" if line.endswith("）") else line + "（本章读完）"
        )
    if extra_note:
        lines.extend(part for part in extra_note.splitlines() if part.strip())
    return "\n".join(lines)[:2000]


def sync_day(
    plan: dict[str, Any],
    local_date: str,
    tasks: list[dict[str, Any]],
    *,
    event_id: str | None = None,
    card_id: str | None = None,
    extra_note: str = "",
) -> dict[str, Any]:
    """Create the day's calendar block and kanban card if they are missing."""
    study = [task for task in tasks if task["kind"] == "study"]
    minutes = sum(int(task.get("minutes", 0)) for task in study)
    result: dict[str, Any] = {"calendar": _status(True, event_id), "card": _status(True, card_id)}

    if event_id is None:
        try:
            from deeptutor.services.calendar.models import EventInput
            from deeptutor.services.calendar.store import get_calendar_store

            day = date.fromisoformat(local_date)
            payload = EventInput(
                title=_day_title(plan, local_date, minutes),
                notes=_day_note(tasks, extra_note),
                all_day=True,
                start=day.isoformat(),
                end=(day + timedelta(days=1)).isoformat(),
            )
            created = get_calendar_store().create(payload)
            result["calendar"] = _status(True, created["event"].id)
        except Exception as exc:  # noqa: BLE001 - surface, never block study
            result["calendar"] = _status(False, None, f"{type(exc).__name__}: {exc}")

    if card_id is None:
        try:
            from deeptutor.services.task_board import CreateCard, get_task_board_store

            card = CreateCard(
                title=f"{local_date} 学习：{plan['subject']}（{minutes} 分钟）",
                note=_day_note(tasks, extra_note),
                due_date=local_date,
            )
            board = get_task_board_store().create(card, "")
            created_id = _find_card_id(board, card.title)
            result["card"] = _status(True, created_id)
            if created_id and study and all(task.get("status") == "done" for task in study):
                refreshed = update_day_card(created_id, tasks, done=True)
                if not refreshed["ok"]:
                    result["card"] = _status(False, created_id, refreshed["error"])
        except Exception as exc:  # noqa: BLE001
            result["card"] = _status(False, None, f"{type(exc).__name__}: {exc}")
    return result


def _find_card_id(board: Any, title: str) -> str | None:
    """The newest live card with this title.

    Replanning archives last round's card but leaves it on the board; matching
    the first title hit linked new tasks to that archived card.
    """
    matches = [
        card
        for card in getattr(board, "cards", []) or []
        if getattr(card, "title", None) == title
        and not getattr(card, "archived", False)
    ]
    if not matches:
        return None
    newest = max(matches, key=lambda card: getattr(card, "created_at", "") or "")
    return getattr(newest, "id", None)


def update_day_card(
    card_id: str,
    tasks: list[dict[str, Any]],
    *,
    done: bool,
    extra_note: str = "",
) -> dict[str, Any]:
    """Keep the card's note and status in step with the plan."""
    try:
        from deeptutor.services.task_board import UpdateCard, get_task_board_store

        note = _day_note(tasks)
        if extra_note:
            note = f"{note}\n{extra_note}"[:2000]
        store = get_task_board_store()
        board = store.update(
            card_id,
            UpdateCard(note=note, status="done" if done else "todo"),
            workspace_id="",
        )
        return _status(True, card_id if board is not None else None)
    except Exception as exc:  # noqa: BLE001
        return _status(False, card_id, f"{type(exc).__name__}: {exc}")


def calendar_starts(
    event_ids: list[str], start: str, end: str, tzid: str = "UTC"
) -> dict[str, str]:
    """Where each mirrored event currently sits, keyed by event id.

    ``occurrences`` takes aware instants and returns ``{"occurrences": [...],
    "revision": ...}``; feeding it ISO date strings iterated the dict keys and
    silently produced an empty map. A missing or rotated event simply drops out
    of the map, but a calendar read failure raises so the caller does not report
    "already in sync" when it could not look at all.
    """
    wanted = {event_id for event_id in event_ids if event_id}
    if not wanted:
        return {}
    try:
        from deeptutor.services.calendar.recurrence import parse_instant
        from deeptutor.services.calendar.store import get_calendar_store

        zone = ZoneInfo(tzid)
        first = datetime.combine(date.fromisoformat(start), time(), zone).astimezone(
            timezone.utc
        )
        last = datetime.combine(
            date.fromisoformat(end) + timedelta(days=1), time(), zone
        ).astimezone(timezone.utc)
        payload = get_calendar_store().occurrences(first, last, tzid)
        occurrences = payload.get("occurrences", []) if isinstance(payload, dict) else payload
        starts: dict[str, str] = {}
        for occurrence in occurrences:
            if occurrence.event_id not in wanted:
                continue
            raw = str(occurrence.start)
            if "T" in raw:
                raw = parse_instant(raw).astimezone(zone).date().isoformat()
            starts[occurrence.event_id] = raw[:10]
        return starts
    except Exception as exc:  # noqa: BLE001 - the caller reports the failure
        raise CalendarReadError(f"{type(exc).__name__}: {exc}") from exc


def move_card(card_id: str, due_date: str) -> dict[str, Any]:
    try:
        from deeptutor.services.task_board import UpdateCard, get_task_board_store

        get_task_board_store().update(
            card_id, UpdateCard(due_date=due_date), workspace_id=""
        )
        return _status(True, card_id)
    except Exception as exc:  # noqa: BLE001
        return _status(False, card_id, f"{type(exc).__name__}: {exc}")


def drop_day(event_id: str | None, card_id: str | None) -> dict[str, Any]:
    """Remove a mirrored day during replanning."""
    result: dict[str, Any] = {"calendar": _status(True, event_id), "card": _status(True, card_id)}
    if event_id:
        try:
            from deeptutor.services.calendar.store import get_calendar_store

            get_calendar_store().delete(event_id)
        except Exception as exc:  # noqa: BLE001
            result["calendar"] = _status(False, event_id, f"{type(exc).__name__}: {exc}")
    if card_id:
        try:
            from deeptutor.services.task_board import UpdateCard, get_task_board_store

            get_task_board_store().update(card_id, UpdateCard(archived=True), workspace_id="")
        except Exception as exc:  # noqa: BLE001
            result["card"] = _status(False, card_id, f"{type(exc).__name__}: {exc}")
    return result


def ensure_subject_doc(plan: dict[str, Any]) -> dict[str, Any]:
    """One parent page per subject; chapter notes hang below it."""
    title = f"{plan['subject']} · 学习笔记"
    existing = plan.get("notes_parent_doc_id")
    try:
        from deeptutor.co_writer.storage import get_co_writer_storage

        storage = get_co_writer_storage()
        if existing and storage.load_document(existing) is not None:
            return _status(True, existing)
        for summary in storage.list_documents():
            if summary.title == title and not summary.parent_id:
                return _status(True, summary.id)
        document = storage.create_document(
            title=title,
            content=(
                f"# {plan['subject']} 学习笔记\n\n"
                f"学习区间：{plan['start_date']} → {plan['end_date']}\n\n"
                "每章一篇子页面，由学习计划生成草稿后确认归档。\n"
            ),
        )
        return _status(True, document.id)
    except Exception as exc:  # noqa: BLE001
        return _status(False, existing, f"{type(exc).__name__}: {exc}")


def create_note_doc(
    plan: dict[str, Any], chapter: str, title: str, body: str, parent_id: str | None
) -> dict[str, Any]:
    try:
        from deeptutor.co_writer.storage import get_co_writer_storage

        document = get_co_writer_storage().create_document(
            title=title,
            content=body[:MAX_NOTE_CHARS],
            parent_id=parent_id or None,
        )
        return _status(True, document.id)
    except Exception as exc:  # noqa: BLE001
        return _status(False, None, f"{type(exc).__name__}: {exc}")


__all__ = [
    "CalendarReadError",
    "PlanError",
    "calendar_starts",
    "create_note_doc",
    "drop_day",
    "ensure_subject_doc",
    "move_card",
    "sync_day",
    "update_day_card",
]
