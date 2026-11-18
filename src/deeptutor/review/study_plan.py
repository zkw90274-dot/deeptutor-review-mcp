"""Multi-week study planning on top of the review library.

Why this is not ``ReviewSpec.plan``: that field participates in the content
hash, so moving an end date after the learner has answered raises
``SpecHistoryConflict``. A study plan is mutable by nature, so it lives in its
own tables (``review_plans`` / ``review_daily_tasks`` / ``review_notes``) and
only references chapters by name.

The planner is deterministic and LLM-free: reading effort comes from the raw
material size, the daily budget is split into new-study and review time, and
the schedule packs whole sessions into days. It refuses to produce an
infeasible plan instead of silently compressing it.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

#: Reading speed for Chinese technical notes (characters per minute).
CHARS_PER_MINUTE = 500
#: Share of the daily budget that belongs to review rather than new material.
REVIEW_RATIO = 0.4
#: A session shorter than this is not worth opening the material for.
MIN_SESSION_MINUTES = 15
#: A day's tail shorter than this is too small to name as its own task.
MIN_TAIL_MINUTES = 10
#: Every seventh day runs at half capacity, so a missed day has somewhere to go.
BUFFER_EVERY = 7
DEFAULT_DAILY_MINUTES = 60
MIN_DAILY_MINUTES = 15
MAX_DAILY_MINUTES = 240
MAX_PLAN_DAYS = 365


class PlanError(ValueError):
    """The requested plan cannot be built as specified."""


@dataclass(frozen=True)
class Part:
    """One sitting's worth of reading, named from the note itself."""

    label: str
    minutes: int
    source: str = ""
    #: How many days this document has already been cut across (0 = whole).
    index: int = 0


@dataclass(frozen=True)
class ChapterWork:
    name: str
    minutes: int
    parts: tuple[Part, ...] = ()



def estimate_minutes(chars: int) -> int:
    """Reading minutes for a chapter, with a floor so tiny chapters still count."""
    return max(MIN_SESSION_MINUTES, int(math.ceil(chars / CHARS_PER_MINUTE)))


def chapter_work(subject: str, chapters: list[str]) -> list[ChapterWork]:
    from deeptutor.review.generate import chapter_workloads, infer_chapters

    try:
        available = infer_chapters(subject)
    except (FileNotFoundError, ValueError) as exc:
        raise PlanError(f"unknown or invalid knowledge base: {subject}") from exc
    known = {chapter.name for chapter in available}
    missing = [name for name in chapters if name not in known]
    if missing:
        raise PlanError(
            "unknown chapter(s): "
            + ", ".join(missing)
            + "; available: "
            + ", ".join(sorted(known))
        )
    workloads = chapter_workloads(subject)
    return [
        ChapterWork(name=name, minutes=estimate_minutes(workloads.get(name, 0)))
        for name in chapters
    ]


def _days(start: date, end: date) -> list[date]:
    if end < start:
        raise PlanError("end_date must not be before start_date")
    count = (end - start).days + 1
    if count > MAX_PLAN_DAYS:
        raise PlanError(f"plan spans {count} days; the limit is {MAX_PLAN_DAYS}")
    return [start + timedelta(days=offset) for offset in range(count)]


def _capacities(days: list[date], study_minutes_per_day: int) -> list[int]:
    capacities = []
    for index in range(len(days)):
        if (index + 1) % BUFFER_EVERY == 0:
            capacities.append(max(MIN_SESSION_MINUTES, study_minutes_per_day // 2))
        else:
            capacities.append(study_minutes_per_day)
    return capacities


def _sessions(work: ChapterWork, per_day: int) -> list[int]:
    parts = max(1, math.ceil(work.minutes / per_day))
    base = work.minutes // parts
    remainder = work.minutes - base * parts
    return [base + (1 if index < remainder else 0) for index in range(parts)]


_CN_NUMERALS = "一二三四五六七八九十"
_SLICE_SUFFIX = re.compile(r"（(?:[^（）]*，)?[一二三四五六七八九十]）$")


def _slice_label(part: Part, ordinal: int) -> str:
    """X（一）/（二）/（三）for a document read across several days.

    （上）/（下）only worked for two days; a long 习题精讲 read over three
    produced the nonsense 「（下）（上）」.
    """
    base = _SLICE_SUFFIX.sub("", part.label)
    mark = _CN_NUMERALS[ordinal - 1] if 1 <= ordinal <= 10 else str(ordinal)
    if base.endswith("）"):
        return f"{base[:-1]}，{mark}）"
    return f"{base}（{mark}）"


def _label_for(part: Part) -> str:
    return _slice_label(part, part.index + 1) if part.index else part.label


def _part_detail(part: Part, minutes: int) -> str:
    """What to open, and for how long (the planner overwrites nothing here)."""
    if part.source:
        return f"约 {minutes} 分钟；资料：{part.source}"
    return f"约 {minutes} 分钟"


def _pack_tasks(
    work: list[ChapterWork],
    days: list[date],
    capacities: list[int],
    study_per_day: int,
    *,
    split: bool,
) -> list[dict[str, Any]]:
    """Fill days in order; ``split`` lets a session continue on the next day.

    Greedy whole-session packing can overflow a short buffer day while leaving
    usable minutes behind on earlier days, even when total capacity suffices.
    The first pass keeps sessions whole; only a would-be overflow retries with
    session splitting, so ordinary plans keep their clean session boundaries.
    """
    tasks: list[dict[str, Any]] = []
    cursor = 0
    remaining = capacities[0]
    day_capacity = capacities[0]
    current_day = days[0]
    seq = 1

    def next_day() -> None:
        """Move to the next day that has study minutes.

        A zero-capacity day means "no session today" (an explicit timetable can
        skip a weekday). Landing on one let the tiny-tail branch emit a few
        minutes on a day the learner never offered, so zero days are skipped.
        """
        nonlocal cursor, remaining, day_capacity, current_day, seq
        while True:
            cursor += 1
            if cursor >= len(days):
                raise PlanError("schedule overflow")
            if capacities[cursor] > 0:
                break
        current_day = days[cursor]
        remaining = capacities[cursor]
        day_capacity = capacities[cursor]
        seq = 1

    def emit(kind: str, title: str, detail: str, chapter: str, minutes: int) -> None:
        nonlocal seq
        tasks.append(
            {
                "local_date": current_day.isoformat(),
                "seq": seq,
                "kind": kind,
                "title": title,
                "detail": detail,
                "chapter": chapter,
                "minutes": minutes,
                "status": "todo",
            }
        )
        seq += 1

    started: set[str] = set()

    def begin(item: ChapterWork) -> None:
        """Emit「开始：」only when the chapter's first reading task really lands."""
        if item.name not in started:
            started.add(item.name)
            emit("milestone", f"开始：{item.name}", "", item.name, 0)

    for item in work:
        if remaining <= 0:
            next_day()
        if item.parts:
            queue = list(item.parts)
            while queue:
                if remaining <= 0:
                    next_day()
                part = queue[0]
                leftover = part.minutes - remaining
                if part.minutes <= remaining:
                    begin(item)
                    emit(
                        "study",
                        f"{item.name} · {_label_for(part)}",
                        _part_detail(part, part.minutes),
                        item.name,
                        part.minutes,
                    )
                    remaining -= part.minutes
                    queue.pop(0)
                    continue
                has_next = any(
                    capacities[pos] > 0 for pos in range(cursor + 1, len(days))
                )
                if leftover < MIN_TAIL_MINUTES or not has_next:
                    # The part nearly fits (a sliver of the day is left) or this
                    # is the final session: read it whole instead of leaving a
                    # four-minute continuation for tomorrow.
                    begin(item)
                    emit(
                        "study",
                        f"{item.name} · {_label_for(part)}",
                        _part_detail(part, part.minutes),
                        item.name,
                        part.minutes,
                    )
                    remaining -= part.minutes
                    queue.pop(0)
                    continue
                if remaining >= MIN_TAIL_MINUTES:
                    begin(item)
                    emit(
                        "study",
                        f"{item.name} · {_slice_label(part, part.index + 1)}",
                        _part_detail(part, remaining),
                        item.name,
                        remaining,
                    )
                    queue[0] = Part(
                        label=part.label,
                        minutes=leftover,
                        source=part.source,
                        index=part.index + 1,
                    )
                    remaining = 0
                    next_day()
                    continue
                # Less than MIN_TAIL_MINUTES left: a timetable day ends there.
                # Emitting the part anyway overflowed a 90-minute day by 9.
                remaining = 0
                next_day()
            emit("milestone", f"完成：{item.name}", "", item.name, 0)
            continue
        sessions = _sessions(item, study_per_day)
        for index, minutes in enumerate(sessions, start=1):
            label = (
                f"{item.name} · 第 {index}/{len(sessions)} 段"
                if len(sessions) > 1
                else item.name
            )
            left = minutes
            first = True
            while left > 0:
                if remaining <= 0:
                    next_day()
                if not split and left > remaining:
                    next_day()
                    continue
                take = min(left, remaining)
                begin(item)
                emit(
                    "study",
                    label if first else f"{label}（续）",
                    f"约 {take} 分钟；读 raw 资料，完成后标记本节",
                    item.name,
                    take,
                )
                remaining -= take
                left -= take
                first = False
                if left > 0:
                    next_day()
        emit("milestone", f"完成：{item.name}", "", item.name, 0)
    return tasks


def study_minutes_per_day(daily_minutes: int, review_ratio: float = REVIEW_RATIO) -> int:
    """Study budget per day once the review share is set aside."""
    return max(
        MIN_SESSION_MINUTES, int(round(daily_minutes * (1.0 - review_ratio)))
    )


def capacities_from_budget(
    days: list[date], weekly_budget: dict[int, int], review_ratio: float = 0.0
) -> list[int]:
    """Per-day study minutes from a weekday budget; a missing weekday is a day off.

    ``weekly_budget`` maps ``date.weekday()`` (0 = Monday) to the minutes the
    learner has that day. An explicit timetable means the minutes themselves, so
    ``review_ratio`` defaults to 0: "90 minutes on Monday" is not "90 minutes of
    which 40% is review".
    """
    capacities: list[int] = []
    for day in days:
        minutes = int(weekly_budget.get(day.weekday(), 0))
        if minutes <= 0:
            capacities.append(0)
            continue
        capacities.append(
            minutes if not review_ratio else study_minutes_per_day(minutes, review_ratio)
        )
    return capacities


def weekly_end_date(
    start: date,
    required: int,
    weekly_budget: dict[int, int],
    review_ratio: float = 0.0,
    limit: int = MAX_PLAN_DAYS,
) -> date:
    """First day on which a weekday budget absorbs ``required`` study minutes."""
    total = 0
    for offset in range(limit):
        day = start + timedelta(days=offset)
        total += capacities_from_budget([day], weekly_budget, review_ratio)[0]
        if total >= required:
            return day
    raise PlanError(f"a weekly budget cannot absorb {required} minutes in {limit} days")


def build_schedule(
    work: list[ChapterWork],
    start: date,
    end: date,
    daily_minutes: int = DEFAULT_DAILY_MINUTES,
    review_ratio: float = REVIEW_RATIO,
    first_day_used: int = 0,
    day_capacities: list[int] | None = None,
) -> dict[str, Any]:
    """Build daily study tasks, or explain why the request is infeasible.

    ``day_capacities`` replaces the uniform budget with explicit study minutes per
    day (0 = a day off). An uneven week - "90 minutes Mon/Wed/Fri, 120 Tue/Thu,
    120 Saturday" - cannot be expressed by one daily number, and averaging it
    would put work on days the learner never offered.
    """
    days = _days(start, end)
    if day_capacities is not None:
        if len(day_capacities) != len(days):
            raise PlanError("day_capacities must cover every day of the plan")
        capacities = [max(0, int(minutes)) for minutes in day_capacities]
        if sum(capacities) == 0:
            raise PlanError("day_capacities has no study minutes")
        study_per_day = max(MIN_SESSION_MINUTES, max(capacities))
    else:
        if not MIN_DAILY_MINUTES <= daily_minutes <= MAX_DAILY_MINUTES:
            raise PlanError(
                f"daily_minutes must be between {MIN_DAILY_MINUTES} and {MAX_DAILY_MINUTES}"
            )
        study_per_day = study_minutes_per_day(daily_minutes, review_ratio)
        capacities = _capacities(days, study_per_day)
    if first_day_used > 0:
        # Work already done today still consumes today's study budget.
        capacities[0] = max(0, capacities[0] - first_day_used)
    required = sum(item.minutes for item in work)
    capacity = sum(capacities)

    if required > capacity:
        # How many extra days would the same budget need?
        extra = required - capacity
        average = max(MIN_SESSION_MINUTES, sum(capacities) // max(1, len(days)))
        extra_days = int(math.ceil(extra / average))
        return {
            "feasible": False,
            "required_minutes": required,
            "capacity_minutes": capacity,
            "study_minutes_per_day": study_per_day,
            "suggested_end_date": (end + timedelta(days=extra_days)).isoformat(),
            "tasks": [],
        }

    try:
        tasks = _pack_tasks(work, days, capacities, study_per_day, split=False)
    except PlanError:
        tasks = _pack_tasks(work, days, capacities, study_per_day, split=True)

    return {
        "feasible": True,
        "required_minutes": required,
        "capacity_minutes": capacity,
        "study_minutes_per_day": study_per_day,
        "tasks": tasks,
    }


def new_plan_id() -> str:
    return "plan_" + uuid.uuid4().hex[:12]


def note_body(plan: dict[str, Any], chapter: str, spec: Any | None,
              tasks: list[dict[str, Any]]) -> str:
    """Grounded draft: review-pack concepts and pitfalls, never invented text."""
    lines = [f"# {chapter} · 学习笔记（草稿）", ""]
    lines.append(f"- 科目：{plan['subject']}")
    lines.append(f"- 学习区间：{plan['start_date']} → {plan['end_date']}")
    lines.append("")

    lines.append("## 知识点地图")
    if spec is None:
        lines.append("- 该章尚未生成复习包；先生成后再补全本节。")
    else:
        for concept in spec.concepts:
            definition = (concept.definition or "").strip()
            lines.append(f"- **{concept.name}**" + (f"：{definition}" if definition else ""))
    lines.append("")

    lines.append("## 核心公式")
    formulas: list[str] = []
    if spec is not None:
        for concept in spec.concepts:
            for formula in concept.formulas:
                latex = getattr(formula, "latex", "") or ""
                if latex:
                    formulas.append(f"- `{latex}`")
    lines.extend(formulas or ["- （本章复习包暂无可摘录公式）"])
    lines.append("")

    lines.append("## 易错点")
    pitfalls: list[str] = []
    if spec is not None:
        for item in spec.items:
            aids = getattr(item, "study_aids", None)
            for pitfall in getattr(aids, "pitfalls", []) or []:
                pitfalls.append(f"- {pitfall}")
    lines.extend(pitfalls[:8] or ["- （暂无结构化易错点，复习时补充）"])
    lines.append("")

    lines.append("## 待复习清单")
    reviewed = [task for task in tasks if task["kind"] == "study" and task["status"] == "done"]
    for task in reviewed:
        lines.append(f"- {task['title']}（{task['local_date']}）")
    if not reviewed:
        lines.append("- （本章尚未标记完成任何学习段）")
    lines.append("")
    lines.append("---")
    lines.append("来源：DeepTutor 知识库 / Review Studio；草稿由学习计划生成，确认后归档。")
    return "\n".join(lines)
