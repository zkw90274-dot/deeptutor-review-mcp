"""Today's queue: due review, new material, and a bounded backlog policy.

The queue is derived, never stored. Every call replays the append-only event
log, so a stale materialized state cannot change what the learner sees.

Budget policy (roadmap §3, Phase 0.5 defaults):

* hard cap 12 items, whatever the plan says;
* target 70% review / 30% new, with unused slots donated to the other side;
* when overdue review exceeds twice the review allocation, new material is
  suspended until the backlog is under control;
* one item per concept is preferred in adjacent positions; concepts with more
  items rotate by "least recently shown";
* when ``plan.exam_date`` is set, the minute budget scales for the sprint
  (4-14 days out) and the final week (0-3 days out); the item cap stays 12.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal
from zoneinfo import ZoneInfo

from deeptutor.review.adapter import retrievability, scheduler_for
from deeptutor.review.core_state import ConceptState, states_for_spec
from deeptutor.review.models import Concept, Item, ReviewSpec
from deeptutor.review.timeutil import iso, local_date, tomorrow_window, zone

MAX_ITEMS = 12
REVIEW_RATIO = 0.7
BACKLOG_MULTIPLIER = 2.0
WEAKNESS_LIMIT = 5
RECOVERY_AFTER_DAYS = 3
RECOVERY_LIMIT = 8
DAY_SECONDS = 86400.0

# Exam countdown (Phase 34). The minute budget is the only thing scaled: the
# 12-item hard cap is an invariant of the queue, not a function of the plan.
FINAL_WEEK_DAYS = 3
SPRINT_DAYS = 14
SPRINT_FACTOR = 1.5
FINAL_WEEK_FACTOR = 2.0
SPRINT_CAP_MINUTES = 30
FINAL_WEEK_CAP_MINUTES = 45
# Final-week ordering: a sub-millisecond nudge that only breaks exact ties in
# "least recently shown", so weaker concepts surface first without displacing
# genuinely more-overdue material.
WEAK_TIEBREAK_BOOST = 1e-6


def _exam_plan(
    spec: ReviewSpec, now: float, tz: ZoneInfo
) -> tuple[dict[str, Any], int]:
    """Return the ``exam`` block and the minute budget it implies.

    ``days_left`` is counted in the plan timezone, the same day boundary the
    rest of the queue uses. A date in the past keeps ``normal`` mode: the
    sprint is over, and a negative countdown must not keep inflating the plan.
    """
    base = spec.plan.daily_minutes
    exam_date = spec.plan.exam_date
    if exam_date is None:
        return {"exam_date": None, "days_left": None, "mode": "normal"}, base
    days_left = (exam_date - local_date(now, tz)).days
    if 0 <= days_left <= FINAL_WEEK_DAYS:
        mode = "final_week"
        planned = min(FINAL_WEEK_CAP_MINUTES, int(round(base * FINAL_WEEK_FACTOR)))
    elif FINAL_WEEK_DAYS < days_left <= SPRINT_DAYS:
        mode = "sprint"
        planned = min(SPRINT_CAP_MINUTES, int(round(base * SPRINT_FACTOR)))
    else:
        mode = "normal"
        planned = base
    return {
        "exam_date": exam_date.isoformat(),
        "days_left": days_left,
        "mode": mode,
    }, planned


def weakness_score(state: Any, predicted: float, task: Any | None, *, now: float) -> float:
    """Weakness used by the "practise this again" entry point.

    Roadmap formula: ``0.55*(1-mastery) + 0.25*lapse_rate + 0.20*overdue_ratio``.
    Core ``ReviewTask`` data is exposed alongside the score, but the score
    itself stays deterministic and local so an unavailable core degrades to
    the same formula rather than a different one.
    """
    mastery = min(1.0, max(0.0, float(predicted)))
    review_count = max(0, int(getattr(state, "review_count", 0) or 0))
    lapse_count = max(0, int(getattr(state, "lapse_count", 0) or 0))
    attempts = review_count + lapse_count
    lapse_rate = (lapse_count / attempts) if attempts else 0.0
    due_at = float(getattr(state, "next_review_at", now) or now)
    if due_at >= now:
        overdue_ratio = 0.0
    else:
        anchor = getattr(state, "last_scheduled_at", None) or getattr(
            state, "last_review_at", None
        )
        interval = max(86400.0, due_at - float(anchor)) if anchor else 86400.0
        overdue_ratio = min(1.0, (now - due_at) / interval)
    score = 0.55 * (1.0 - mastery) + 0.25 * lapse_rate + 0.20 * overdue_ratio
    return round(min(1.0, max(0.0, score)), 3)


def _item_payload(
    item: Item,
    concept: Concept,
    *,
    status: str,
    due_at: float,
    predicted: float,
    last_seen_at: float | None,
    tz: ZoneInfo,
    state_source: str = "review",
    state_reason: str = "",
    weakness: float = 0.0,
    task: Any | None = None,
) -> dict[str, Any]:
    reason = {
        "due": "到期复习",
        "new": "新内容",
        "repeat": "本会话再练一次",
    }.get(status, status)
    return {
        "item_id": item.id,
        "concept_id": concept.id,
        "concept_key": concept.key,
        "concept_name": concept.name,
        "kind": item.kind,
        "stem": item.stem,
        "options": [{"label": option.label, "text": option.text} for option in item.options],
        "difficulty": item.difficulty,
        "status": status,
        "reason": reason,
        "state_source": state_source,
        **({"state_reason": state_reason} if state_reason else {}),
        "due_at": iso(due_at, tz),
        "due_at_ts": due_at,
        "predicted_retrievability": round(predicted, 4),
        "weakness": weakness,
        **(
            {
                "core_priority": int(getattr(task, "priority", 0) or 0),
                "core_forgetting_risk": round(
                    float(getattr(task, "forgetting_risk", 0.0) or 0.0), 3
                ),
                "core_reason": str(getattr(task, "reason", "") or ""),
            }
            if task is not None
            else {}
        ),
        "last_seen_at": iso(last_seen_at, tz),
    }


def _interleave(
    entries: list[dict[str, Any]],
    concepts: dict[str, list[dict[str, Any]]],
    priority: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Round-robin by concept, preserving the caller's within-concept order.

    ``priority`` is a small tiebreak applied only when two concept groups have
    the same size, so an exam-week weakness boost survives the regrouping.
    """
    queues = {concept_id: list(rows) for concept_id, rows in concepts.items()}
    boosts = priority or {}
    # Start with the largest concept group so its items are separated by the
    # smaller groups (classic "rearrange k distance apart" greedy order).
    order = sorted(
        queues,
        key=lambda concept_id: (
            -len(queues[concept_id]),
            boosts.get(concept_id, 0.0),
            concept_id,
        ),
    )
    result: list[dict[str, Any]] = []
    while True:
        added = False
        for concept_id in order:
            rows = queues[concept_id]
            if not rows:
                continue
            result.append(rows.pop(0))
            added = True
        if not added:
            break
    # The round-robin above can exhaust one concept early; keep deterministic.
    if len(result) != len(entries):
        seen = {row["item_id"] for row in result}
        result.extend(row for row in entries if row["item_id"] not in seen)
    return result


def build_today(
    spec: ReviewSpec,
    store: Any,
    *,
    now: float,
    session_id: str,
    states: dict[str, ConceptState] | None = None,
    record_views: bool = True,
    allow_recovery: bool = True,
    activity: dict[tuple[str, str], float] | None = None,
    candidate_preview: bool = False,
) -> dict[str, Any]:
    tz = zone(spec.plan.timezone)
    exam, planned_minutes = _exam_plan(spec, now, tz)
    scheduler = scheduler_for(spec)
    resolved = states or states_for_spec(spec, store, now=now)

    concept_states: dict[str, Any] = {}
    state_sources: dict[str, str] = {}
    state_initial: dict[str, bool] = {}
    state_reasons: dict[str, str] = {}
    state_tasks: dict[str, Any | None] = {}
    concept_weakness: dict[str, float] = {}
    concept_predicted: dict[str, float] = {}
    activity = store.item_activity(spec.id) if activity is None else activity
    concept_keys = {concept.key for concept in spec.concepts}
    last_seen_by_item: dict[str, float] = {}
    for (concept_key, item_id), timestamp in activity.items():
        if concept_key in concept_keys:
            last_seen_by_item[item_id] = max(timestamp, last_seen_by_item.get(item_id, timestamp))
    for concept in spec.concepts:
        concept_state = resolved[concept.id]
        concept_states[concept.id] = concept_state.state
        state_sources[concept.id] = concept_state.source
        state_initial[concept.id] = concept_state.initial
        state_reasons[concept.id] = concept_state.reason
        state_tasks[concept.id] = concept_state.task
        predicted = retrievability(
            concept, concept_state.state, now=now, scheduler=scheduler
        )
        concept_predicted[concept.id] = predicted
        concept_weakness[concept.id] = weakness_score(
            concept_state.state, predicted, concept_state.task, now=now
        )

    review_rows: list[dict[str, Any]] = []
    new_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []

    for concept in spec.concepts:
        state = concept_states[concept.id]
        concept_due = state.next_review_at <= now
        predicted = retrievability(concept, state, now=now, scheduler=scheduler)
        for item in spec.items_for_concept(concept.id):
            last_seen = last_seen_by_item.get(item.id)
            if last_seen is None:
                core_is_known = (
                    state_sources[concept.id] == "core"
                    and not state_initial[concept.id]
                )
                if core_is_known and not concept_due:
                    # The core already knows this objective and it is not due;
                    # it is not "new" just because Review has no local event.
                    continue
                if core_is_known and concept_due:
                    review_rows.append(
                        _item_payload(
                            item,
                            concept,
                            status="due",
                            due_at=state.next_review_at,
                            predicted=predicted,
                            last_seen_at=None,
                            tz=tz,
                            state_source="core",
                            weakness=concept_weakness[concept.id],
                            task=state_tasks[concept.id],
                        )
                    )
                    continue
                new_rows.append(
                    _item_payload(
                        item,
                        concept,
                        status="new",
                        due_at=now,
                        predicted=1.0,
                        last_seen_at=None,
                        tz=tz,
                        state_source=state_sources[concept.id],
                        state_reason=state_reasons[concept.id],
                        weakness=concept_weakness[concept.id],
                        task=state_tasks[concept.id],
                    )
                )
            elif concept_due:
                review_rows.append(
                    _item_payload(
                        item,
                        concept,
                        status="due",
                        due_at=state.next_review_at,
                        predicted=predicted,
                        last_seen_at=last_seen,
                        tz=tz,
                        state_source=state_sources[concept.id],
                        state_reason=state_reasons[concept.id],
                        weakness=concept_weakness[concept.id],
                        task=state_tasks[concept.id],
                    )
                )

    # "Again" is repeated once in the same session, but only once: the queue
    # must not become an infinite loop for a learner who keeps failing.
    session_events = store.events_for_session(session_id) if session_id else []
    again_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in session_events:
        if event["rating"] == "again":
            again_by_item[str(event["item_id"])].append(event)
    queued_ids = {row["item_id"] for row in review_rows + new_rows}
    for item_id, again_events in again_by_item.items():
        if len(again_events) != 1 or item_id in queued_ids:
            continue
        try:
            item = spec.item_by_id(item_id)
        except KeyError:
            continue
        concept = spec.concept_by_id(item.concept_ids[0])
        later = [
            event
            for event in session_events
            if str(event["item_id"]) == item_id and float(event["ts"]) > float(again_events[0]["ts"])
        ]
        if later:
            continue
        state = concept_states[concept.id]
        repeat_rows.append(
            _item_payload(
                item,
                concept,
                status="repeat",
                due_at=state.next_review_at,
                predicted=retrievability(concept, state, now=now, scheduler=scheduler),
                last_seen_at=float(again_events[0]["ts"]),
                tz=tz,
                state_source=state_sources[concept.id],
                state_reason=state_reasons[concept.id],
                weakness=concept_weakness[concept.id],
                task=state_tasks[concept.id],
            )
        )

    # "Least recently shown" rotation: without this, a concept with more
    # items than the daily slice would starve the tail forever.
    last_views = store.last_view_map(spec.id)
    last_activity = max(activity.values(), default=None)
    gap_days = (
        max(0.0, (now - last_activity) / DAY_SECONDS)
        if last_activity is not None
        else None
    )
    recovering = (
        allow_recovery
        and gap_days is not None
        and gap_days >= RECOVERY_AFTER_DAYS
    )
    weak_boost: dict[str, float] = {}
    if recovering:
        # Recovery is not a backlog report: show the most important handful
        # and do not let new material push them out.
        review_rows.sort(key=lambda row: (-row["weakness"], row["item_id"]))
    else:
        # In the final week, give the weakest third of concepts a tiny edge
        # when two items were last seen at exactly the same instant. The boost
        # is 1e-6 against timestamps measured in seconds, so it is a tiebreak
        # only: a more-overdue item still sorts first.
        if exam["mode"] == "final_week" and concept_weakness:
            ranked = sorted(
                concept_weakness, key=lambda cid: (-concept_weakness[cid], cid)
            )
            for concept_id in ranked[: max(1, (len(ranked) + 2) // 3)]:
                weak_boost[concept_id] = -WEAK_TIEBREAK_BOOST
        review_rows.sort(
            key=lambda row: (
                last_views.get(row["item_id"], 0.0),
                weak_boost.get(row["concept_id"], 0.0),
                row["concept_id"],
                row["item_id"],
            )
        )
    new_rows.sort(
        key=lambda row: (last_views.get(row["item_id"], 0.0), row["item_id"])
    )

    if spec.items:
        total_budget = min(RECOVERY_LIMIT if recovering else MAX_ITEMS, max(1, len(spec.items)))
    else:
        total_budget = 0
    if recovering:
        review_cap = total_budget
        new_cap = 0
        backlog = False
    else:
        review_cap = int(round(total_budget * REVIEW_RATIO))
        new_cap = total_budget - review_cap
        backlog = len(review_rows) > BACKLOG_MULTIPLIER * max(1, review_cap)
        if backlog:
            new_cap = 0

    review_take = min(len(review_rows), review_cap)
    new_take = min(len(new_rows), new_cap)
    # Donate unused slots to the other side; the hard cap still applies.
    # Recovery deliberately does not donate: a returning learner should not
    # get new material just because the due pool is shorter than 8.
    if review_take < review_cap and not recovering:
        new_take = min(len(new_rows), total_budget - review_take)
    elif new_take < new_cap and not backlog:
        review_take = min(len(review_rows), total_budget - new_take)

    selected_review = review_rows[:review_take]
    selected_new = new_rows[:new_take]
    remaining = total_budget - len(selected_review) - len(selected_new)
    if remaining > 0:
        # Fill any leftover from whichever pool still has content.
        selected_review.extend(review_rows[review_take : review_take + remaining])
        remaining = total_budget - len(selected_review) - len(selected_new)
        if remaining > 0 and not recovering:
            selected_new.extend(new_rows[new_take : new_take + remaining])

    # Round-robin within each pool by concept, then interleave the two pools.
    def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[row["concept_id"]].append(row)
        return buckets

    review_ordered = _interleave(
        selected_review, grouped(selected_review), weak_boost
    )
    new_ordered = _interleave(selected_new, grouped(selected_new), weak_boost)

    entries: list[dict[str, Any]] = []
    review_index = 0
    new_index = 0
    # 7 review then 3 new is a deterministic approximation of 70:30; when a
    # pool runs dry the other keeps going.
    while review_index < len(review_ordered) or new_index < len(new_ordered):
        for _ in range(int(round(10 * REVIEW_RATIO))):
            if review_index < len(review_ordered):
                entries.append(review_ordered[review_index])
                review_index += 1
        for _ in range(int(round(10 * (1 - REVIEW_RATIO)))):
            if new_index < len(new_ordered):
                entries.append(new_ordered[new_index])
                new_index += 1
        if review_index >= len(review_ordered) and new_index >= len(new_ordered):
            break

    used = {row["item_id"] for row in entries}
    for row in repeat_rows:
        if len(entries) >= total_budget:
            break
        if row["item_id"] not in used:
            retry_row = dict(row)
            # Number of items shown before this retest. RS-1002 targets >=5;
            # a very short session cannot manufacture filler items, so the
            # actual gap is reported instead of pretending it was 5.
            retry_row["retry_gap"] = len(entries)
            entries.append(retry_row)
            used.add(row["item_id"])

    if candidate_preview:
        due_ids = {row["item_id"] for row in review_rows}
        new_ids = {row["item_id"] for row in new_rows} - due_ids
        return {
            "items": entries,
            "due": len(due_ids),
            "new": len(new_ids),
            "planned_minutes": planned_minutes,
            "budget": total_budget,
        }

    if record_views:
        store.record_views(
            spec.id, session_id, [row["item_id"] for row in entries], now=now
        )

    tomorrow_date, end_today, end_tomorrow = tomorrow_window(now, spec.plan.timezone)
    due_tomorrow = 0
    for concept in spec.concepts:
        due_at = concept_states[concept.id].next_review_at
        if end_today <= due_at < end_tomorrow:
            due_tomorrow += len(spec.items_for_concept(concept.id))

    estimated_minutes = max(
        1,
        int(round(planned_minutes * len(entries) / max(1, total_budget))),
    ) if entries else 0

    weak_concepts = []
    for concept in spec.concepts:
        task = state_tasks.get(concept.id)
        weak_concepts.append(
            {
                "concept_id": concept.id,
                "concept_key": concept.key,
                "concept_name": concept.name,
                "weakness": concept_weakness[concept.id],
                "predicted_retrievability": round(concept_predicted[concept.id], 4),
                "state_source": state_sources[concept.id],
                "due_at": iso(concept_states[concept.id].next_review_at, tz),
                "due_at_ts": concept_states[concept.id].next_review_at,
                "item_ids": [item.id for item in spec.items_for_concept(concept.id)],
                "item_count": len(spec.items_for_concept(concept.id)),
                **(
                    {
                        "core_priority": int(getattr(task, "priority", 0) or 0),
                        "core_forgetting_risk": round(
                            float(getattr(task, "forgetting_risk", 0.0) or 0.0), 3
                        ),
                        "core_reason": str(getattr(task, "reason", "") or ""),
                    }
                    if task is not None
                    else {}
                ),
            }
        )
    weak_concepts.sort(key=lambda row: (-row["weakness"], row["concept_id"]))
    selected_counts = {"due": 0, "new": 0, "repeat": 0}
    for row in entries:
        if row["status"] in selected_counts:
            selected_counts[row["status"]] += 1
    recovery = {
        "active": recovering,
        "days_since_last_review": round(gap_days, 1) if gap_days is not None else None,
        "limit": total_budget if recovering else None,
    }
    if recovering:
        # RS-1005 is explicit: a returning learner must not be shown a backlog
        # number. ``counts.due`` therefore reports what this session selected,
        # not how many cards exist.
        counts = {**selected_counts, "selected": len(entries)}
        empty_message = (
            f"已恢复复习。先完成最重要的 {total_budget} 项，不需要补完所有积压。"
            if total_budget
            else "今天是恢复日，暂时没有到期内容。"
        )
    else:
        counts = {
            "due": len(review_rows),
            "new": len(new_rows),
            "repeat": len(repeat_rows),
            "selected": len(entries),
        }
        empty_message = "今天没有到期内容。可以生成新内容，或明天再来。"

    return {
        "spec_id": spec.id,
        "session_id": session_id,
        "generated_at": iso(now, tz),
        "timezone": spec.plan.timezone,
        "budget": total_budget,
        "planned_minutes": planned_minutes,
        "estimated_minutes": estimated_minutes,
        "exam": exam,
        "items": entries,
        "counts": counts,
        "backlog": backlog,
        "recovery": recovery,
        "weak_concepts": weak_concepts[:WEAKNESS_LIMIT],
        "tomorrow": {
            "date": tomorrow_date,
            "due_items": due_tomorrow,
        },
        "empty_message": None if entries else empty_message,
    }


def concept_state_map(
    spec: ReviewSpec,
    store: Any,
    *,
    now: float,
    states: dict[str, ConceptState] | None = None,
) -> dict[str, Any]:
    """Return every concept state; used by export snapshots and completion."""
    resolved = states or states_for_spec(spec, store, now=now)
    return {concept.id: resolved[concept.id].state for concept in spec.concepts}


def build_weakness(
    spec: ReviewSpec,
    store: Any,
    *,
    now: float,
    states: dict[str, ConceptState] | None = None,
    limit: int = WEAKNESS_LIMIT,
) -> list[dict[str, Any]]:
    """Rank concepts by the W5 weakness formula without recording views."""
    resolved = states or states_for_spec(spec, store, now=now)
    scheduler = scheduler_for(spec)
    tz = zone(spec.plan.timezone)
    rows: list[dict[str, Any]] = []
    for concept in spec.concepts:
        concept_state = resolved[concept.id]
        predicted = retrievability(
            concept, concept_state.state, now=now, scheduler=scheduler
        )
        task = concept_state.task
        rows.append(
            {
                "concept_id": concept.id,
                "concept_key": concept.key,
                "concept_name": concept.name,
                "weakness": weakness_score(
                    concept_state.state, predicted, task, now=now
                ),
                "predicted_retrievability": round(predicted, 4),
                "state_source": concept_state.source,
                "due_at": iso(concept_state.state.next_review_at, tz),
                "due_at_ts": concept_state.state.next_review_at,
                "item_ids": [item.id for item in spec.items_for_concept(concept.id)],
                "item_count": len(spec.items_for_concept(concept.id)),
                **(
                    {
                        "core_priority": int(getattr(task, "priority", 0) or 0),
                        "core_forgetting_risk": round(
                            float(getattr(task, "forgetting_risk", 0.0) or 0.0), 3
                        ),
                        "core_reason": str(getattr(task, "reason", "") or ""),
                    }
                    if task is not None
                    else {}
                ),
            }
        )
    rows.sort(key=lambda row: (-row["weakness"], row["concept_id"]))
    return rows[: max(1, int(limit))]


SessionMode = Literal["mixed", "review", "new", "weak"]
SESSION_MODES: tuple[str, ...] = ("mixed", "review", "new", "weak")


def build_session_plan(
    spec: ReviewSpec,
    store: Any,
    *,
    now: float,
    states: dict[str, ConceptState] | None = None,
    mode: SessionMode = "mixed",
    limit: int | None = None,
    minutes: int | None = None,
    session_id: str = "",
) -> dict[str, Any]:
    """Build a bounded session without recording views.

    ``mixed`` is the existing Today policy (12 items / plan.daily_minutes /
    70:30 / backlog degradation). ``review`` and ``new`` filter that plan to
    one pool. ``weak`` keeps the weakest concepts first, which is the data
    behind the "practise this again" entry point.

    ``limit`` and ``minutes`` can only shrink the plan; they never extend it
    beyond Today's budget. This is deliberately a read-only builder: callers
    that show items must still call Today (or explicitly record views) when
    the session actually starts.
    """
    if mode not in SESSION_MODES:
        raise ValueError(f"unknown session mode: {mode}")
    today = build_today(
        spec,
        store,
        now=now,
        session_id=session_id,
        states=states,
        record_views=False,
        # Recovery trims the default mixed plan. An explicit "give me new
        # material" or "practise weak concepts" request is an override: the
        # learner asked for that pool, so silently returning nothing for the
        # whole recovery window would be worse than trusting the request.
        allow_recovery=mode == "mixed",
    )
    items = list(today["items"])
    weakness_by_concept = {
        row["concept_id"]: row["weakness"]
        for row in today.get("weak_concepts", [])
    }
    if mode == "review":
        items = [row for row in items if row["status"] == "due"]
    elif mode == "new":
        items = [row for row in items if row["status"] == "new"]
    elif mode == "weak":
        weak_rows = [
            row for row in today.get("weak_concepts", []) if row["weakness"] > 0.0
        ]
        weak_ids = {row["concept_id"] for row in weak_rows}
        items = [row for row in items if row["concept_id"] in weak_ids]
        items.sort(
            key=lambda row: (
                -weakness_by_concept.get(row["concept_id"], 0.0),
                row["item_id"],
            )
        )

    cap = MAX_ITEMS
    if limit is not None:
        cap = min(cap, max(1, int(limit)))
    if minutes is not None:
        # The ratio is taken against the same (possibly exam-scaled) budget
        # Today reported, so an explicit minute request stays proportional.
        planned = max(1, today["planned_minutes"])
        requested = max(1, min(int(minutes), planned))
        cap = min(cap, max(1, int(round(MAX_ITEMS * requested / planned))))
    items = items[:cap]

    counts = {"due": 0, "new": 0, "repeat": 0}
    for row in items:
        if row["status"] in counts:
            counts[row["status"]] += 1
    estimated_minutes = (
        max(
            1,
            int(
                round(
                    today["planned_minutes"]
                    * len(items)
                    / max(1, today["budget"])
                )
            ),
        )
        if items
        else 0
    )
    recovery = today.get("recovery", {"active": False})
    # The message must describe what the caller actually got, not what Today
    # would have returned before the mode filter.
    empty_message = {
        "mixed": "今天没有到期内容。可以生成新内容，或明天再来。",
        "review": "今天没有到期内容。",
        "new": "没有可用的新内容。",
        "weak": "当前没有需要再练的薄弱概念。",
    }[mode]
    if not items and mode == "mixed" and recovery.get("active"):
        empty_message = today.get("empty_message") or empty_message
    return {
        "spec_id": spec.id,
        "session_id": session_id,
        "generated_at": today["generated_at"],
        "timezone": spec.plan.timezone,
        "mode": mode,
        "budget": today["budget"],
        "selected": len(items),
        "planned_minutes": today["planned_minutes"],
        "estimated_minutes": estimated_minutes,
        "items": items,
        "counts": counts,
        "backlog": today["backlog"],
        "recovery": recovery,
        "weak_concepts": today.get("weak_concepts", []),
        "empty_message": None if items else empty_message,
    }
