from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from functools import cache
from typing import Any

from deeptutor.review.core_state import load_core_progress_read_only, states_for_spec
from deeptutor.review.export_tree import split_pack_title
from deeptutor.review.queue import REVIEW_RATIO, build_today
from deeptutor.review.service import ReviewService


def _round_robin(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subjects = defaultdict(deque)
    for item in items:
        subjects[item["subject"]].append(item)
    order = deque(subjects.values())
    result = []
    while order:
        subject = order.popleft()
        result.append(subject.popleft())
        if subject:
            order.append(subject)
    return result


def _counts(due: int = 0, new: int = 0) -> dict[str, int]:
    return {"due": due, "new": new, "total": due + new}


def plan_mixed_queue(
    service: ReviewService,
    *,
    spec_ids: set[str] | None,
    daily_minutes: int,
    limit: int,
    include_draft: bool,
) -> dict[str, Any]:
    if not service.store.read_only:
        raise ValueError("A mixed queue requires a read-only review store")
    now = time.time()
    load_progress = cache(load_core_progress_read_only)
    library = _counts()
    filtered = _counts()
    seen_specs = set()
    packs = {}
    pool: dict[str, list[dict[str, Any]]] = {"due": [], "new": []}
    omitted = 0
    for spec in service._iter_specs():
        if spec is None:
            omitted += 1
            continue
        seen_specs.add(spec.id)
        quality = service._quality_summary(spec)
        state = quality["quality_state"]
        if state == "rejected" or (state == "draft" and not include_draft):
            continue
        states = states_for_spec(spec, service.store, now=now, progress_loader=load_progress)
        preview = build_today(
            spec, service.store, now=now, session_id="", states=states,
            record_views=False, candidate_preview=True,
        )
        backlog = _counts(preview["due"], preview["new"])
        for key, count in backlog.items():
            library[key] += count
        if spec_ids is not None and spec.id not in spec_ids:
            continue
        for key, count in backlog.items():
            filtered[key] += count
        subject, _ = split_pack_title(spec.title)
        minutes = max(1, math.ceil(preview["planned_minutes"] / max(1, preview["budget"])))
        packs[spec.id] = {
            "spec_id": spec.id, "subject": subject, "title": spec.title,
            "quality_state": state, "quality_warning": quality["quality_warning"],
            "backlog": backlog, "selected": {**_counts(), "estimated_minutes": 0},
        }
        seen_items = set()
        for row in sorted(preview["items"], key=lambda row: row["status"] != "due"):
            if row["item_id"] in seen_items:
                continue
            seen_items.add(row["item_id"])
            pool[row["status"]].append({
                "spec_id": spec.id, "subject": subject, "estimated_minutes": minutes,
                **{key: row[key] for key in ("item_id", "kind", "stem", "options", "difficulty", "status", "due_at")},
            })

    due = deque(_round_robin(pool["due"]))
    new = deque(_round_robin(pool["new"]))
    selected: dict[str, list[dict[str, Any]]] = {"due": [], "new": []}
    remaining = daily_minutes

    def take(source: deque, count: int) -> None:
        nonlocal remaining
        while source and count > 0:
            item = source.popleft()
            minutes = item["estimated_minutes"]
            if minutes <= remaining:
                selected[item["status"]].append(item)
                remaining -= minutes
                count -= 1

    take(due, max(1, round(limit * REVIEW_RATIO)))
    take(new, limit - len(selected["due"]))
    take(due, limit - len(selected["due"]) - len(selected["new"]))
    items = selected["due"] + selected["new"]
    for item in items:
        counts = packs[item["spec_id"]]["selected"]
        counts[item["status"]] += 1
        counts["total"] += 1
        counts["estimated_minutes"] += item["estimated_minutes"]
    return {
        "daily_minutes": daily_minutes,
        "limit": limit,
        "items": items,
        "packs": list(packs.values()),
        "selected": {**_counts(len(selected["due"]), len(selected["new"])), "estimated_minutes": daily_minutes - remaining},
        "filtered_backlog": filtered,
        "library_backlog": library,
        "omitted": omitted,
        "unavailable_spec_ids": sorted((spec_ids or set()) - seen_specs),
    }
