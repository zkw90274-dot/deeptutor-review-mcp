"""Authoritative retention state for linked concepts.

A linked Review concept does not own its due date any more: the core mastery
path owns it, because the same objective may also receive Book or Mastery
evidence. Review events remain the history of Review answers; this module only
decides which state is displayed.

Unlinked concepts keep the Phase 0.5 behavior: replay ``review_events``.

Fallback rules are deliberately conservative: if the core cannot produce a
state, the concept falls back to Review replay even when it has events. A
synthetic "new" core state must never hide a card the learner has already
answered.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from deeptutor.learning.models import RepetitionState
from deeptutor.learning.policy import find_knowledge_point
from deeptutor.learning.scheduler import SpacedRepetitionScheduler

from deeptutor.review.adapter import replay_state, scheduler_for
from deeptutor.review.models import Concept, ReviewSpec

try:  # DeepTutor >= 1.7.2 binds a per-path review strategy.
    from deeptutor.learning.review_strategy import (
        scheduler_for_progress as _strategy_scheduler,
    )
except ImportError:  # DeepTutor 1.7.0 has a single fixed interval sequence.
    _strategy_scheduler = None

logger = logging.getLogger(__name__)

StateSource = Literal["core", "review"]
# Optional deployment-level allowlist. Unset means "single-user local": the
# spec author is the data owner, which is the Phase 0.5/1 trust assumption.
ALLOWED_PATHS_ENV = "DEEPTUTOR_REVIEW_ALLOWED_MASTERY_PATHS"


def allowed_mastery_paths() -> frozenset[str] | None:
    raw = os.environ.get(ALLOWED_PATHS_ENV, "")
    values = frozenset(part.strip() for part in raw.split(",") if part.strip())
    return values or None


def linkage_allowed(path_id: str) -> bool:
    """Fail closed when a deployment declared an ownership allowlist."""
    allowed = allowed_mastery_paths()
    return allowed is None or path_id in allowed


@dataclass(frozen=True)
class ConceptState:
    state: RepetitionState
    source: StateSource
    # True only for a synthetic core initial state that has not been
    # persisted. The queue treats it as new content, not as evidence that the
    # learner has already seen the objective.
    initial: bool = False
    # Client-visible reason when the source is "review". Empty for core.
    reason: str = ""
    # The core ReviewTask for this objective, when the path exposes one.
    # Kept as Any so this adapter does not import the core's model at module
    # load time; queue.py reads the fields it needs defensively.
    task: Any | None = None


def load_core_progress(path_id: str) -> Any | None:
    """Load a mastery aggregate without making Today depend on core health."""
    if not path_id or not linkage_allowed(path_id):
        return None
    try:
        from deeptutor.learning.storage import LearningStore

        return LearningStore().load(path_id)
    except Exception as exc:  # noqa: BLE001 - Today falls back to Review state
        # repr keeps log lines single-line and avoids exception-class-only
        # messages hiding the path that failed.
        logger.warning("Could not load core mastery path %r: %r", path_id, exc)
        return None


def _kp_index(progress: Any) -> dict[str, Any]:
    """One pass over the path instead of one linear scan per concept."""
    index: dict[str, Any] = {}
    for module in getattr(progress, "modules", []) or []:
        for kp in getattr(module, "knowledge_points", []) or []:
            index[str(kp.id)] = kp
    return index


def _core_scheduler(progress: Any) -> SpacedRepetitionScheduler:
    """Use the path's bound strategy when the core exposes one.

    DeepTutor 1.7.2 added per-path interval sequences. Hydrating a legacy
    ``stability=0`` snapshot with the default sequence produces the wrong
    stability (and therefore the wrong retrievability) when the path uses a
    custom strategy. The 1.7.0 fallback keeps the old behavior.
    """
    if _strategy_scheduler is not None:
        try:
            return _strategy_scheduler(progress)
        except Exception as exc:  # noqa: BLE001 - a bad binding must not break Today
            logger.warning(
                "Could not resolve core review strategy for %r: %r",
                getattr(progress, "book_id", ""),
                exc,
            )
    return SpacedRepetitionScheduler(
        desired_retention=float(getattr(progress, "desired_retention", 0.9))
    )


def _core_tasks(progress: Any, *, now: float) -> dict[str, Any]:
    """Map knowledge_point_id -> core ReviewTask for weakness/priority.

    ``build_review_queue`` hydrates its own copy of the aggregate in place;
    ``load_core_progress`` returns a fresh object, so this never writes back.
    """
    try:
        tasks = _core_scheduler(progress).build_review_queue(progress, now=now)
    except Exception as exc:  # noqa: BLE001 - Today must survive core drift
        logger.warning(
            "Could not build core review queue for %r: %r",
            getattr(progress, "book_id", ""),
            exc,
        )
        return {}
    return {str(task.knowledge_point_id): task for task in tasks}


def _core_state_for_indexed(
    spec: ReviewSpec,
    concept: Concept,
    progress: Any,
    kp_index: dict[str, Any],
    *,
    now: float,
) -> tuple[RepetitionState | None, bool]:
    """Return ``(state, is_synthetic_initial)`` for a linked objective."""
    if not spec.mastery_path_id or not concept.knowledge_point_id:
        return None, False
    kp = kp_index.get(concept.knowledge_point_id)
    if kp is None:
        return None, False

    state = progress.repetition_states.get(kp.id)
    if state is not None:
        # Legacy snapshots store stability=0. Core hydrates before reading;
        # without this, retrievability is computed from an empty stability and
        # displays 0% for a card reviewed yesterday.
        scheduler = _core_scheduler(progress)
        return scheduler.hydrate(state.model_copy(deep=True), kp.type), False

    # A brand-new linked objective has no persisted state yet. Use the core's
    # type and retention so its first displayed due date matches what the core
    # will create when the first evidence arrives; do not persist this.
    retention = float(progress.desired_retention)
    scheduler = _core_scheduler(progress)
    return (
        scheduler.get_initial_state(
            kp.type,
            now=now,
            desired_retention=retention,
        ),
        True,
    )


def core_state_for(
    spec: ReviewSpec,
    concept: Concept,
    progress: Any | None,
    *,
    now: float,
) -> RepetitionState | None:
    """Return the linked core state, or ``None`` when Review must fall back."""
    if progress is None:
        return None
    state, _ = _core_state_for_indexed(
        spec, concept, progress, _kp_index(progress), now=now
    )
    return state


def states_for_spec(
    spec: ReviewSpec,
    store: Any,
    *,
    now: float,
) -> dict[str, ConceptState]:
    """Resolve every concept once: core for linked, local replay otherwise."""
    has_linkage = bool(spec.mastery_path_id) and any(
        concept.knowledge_point_id for concept in spec.concepts
    )
    progress = load_core_progress(spec.mastery_path_id) if has_linkage else None
    kp_index = _kp_index(progress) if progress is not None else {}
    task_by_kp = _core_tasks(progress, now=now) if progress is not None else {}
    scheduler = scheduler_for(spec)
    # One query for the whole spec instead of one connection per concept; the
    # rows stay rowid-ordered, so each concept's replay sees the same history.
    local_events_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in store.events_for_spec(spec.id):
        local_events_by_key[str(event["concept_key"])].append(event)
    result: dict[str, ConceptState] = {}
    for concept in spec.concepts:
        local_events = local_events_by_key.get(concept.key, [])
        task = task_by_kp.get(str(concept.knowledge_point_id)) if concept.knowledge_point_id else None
        core: RepetitionState | None = None
        initial = False
        if progress is not None:
            core, initial = _core_state_for_indexed(
                spec, concept, progress, kp_index, now=now
            )
        if core is not None and not (initial and local_events):
            # A synthetic core state means "the core has never seen this
            # objective". If Review has answers for it, the core write did not
            # land (projection failed); replaying the local history is the only
            # way the card does not silently disappear from Today.
            result[concept.id] = ConceptState(
                state=core,
                source="core",
                initial=initial,
                task=task,
            )
            continue
        local = replay_state(
            concept,
            local_events,
            now=now,
            scheduler=scheduler,
        )
        reason = (
            "core_state_missing_local_history"
            if core is not None
            else ("core_unavailable" if has_linkage else "no_linkage")
        )
        result[concept.id] = ConceptState(
            state=local,
            source="review",
            initial=False,
            reason=reason,
            task=task,
        )
    return result


__all__ = [
    "ALLOWED_PATHS_ENV",
    "ConceptState",
    "allowed_mastery_paths",
    "core_state_for",
    "linkage_allowed",
    "load_core_progress",
    "states_for_spec",
]
