"""Project Review attempts into the core mastery path.

The Review event log stays authoritative. This module is a one-way adapter:
it builds one ``AssessmentRecord`` and calls the core's public
``record_assessment`` entry point. It never writes mastery tables directly and
never lets a projection failure roll back an answer that the learner already
submitted.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Mapping

from deeptutor.learning.assessment import AssessmentRecord, record_assessment

from deeptutor.review.models import Concept, Item, ReviewSpec
from deeptutor.review.storage import ReviewStore

logger = logging.getLogger(__name__)

# Only the diagnostic read is bounded. The core write is a multi-step,
# transaction-backed operation; cancelling it with wait_for can leave the
# attempt recorded but its linked evidence missing.
DIAGNOSTIC_TIMEOUT_SECONDS = 2.0

# ``hard`` means "remembered with effort", not "wrong". The core scheduler
# already reads quality=0.55 as the success branch; result=partial would make
# the Question Bank card record is_correct=False and a declined trend.
_RATING_TO_RESULT = {
    "again": "incorrect",
    "hard": "correct",
    "good": "correct",
}
_RATING_TO_CORRECT = {
    "again": False,
    "hard": True,
    "good": True,
}


def _choice_text(item: Item, index: object) -> str:
    """Render a choice index as ``B. text``; leave non-choice answers alone."""
    if isinstance(index, bool) or not isinstance(index, int):
        return str(index if index is not None else "")
    if 0 <= index < len(item.options):
        option = item.options[index]
        return f"{option.label}. {option.text}"
    return str(index)


def _answer_text(item: Item, answer: object) -> str:
    if item.kind == "single_choice":
        return _choice_text(item, answer)
    if answer is None:
        return ""
    return str(answer)


def _correct_answer_text(item: Item) -> str:
    if item.kind == "single_choice":
        if item.answer.choice_index is None:
            return ""
        return _choice_text(item, item.answer.choice_index)
    if item.kind == "numeric":
        return "" if item.answer.value is None else str(item.answer.value)
    return item.answer.text


def build_assessment_record(
    spec: ReviewSpec,
    item: Item,
    concept: Concept,
    event: Mapping[str, Any],
    *,
    user_answer: object = None,
) -> AssessmentRecord | None:
    """Build the core record, or ``None`` when this concept is review-only."""
    if not spec.mastery_path_id or not concept.knowledge_point_id:
        return None
    rating = str(event.get("rating") or "")
    if rating not in _RATING_TO_RESULT:
        return None
    return AssessmentRecord(
        origin_type="external_import",
        origin_ref=spec.id,
        turn_id=str(event.get("session_id") or ""),
        question_id=item.id,
        source="book" if spec.book_id else "import",
        assessment_type="review",
        result=_RATING_TO_RESULT[rating],
        is_correct=_RATING_TO_CORRECT[rating],
        question=item.stem,
        question_type=item.kind,
        options={option.label: option.text for option in item.options},
        user_answer=_answer_text(item, user_answer),
        correct_answer=_correct_answer_text(item),
        explanation=item.explanation,
        difficulty=item.difficulty,
        material_id=spec.id,
        material_title=spec.title,
        section_id=concept.id,
        section_title=concept.name,
        mastery_path_id=spec.mastery_path_id,
        knowledge_point_id=concept.knowledge_point_id,
        attempt_count=1,
        # The event row is authoritative: a hinted answer must reach core as
        # hinted, or the mastery policy can never label it
        # practiced_with_help.
        hints_used=int(event.get("hints_used") or 0),
        confidence=event.get("confidence"),
        response_time=max(0.0, float(event.get("elapsed_ms") or 0) / 1000.0),
        quality=float(event.get("quality") or 0.0),
        created_at=float(event["ts"]),
        attempt_id=f"review:{event['event_id']}",
    )


def _linkage_preflight(
    path_id: str, knowledge_point_id: str, *, now: float
) -> tuple[bool, str, str, float | None, float | None, bool]:
    """Check the target before any core write and read its pre-answer state.

    Core's ``_is_trusted_linkage`` clears an invalid linkage *and persists the
    empty relationship on the attempt*. A later retry can then never heal it.
    Preflighting keeps an invalid spec from creating that permanent record, so
    creating the path and retrying the same submission can still succeed.

    Returns ``(ok, reason, kp_type, desired_retention, predicted_before,
    had_state)``. ``predicted_before`` is the core's own retrievability at
    answer time; it is the honest value for the response once the core owns the
    due date. ``had_state`` tells the caller whether the core already held
    evidence from another surface, in which case a local-vs-core due comparison
    is meaningless.
    """
    try:
        from deeptutor.learning.policy import find_knowledge_point
        from deeptutor.learning.scheduler import SpacedRepetitionScheduler
        from deeptutor.learning.storage import LearningStore

        from deeptutor.review.core_state import linkage_allowed

        if not linkage_allowed(path_id):
            # Deployment declared an ownership allowlist and this path is not
            # in it. Reject before the core persists an attempt.
            return False, "linkage_not_allowed", "", None, None, False
        progress = LearningStore().load(path_id)
    except Exception as exc:  # noqa: BLE001 - reported as an invalid target
        logger.warning("Mastery linkage preflight failed for %r: %r", path_id, exc)
        return False, "linkage_unavailable", "", None, None, False
    if progress is None or progress.book_id != path_id:
        return False, "path_not_found", "", None, None, False
    kp, _, _ = find_knowledge_point(progress, knowledge_point_id)
    if kp is None:
        return False, "knowledge_point_not_found", "", None, None, False

    retention = float(progress.desired_retention)
    scheduler = SpacedRepetitionScheduler(desired_retention=retention)
    state = progress.repetition_states.get(kp.id)
    if state is None:
        return True, "", str(kp.type.value), retention, 1.0, False
    predicted_before = scheduler.retrievability(
        scheduler.hydrate(state.model_copy(deep=True), kp.type), now=now
    )
    return (
        True,
        "",
        str(kp.type.value),
        retention,
        round(predicted_before, 6),
        True,
    )


def _read_core_due(
    path_id: str, knowledge_point_id: str
) -> tuple[float | None, float | None]:
    """Return ``(next_review_at, desired_retention)`` without raising.

    Reading the path is diagnostics only. The evidence write has already
    happened, so an unavailable path must not turn a successful attempt into
    an error response.
    """
    try:
        from deeptutor.learning.storage import LearningStore

        progress = LearningStore().load(path_id)
        if progress is None:
            return None, None
        state = progress.repetition_states.get(knowledge_point_id)
        return (
            None if state is None else float(state.next_review_at),
            float(progress.desired_retention),
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must not fail the attempt
        logger.warning(
            "Could not read core retention state for path %s: %r", path_id, exc
        )
        return None, None


def _result(
    status: str,
    *,
    attempt_id: str = "",
    diagnostics: list[str] | None = None,
    due_date_consistent: bool | None = None,
    core_next_due_ts: float | None = None,
    core_predicted_retrievability: float | None = None,
    error: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "attempt_id": attempt_id,
        "diagnostics": diagnostics or [],
        "due_date_consistent": due_date_consistent,
        "core_next_due_ts": core_next_due_ts,
    }
    if core_predicted_retrievability is not None:
        payload["core_predicted_retrievability"] = core_predicted_retrievability
    if error:
        payload["error"] = error
    return payload


async def _project_attempt(
    store: ReviewStore,
    spec_id: str,
    attempt_result: Mapping[str, Any],
    *,
    user_answer: object = None,
) -> dict[str, Any]:
    """Project one already-persisted Review attempt into the mastery path."""
    try:
        spec = await asyncio.to_thread(store.get_spec, spec_id)
        item = spec.item_by_id(str(attempt_result.get("item_id") or ""))
        concept = spec.concept_by_id(str(attempt_result.get("concept_id") or ""))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Mastery projection could not resolve %s: %r", spec_id, exc)
        return _result("failed", error="projection_error")

    if not spec.mastery_path_id or not concept.knowledge_point_id:
        return _result("skipped", diagnostics=["no_mastery_linkage"])

    answer_ts = attempt_result.get("event_ts")
    if not isinstance(answer_ts, (int, float)):
        answer_ts = time.time()
    linked, reason, core_type, core_retention, core_predicted, had_state = await asyncio.to_thread(
        _linkage_preflight,
        spec.mastery_path_id,
        concept.knowledge_point_id,
        now=float(answer_ts),
    )
    if not linked:
        # Do not call the core: its first call would persist the cleared
        # linkage and make every retry permanently non-healing.
        return _result("invalid_linkage", diagnostics=[reason])

    submission_id = str(attempt_result.get("submission_id") or "")
    event = await asyncio.to_thread(store.event_by_submission, submission_id)
    if event is None:
        return _result("failed", diagnostics=["event_not_found"])

    record = build_assessment_record(
        spec, item, concept, event, user_answer=user_answer
    )
    if record is None:
        return _result("skipped", diagnostics=["invalid_rating_or_linkage"])

    attempt_id = record.attempt_id
    try:
        # Do not wrap this multi-step core write in wait_for: cancellation can
        # leave the immutable attempt committed while linked evidence is not.
        outcome = await record_assessment(record)
    except Exception as exc:  # noqa: BLE001 - Review answer is already durable
        logger.warning(
            "Mastery projection failed for %s/%r: %r",
            spec_id,
            item.id,
            exc,
            exc_info=True,
        )
        return _result("failed", attempt_id=attempt_id, error="projection_error")

    diagnostics = list(outcome.diagnostics)
    if "dropped_invalid_mastery_linkage" in diagnostics:
        status = "invalid_linkage"
    elif outcome.attempt_recorded or "linked_retention_updated" in diagnostics:
        # A retry may find the assessment attempt already recorded while the
        # linked evidence was not yet applied; core heals it and reports it.
        status = "applied"
    else:
        status = "duplicate"

    core_due: float | None = None
    if status in {"applied", "duplicate"} and not had_state:
        # Only the first-evidence case needs a comparison, and only then is the
        # extra core read worth it. Once other surfaces have evidence the
        # response uses the core due date directly (apply_core_state).
        try:
            core_due, _ = await asyncio.wait_for(
                asyncio.to_thread(
                    _read_core_due,
                    spec.mastery_path_id,
                    concept.knowledge_point_id,
                ),
                timeout=DIAGNOSTIC_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            diagnostics.append("due_read_timeout")
    review_due = attempt_result.get("next_due_ts")
    consistent: bool | None = None
    if had_state:
        # Core already held evidence from another surface. Comparing the local
        # Review replay against it is expected to differ and says nothing about
        # correctness; the response uses the core due date after projection.
        diagnostics.append("core_state_precedes_review")
    elif core_due is not None and isinstance(review_due, (int, float)):
        consistent = abs(core_due - float(review_due)) <= 1.0
        if not consistent:
            diagnostics.append("due_date_mismatch")
    if core_retention is not None:
        if abs(core_retention - spec.plan.target_retention) > 1e-9:
            diagnostics.append("retention_mismatch")
    if core_type and core_type != concept.knowledge_type:
        diagnostics.append("knowledge_type_mismatch")

    return _result(
        status,
        attempt_id=attempt_id,
        diagnostics=diagnostics,
        due_date_consistent=consistent,
        core_next_due_ts=core_due,
        core_predicted_retrievability=core_predicted,
    )


async def project_attempt(
    store: ReviewStore,
    spec_id: str,
    attempt_result: Mapping[str, Any],
    *,
    user_answer: object = None,
) -> dict[str, Any]:
    """Public boundary: projection failure must never fail the Review request."""
    try:
        return await _project_attempt(
            store, spec_id, attempt_result, user_answer=user_answer
        )
    except Exception as exc:  # noqa: BLE001 - contract: never raise to the route
        logger.warning(
            "Mastery projection boundary failed for %s: %r",
            spec_id,
            exc,
            exc_info=True,
        )
        return _result("failed", error="projection_error")


__all__ = ["build_assessment_record", "project_attempt"]
