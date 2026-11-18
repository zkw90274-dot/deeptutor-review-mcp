"""Review Studio facade: spec ingestion, today queue, attempts, export.

The HTTP layer stays thin. Every operation that changes state goes through
``ReviewStore``; every due date goes through ``adapter`` and therefore through
``deeptutor.learning.scheduler``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from deeptutor.__version__ import __version__

from deeptutor.review.adapter import (
    RATING_MAP,
    apply_rating,
    initial_state,
    iso_from_timestamp,
    replay_state,
    scheduler_for,
)
from deeptutor.review.export import build_html
from deeptutor.review.models import (
    Concept,
    Item,
    QualityReport,
    ReviewSpec,
    canonical_json,
    normalize_text,
)
from deeptutor.review.correction import answer_text as _answer_text
from deeptutor.review.correction import build_correction
from deeptutor.review.core_state import states_for_spec
from deeptutor.review.metrics import (
    completion_rate,
    learner_calibration,
    retention_buckets,
)
from deeptutor.review.qa import run_gates
from deeptutor.review.queue import (
    build_session_plan,
    build_today,
    build_weakness,
    concept_state_map,
)
from deeptutor.review.storage import (
    ACTIVE_GAP_SECONDS,
    MAX_OPEN_SESSION_SECONDS,
    ReviewStore,
    SessionNotFound,
    SpecIntegrityError,
    SpecNotFound,
    StateTransition,
    SubmissionConflict,
)
from deeptutor.review.timeutil import tomorrow_window

logger = logging.getLogger(__name__)

#: Row ceiling for the read-only metrics aggregates. Chosen so a wide window on
#: a heavily used spec stays a bounded read (roughly tens of MB) instead of
#: materialising every event ever recorded. Hitting it sets ``truncated``.
MAX_METRIC_EVENTS = 50_000

#: Upper bound for one attempt's elapsed time (24 hours). SQLite stores
#: integers as signed 64-bit; an unbounded model-supplied number would raise
#: ``OverflowError`` inside the write and surface as a 500.
MAX_ELAPSED_MS = 86_400_000

#: Upper bound for the learner's own wording behind a rating. A quote is
#: evidence, not content: it is clipped so a model cannot turn the event row
#: into a transcript.
MAX_LEARNER_QUOTE_CHARS = 500

#: Per-spec locks around "find the open session or create one". The lookup and
#: the insert are separate transactions, so two concurrent ``today`` calls can
#: both miss and create two sessions without this. Process-local, like the
#: spec write lock.
_SESSION_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_SESSION_LOCKS_GUARD = threading.Lock()


def _session_lock(home: Any, spec_id: str) -> threading.Lock:
    key = (str(home), spec_id)
    with _SESSION_LOCKS_GUARD:
        lock = _SESSION_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SESSION_LOCKS[key] = lock
        return lock


class QualityGateFailed(ValueError):
    def __init__(self, report: QualityReport) -> None:
        super().__init__("ReviewSpec failed QA gates")
        self.report = report


@dataclass
class AttemptInput:
    item_id: str
    rating: str
    session_id: str = ""
    submission_id: str = ""
    answer: Any = None
    elapsed_ms: int = 0
    #: How many hints the learner asked for before answering (0-2). Like a
    #: same-session retest, answering after a hint is not unaided recall.
    hints_used: int = 0
    #: Learner's own confidence in their answer, 0-100, when they volunteer
    #: one. Metadata for calibration only; never changes the rating.
    confidence: float | None = None
    #: The learner's own words for the rating ("会，但想了很久"). Recorded so
    #: "the rating came from the learner" is auditable after the fact instead
    #: of being a prompt-only promise. Never used by the scheduler.
    learner_quote: str | None = None


def _resolve_calibration(
    spec: ReviewSpec, payload: dict[str, Any]
) -> dict[str, Any]:
    """Add the readable concept id and name next to the stored hashed key.

    Events persist ``Concept.key`` (a hash); a human reading the metrics
    endpoint needs the spec's own ids. Unknown keys are left blank, never
    guessed.
    """
    rows = payload.get("by_concept") or []
    resolved: list[dict[str, Any]] = []
    for row in rows:
        entry = dict(row)
        try:
            concept = spec.concept_by_key(str(entry.get("concept_key") or ""))
        except KeyError:
            concept = None
        entry["concept_id"] = getattr(concept, "id", "")
        entry["concept_name"] = getattr(concept, "name", "")
        resolved.append(entry)
    payload = dict(payload)
    payload["by_concept"] = resolved
    return payload


def _payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def grade_answer(item: Item, answer: Any) -> bool | None:
    """Grade the learner's submitted answer. ``None`` means "not graded".

    Self-rating drives the scheduler; this field is informational and is never
    allowed to silently change the rating the learner chose.
    """
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        return None
    if item.kind == "single_choice":
        if isinstance(answer, int):
            index = answer
        else:
            text = str(answer).strip()
            index = None
            if text.isdigit():
                index = int(text)
            else:
                for position, option in enumerate(item.options):
                    if text.upper() == option.label.upper():
                        index = position
                        break
        if index is None:
            return False
        return index == item.answer.choice_index
    if item.kind == "numeric":
        try:
            value = float(answer)
        except (TypeError, ValueError):
            return False
        expected = item.answer.value
        if expected is None:
            return False
        tolerance = max(item.answer.tolerance, 1e-9)
        return abs(value - expected) <= tolerance
    expected_text = normalize_text(item.answer.text)
    actual_text = normalize_text(str(answer))
    return actual_text == expected_text or (
        len(actual_text) >= 3 and actual_text in expected_text
    )


def _attempt_response(
    store: ReviewStore,
    spec: ReviewSpec,
    item: Item,
    concept: Concept,
    event: dict[str, Any],
    *,
    created: bool,
    request: AttemptInput,
    now: float,
    scheduler: Any,
    retry: bool = False,
) -> dict[str, Any]:
    # A replay must describe the original event, not the state "now": using
    # the wall clock would report a shorter remaining interval than the
    # ``scheduled_interval_days`` recorded on the event, so the same
    # submission would return different due dates each time it is replayed.
    reference = now if created else float(event["ts"])
    state = replay_state(
        concept,
        store.events_for_concept(spec.id, concept.key),
        now=reference,
        scheduler=scheduler,
    )
    correct = grade_answer(item, request.answer)
    correction = (
        build_correction(spec, item)
        if event["rating"] == "again" or correct is False
        else None
    )
    return {
        "spec_id": spec.id,
        "event_id": event["event_id"],
        "item_id": item.id,
        "concept_id": concept.id,
        "concept_key": concept.key,
        "session_id": event["session_id"],
        "submission_id": event["submission_id"],
        "created": created,
        "rating": event["rating"],
        "requested_rating": request.rating,
        "retry": retry,
        "result": event["result"],
        "quality": event["quality"],
        "hints_used": int(event.get("hints_used") or 0),
        "confidence": event.get("confidence"),
        "learner_quote": event.get("learner_quote"),
        "correct": correct,
        "answer": _answer_text(item),
        "explanation": item.explanation,
        "study_aids": {
            "assumptions": list(item.assumptions),
            "self_check": item.self_check,
            "pitfalls": list(item.pitfalls),
            "table": item.study_table,
        },
        "correction": correction,
        "source_ids": item.source_ids,
        "predicted_retrievability": round(float(event["predicted_retrievability"]), 4),
        "scheduled_interval_days": round(float(event["scheduled_interval_days"]), 6),
        # Raw values recorded on the Review event. ``apply_core_state`` may
        # replace the two fields above with core-authoritative values; these
        # keep the local transition visible for audit and tests.
        "event_predicted_retrievability": round(
            float(event["predicted_retrievability"]), 4
        ),
        "event_scheduled_interval_days": round(
            float(event["scheduled_interval_days"]), 6
        ),
        "event_ts": float(event["ts"]),
        "next_due": iso_from_timestamp(state.next_review_at, spec.plan.timezone),
        "next_due_ts": state.next_review_at,
        "lapse_count": state.lapse_count,
        "review_count": state.review_count,
    }


class ReviewService:
    def __init__(self, store: ReviewStore | None = None) -> None:
        self.store = store or ReviewStore()

    def _session_for(
        self, spec: ReviewSpec, session_id: str | None, now: float
    ) -> str:
        """Join the open sitting (or create one) under the database write lock.

        ``store.open_session`` does the lookup and the insert in one
        transaction, so two workers cannot each mint a sitting. An explicit
        ``session_id`` still goes through ``ensure_session``: the caller asked
        for that specific row, not for whatever is currently open.
        """
        if session_id:
            self.store.ensure_session(
                spec.id, session_id, spec.plan.daily_minutes, now=now
            )
            return session_id
        session, _ = self.store.open_session(
            spec.id, uuid.uuid4().hex, spec.plan.daily_minutes, now=now
        )
        return str(session["session_id"])

    # -- specs --------------------------------------------------------------
    def put_spec(
        self,
        payload: ReviewSpec | dict[str, Any],
        *,
        overwrite: bool = True,
        allow_history_rewrite: bool = False,
    ) -> dict[str, Any]:
        # Round-trip through validation even for a model instance: Pydantic's
        # model_copy(update=...) does not re-run validators, so a copied model
        # can carry a stale content_hash that would poison later reads.
        spec = ReviewSpec.model_validate(
            payload.model_dump(mode="json") if isinstance(payload, ReviewSpec) else payload
        )
        # Validate the timezone only on the write path. Putting it in the model
        # would also run while reading legacy rows, so one historical typo
        # would take the whole listing and export surface down.
        try:
            ZoneInfo(spec.plan.timezone)
        except Exception as exc:
            raise ValueError(f"unknown timezone: {spec.plan.timezone}") from exc
        report = run_gates(spec)
        if not report.passed:
            raise QualityGateFailed(report)
        stored, created = self.store.put_spec(
            spec.model_copy(update={"quality": report}),
            overwrite=overwrite,
            allow_history_rewrite=allow_history_rewrite,
        )
        return {
            "spec_id": stored.id,
            "content_hash": stored.content_hash,
            "created": created,
            "quality": stored.quality.model_dump(mode="json"),
            "counts": {
                "concepts": len(stored.concepts),
                "sections": len(stored.sections),
                "items": len(stored.items),
            },
        }

    def list_specs(self) -> list[dict[str, Any]]:
        return self.store.list_specs()

    def spec_meta(self, spec_id: str) -> dict[str, Any]:
        meta = self.store.spec_meta(spec_id)
        return self._describe_spec(meta, lambda: self.store.get_spec(spec_id))

    def spec_page(self, *, limit: int, offset: int) -> tuple[int, list[dict[str, Any]]]:
        count, rows = self.store.spec_page(limit=limit, offset=offset)
        details = []
        for row in rows:
            payload = row.pop("spec_json")
            details.append(self._describe_spec(row, lambda: self.store.decode_spec(row["spec_id"], payload)))
        return count, details

    def _describe_spec(self, meta: dict[str, Any], load_spec: Callable[[], ReviewSpec]) -> dict[str, Any]:
        spec_id = meta["spec_id"]
        try:
            spec = load_spec()
        except (ValidationError, SpecIntegrityError):
            # One damaged row must not take down the overview or the agent's
            # pack list. Report the pack as failed, with a stable title, and
            # let the caller decide to regenerate/re-import it.
            return {
                **meta,
                "title": spec_id,
                "quality": QualityReport(
                    passed=False,
                    warnings=["stored pack failed validation"],
                ).model_dump(mode="json"),
                "counts": {"concepts": 0, "sections": 0, "items": 0},
                "warning": "stored pack failed validation; regenerate or re-import it",
            }
        return {
            **meta,
            "title": spec.title,
            # Re-run the gates: a pack stored before a new gate existed must
            # not keep advertising an old "passed" report (G9 is the current
            # example). The content hash excludes quality, so this is cheap.
            "quality": run_gates(spec).model_dump(mode="json"),
            "counts": {
                "concepts": len(spec.concepts),
                "sections": len(spec.sections),
                "items": len(spec.items),
            },
        }

    # -- today --------------------------------------------------------------
    def today(
        self,
        spec_id: str,
        *,
        session_id: str | None = None,
        now: float | None = None,
        allow_recovery: bool = True,
        read_only: bool = False,
    ) -> dict[str, Any]:
        spec = self.store.get_spec(spec_id)
        moment = time.time() if now is None else now
        if read_only:
            # A true read: no session is opened and no view is recorded. This
            # is what ``--read-only`` promises; the default path deliberately
            # reuses the open sitting and records views for the Today panel.
            states = states_for_spec(spec, self.store, now=moment)
            return build_today(
                spec,
                self.store,
                now=moment,
                session_id="",
                states=states,
                record_views=False,
                allow_recovery=allow_recovery,
            )
        # Reuse the open session instead of minting one per call. The agent
        # path calls ``today`` every time the learner asks "what now?", and a
        # fresh session per call inflates the session count while the learner
        # is still in the same sitting.
        with _session_lock(self.store.home, spec.id):
            sid = self._session_for(spec, session_id, moment)
        states = states_for_spec(spec, self.store, now=moment)
        return build_today(
            spec,
            self.store,
            now=moment,
            session_id=sid,
            states=states,
            allow_recovery=allow_recovery,
        )

    def due_overview(
        self,
        *,
        subject: str | None = None,
        now: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Cross-pack due summary; never opens a session and never records a view.

        ``review_specs`` answers "what exists" and ``review_today`` answers
        "what is in *this* pack's queue". The highest-frequency question —
        "what should I review today?" — therefore used to cost one
        ``review_today`` call per pack, and in normal mode every one of those
        calls opens or reuses a session and records views (measured on the
        43-pack library: 43 calls, 42 new sessions, 223 views). This method
        runs the same queue read-only and aggregates.

        ``queued`` is exactly what ``review_today`` would present for that pack
        today (due review + new material, capped by the queue budget). ``due``
        and ``new`` are the queue's own split; a pack in recovery mode
        deliberately reports a smaller ``due`` (RS-1005: a returning learner is
        not shown a backlog number), and ``recovery`` says so instead of
        leaving the reader to guess why the numbers disagree.
        """
        moment = time.time() if now is None else now
        needle = (subject or "").strip()
        rows: list[dict[str, Any]] = []
        totals = {"queued": 0, "due": 0, "new": 0, "repeat": 0, "estimated_minutes": 0}
        omitted = 0
        for spec in self._iter_specs():
            if spec is None:
                omitted += 1
                continue
            title = getattr(spec, "title", "") or ""
            if needle and needle not in title:
                continue
            states = states_for_spec(spec, self.store, now=moment)
            activity = self.store.item_activity(spec.id)
            today = build_today(
                spec,
                self.store,
                now=moment,
                session_id="",
                states=states,
                record_views=False,
                activity=activity,
            )
            queued = len(today.get("items") or [])
            if not queued:
                continue
            counts = today.get("counts") or {}
            last = max(activity.values(), default=None)
            # Imported lazily: ``export_tree`` imports this module at load
            # time, so a module-level import here would be circular.
            from deeptutor.review.export_tree import split_pack_title

            pack_subject, _chapter = split_pack_title(title)
            row = {
                "spec_id": spec.id,
                "title": title,
                "subject": pack_subject,
                "queued": queued,
                "due": int(counts.get("due") or 0),
                "new": int(counts.get("new") or 0),
                "repeat": int(counts.get("repeat") or 0),
                "recovery": bool((today.get("recovery") or {}).get("active")),
                "estimated_minutes": int(today.get("estimated_minutes") or 0),
                "last_reviewed_at": (
                    iso_from_timestamp(last, spec.plan.timezone) if last is not None else None
                ),
            }
            rows.append(row)
            for key in ("queued", "due", "new", "repeat", "estimated_minutes"):
                totals[key] += row[key]
        rows.sort(key=lambda row: (-row["due"], -row["queued"], row["spec_id"]))
        return {
            "generated_at": iso_from_timestamp(moment),
            "subject": needle or None,
            "count": len(rows),
            "omitted": omitted,
            "totals": totals,
            "packs": rows[offset:offset + limit],
        }

    def _iter_specs(self) -> Iterator[ReviewSpec | None]:
        with self.store.spec_rows() as rows:
            for row in rows:
                try:
                    yield self.store.decode_spec(row["spec_id"], row["spec_json"])
                except (SpecIntegrityError, ValidationError):
                    yield None

    def weakness(
        self, spec_id: str, *, limit: int = 5, now: float | None = None
    ) -> dict[str, Any]:
        """Read-only weak-concept ranking; never records a view."""
        spec = self.store.get_spec(spec_id)
        moment = time.time() if now is None else now
        states = states_for_spec(spec, self.store, now=moment)
        rows = build_weakness(
            spec,
            self.store,
            now=moment,
            states=states,
            limit=limit,
        )
        return {
            "spec_id": spec.id,
            "generated_at": iso_from_timestamp(moment, spec.plan.timezone),
            "limit": limit,
            "weak_concepts": rows,
        }

    def session_plan(
        self,
        spec_id: str,
        *,
        mode: str = "mixed",
        limit: int | None = None,
        minutes: int | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Read-only session plan; never creates a session or records views."""
        spec = self.store.get_spec(spec_id)
        moment = time.time() if now is None else now
        states = states_for_spec(spec, self.store, now=moment)
        return build_session_plan(
            spec,
            self.store,
            now=moment,
            states=states,
            mode=mode,  # type: ignore[arg-type]
            limit=limit,
            minutes=minutes,
        )

    def metrics(
        self, spec_id: str, *, days: int = 7, now: float | None = None
    ) -> dict[str, Any]:
        """Read-only aggregates for one spec (RS-1102).

        Nothing here writes: both aggregates are pure functions over rows the
        store already holds. ``days`` bounds the window, and ``MAX_METRIC_EVENTS``
        bounds the rows loaded even when the window is wide: a spec that has
        collected a full per-concept history must not turn this endpoint into a
        multi-hundred-megabyte read. When the cap is hit the payload says so
        instead of silently reporting a partial sample as complete.
        """
        spec = self.store.get_spec(spec_id)
        moment = time.time() if now is None else now
        if days <= 0:
            raise ValueError("days must be positive")
        since = moment - days * 86400.0
        events = self.store.events_for_spec(
            spec.id, since=since, limit=MAX_METRIC_EVENTS + 1
        )
        truncated = len(events) > MAX_METRIC_EVENTS
        if truncated:
            # ``events_for_spec`` returns oldest-first; the cap must drop the
            # oldest rows, not the newest ones the window is about.
            events = events[-MAX_METRIC_EVENTS:]
        sessions = self.store.sessions_for_spec(spec.id, since=since)
        # Metrics is read-only, so it must not close rows on disk; treat an
        # open session past the rollover age as completed in this snapshot.
        # Otherwise a chat-only learner who never calls today again stays at
        # ``completed=0`` forever even though the sitting is long over.
        for session in sessions:
            if session.get("ended_at") is None:
                try:
                    started = float(session["started_at"])
                except (KeyError, TypeError, ValueError):
                    continue
                if started < moment - MAX_OPEN_SESSION_SECONDS:
                    session["ended_at"] = started + MAX_OPEN_SESSION_SECONDS
        # ``completion_rate`` deliberately does not join the two sequences, so
        # the caller must hand it one consistent scope. Restrict the events to
        # the sessions that are in the window: otherwise a session that started
        # before the window contributes attempts to a numerator whose session is
        # not in the denominator, and the rate describes two different periods.
        window_sessions = {str(row["session_id"]) for row in sessions}
        scoped = [event for event in events if str(event["session_id"]) in window_sessions]
        # ``today`` materialises a session so it can record item views, but a
        # session with no answers is a browsing artifact. Letting it into the
        # denominator reports a completion rate for a sitting the learner never
        # started; only sessions with at least one answer count as a sitting.
        answered_sessions = {str(event["session_id"]) for event in scoped}
        sessions = [
            session
            for session in sessions
            if str(session["session_id"]) in answered_sessions
        ]
        return {
            "spec_id": spec.id,
            "generated_at": iso_from_timestamp(moment, spec.plan.timezone),
            "window_days": days,
            "retention": retention_buckets(events, now=moment, days=days),
            "completion": completion_rate(sessions, scoped),
            # Calibration reads the whole window, like retention: it describes
            # the learner's self-assessment, not just the sessions that
            # completed, and needs the per-concept grouping the raw events
            # already carry.
            "learner_calibration": _resolve_calibration(
                spec, learner_calibration(events)
            ),
            # ``truncated`` is the honest caveat: the retention buckets and the
            # attempt count describe a prefix of a very large window, not all
            # of it. ``sampled_*`` tells "no activity" apart from "not loaded".
            "sampled_events": len(events),
            "sampled_sessions": len(sessions),
            "truncated": truncated,
        }

    def mappings(self, spec_id: str) -> dict[str, Any]:
        """Read-only concept-to-knowledge-point table for one spec (RS-203).

        Pure projection over the stored spec: no extra storage, no writes, no
        pagination (a spec carries at most tens of concepts). Every item is
        listed under every concept it names, in spec order; an item that names
        several concepts therefore appears more than once across the table,
        which is what "the items of this concept" means at the data level.
        """
        spec = self.store.get_spec(spec_id)
        items_by_concept: dict[str, list[str]] = {
            concept.id: [] for concept in spec.concepts
        }
        # ``ReviewSpec`` already rejects items naming unknown concepts, so the
        # bucket always exists; the lookup keeps the order of ``spec.items``.
        for item in spec.items:
            for concept_id in item.concept_ids:
                items_by_concept[concept_id].append(item.id)
        concepts = [
            {
                "concept_id": concept.id,
                "title": concept.name,
                "knowledge_point_id": concept.knowledge_point_id,
                "item_ids": items_by_concept[concept.id],
            }
            for concept in spec.concepts
        ]
        return {
            "mastery_path_id": spec.mastery_path_id,
            "concepts": concepts,
            "total_concepts": len(concepts),
            "linked_concepts": sum(
                1 for concept in spec.concepts if concept.knowledge_point_id
            ),
        }

    # -- attempts -----------------------------------------------------------
    def attempt(
        self, spec_id: str, request: AttemptInput, *, now: float | None = None
    ) -> dict[str, Any]:
        spec = self.store.get_spec(spec_id)
        moment = time.time() if now is None else now
        try:
            item = spec.item_by_id(request.item_id)
        except KeyError as exc:
            raise KeyError(f"unknown item: {request.item_id}") from exc
        if request.rating not in RATING_MAP:
            raise ValueError(f"unknown rating: {request.rating}")
        raw_hints = request.hints_used
        if isinstance(raw_hints, bool) or not isinstance(raw_hints, int):
            raise ValueError("hints_used must be an integer from 0 to 2")
        hints_used = max(0, min(2, raw_hints))
        confidence = request.confidence
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise ValueError("confidence must be a number between 0 and 100")
            confidence = float(confidence)
            if not math.isfinite(confidence) or not (0.0 <= confidence <= 100.0):
                raise ValueError("confidence must be a finite number between 0 and 100")
        concept = spec.concept_by_id(item.concept_ids[0])
        scheduler = scheduler_for(spec)
        submission_id = request.submission_id or uuid.uuid4().hex
        elapsed_ms = request.elapsed_ms
        if isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, int):
            raise ValueError("elapsed_ms must be an integer")
        if not 0 <= elapsed_ms <= MAX_ELAPSED_MS:
            raise ValueError(
                f"elapsed_ms must be between 0 and {MAX_ELAPSED_MS}"
            )
        learner_quote = request.learner_quote
        if learner_quote is not None:
            if not isinstance(learner_quote, str):
                raise ValueError("learner_quote must be a string")
            learner_quote = learner_quote.strip()[:MAX_LEARNER_QUOTE_CHARS] or None
        # Session is metadata, not identity: a client that retries without a
        # session id must still get the original event back. Hint usage and
        # confidence ARE identity: they change what the event means.
        identity: dict[str, Any] = {
            "spec_id": spec.id,
            "item_id": item.id,
            "rating": request.rating,
            "answer": request.answer,
            # Timing is telemetry, not identity.
        }
        # Hint usage and confidence are identity only when present: events
        # written before schema v3 hashed without these keys, and a replay of
        # such a submission must still match instead of reporting a conflict.
        if hints_used:
            identity["hints_used"] = hints_used
        if confidence is not None:
            identity["confidence"] = confidence
        payload_hash = _payload_hash(identity)

        existing = self.store.event_by_submission(submission_id)
        if existing is not None:
            if existing["payload_hash"] != payload_hash:
                raise SubmissionConflict(
                    "submission_id was already used with a different payload",
                    existing,
                )
            event_concept = concept
            try:
                event_concept = spec.concept_by_key(existing["concept_key"])
            except KeyError:
                pass
            # A replay must describe the event the same way the first response
            # did. ``retry`` is derived from the event history, not from the
            # (possibly absent) flag on this call.
            replay_session = str(existing.get("session_id") or "")
            replay_retry = False
            if replay_session:
                replay_events = self.store.events_for_session(replay_session)
                positions = [
                    index
                    for index, event in enumerate(replay_events)
                    if event["event_id"] == existing["event_id"]
                ]
                if positions and positions[0] > 0:
                    replay_prior = [
                        event
                        for event in replay_events[: positions[0]]
                        if event["item_id"] == item.id
                    ]
                    replay_retry = bool(replay_prior) and (
                        replay_prior[-1]["rating"] == "again"
                    )
            return _attempt_response(
                self.store,
                spec,
                item,
                event_concept,
                existing,
                created=False,
                request=request,
                now=moment,
                scheduler=scheduler,
                retry=replay_retry,
            )

        with _session_lock(self.store.home, spec.id):
            sid = self._session_for(spec, request.session_id, moment)
        session_events = self.store.events_for_session(sid)
        prior_item_events = [
            event for event in session_events if event["item_id"] == item.id
        ]
        # RS-1002: the second answer in the same session after an "again" is
        # the retest. Correcting it maps a self-reported "good" to "hard":
        # the learner recognised the material after seeing the answer, which
        # is not the same as unaided recall.
        is_retry = bool(prior_item_events) and prior_item_events[-1]["rating"] == "again"
        # Answering after a hint is the same story as a same-session retest:
        # the learner got there with help, which is not unaided recall, so a
        # self-reported "good" is stored as "hard".
        helped = is_retry or hints_used > 0
        effective_rating = (
            "hard" if helped and request.rating == "good" else request.rating
        )

        def transition(_state_json: str | None) -> StateTransition:
            # Replay instead of trusting the materialized row: the event log
            # is authoritative even if a previous writer crashed mid-update.
            history = self.store.events_for_concept(spec.id, concept.key)
            state = (
                replay_state(concept, history, now=moment, scheduler=scheduler)
                if history
                else initial_state(concept, now=moment, scheduler=scheduler)
            )
            state, predicted, interval_days = apply_rating(
                state,
                concept,
                effective_rating,
                now=moment,
                item_id=item.id,
                session_id=sid,
                scheduler=scheduler,
            )
            return StateTransition(state, predicted, interval_days)

        event, created = self.store.append_event(
            submission_id=submission_id,
            spec_id=spec.id,
            item_id=item.id,
            concept_key=concept.key,
            session_id=sid,
            ts=moment,
            rating=effective_rating,
            result=RATING_MAP[effective_rating][0],
            quality=RATING_MAP[effective_rating][1],
            elapsed_ms=elapsed_ms,
            payload_hash=payload_hash,
            app_version=__version__,
            transition=transition,
            hints_used=hints_used,
            confidence=confidence,
            learner_quote=learner_quote,
        )
        return _attempt_response(
            self.store,
            spec,
            item,
            concept,
            event,
            created=created,
            request=request,
            now=moment,
            scheduler=scheduler,
            retry=is_retry,
        )

    def apply_core_state(
        self,
        spec_id: str,
        result: dict[str, Any],
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Overlay the authoritative core state after a successful projection.

        The Review event remains durable even if this read fails; in that case
        the response keeps its local replay values and says so via
        ``state_source`` plus a reason. The overlay also recomputes
        ``scheduled_interval_days`` from the core due date, because the local
        transition interval no longer describes the answer once core holds
        evidence from other surfaces.
        """
        moment = time.time() if now is None else now
        try:
            spec = self.store.get_spec(spec_id)
            concept = spec.concept_by_id(str(result.get("concept_id") or ""))
            resolved = states_for_spec(spec, self.store, now=moment)
            concept_state = resolved[concept.id]
            result["state_source"] = concept_state.source
            if concept_state.reason:
                result["state_source_reason"] = concept_state.reason
            if concept_state.source == "core":
                state = concept_state.state
                result["next_due_ts"] = state.next_review_at
                result["next_due"] = iso_from_timestamp(
                    state.next_review_at, spec.plan.timezone
                )
                result["lapse_count"] = state.lapse_count
                result["review_count"] = state.review_count
                event_ts = result.get("event_ts")
                if isinstance(event_ts, (int, float)):
                    result["scheduled_interval_days"] = round(
                        max(0.0, (state.next_review_at - float(event_ts)) / 86400.0),
                        6,
                    )
                core_predicted = (result.get("mastery_projection") or {}).get(
                    "core_predicted_retrievability"
                )
                if isinstance(core_predicted, (int, float)):
                    result["predicted_retrievability"] = round(
                        float(core_predicted), 4
                    )
        except Exception as exc:  # noqa: BLE001 - never turn a display read into a 5xx
            logger.warning("Could not apply core state for %s: %r", spec_id, exc)
            result.setdefault("state_source", "review")
            result.setdefault("state_source_reason", "core_unavailable")
        return result

    # -- session completion -------------------------------------------------
    def complete(
        self, spec_id: str, session_id: str, *, now: float | None = None
    ) -> dict[str, Any]:
        spec = self.store.get_spec(spec_id)
        moment = time.time() if now is None else now
        session = self.store.get_session(session_id)
        if session is None or session["spec_id"] != spec.id:
            raise SessionNotFound(session_id)
        ended = self.store.complete_session(session_id, now=moment)
        events = self.store.events_for_session(session_id)
        # ``completed`` is unique items, not raw events: a retest is the same
        # card answered twice, and counting it twice lets "completed" exceed
        # the session length. Attempt-level counts stay available separately.
        answered_items: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            answered_items.setdefault(str(event["item_id"]), []).append(event)
        completed = len(answered_items)
        correct = sum(
            1
            for item_events in answered_items.values()
            if item_events[-1]["result"] == "correct"
        )
        attempts = len(events)
        retests = attempts - completed
        started_at = float(session["started_at"])
        ended_at = float(ended["ended_at"]) if ended and ended["ended_at"] else moment
        stamps = [started_at, *self.store.session_activity(session_id)]
        active_seconds = sum(
            min(max(0.0, later - earlier), ACTIVE_GAP_SECONDS)
            for earlier, later in zip(stamps, stamps[1:])
        )
        # Wall-clock span includes hours of idling for a session that stayed
        # open; report the active window (gaps capped at 5 minutes) instead.
        actual_minutes = round(active_seconds / 60.0, 1)
        tomorrow_date, start, end = tomorrow_window(moment, spec.plan.timezone)
        states = states_for_spec(spec, self.store, now=moment)
        due_tomorrow = 0
        for concept in spec.concepts:
            due_at = states[concept.id].state.next_review_at
            if start <= due_at < end:
                due_tomorrow += len(spec.items_for_concept(concept.id))
        return {
            "spec_id": spec.id,
            "session_id": session_id,
            "completed": completed,
            "correct": correct,
            "accuracy": round(correct / completed, 4) if completed else 0.0,
            "attempts": attempts,
            "retests": retests,
            "started_at": iso_from_timestamp(started_at, spec.plan.timezone),
            "ended_at": iso_from_timestamp(ended_at, spec.plan.timezone),
            "actual_minutes": actual_minutes,
            "tomorrow": {"date": tomorrow_date, "due_items": due_tomorrow},
        }

    # -- export -------------------------------------------------------------
    def export(
        self,
        spec_id: str,
        *,
        include_snapshot: bool = False,
        now: float | None = None,
    ) -> str:
        spec = self.store.get_spec(spec_id)
        report = run_gates(spec)
        if not report.passed:
            raise QualityGateFailed(report)
        moment = time.time() if now is None else now
        states = states_for_spec(spec, self.store, now=moment)
        return build_html(
            spec,
            concept_state_map(spec, self.store, now=moment, states=states),
            now=moment,
            include_snapshot=include_snapshot,
            quality=report,
        )
