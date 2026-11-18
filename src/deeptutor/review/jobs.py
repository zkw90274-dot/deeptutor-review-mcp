"""Asynchronous generation jobs for ``review_generate`` (Phase 32).

Generating a chapter pack takes minutes, well past the MCP tool timeout, so the
tool starts a job and polls it by ``job_id`` instead of blocking. Job state
lives in ``<review_home>/jobs/<job_id>.json``: every process that resolves the
same review home — the MCP stdio child, a web server, a script — reads the same
record without sharing memory. Writes are atomic (a temporary file in the same
directory, then :func:`os.replace`), so a poller never observes a half-written
job.

Generation runs in a daemon thread owned by the process that accepted the call.
If that process dies mid-run the record stays ``running``; :func:`is_stale`
lets pollers stop waiting on an abandoned job instead of hanging forever. The
record is deliberately not rewritten to ``failed`` in that case, because the
real cause is unknown and the store must not invent one.
"""

from __future__ import annotations

import contextvars
import json
import logging
import math
import os
import re
import secrets
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from deeptutor.review.storage import SpecExists, SpecNotFound, review_home

logger = logging.getLogger(__name__)

#: A job whose heartbeat is older than this is reported stale. The runner
#: cannot heartbeat while the single generation call blocks, so the window
#: must cover the whole call; one hour fits the slowest legitimate batch
#: (20 written questions on a slow provider) with margin.
JOB_STALE_SECONDS = 3600

#: Only ids this module minted may become path components. ``fullmatch`` is
#: used below, so a trailing newline cannot slip past the anchors. Four to
#: eight hex characters: older ids carry four, new ids eight (collision odds
#: of the four-char form at one id per second were too close for comfort).
_JOB_ID = re.compile(r"^gen-[0-9]+-[0-9a-f]{4,8}$")

#: Failure messages are read by the model; keep them one paragraph long but
#: leave room for the first gate details (the field path is what makes a QA
#: rejection actionable).
_MAX_ERROR_CHARS = 600

#: Windows closes the replace/read race with a bounded retry: a concurrent
#: reader keeps the target open without delete sharing, so ``os.replace``
#: fails with WinError 5 for the few milliseconds the read lasts (and the
#: reader can see a transient ``OSError`` while a replace is in flight).
_WRITE_ATTEMPTS = 8
_WRITE_DELAY = 0.025
_READ_ATTEMPTS = 4
_READ_DELAY = 0.01

#: How long a start waits for the cross-process start lock. The lock is held
#: for one directory scan plus one JSON write, so this only matters when a
#: process died while holding it.
_START_LOCK_SECONDS = 5.0


def _retry_os_error(action: Callable[[], Any], *, attempts: int, delay: float) -> Any:
    """Retry a filesystem operation across the replace/read window."""
    last: OSError | None = None
    for _ in range(attempts):
        try:
            return action()
        except FileNotFoundError:
            raise  # a missing file is not a race; fail fast
        except OSError as exc:
            last = exc
            time.sleep(delay)
    raise last  # type: ignore[misc]  # the loop always assigns it


def jobs_dir() -> Path:
    """Return the job directory, creating it on first use."""
    path = review_home() / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _new_job_id() -> str:
    return f"gen-{int(time.time())}-{secrets.token_hex(4)}"


@contextmanager
def _generation_start_lock() -> Iterator[bool]:
    """Serialise "find an active job, else create one" across processes.

    ``mkdir`` is atomic on every platform and needs no extra dependency. The
    critical section is a directory scan and one write, so the wait is short;
    if the lock cannot be taken (a crashed holder left the directory behind),
    the caller still proceeds — a duplicate job is worse than no dedup, but a
    failed start is worse than both.
    """
    lock = jobs_dir() / ".generation-start.lock"
    deadline = time.monotonic() + _START_LOCK_SECONDS
    acquired = False
    while time.monotonic() < deadline:
        try:
            lock.mkdir()
            acquired = True
            break
        except FileExistsError:
            time.sleep(0.01)
    if not acquired:
        try:
            lock.rmdir()
            lock.mkdir()
            acquired = True
        except OSError:
            acquired = False
    try:
        yield acquired
    finally:
        if acquired:
            try:
                lock.rmdir()
            except OSError:
                pass


def _same_request(
    job: dict[str, Any],
    *,
    num_questions: int,
    question_types: list[str] | None,
    difficulty: str,
    replace: bool,
) -> bool:
    types = list(question_types) if question_types else []
    return (
        int(job.get("num_questions", -1)) == int(num_questions)
        and (job.get("question_types") or []) == types
        and job.get("difficulty") == difficulty
        and bool(job.get("replace")) == bool(replace)
    )


def find_active_generation_job(kb: str, chapter: str) -> dict[str, Any] | None:
    """The newest queued/running, still-heartbeating job for this chapter.

    Corrupt or half-written records read as no job: the scan must never stop
    a start because some unrelated file is unreadable.
    """
    newest: dict[str, Any] | None = None
    for path in jobs_dir().glob("gen-*.json"):
        try:
            job = load_job(path.stem)
        except (OSError, ValueError):
            continue
        if job is None or job.get("status") not in ("queued", "running"):
            continue
        if job.get("kb") != kb or job.get("chapter") != chapter:
            continue
        if is_stale(job):
            continue
        if newest is None or float(job.get("created_at", 0)) >= float(
            newest.get("created_at", 0)
        ):
            newest = job
    return newest


def _job_path(job_id: str) -> Path:
    """Resolve ``job_id`` to a file inside the jobs directory.

    Raises ``ValueError`` for anything that is not a minted id, so a hostile
    or corrupted id can never escape the jobs directory.
    """
    if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
        raise ValueError(f"not a job id: {job_id!r}")
    return jobs_dir() / f"{job_id}.json"


def create_job(
    *,
    kb: str,
    chapter: str,
    num_questions: int,
    question_types: list[str] | None,
    difficulty: str,
    replace: bool = False,
) -> dict[str, Any]:
    """Build a ``queued`` job, persist it, and return the record."""
    now = time.time()
    job: dict[str, Any] = {
        "job_id": _new_job_id(),
        "kb": kb,
        "chapter": chapter,
        "num_questions": int(num_questions),
        "question_types": list(question_types) if question_types else None,
        "difficulty": difficulty,
        "replace": bool(replace),
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "spec_id": None,
        "qa_count": None,
        "skipped": None,
        "qa_passed": None,
        "counts": None,
        "warnings": [],
        "error": None,
    }
    save_job(job)
    return job


def load_job(job_id: str) -> dict[str, Any] | None:
    """Read a job record.

    A missing file, unreadable bytes, corrupt JSON, or an id that does not
    belong to the jobs directory all read as ``None`` — pollers treat them the
    same way: no such job.
    """
    try:
        path = _job_path(job_id)
    except ValueError:
        return None
    try:
        text = _retry_os_error(
            lambda: path.read_text(encoding="utf-8"),
            attempts=_READ_ATTEMPTS,
            delay=_READ_DELAY,
        )
    except FileNotFoundError:
        return None
    except OSError:
        # A live store error is not "no such job": the caller must report an
        # unavailable store instead of inviting a duplicate generation.
        raise
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    if not isinstance(payload, dict) or payload.get("job_id") != job_id:
        return None
    return payload


def save_job(job: dict[str, Any]) -> None:
    """Atomically write ``job``, refreshing ``updated_at``.

    The write doubles as the heartbeat: every status transition stamps the
    file, which is what :func:`is_stale` measures. The temporary file is
    created in the jobs directory so :func:`os.replace` stays atomic on both
    Windows and POSIX.
    """
    job["updated_at"] = time.time()
    target = _job_path(str(job.get("job_id", "")))
    temp = target.with_name(f".{target.name}.{os.getpid()}-{secrets.token_hex(4)}.tmp")
    try:
        temp.write_text(
            json.dumps(job, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _retry_os_error(
            lambda: os.replace(temp, target),
            attempts=_WRITE_ATTEMPTS,
            delay=_WRITE_DELAY,
        )
    except OSError:
        temp.unlink(missing_ok=True)
        raise


def is_stale(job: dict[str, Any], *, now: float | None = None) -> bool:
    """Whether a ``queued`` or ``running`` job has stopped heartbeating.

    A queued record can go stale too: a crash between the file write and
    ``Thread.start`` leaves a job nobody will ever run. Only active statuses
    go stale; an active record without a usable ``updated_at`` is treated as
    stale, because it cannot prove it is alive.
    """
    if job.get("status") not in ("queued", "running"):
        return False
    reference = time.time() if now is None else float(now)
    try:
        updated = float(job["updated_at"])
    except (KeyError, TypeError, ValueError):
        return True
    if not math.isfinite(updated):
        return True  # NaN/infinity cannot prove the job is alive
    return reference - updated > JOB_STALE_SECONDS


def start_generation_job(
    *,
    kb: str,
    chapter: str,
    num_questions: int,
    question_types: list[str] | None,
    difficulty: str,
    replace: bool = False,
    runner: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Create a job and run generation in a daemon thread.

    Returns the ``queued`` record immediately so the tool call stays inside
    its timeout; the caller polls :func:`load_job` with the returned
    ``job_id``. ``replace`` allows the run to overwrite an existing pack with
    the same deterministic id; without it the run fails instead of replacing
    the learner's history. Repeating the same request for the same chapter
    while that run is still queued or running returns the existing record with
    ``reused`` set instead of starting a duplicate. Different options start a
    separate job: the caller asked for different content, so silently
    substituting the first run would be wrong. ``runner`` is the test seam that
    substitutes the LLM pipeline.
    """
    with _generation_start_lock():
        active = find_active_generation_job(kb, chapter)
        if active is not None and _same_request(
            active,
            num_questions=num_questions,
            question_types=question_types,
            difficulty=difficulty,
            replace=replace,
        ):
            reused = dict(active)
            reused["reused"] = True
            return reused
        job = create_job(
            kb=kb,
            chapter=chapter,
            num_questions=num_questions,
            question_types=question_types,
            difficulty=difficulty,
            replace=replace,
        )
    # threading.Thread does not inherit contextvars; the web request's user
    # scope (and with it review_home / the KB root) would silently reset to
    # the default scope inside the runner. Copy the context so the background
    # job writes to the same store that created it.
    context = contextvars.copy_context()
    thread = threading.Thread(
        target=context.run,
        args=(runner or _run_generation, job),
        daemon=True,
        name=f"review-gen-{job['job_id']}",
    )
    thread.start()
    return job


def _complete_concept_notes(prompt: str, system_prompt: str) -> str:
    """Synchronous completion for concept notes; the job runs off-loop.

    Providers occasionally return an empty body (rate limit, a dropped
    response, a transient 5xx mapped to an empty string). The notes layer
    treats that as "no notes", which silently removes study aids and the
    comparison table; one short retry costs a second and keeps the best-effort
    promise without turning a transport hiccup into a permanent gap.
    """

    from deeptutor.services.llm.client import get_llm_client

    client = get_llm_client()
    for attempt in range(2):
        text = client.complete_sync(prompt, system_prompt)
        if str(text or "").strip():
            return text
        if attempt == 0:
            time.sleep(0.5)
    return ""


def _run_generation(job: dict[str, Any]) -> None:
    """Run the real pipeline; never leave the job ``running`` on error.

    Every heavy import is deferred into the function body: ``jobs`` must stay
    importable in a checkout without the LLM stack, because the tool layer
    reads job state regardless. The imports sit in their own ``try`` so an
    ``ImportError`` also lands as a ``failed`` job instead of escaping the
    thread and stranding the record.
    """
    job["status"] = "running"
    try:
        save_job(job)
    except OSError:
        # The store stayed unwritable through the retry window: the record
        # remains ``queued`` and goes stale, which is the honest state.
        # Killing nothing else in the thread is not possible from here.
        return
    try:
        from deeptutor.review.generate import generate_spec_from_kb
        from deeptutor.review.service import QualityGateFailed, ReviewService
    except Exception as exc:  # noqa: BLE001 - a job must never stay running
        _fail(job, f"{type(exc).__name__}: {exc}")
        return

    kwargs = {
        "kb_name": job["kb"],
        "chapter": job["chapter"],
        "num_questions": job["num_questions"],
        "question_types": job["question_types"],
        "difficulty": job["difficulty"],
        "llm_complete": _complete_concept_notes,
    }
    try:
        try:
            result = generate_spec_from_kb(**kwargs)
        except Exception as exc:  # noqa: BLE001 - classified below
            # A transport failure (timeout, dropped connection) usually
            # succeeds on a second attempt and costs nothing the first one
            # produced; anything else is a real failure and propagates.
            if not _is_network_error(exc):
                raise
            logger.warning(
                "generation job %s hit a network error, retrying once: %r",
                job.get("job_id"),
                exc,
            )
            result = generate_spec_from_kb(**kwargs)
        if result.spec is None:
            job["warnings"] = list(getattr(result, "warnings", None) or [])
            job["qa_count"] = result.qa_count
            job["skipped"] = result.skipped
            _fail(
                job,
                "no questions could be generated for chapter "
                f"{job['chapter']!r} in knowledge base {job['kb']!r}",
            )
            return
        service = ReviewService()
        warnings = list(getattr(result, "warnings", None) or [])
        job.update(qa_count=result.qa_count, skipped=result.skipped, warnings=warnings)
        spec = _reuse_legacy_spec_id(
            service,
            result.spec,
            getattr(result, "kb_name", job["kb"]),
            getattr(result, "chapter", job["chapter"]),
            warnings,
        )
        replace = bool(job.get("replace", False))
        store = getattr(service, "store", None)
        spec_id = getattr(spec, "id", None)
        # Regenerating a chapter resolves to the same deterministic id, so a
        # plain retry would silently replace the pack the learner's events
        # and mastery state point at. Refuse unless the caller said so
        # explicitly; the store re-checks under its per-spec lock.
        history: dict[str, int] = {}
        if store is not None and isinstance(spec_id, str):
            try:
                store.spec_meta(spec_id)
            except SpecNotFound:
                pass
            else:
                if not replace:
                    job["spec_id"] = spec_id
                    _fail(job, _existing_pack_message(spec_id))
                    return
                history = store.history_counts(spec_id)
        try:
            stored = service.put_spec(
                spec,
                overwrite=replace,
                allow_history_rewrite=replace,
            )
            if any(history.values()):
                warnings.append(_history_rewrite_warning(str(spec_id), history))
        except SpecExists:
            job["spec_id"] = getattr(spec, "id", None)
            _fail(job, _existing_pack_message(str(getattr(spec, "id", ""))))
            return
        job.update(
            status="done",
            spec_id=stored.get("spec_id"),
            qa_count=result.qa_count,
            skipped=result.skipped,
            qa_passed=bool((stored.get("quality") or {}).get("passed", False)),
            counts=stored.get("counts"),
            warnings=warnings,
            error=None,
        )
    except QualityGateFailed as exc:
        _fail(job, _gate_summary(exc))
        return
    except Exception as exc:  # noqa: BLE001 - a job must never stay running
        _fail(job, f"{type(exc).__name__}: {exc}")
        return
    _finish_save(job)


def _reuse_legacy_spec_id(
    service: Any,
    spec: Any,
    kb_name: str,
    chapter: str,
    warnings: list[str],
) -> Any:
    """Keep a pre-Phase-35 pack's id so its history is not orphaned.

    Phase 35 appended a collision digest to generated ids. Regenerating the
    same chapter would otherwise create a second pack with the same title
    while the learner's events, sessions and mastery state stay under the old
    id. Reuse the legacy id only when the new id does not already exist.
    """
    try:
        from deeptutor.review.generate import legacy_default_spec_id
    except ImportError:
        # Tests replace the generation module with a one-function stub.
        return spec
    from pydantic import ValidationError

    from deeptutor.review.models import ReviewSpec
    from deeptutor.review.storage import SpecIntegrityError, SpecNotFound

    store = getattr(service, "store", None)
    if store is None or not hasattr(spec, "id"):
        # Unit-test doubles need not model the store; production always does.
        return spec
    try:
        store.spec_meta(spec.id)
        return spec  # the new-format pack is already in use
    except SpecNotFound:
        pass
    legacy = legacy_default_spec_id(kb_name, chapter)
    if legacy == spec.id:
        return spec
    try:
        legacy_spec = store.get_spec(legacy)
    except (SpecNotFound, ValidationError, SpecIntegrityError):
        return spec
    # The legacy id is a lossy slug: "Chapter 2" and "Chapter-2" collide.
    # Reusing it for the other chapter would overwrite that pack. The source
    # id records the exact kb/chapter identity.
    identity_source = f"kb:{kb_name}·{chapter}"
    if identity_source not in (legacy_spec.sources or {}):
        warnings.append(
            "legacy spec id belongs to a different chapter; kept the new id"
        )
        return spec
    old_id = spec.id
    payload = spec.model_dump(mode="json")
    payload["id"] = legacy
    payload["content_hash"] = ""
    # Item ids embed the spec id (``<spec>-q001``). Keep them on the legacy
    # prefix too, otherwise the learner's existing events (which reference
    # ``<legacy>-q001``) would point at items that no longer exist.
    old_prefix = f"{old_id}-"

    def move(item_id: Any) -> Any:
        if isinstance(item_id, str) and item_id.startswith(old_prefix):
            return f"{legacy}-{item_id[len(old_prefix):]}"
        return item_id

    for item in payload.get("items", []):
        item["id"] = move(item.get("id", ""))
        if item.get("variant_of"):
            item["variant_of"] = move(item["variant_of"])
    for section in payload.get("sections", []):
        for block in section.get("blocks", []):
            if block.get("item_id"):
                block["item_id"] = move(block["item_id"])
    warnings.append(f"reused legacy spec id {legacy}")
    return ReviewSpec.model_validate(payload)


def _gate_summary(exc: Any) -> str:
    """One-line summary of a QA-gate rejection for the model to act on.

    The gate name alone ("G10-math-balance") does not tell the caller which
    field broke the rule. Append the first details, which carry the field path
    (``items.[2].stem``), so the next generation can fix exactly that item.
    """
    report = getattr(exc, "report", None)
    failed = [
        gate
        for gate in getattr(report, "gates", [])
        if not getattr(gate, "passed", True)
    ]
    parts: list[str] = []
    for gate in failed:
        name = str(getattr(gate, "gate", "") or "gate")
        message = str(getattr(gate, "message", "") or "").strip()
        entry = f"{name}: {message}" if message else name
        details = [
            str(detail).strip()
            for detail in (getattr(gate, "details", None) or [])
            if str(detail).strip()
        ]
        if details:
            shown = "; ".join(details[:3])
            if len(details) > 3:
                shown += f"; (+{len(details) - 3} more)"
            entry += f" [{shown}]"
        parts.append(entry)
    if not parts:
        return f"quality gates failed: {exc}"
    return "quality gates failed: " + "; ".join(parts)


def _history_rewrite_warning(spec_id: str, counts: dict[str, int]) -> str:
    return (
        f"replaced pack {spec_id!r} although it already had "
        f"{counts['events']} review event(s), {counts['states']} state row(s) "
        f"and {counts['sessions']} session(s); history entries that reference "
        "removed item/concept keys no longer resolve"
    )


def _existing_pack_message(spec_id: str) -> str:
    """Failure text that tells the caller how to proceed deliberately."""
    return (
        f"pack {spec_id!r} already exists; start the generation again with "
        "replace=true only after the learner explicitly asks to replace it"
    )


def _fail(job: dict[str, Any], message: str) -> None:
    job["status"] = "failed"
    job["error"] = _clip(message)
    _finish_save(job)


#: Exception class names (openai SDK and friends) that mean "the transport
#: failed", not "the request was wrong". Matched by name so the runner stays
#: importable without the LLM stack.
_NETWORK_ERROR_NAMES = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "APIConnectionTimeoutError",
        "InternalServerError",
        "RateLimitError",
        "ReadTimeout",
        "ConnectTimeout",
    }
)


def _is_network_error(exc: BaseException) -> bool:
    """Whether any link in the exception chain is a transport failure."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in _NETWORK_ERROR_NAMES:
            return True
        current = current.__cause__ or current.__context__
    return False


def _finish_save(job: dict[str, Any]) -> None:
    """Persist a terminal state without raising out of the runner thread.

    The work itself has already happened (a pack stored, or an error known);
    a store that stays unwritable cannot fall back to another state, so log
    loudly instead of losing the thread to an exception.
    """
    try:
        save_job(job)
    except OSError:
        logger.exception(
            "could not persist the terminal state of job %s", job.get("job_id")
        )


def _clip(text: str, limit: int = _MAX_ERROR_CHARS) -> str:
    """Collapse whitespace and truncate to ``limit`` characters."""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
