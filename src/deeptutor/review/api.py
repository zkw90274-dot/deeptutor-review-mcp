"""Review Studio HTTP router.

Mounted by the core module loader from ``modules/manifests/review.json`` with
``auth="authenticated"``. Core 1.7.2 has no learning-surface mapping for
``/api/review`` (its ``_learning_surface_for_path`` would return an empty
surface and reject every learner), so the route requires an authenticated
user of any role. Revisit this when core exposes a ``review`` learning
surface. The router is intentionally free of scheduling logic; it validates
transport concerns and delegates to ``ReviewService``.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import math
import sqlite3
from importlib.resources import files
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from deeptutor.review.projection import project_attempt
from deeptutor.review.service import (
    MAX_ELAPSED_MS,
    MAX_LEARNER_QUOTE_CHARS,
    AttemptInput,
    QualityGateFailed,
    ReviewService,
)
from deeptutor.review.storage import (
    ReviewStore,
    ReviewStoreError,
    ReviewStoreIOError,
    SessionConflict,
    SessionNotFound,
    SpecExists,
    SpecHistoryConflict,
    SpecIntegrityError,
    SpecNotFound,
    SubmissionConflict,
)

MAX_REQUEST_BYTES = 2_000_000
logger = logging.getLogger(__name__)


class _NonFiniteNumber(ValueError):
    """JSON contained NaN/Infinity or a number that overflows to infinity."""


def _parse_finite_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise _NonFiniteNumber(raw)
    return value


def _reject_non_finite_constant(raw: str) -> float:
    # json.loads accepts the non-standard NaN/Infinity/-Infinity literals.
    raise _NonFiniteNumber(raw)


def _reject_non_finite_json(body: bytes) -> None:
    """Parse once at the transport edge so no NaN reaches pydantic or storage."""
    json.loads(
        body,
        parse_float=_parse_finite_float,
        parse_constant=_reject_non_finite_constant,
    )


def _json_safe(value: Any) -> Any:
    """Make error details serializable: NaN/Infinity cannot be JSON encoded."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


class _LimitedRoute(APIRoute):
    """Reject oversized bodies before FastAPI materializes them in memory.

    Content-Length is only a fast path: chunked or HTTP/2 requests may omit it
    and would otherwise bypass the limit. The route pre-reads the ASGI body
    with a byte counter, then replays the cached body to FastAPI. Raising from
    inside the receive callable is not reliable because FastAPI converts body
    read failures into a 400 before the route's exception mapping runs.
    """

    def get_route_handler(self):
        original = super().get_route_handler()

        async def limited(request: Request):
            origin = request.headers.get("origin")
            if origin:
                from deeptutor.api.main import _build_cors_settings

                allowed_origins = _build_cors_settings()["allow_origins"]
                from deeptutor.services.config.runtime_settings import load_system_settings

                frontend_port = str(load_system_settings()["frontend_port"])
                allowed_origins = [*allowed_origins, f"http://[::1]:{frontend_port}", "http://[::1]:3000"]
                if origin not in allowed_origins:
                    raise HTTPException(
                        status_code=403,
                        detail={"code": "untrusted_origin", "message": "browser origin is not allowed for review; configure CORS_ORIGINS with the trusted frontend origin"},
                    )
            raw = request.headers.get("content-length")
            if raw is not None:
                try:
                    size = int(raw)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={"code": "invalid_content_length"},
                    ) from exc
                if size > MAX_REQUEST_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail={
                            "code": "payload_too_large",
                            "message": f"request body exceeds {MAX_REQUEST_BYTES} bytes",
                        },
                    )

            receive = request._receive
            parts: list[bytes] = []
            received = 0
            while True:
                message = await receive()
                if message.get("type") != "http.request":
                    break
                chunk = message.get("body", b"")
                received += len(chunk)
                if received > MAX_REQUEST_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail={
                            "code": "payload_too_large",
                            "message": f"request body exceeds {MAX_REQUEST_BYTES} bytes",
                        },
                    )
                parts.append(chunk)
                if not message.get("more_body", False):
                    break

            body = b"".join(parts)
            if body:
                # FastAPI parses a header-less JSON body too, so the check
                # cannot depend on Content-Type. A body that is not JSON keeps
                # FastAPI's normal validation path.
                try:
                    _reject_non_finite_json(body)
                except _NonFiniteNumber:
                    return JSONResponse(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        content={
                            "detail": {
                                "code": "invalid_number",
                                "message": "JSON numbers must be finite",
                            }
                        },
                    )
                except RecursionError:
                    # A pathologically nested document makes Python's parser
                    # hit its recursion limit; FastAPI would surface that as a
                    # 500 because it only catches JSONDecodeError.
                    return JSONResponse(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        content={
                            "detail": {
                                "code": "invalid_json",
                                "message": "JSON is too deeply nested",
                            }
                        },
                    )
                except (ValueError, UnicodeDecodeError):
                    pass
            delivered = False

            async def replay_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return {"type": "http.disconnect"}

            request._receive = replay_receive
            return await original(request)

        return limited


router = APIRouter(route_class=_LimitedRoute)


class AttemptBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    rating: Literal["again", "hard", "good"]
    session_id: str = ""
    submission_id: str = ""
    answer: Any = None
    elapsed_ms: int = Field(default=0, ge=0, le=MAX_ELAPSED_MS)
    #: The learner's own words behind the rating. Stored with the event so the
    #: "the learner rated it, not the model" rule is auditable; never used by
    #: the scheduler and never part of the submission identity.
    learner_quote: str | None = Field(default=None, max_length=MAX_LEARNER_QUOTE_CHARS)

    @field_validator("elapsed_ms", mode="before")
    @classmethod
    def _elapsed_is_integer(cls, value: object) -> object:
        # ``True`` and ``"100"`` both satisfy pydantic's int coercion; only a
        # real JSON integer is a duration, matching the hint/confidence rules.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("elapsed_ms must be an integer")
        return value


class CompleteBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)


class BookAttemptBody(BaseModel):
    """Book reader bridge: correctness is a bool, not a three-level rating."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    is_correct: bool
    session_id: str = ""
    submission_id: str = ""
    answer: Any = None
    elapsed_ms: int = Field(default=0, ge=0, le=MAX_ELAPSED_MS)

    @field_validator("elapsed_ms", mode="before")
    @classmethod
    def _elapsed_is_integer(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("elapsed_ms must be an integer")
        return value


class BookImportBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    book_id: str = Field(min_length=1)
    spec_id: str | None = None
    mastery_path_id: str = ""


def _service() -> ReviewService:
    return ReviewService()


def _not_found(code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"code": code, "message": message},
    )


def _handle(exc: Exception) -> HTTPException:
    if isinstance(exc, sqlite3.Error):
        logger.warning("review database unavailable: %s", exc)
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "storage_unavailable", "message": "review storage is temporarily unavailable; retry later"},
        )
    if isinstance(exc, OverflowError):
        # A value too large for SQLite's signed 64-bit integer is a client
        # input problem. Report it without the exception text: the raw message
        # names internal types and storage details.
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "value_out_of_range",
                "message": "a numeric value is out of the supported range",
            },
        )
    if isinstance(exc, ValidationError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            # ``errors()`` echoes the offending input; a non-finite float there
            # makes the error response itself unserializable (500). Sanitize as
            # a last line of defense behind the transport check.
            detail=_json_safe(jsonable_encoder(exc.errors(include_url=False))),
        )
    if isinstance(exc, (SpecNotFound, SessionNotFound)):
        return _not_found("not_found", str(exc))
    if isinstance(exc, SubmissionConflict):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "submission_conflict",
                "message": str(exc),
                "existing_event_id": (exc.existing or {}).get("event_id"),
            },
        )
    if isinstance(exc, SessionConflict):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "session_conflict", "message": str(exc)},
        )
    if isinstance(exc, SpecHistoryConflict):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "spec_history_conflict",
                "message": str(exc),
                "counts": exc.counts,
            },
        )
    if isinstance(exc, SpecExists):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "spec_exists", "message": str(exc)},
        )
    if isinstance(exc, QualityGateFailed):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "quality_gate_failed",
                "message": "ReviewSpec failed deterministic QA gates",
                "quality": exc.report.model_dump(mode="json"),
            },
        )
    if isinstance(exc, KeyError):
        return _not_found("item_not_found", str(exc))
    if isinstance(exc, ReviewStoreIOError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "storage_unavailable",
                "message": "review storage could not write the spec snapshot",
            },
        )
    if isinstance(exc, ReviewStoreError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "review_store_conflict", "message": str(exc)},
        )
    if isinstance(exc, ValueError):
        # Log the concrete reason, but do not echo arbitrary exception text to
        # the caller: the json encoding layer uses ValueError for internal
        # complaints (e.g. "Out of range float values ...") that leak types.
        logger.warning("review request rejected: %s", exc)
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_request", "message": "request is invalid"},
        )
    raise exc


def _qa_summary(quality: dict[str, Any]) -> str:
    warnings = len(quality.get("warnings") or [])
    if quality.get("passed"):
        return f"通过，{warnings} 条警告" if warnings else "通过，无警告"
    return f"未通过，{warnings} 条警告"


def _overview_card(service: ReviewService, meta: dict[str, Any]) -> str:
    """Render one read-only spec card; every learner-visible string is escaped."""
    spec_id = str(meta.get("spec_id") or "")
    detail = service.spec_meta(spec_id)
    plan = service.session_plan(spec_id, mode="mixed")
    title = html.escape(str(detail.get("title") or spec_id), quote=True)
    safe_id = html.escape(spec_id, quote=True)
    counts = detail.get("counts") or {}
    quality = detail.get("quality") or {}
    plan_counts = plan.get("counts") or {}
    due = int(plan_counts.get("due") or 0)
    # ``new`` is what a fresh spec has instead of due cards. Showing only "due"
    # would render an empty-looking card next to a queue of three new items.
    new = int(plan_counts.get("new") or 0)
    # ``selected`` is a top-level field of the session plan, not part of counts.
    selected = int(plan.get("selected") or 0)
    qa_class = "ok" if quality.get("passed") else "bad"
    return (
        f'<article class="card">'
        f"<h2>{title}</h2>"
        f'<p class="meta">ID <b>{safe_id}</b>'
        f' · 条目 <b>{int(counts.get("items") or 0)}</b>'
        f' · 概念 <b>{int(counts.get("concepts") or 0)}</b></p>'
        f'<ul class="facts">'
        f'<li>QA：<b class="{qa_class}">{html.escape(_qa_summary(quality))}</b></li>'
        f"<li>今日待复习：<b>{due}</b> 项</li>"
        f"<li>今日新内容：<b>{new}</b> 项</li>"
        f"<li>今日队列：<b>{selected}</b> 项 · 预算 {int(plan.get('budget') or 0)}</li>"
        f"</ul>"
        f'<p class="cta"><a href="/api/review/{safe_id}/export">导出离线复习页</a></p>'
        f"</article>"
    )


def _overview_error_card(meta: dict[str, Any]) -> str:
    """A placeholder for one unreadable pack; it must not hide the others."""
    spec_id = str(meta.get("spec_id") or "")
    safe_id = html.escape(spec_id, quote=True)
    return (
        f'<article class="card">'
        f"<h2>{safe_id}</h2>"
        f'<p class="meta">ID <b>{safe_id}</b></p>'
        f'<p class="meta"><b class="bad">该复习包已损坏，无法读取；'
        f"请重新生成或重新导入。</b></p>"
        f"</article>"
    )


def _overview_storage_error_card(meta: dict[str, Any]) -> str:
    """A transient store failure must not read as data damage: telling the
    learner to regenerate a healthy pack throws their work away."""
    spec_id = str(meta.get("spec_id") or "")
    safe_id = html.escape(spec_id, quote=True)
    return (
        f'<article class="card">'
        f"<h2>{safe_id}</h2>"
        f'<p class="meta">ID <b>{safe_id}</b></p>'
        f'<p class="meta"><b class="warn">复习存储暂时不可用，稍后刷新本页即可；'
        f"这份复习包本身没有损坏，无需重新生成。</b></p>"
        f"</article>"
    )


def _overview_page(service: ReviewService) -> str:
    metas = service.list_specs()
    cards = []
    for meta in metas:
        try:
            cards.append(_overview_card(service, meta))
        except Exception as exc:  # noqa: BLE001 - one bad pack must not blank the page
            # A damaged row is expected data, not an application crash: log
            # the reason without a full traceback so the catalog does not
            # manufacture stack traces every time the page is opened.
            logger.warning(
                "review overview card failed for %r: %s",
                meta.get("spec_id"),
                exc,
            )
            if isinstance(exc, (ValidationError, SpecIntegrityError)):
                cards.append(_overview_error_card(meta))
            else:
                # Everything else the read path raises (a locked/read-only DB,
                # a failed migration) is an availability problem, not damage.
                cards.append(_overview_storage_error_card(meta))
    cards = "".join(cards)
    if not metas:
        cards = (
            '<p class="empty">还没有任何复习包。在聊天里让 AI 生成复习内容，'
            "或通过 <code>PUT /api/review/specs/{spec_id}</code> 导入一个 "
            "ReviewSpec 后，这里会列出它。</p>"
        )
    summary = f"共 {len(metas)} 个复习包"
    template = (
        files("deeptutor.review")
        .joinpath("templates/overview.html")
        .read_text(encoding="utf-8")
    )
    if (
        template.count("{{TITLE}}") != 1
        or template.count("{{SUMMARY}}") != 1
        or template.count("{{CARDS}}") != 1
    ):
        # Shipping drift is a server bug, not a bad request: _handle re-raises
        # anything it does not know, so this becomes a 500 rather than a 400.
        raise RuntimeError("template token drift in overview page")
    return (
        template.replace("{{TITLE}}", "复习工坊")
        .replace("{{SUMMARY}}", html.escape(summary))
        .replace("{{CARDS}}", cards)
    )


@router.get("/ui", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    """Read-only overview page: spec cards, today's due counts, export links."""
    try:
        page = _overview_page(_service())
    except Exception as exc:  # noqa: BLE001 - mapped to typed HTTP errors
        raise _handle(exc) from exc
    return HTMLResponse(content=page, headers={"Cache-Control": "no-store"})


@router.put("/specs/{spec_id}")
def put_spec(
    spec_id: str,
    payload: dict[str, Any],
    response: Response,
    allow_history_rewrite: bool = Query(default=False),
) -> dict[str, Any]:
    if payload.get("id") not in (None, spec_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "spec_id_mismatch",
                "message": "body.id must match the path spec_id",
            },
        )
    payload = {**payload, "id": spec_id}
    try:
        result = _service().put_spec(
            payload, allow_history_rewrite=allow_history_rewrite
        )
    except Exception as exc:  # noqa: BLE001 - mapped to typed HTTP errors
        raise _handle(exc) from exc
    response.status_code = status.HTTP_201_CREATED if result["created"] else status.HTTP_200_OK
    return result


@router.get("/specs")
def list_specs() -> dict[str, Any]:
    try:
        return {"specs": _service().list_specs()}
    except Exception as exc:  # noqa: BLE001 - mapped to typed HTTP errors
        raise _handle(exc) from exc


@router.get("/specs/{spec_id}")
def get_spec(spec_id: str) -> dict[str, Any]:
    try:
        return _service().spec_meta(spec_id)
    except Exception as exc:  # noqa: BLE001
        raise _handle(exc) from exc


@router.get("/{spec_id}/today", deprecated=True)
def today(
    spec_id: str,
    session_id: str | None = Query(default=None),
    read_only: bool = Query(default=False, description="Preview without opening a session or recording views."),
    include_draft: bool = Query(default=False),
) -> dict[str, Any]:
    if not read_only:
        return start_today(spec_id, session_id, include_draft)
    try:
        return ReviewService(ReviewStore(read_only=True)).today(spec_id, read_only=True, include_draft=include_draft)
    except Exception as exc:  # noqa: BLE001
        raise _handle(exc) from exc


@router.post("/{spec_id}/today")
def start_today(
    spec_id: str,
    session_id: str | None = Query(default=None),
    include_draft: bool = Query(default=False),
) -> dict[str, Any]:
    try:
        return _service().today(spec_id, session_id=session_id, include_draft=include_draft)
    except Exception as exc:  # noqa: BLE001
        raise _handle(exc) from exc


@router.get("/{spec_id}/weakness")
def weakness(
    spec_id: str,
    limit: int = Query(default=5, ge=1, le=20),
) -> dict[str, Any]:
    try:
        return _service().weakness(spec_id, limit=limit)
    except Exception as exc:  # noqa: BLE001
        raise _handle(exc) from exc


@router.get("/{spec_id}/session")
def session_plan(
    spec_id: str,
    mode: Literal["mixed", "review", "new", "weak"] = Query(default="mixed"),
    limit: int | None = Query(default=None, ge=1, le=12),
    minutes: int | None = Query(default=None, ge=1, le=60),
) -> dict[str, Any]:
    try:
        return ReviewService(ReviewStore(read_only=True)).session_plan(
            spec_id, mode=mode, limit=limit, minutes=minutes
        )
    except Exception as exc:  # noqa: BLE001
        raise _handle(exc) from exc


async def _run_attempt(
    spec_id: str,
    attempt: AttemptInput,
    *,
    answer: Any,
) -> dict[str, Any]:
    try:
        # Construction probes the SQLite schema; keep it off the event loop.
        service = await asyncio.to_thread(_service)
        result = await asyncio.to_thread(service.attempt, spec_id, attempt)
    except Exception as exc:  # noqa: BLE001
        raise _handle(exc) from exc
    # Projection is deliberately after the Review event is durable. A core
    # failure is reported in-band and never turns a saved answer into a 5xx.
    projection = await project_attempt(
        service.store, spec_id, result, user_answer=answer
    )
    result["mastery_projection"] = projection
    if projection.get("status") in {"applied", "duplicate"}:
        result = await asyncio.to_thread(service.apply_core_state, spec_id, result)
    else:
        result.setdefault("state_source", "review")
        # A pack without a mastery linkage is a designed state, not a failure.
        # ``project_attempt`` reports the specific cause in ``diagnostics``;
        # reserve ``projection_failed`` for an actual projection error.
        diagnostics = [str(item) for item in projection.get("diagnostics") or []]
        if "no_mastery_linkage" in diagnostics:
            reason = "no_linkage"
        elif diagnostics:
            reason = diagnostics[0]
        elif projection.get("error"):
            reason = str(projection["error"])
        else:
            reason = "projection_failed"
        result.setdefault("state_source_reason", reason)
    result.pop("grader_basis", None)
    return result


@router.get("/{spec_id}/metrics")
def metrics(
    spec_id: str,
    days: int = Query(default=7, ge=1, le=365),
    include_objective: bool = Query(default=False),
) -> dict[str, Any]:
    try:
        result = _service().metrics(spec_id, days=days)
        if not include_objective:
            result.pop("objective_accuracy", None)
        return result
    except Exception as exc:  # noqa: BLE001
        raise _handle(exc) from exc


@router.get("/{spec_id}/mappings")
def mappings(spec_id: str) -> dict[str, Any]:
    try:
        return {"spec_id": spec_id, **_service().mappings(spec_id)}
    except Exception as exc:  # noqa: BLE001
        raise _handle(exc) from exc


@router.post("/{spec_id}/attempt")
async def attempt(spec_id: str, body: AttemptBody) -> dict[str, Any]:
    return await _run_attempt(
        spec_id,
        AttemptInput(
            item_id=body.item_id,
            rating=body.rating,
            session_id=body.session_id,
            submission_id=body.submission_id,
            answer=body.answer,
            elapsed_ms=body.elapsed_ms,
            learner_quote=body.learner_quote,
        ),
        answer=body.answer,
    )


@router.post("/{spec_id}/book-attempt")
async def book_attempt(spec_id: str, body: BookAttemptBody) -> dict[str, Any]:
    """Bridge a Book quiz/flashcard attempt into the Review event log.

    Book questions carry a boolean correctness, not the three-level rating.
    Mapping is intentionally conservative: correct -> good, wrong -> again.
    """
    return await _run_attempt(
        spec_id,
        AttemptInput(
            item_id=body.item_id,
            rating="good" if body.is_correct else "again",
            session_id=body.session_id,
            submission_id=body.submission_id,
            answer=body.answer,
            elapsed_ms=body.elapsed_ms,
        ),
        answer=body.answer,
    )


@router.post("/imports/book")
def import_book(
    body: BookImportBody,
    response: Response,
    allow_history_rewrite: bool = Query(default=False),
) -> dict[str, Any]:
    """Convert an existing compiled Book into a ReviewSpec and store it."""
    try:
        from deeptutor.review.book_import import build_spec_from_pages, load_book_pages

        book, pages = load_book_pages(body.book_id)
        warnings: list[str] = []
        spec = build_spec_from_pages(
            book,
            pages,
            spec_id=body.spec_id,
            mastery_path_id=body.mastery_path_id,
            warnings=warnings,
        )
        result = _service().put_spec(
            spec, allow_history_rewrite=allow_history_rewrite
        )
        response.status_code = (
            status.HTTP_201_CREATED if result["created"] else status.HTTP_200_OK
        )
        return {
            **result,
            "spec_id": spec.id,
            "book_id": body.book_id,
            "items": len(spec.items),
            "concepts": len(spec.concepts),
            "warnings": warnings,
        }
    except Exception as exc:  # noqa: BLE001
        raise _handle(exc) from exc


@router.post("/{spec_id}/session/complete")
def complete_session(spec_id: str, body: CompleteBody) -> dict[str, Any]:
    try:
        return _service().complete(spec_id, body.session_id)
    except Exception as exc:  # noqa: BLE001
        raise _handle(exc) from exc


# GET is what the overview page's links and the agent's ``review_export_link``
# hand out — a browser can only follow a GET. POST stays for API callers that
# already use it; both run the same implementation.
@router.get("/{spec_id}/export")
@router.post("/{spec_id}/export")
def export(
    spec_id: str,
    include_snapshot: bool = Query(default=False),
) -> HTMLResponse:
    try:
        html = _service().export(spec_id, include_snapshot=include_snapshot)
    except Exception as exc:  # noqa: BLE001
        raise _handle(exc) from exc
    filename = f"review-{spec_id}.html"
    return HTMLResponse(
        content=html,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )
