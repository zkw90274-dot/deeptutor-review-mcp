"""Review Studio agent tools: the tools a chat model mounts.

Review Studio previously reached the learner only through HTTP (the offline
export and the overview page). These tools make the same
:class:`~deeptutor.review.service.ReviewService` reachable from an ordinary
chat, in the same shape as the mastery navigation tools
(``deeptutor.tools.mastery_nav``).

Eleven tools, split along what the learner is asking for:

* ``review_specs`` — what can I review, and how big is each pack?
* ``review_due`` — which packs have work today, across the whole library?
* ``review_today`` — what is due right now, and what does each item ask?
* ``review_attempt`` — record the learner's **own** self-assessment.
* ``review_weakness`` — what am I weakest at, read-only.
* ``review_export_link`` — hand over the offline review page.
* ``review_export_tree`` — write the whole pack collection to a folder tree.
* ``review_book_figures`` — which book section owns which diagram.
* ``review_generate`` — turn one knowledge-base chapter into a new pack.
* ``review_hint`` — a two-step hint that never states the answer.
* ``review_plan`` — schedule chapters into daily study and review tasks.

``review_due`` exists because the most common question ("what should I review
today?") otherwise costs one ``review_today`` per pack, and every one of those
opens or reuses a session and records views: a read in name, a write in fact.

Two properties keep them safe to mount permanently:

**The model never grades.** ``review_attempt`` only transcribes a rating the
learner spoke aloud (``again`` / ``hard`` / ``good``). The scheduler that plans
the learner's future reviews is driven by that rating, so a rating the model
invented would corrupt the only record of what the learner actually
remembers. The description states this in the terms the model reads, and the
tool takes no answer argument it could use to compute a rating.

**No answer leaks before the learner answers.** ``review_today`` deliberately
strips the answer and explanation from each item; they only appear once an
attempt has been recorded, so the model cannot read the answer out loud while
asking the question.

Every payload carries an ``instruction`` field telling the model what to do
next and what not to do. A missing spec or item is a *soft* failure: the tool
succeeds, returns an ``error`` plus a redirecting instruction, and the model
can recover by calling ``review_specs`` / ``review_today``. Only an internal
fault (a broken store, an unexpected exception) returns ``success=False``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import unicodedata
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Any

from deeptutor.core.tool_protocol import (
    BaseTool,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from deeptutor.review import capabilities, export_tree, materials, plan_service
from deeptutor.review.adapter import RATING_MAP
from deeptutor.review.projection import project_attempt
from deeptutor.review.qa import run_gates
from deeptutor.review.service import AttemptInput, ReviewService
from deeptutor.review.storage import (
    ReadOnlyStoreUnavailable,
    ReviewStore,
    SessionConflict,
    SpecNotFound,
    SubmissionConflict,
    review_home,
)

logger = logging.getLogger(__name__)

#: Metadata fields (titles, concept names, reasons) are clipped tight: they
#: are labels, not content. Stems and options are clipped far more generously:
#: the model reads them aloud "exactly as given", and real generated packs
#: carry stems up to ~900 characters and explanations past 3000 (measured on
#: the stored kb-* packs). A tight clip would make the model dictate a
#: truncated question while the instruction claims it is verbatim.
TEXT_LIMIT = 200
STEM_LIMIT = 2000
OPTION_LIMIT = 800

#: The learner's own wording behind a rating is stored as evidence, not as
#: content: clip it so one rambling sentence cannot turn the event row into a
#: transcript. Matches ``service.MAX_LEARNER_QUOTE_CHARS``.
LEARNER_QUOTE_LIMIT = 500

#: The weakness tool always answers with the top five, matching the Today
#: panel and ``GET /api/review/{id}/weakness``. The tool surface stays narrow on
#: purpose: the model does not need to tune a ranking it cannot interpret.
WEAKNESS_LIMIT = 5

#: Relative on purpose (see the frozen Phase 29 contract): the chat frontend
#: resolves it against its own origin, and an absolute URL would pin the link to
#: whatever host the agent happened to run on.
EXPORT_URL_TEMPLATE = "/api/review/{spec_id}/export"

#: Optional public origin for external MCP clients. The chat frontend already
#: resolves the relative ``url``; an agent outside this deployment needs an
#: absolute link it can hand to the learner. Configuration is explicit so a
#: local run never advertises a host it cannot serve.
PUBLIC_BASE_URL_ENV = "DEEPTUTOR_REVIEW_PUBLIC_BASE_URL"
PUBLIC_BASE_URL_FALLBACK_ENV = "DEEPTUTOR_PUBLIC_BASE_URL"


def _public_base_url() -> str:
    """The configured public origin, or ``""`` when unset/invalid."""
    raw = (
        os.environ.get(PUBLIC_BASE_URL_ENV)
        or os.environ.get(PUBLIC_BASE_URL_FALLBACK_ENV)
        or ""
    ).strip()
    if not re.match(r"^https?://[^\s/]+(?:/[^\s]*)?$", raw, re.IGNORECASE):
        return ""
    return raw.rstrip("/")

#: The tools every surface serves (chat, MCP server, skill). The tuple is
#: frozen by tests on all three surfaces; add tool names here first.
REVIEW_AGENT_TOOL_NAMES: tuple[str, ...] = (
    "review_specs",
    "review_due",
    "review_today",
    "review_attempt",
    "review_weakness",
    "review_export_link",
    "review_export_tree",
    "review_book_figures",
    "review_generate",
    "review_hint",
    "review_plan",
)

#: Generation bounds. The schema advertises them and ``execute`` clamps to the
#: same numbers, because a model can ignore the schema.
DEFAULT_NUM_QUESTIONS = 12
MIN_NUM_QUESTIONS = 5
MAX_NUM_QUESTIONS = 20
DEFAULT_QUESTION_TYPES: tuple[str, ...] = ("choice", "written")
ALLOWED_QUESTION_TYPES = frozenset(DEFAULT_QUESTION_TYPES)


# -- shared result helpers --------------------------------------------------


def _result(payload: dict[str, Any], *, tool: str, success: bool = True) -> ToolResult:
    """JSON content plus the ``review_xxx`` metadata mirror.

    The mirror key is the tool name, matching the acceptance script and the
    convention the mastery navigation tools use: a frontend reader does not
    have to know which tool produced the payload.
    """
    return ToolResult(
        content=json.dumps(payload, ensure_ascii=False),
        success=success,
        metadata={tool: payload},
    )


def _failure(message: str) -> ToolResult:
    """Internal fault: the model cannot fix this by calling another tool."""
    return ToolResult(content=message, success=False)


def _internal_failure(tool: str, exc: Exception) -> ToolResult:
    if isinstance(exc, ReadOnlyStoreUnavailable):
        return _soft(
            tool=tool,
            error=str(exc),
            instruction="Ask the operator to initialize or migrate the Review Studio database in write mode.",
        )
    logger.warning("%s failed: %r", tool, exc, exc_info=True)
    return _failure(
        f"{tool} failed ({type(exc).__name__}). The review store may be "
        "unavailable or the call hit an internal error; tell the learner so "
        "instead of retrying in a loop."
    )


def _soft(
    *,
    tool: str,
    error: str,
    instruction: str,
    **fields: Any,
) -> ToolResult:
    """A recoverable miss: success, plus an error the model can act on.

    Phase 29 §5 freezes this shape (``success=True`` + ``error`` +
    ``instruction``) for a spec or item that does not exist. It is deliberately
    not an exception path: the model has a next move (list the specs, list the
    queue) and the tool must stay usable inside the same turn.
    """
    return _result({**fields, "error": error, "instruction": instruction}, tool=tool)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _resolve_export_dir(requested: str) -> tuple[Path | None, str | None]:
    """Keep export trees inside the review home unless explicitly allowed."""
    base = (review_home() / export_tree.DEFAULT_FOLDER).resolve()
    roots = [base]
    roots.extend(
        Path(extra).expanduser().resolve()
        for extra in capabilities.current().allowed_out_dirs
    )
    path = Path(requested).expanduser().resolve() if requested else base
    if any(path == root or root in path.parents for root in roots):
        return path, None
    return None, (
        "Use the default export folder or an explicitly allowed directory. Pass "
        "--allow-out-dir when starting the server if this location is intentional."
    )


def _clip(value: Any, limit: int | None = None) -> str:
    """Collapse whitespace and bound one learner-visible field."""
    # Read the module constant at call time rather than binding it as a default:
    # tests and operators tune the clip through the module attribute.
    if limit is None:
        limit = TEXT_LIMIT
    text = _text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _clip_verbatim(value: Any, limit: int) -> str:
    """Bound a content field without flattening it.

    Stems and option texts carry deliberate line breaks (sub-questions,
    displayed formulas). The metadata ``_clip`` collapses whitespace; content
    must arrive exactly as stored, so this variant only trims the ends and
    truncates past ``limit``.
    """
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _service() -> ReviewService:
    """A service for this call, resolving the store from the environment.

    Mirrors ``deeptutor.review.api._service``: the HTTP layer and the tool layer
    must read the same ``DEEPTUTOR_REVIEW_HOME``, and neither may cache a store
    across a home change (tests and the acceptance script both swap it).
    """
    return ReviewService(ReviewStore(read_only=True)) if capabilities.current().read_only else ReviewService()


async def _service_and_spec(
    *, tool: str, spec_id: Any, missing_instruction: str
) -> tuple[ReviewService | None, Any | None, ToolResult | None]:
    """Resolve a model-supplied spec id, or the soft failure to return instead."""
    ref = _text(spec_id)
    if not ref:
        return (
            None,
            None,
            _soft(
                tool=tool,
                error="spec_id is required",
                instruction=(
                    "Call review_specs first and pass one of the returned ids "
                    "verbatim."
                ),
            ),
        )
    try:
        service = await asyncio.to_thread(_service)
        spec = await asyncio.to_thread(service.store.get_spec, ref)
    except SpecNotFound:
        return (
            None,
            None,
            _soft(
                tool=tool,
                error=f"spec not found: {ref}",
                instruction=missing_instruction,
                spec_id=ref,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - mapped to an explicit failure
        return None, None, _internal_failure(tool, exc)
    return service, spec, None
# -- the eight tools --------------------------------------------------------


def _page_parameters() -> list[ToolParameter]:
    return [
        ToolParameter(name="limit", type="integer", required=False, minimum=1, maximum=100,
                      description="Page size, default 50 and maximum 100."),
        ToolParameter(name="offset", type="integer", required=False, minimum=0, maximum=1_000_000,
                      description="First row to return. Use next_offset from the previous result."),
    ]


def _read_page(kwargs: dict[str, Any]) -> tuple[int, int]:
    limit = kwargs.get("limit", 50)
    offset = kwargs.get("offset", 0)
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer from 1 to 100")
    if type(offset) is not int or not 0 <= offset <= 1_000_000:
        raise ValueError("offset must be an integer from 0 to 1000000")
    return limit, offset


def _page_result(count: int, offset: int, returned: int) -> dict[str, Any]:
    return {
        "count": count,
        "offset": offset,
        "next_offset": offset + returned if offset + returned < count else None,
    }


class ReviewSpecsTool(BaseTool):
    """The review packs available to this learner."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="review_specs",
            description=(
                "List the Review Studio packs available to this learner: each "
                "pack's id, title, how many items and concepts it holds, and "
                "whether it passed the generation quality gates. Call this "
                "first whenever the learner asks to review, revise, drill or "
                "be quizzed on something — every tool that reads or records "
                "takes a spec_id. Follow next_offset to see more stored packs. "
                "Packs that fail current validation carry a warning or failed QA status."
            ),
            parameters=_page_parameters(),
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        tool = "review_specs"
        try:
            limit, offset = _read_page(kwargs)
        except ValueError as exc:
            return _soft(tool=tool, error=str(exc), instruction="Pass a valid limit and offset.")
        try:
            service = await asyncio.to_thread(_service)
            count, details = await asyncio.to_thread(service.spec_page, limit=limit, offset=offset)
            rows: list[dict[str, Any]] = []
            for detail in details:
                spec_id = detail["spec_id"]
                quality = detail.get("quality") or {}
                counts = detail.get("counts") or {}
                row = {
                    "id": spec_id,
                    "title": _clip(detail.get("title") or spec_id),
                    "items": int(counts.get("items") or 0),
                    "concepts": int(counts.get("concepts") or 0),
                    "qa_passed": bool(quality.get("passed")),
                    "qa_warnings": len(quality.get("warnings") or []),
                    "updated_at": detail.get("updated_at"),
                }
                if detail.get("warning"):
                    row["warning"] = _clip(str(detail["warning"]), 200)
                rows.append(row)
        except Exception as exc:  # noqa: BLE001
            return _internal_failure(tool, exc)

        if rows:
            instruction = (
                "Match what the learner said against these titles and ids, then "
                "call review_today with the chosen id. Never invent or guess a "
                "spec_id, and do not call review_today for more than one pack "
                "unless the learner asks for a comparison."
            )
        elif count:
            instruction = "This page is empty. Start again at offset 0 or use a previous page."
        else:
            instruction = (
                "This learner has no review packs yet. Say so plainly and offer "
                "to build one from their material; do not invent a spec_id or "
                "describe packs that do not exist."
            )
        page = _page_result(count, offset, len(rows))
        if page["next_offset"] is not None:
            instruction += " Pass next_offset as offset to see more packs."
        return _result(
            {**page, "specs": rows, "instruction": instruction},
            tool=tool,
        )


class ReviewDueTool(BaseTool):
    """Cross-pack due summary: which packs have work today, and how much."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="review_due",
            description=(
                "Summarise what is due across every pack in one read-only call: "
                "for each pack that has work today, how many items its queue "
                "would present, how many are due review vs new material, the "
                "estimated minutes, and when it was last reviewed. Use it to "
                "answer 'what should I review today?' or 'which pack should I "
                "start with?' before choosing one pack for review_today. "
                "Unlike review_today it opens no session and records nothing, "
                "so it is safe to call repeatedly, and it does not return any "
                "question text — call review_today for the actual items. "
                "Totals cover all matching packs. Follow next_offset for more rows."
            ),
            parameters=[
                ToolParameter(
                    name="subject",
                    type="string",
                    required=False,
                    description=(
                        "Optional subject filter: only packs whose title "
                        "contains this text (case-sensitive substring)."
                    ),
                ),
                *_page_parameters(),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        tool = "review_due"
        subject = _text(kwargs.get("subject"))
        try:
            limit, offset = _read_page(kwargs)
        except ValueError as exc:
            return _soft(tool=tool, error=str(exc), instruction="Pass a valid limit and offset.")
        try:
            service = await asyncio.to_thread(_service)
            overview = await asyncio.to_thread(
                service.due_overview, subject=subject or None, limit=limit, offset=offset
            )
        except Exception as exc:  # noqa: BLE001
            return _internal_failure(tool, exc)

        rows: list[dict[str, Any]] = []
        for row in overview.get("packs") or []:
            rows.append(
                {
                    "spec_id": _text(row.get("spec_id")),
                    "title": _clip(row.get("title")),
                    "subject": _clip(row.get("subject")),
                    "queued": int(row.get("queued") or 0),
                    "due": int(row.get("due") or 0),
                    "new": int(row.get("new") or 0),
                    "repeat": int(row.get("repeat") or 0),
                    "estimated_minutes": int(row.get("estimated_minutes") or 0),
                    "recovery": bool(row.get("recovery")),
                    "last_reviewed_at": row.get("last_reviewed_at"),
                }
            )
        totals = overview.get("totals") or {}
        page = _page_result(overview["count"], offset, len(rows))
        if rows:
            instruction = (
                "These are queue counts, not question text: pick one pack with "
                "the learner (or the largest `queued` when they did not say), "
                "then call review_today with its `spec_id` to start. `due` plus "
                "`new` may be larger than `queued` when the daily budget cut the "
                "queue; do not tell the learner they must clear every one. A "
                "pack with `recovery: true` deliberately shows only a handful — "
                "say that resuming is enough and do not call it a backlog."
            )
        elif page["count"]:
            instruction = "This page is empty. Use a previous page to see the queued packs."
        else:
            instruction = (
                "Nothing is queued today"
                + (f" for packs matching {subject!r}" if subject else "")
                + ". Say so plainly; offer to check review_specs for packs, or "
                "to generate new material. Do not invent due items."
            )
        if page["next_offset"] is not None:
            instruction += " Pass next_offset as offset with the same subject to see more packs."
        if overview["omitted"]:
            instruction += " Some stored packs could not be validated. Check review_specs for warnings."
        return _result(
            {
                "generated_at": overview.get("generated_at"),
                "subject": overview.get("subject"),
                **page,
                "omitted": overview["omitted"],
                "totals": {
                    "queued": int(totals.get("queued") or 0),
                    "due": int(totals.get("due") or 0),
                    "new": int(totals.get("new") or 0),
                    "repeat": int(totals.get("repeat") or 0),
                    "estimated_minutes": int(totals.get("estimated_minutes") or 0),
                },
                "packs": rows,
                "instruction": instruction,
            },
            tool=tool,
        )


class ReviewTodayTool(BaseTool):
    """Today's queue for one pack, with the answers withheld."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="review_today",
            description=(
                "Return today's review queue for one Review Studio pack: the "
                "item budget, the planned minutes, and each queued item's "
                "kind, stem and options. Answers and explanations are "
                "deliberately "
                "not included, so the questions can be asked honestly. Use it "
                "when the learner says they want to start or continue "
                "reviewing; then read one item at a time and record the "
                "learner's own rating with review_attempt."
            ),
            parameters=[
                ToolParameter(
                    name="spec_id",
                    type="string",
                    description="Pack id from review_specs (verbatim).",
                )
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        tool = "review_today"
        service, spec, error = await _service_and_spec(
            tool=tool,
            spec_id=kwargs.get("spec_id"),
            missing_instruction=(
                "That pack does not exist. Call review_specs for the ids this "
                "learner actually has, then call review_today again."
            ),
        )
        if error is not None or service is None or spec is None:
            return error or _failure("review_today could not resolve the pack.")

        try:
            today = await asyncio.to_thread(
                service.today, spec.id, read_only=capabilities.current().read_only
            )
        except Exception as exc:  # noqa: BLE001
            return _internal_failure(tool, exc)

        items: list[dict[str, Any]] = []
        for row in today.get("items") or []:
            item: dict[str, Any] = {
                "item_id": row.get("item_id", ""),
                "concept_id": row.get("concept_id", ""),
                "kind": row.get("kind", ""),
                "status": row.get("status", ""),
                "reason": _clip(row.get("reason")),
                "difficulty": row.get("difficulty", ""),
                "stem": _clip_verbatim(row.get("stem"), STEM_LIMIT),
            }
            options = [
                {
                    "label": _text(option.get("label")),
                    "text": _clip_verbatim(option.get("text"), OPTION_LIMIT),
                }
                for option in row.get("options") or []
            ]
            if options:
                item["options"] = options
            items.append(item)

        session_id = str(today.get("session_id") or "")
        if items:
            instruction = (
                f"Read the first item's stem to the learner exactly as given, "
                f"with its options when present, and do not reveal or hint at "
                f"the answer — you do not have it. If the learner asks for the "
                f"answer, asks you to confirm a guess, or states what they "
                f"think it is before answering, neither confirm nor deny and do "
                f"not use your own subject knowledge to solve it for them: "
                f"they must answer first. Then stop and wait: the "
                f"learner answers, and separately says how it went. Only after "
                f"that, call review_attempt with spec_id={spec.id!r}, that "
                f"item_id, the rating they gave in their own words, and "
                f"session_id={session_id!r} (so a retest in this same session is "
                f"scored correctly). Work through the list in order; do not "
                f"invent items, and do not ask the learner to rate the item for "
                f"you."
            )
        else:
            instruction = (
                "Nothing is due in this pack today. Tell the learner that, and "
                "offer review_weakness (practise the weakest concepts) or "
                "review_export_link (take the offline page). Do not invent "
                "items or claim a backlog you were not given."
            )

        return _result(
            {
                "spec_id": spec.id,
                "title": _clip(getattr(spec, "title", "")),
                "session_id": session_id,
                "budget": today.get("budget", 0),
                "planned_minutes": today.get("planned_minutes", 0),
                "estimated_minutes": today.get("estimated_minutes", 0),
                "counts": today.get("counts") or {},
                "backlog": bool(today.get("backlog")),
                "recovery_active": bool((today.get("recovery") or {}).get("active")),
                "exam": today.get("exam"),
                "items": items,
                "empty_message": today.get("empty_message"),
                "instruction": instruction,
            },
            tool=tool,
        )


class ReviewHintTool(BaseTool):
    """A deterministic hint ladder: help without ever restating the answer."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="review_hint",
            description=(
                "Give the learner a hint from the hint ladder for one review item without "
                "revealing the answer. Level 1 points at the tested concept; "
                "level 2 restates the concept's definition (and formula when "
                "the pack has one). The text is derived deterministically "
                "from the pack and checked against the correct answer and "
                "every option; when that check fails the tool returns a "
                "source-guidance fallback instead. Offer level 1 first, level "
                "2 only if the learner is still stuck. Count every hint you "
                "showed and pass that count as hints_used to review_attempt. "
                "When the learner gives up, take the answer path through the "
                "attempt flow instead of extending the hints."
            ),
            parameters=[
                ToolParameter(
                    name="spec_id",
                    type="string",
                    description="Pack id from review_specs / review_today (verbatim).",
                ),
                ToolParameter(
                    name="item_id",
                    type="string",
                    description="Item id from review_today (verbatim).",
                ),
                ToolParameter(
                    name="level",
                    type="integer",
                    description="1 = concept direction, 2 = definition/formula. Offer 1 first.",
                    required=False,
                    default=1,
                    minimum=1,
                    maximum=2,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        tool = "review_hint"
        item_id = _text(kwargs.get("item_id"))
        level = _bounded_int(kwargs.get("level"), default=1, minimum=1, maximum=2)
        service, spec, error = await _service_and_spec(
            tool=tool,
            spec_id=kwargs.get("spec_id"),
            missing_instruction=(
                "That pack does not exist. Call review_specs for the ids this "
                "learner has, then call review_hint again."
            ),
        )
        if error is not None or service is None or spec is None:
            return error or _failure("review_hint could not resolve the pack.")
        if not item_id:
            return _soft(
                tool=tool,
                error="item_id is required",
                instruction=(
                    "Call review_today first and use one of its item ids "
                    "verbatim."
                ),
                spec_id=spec.id,
            )
        try:
            item = spec.item_by_id(item_id)
            concept = spec.concept_by_id(item.concept_ids[0])
        except KeyError:
            return _soft(
                tool=tool,
                error=f"unknown item: {item_id}",
                instruction=(
                    "That item is not in this pack. Call review_today for the "
                    "current queue and use one of its item ids."
                ),
                spec_id=spec.id,
                item_id=item_id,
            )
        except Exception as exc:  # noqa: BLE001
            return _internal_failure(tool, exc)

        # Level 1 only names the tested concept. A one-line summary often
        # restates the reasoning (or the answer) in different words, which the
        # lexical leak gate cannot reliably detect; the summary therefore
        # belongs to level 2, where asking for the full definition is expected.
        # Generated packs may only carry a long concept name, so level 1 uses a
        # bounded brief of it; the leak gate below still applies.
        name_hint = _text(getattr(concept, "name", ""))
        summary = _text(getattr(concept, "summary_one_line", ""))
        if level == 1:
            hint_text = _hint_brief(name_hint)
        else:
            definition = _text(getattr(concept, "definition", ""))
            formulas = [
                _text(getattr(formula, "latex", ""))
                for formula in (getattr(concept, "formulas", None) or [])
            ]
            formulas = [formula for formula in formulas if formula]
            if definition:
                hint_text = definition
            elif summary:
                # A pack may store only the one-line summary. Level 2 is the
                # definition rung, so present the summary as a prompt to
                # complete the full definition and its boundary. Truncate the
                # summary first so the suffix survives the final 800-char clip
                # even for a very long summary.
                suffix = "（请补全完整定义与适用条件，再对照来源材料核对）"
                budget = max(1, HINT_TEXT_CHARS - len(suffix))
                hint_text = f"{_hint_brief(summary, budget)}{suffix}"
            else:
                # Generated packs may carry only a concept name. Returning the
                # level-1 brief verbatim would make the second rung useless;
                # point at the definition and its boundary instead. The leak
                # gate below still runs on the result.
                hint_text = (
                    f"回忆「{name_hint}」的完整定义与适用条件，"
                    "再对照来源材料核对这些细节。"
                )
            if formulas:
                hint_text = (
                    f"{hint_text} 相关公式：{formulas[0]}"
                    if hint_text
                    else f"相关公式：{formulas[0]}"
                )

        fallback = False
        source: dict[str, Any] | None = None
        paraphrased = _paraphrase_leaks(hint_text, item)
        verbatim = (
            not hint_text
            or _hint_leaks(hint_text, item)
            or _written_hint_leaks(hint_text, item)
        )
        if verbatim or paraphrased:
            hint_text, source = _safe_hint(level, item, spec)
            fallback = True

        return _result(
            {
                "spec_id": spec.id,
                "item_id": item.id,
                "level": level,
                "hint": _clip_verbatim(hint_text, HINT_TEXT_CHARS),
                "fallback": fallback,
                "source": source,
                "instruction": (
                    "Read this hint to the learner and let them try again; do "
                    "not add the answer or extra explanation beyond the hint. "
                    "If they used hints and then answer, pass hints_used to "
                    "review_attempt so the scheduler does not treat it as "
                    "unaided recall."
                ),
            },
            tool=tool,
        )


#: Level-1 hints cut a long concept name down to this many characters so the
#: first rung gives direction without dictating the whole statement.
HINT_BRIEF_CHARS = 80

#: Final ceiling for one hint payload. Level 2 reserves room for its suffix
#: before truncating the concept text, so the two rungs never collapse into
#: the same 800 characters.
HINT_TEXT_CHARS = 800


def _hint_brief(text: str, limit: int = HINT_BRIEF_CHARS) -> str:
    """A bounded brief of a long concept name, used at hint level 1.

    Cut at a natural punctuation boundary when one exists near the limit; a
    hard character cut leaves half a term such as "最佳导通…" and makes the
    learner guess the rest.
    """
    if len(text) <= limit:
        return text
    window = text[: limit - 1]
    best = -1
    for separator in ("；", "，", "、", "：", ";", ",", ":", " "):
        best = max(best, window.rfind(separator))
    if best >= 12:
        return window[: best + 1].rstrip() + "…"
    return window + "…"


def _bigrams(text: str) -> set[str]:
    normalized = re.sub(r"[^0-9A-Za-z一-鿿]+", "", text)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[index : index + 2] for index in range(len(normalized) - 1)}


def _paraphrase_leaks(hint: str, item: Any) -> bool:
    """Detect choice conclusions and distinctive phrases without length dilution."""
    if getattr(item, "kind", "") != "single_choice":
        return False
    answer = getattr(item, "answer", None)
    options = list(getattr(item, "options", None) or [])
    choice_index = getattr(answer, "choice_index", None)
    if isinstance(choice_index, int) and 0 <= choice_index < len(options):
        target = str(getattr(options[choice_index], "text", ""))
    else:
        target = str(getattr(answer, "text", "") or "")
    hint_grams = _bigrams(hint)
    target_grams = _bigrams(target)
    if not hint_grams or not target_grams:
        return False
    compact_hint = _hint_compact(hint)
    compact_target = _hint_compact(target)
    compact_stem = _hint_compact(str(getattr(item, "stem", "")))
    formula_hint = _hint_math_text(hint)
    for fragment in re.findall(r"[A-Za-z0-9α-ωΑ-Ω_{}^=+*/|<>-]{4,}", re.sub(r"\s+", "", formula_hint)):
        formula = _hint_compact(fragment)
        if len(formula) >= 4 and formula in compact_target and formula not in compact_stem:
            return True
    # Only phrases the containing question and the distractors do not already
    # use are distinctive: a 4-gram every option shares is the topic, not the
    # answer.
    stem_phrases = _four_grams(compact_stem)
    other_phrases = {
        phrase
        for index, option in enumerate(options)
        if index != choice_index
        for phrase in _four_grams(_hint_compact(str(getattr(option, "text", ""))))
    }
    hint_phrases = _four_grams(compact_hint) - stem_phrases
    target_phrases = _four_grams(compact_target) - stem_phrases
    if (hint_phrases & target_phrases) - other_phrases:
        return True
    # The bigram ratio is measured on the same reduced sets. Without the
    # subtraction a short option saturates: it shares half of its bigrams with
    # any hint that merely reuses the stem's wording.
    stem_grams = _bigrams(str(getattr(item, "stem", "")))
    other_grams = {
        gram
        for index, option in enumerate(options)
        if index != choice_index
        for gram in _bigrams(str(getattr(option, "text", "")))
    }
    hint_grams = hint_grams - stem_grams - other_grams
    target_grams = target_grams - stem_grams - other_grams
    if not hint_grams or not target_grams:
        return False
    shared = hint_grams & target_grams
    return (
        len(shared) / len(hint_grams) >= 0.25
        or len(shared) / len(target_grams) >= 0.4
    )


def _four_grams(text: str) -> set[str]:
    return {text[offset : offset + 4] for offset in range(len(text) - 3)}


def _hint_compact(text: str) -> str:
    """Drop layout characters only, keeping the casing that carries meaning.

    ``mW``/``MW`` and ``ω``/``Ω`` differ by case alone, so this no longer
    casefolds. It also no longer deletes every ``的``: that spliced characters
    which never touched in the source and fabricated 4-grams the pack never
    contained. NFKC (inside :func:`_hint_math_text`) already folds widths.
    """
    normalized = _hint_math_text(text)
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fffα-ωΑ-Ω]", "", normalized)


def _hint_math_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\\(?:cdot|times)\b", "*", text)
    text = re.sub(r"\\[,!;:]|\\(?:quad|qquad|enspace|thinspace)\b", " ", text)
    text = re.sub(r"\\(?:mathrm|textrm|text|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:d?frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\(?:eta|xi|theta)\b", lambda match: {"\\eta": "η", "\\xi": "ξ", "\\theta": "θ"}[match.group()], text)
    symbols = {"varphi": "φ", "phi": "φ", "pi": "π", "omega": "ω", "lt": "<", "gt": ">", "vert": "|"}
    text = re.sub(r"\\(varphi|phi|pi|omega|lt|gt|vert)(?![A-Za-z])", lambda match: symbols[match.group(1)], text)
    return text.replace("\\%", "%").replace("$", "").replace("{", "").replace("}", "").replace("_", "")


def _hint_numbers(text: str) -> set[Decimal]:
    values: set[Decimal] = set()
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    coefficient = r"(?:[A-Za-zα-ωΑ-Ω]|\s*[×*·]\s*[A-Za-zα-ωΑ-Ω]|\s+[α-ωΑ-Ω])"
    units = r"(?:W|V|A|Hz|Ω)\b"

    def collect_fraction(match: re.Match[str]) -> str:
        suffix = text[match.end():match.end() + 64]
        if not re.match(coefficient, suffix) or re.match(units, suffix):
            try:
                values.add(Decimal(match.group(1)) / Decimal(match.group(2)))
            except DecimalException:
                pass
        return " " * len(match.group())

    remainder = re.sub(rf"\(({number})\)\s*/\s*\(({number})\)", collect_fraction, text)
    for match in re.finditer(rf"(?<![A-Za-z0-9_.])({number})(?![\d.])", remainder):
        suffix = remainder[match.end():match.end() + 64]
        if re.match(coefficient, suffix) and not re.match(units, suffix):
            continue
        try:
            values.add(Decimal(match.group(1)))
        except DecimalException:
            continue
    return values


#: SI prefixes a hint may attach to a unit, as written: ``mW`` is a milliwatt
#: and ``MW`` a megawatt, so this table is never casefolded.
_UNIT_PREFIXES = {
    "": Decimal(1),
    "p": Decimal("1e-12"),
    "n": Decimal("1e-9"),
    "u": Decimal("1e-6"),
    "μ": Decimal("1e-6"),
    "µ": Decimal("1e-6"),
    "m": Decimal("1e-3"),
    "k": Decimal("1e3"),
    "M": Decimal("1e6"),
    "G": Decimal("1e9"),
}

_UNIT_QUANTITY = re.compile(
    r"(?<![A-Za-z0-9_.])([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*"
    r"([pnumkMGμµ]?)(Hz|Ω|V|A|W|F|s|%)(?![A-Za-z0-9])"
)


def _hint_quantities(text: str) -> set[tuple[Decimal, str]]:
    """Value+unit pairs normalized to base units: 500 mA -> (0.5, A), 25% -> (0.25, %)."""
    quantities: set[tuple[Decimal, str]] = set()
    for match in _UNIT_QUANTITY.finditer(_hint_math_text(text)):
        unit = match.group(3)
        try:
            value = Decimal(match.group(1))
            value = value / 100 if unit == "%" else value * _UNIT_PREFIXES[match.group(2)]
        except (DecimalException, KeyError):
            continue
        quantities.add((value, unit))
    return quantities


def _label_core(label: str) -> str:
    """The bare label token: NFKC plus surrounding punctuation removed ("（A）" -> "A")."""
    normalized = unicodedata.normalize("NFKC", str(label or "")).strip()
    return re.sub(r"^[^0-9A-Za-z\u4e00-\u9fff]+|[^0-9A-Za-z\u4e00-\u9fff]+$", "", normalized)


def _label_leaks(hint: str, label: str) -> bool:
    """Whether the hint names ``label`` as the answer, in Chinese or English.

    Callers only pass the *correct* option's label, so naming a distractor is
    not a leak. The label must stand alone as a token: "正确的是B" counts,
    "正确的是BC" does not name option B.
    """
    core = _label_core(label)
    if not core:
        return False
    token = rf"(?<![A-Za-z0-9]){re.escape(core)}(?![A-Za-z0-9])"
    text = unicodedata.normalize("NFKC", _hint_math_text(hint))
    chinese = (
        rf"(?:正确(?:答案|选项)?(?:的)?(?:是|为)?|答案(?:是|为)?|"
        rf"应选择|应选|选择|选)\s*[：:]?\s*(?:选项\s*)?{token}"
    )
    english = (
        rf"(?:{token}\s+is\s+(?:the\s+)?correct(?:\s+(?:answer|option|choice))?"
        rf"|correct\s+(?:answer|choice|option)(?:\s+is)?\s*[：:]?\s*(?:option\s*)?{token}"
        rf"|answer\s*[:：]\s*(?:option\s*)?{token}"
        rf"|(?:choose|select)\s*[:：]?\s*(?:option\s*)?{token})"
    )
    return bool(re.search(f"(?:{chinese}|{english})", text, re.IGNORECASE))


#: A one-character answer only counts as revealed when a phrase points at it
#: and the character ends the clause: "答案是对的", not "对错参半".
_REVEAL_TAIL = r"(?=$|[\s，。；、,;:：!！?？）)】」]|的|了)"


def _short_answer_reveals(hint: str, term: str) -> bool:
    """Whether the hint spells out a one-character answer after a reveal phrase."""
    needle = str(term or "").strip()
    if len(needle) != 1:
        return False
    text = unicodedata.normalize("NFKC", _hint_math_text(hint))
    return bool(re.search(
        rf"(?:正确(?:答案|选项)?|答案|结论|判断|应选择|应选|选择|选)\s*"
        rf"(?:是|为|应为|就是)?\s*[：:]?\s*(?:选项\s*)?{re.escape(needle)}{_REVEAL_TAIL}",
        text,
    ))


def _written_hint_leaks(hint: str, item: Any) -> bool:
    if getattr(item, "kind", "") not in {"short_answer", "flashcard", "numeric"}:
        return False
    answer = str(getattr(getattr(item, "answer", None), "text", "") or "")
    stem = str(getattr(item, "stem", ""))
    normalized_answer = _hint_math_text(answer)
    normalized_hint = _hint_math_text(hint)
    normalized_stem = _hint_math_text(stem)
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    scalar = rf"(?:{number}|\({number}\)\s*/\s*\({number}\))"
    results = re.findall(
        rf"(?:=|≈|为|等于|得到|得出)\s*({scalar})(?![\d.])"
        rf"(?=\s*(?:$|[,，。；;、%）)\]]|与|和|并|且|[WVAΩ]\b|瓦|伏|安|欧姆))",
        normalized_answer,
    )
    results.extend(re.findall(
        rf"(?<![A-Za-z0-9_.])({number})\s*(?:%|W\b|V\b|A\b|Ω|瓦|伏|安|欧姆)",
        normalized_answer,
    ))
    result_values = set().union(*(_hint_numbers(result) for result in results))
    value = getattr(getattr(item, "answer", None), "value", None)
    if value is not None:
        result_values.add(Decimal(str(value)))
    if (result_values & _hint_numbers(normalized_hint)) - _hint_numbers(normalized_stem):
        return True
    if _short_answer_reveals(hint, answer):
        return True
    compact_hint = _hint_compact(hint)
    compact_stem = _hint_compact(stem)
    compact_answer = _hint_compact(answer)
    if (
        len(compact_answer) >= 2 and compact_answer in compact_hint
        and compact_answer not in compact_stem
    ):
        return True
    # Unit-aware comparison: "500 mA" states the same quantity as an answer of
    # 0.5 A once both sides are normalized to base units. A unit-less answer
    # bridges through its magnitude, so answer 1000 still catches "1 kΩ".
    hint_quantities = _hint_quantities(normalized_hint)
    answer_quantities = _hint_quantities(normalized_answer)
    stem_quantities = _hint_quantities(normalized_stem)
    if (hint_quantities & answer_quantities) - stem_quantities:
        return True
    stem_numbers = _hint_numbers(normalized_stem)
    answer_numbers = _hint_numbers(normalized_answer)
    answer_value = getattr(getattr(item, "answer", None), "value", None)
    if answer_value is not None:
        answer_numbers.add(Decimal(str(answer_value)))
    hint_magnitudes = {magnitude for magnitude, _ in hint_quantities}
    if (
        (hint_magnitudes & answer_numbers)
        - stem_numbers
        - {magnitude for magnitude, _ in stem_quantities}
    ):
        return True
    # A pack may state a tolerance instead of an exact value; any hint number
    # inside that band still gives the answer away.
    tolerance = getattr(getattr(item, "answer", None), "tolerance", 0.0) or 0.0
    if answer_value is not None and tolerance > 0:
        target = Decimal(str(answer_value))
        band = Decimal(str(tolerance))
        candidates = (_hint_numbers(normalized_hint) | hint_magnitudes) - stem_numbers
        if any(abs(number - target) <= band for number in candidates):
            return True
    if re.search(r"推导|写出|给出.*(?:公式|关系|表达式)|derive|derive.*relation", stem, re.IGNORECASE):
        equations = re.findall(r"[A-Za-zα-ωΑ-Ω][A-Za-z0-9_{}₀-₉]*=[A-Za-z0-9_α-ωΑ-Ω+*/^=().{}₀-₉-]{2,100}", re.sub(r"\s+", "", normalized_hint))
        answer_equations = _hint_compact(normalized_answer)
        if any(
            len(_hint_compact(equation)) >= 4
            and _hint_compact(equation) in answer_equations
            and _hint_compact(equation) not in compact_stem
            for equation in equations
        ):
            return True
    for clause in re.split(
        r"[，。；;:\n]|(?<!\d)\.(?!\d)|\b(?:because|since)\b|因为|由于",
        answer, flags=re.IGNORECASE,
    ):
        compact_clause = _hint_compact(clause)
        if (
            len(compact_clause) >= 8
            and compact_clause in compact_hint
            and compact_clause not in compact_stem
        ):
            return True
        conclusion = re.search(
            r"(?:应(?:当)?(?:选择|工作在)|工作在|处于|必须|只能|因此|所以|结果(?:为|是))(.{4,})",
            clause,
        )
        if conclusion:
            phrase = _hint_compact(conclusion.group(1))
            if len(phrase) >= 4 and phrase in compact_hint and phrase not in compact_stem:
                return True
    return False


_GREEK_COMMANDS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "eta": "η",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "phi": "φ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "xi": "ξ",
}


def _stem_variables(stem: str) -> list[str]:
    """Extract up to four subscripted variable names from the question stem."""
    tokens: list[str] = []
    for match in re.finditer(
        r"(\\[A-Za-z]{1,10}|[A-Za-z]{1,2})_\{?([A-Za-z0-9,]+)\}?", stem
    ):
        base = match.group(1)
        if base.startswith("\\"):
            base = _GREEK_COMMANDS.get(base[1:], base[1:])
        token = f"{base}_{match.group(2)}"
        if token not in tokens:
            tokens.append(token)
        if len(tokens) >= 4:
            break
    return tokens


def _safe_hint(
    level: int, item: Any, spec: Any
) -> tuple[str, dict[str, Any] | None]:
    """A distinct, non-leaking hint used when the concept name paraphrases."""
    if level == 1:
        _, source = _hint_fallback(spec, item)
        variables = _stem_variables(str(getattr(item, "stem", "")))
        if variables:
            return (
                "先梳理题干中 "
                + "、".join(variables)
                + " 之间的关系，再判断本题考的是哪一种状态、条件或波形。",
                source,
            )
        return (
            "先找出题干给出的已知量与所求量之间的关系，再判断本题的考点。",
            source,
        )
    return _hint_fallback(spec, item)


def _hint_leaks(hint: str, item: Any) -> bool:
    """Whether a hint restates the answer or any option verbatim."""
    text = _text(hint)
    if not text:
        return True
    pieces: list[str] = []
    answer = getattr(item, "answer", None)
    options = list(getattr(item, "options", None) or [])
    choice_index = getattr(answer, "choice_index", None)
    chosen = (
        options[choice_index]
        if isinstance(choice_index, int) and 0 <= choice_index < len(options)
        else None
    )
    if chosen is not None and _label_leaks(
        text, str(getattr(chosen, "label", ""))
    ):
        return True
    if chosen is not None:
        pieces.append(getattr(chosen, "text", ""))
    answer_text = getattr(answer, "text", "")
    if answer_text:
        pieces.append(answer_text)
    answer_value = getattr(answer, "value", None)
    if answer_value is not None:
        pieces.append(f"{answer_value:g}")
    for option in options:
        pieces.append(getattr(option, "text", ""))
    for piece in pieces:
        needle = _text(piece)
        # Very short fragments (a lone number, one character) would match
        # almost any prose; only verbatim phrases of four characters or more
        # count as a leak. A one-character answer is caught below, but only
        # when a phrase such as "答案是对" points at it.
        if len(needle) >= 4 and needle in text:
            return True
    if _short_answer_reveals(text, answer_text) or (
        chosen is not None
        and _short_answer_reveals(text, getattr(chosen, "text", ""))
    ):
        return True
    return False


def _hint_fallback(spec: Any, item: Any) -> tuple[str, dict[str, Any] | None]:
    """A source-anchored fallback hint that cannot leak the answer.

    Source metadata is authored content, not a trusted string: a title or
    ref that happens to restate the answer must still degrade to the
    generic guidance.
    """
    # ``ReviewSpec.sources`` is a mapping keyed by source id.
    sources = dict(getattr(spec, "sources", None) or {})
    for source_id in getattr(item, "source_ids", None) or []:
        source = sources.get(str(source_id))
        if source is None:
            continue
        title = _text(getattr(source, "title", "")) or _text(
            getattr(source, "ref", "")
        )
        ref = _text(getattr(source, "ref", ""))
        label = f"《{title}》"
        if ref and ref != title:
            label += f" 的「{ref}」部分"
        anchored = f"回到来源材料{label}，复习相关段落后再作答。"
        if _hint_leaks(anchored, item):
            break
        return anchored, {"title": title, "ref": ref}
    return (
        "先回顾该题涉及的概念要点后再作答（本包没有更细的来源锚点）。",
        None,
    )


class ReviewAttemptTool(BaseTool):
    """Record the learner's own rating; never compute one."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="review_attempt",
            description=(
                "Record the learner's self-assessment of one review item and let "
                "the spaced-repetition scheduler reschedule it. ONLY call after "
                "the learner self-assesses aloud — they said, in their own "
                "words, that they got it easily, with effort, or not at all; map "
                "that to good, hard or again. Never grade on their behalf and "
                "never infer a rating from whether their answer looked right: "
                "this rating is the learner's real review history, and a rating "
                "you invented silently corrupts every future review date. If "
                "their words are vague or noncommittal, ask which of the "
                "three fits before calling this tool; never map a vague "
                "answer to a rating yourself. Use "
                "the item_id and session_id from review_today. Calling again "
                "with the same submission_id returns the original event instead "
                "of recording a second one. Even if the learner asks you to "
                "record a rating for an item they have not answered — for "
                'example "just mark it as known" — decline: present the item '
                "instead and explain that a rating without an attempt would "
                "push their next review date into fiction. Recording it after "
                "they answer is always fine."
            ),
            parameters=[
                ToolParameter(
                    name="spec_id",
                    type="string",
                    description="Pack id from review_specs / review_today (verbatim).",
                ),
                ToolParameter(
                    name="item_id",
                    type="string",
                    description="Item id from review_today (verbatim).",
                ),
                ToolParameter(
                    name="rating",
                    type="string",
                    enum=["again", "hard", "good"],
                    description=(
                        "The learner's own verdict: 'again' = could not recall "
                        "it, 'hard' = recalled with effort, 'good' = recalled "
                        "easily. Transcribe it; never pick one yourself."
                    ),
                ),
                ToolParameter(
                    name="session_id",
                    type="string",
                    description=(
                        "Session id returned by review_today. Pass it back so the "
                        "scheduler can tell a same-session retest from a fresh "
                        "first answer."
                    ),
                    required=False,
                ),
                ToolParameter(
                    name="submission_id",
                    type="string",
                    description=(
                        "Optional idempotency key. Reuse the same value only when "
                        "retrying the exact same rating for the same item; a "
                        "reused key with a different rating is rejected."
                    ),
                    required=False,
                ),
                ToolParameter(
                    name="hints_used",
                    type="integer",
                    description=(
                        "How many review_hint hints you showed for this item "
                        "before the learner answered (0-2). A hinted answer is "
                        "stored as 'hard' when the learner says 'good'."
                    ),
                    required=False,
                    default=0,
                    minimum=0,
                    maximum=2,
                ),
                ToolParameter(
                    name="confidence",
                    type="integer",
                    description=(
                        "The learner's own confidence in their answer, 0-100, "
                        "only when they volunteer it. Recorded for calibration; "
                        "never changes the rating and never invented by you."
                    ),
                    required=False,
                    minimum=0,
                    maximum=100,
                ),
                ToolParameter(
                    name="learner_quote",
                    type="string",
                    description=(
                        "The learner's own words for how it went, transcribed "
                        "as spoken (for example '会，但是想了很久'). Pass it "
                        "whenever they phrased the rating themselves: it is "
                        "stored with the event so the rating can be audited "
                        "later. Never paraphrase, summarise or invent it, and "
                        "never put your own assessment here."
                    ),
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        tool = "review_attempt"
        blocked = capabilities.disabled(tool)
        if blocked:
            return _soft(tool=tool, error=f"{tool} is disabled", instruction=blocked)
        item_id = _text(kwargs.get("item_id"))
        rating = _text(kwargs.get("rating")).lower()
        session_id = _text(kwargs.get("session_id"))
        submission_id = _text(kwargs.get("submission_id"))
        raw_hints = kwargs.get("hints_used")
        hints_used = 0
        if raw_hints not in (None, ""):
            # Strict on purpose: ``int(True)`` is 1 and ``int("2")`` is 2, but
            # neither is the integer the tool schema asked for. A coercing
            # parse hides caller bugs and lets a stringly-typed client write
            # a hint count nobody intended.
            if (
                isinstance(raw_hints, bool)
                or not isinstance(raw_hints, int)
                or not 0 <= raw_hints <= 2
            ):
                return _soft(
                    tool=tool,
                    error="hints_used must be an integer from 0 to 2",
                    instruction=(
                        "Pass the number of hints you showed (0, 1 or 2), or "
                        "omit it when none were used."
                    ),
                    item_id=item_id,
                )
            hints_used = raw_hints
        confidence: float | None = None
        raw_confidence = kwargs.get("confidence")
        if raw_confidence not in (None, ""):
            if isinstance(raw_confidence, bool) or not isinstance(
                raw_confidence, (int, float)
            ):
                return _soft(
                    tool=tool,
                    error="confidence must be a number between 0 and 100",
                    instruction=(
                        "Pass the learner's confidence as a number from 0 to "
                        "100, or omit it when they did not give one."
                    ),
                    item_id=item_id,
                )
            try:
                confidence = round(float(raw_confidence), 1)
            except (TypeError, ValueError):
                return _soft(
                    tool=tool,
                    error="confidence must be a number between 0 and 100",
                    instruction=(
                        "Pass the learner's confidence as a number from 0 to "
                        "100, or omit it when they did not give one."
                    ),
                    item_id=item_id,
                )
            if not (0.0 <= confidence <= 100.0):
                return _soft(
                    tool=tool,
                    error="confidence must be between 0 and 100",
                    instruction=(
                        "Pass the learner's confidence as a number from 0 to "
                        "100, or omit it when they did not give one."
                    ),
                    item_id=item_id,
                )

        if not item_id:
            return _soft(
                tool=tool,
                error="item_id is required",
                instruction=(
                    "Call review_today for this pack and pass one of the item "
                    "ids it returned."
                ),
            )
        if rating not in RATING_MAP:
            return _soft(
                tool=tool,
                error=f"unknown rating: {rating or '(missing)'}",
                instruction=(
                    "rating must be exactly one of again / hard / good, and it "
                    "must come from what the learner said. Ask them how it went "
                    "instead of choosing for them."
                ),
                item_id=item_id,
            )

        learner_quote: str | None = None
        raw_quote = kwargs.get("learner_quote")
        if raw_quote not in (None, ""):
            if not isinstance(raw_quote, str):
                return _soft(
                    tool=tool,
                    error="learner_quote must be a string",
                    instruction=(
                        "Pass the learner's own words verbatim as a string, or "
                        "omit it when they did not phrase the rating themselves."
                    ),
                    item_id=item_id,
                )
            learner_quote = raw_quote.strip()[:LEARNER_QUOTE_LIMIT] or None

        service, spec, error = await _service_and_spec(
            tool=tool,
            spec_id=kwargs.get("spec_id"),
            missing_instruction=(
                "That pack does not exist. Call review_specs, then review_today "
                "for the right pack, and retry with an item id from that queue."
            ),
        )
        if error is not None or service is None or spec is None:
            return error or _failure("review_attempt could not resolve the pack.")

        request = AttemptInput(
            item_id=item_id,
            rating=rating,
            session_id=session_id,
            submission_id=submission_id,
            hints_used=hints_used,
            confidence=confidence,
            learner_quote=learner_quote,
        )
        # Resolve the item here rather than catching ``KeyError`` around
        # ``attempt``: that method also raises ``KeyError`` when a stored spec
        # has a dangling concept reference, and calling a damaged pack "unknown
        # item" would send the model chasing a queue that cannot fix it.
        try:
            spec.item_by_id(item_id)
        except KeyError:
            return _soft(
                tool=tool,
                error=f"unknown item: {item_id}",
                instruction=(
                    "That item is not in this pack. Call review_today for the "
                    "current queue and use one of its item ids; do not reuse ids "
                    "from an older queue."
                ),
                spec_id=spec.id,
                item_id=item_id,
            )
        except Exception as exc:  # noqa: BLE001
            return _internal_failure(tool, exc)

        try:
            result = await asyncio.to_thread(service.attempt, spec.id, request)
        except SessionConflict:
            return _soft(
                tool=tool, error="session_conflict",
                instruction=(
                    "This session is closed or no longer available. Nothing was recorded. "
                    "Call review_today to obtain the current session_id, then retry "
                    "only the learner's actual assessment with that session."
                ),
                spec_id=spec.id, item_id=item_id, session_id=session_id,
            )
        except SubmissionConflict:
            return _soft(
                tool=tool,
                error="submission_id was already used with a different payload",
                instruction=(
                    "Nothing new was recorded. If you are retrying the same "
                    "attempt, pass the SAME submission_id with the exact same "
                    "payload; if this is a genuinely new attempt, a new "
                    "submission_id is fine. Never change the rating to make a "
                    "conflict go away."
                ),
                spec_id=spec.id,
                item_id=item_id,
            )
        except Exception as exc:  # noqa: BLE001
            return _internal_failure(tool, exc)

        # Projection runs after the Review event is durable: a core failure is
        # reported in-band and never turns a saved answer into an error. This
        # mirrors the HTTP attempt route exactly, so a chat and the web UI
        # cannot disagree about one attempt.
        projection = await project_attempt(service.store, spec.id, result)
        result["mastery_projection"] = projection
        if projection.get("status") in {"applied", "duplicate"}:
            try:
                result = await asyncio.to_thread(
                    service.apply_core_state, spec.id, result
                )
            except Exception as exc:  # noqa: BLE001 - display read only
                logger.warning("Could not overlay core state for %s: %r", spec.id, exc)

        created = bool(result.get("created"))
        retry = bool(result.get("retry"))
        if not created:
            instruction = (
                "This submission_id was already recorded, so nothing new was "
                "saved and no review date changed. Do not present it to the "
                "learner as a new attempt; continue with the next item."
            )
        else:
            instruction = (
                "The learner's own rating is recorded, and the next review date "
                "is in `next`. Tell them plainly what the correct answer and "
                "explanation were, and when they will see this card again. If "
                "they rated it 'again', walk them through `correction` before "
                "moving on. Then take the next item from the review_today list. "
                "Never record a rating the learner did not give. "
                "`requested_rating` is what the learner said and `rating` is "
                "what the scheduler stored; `next.due_at_ts` is a raw epoch "
                "(quote `next.due_at` instead); `predicted_retrievability` is a "
                "probability estimate, not a promise; if `mastery_projection."
                "status` is 'skipped', do not claim the learner's mastery was "
                "updated."
            )
            if retry and str(result.get("rating")) != str(
                result.get("requested_rating")
            ):
                instruction += (
                    " This was a retest after an 'again' in the same session, so "
                    "the scheduler recorded it as 'hard' even though the learner "
                    "said 'good' — say that, because seeing the answer first is "
                    "not unaided recall."
                )
            if hints_used and str(result.get("rating")) != str(
                result.get("requested_rating")
            ):
                instruction += (
                    " The learner answered after using hints, so the scheduler "
                    "recorded it as 'hard' even though they said 'good' — say "
                    "that, because a hinted answer is not unaided recall."
                )
            instruction += (
                " `study_aids` (assumptions, self_check, pitfalls) belongs to "
                "the post-answer explanation: use it to consolidate, never to "
                "preview the question."
            )

        return _result(
            {
                "spec_id": spec.id,
                "item_id": result.get("item_id", item_id),
                "concept_id": result.get("concept_id", ""),
                "concept_key": result.get("concept_key", ""),
                "event_id": result.get("event_id", ""),
                "session_id": result.get("session_id", session_id),
                "submission_id": result.get("submission_id", submission_id),
                "created": created,
                "rating": result.get("rating", rating),
                "requested_rating": result.get("requested_rating", rating),
                "retry": retry,
                "hints_used": int(result.get("hints_used") or 0),
                "confidence": result.get("confidence"),
                "learner_quote": result.get("learner_quote"),
                "next": {
                    "due_at": result.get("next_due"),
                    "due_at_ts": result.get("next_due_ts"),
                    "interval_days": result.get("scheduled_interval_days"),
                },
                "predicted_retrievability": result.get("predicted_retrievability"),
                "answer": result.get("answer"),
                "explanation": result.get("explanation"),
                "study_aids": result.get("study_aids"),
                "correction": result.get("correction"),
                "mastery_projection": projection,
                "instruction": instruction,
            },
            tool=tool,
        )


class ReviewWeaknessTool(BaseTool):
    """The weakest concepts in one pack, read-only."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="review_weakness",
            description=(
                "Show the weakest concepts of one Review Studio pack — at most "
                "the five weakest, ranked — with each concept's weakness "
                "score, predicted retrievability, and how much recorded "
                "evidence it rests on. Read-only: it creates no session and "
                "records nothing. Use it when the learner asks what they are "
                "weakest at, or what to prioritise; use review_today to "
                "actually practise."
            ),
            parameters=[
                ToolParameter(
                    name="spec_id",
                    type="string",
                    description="Pack id from review_specs (verbatim).",
                )
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        tool = "review_weakness"
        service, spec, error = await _service_and_spec(
            tool=tool,
            spec_id=kwargs.get("spec_id"),
            missing_instruction=(
                "That pack does not exist. Call review_specs for the ids this "
                "learner has, then call review_weakness again."
            ),
        )
        if error is not None or service is None or spec is None:
            return error or _failure("review_weakness could not resolve the pack.")

        try:
            report = await asyncio.to_thread(
                service.weakness, spec.id, limit=WEAKNESS_LIMIT
            )
            rows: list[dict[str, Any]] = []
            for row in report.get("weak_concepts") or []:
                concept_key = str(row.get("concept_key") or "")
                # "Evidence" is the number of review events this concept has
                # actually recorded. Core-sourced state may also hold evidence
                # from other surfaces; ``state_source`` says which it is, so the
                # count is never presented as the whole history.
                events = await asyncio.to_thread(
                    service.store.events_for_concept, spec.id, concept_key
                ) if concept_key else []
                rows.append(
                    {
                        "concept_id": row.get("concept_id", ""),
                        "concept_key": concept_key,
                        "concept_name": _clip(row.get("concept_name")),
                        "weakness": row.get("weakness"),
                        "predicted_retrievability": row.get("predicted_retrievability"),
                        "evidence_count": len(events),
                        "item_ids": list(row.get("item_ids") or []),
                        "item_count": row.get("item_count", 0),
                        "state_source": row.get("state_source", ""),
                        "due_at": row.get("due_at"),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            return _internal_failure(tool, exc)

        if rows:
            instruction = (
                "Higher `weakness` means weaker. This is at most the five "
                "weakest concepts, not necessarily every concept in the pack. "
                "Use it to answer what the learner is weakest at, or what to "
                "prioritise — then call review_today to practise (it serves "
                "the whole daily queue; there is no per-concept filter). "
                "`due_at` is the next scheduled review time, not a weakness "
                "signal: never treat an early due date as a weak concept. "
                "`evidence_count` counts review events recorded in this pack; "
                "when `state_source` is 'core', the scheduler may also hold "
                "core-side evidence, so zero does not mean never reviewed. Do "
                "not say a concept is mastered or forgotten from this ranking "
                "alone, and do not read raw scores aloud unless the learner "
                "asks for them."
            )
        else:
            instruction = (
                "This pack has no concepts yet, so there is nothing to rank. "
                "Offer review_today instead, and do not invent weak concepts."
            )
        return _result(
            {
                "spec_id": spec.id,
                "title": _clip(getattr(spec, "title", "")),
                "generated_at": report.get("generated_at"),
                "limit": WEAKNESS_LIMIT,
                "weak_concepts": rows,
                "instruction": instruction,
            },
            tool=tool,
        )


class ReviewPlanTool(BaseTool):
    """Plan a multi-week course and mirror it onto the DeepTutor workbench."""

    ACTIONS = (
        "create",
        "today",
        "complete",
        "replan",
        "confirm_note",
        "status",
        "sync_calendar",
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="review_plan",
            description=(
                "Create and run a multi-week study plan for one subject. "
                "`create` takes the knowledge base, the chapter list (exactly "
                "as review_generate lists them), an end date and a daily "
                "budget; the tool spreads the reading across days and refuses "
                "a schedule that does not fit instead of compressing it. Each "
                "planned day is mirrored onto the workbench calendar and task "
                "board (best-effort: the `sync` block says whether it worked). "
                "`complete` marks one study session done and drafts the "
                "chapter note once every session of that chapter is done; "
                "`confirm_note` writes that draft into the Co-Writer notebook. "
                "`today` returns today's tasks plus overdue sessions, "
                "`replan` rebuilds the remaining schedule from today, "
                "`sync_calendar` pulls whole-day moves made by dragging events "
                "on the calendar back into the plan, and "
                "`status` lists plans and progress."
            ),
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="One of: create, today, complete, replan, confirm_note, status, sync_calendar.",
                    enum=list(self.ACTIONS),
                ),
                ToolParameter(
                    name="subject",
                    type="string",
                    required=False,
                    description="Knowledge base name, e.g. 模拟电路 (create).",
                ),
                ToolParameter(
                    name="chapters",
                    type="array",
                    required=False,
                    items={"type": "string"},
                    description="Chapter names exactly as review_generate lists them (create).",
                ),
                ToolParameter(
                    name="start_date",
                    type="string",
                    required=False,
                    description="YYYY-MM-DD; defaults to today (create).",
                ),
                ToolParameter(
                    name="end_date",
                    type="string",
                    required=False,
                    description="YYYY-MM-DD the study must finish by (create/replan).",
                ),
                ToolParameter(
                    name="daily_minutes",
                    type="integer",
                    required=False,
                    minimum=15,
                    maximum=240,
                    description="Total daily budget including review; default 60 (create/replan).",
                ),
                ToolParameter(
                    name="weekly_budget",
                    type="string",
                    required=False,
                    description=(
                        "Seven comma-separated minutes, Monday..Sunday, for an "
                        "uneven week; 0 means a day off (replan). "
                        "Example: 90,120,90,120,90,120,180. The end date is "
                        "computed from it."
                    ),
                ),
                ToolParameter(
                    name="plan_id",
                    type="string",
                    required=False,
                    description="Plan id from create/status.",
                ),
                ToolParameter(
                    name="date",
                    type="string",
                    required=False,
                    description="YYYY-MM-DD for today/complete; defaults to today.",
                ),
                ToolParameter(
                    name="seq",
                    type="integer",
                    required=False,
                    description="Task seq inside that date, from today (complete).",
                ),
                ToolParameter(
                    name="self_note",
                    type="string",
                    required=False,
                    description="One sentence worth remembering; feeds the chapter note (complete).",
                ),
                ToolParameter(
                    name="note_id",
                    type="string",
                    required=False,
                    description="Draft note id returned by complete (confirm_note).",
                ),
            ],
        )

    async def _create(self, store: ReviewStore, kwargs: dict[str, Any]) -> ToolResult:
        tool = "review_plan"
        subject = str(kwargs.get("subject") or "").strip()
        raw_chapters = kwargs.get("chapters")
        if raw_chapters is not None and (
            not isinstance(raw_chapters, list)
            or any(not isinstance(name, str) or not name.strip() for name in raw_chapters)
        ):
            return _soft(
                tool=tool,
                error="chapters must be a list of non-empty chapter names",
                instruction="Call review_generate with only kb for the exact chapter names.",
            )
        chapters = [name.strip() for name in (raw_chapters or [])]
        end_date = str(kwargs.get("end_date") or "").strip()
        if not subject or not chapters or not end_date:
            return _soft(
                tool="review_plan",
                error="create needs subject, chapters and end_date",
                instruction=(
                    "Ask the learner for the subject, the chapter list and the "
                    "end date; call review_generate with only `kb` to list the "
                    "exact chapter names."
                ),
            )
        start_date, invalid = _plan_date(
            kwargs.get("start_date"),
            tool=tool,
            field="start_date",
            default=plan_service.local_today(plan_service.DEFAULT_TIMEZONE).isoformat(),
            instruction="Pass start_date as YYYY-MM-DD, or omit it to start today.",
        )
        if invalid is not None:
            return invalid
        end_date, invalid = _plan_date(
            end_date,
            tool=tool,
            field="end_date",
            instruction="Pass end_date as YYYY-MM-DD; it is the day the study must finish by.",
        )
        if invalid is not None:
            return invalid
        daily_minutes, invalid = _plan_int(
            kwargs.get("daily_minutes"),
            tool=tool,
            field="daily_minutes",
            minimum=15,
            maximum=240,
            default=60,
            instruction=(
                "Pass daily_minutes as an integer between 15 and 240 (total "
                "daily budget, review included), or omit it for 60."
            ),
        )
        if invalid is not None:
            return invalid
        payload = await asyncio.to_thread(
            plan_service.create_plan,
            store,
            subject=subject,
            chapters=chapters,
            start_date=start_date,
            end_date=end_date,
            daily_minutes=daily_minutes,
        )
        if payload.get("ok"):
            days = len({task["local_date"] for task in payload["tasks"]})
            sync_failures = [
                day
                for day, status in (payload.get("sync") or {}).items()
                if not status["calendar"]["ok"] or not status["card"]["ok"]
            ]
            payload["instruction"] = (
                f"Plan created: {len(chapters)} chapters, {days} study days, "
                f"{payload['plan']['study_minutes_per_day']} minutes of new "
                "material per day. "
                + (
                    "Chapters without a review pack yet: "
                    + ", ".join(
                        row["chapter"]
                        for row in payload.get("coverage") or []
                        if not row["has_pack"]
                    )
                    + " — they have no knowledge points or review items until "
                    "review_generate creates one; say so instead of promising "
                    "review for them. "
                    if any(
                        not row["has_pack"] for row in payload.get("coverage") or []
                    )
                    else "Every chapter already has a review pack. "
                )
                + "Tell the learner the end date and the daily amount, and "
                "that they can change the end date with action=replan. "
                + (
                    "The workbench mirror failed for: "
                    + ", ".join(sorted(sync_failures)[:5])
                    + "; the plan still works, only the calendar/task board "
                    "are missing those days."
                    if sync_failures
                    else "Calendar and task board accepted every day."
                )
            )
        else:
            payload["instruction"] = (
                "Do not create the plan. Tell the learner it does not fit and "
                "offer exactly two options: extend the end date to "
                f"{payload.get('suggested_end_date')}, or cut chapters."
            )
        return _result(payload, tool="review_plan")

    async def execute(self, **kwargs: Any) -> ToolResult:
        tool = "review_plan"
        blocked = capabilities.disabled(tool)
        if blocked:
            return _soft(tool=tool, error=f"{tool} is disabled", instruction=blocked)
        action = str(kwargs.get("action") or "").strip().lower()
        if action not in self.ACTIONS:
            return _soft(
                tool=tool,
                error=f"unknown action: {action or '(missing)'}",
                instruction="Call review_plan with action=create|today|complete|replan|confirm_note|status|sync_calendar.",
            )
        try:
            store = ReviewStore()
            if action == "create":
                return await self._create(store, kwargs)

            if action == "today":
                on_date, invalid = _plan_date(
                    kwargs.get("date"),
                    tool=tool,
                    field="date",
                    optional=True,
                    instruction="Pass date as YYYY-MM-DD, or omit it for today.",
                )
                if invalid is not None:
                    return invalid
                payload = await asyncio.to_thread(
                    plan_service.plan_today,
                    store,
                    plan_id=kwargs.get("plan_id"),
                    on_date=on_date,
                    refresh_mirror=True,
                )
                payload["instruction"] = (
                    "These are the planned study tasks for the day. Mark one "
                    "done with action=complete and its date/seq. The `review_due` "
                    "block in this payload counts the due items across this "
                    "subject's packs (read-only; the task board card is refreshed "
                    "with the same count) — "
                    "tell the learner the number, then use review_today to "
                    "practise them. If `overdue` is non-empty, mention it and "
                    "offer replan."
                )
                return _result(payload, tool=tool)

            if action == "complete":
                plan_id = str(kwargs.get("plan_id") or "").strip()
                seq = kwargs.get("seq")
                if not plan_id or seq is None:
                    return _soft(
                        tool=tool,
                        error="complete needs plan_id and seq",
                        instruction="Call action=today first and pass the plan_id/date/seq of the task.",
                    )
                checked_seq, invalid = _plan_int(
                    seq,
                    tool=tool,
                    field="seq",
                    minimum=1,
                    maximum=10_000,
                    instruction=(
                        "Pass the integer seq from action=today (for example 1), "
                        "not a string or a whole task object."
                    ),
                )
                if invalid is not None:
                    return invalid
                assert checked_seq is not None  # required above; kept for mypy
                timezone = (store.get_plan(plan_id) or {}).get("timezone") or (
                    plan_service.DEFAULT_TIMEZONE
                )
                local_date, invalid = _plan_date(
                    kwargs.get("date"),
                    tool=tool,
                    field="date",
                    default=plan_service.local_today(timezone).isoformat(),
                    instruction="Pass date as YYYY-MM-DD, or omit it for today.",
                )
                if invalid is not None:
                    return invalid
                payload = await asyncio.to_thread(
                    plan_service.complete_task,
                    store,
                    plan_id=plan_id,
                    local_date=local_date,
                    seq=checked_seq,
                    self_note=str(kwargs.get("self_note") or ""),
                )
                payload["instruction"] = (
                    "Marked done. "
                    + (
                        "A chapter note draft is ready; summarise it briefly and "
                        "ask before calling action=confirm_note."
                        if payload.get("note")
                        else "Continue with the next task, or finish for today."
                    )
                )
                return _result(payload, tool=tool)

            if action == "replan":
                plan_id = str(kwargs.get("plan_id") or "").strip()
                if not plan_id:
                    return _soft(
                        tool=tool,
                        error="replan needs plan_id",
                        instruction="Call action=status to list plans first.",
                    )
                weekly_text = str(kwargs.get("weekly_budget") or "").strip()
                weekly_budget = None
                if weekly_text:
                    try:
                        minutes = [int(part) for part in weekly_text.split(",")]
                    except ValueError:
                        minutes = []
                    if len(minutes) != 7 or any(value < 0 for value in minutes):
                        return _soft(
                            tool=tool,
                            error=(
                                "weekly_budget needs seven comma-separated minutes "
                                "(Monday..Sunday)"
                            ),
                            instruction="Example: 90,120,90,120,90,120,180; 0 = day off.",
                        )
                    weekly_budget = {
                        day: value for day, value in enumerate(minutes)
                    }
                end_date, invalid = _plan_date(
                    kwargs.get("end_date"),
                    tool=tool,
                    field="end_date",
                    optional=True,
                    instruction="Pass end_date as YYYY-MM-DD, or omit it to keep the plan's end date.",
                )
                if invalid is not None:
                    return invalid
                daily_minutes, invalid = _plan_int(
                    kwargs.get("daily_minutes"),
                    tool=tool,
                    field="daily_minutes",
                    minimum=15,
                    maximum=240,
                    optional=True,
                    instruction=(
                        "Pass daily_minutes as an integer between 15 and 240 "
                        "(total daily budget, review included), or omit it."
                    ),
                )
                if invalid is not None:
                    return invalid
                payload = await asyncio.to_thread(
                    plan_service.replan,
                    store,
                    plan_id=plan_id,
                    end_date=end_date,
                    daily_minutes=daily_minutes,
                    weekly_budget=weekly_budget,
                )
                payload["instruction"] = (
                    "The remaining schedule was rebuilt from today. Tell the "
                    "learner the new end date and daily amount."
                    if payload.get("ok")
                    else "Replanning failed: relay the reason and the suggested "
                    "end date, and do not claim the plan changed."
                )
                return _result(payload, tool=tool)

            if action == "confirm_note":
                note_id = str(kwargs.get("note_id") or "").strip()
                if not note_id:
                    return _soft(
                        tool=tool,
                        error="confirm_note needs note_id",
                        instruction="Use the note id returned by action=complete.",
                    )
                payload = await asyncio.to_thread(
                    plan_service.confirm_note, store, note_id=note_id
                )
                payload["instruction"] = (
                    "The note is now in the Co-Writer notebook under the "
                    "subject page."
                    if payload.get("ok")
                    else "The notebook is unavailable; the note stays a draft. "
                    "Tell the learner it is saved locally and can be confirmed later."
                )
                return _result(payload, tool=tool)

            if action == "sync_calendar":
                plan_id = str(kwargs.get("plan_id") or "").strip()
                if not plan_id:
                    return _soft(
                        tool=tool,
                        error="sync_calendar needs plan_id",
                        instruction="Call action=status to list plans first.",
                    )
                payload = await asyncio.to_thread(
                    plan_service.sync_from_calendar, store, plan_id=plan_id
                )
                payload["instruction"] = (
                    "Calendar moves were read back into the plan; report how "
                    "many days moved, any day that was skipped because the "
                    "target date already had tasks, and any card that did not "
                    "move. If ok is false, the calendar could not be read and "
                    "the plan is unchanged — tell the learner to retry later."
                )
                return _result(payload, tool=tool)

            payload = await asyncio.to_thread(
                plan_service.plan_status, store, plan_id=kwargs.get("plan_id")
            )
            payload["instruction"] = (
                "Read-only summary. Use action=today for the actual tasks."
            )
            return _result(payload, tool=tool)
        except plan_service.PlanError as exc:
            return _soft(
                tool=tool,
                error=str(exc),
                instruction=(
                    "The request cannot be met as stated. Relay the reason and "
                    "ask for a corrected date, chapter list or plan id."
                ),
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary
            return _internal_failure(tool, exc)


class ReviewExportLinkTool(BaseTool):
    """Hand the learner the export page URL for one pack."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="review_export_link",
            description=(
                "Return the URL of the printable review page for one Review "
                "Studio pack. The page itself renders self-contained, including "
                "figures, so once open it works offline — but the URL is served "
                "by the DeepTutor web app, so this tool needs that app running "
                "and the learner signed in; it does not work in a standalone "
                "deployment. Use review_export_tree when the learner needs an "
                "offline copy with no web app at all. This call changes nothing "
                "and starts no session."
            ),
            parameters=[
                ToolParameter(
                    name="spec_id",
                    type="string",
                    description="Pack id from review_specs (verbatim).",
                )
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        tool = "review_export_link"
        _, spec, error = await _service_and_spec(
            tool=tool,
            spec_id=kwargs.get("spec_id"),
            missing_instruction=(
                "That pack does not exist. Call review_specs for the ids this "
                "learner has, then call review_export_link again."
            ),
        )
        if error is not None or spec is None:
            return error or _failure("review_export_link could not resolve the pack.")

        quality = run_gates(spec)
        if not quality.passed:
            return _soft(
                tool=tool,
                error="quality_gate_failed",
                instruction=(
                    "This pack failed quality gates and cannot be exported. "
                    "Explain the failed gates and ask the learner to regenerate "
                    "or correct the pack before requesting another export."
                ),
                spec_id=spec.id,
                quality=quality.model_dump(mode="json"),
            )
        url = EXPORT_URL_TEMPLATE.format(spec_id=spec.id)
        payload: dict[str, Any] = {
            "spec_id": spec.id,
            "title": _clip(getattr(spec, "title", "")),
            "url": url,
        }
        base = _public_base_url()
        if base:
            # An external MCP client has no browser origin to resolve the
            # relative path against; the deployment's configured origin is the
            # only safe absolute link. Keep ``url`` for existing consumers.
            payload["absolute_url"] = f"{base}{url}"
        payload["instruction"] = (
            "Give the learner this link (or the export card the UI renders for "
            "it): opening it in a browser downloads a self-contained review "
            "page for this pack. The link is served by the DeepTutor web app, "
            "so if that app is not running for this learner, say so and offer "
            "review_export_tree instead of handing over a dead link. Use "
            "`absolute_url` when present; otherwise `url` is relative to the "
            "host app. Do not paste the page's contents into the chat, and do "
            "not claim you opened it yourself — you cannot."
        )
        return _result(payload, tool=tool)


class ReviewExportTreeTool(BaseTool):
    """Write every pack into one subject-classified folder tree on disk."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="review_export_tree",
            description=(
                "Write the learner's whole pack collection to one classified "
                "folder tree on disk: a folder per subject, one self-contained "
                "HTML page per pack, and an index.html that links them all, so "
                "someone opening the folder can tell subjects apart without "
                "guessing. The tree is rebuilt from the store on every call, so "
                "renamed chapters leave no stale page and no dated snapshot "
                "folders accumulate; packs failing the quality gates are "
                "skipped and reported. Use it when the learner wants everything "
                "on disk (backup, print run, another machine) rather than one "
                "link. It changes no review state and starts no session."
            ),
            parameters=[
                ToolParameter(
                    name="out_dir",
                    type="string",
                    description=(
                        "Dedicated destination folder, created if missing. Omit to use "
                        "<review home>/exports. Existing personal or legacy HTML with "
                        "a conflicting name is preserved."
                    ),
                    required=False,
                ),
                ToolParameter(
                    name="subjects",
                    type="array",
                    description=(
                        "Only export packs whose title prefix (the part before "
                        "' · ') is one of these subject names. Omit for every "
                        "subject."
                    ),
                    required=False,
                    items={"type": "string"},
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        tool = "review_export_tree"
        blocked = capabilities.disabled(tool)
        if blocked:
            return _soft(tool=tool, error=f"{tool} is disabled", instruction=blocked)
        try:
            service = await asyncio.to_thread(_service)
            requested = _text(kwargs.get("out_dir"))
            out_dir, out_dir_error = _resolve_export_dir(requested)
            if out_dir_error:
                return _soft(
                    tool=tool,
                    error="out_dir is outside the allowed roots",
                    instruction=out_dir_error,
                )
            raw_subjects = kwargs.get("subjects") or ()
            if isinstance(raw_subjects, str):
                raw_subjects = [raw_subjects]
            subjects = [str(name).strip() for name in raw_subjects if str(name).strip()] or None
            summary = await asyncio.to_thread(
                export_tree.export_tree, service, out_dir, subjects=subjects
            )
        except export_tree.ExportConflict as exc:
            return _soft(
                tool=tool,
                error=str(exc),
                instruction="Keep the existing file and choose a new export directory.",
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary
            return _internal_failure(tool, exc)

        payload: dict[str, Any] = dict(summary)
        payload["instruction"] = (
            f"Tell the learner the folder is written: {payload['out_dir']} "
            f"({payload['packs']} packs, {payload['questions']} questions). "
            "Opening index.html lists every subject and chapter; each pack page "
            "opens on its own, offline. These pages are the spaced-repetition "
            "packs for that chapter, not the living book's in-section quizzes. "
            + (
                "These packs were skipped because they failed the quality "
                "gates; say so and offer regeneration instead of claiming they "
                "were written: "
                + "; ".join(f"{item['title']} ({item['reason']})" for item in payload["skipped"])
                + ". "
                if payload["skipped"]
                else ""
            )
            + "Do not paste pack contents into the chat."
        )
        return _result(payload, tool=tool)


# -- generation helpers -----------------------------------------------------


def _generate_module() -> Any:
    """Import the knowledge-base generator on demand.

    ``deeptutor.review.generate`` is a separate work stream (Phase 31 WS-1).
    Importing inside the call keeps every other review tool usable in a
    checkout that predates it, and lets tests substitute the module.
    """
    import importlib

    return importlib.import_module("deeptutor.review.generate")


def _elapsed_seconds(job: dict[str, Any]) -> int:
    """Whole seconds since the job was created, tolerating a corrupt record.

    Job files sit in a user-writable data directory; a hand-edited or
    truncated ``created_at`` (string, list, NaN, infinity) must degrade to
    zero rather than crash the poll. A clock-skewed future stamp likewise
    reads as zero, never a negative age.
    """
    try:
        created = float(job.get("created_at"))
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(created):
        return 0
    return max(0, round(time.time() - created))


def _jobs_module() -> Any:
    """Import the background job store on demand.

    Same seam as ``_generate_module``: importing inside the call lets tests
    substitute the module and keeps every other review tool usable even if
    the job store is absent from an older checkout.
    """
    import importlib

    return importlib.import_module("deeptutor.review.jobs")


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    """Validate a model-supplied count against the advertised range.

    Only a real integer counts: ``True`` is not 1, ``"6"`` is not 6, and
    ``6.0`` is not the integer the tool schema declared. Anything else falls
    back to the documented default instead of being silently coerced.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(minimum, min(maximum, value))


def _plan_int(
    value: Any,
    *,
    tool: str,
    field: str,
    minimum: int,
    maximum: int,
    default: int | None = None,
    optional: bool = False,
    instruction: str,
) -> tuple[int | None, ToolResult | None]:
    """A plan argument that must be a real integer in range.

    ``review_plan`` declares twelve optional parameters with one ``enum``
    action, so the schema cannot enforce per-action types. A model that sends
    ``"60"`` or ``"abc"`` used to reach ``int(...)`` and surface as an internal
    failure — an unrecoverable dead end for the agent. Returning a soft failure
    keeps the single next move (fix the argument and retry) available.

    ``optional=True`` means "absent is fine" (``replan`` may keep the plan's
    current value); it is distinct from "absent falls back to ``default``"
    (``create`` starts at 60).
    """
    if value is None:
        if optional:
            return None, None
        if default is None:
            return None, _soft(tool=tool, error=f"{field} is required", instruction=instruction)
        return default, None
    if isinstance(value, bool) or not isinstance(value, int):
        return None, _soft(
            tool=tool,
            error=f"{field} must be an integer",
            instruction=instruction,
        )
    if not minimum <= value <= maximum:
        return None, _soft(
            tool=tool,
            error=f"{field} must be between {minimum} and {maximum}",
            instruction=instruction,
        )
    return value, None


def _plan_date(
    value: Any,
    *,
    tool: str,
    field: str,
    default: str | None = None,
    optional: bool = False,
    instruction: str,
) -> tuple[str | None, ToolResult | None]:
    """A ``YYYY-MM-DD`` argument, validated before it reaches ``fromisoformat``.

    ``date.fromisoformat`` raises ``ValueError``, which the plan service does
    not always wrap (``replan`` parses the end date directly), so an
    ``2026-13-40`` used to become an internal failure rather than a correctable
    miss.

    ``optional=True`` means "absent means keep what the plan already has".
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        if optional:
            return None, None
        if default is None:
            return None, _soft(tool=tool, error=f"{field} is required", instruction=instruction)
        return default, None
    try:
        return plan_service.parse_plan_date(value, field=field).isoformat(), None
    except plan_service.PlanError:
        return None, _soft(
            tool=tool,
            error=f"{field} must be a date in YYYY-MM-DD form",
            instruction=instruction,
        )


def _question_types(value: Any) -> list[str]:
    """Keep the known kinds in the caller's order, falling back to both."""
    if value is None:
        return list(DEFAULT_QUESTION_TYPES)
    entries = [value] if isinstance(value, str) else value
    if not isinstance(entries, (list, tuple, set)):
        return list(DEFAULT_QUESTION_TYPES)
    picked: list[str] = []
    for entry in entries:
        name = _text(entry).lower()
        if name in ALLOWED_QUESTION_TYPES and name not in picked:
            picked.append(name)
    return picked or list(DEFAULT_QUESTION_TYPES)


class ReviewGenerateTool(BaseTool):
    """Turn one knowledge-base chapter into a review pack."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="review_generate",
            description=(
                "Generate a new Review Studio pack from one chapter of a "
                "knowledge base, so the learner can review material that has "
                "no pack yet. Three call modes: with kb only it lists the "
                "chapters; with kb and chapter it starts a background "
                "generation job and returns a job_id immediately; with kb and "
                "job_id it reports that job's progress. Generation runs in "
                "the background and takes minutes, so do it only when the "
                "learner asks for a new pack, never speculatively. The pack "
                "is checked by the quality gates before it is stored; a pack "
                "that fails the gates is not saved, and you must not tell "
                "the learner it is ready until a poll reports done."
            ),
            parameters=[
                ToolParameter(
                    name="kb",
                    type="string",
                    description=(
                        "Knowledge base name, as the learner names it. Required."
                    ),
                ),
                ToolParameter(
                    name="chapter",
                    type="string",
                    description=(
                        "Chapter name exactly as returned by the chapter-listing "
                        "call. Omit it (call with kb only) to list the chapters."
                    ),
                    required=False,
                ),
                ToolParameter(
                    name="num_questions",
                    type="integer",
                    description=(
                        "How many questions to generate. Must be between "
                        f"{MIN_NUM_QUESTIONS} and {MAX_NUM_QUESTIONS}; values "
                        "outside are rejected by the schema."
                    ),
                    required=False,
                    default=DEFAULT_NUM_QUESTIONS,
                    minimum=MIN_NUM_QUESTIONS,
                    maximum=MAX_NUM_QUESTIONS,
                ),
                ToolParameter(
                    name="question_types",
                    type="array",
                    description=(
                        "Question kinds to generate: choice and/or written. "
                        "Defaults to both."
                    ),
                    required=False,
                    default=list(DEFAULT_QUESTION_TYPES),
                    items={
                        "type": "string",
                        "enum": list(DEFAULT_QUESTION_TYPES),
                    },
                ),
                ToolParameter(
                    name="difficulty",
                    type="string",
                    description=(
                        "Optional difficulty hint passed to the generator "
                        "(for example easy, medium or hard)."
                    ),
                    required=False,
                ),
                ToolParameter(
                    name="job_id",
                    type="string",
                    description=(
                        "Job id returned when a generation started. Poll with "
                        "it (together with kb) until the status is done or "
                        "failed. Do not pass chapter at the same time."
                    ),
                    required=False,
                ),
                ToolParameter(
                    name="replace",
                    type="boolean",
                    description=(
                        "Default false. Regenerating a chapter resolves to the "
                        "same deterministic pack id, so the run is rejected if "
                        "that pack already exists. Pass true only when the "
                        "learner explicitly asks to replace the existing pack."
                    ),
                    required=False,
                    default=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        tool = "review_generate"
        blocked = capabilities.disabled(tool)
        if blocked:
            return _soft(tool=tool, error=f"{tool} is disabled", instruction=blocked)
        kb_name = _text(kwargs.get("kb"))
        if not kb_name:
            return _soft(
                tool=tool,
                error="kb is required",
                instruction=(
                    "Ask the learner which knowledge base to generate from, "
                    "then call review_generate again with that name."
                ),
            )
        job_id = _text(kwargs.get("job_id"))
        chapter = _text(kwargs.get("chapter"))
        if job_id:
            if chapter:
                return _soft(
                    tool=tool,
                    error=(
                        "pass either chapter (to start a generation) or "
                        "job_id (to poll one), not both"
                    ),
                    instruction=(
                        "Keep the job_id from the start call and poll with "
                        "only kb and job_id. To start a new generation, pass "
                        "kb and chapter without job_id."
                    ),
                    job_id=job_id,
                )
            return await self._poll_job(tool, kb_name, job_id)

        try:
            generator = _generate_module()
        except Exception as exc:  # pragma: no cover - only on a broken install
            logger.warning(
                "review_generate cannot import the generator: %r", exc, exc_info=True
            )
            return _failure(
                "review_generate is unavailable (the knowledge-base generator "
                "module could not be imported). Tell the learner generation is "
                "not available right now instead of retrying."
            )

        if not chapter:
            return await self._list_chapters(generator, kb_name)
        return await self._start(
            generator,
            kb_name=kb_name,
            chapter=chapter,
            num_questions=_bounded_int(
                kwargs.get("num_questions"),
                default=DEFAULT_NUM_QUESTIONS,
                minimum=MIN_NUM_QUESTIONS,
                maximum=MAX_NUM_QUESTIONS,
            ),
            question_types=_question_types(kwargs.get("question_types")),
            difficulty=_text(kwargs.get("difficulty")),
            # Only the literal boolean opts in: "true", 1 and other truthy
            # transport values must not replace an existing pack.
            replace=kwargs.get("replace") is True,
        )

    async def _list_chapters(self, generator: Any, kb_name: str) -> ToolResult:
        tool = "review_generate"
        try:
            chapters = await asyncio.to_thread(generator.infer_chapters, kb_name)
        except (ValueError, KeyError, OSError) as exc:
            # An unknown or malformed knowledge base name is a recoverable
            # miss, not a fault: the model can check the name and call again.
            logger.info("review_generate could not read %r: %r", kb_name, exc)
            return _soft(
                tool=tool,
                error=(
                    f"could not read knowledge base {kb_name!r} "
                    f"({type(exc).__name__})"
                ),
                instruction=(
                    "Check the knowledge base name with the knowledge-base "
                    "tools, then call review_generate again with the exact "
                    "name. If the learner does not have this knowledge base, "
                    "say so instead of retrying."
                ),
                kb=kb_name,
                mode="chapters",
            )
        except Exception as exc:
            return _internal_failure(tool, exc)

        rows = [
            {
                "name": _text(getattr(item, "name", "")),
                "doc_count": int(getattr(item, "doc_count", 0) or 0),
            }
            for item in (chapters or [])
        ]
        if not rows:
            return _soft(
                tool=tool,
                error=f"no chapters found in knowledge base {kb_name!r}",
                instruction=(
                    "Tell the learner this knowledge base has no "
                    "chapter-structured documents. If they already have packs, "
                    "call review_specs instead."
                ),
                kb=kb_name,
                mode="chapters",
            )
        return _result(
            {
                "kb": kb_name,
                "mode": "chapters",
                "chapters": rows,
                "instruction": (
                    "Ask the learner which chapter to turn into a review pack, "
                    f"then call review_generate again with kb={kb_name!r} and "
                    "that chapter name verbatim. Tell the learner generation "
                    "runs in the background and takes minutes."
                ),
            },
            tool=tool,
        )

    async def _start(
        self,
        generator: Any,
        *,
        kb_name: str,
        chapter: str,
        num_questions: int,
        question_types: list[str],
        difficulty: str,
        replace: bool = False,
    ) -> ToolResult:
        tool = "review_generate"
        # Preflight the chapter before any job exists: an unknown knowledge
        # base or chapter has to stay a recoverable miss in this turn, not a
        # background job that fails minutes later.
        try:
            chapters = await asyncio.to_thread(generator.infer_chapters, kb_name)
        except FileNotFoundError as exc:
            logger.info("review_generate could not read %r: %r", kb_name, exc)
            return _soft(
                tool=tool,
                error=f"knowledge base not found: {kb_name!r}",
                instruction=(
                    "Check the knowledge base name with the knowledge-base "
                    "tools, then call review_generate again with the exact "
                    "name. If the learner does not have this knowledge base, "
                    "say so instead of retrying."
                ),
                kb=kb_name,
                chapter=chapter,
            )
        except (ValueError, KeyError, OSError) as exc:
            logger.info("review_generate could not read %r: %r", kb_name, exc)
            return _soft(
                tool=tool,
                error=f"could not read knowledge base {kb_name!r} ({exc})",
                instruction=(
                    "Call review_generate with only kb to check the chapter "
                    "list. If the knowledge base cannot be read, tell the "
                    "learner instead of retrying."
                ),
                kb=kb_name,
                chapter=chapter,
            )
        except Exception as exc:
            return _internal_failure(tool, exc)

        names = [_text(getattr(item, "name", "")) for item in (chapters or [])]
        if chapter not in names:
            return _soft(
                tool=tool,
                error=(
                    f"no chapter matching {chapter!r} in knowledge base "
                    f"{kb_name!r}"
                ),
                instruction=(
                    "Call review_generate with only kb to list the chapters, "
                    "then pass one of those names verbatim. Do not retry the "
                    "same chapter name."
                ),
                kb=kb_name,
                chapter=chapter,
                chapters=names,
            )

        try:
            job = _jobs_module().start_generation_job(
                kb=kb_name,
                chapter=chapter,
                num_questions=num_questions,
                question_types=question_types,
                difficulty=difficulty,
                replace=replace,
            )
        except Exception as exc:
            return _internal_failure(tool, exc)

        job_id = str(job.get("job_id") or "")
        reused = bool(job.get("reused"))
        if reused:
            instruction = (
                f"A generation for this chapter is already running (job "
                f"{job_id}); this call reused it instead of starting a "
                "duplicate. Do not start another generation. Poll every ~20 "
                "seconds by calling review_generate again with kb and that "
                "job_id."
            )
        else:
            instruction = (
                f"Generation started in the background (job {job_id}). "
                "Tell the learner it takes minutes. Poll every ~20 "
                "seconds by calling review_generate again with kb and "
                "that job_id; do not start another generation for the "
                "same chapter while this one runs."
            )
        return _result(
            {
                "kb": kb_name,
                "chapter": chapter,
                "job_id": job_id,
                "status": str(job.get("status") or "queued"),
                "reused": reused,
                "instruction": instruction,
            },
            tool=tool,
        )

    async def _poll_job(self, tool: str, kb_name: str, job_id: str) -> ToolResult:
        try:
            jobs = _jobs_module()
            job = await asyncio.to_thread(jobs.load_job, job_id)
        except Exception as exc:
            return _internal_failure(tool, exc)
        if job is None:
            return _soft(
                tool=tool,
                error=f"unknown generation job: {job_id!r}",
                instruction=(
                    "Use the job_id exactly as returned when the generation "
                    "started. If it is lost, call review_specs to see whether "
                    "a pack appeared, or start a new generation with kb and "
                    "chapter."
                ),
                job_id=job_id,
            )
        job_kb = str(job.get("kb") or "")
        if job_kb and kb_name and job_kb != kb_name:
            return _soft(
                tool=tool,
                error=(
                    f"this job belongs to knowledge base {job_kb!r}, not "
                    f"{kb_name!r}"
                ),
                instruction=(
                    f"Poll again with kb={job_kb!r} and the same job_id. Do "
                    "not start a new generation for the other knowledge base."
                ),
                job_id=job_id,
            )

        status = str(job.get("status") or "unknown")
        payload: dict[str, Any] = {
            "job_id": job_id,
            "kb": job.get("kb"),
            "chapter": job.get("chapter"),
            "status": status,
            "spec_id": job.get("spec_id"),
            "replace": job.get("replace", False),
            "qa_count": job.get("qa_count"),
            "skipped": job.get("skipped"),
            "error": job.get("error"),
            "elapsed_seconds": _elapsed_seconds(job),
            "warnings": [warning for warning in (job.get("warnings") or []) if isinstance(warning, str)] if isinstance(job.get("warnings"), list) else [],
        }
        if status in ("queued", "running"):
            if jobs.is_stale(job):
                payload["stale"] = True
                payload["instruction"] = (
                    "The job has not updated for over an hour. The service "
                    "may have restarted and interrupted it, though a very "
                    "large batch can still be running. Tell the learner, "
                    "check review_specs in case the pack was stored anyway, "
                    "and only then offer to restart the generation."
                )
            else:
                payload["instruction"] = (
                    "Still generating. Poll again in ~20 seconds with the "
                    "same kb and job_id. Do not start a second generation "
                    "for this chapter while this job runs."
                )
            return _result(payload, tool=tool)
        if status == "done":
            spec_id = job.get("spec_id")
            count = job.get("qa_count")
            counts = job.get("counts")
            warnings = job.get("warnings", [])
            if (
                not isinstance(spec_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", spec_id)
                or spec_id.endswith(".")
                or job.get("qa_passed") is not True
                or type(count) is not int or count <= 0
                or not isinstance(counts, dict) or type(counts.get("items")) is not int
                or counts["items"] != count
                or not isinstance(warnings, list) or any(not isinstance(warning, str) for warning in warnings)
                or job.get("error")
            ):
                # Jobs written before the strict schema existed may carry only
                # ``status``/``spec_id``. The stored pack is the ground truth,
                # so verify the id against it instead of discarding a real
                # result; a pack that is truly absent keeps the refusal.
                recovered = await self._recover_done_job(tool, job, spec_id)
                if recovered is not None:
                    return recovered
                return _soft(
                    tool=tool, error="invalid_job_record", job_id=job_id, status="invalid",
                    instruction=(
                        "The completed job record is inconsistent; success cannot be confirmed. "
                        "Call review_specs to check whether a valid pack exists. "
                        "Do not claim it was saved or passed quality gates from this record."
                    ),
                )
            spec_id = str(job.get("spec_id") or "")
            payload["qa_passed"] = job.get("qa_passed")
            payload["counts"] = job.get("counts")
            payload["warnings"] = list(job.get("warnings") or [])
            payload["instruction"] = (
                f"The pack {spec_id} is saved and passed the quality gates. "
                "Call review_today with that spec_id to start the first "
                "session, or review_specs to show it with the learner's "
                "other packs. Tell the learner how many items it holds."
            )
            return _result(payload, tool=tool)
        if status == "failed":
            payload["instruction"] = (
                "The generation failed and no pack was stored. Report the "
                "error to the learner and do not retry the same call "
                "unchanged. When the error lists gate details, use the field "
                "path in each detail (for example items.[2].stem) to locate "
                "the offending content; a ____ placeholder inside math must "
                "be written as \\underline{\\hspace{2em}}. If the error says "
                "a pack already exists, the existing pack was left untouched: "
                "ask whether to keep it or replace it, and pass replace=true "
                "only on explicit confirmation; otherwise retry at most once "
                "with fewer questions or choice-only question types."
            )
        else:
            payload["instruction"] = (
                "This job record carries an unrecognized status. Poll again; "
                "if it persists, call review_specs to check whether a pack "
                "was stored, then start a new generation only if needed."
            )
        return _result(payload, tool=tool)

    async def _recover_done_job(
        self, tool: str, job: dict[str, Any], spec_id: Any
    ) -> ToolResult | None:
        """Confirm a ``done`` job against the pack it claims to have stored.

        Records written before the strict done-schema (or truncated ones that
        lost ``qa_count``/``counts``/``qa_passed``) still name a real pack.
        The pack itself is trustworthy evidence — ``spec_meta`` re-runs the
        current gates and reports its live counts — so a record whose id
        resolves to a stored pack is completed from that pack. ``None`` means
        there is no such pack and the caller must keep refusing.
        """
        if job.get("error"):
            return None
        if (
            not isinstance(spec_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", spec_id)
            or spec_id.endswith(".")
        ):
            return None
        try:
            service = await asyncio.to_thread(_service)
            meta = await asyncio.to_thread(service.spec_meta, spec_id)
        except SpecNotFound:
            return None
        except Exception as exc:  # noqa: BLE001 - store outage is an internal failure
            return _internal_failure(tool, exc)
        counts = meta.get("counts")
        items = counts.get("items") if isinstance(counts, dict) else None
        if type(items) is not int or items <= 0:
            return None
        quality = meta.get("quality") if isinstance(meta.get("quality"), dict) else {}
        passed = quality.get("passed") is True
        warnings = [
            warning for warning in (quality.get("warnings") or []) if isinstance(warning, str)
        ]
        payload: dict[str, Any] = {
            "job_id": job.get("job_id"),
            "kb": job.get("kb"),
            "chapter": job.get("chapter"),
            "status": "done",
            "spec_id": spec_id,
            "replace": job.get("replace", False),
            "qa_count": items,
            "skipped": job.get("skipped"),
            "error": None,
            "elapsed_seconds": _elapsed_seconds(job),
            "warnings": warnings,
            "qa_passed": passed,
            "counts": {
                key: counts.get(key) if type(counts.get(key)) is int else 0
                for key in ("concepts", "sections", "items")
            },
            "legacy_record": True,
        }
        if passed:
            payload["instruction"] = (
                f"The pack {spec_id} is stored and currently passes the "
                "quality gates. The job record predates the current schema, "
                "so these counts were read from the stored pack, not the "
                "record. Call review_today with that spec_id to start the "
                "first session, or review_specs to show it with the "
                "learner's other packs."
            )
        else:
            payload["instruction"] = (
                f"The pack {spec_id} is stored but does not pass the current "
                "quality gates. Do not tell the learner it passed. Call "
                "review_specs so they can see it and its warning count; offer "
                "to regenerate it with replace=true only if they explicitly "
                "ask to replace it."
            )
        return _result(payload, tool=tool)


FIGURE_INSTRUCTION = (
    "Figures are inline SVG drawn inside the built book; this index only says "
    "which section owns which one. When you send the learner to a section, name "
    "the figure the way the book labels it (图 2-1) and say in one clause what it "
    "shows; a section absent from this index has no diagram, so never promise "
    "one. Never paste SVG or base64 into the chat. Review pack pages are a "
    "different surface and carry no book figures."
)


def _figure_sort_key(number: str) -> tuple[int, int, str]:
    """Sort 2.10 after 2.9; appendix letters go last."""
    match = re.match(r"(\d+)(?:\.(\d+))?", str(number))
    if not match:
        return (10**6, 0, str(number))
    return (int(match.group(1)), int(match.group(2) or 0), "")


class ReviewBookFiguresTool(BaseTool):
    """Tell the agent which sections of the living books carry a diagram."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="review_book_figures",
            description=(
                "List the diagrams the living books print, section by section, "
                "so a reading pointer can name the figure instead of sending the "
                "learner into a wall of text. Each entry carries the section "
                "number, its title, the book anchor and the caption. Read-only: "
                "it changes no review state and starts no session."
            ),
            parameters=[
                ToolParameter(
                    name="subject",
                    type="string",
                    description=(
                        "Only report this subject (the name the book file uses). "
                        "Omit for every indexed book."
                    ),
                    required=False,
                ),
                ToolParameter(
                    name="chapter",
                    type="string",
                    description=(
                        "Only report the sections a planned chapter owns, e.g. "
                        "第2讲 or 第1章_离散时间信号的由来."
                    ),
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        tool = "review_book_figures"
        try:
            index = await asyncio.to_thread(materials.load_book_figures)
        except Exception as exc:  # noqa: BLE001 - tool boundary
            return _internal_failure(tool, exc)

        subject = _text(kwargs.get("subject"))
        chapter = _text(kwargs.get("chapter"))
        if not index:
            return _result(
                {
                    "error": "no figure index",
                    "instruction": (
                        "The books are not indexed, so nothing can be said about "
                        "their figures. An operator runs "
                        "scripts/build-book-figures.py against the built books to "
                        "write book-figures.json into the review home. Do not "
                        "guess which sections have a diagram."
                    ),
                },
                tool=tool,
            )

        raw_books = index.get("books")
        books = raw_books if isinstance(raw_books, dict) else {}
        raw_chapters = index.get("chapters")
        chapters = raw_chapters if isinstance(raw_chapters, dict) else {}

        if chapter:
            rows = chapters.get(chapter)
            if not isinstance(rows, dict):
                return _result(
                    {
                        "error": f"chapter {chapter} has no figure map",
                        "chapters": sorted(str(key) for key in chapters),
                        "instruction": (
                            "This chapter is not mapped onto book sections, so its "
                            "figures are unknown. Say that instead of naming a "
                            "figure; the card still names the note to read."
                        ),
                    },
                    tool=tool,
                )
            sections = []
            for number in sorted(rows, key=_figure_sort_key):
                figures = rows.get(number) or []
                if figures:
                    sections.append({"no": number, "figures": figures})
            return _result(
                {
                    "chapter": chapter,
                    "sections": sections,
                    "figures": sum(len(entry["figures"]) for entry in sections),
                    "instruction": FIGURE_INSTRUCTION,
                },
                tool=tool,
            )

        wanted = [subject] if subject else sorted(str(key) for key in books)
        payload_books: dict[str, Any] = {}
        for name in wanted:
            found = books.get(name)
            if not isinstance(found, dict):
                continue
            sections = []
            for number in sorted(found, key=_figure_sort_key):
                entry = found[number]
                figures = entry.get("figures") or []
                if not figures:
                    continue
                sections.append(
                    {
                        "no": number,
                        "title": entry.get("title", ""),
                        "anchor": entry.get("anchor", ""),
                        "figures": figures,
                    }
                )
            payload_books[name] = {
                "sections": sections,
                "figures": sum(len(entry["figures"]) for entry in sections),
            }
        if subject and not payload_books:
            return _result(
                {
                    "error": f"subject {subject} has no indexed book",
                    "subjects": sorted(str(key) for key in books),
                    "instruction": "Name the subjects that exist; invent no figures.",
                },
                tool=tool,
            )
        chapter_map = {
            str(key): sorted(str(no) for no in value)
            for key, value in chapters.items()
            if isinstance(value, dict)
        }
        return _result(
            {
                "generated_at": index.get("generated_at", ""),
                "books": payload_books,
                "total_figures": sum(entry["figures"] for entry in payload_books.values()),
                "chapters": chapter_map,
                "instruction": FIGURE_INSTRUCTION,
            },
            tool=tool,
        )


__all__ = [
    "REVIEW_AGENT_TOOL_NAMES",
    "ReviewAttemptTool",
    "ReviewBookFiguresTool",
    "ReviewDueTool",
    "ReviewExportLinkTool",
    "ReviewExportTreeTool",
    "ReviewGenerateTool",
    "ReviewHintTool",
    "ReviewPlanTool",
    "ReviewSpecsTool",
    "ReviewTodayTool",
    "ReviewWeaknessTool",
]
