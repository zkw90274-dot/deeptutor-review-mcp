"""Regressions for the 2026-11-08 deep audit.

Each test pins one finding from ``test-report-deep-audit-2026-11-08.md``:
a P1 security/correctness boundary or a P2 data-integrity defect.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest
from pydantic import ValidationError

from deeptutor.review import tikz
from deeptutor.review.book_import import _safe_id
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.generate import _default_spec_id
from deeptutor.review.models import ReviewSpec
from deeptutor.review.qa import run_gates
from deeptutor.review.service import AttemptInput, ReviewService
from deeptutor.review.storage import SessionConflict

ITEM = "item-resonance-choice"
SESSION = "audit-session"


def _payload() -> dict:
    payload = build_fixture_spec().model_dump(mode="json")
    # Mutated payloads must not carry the old hash; an empty value asks the
    # model to recompute it.
    payload["content_hash"] = ""
    return payload


def _attempt(**overrides) -> AttemptInput:
    values = {
        "item_id": ITEM,
        "rating": "good",
        "session_id": SESSION,
        "submission_id": "audit-submission",
    }
    values.update(overrides)
    return AttemptInput(**values)


# -- P1-1: TeX ^^ escapes ------------------------------------------------


def test_tikz_rejects_tex_caret_escape() -> None:
    with pytest.raises(tikz.TikzRenderError):
        tikz._prepare_source(r"\^^69nput{/etc/passwd}")


def test_tikz_still_allows_a_single_caret() -> None:
    assert tikz._prepare_source(r"\node at (0,0) {$x^2$};")


def test_tikz_caret_escape_inside_a_comment_is_ignored() -> None:
    assert tikz._prepare_source("% ^^69 is a comment\n\\draw (0,0)--(1,1);") == (
        "\\draw (0,0)--(1,1);"
    )


# -- P1-2: session inflation ---------------------------------------------


def test_repeated_today_reuses_the_open_session(service: ReviewService) -> None:
    service.put_spec(build_fixture_spec())
    session_ids = {
        service.today(SPEC_ID, now=1_800_000_000.0)["session_id"] for _ in range(5)
    }
    assert len(session_ids) == 1
    assert len(service.store.sessions_for_spec(SPEC_ID)) == 1


def test_completion_rate_ignores_browsing_only_sessions(
    service: ReviewService,
) -> None:
    service.put_spec(build_fixture_spec())
    today = service.today(SPEC_ID, now=1_800_000_000.0)

    browsing = service.metrics(SPEC_ID, days=7, now=1_800_000_100.0)
    assert browsing["completion"]["sessions"] == 0
    assert browsing["completion"]["rate"] is None

    service.attempt(
        SPEC_ID,
        _attempt(session_id=today["session_id"]),
        now=1_800_000_200.0,
    )
    service.complete(SPEC_ID, today["session_id"], now=1_800_000_300.0)

    answered = service.metrics(SPEC_ID, days=7, now=1_800_000_400.0)
    assert answered["completion"]["sessions"] == 1
    assert answered["completion"]["completed"] == 1
    assert answered["completion"]["rate"] == 1.0


# -- P2-1: writes into a closed session ----------------------------------


def test_attempt_into_a_completed_session_is_rejected(
    service: ReviewService,
) -> None:
    service.put_spec(build_fixture_spec())
    today = service.today(SPEC_ID, now=1_800_000_000.0)
    service.complete(SPEC_ID, today["session_id"], now=1_800_000_100.0)

    with pytest.raises(SessionConflict):
        service.attempt(
            SPEC_ID,
            _attempt(session_id=today["session_id"]),
            now=1_800_000_200.0,
        )


# -- P2-2: concurrent put_spec -------------------------------------------


def test_concurrent_put_spec_keeps_snapshot_and_row_in_sync(
    service: ReviewService,
) -> None:
    service.put_spec(build_fixture_spec())
    errors: list[BaseException] = []

    def writer(index: int) -> None:
        payload = _payload()
        payload["title"] = f"并发标题 {index}"
        try:
            ReviewService(service.store).put_spec(payload)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(index,)) for index in range(24)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    stored = service.store.get_spec(SPEC_ID)
    snapshot_path = service.store.home / f"review_{SPEC_ID}" / "spec.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["content_hash"] == stored.content_hash
    assert snapshot["title"] == stored.title


def test_attempt_tool_rejects_coerced_hint_counts(service: ReviewService) -> None:
    from deeptutor.review.agent_tools import ReviewAttemptTool

    tool = ReviewAttemptTool()
    for raw in (True, "2", 99):
        result = asyncio.run(
            tool.execute(
                spec_id=SPEC_ID,
                item_id=ITEM,
                rating="good",
                hints_used=raw,
            )
        )
        assert result.success is True  # soft failure, not an exception
        assert "hints_used" in json.loads(result.content)["error"]


# -- P2-3: replay reports the original due date --------------------------


def test_replay_reports_the_original_due_date(service: ReviewService) -> None:
    service.put_spec(build_fixture_spec())
    first = service.attempt(SPEC_ID, _attempt(), now=1_800_000_000.0)
    replay = service.attempt(
        SPEC_ID, _attempt(), now=1_800_000_000.0 + 3 * 86400.0
    )

    assert replay["created"] is False
    assert replay["next_due_ts"] == first["next_due_ts"]
    assert (
        replay["scheduled_interval_days"] == first["scheduled_interval_days"]
    )


# -- P2-4: QA blind spots ------------------------------------------------


def test_blank_stem_is_rejected_by_the_model() -> None:
    payload = _payload()
    payload["items"][0]["stem"] = "   "
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_duplicate_option_text_is_rejected() -> None:
    payload = _payload()
    options = payload["items"][0]["options"]
    options[1]["text"] = options[0]["text"]
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_numeric_answer_must_be_a_standalone_number() -> None:
    payload = _payload()
    numeric = next(item for item in payload["items"] if item["kind"] == "numeric")
    numeric["answer"]["value"] = 2.0
    numeric["explanation"] = "解释里的数字是 12，与答案无关。"
    spec = ReviewSpec.model_validate(payload)

    report = run_gates(spec)
    assert report.passed is False
    assert any(
        gate.gate == "G4-answer-key" and not gate.passed for gate in report.gates
    )


def test_short_answer_must_appear_in_the_explanation() -> None:
    payload = _payload()
    concept_id = payload["concepts"][0]["id"]
    payload["items"].append(
        {
            "id": "item-short",
            "concept_ids": [concept_id],
            "kind": "short_answer",
            "stem": "关键步骤是什么？",
            "options": [],
            "answer": {"text": "先调谐再测量"},
            "explanation": "解释与答案键完全无关。",
            "source_ids": [payload["items"][0]["source_ids"][0]],
        }
    )
    payload["sections"][0]["blocks"].append({"kind": "item", "item_id": "item-short"})
    spec = ReviewSpec.model_validate(payload)

    report = run_gates(spec)
    assert report.passed is False
    assert any(
        gate.gate == "G4-answer-key" and not gate.passed for gate in report.gates
    )


# -- P2-5: numeric input boundaries --------------------------------------


def test_elapsed_ms_out_of_range_is_rejected_by_the_service(
    service: ReviewService,
) -> None:
    service.put_spec(build_fixture_spec())
    with pytest.raises(ValueError):
        service.attempt(SPEC_ID, _attempt(elapsed_ms=2**70), now=1_800_000_000.0)
    with pytest.raises(ValueError):
        service.attempt(SPEC_ID, _attempt(elapsed_ms=True), now=1_800_000_000.0)


def test_elapsed_ms_out_of_range_is_a_422(client) -> None:
    spec = _payload()
    assert client.put(f"/api/review/specs/{SPEC_ID}", json=spec).status_code == 201
    huge = client.post(
        f"/api/review/{SPEC_ID}/attempt",
        json={"item_id": ITEM, "rating": "good", "elapsed_ms": 2**70},
    )
    assert huge.status_code == 422
    boolean = client.post(
        f"/api/review/{SPEC_ID}/attempt",
        json={"item_id": ITEM, "rating": "good", "elapsed_ms": True},
    )
    assert boolean.status_code == 422
    # Transport strings and fractional JSON numbers are not durations either;
    # pydantic would otherwise coerce both into an int.
    text = client.post(
        f"/api/review/{SPEC_ID}/attempt",
        json={"item_id": ITEM, "rating": "good", "elapsed_ms": "100"},
    )
    assert text.status_code == 422
    fractional = client.post(
        f"/api/review/{SPEC_ID}/attempt",
        json={"item_id": ITEM, "rating": "good", "elapsed_ms": 100.0},
    )
    assert fractional.status_code == 422


# -- P2-6: timezone validation -------------------------------------------


def test_unknown_timezone_is_rejected_on_write_only(
    service: ReviewService,
) -> None:
    payload = _payload()
    payload["plan"]["timezone"] = "Mars/Olympus"
    # Reads must keep working: a legacy row with a bad zone stays parseable
    # and timeutil falls back to UTC with a warning.
    legacy = ReviewSpec.model_validate(payload)
    with pytest.raises(ValueError):
        service.put_spec(legacy)


# -- P2-7: id collisions -------------------------------------------------


def test_generated_spec_ids_stay_distinct_when_slugs_collide() -> None:
    assert _default_spec_id("KB", "CS 101") != _default_spec_id("KB", "CS-101")


def test_book_generated_ids_stay_distinct_for_cjk_ids() -> None:
    first = _safe_id("第1章", prefix="src-", fallback="src-book-page", unique=True)
    second = _safe_id("第2章", prefix="src-", fallback="src-book-page", unique=True)
    assert first != second
    # Explicit ids that callers rely on are preserved verbatim.
    assert _safe_id("item-custom", prefix="", fallback="x") == "item-custom"


# -- P2-9: hint ladder and P2-10: projection reason ----------------------

def test_generated_hint_levels_are_distinct(service: ReviewService) -> None:
    from deeptutor.review.agent_tools import ReviewHintTool

    payload = _payload()
    for concept in payload["concepts"]:
        concept["summary_one_line"] = ""
        concept["definition"] = ""
        concept["formulas"] = []
    service.put_spec(payload)

    tool = ReviewHintTool()
    first = json.loads(
        asyncio.run(tool.execute(spec_id=SPEC_ID, item_id=ITEM, level=1)).content
    )
    second = json.loads(
        asyncio.run(tool.execute(spec_id=SPEC_ID, item_id=ITEM, level=2)).content
    )
    assert first["hint"] != second["hint"]


def test_skipped_projection_reports_no_linkage(client) -> None:
    spec = _payload()
    assert client.put(f"/api/review/specs/{SPEC_ID}", json=spec).status_code == 201
    response = client.post(
        f"/api/review/{SPEC_ID}/attempt",
        json={"item_id": ITEM, "rating": "good", "submission_id": "projection-1"},
    )
    assert response.status_code == 200
    assert response.json()["state_source_reason"] == "no_linkage"


# -- P1-3: template token collision --------------------------------------


def test_export_payload_cannot_inject_template_tokens(
    service: ReviewService,
) -> None:
    payload = _payload()
    payload["items"][0]["stem"] = (
        "题干 {{MATHJAX}} {{TITLE}} {{PAYLOAD}} 结束"
    )
    service.put_spec(payload)

    html = service.export(SPEC_ID, now=1_800_000_000.0)
    assert html.count("window.MathJax={") == 1
    # The literals stay inside the JSON payload instead of being substituted.
    assert "{{MATHJAX}}" in html
    assert "{{TITLE}}" in html
    assert "{{PAYLOAD}}" in html
    assert html == service.export(SPEC_ID, now=1_800_000_000.0)


def test_export_endpoint_survives_template_tokens(client) -> None:
    payload = _payload()
    payload["items"][0]["stem"] = "题干 {{MATHJAX}} {{PAYLOAD}} 结束"
    assert client.put(f"/api/review/specs/{SPEC_ID}", json=payload).status_code == 201

    first = client.post(f"/api/review/{SPEC_ID}/export")
    second = client.post(f"/api/review/{SPEC_ID}/export")
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.content == second.content
    text = first.content.decode("utf-8")
    assert text.count("window.MathJax={") == 1
    assert "{{MATHJAX}}" in text
    assert "{{PAYLOAD}}" in text
