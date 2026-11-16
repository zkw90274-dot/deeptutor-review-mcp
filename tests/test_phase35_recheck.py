"""Regressions for the Phase 35 recheck (N-01 ... N-11).

Each test pins one finding from ``test-report-phase35-recheck-2026-11-08.md``.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading

import pytest
from pydantic import ValidationError

from deeptutor.review import tikz
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.generate import _default_spec_id, legacy_default_spec_id
from deeptutor.review.jobs import _reuse_legacy_spec_id
from deeptutor.review.models import ReviewSpec
from deeptutor.review.qa import run_gates
from deeptutor.review.service import ReviewService

ITEM = "item-resonance-choice"
NOW = 1_800_000_000.0


def _payload() -> dict:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    return payload


def _put_fixture(client) -> None:
    assert (
        client.put(f"/api/review/specs/{SPEC_ID}", json=_payload()).status_code
        == 201
    )


# -- N-01: TeX file reads ------------------------------------------------


def test_tikz_rejects_double_at_input() -> None:
    with pytest.raises(tikz.TikzRenderError):
        tikz._prepare_source(r"\makeatletter\@@input C:/Windows/win.ini\makeatother")


def test_tikz_rejects_file_reading_packages() -> None:
    with pytest.raises(tikz.TikzRenderError):
        tikz._prepare_source(
            "\\usepackage{catchfile}\n"
            r"\CatchFileDef{\secret}{C:/Windows/win.ini}{}"
            r"\node[draw] at (0,0) {\secret};"
        )


def test_tikz_allows_safe_preamble() -> None:
    source = (
        "\\usepackage{amsmath}\n"
        "\\usetikzlibrary{circuits.ee.IEC}\n"
        r"\draw (0,0) -- (1,1);"
    )
    assert tikz._prepare_source(source) == source


# -- N-02: one bad legacy row must not take the module down -------------


def test_bad_timezone_row_stays_readable(client) -> None:
    service = ReviewService()
    service.put_spec(build_fixture_spec())

    bad_payload = _payload()
    bad_payload["id"] = "bad-tz"
    bad_payload["plan"]["timezone"] = "Mars/Olympus"
    service.store.put_spec(ReviewSpec.model_validate(bad_payload))

    assert client.get("/api/review/ui").status_code == 200
    detail = client.get("/api/review/specs/bad-tz")
    assert detail.status_code == 200
    # The queue uses the documented UTC fallback instead of refusing to run.
    assert client.get("/api/review/bad-tz/today").status_code == 200
    ids = {row["spec_id"] for row in client.get("/api/review/specs").json()["specs"]}
    assert {"fixture-hf-resonance", "bad-tz"} <= ids


def test_corrupt_row_degrades_to_a_warning(client) -> None:
    service = ReviewService()
    service.put_spec(build_fixture_spec())
    with sqlite3.connect(service.store.db_path) as conn:
        conn.execute(
            "INSERT INTO review_specs "
            "(spec_id, content_hash, created_at, updated_at, spec_json) "
            "VALUES (?, ?, ?, ?, ?)",
            ("broken", "x", NOW, NOW, '{"schema_version": "1.0"}'),
        )

    meta = client.get("/api/review/specs/broken")
    assert meta.status_code == 200
    assert meta.json()["warning"]
    assert client.get("/api/review/ui").status_code == 200

    # The agent catalog degrades the same way instead of failing entirely.
    from deeptutor.review.agent_tools import ReviewSpecsTool

    result = asyncio.run(ReviewSpecsTool().execute())
    body = json.loads(result.content)
    broken = [row for row in body["specs"] if row["id"] == "broken"]
    assert broken and broken[0]["warning"]


# -- N-03: non-finite JSON numbers --------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        b'{"item_id":"item-resonance-choice","rating":"good","elapsed_ms":NaN}',
        b'{"item_id":"item-resonance-choice","rating":"good","confidence":NaN}',
        b'{"item_id":"item-resonance-choice","rating":"good","answer":{"v":Infinity}}',
        b'{"item_id":"item-resonance-choice","rating":"good","elapsed_ms":1e400}',
    ],
)
def test_non_finite_json_is_a_422(client, body: bytes) -> None:
    _put_fixture(client)
    response = client.post(
        f"/api/review/{SPEC_ID}/attempt",
        content=body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_number"


# -- N-04: forged MathJax sentinel --------------------------------------


def test_title_cannot_forge_the_mathjax_sentinel(service: ReviewService) -> None:
    payload = _payload()
    payload["title"] = "\x00MATHJAX\x00"
    service.put_spec(payload)

    first = service.export(SPEC_ID, now=NOW)
    second = service.export(SPEC_ID, now=NOW)
    assert first == second
    assert first.count("window.MathJax={") == 1
    assert len(first.encode("utf-8")) < 3_000_000


# -- N-06: legacy spec id migration -------------------------------------


def test_regeneration_reuses_the_legacy_spec_id(service: ReviewService) -> None:
    kb, chapter = "高频", "第3章"
    legacy = legacy_default_spec_id(kb, chapter)
    legacy_payload = _payload()
    legacy_payload["id"] = legacy
    legacy_payload["sources"]["kb:高频·第3章"] = {
        "id": "kb:高频·第3章",
        "kind": "doc",
        "title": "高频",
        "kb": "高频",
        "ref": "第3章",
    }
    service.store.put_spec(ReviewSpec.model_validate(legacy_payload))

    new_payload = _payload()
    new_id = _default_spec_id(kb, chapter)
    new_payload["id"] = new_id
    old_item_id = new_payload["items"][0]["id"]
    new_item_id = f"{new_id}-{old_item_id}"
    new_payload["items"][0]["id"] = new_item_id
    for block in new_payload["sections"][0]["blocks"]:
        if block.get("item_id") == old_item_id:
            block["item_id"] = new_item_id
    new_spec = ReviewSpec.model_validate(new_payload)

    warnings: list[str] = []
    reused = _reuse_legacy_spec_id(service, new_spec, kb, chapter, warnings)
    assert reused.id == legacy
    # Item ids follow the spec id, so old events keep pointing at real items.
    assert reused.items[0].id == f"{legacy}-{old_item_id}"
    assert any(
        block.item_id == f"{legacy}-{old_item_id}"
        for section in reused.sections
        for block in section.blocks
        if block.kind == "item"
    )
    assert warnings and legacy in warnings[0]


def test_legacy_reuse_refuses_a_different_chapter(
    service: ReviewService,
) -> None:
    """Chapter 2 and Chapter-2 share a legacy slug; reuse must not overwrite."""
    kb = "Physics"
    legacy = legacy_default_spec_id(kb, "Chapter 2")
    assert legacy == legacy_default_spec_id(kb, "Chapter-2")

    legacy_payload = _payload()
    legacy_payload["id"] = legacy
    legacy_payload["sources"][f"kb:{kb}·Chapter 2"] = {
        "id": f"kb:{kb}·Chapter 2",
        "kind": "doc",
        "title": kb,
        "kb": kb,
        "ref": "Chapter 2",
    }
    service.store.put_spec(ReviewSpec.model_validate(legacy_payload))

    new_payload = _payload()
    new_payload["id"] = _default_spec_id(kb, "Chapter-2")
    new_spec = ReviewSpec.model_validate(new_payload)

    warnings: list[str] = []
    kept = _reuse_legacy_spec_id(service, new_spec, kb, "Chapter-2", warnings)
    assert kept.id == new_spec.id
    assert any("different chapter" in warning for warning in warnings)


def test_new_id_wins_when_it_already_exists(service: ReviewService) -> None:
    kb, chapter = "高频", "第3章"
    legacy_payload = _payload()
    legacy_payload["id"] = legacy_default_spec_id(kb, chapter)
    service.store.put_spec(ReviewSpec.model_validate(legacy_payload))

    new_payload = _payload()
    new_payload["id"] = _default_spec_id(kb, chapter)
    new_spec = ReviewSpec.model_validate(new_payload)
    service.store.put_spec(new_spec)

    warnings: list[str] = []
    kept = _reuse_legacy_spec_id(service, new_spec, kb, chapter, warnings)
    assert kept.id == new_spec.id
    assert warnings == []


# -- N-07: summary-only hint ladder -------------------------------------


def test_summary_only_hint_levels_are_distinct(service: ReviewService) -> None:
    from deeptutor.review.agent_tools import ReviewHintTool

    payload = _payload()
    for concept in payload["concepts"]:
        concept["summary_one_line"] = "这是摘要"
        concept["definition"] = ""
        concept["formulas"] = []
    service.put_spec(payload)

    tool = ReviewHintTool()
    level_one = json.loads(
        asyncio.run(tool.execute(spec_id=SPEC_ID, item_id=ITEM, level=1)).content
    )
    level_two = json.loads(
        asyncio.run(tool.execute(spec_id=SPEC_ID, item_id=ITEM, level=2)).content
    )
    assert level_one["hint"]
    assert "这是摘要" not in level_one["hint"]
    assert level_two["hint"] != level_one["hint"]
    assert "这是摘要" in level_two["hint"]
    assert "完整定义" in level_two["hint"]


# -- N-08: sessions roll over -------------------------------------------


def test_open_session_rolls_over_after_a_day(service: ReviewService) -> None:
    service.put_spec(build_fixture_spec())
    first = service.today(SPEC_ID, now=NOW)
    service.attempt(
        SPEC_ID,
        _attempt(session_id=first["session_id"], submission_id="rollover-1"),
        now=NOW + 10.0,
    )

    later = service.today(SPEC_ID, now=NOW + 30 * 86400.0)
    assert later["session_id"] != first["session_id"]
    sessions = {row["session_id"]: row for row in service.store.sessions_for_spec(SPEC_ID)}
    assert sessions[first["session_id"]]["ended_at"] is not None


def _attempt(*, session_id: str, submission_id: str):
    from deeptutor.review.service import AttemptInput

    return AttemptInput(
        item_id=ITEM,
        rating="good",
        session_id=session_id,
        submission_id=submission_id,
    )


# -- N-09: metrics truncation keeps the newest events -------------------


def test_metrics_truncation_keeps_the_newest_events(
    service: ReviewService, monkeypatch: pytest.MonkeyPatch
) -> None:
    service.put_spec(build_fixture_spec())
    today = service.today(SPEC_ID, now=NOW)
    for index in range(3):
        service.attempt(
            SPEC_ID,
            _attempt(
                session_id=today["session_id"],
                submission_id=f"metric-{index}",
            ),
            now=NOW + 10.0 + index,
        )

    from deeptutor.review import service as service_module

    monkeypatch.setattr(service_module, "MAX_METRIC_EVENTS", 2)
    captured: dict[str, list[str]] = {}

    def fake_retention(events, *, now, days):
        captured["ids"] = [event["submission_id"] for event in events]
        return {}

    monkeypatch.setattr(service_module, "retention_buckets", fake_retention)
    service.metrics(SPEC_ID, days=7, now=NOW + 100.0)
    assert captured["ids"] == ["metric-1", "metric-2"]


# -- N-10: remaining QA false positives ---------------------------------


@pytest.mark.parametrize("stem", ["\u200b", "\x00", "\n\t"])
def test_format_only_stem_is_rejected(stem: str) -> None:
    payload = _payload()
    payload["items"][0]["stem"] = stem
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_numeric_answer_does_not_match_a_decimal_fragment() -> None:
    payload = _payload()
    numeric = next(item for item in payload["items"] if item["kind"] == "numeric")
    numeric["answer"]["value"] = 2.0
    numeric["explanation"] = "量程是 2.5，单位是安培。"
    spec = ReviewSpec.model_validate(payload)

    report = run_gates(spec)
    assert report.passed is False
    assert any(
        gate.gate == "G4-answer-key" and not gate.passed for gate in report.gates
    )


def test_non_json_body_does_not_500(client) -> None:
    _put_fixture(client)
    response = client.post(
        f"/api/review/{SPEC_ID}/attempt",
        content=b"hello",
        headers={"content-type": "text/plain"},
    )
    assert response.status_code in (400, 422)
    assert response.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize(
    "source",
    [
        r"\ExplSyntaxOn\file_input:n{C:/Windows/win.ini}\ExplSyntaxOff",
        r"\ExplSyntaxOn\use:c{input}{C:/Windows/win.ini}\ExplSyntaxOff",
        r"\node at (0,0) {\pgfimage{C:/Windows/win.ini}};",
    ],
)
def test_tikz_rejects_expl3_and_pgf_image_readers(source: str) -> None:
    with pytest.raises(tikz.TikzRenderError):
        tikz._prepare_source(source)


def test_deeply_nested_json_never_crashes_the_endpoint(client) -> None:
    """Pathological nesting must be handled, not 500.

    CPython 3.12's C scanner parses deeply nested arrays iteratively, so the
    request can legitimately be accepted there; older interpreters hit the
    recursion limit and answer 422 invalid_json. Both outcomes are fine, a
    5xx or a dropped connection is not.
    """
    _put_fixture(client)
    nested = b"[" * 5000 + b"0" + b"]" * 5000
    body = (
        b'{"item_id":"item-resonance-choice","rating":"good","answer":'
        + nested
        + b"}"
    )
    response = client.post(
        f"/api/review/{SPEC_ID}/attempt",
        content=body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code in (200, 422), response.status_code
    if response.status_code == 422:
        assert response.json()["detail"]["code"] == "invalid_json"


def test_very_long_summary_keeps_the_hint_levels_apart(
    service: ReviewService,
) -> None:
    from deeptutor.review.agent_tools import HINT_TEXT_CHARS, ReviewHintTool

    payload = _payload()
    for concept in payload["concepts"]:
        concept["summary_one_line"] = "定" * (HINT_TEXT_CHARS + 100)
        concept["definition"] = ""
        concept["formulas"] = []
    service.put_spec(payload)

    tool = ReviewHintTool()
    level_one = json.loads(
        asyncio.run(tool.execute(spec_id=SPEC_ID, item_id=ITEM, level=1)).content
    )
    level_two = json.loads(
        asyncio.run(tool.execute(spec_id=SPEC_ID, item_id=ITEM, level=2)).content
    )
    assert level_one["hint"] != level_two["hint"]
    assert "完整定义" in level_two["hint"]
    assert len(level_two["hint"]) <= HINT_TEXT_CHARS


def test_row_key_mismatch_degrades_instead_of_writing_elsewhere(client) -> None:
    service = ReviewService()
    service.put_spec(build_fixture_spec())
    payload = _payload()
    payload["id"] = "pack-b"
    embedded = ReviewSpec.model_validate(payload)
    with sqlite3.connect(service.store.db_path) as conn:
        conn.execute(
            "INSERT INTO review_specs "
            "(spec_id, content_hash, created_at, updated_at, spec_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "pack-a",
                embedded.content_hash,
                NOW,
                NOW,
                json.dumps(embedded.model_dump(mode="json")),
            ),
        )

    meta = client.get("/api/review/specs/pack-a")
    assert meta.status_code == 200
    assert meta.json()["warning"]
    # The queue refuses the pack instead of writing a session under pack-b.
    assert client.get("/api/review/pack-a/today").status_code == 409


def test_explicit_session_id_obeys_the_rollover(service: ReviewService) -> None:
    from deeptutor.review.storage import SessionConflict

    service.put_spec(build_fixture_spec())
    today = service.today(SPEC_ID, now=NOW)
    with pytest.raises(SessionConflict):
        service.attempt(
            SPEC_ID,
            _attempt(session_id=today["session_id"], submission_id="expired-1"),
            now=NOW + 73 * 3600.0,
        )
    session = service.store.get_session(today["session_id"])
    assert session["ended_at"] == session["started_at"] + 72 * 3600


def test_metrics_treats_stale_open_sessions_as_completed_without_writing(
    service: ReviewService,
) -> None:
    service.put_spec(build_fixture_spec())
    today = service.today(SPEC_ID, now=NOW)
    service.attempt(
        SPEC_ID,
        _attempt(session_id=today["session_id"], submission_id="metric-stale"),
        now=NOW + 10.0,
    )

    metrics = service.metrics(SPEC_ID, days=7, now=NOW + 4 * 86400.0)
    assert metrics["completion"]["sessions"] == 1
    assert metrics["completion"]["completed"] == 1
    assert metrics["completion"]["rate"] == 1.0
    # Read-only: the row itself is untouched.
    assert service.store.get_session(today["session_id"])["ended_at"] is None


def test_concurrent_today_creates_one_session(service: ReviewService) -> None:
    service.put_spec(build_fixture_spec())
    barrier = threading.Barrier(2)
    session_ids: list[str] = []

    def call() -> None:
        barrier.wait()
        session_ids.append(service.today(SPEC_ID, now=NOW)["session_id"])

    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(set(session_ids)) == 1
    assert len(service.store.sessions_for_spec(SPEC_ID)) == 1


def test_metrics_truncation_orders_by_event_time(
    service: ReviewService, monkeypatch: pytest.MonkeyPatch
) -> None:
    service.put_spec(build_fixture_spec())
    today = service.today(SPEC_ID, now=NOW)
    for offset, submission in ((300, "ts-300"), (100, "ts-100"), (200, "ts-200")):
        service.attempt(
            SPEC_ID,
            _attempt(session_id=today["session_id"], submission_id=submission),
            now=NOW + offset,
        )

    from deeptutor.review import service as service_module

    monkeypatch.setattr(service_module, "MAX_METRIC_EVENTS", 2)
    captured: dict[str, list[str]] = {}

    def fake_retention(events, *, now, days):
        captured["ids"] = [event["submission_id"] for event in events]
        return {}

    monkeypatch.setattr(service_module, "retention_buckets", fake_retention)
    service.metrics(SPEC_ID, days=7, now=NOW + 400.0)
    assert captured["ids"] == ["ts-200", "ts-300"]


def test_tikz_rejects_xetexpicfile_and_if_file_exists() -> None:
    for source in (
        r"\XeTeXpicfile `C:/Windows/win.ini`",
        r"\IfFileExists{C:/Windows/win.ini}{yes}{no}",
    ):
        with pytest.raises(tikz.TikzRenderError):
            tikz._prepare_source(source)


def test_xetexpicfile_never_reaches_the_compiler(monkeypatch) -> None:
    def forbidden_resolve():
        raise AssertionError("the scanner must reject before resolving tools")

    monkeypatch.setattr(tikz, "_resolve_tools", forbidden_resolve)
    with pytest.raises(tikz.TikzRenderError):
        tikz.render_tikz_document(r"\XeTeXpicfile `C:/Windows/win.ini`")


def test_export_does_not_embed_a_xetexpicfile_image(
    service: ReviewService,
) -> None:
    from deeptutor.review.models import FigureSpec

    payload = _payload()
    payload["figures"].append(
        {
            "id": "fig-pic",
            "kind": "tikz",
            "title": "图片注入",
            "tikz": r"\XeTeXpicfile `C:/Windows/win.ini`",
        }
    )
    service.put_spec(ReviewSpec.model_validate(payload))

    html = service.export(SPEC_ID, now=NOW)
    # The sanitizer's own regex mentions ``data:image``; the embedded payload
    # marker is the base64 prefix.
    assert "data:image/png;base64," not in html
    assert "xetexpicfile" in html.lower()  # placeholder warning names the cause


@pytest.mark.parametrize("explanation", ["量程是 .2 安培。", "量程是 2e3 安培。"])
def test_numeric_answer_does_not_match_fragments(explanation: str) -> None:
    payload = _payload()
    numeric = next(item for item in payload["items"] if item["kind"] == "numeric")
    numeric["answer"]["value"] = 2.0
    numeric["explanation"] = explanation
    spec = ReviewSpec.model_validate(payload)

    report = run_gates(spec)
    assert report.passed is False
    assert any(
        gate.gate == "G4-answer-key" and not gate.passed for gate in report.gates
    )


def test_tikz_allows_audited_drawing_packages() -> None:
    for package in ("circuitikz", "tikz-3dplot"):
        source = (
            f"\\usepackage{{{package}}}\n" + r"\draw (0,0) -- (1,1);"
        )
        assert tikz._prepare_source(source) == source
