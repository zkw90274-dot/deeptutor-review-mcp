"""WS-A: the five Review Studio agent tools.

The tests drive the tools the way a chat model would: call them through
``execute`` with model-shaped arguments, parse the JSON body the model reads,
and check the ``instruction`` field and the soft-failure contract from
``docs/maintainer-notes.md`` §5.

No async test plugin is installed in this package, so each call is wrapped in
``asyncio.run`` the same way ``tests/test_projection.py`` does.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import types
from typing import Any

import pytest

from deeptutor.core.tool_protocol import BaseTool, ToolResult
from deeptutor.review import agent_tools
from deeptutor.review.agent_tools import (
    REVIEW_AGENT_TOOL_NAMES,
    ReviewAttemptTool,
    ReviewBookFiguresTool,
    ReviewDueTool,
    ReviewExportLinkTool,
    ReviewGenerateTool,
    ReviewHintTool,
    ReviewPlanTool,
    ReviewSpecsTool,
    ReviewTodayTool,
    ReviewWeaknessTool,
)
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.service import ReviewService
from deeptutor.review.mcp_server import TOOL_CLASSES

TOOLS: dict[str, type[BaseTool]] = {
    "review_specs": ReviewSpecsTool,
    "review_due": ReviewDueTool,
    "review_today": ReviewTodayTool,
    "review_attempt": ReviewAttemptTool,
    "review_weakness": ReviewWeaknessTool,
    "review_export_link": ReviewExportLinkTool,
    "review_book_figures": ReviewBookFiguresTool,
    "review_generate": ReviewGenerateTool,
    "review_plan": ReviewPlanTool,
}


def _run(tool: BaseTool, **kwargs: Any) -> ToolResult:
    return asyncio.run(tool.execute(**kwargs))


def _payload(result: ToolResult, *, tool: str) -> dict[str, Any]:
    """Assert the frozen result shape, then return the parsed payload."""
    assert result.success is True, result.content
    payload = json.loads(result.content)
    assert isinstance(payload, dict)
    assert "instruction" in payload and payload["instruction"]
    assert result.metadata[tool] == payload
    return payload


@pytest.fixture()
def seeded(service: ReviewService) -> ReviewService:
    service.put_spec(build_fixture_spec())
    return service


def test_hint_ledger_downgrades_good(seeded: ReviewService) -> None:
    today = _payload(_run(ReviewTodayTool(), spec_id=SPEC_ID, include_draft=True), tool="review_today")
    args = dict(spec_id=SPEC_ID, session_id=today["session_id"], item_id=today["items"][0]["item_id"])
    _run(ReviewHintTool(), **args, level=1)
    _run(ReviewHintTool(), **args, level=2)
    attempt = dict(**args, rating="good", hints_used=0, submission_id="hint-ledger")
    result = _payload(_run(ReviewAttemptTool(), **attempt), tool="review_attempt")
    assert result["rating"] == "hard"
    assert result["hints_used"] == 2
    assert seeded.store.event_by_submission("hint-ledger")["hints_issued"] == 2
    _run(ReviewHintTool(), **args, level=2)
    replay = _payload(_run(ReviewAttemptTool(), **attempt), tool="review_attempt")
    assert replay["event_id"] == result["event_id"]
    assert replay["created"] is False


def test_mcp_attempt_requires_shown_item(seeded: ReviewService, fixture_spec) -> None:
    result = _payload(_run(ReviewAttemptTool(), spec_id=SPEC_ID, item_id=fixture_spec.items[0].id, rating="again"), tool="review_attempt")
    assert result["error"] == "item_not_shown"
    assert seeded.store.event_count(SPEC_ID) == 0


def test_mcp_attempt_hides_answer_until_reveal(seeded: ReviewService, monkeypatch) -> None:
    monkeypatch.setenv("DEEPTUTOR_REVIEW_MAX_REVEALS_PER_SESSION", "2")
    today = _payload(_run(ReviewTodayTool(), spec_id=SPEC_ID, include_draft=True), tool="review_today")
    result = _payload(_run(ReviewAttemptTool(), spec_id=SPEC_ID, item_id=today["items"][0]["item_id"], session_id=today["session_id"], rating="again"), tool="review_attempt")
    assert result["event_id"] == result["reveal_token"]
    assert not {"answer", "explanation", "correction", "study_aids"}.intersection(result)
    reveal = TOOL_CLASSES["review_reveal"]()
    args = dict(spec_id=SPEC_ID, event_id=result["event_id"])
    first = _payload(_run(reveal, **args), tool="review_reveal")
    second = _payload(_run(reveal, **args), tool="review_reveal")
    assert first == second and first["answer"] and first["correction"]
    blocked = _payload(_run(reveal, **args), tool="review_reveal")
    assert blocked["error"] == "reveal_limit"
    assert "answer" not in blocked


# -- shared contract --------------------------------------------------------


def test_all_tools_are_base_tools_with_frozen_names() -> None:
    assert REVIEW_AGENT_TOOL_NAMES == (
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
        "review_reveal",
        "review_quality",
        "review_quality_decision",
        "review_correct",
        "review_metrics",
        "review_session",
        "review_link",
        "review_session_plan",
        "review_artifact_status",
    )
    for name, tool_type in TOOLS.items():
        tool = tool_type()
        assert isinstance(tool, BaseTool)
        definition = tool.get_definition()
        assert definition.name == name
        # descriptions are English prose aimed at a model: no CJK characters,
        # even though the learner's material itself is Chinese.
        assert len(definition.description) > 80
        assert not any("一" <= char <= "鿿" for char in definition.description)


def test_spec_id_parameter_is_required_on_the_four_addressed_tools() -> None:
    for name in ("review_today", "review_weakness", "review_export_link"):
        definition = TOOLS[name]().get_definition()
        assert definition.parameters[0].name == "spec_id"
        assert definition.parameters[0].required is True
    attempt = TOOLS["review_attempt"]().get_definition()
    by_name = {parameter.name: parameter for parameter in attempt.parameters}
    assert set(by_name) == {
        "spec_id",
        "item_id",
        "rating",
        "session_id",
        "submission_id",
        "hints_used",
        "confidence",
        "learner_quote",
        "answer",
        "elapsed_ms",
    }
    assert by_name["rating"].enum == ["again", "hard", "good"]
    assert by_name["submission_id"].required is False
    assert by_name["session_id"].required is False


def test_attempt_description_locks_the_self_assessment_rule() -> None:
    description = TOOLS["review_attempt"]().get_definition().description
    assert "ONLY call after the learner self-assesses aloud" in description
    assert "Never grade on their behalf" in description
    lowered = description.lower()
    # the acceptance script checks these two words; keep both spellings.
    assert "self-assess" in lowered
    assert "never grade" in lowered


# -- review_specs -----------------------------------------------------------


def test_specs_lists_packs_with_counts_and_qa(seeded: ReviewService) -> None:
    payload = _payload(_run(ReviewSpecsTool()), tool="review_specs")
    assert payload["count"] == 1
    row = payload["specs"][0]
    assert row["id"] == SPEC_ID
    assert row["title"]
    assert row["items"] == 3
    assert row["concepts"] == 2
    assert row["qa_passed"] is True
    assert "review_today" in payload["instruction"]


def test_specs_on_an_empty_store_redirects_instead_of_inventing(
    service: ReviewService,
) -> None:
    payload = _payload(_run(ReviewSpecsTool()), tool="review_specs")
    assert payload["count"] == 0
    assert payload["specs"] == []
    assert "no review packs" in payload["instruction"]


def test_specs_reports_an_internal_failure(
    service: ReviewService, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> ReviewService:
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(agent_tools, "_service", boom)
    result = _run(ReviewSpecsTool())
    assert result.success is False
    assert "review_specs failed" in result.content


def test_specs_orders_by_recency_numerically(service: ReviewService) -> None:
    """``updated_at`` is an epoch float; a digit boundary must not reorder it."""

    for spec_id, timestamp in (("older", 9_999_999_999.0), ("newer", 10_000_000_000.0)):
        spec = build_fixture_spec().model_copy(update={"id": spec_id, "content_hash": ""})
        service.put_spec(spec)
        with service.store._transaction() as conn:
            conn.execute("UPDATE review_specs SET updated_at = ? WHERE spec_id = ?", (timestamp, spec_id))
    payload = _payload(_run(ReviewSpecsTool()), tool="review_specs")
    assert [row["id"] for row in payload["specs"]] == ["newer", "older"]


# -- review_today -----------------------------------------------------------


def test_today_returns_items_without_leaking_answers(seeded: ReviewService) -> None:
    payload = _payload(
        _run(ReviewTodayTool(), spec_id=SPEC_ID), tool="review_today"
    )
    assert payload["spec_id"] == SPEC_ID
    assert payload["budget"] >= 3
    assert payload["planned_minutes"] > 0
    assert payload["session_id"]
    assert len(payload["items"]) == 3
    first = payload["items"][0]
    assert first["item_id"]
    assert first["kind"] in {"single_choice", "numeric", "flashcard"}
    assert first["stem"]
    # Red line: the answer only appears after review_attempt records one.
    for item in payload["items"]:
        assert "answer" not in item
        assert "explanation" not in item
        assert "self_check" not in item
        assert "pitfalls" not in item
        assert "assumptions" not in item
        assert "study_aids" not in item
    assert "review_attempt" in payload["instruction"]
    assert payload["session_id"] in payload["instruction"]


def test_today_clips_long_stems(
    service: ReviewService, monkeypatch: pytest.MonkeyPatch
) -> None:
    long_stem = "题干" * 400
    monkeypatch.setattr(
        agent_tools,
        "STEM_LIMIT",
        40,
    )
    spec = build_fixture_spec()
    payload = spec.model_dump(mode="json")
    payload["items"][0]["stem"] = long_stem
    payload["content_hash"] = ""
    service.put_spec(payload)
    result = _payload(_run(ReviewTodayTool(), spec_id=SPEC_ID), tool="review_today")
    stems = [item["stem"] for item in result["items"]]
    assert all(len(stem) <= 40 for stem in stems)
    assert any(stem.endswith("…") for stem in stems)


def test_today_does_not_truncate_realistic_stems_or_options(
    service: ReviewService,
) -> None:
    """A 900-char stem (real generated packs go that long) must arrive whole."""
    stem = "某电子线路如图所示，已知参数如下：" + "分析过程" * 180  # ~900 chars
    option_text = "当一个放大器的输入输出阻抗与滤波网络的匹配条件满足时，" * 12
    spec = build_fixture_spec()
    payload = spec.model_dump(mode="json")
    payload["items"][0]["stem"] = stem
    options = payload["items"][0].get("options") or []
    if options:
        options[0]["text"] = option_text
    payload["content_hash"] = ""
    service.put_spec(payload)

    result = _payload(_run(ReviewTodayTool(), spec_id=SPEC_ID), tool="review_today")
    served = {item["item_id"]: item for item in result["items"]}
    item = served[payload["items"][0]["id"]]
    assert item["stem"] == stem
    if options:
        assert item["options"][0]["text"] == option_text


def test_today_with_an_unknown_spec_is_a_soft_failure(
    service: ReviewService,
) -> None:
    payload = _payload(
        _run(ReviewTodayTool(), spec_id="no-such-pack"), tool="review_today"
    )
    assert payload["error"] == "spec not found: no-such-pack"
    assert "review_specs" in payload["instruction"]


def test_today_without_a_spec_id_is_a_soft_failure(service: ReviewService) -> None:
    payload = _payload(_run(ReviewTodayTool()), tool="review_today")
    assert payload["error"] == "spec_id is required"
    assert "review_specs" in payload["instruction"]


def test_today_reports_an_internal_failure(
    seeded: ReviewService, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenToday(ReviewService):
        def today(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("store corrupted")

    monkeypatch.setattr(
        agent_tools, "_service", lambda: _BrokenToday(seeded.store)
    )
    result = _run(ReviewTodayTool(), spec_id=SPEC_ID)
    assert result.success is False
    assert "review_today failed" in result.content


# -- review_attempt ---------------------------------------------------------


def test_today_item_id_relays_into_attempt_and_advances_the_queue(
    seeded: ReviewService,
) -> None:
    queue = _payload(_run(ReviewTodayTool(), spec_id=SPEC_ID), tool="review_today")
    item_id = queue["items"][0]["item_id"]

    attempt = _payload(
        _run(
            ReviewAttemptTool(),
            spec_id=SPEC_ID,
            item_id=item_id,
            rating="good",
            session_id=queue["session_id"],
            submission_id="relay-1",
        ),
        tool="review_attempt",
    )
    assert attempt["item_id"] == item_id
    assert attempt["session_id"] == queue["session_id"]
    assert attempt["created"] is True
    assert attempt["rating"] == "good"
    assert attempt["event_id"]
    assert attempt["next"]["due_at"]
    assert attempt["mastery_projection"]["status"] in {
        "skipped",
        "applied",
        "duplicate",
        "invalid_linkage",
    }
    # The answer is released only here, together with its explanation and
    # structured study aids.
    assert attempt["answer"]
    assert "explanation" in attempt
    assert set(attempt["study_aids"]) == {
        "assumptions",
        "self_check",
        "pitfalls",
        "table",
    }
    assert (
        "review_today" in attempt["instruction"]
        or "next item" in attempt["instruction"]
    )

    after = _payload(_run(ReviewTodayTool(), spec_id=SPEC_ID), tool="review_today")
    due_after = {row["item_id"] for row in after["items"] if row["status"] == "due"}
    assert item_id not in due_after


def test_attempt_is_idempotent_for_a_repeated_submission_id(
    seeded: ReviewService,
) -> None:
    first = _payload(
        _run(
            ReviewAttemptTool(),
            spec_id=SPEC_ID,
            item_id="item-resonance-choice",
            rating="good",
            submission_id="idem-1",
        ),
        tool="review_attempt",
    )
    replay = _payload(
        _run(
            ReviewAttemptTool(),
            spec_id=SPEC_ID,
            item_id="item-resonance-choice",
            rating="good",
            submission_id="idem-1",
        ),
        tool="review_attempt",
    )
    assert replay["created"] is False
    assert replay["event_id"] == first["event_id"]
    assert "already recorded" in replay["instruction"]
    assert seeded.store.event_count(SPEC_ID) == 1


def test_reused_submission_id_with_a_different_rating_is_a_soft_conflict(
    seeded: ReviewService,
) -> None:
    _run(
        ReviewAttemptTool(),
        spec_id=SPEC_ID,
        item_id="item-resonance-choice",
        rating="good",
        submission_id="conflict-1",
    )
    payload = _payload(
        _run(
            ReviewAttemptTool(),
            spec_id=SPEC_ID,
            item_id="item-resonance-choice",
            rating="again",
            submission_id="conflict-1",
        ),
        tool="review_attempt",
    )
    assert "submission_id was already used" in payload["error"]
    assert "Never change the rating" in payload["instruction"]
    assert seeded.store.event_count(SPEC_ID) == 1


def test_attempt_rejects_an_unknown_item_without_writing(
    seeded: ReviewService,
) -> None:
    payload = _payload(
        _run(
            ReviewAttemptTool(),
            spec_id=SPEC_ID,
            item_id="item-does-not-exist",
            rating="hard",
        ),
        tool="review_attempt",
    )
    assert payload["error"] == "unknown item: item-does-not-exist"
    assert "review_today" in payload["instruction"]
    assert seeded.store.event_count(SPEC_ID) == 0


def test_attempt_rejects_a_rating_the_model_invented(
    seeded: ReviewService,
) -> None:
    payload = _payload(
        _run(
            ReviewAttemptTool(),
            spec_id=SPEC_ID,
            item_id="item-resonance-choice",
            rating="excellent",
        ),
        tool="review_attempt",
    )
    assert "unknown rating" in payload["error"]
    assert "again / hard / good" in payload["instruction"]
    assert seeded.store.event_count(SPEC_ID) == 0


def test_attempt_with_an_unknown_spec_is_a_soft_failure(
    seeded: ReviewService,
) -> None:
    payload = _payload(
        _run(
            ReviewAttemptTool(),
            spec_id="no-such-pack",
            item_id="item-resonance-choice",
            rating="good",
        ),
        tool="review_attempt",
    )
    assert payload["error"] == "spec not found: no-such-pack"
    assert "review_specs" in payload["instruction"]


def test_attempt_reports_an_internal_failure(
    seeded: ReviewService, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenAttempt(ReviewService):
        def attempt(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("database is locked")

    monkeypatch.setattr(
        agent_tools, "_service", lambda: _BrokenAttempt(seeded.store)
    )
    result = _run(
        ReviewAttemptTool(),
        spec_id=SPEC_ID,
        item_id="item-resonance-choice",
        rating="good",
    )
    assert result.success is False
    assert "review_attempt failed" in result.content
    assert seeded.store.event_count(SPEC_ID) == 0


def test_attempt_marks_a_same_session_retest(seeded: ReviewService) -> None:
    queue = _payload(_run(ReviewTodayTool(), spec_id=SPEC_ID), tool="review_today")
    item_id = queue["items"][0]["item_id"]
    session_id = queue["session_id"]
    _run(
        ReviewAttemptTool(),
        spec_id=SPEC_ID,
        item_id=item_id,
        rating="again",
        session_id=session_id,
        submission_id="retest-1",
    )
    retest = _payload(
        _run(
            ReviewAttemptTool(),
            spec_id=SPEC_ID,
            item_id=item_id,
            rating="good",
            session_id=session_id,
            submission_id="retest-2",
        ),
        tool="review_attempt",
    )
    assert retest["retry"] is True
    assert retest["rating"] == "hard"
    assert retest["requested_rating"] == "good"
    assert "not unaided recall" in retest["instruction"]


# -- review_weakness --------------------------------------------------------


def test_weakness_ranks_concepts_and_counts_evidence(
    seeded: ReviewService,
) -> None:
    _run(
        ReviewAttemptTool(),
        spec_id=SPEC_ID,
        item_id="item-resonance-choice",
        rating="again",
        session_id="weak-session",
    )
    payload = _payload(
        _run(ReviewWeaknessTool(), spec_id=SPEC_ID), tool="review_weakness"
    )
    assert payload["limit"] == 5
    rows = payload["weak_concepts"]
    assert {row["concept_id"] for row in rows} == {
        "c-series-resonance",
        "c-quality-factor",
    }
    scores = [row["weakness"] for row in rows]
    assert scores == sorted(scores, reverse=True)
    attempted = next(row for row in rows if row["concept_id"] == "c-series-resonance")
    assert attempted["evidence_count"] == 1
    # The item ids ride along so a caller can tell which queue entry this
    # concept's evidence came from.
    assert "item-resonance-choice" in attempted["item_ids"]
    untouched = next(row for row in rows if row["concept_id"] == "c-quality-factor")
    assert untouched["evidence_count"] == 0
    assert all(row["concept_name"] for row in rows)


def test_weakness_is_read_only(seeded: ReviewService) -> None:
    before = seeded.store.event_count(SPEC_ID)
    _payload(_run(ReviewWeaknessTool(), spec_id=SPEC_ID), tool="review_weakness")
    assert seeded.store.event_count(SPEC_ID) == before
    assert seeded.store.list_specs()


def test_weakness_with_an_unknown_spec_is_a_soft_failure(
    service: ReviewService,
) -> None:
    payload = _payload(
        _run(ReviewWeaknessTool(), spec_id="no-such-pack"), tool="review_weakness"
    )
    assert payload["error"] == "spec not found: no-such-pack"
    assert "review_specs" in payload["instruction"]


def test_weakness_reports_an_internal_failure(
    seeded: ReviewService, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenWeakness(ReviewService):
        def weakness(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("replay failed")

    monkeypatch.setattr(
        agent_tools, "_service", lambda: _BrokenWeakness(seeded.store)
    )
    result = _run(ReviewWeaknessTool(), spec_id=SPEC_ID)
    assert result.success is False
    assert "review_weakness failed" in result.content


# -- review_export_link -----------------------------------------------------


def test_export_link_is_a_relative_url_for_the_pack(seeded: ReviewService) -> None:
    payload = _payload(
        _run(ReviewExportLinkTool(), spec_id=SPEC_ID), tool="review_export_link"
    )
    assert payload["url"] == f"/api/review/{SPEC_ID}/export"
    assert "://" not in payload["url"]
    assert "browser" in payload["instruction"]


def test_export_link_with_an_unknown_spec_is_a_soft_failure(
    service: ReviewService,
) -> None:
    payload = _payload(
        _run(ReviewExportLinkTool(), spec_id="no-such-pack"),
        tool="review_export_link",
    )
    assert payload["error"] == "spec not found: no-such-pack"
    assert "review_specs" in payload["instruction"]


def test_export_link_reports_an_internal_failure(
    service: ReviewService, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> ReviewService:
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(agent_tools, "_service", boom)
    result = _run(ReviewExportLinkTool(), spec_id=SPEC_ID)
    assert result.success is False
    assert "review_export_link failed" in result.content


# -- review_generate (Phase 31 WS-2) ----------------------------------------


@pytest.fixture()
def generate_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """The WS-1 generator module, real if present, otherwise a stand-in.

    ``deeptutor.review.generate`` is owned by the parallel work stream. Until
    it lands, inject a stub into ``sys.modules``; once it lands, the real
    module is used and its functions are monkeypatched in place.
    """
    try:
        from deeptutor.review import generate
    except ImportError:
        generate = types.ModuleType("deeptutor.review.generate")
        # Placeholders so ``monkeypatch.setattr`` can replace them; the real
        # module already defines both.
        setattr(generate, "infer_chapters", lambda kb_name: [])
        setattr(generate, "generate_spec_from_kb", lambda **kwargs: None)
        monkeypatch.setitem(sys.modules, "deeptutor.review.generate", generate)
    return generate

def test_generate_tool_definition_is_english_and_bounded() -> None:
    definition = ReviewGenerateTool().get_definition()
    assert definition.name == "review_generate"
    assert len(definition.description) > 80
    assert not any("一" <= char <= "鿿" for char in definition.description)
    by_name = {parameter.name: parameter for parameter in definition.parameters}
    assert list(by_name) == [
        "kb",
        "chapter",
        "num_questions",
        "question_types",
        "difficulty",
        "job_id",
        "wait_seconds",
        "replace",
    ]
    assert by_name["kb"].required is True
    assert by_name["chapter"].required is False
    assert by_name["num_questions"].default == 12
    assert by_name["num_questions"].minimum == 5
    assert by_name["num_questions"].maximum == 20
    assert by_name["question_types"].type == "array"
    assert by_name["question_types"].items == {
        "type": "string",
        "enum": ["choice", "written"],
    }
    assert by_name["difficulty"].required is False
    assert by_name["job_id"].required is False
    assert by_name["wait_seconds"].required is False
    assert by_name["wait_seconds"].default == by_name["wait_seconds"].minimum == 0
    assert by_name["wait_seconds"].maximum == 60
    assert by_name["replace"].type == "boolean"
    assert by_name["replace"].required is False
    assert by_name["replace"].default is False


@pytest.fixture()
def jobs_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A fake job store: these tests assert the tool's start/poll contract.

    The real store is covered by ``test_jobs.py``; here the module is always
    replaced so the tool tests never spawn background threads.
    """
    jobs = types.ModuleType("deeptutor.review.jobs")

    def fake_start(**kwargs: Any) -> dict[str, Any]:
        return {"job_id": "gen-1-abcd", "status": "queued", **kwargs}

    def fake_load(job_id: str) -> dict[str, Any] | None:
        return None

    def fake_is_stale(job: dict[str, Any], *, now: float | None = None) -> bool:
        if job.get("status") not in ("queued", "running"):
            return False
        stamp = float(job.get("updated_at") or 0)
        return (now or time.time()) - stamp > 1800

    setattr(jobs, "start_generation_job", fake_start)
    setattr(jobs, "load_job", fake_load)
    setattr(jobs, "is_stale", fake_is_stale)
    monkeypatch.setitem(sys.modules, "deeptutor.review.jobs", jobs)
    return jobs


def _chapter(name: str = "第1章_绪论") -> list[Any]:
    return [types.SimpleNamespace(name=name, doc_count=3)]


def test_generate_without_kb_is_a_soft_failure() -> None:
    payload = _payload(_run(ReviewGenerateTool()), tool="review_generate")
    assert payload["error"] == "kb is required"
    assert "knowledge base" in payload["instruction"]


def test_generate_lists_chapters_when_chapter_is_omitted(
    generate_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_infer(kb_name: str) -> list[Any]:
        calls.append(kb_name)
        return [
            types.SimpleNamespace(name="第1章_绪论", doc_count=3),
            types.SimpleNamespace(name="实验一_高频小信号谐振放大器", doc_count=2),
        ]

    monkeypatch.setattr(generate_module, "infer_chapters", fake_infer)
    payload = _payload(_run(ReviewGenerateTool(), kb="hf"), tool="review_generate")
    assert calls == ["hf"]
    assert payload["mode"] == "chapters"
    assert payload["chapters"] == [
        {"name": "第1章_绪论", "doc_count": 3},
        {"name": "实验一_高频小信号谐振放大器", "doc_count": 2},
    ]
    assert "review_generate" in payload["instruction"]


def test_generate_soft_fails_for_an_unknown_knowledge_base(
    generate_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_infer(kb_name: str) -> list[Any]:
        raise FileNotFoundError(kb_name)

    monkeypatch.setattr(generate_module, "infer_chapters", fake_infer)
    payload = _payload(
        _run(ReviewGenerateTool(), kb="missing-kb"), tool="review_generate"
    )
    assert payload["kb"] == "missing-kb"
    assert payload["mode"] == "chapters"
    assert "missing-kb" in payload["error"]


def test_generate_start_soft_fails_for_an_unknown_knowledge_base(
    generate_module: types.ModuleType,
    jobs_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_infer(kb_name: str) -> list[Any]:
        raise FileNotFoundError(kb_name)

    started: list[dict[str, Any]] = []

    def fake_start(**kwargs: Any) -> dict[str, Any]:
        started.append(kwargs)
        return {"job_id": "gen-1-abcd", "status": "queued"}

    monkeypatch.setattr(generate_module, "infer_chapters", fake_infer)
    monkeypatch.setattr(jobs_module, "start_generation_job", fake_start)
    payload = _payload(
        _run(ReviewGenerateTool(), kb="missing-kb", chapter="第1章_绪论"),
        tool="review_generate",
    )
    assert payload["error"] == "knowledge base not found: 'missing-kb'"
    assert started == []


def test_generate_starts_a_background_job(
    generate_module: types.ModuleType,
    jobs_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_infer(kb_name: str) -> list[Any]:
        return _chapter("第2章_谐振")

    def fake_start(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"job_id": "gen-123-abcd", "status": "queued"}

    monkeypatch.setattr(generate_module, "infer_chapters", fake_infer)
    monkeypatch.setattr(jobs_module, "start_generation_job", fake_start)
    payload = _payload(
        _run(
            ReviewGenerateTool(),
            kb="hf",
            chapter="第2章_谐振",
            num_questions=8,
            question_types=["choice"],
            difficulty="medium",
        ),
        tool="review_generate",
    )
    assert seen == {
        "kb": "hf",
        "chapter": "第2章_谐振",
        "num_questions": 8,
        "question_types": ["choice"],
        "difficulty": "medium",
        "replace": False,
    }
    assert payload["job_id"] == "gen-123-abcd"
    assert payload["status"] == "queued"
    assert payload["chapter"] == "第2章_谐振"
    assert "job_id" in payload["instruction"]
    assert "20" in payload["instruction"]  # poll cadence


def test_generate_clamps_counts_and_drops_unknown_question_types(
    generate_module: types.ModuleType,
    jobs_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_infer(kb_name: str) -> list[Any]:
        return _chapter("第1章")

    def fake_start(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"job_id": "gen-1-abcd", "status": "queued"}

    monkeypatch.setattr(generate_module, "infer_chapters", fake_infer)
    monkeypatch.setattr(jobs_module, "start_generation_job", fake_start)
    _payload(
        _run(
            ReviewGenerateTool(),
            kb="hf",
            chapter="第1章",
            num_questions=99,
            question_types=["bogus", "written", "written"],
        ),
        tool="review_generate",
    )
    assert seen["num_questions"] == 20
    assert seen["question_types"] == ["written"]

    seen.clear()
    _payload(
        _run(
            ReviewGenerateTool(),
            kb="hf",
            chapter="第1章",
            num_questions="many",
            question_types="choice",
        ),
        tool="review_generate",
    )
    assert seen["num_questions"] == 12
    assert seen["question_types"] == ["choice"]


def test_generate_soft_fails_when_the_chapter_does_not_match(
    generate_module: types.ModuleType,
    jobs_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_infer(kb_name: str) -> list[Any]:
        return _chapter("第1章_绪论")

    monkeypatch.setattr(generate_module, "infer_chapters", fake_infer)
    payload = _payload(
        _run(ReviewGenerateTool(), kb="hf", chapter="不存在的章"),
        tool="review_generate",
    )
    assert "no chapter matching" in payload["error"]
    assert payload["chapters"] == ["第1章_绪论"]
    assert "review_generate" in payload["instruction"]


def test_generate_fails_when_the_job_store_breaks(
    generate_module: types.ModuleType,
    jobs_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_infer(kb_name: str) -> list[Any]:
        return _chapter("第1章")

    def fake_start(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("disk full")

    monkeypatch.setattr(generate_module, "infer_chapters", fake_infer)
    monkeypatch.setattr(jobs_module, "start_generation_job", fake_start)
    result = _run(ReviewGenerateTool(), kb="hf", chapter="第1章")
    assert result.success is False
    assert "review_generate failed" in result.content


def test_generate_rejects_chapter_and_job_id_together(
    jobs_module: types.ModuleType,
) -> None:
    payload = _payload(
        _run(
            ReviewGenerateTool(), kb="hf", chapter="第1章", job_id="gen-1-abcd"
        ),
        tool="review_generate",
    )
    assert "not both" in payload["error"]
    assert "job_id" in payload["instruction"]


def _poll(
    jobs_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    job: dict[str, Any] | None,
    kb: str = "hf",
) -> dict[str, Any]:
    monkeypatch.setattr(jobs_module, "load_job", lambda job_id: job)
    return _payload(
        _run(ReviewGenerateTool(), kb=kb, job_id="gen-1-abcd"),
        tool="review_generate",
    )


def test_poll_reports_running(
    jobs_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _poll(
        jobs_module,
        monkeypatch,
        {
            "job_id": "gen-1-abcd",
            "kb": "hf",
            "chapter": "第1章",
            "status": "running",
            "created_at": time.time(),
            "updated_at": time.time(),
        },
    )
    assert payload["status"] == "running"
    assert payload["elapsed_seconds"] >= 0
    assert "Still generating" in payload["instruction"]
    assert "stale" not in payload


def test_poll_reports_done_with_the_stored_pack(
    jobs_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _poll(
        jobs_module,
        monkeypatch,
        {
            "job_id": "gen-1-abcd",
            "kb": "hf",
            "chapter": "第1章",
            "status": "done",
            "spec_id": "kb-deadbeef00-1",
            "qa_count": 6,
            "skipped": 0,
            "qa_passed": True,
            "counts": {"concepts": 3, "sections": 3, "items": 6},
            "created_at": time.time(),
            "updated_at": time.time(),
        },
    )
    assert payload["status"] == "done"
    assert payload["spec_id"] == "kb-deadbeef00-1"
    assert payload["qa_count"] == 6
    assert payload["qa_passed"] is True
    assert payload["counts"] == {"concepts": 3, "sections": 3, "items": 6}
    assert "review_today" in payload["instruction"]


def test_poll_recovers_a_legacy_done_record_from_the_stored_pack(
    seeded: ReviewService,
    jobs_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pre-schema records carried only status and spec_id; the pack they name
    # is still the ground truth.
    payload = _poll(
        jobs_module,
        monkeypatch,
        {
            "job_id": "gen-1-abcd",
            "kb": "hf",
            "chapter": "第1章",
            "status": "done",
            "spec_id": SPEC_ID,
            "created_at": time.time(),
            "updated_at": time.time(),
        },
    )
    assert payload["status"] == "done"
    assert payload["spec_id"] == SPEC_ID
    assert payload["qa_passed"] is True
    assert payload["qa_count"] == 3
    assert payload["counts"] == {"concepts": 2, "sections": 1, "items": 3}
    assert payload["legacy_record"] is True
    assert "read from the stored pack" in payload["instruction"]
    assert "review_today" in payload["instruction"]


def test_poll_refuses_a_legacy_done_record_whose_pack_is_missing(
    service: ReviewService,
    jobs_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _poll(
        jobs_module,
        monkeypatch,
        {
            "job_id": "gen-1-abcd",
            "kb": "hf",
            "chapter": "第1章",
            "status": "done",
            "spec_id": "fixture-test",
            "created_at": time.time(),
            "updated_at": time.time(),
        },
    )
    assert payload["error"] == "invalid_job_record"
    assert "review_specs" in payload["instruction"]


def test_poll_recovers_a_legacy_record_but_reports_a_failing_pack_truthfully(
    jobs_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Stub:
        @staticmethod
        def spec_meta(spec_id: str) -> dict[str, Any]:
            return {
                "spec_id": spec_id,
                "quality": {"passed": False, "warnings": ["G4: answer leaked"]},
                "counts": {"concepts": 2, "sections": 1, "items": 4},
            }

    monkeypatch.setattr(agent_tools, "_service", lambda: _Stub())
    payload = _poll(
        jobs_module,
        monkeypatch,
        {
            "job_id": "gen-1-abcd",
            "kb": "hf",
            "chapter": "第1章",
            "status": "done",
            "spec_id": "kb-deadbeef00-1",
            "created_at": time.time(),
            "updated_at": time.time(),
        },
    )
    assert payload["status"] == "done"
    assert payload["qa_passed"] is False
    assert payload["counts"]["items"] == 4
    assert payload["warnings"] == ["G4: answer leaked"]
    assert "does not pass the current quality gates" in payload["instruction"]
    assert "Do not tell the learner it passed" in payload["instruction"]


def test_poll_fails_as_internal_when_the_recovery_read_breaks(
    jobs_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Broken:
        @staticmethod
        def spec_meta(spec_id: str) -> dict[str, Any]:
            raise PermissionError("store outage")

    monkeypatch.setattr(agent_tools, "_service", lambda: _Broken())
    monkeypatch.setattr(
        jobs_module,
        "load_job",
        lambda job_id: {
            "job_id": "gen-1-abcd",
            "kb": "hf",
            "chapter": "第1章",
            "status": "done",
            "spec_id": "kb-deadbeef00-1",
            "created_at": time.time(),
            "updated_at": time.time(),
        },
    )
    result = _run(ReviewGenerateTool(), kb="hf", job_id="gen-1-abcd")
    assert result.success is False
    assert "review_generate failed" in result.content


def test_poll_reports_failure(
    jobs_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _poll(
        jobs_module,
        monkeypatch,
        {
            "job_id": "gen-1-abcd",
            "kb": "hf",
            "chapter": "第1章",
            "status": "failed",
            "error": "RuntimeError: model unavailable",
            "created_at": time.time(),
            "updated_at": time.time(),
        },
    )
    assert payload["status"] == "failed"
    assert "model unavailable" in payload["error"]
    assert "no pack was stored" in payload["instruction"]


def test_poll_flags_a_stale_job(
    jobs_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _poll(
        jobs_module,
        monkeypatch,
        {
            "job_id": "gen-1-abcd",
            "kb": "hf",
            "chapter": "第1章",
            "status": "running",
            "created_at": time.time() - 7200,
            "updated_at": time.time() - 7200,
        },
    )
    assert payload["stale"] is True
    assert "restarted" in payload["instruction"]


def test_poll_soft_fails_for_an_unknown_job(
    jobs_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _poll(jobs_module, monkeypatch, None)
    assert "unknown generation job" in payload["error"]
    assert "review_specs" in payload["instruction"]


def test_poll_fails_as_internal_when_the_store_errors(
    jobs_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(job_id: str) -> None:
        raise PermissionError("store outage")

    monkeypatch.setattr(jobs_module, "load_job", boom)
    result = _run(ReviewGenerateTool(), kb="hf", job_id="gen-1-abcd")
    assert result.success is False
    assert "review_generate failed" in result.content


def test_poll_rejects_a_mismatched_knowledge_base(
    jobs_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = {
        "job_id": "gen-1-abcd",
        "kb": "另一门课",
        "chapter": "第1章",
        "status": "running",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    payload = _poll(jobs_module, monkeypatch, job, kb="hf")
    assert "belongs to knowledge base" in payload["error"]
    assert "另一门课" in payload["instruction"]
