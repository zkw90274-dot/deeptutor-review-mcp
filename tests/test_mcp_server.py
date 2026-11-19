"""WS-1 acceptance: the Review Studio MCP server.

The suite drives the real low-level ``Server`` through the SDK's in-memory
transport, so ``list_tools`` and ``call_tool`` run exactly as they do for a
stdio or Streamable HTTP client — without opening a port or spawning a child
process.

The schema test is deliberately a *source* comparison: the listed MCP tools
must be the agent tools' own ``get_definition()`` output, not a hand-written
copy that can drift. The call tests use an isolated ``DEEPTUTOR_REVIEW_HOME``
and a seeded fixture pack, so they never touch a learner's real workspace.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult

from deeptutor.core.tool_protocol import ToolResult
from deeptutor.review.agent_tools import (
    REVIEW_AGENT_TOOL_NAMES,
    ReviewSpecsTool,
)
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.mcp_server import (
    TOOL_CLASSES,
    build_server,
    create_http_app,
)
from deeptutor.review.service import ReviewService


@pytest.fixture()
def review_home(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point the review store (and the core store it may read) at tmp_path."""
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(tmp_path / "review"))
    monkeypatch.setenv("DEEPTUTOR_HOME", str(tmp_path / "core-home"))
    return tmp_path / "review"


def _seed_fixture() -> None:
    ReviewService().put_spec(build_fixture_spec())


def _payload(result: CallToolResult) -> dict[str, Any]:
    """A successful tool call: non-empty text, parsed as a JSON object."""
    assert result.isError is False, result.content
    assert result.content, "tool returned no content"
    text = result.content[0].text
    assert text.strip(), "tool returned blank text"
    payload = json.loads(text)
    assert isinstance(payload, dict)
    return payload


def _run(scenario):
    return asyncio.run(scenario())


def test_each_listed_tool_carries_annotations() -> None:
    """Every tool advertises read/write risk; clients gate on these hints.

    The mapping is deliberately not derived from the tool names: ``review_today``
    is a write (session + views), ``review_attempt`` is only idempotent when the
    caller supplies ``submission_id``, and ``review_generate`` can overwrite an
    existing pack. Pin all three so a future "cleanup" cannot quietly re-label
    them.
    """
    from deeptutor.review.mcp_server import TOOL_ANNOTATIONS

    assert set(TOOL_ANNOTATIONS) == set(TOOL_CLASSES)

    async def scenario():
        async with create_connected_server_and_client_session(build_server()) as session:
            return (await session.list_tools()).tools

    tools = _run(scenario)
    assert [tool.name for tool in tools] == list(REVIEW_AGENT_TOOL_NAMES)
    for listed in tools:
        assert listed.annotations is not None, listed.name
        expected = TOOL_ANNOTATIONS[listed.name]
        for key, value in expected.items():
            assert getattr(listed.annotations, key) is value, (listed.name, key)

    read_only = {
        tool.name for tool in tools if tool.annotations.readOnlyHint
    }
    assert read_only == {
        "review_specs",
        "review_due",
        "review_weakness",
        "review_export_link",
        "review_book_figures",
        "review_metrics",
        "review_session_plan",
        "review_artifact_status",
    }
    today = next(tool for tool in tools if tool.name == "review_today")
    assert today.annotations.readOnlyHint is False
    tree = next(tool for tool in tools if tool.name == "review_export_tree")
    assert tree.annotations.destructiveHint is True
    # Byte-stable for the same store state; the cleanup side effect is already
    # covered by destructiveHint.
    assert tree.annotations.idempotentHint is True

    attempt = next(tool for tool in tools if tool.name == "review_attempt")
    # Without submission_id every call appends an event, so a client must not
    # auto-retry it as if it were idempotent.
    assert attempt.annotations.idempotentHint is False

    generate = next(tool for tool in tools if tool.name == "review_generate")
    # replace=True overwrites the pack and rewrites history protection.
    assert generate.annotations.destructiveHint is True
    assert generate.annotations.openWorldHint is True


def test_listed_tools_are_generated_from_agent_definitions() -> None:
    """Name, description and every schema field come from ``get_definition``."""
    assert tuple(TOOL_CLASSES) == REVIEW_AGENT_TOOL_NAMES
    definitions = {
        name: tool_type().get_definition() for name, tool_type in TOOL_CLASSES.items()
    }

    async def scenario():
        async with create_connected_server_and_client_session(build_server()) as session:
            return (await session.list_tools()).tools

    tools = _run(scenario)
    assert [tool.name for tool in tools] == list(REVIEW_AGENT_TOOL_NAMES)

    for listed in tools:
        definition = definitions[listed.name]
        assert listed.description == definition.description
        assert listed.inputSchema == {
            **definition.to_openai_schema()["function"]["parameters"],
            "additionalProperties": False,
        }

    # Pin the async-generation parameter explicitly: if the definition and
    # the MCP schema ever drifted together, the equality above would still
    # pass, so assert the contract directly.
    generate = next(tool for tool in tools if tool.name == "review_generate")
    assert "job_id" in generate.inputSchema["properties"]
    assert generate.inputSchema["required"] == ["kb"]


def test_each_tool_call_returns_json(review_home) -> None:
    """Every non-generating tool is callable once and returns JSON text.

    ``review_generate`` is exercised separately (its chapter-listing mode)
    because starting a real generation job needs the LLM pipeline.
    """
    _seed_fixture()

    async def scenario() -> dict[str, dict[str, Any]]:
        async with create_connected_server_and_client_session(build_server()) as session:
            specs = _payload(await session.call_tool("review_specs", {}))
            assert [row["id"] for row in specs["specs"]] == [SPEC_ID]

            today = _payload(await session.call_tool("review_today", {"spec_id": SPEC_ID, "include_draft": True}))
            assert today["items"], "fixture pack has no due item to review"
            item_id = today["items"][0]["item_id"]

            hint = _payload(
                await session.call_tool(
                    "review_hint",
                    {"spec_id": SPEC_ID, "item_id": item_id, "level": 1, "session_id": today["session_id"]},
                )
            )
            assert hint["item_id"] == item_id
            assert hint["hint"]

            attempt = _payload(
                await session.call_tool(
                    "review_attempt",
                    {
                        "spec_id": SPEC_ID,
                        "item_id": item_id,
                        "rating": "good",
                        "session_id": today["session_id"],
                    },
                )
            )
            assert attempt["item_id"] == item_id
            assert attempt["rating"] == "hard"
            assert "answer" not in attempt
            feedback = _payload(await session.call_tool("review_reveal", {"spec_id": SPEC_ID, "event_id": attempt["event_id"]}))
            assert feedback["answer"]

            weakness = _payload(
                await session.call_tool("review_weakness", {"spec_id": SPEC_ID})
            )
            due = _payload(await session.call_tool("review_due", {"include_draft": True}))
            assert due["count"] == 1
            assert due["packs"][0]["spec_id"] == SPEC_ID
            export = _payload(
                await session.call_tool("review_export_link", {"spec_id": SPEC_ID})
            )
            tree = _payload(
                await session.call_tool(
                    "review_export_tree", {"out_dir": str(review_home / "exports")}
                )
            )
            figures = _payload(
                await session.call_tool("review_book_figures", {"subject": "nope"})
            )
            plan = _payload(await session.call_tool("review_plan", {"action": "status"}))
            assert plan["plans"] == []
            quality = _payload(await session.call_tool("review_quality", {"spec_id": SPEC_ID}))
            assert quality["state"] == "draft"
            metrics = _payload(await session.call_tool("review_metrics", {"spec_id": SPEC_ID}))
            assert metrics["objective_accuracy"]["judged"] == 0
            ended = _payload(await session.call_tool("review_session", {
                "action": "end", "spec_id": SPEC_ID, "session_id": today["session_id"],
            }))
            assert ended["completed"] == 1
            corrected = _payload(await session.call_tool("review_correct", {
                "spec_id": SPEC_ID, "item_id": item_id,
                "patch": {"stem": today["items"][0]["stem"] + " 请说明理由。"},
            }))
            assert corrected["revision"] == 1
            links = _payload(await session.call_tool("review_link", {"action": "list", "spec_id": SPEC_ID}))
            assert links["linked_concepts"] == 0
            mixed = _payload(await session.call_tool("review_session_plan", {
                "spec_ids": [SPEC_ID], "include_draft": True, "limit": 2,
            }))
            assert 0 < len(mixed["items"]) <= 2
            artifacts = _payload(await session.call_tool("review_artifact_status", {"spec_id": SPEC_ID}))
            assert artifacts["count"] == 1
            assert artifacts["items"][0]["status"] == "unknown"
            quality_decision = _payload(await session.call_tool("review_quality_decision", {
                "spec_id": SPEC_ID, "decision": "verify", "note": "test approval",
            }))
            assert quality_decision["quality_state"] == "verified"
            specs_after_decision = _payload(await session.call_tool("review_specs", {}))
            assert specs_after_decision["specs"][0]["quality_state"] == "verified"
            return {
                "review_specs": specs,
                "review_due": due,
                "review_today": today,
                "review_attempt": attempt,
                "review_weakness": weakness,
                "review_export_link": export,
                "review_export_tree": tree,
                "review_book_figures": figures,
                "review_hint": hint,
                "review_plan": plan,
                "review_reveal": feedback,
                "review_quality": quality,
                "review_quality_decision": quality_decision,
                "review_correct": corrected,
                "review_metrics": metrics,
                "review_session": ended,
                "review_link": links,
                "review_session_plan": mixed,
                "review_artifact_status": artifacts,
            }

    payloads = _run(scenario)
    # ``review_generate`` calls the real LLM, so this local-only sweep skips
    # it; its MCP wiring is covered by the TOOL_CLASSES/names equality test
    # and its behaviour by tests/test_agent_tools.py with a fake generator.
    assert set(payloads) | {"review_generate"} == set(REVIEW_AGENT_TOOL_NAMES)
    for name, payload in payloads.items():
        assert payload.get("instruction"), f"{name} lost its instruction field"
    assert payloads["review_export_link"]["url"].endswith(f"/{SPEC_ID}/export")


def test_unknown_argument_is_an_mcp_error() -> None:
    async def scenario():
        async with create_connected_server_and_client_session(build_server()) as session:
            return await session.call_tool("review_due", {"subjects": ["sample"]})

    result = _run(scenario)
    assert result.isError
    assert "subjects" in result.content[0].text


def test_unknown_tool_is_an_mcp_error() -> None:
    """A name the server does not serve surfaces as ``isError``, not silence."""

    async def scenario():
        async with create_connected_server_and_client_session(build_server()) as session:
            return await session.call_tool("review_nope", {})

    result = _run(scenario)
    assert result.isError is True
    assert result.content
    assert "review_nope" in result.content[0].text


def test_unknown_spec_is_a_soft_failure(review_home) -> None:
    """A recoverable miss stays a successful call carrying error + instruction."""
    _seed_fixture()

    async def scenario():
        async with create_connected_server_and_client_session(build_server()) as session:
            return await session.call_tool(
                "review_attempt",
                {
                    "spec_id": "spec-that-does-not-exist",
                    "item_id": "item-x",
                    "rating": "good",
                },
            )

    payload = _payload(_run(scenario))
    assert payload["error"] == "spec not found: spec-that-does-not-exist"
    assert payload["instruction"]


def test_failed_tool_maps_to_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ToolResult.success=False`` is the only path that sets ``isError``."""

    class FailingSpecsTool(ReviewSpecsTool):
        async def execute(self, **kwargs: Any) -> ToolResult:
            return ToolResult(content="store unavailable", success=False)

    monkeypatch.setitem(TOOL_CLASSES, "review_specs", FailingSpecsTool)

    async def scenario():
        async with create_connected_server_and_client_session(build_server()) as session:
            return await session.call_tool("review_specs", {})

    result = _run(scenario)
    assert result.isError is True
    assert result.content[0].text == "store unavailable"


def test_http_app_exposes_the_configured_path() -> None:
    """The hosted transport mounts the MCP endpoint where the CLI says it does."""
    app = create_http_app(build_server(), path="/custom-mcp")
    paths = [route.path for route in app.routes if hasattr(route, "path")]
    assert "/custom-mcp" in paths

def test_instructions_are_deployment_neutral() -> None:
    """A published wheel must not carry one learner's paths or counts."""
    from deeptutor.review.mcp_server import INSTRUCTIONS

    assert "DEEPTUTOR_REVIEW_HOME" in INSTRUCTIONS
    assert "/opt/course-notes" not in INSTRUCTIONS
    assert len(INSTRUCTIONS) <= 2800
    assert "{skill}" in INSTRUCTIONS


def test_extra_context_appended_from_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators add local material through a note, not by editing the wheel."""
    from deeptutor.review.mcp_server import build_instructions

    note = tmp_path / "ctx.md"
    note.write_text("本机目录：/opt/course-notes", encoding="utf-8")
    monkeypatch.setenv("DEEPTUTOR_REVIEW_MCP_CONTEXT", str(note))
    text = build_instructions()
    assert text.endswith("本机目录：/opt/course-notes")
    assert "DEEPTUTOR_REVIEW_HOME" in text


def test_extra_context_missing_file_is_ignored(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale env path degrades to the generic text instead of failing."""
    from deeptutor.review.mcp_server import INSTRUCTIONS, build_instructions

    monkeypatch.setenv("DEEPTUTOR_REVIEW_MCP_CONTEXT", str(tmp_path / "absent.md"))
    instructions = build_instructions()
    assert instructions.startswith(INSTRUCTIONS.split("{skill}")[0].format())
    assert "review://spec/{spec_id}/export" in instructions
    assert "SKILL.md" in instructions
    assert "---" not in instructions

def _skill_text() -> str:
    from _layout import skill_dir

    return (skill_dir() / "SKILL.md").read_text(encoding="utf-8")


def test_skill_doc_carries_the_plan_naming_rule() -> None:
    """The detail moved out of the handshake, not out of the product."""
    text = _skill_text()
    assert "第 i/n 段" in text
    assert "（一）（二）（三）" in text
    assert "读什么" in text and "时长切成段" in text
    assert "「复习包：」" in text


def test_skill_doc_rules_out_a_separate_book_pointer() -> None:
    text = _skill_text()
    assert "book-sections.json" in text
    assert "不再输出独立的" in text


def test_skill_doc_carries_the_material_style() -> None:
    """One source per subject, a living book, and the Review Studio skin."""
    text = _skill_text()
    assert "只有一份主资料" in text
    assert "单文件离线交互活书" in text
    assert "绝不手改产物 HTML" in text
    assert "#0f766e" in text and "#5ec8bd" in text
    assert "s<章>-<节>" in text


def test_skill_doc_carries_the_book_performance_rules() -> None:
    """A dense book must typeset and lay out lazily, or it stalls on open."""
    text = _skill_text()
    assert "startup: { typeset: false }" in text
    assert "content-visibility: auto" in text
    assert "contain-intrinsic-size" in text
    assert 'behavior: "instant"' in text
    assert "打印按钮" in text and "window.print()" in text
    assert "100ms" in text
    # The three rules that make the lazy path actually cheap.
    assert "IntersectionObserver" in text
    assert "900px" in text
    assert "160ms" in text
    assert "DOM 查询要缓存" in text
    # The rules stay generic: no book names, no local paths.
    assert "secret-subject" not in text and "/opt/course-notes" not in text
