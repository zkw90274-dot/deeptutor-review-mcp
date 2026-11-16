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
        assert listed.inputSchema == definition.to_openai_schema()["function"][
            "parameters"
        ]

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

            today = _payload(await session.call_tool("review_today", {"spec_id": SPEC_ID}))
            assert today["items"], "fixture pack has no due item to review"
            item_id = today["items"][0]["item_id"]

            hint = _payload(
                await session.call_tool(
                    "review_hint",
                    {"spec_id": SPEC_ID, "item_id": item_id, "level": 1},
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
            assert attempt["rating"] == "good"

            weakness = _payload(
                await session.call_tool("review_weakness", {"spec_id": SPEC_ID})
            )
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
            return {
                "review_specs": specs,
                "review_today": today,
                "review_attempt": attempt,
                "review_weakness": weakness,
                "review_export_link": export,
                "review_export_tree": tree,
                "review_book_figures": figures,
                "review_hint": hint,
                "review_plan": plan,
            }

    payloads = _run(scenario)
    # ``review_generate`` calls the real LLM, so this local-only sweep skips
    # it; its MCP wiring is covered by the TOOL_CLASSES/names equality test
    # and its behaviour by tests/test_agent_tools.py with a fake generator.
    assert set(payloads) | {"review_generate"} == set(REVIEW_AGENT_TOOL_NAMES)
    for name, payload in payloads.items():
        assert payload.get("instruction"), f"{name} lost its instruction field"
    assert payloads["review_export_link"]["url"].endswith(f"/{SPEC_ID}/export")


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
    assert build_instructions() == INSTRUCTIONS.strip()

def test_instructions_carry_the_plan_naming_rule() -> None:
    """Plan tasks follow the material; the rule travels with the wheel."""
    from deeptutor.review.mcp_server import INSTRUCTIONS

    assert "part i/n" in INSTRUCTIONS
    assert "(一)(二)(三)" in INSTRUCTIONS
    assert "reading tasks after the material" in INSTRUCTIONS
    assert "复习包：<chapter>" in INSTRUCTIONS


def test_instructions_rule_out_a_separate_book_pointer() -> None:
    """The heading names the book chapter, so a pointer line is noise."""
    from deeptutor.review.mcp_server import INSTRUCTIONS

    assert "book-sections.json" in INSTRUCTIONS
    assert "Never add a separate pointer line" in INSTRUCTIONS


def test_instructions_carry_the_material_style() -> None:
    """One source per subject, a living book, and the Review Studio skin."""
    from deeptutor.review.mcp_server import INSTRUCTIONS

    assert "One main study document per subject" in INSTRUCTIONS
    assert "interactive living book" in INSTRUCTIONS
    assert "Never hand-edit the built HTML" in INSTRUCTIONS
    assert "#0f766e" in INSTRUCTIONS and "#5ec8bd" in INSTRUCTIONS
    assert "s<chapter>-<n>" in INSTRUCTIONS
    # Still deployment-neutral: no local directories or counts.
    assert "/opt/course-notes" not in INSTRUCTIONS


def test_instructions_carry_the_book_performance_rules() -> None:
    """A dense book must typeset and lay out lazily, or it stalls on open."""
    from deeptutor.review.mcp_server import INSTRUCTIONS

    assert "startup: { typeset: false }" in INSTRUCTIONS
    assert "content-visibility: auto" in INSTRUCTIONS
    assert "contain-intrinsic-size" in INSTRUCTIONS
    assert 'behavior "instant"' in INSTRUCTIONS
    assert "A print control" in INSTRUCTIONS
    assert "long task over 100ms" in INSTRUCTIONS
    # The three rules that make the lazy path actually cheap.
    assert "IntersectionObserver" in INSTRUCTIONS
    assert "900px" in INSTRUCTIONS
    assert "160ms" in INSTRUCTIONS
    assert "the DOM queries" in INSTRUCTIONS
    # The rules stay generic: no book names, no local paths.
    assert "secret-subject" not in INSTRUCTIONS and "/opt/course-notes" not in INSTRUCTIONS
