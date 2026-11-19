from __future__ import annotations

import asyncio
import json
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import ValidationError, validate
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import AnyUrl

from deeptutor.review import agent_tools, core_state, export, jobs, tikz
from deeptutor.review.mcp_server import build_server
from deeptutor.review.models import FigureSpec
from deeptutor.review.storage import ReviewStore


_MANIFEST = Path(__file__).resolve().parents[1] / "src/deeptutor/modules/manifests/review.json"


class _OfflinePage(HTMLParser):
    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in {"src", "href"} and value:
                assert value.startswith(("#", "data:")), (tag, name, value)


def test_structured_output_matches_legacy_text(service, fixture_spec):
    service.put_spec(fixture_spec)
    names = {"review_quality", "review_metrics", "review_artifact_status"}
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))["mcp"]
    assert set(manifest["structured_output_tools"]) == names

    async def scenario():
        async with create_connected_server_and_client_session(build_server()) as session:
            tools = (await session.list_tools()).tools
            assert len(tools) == 20
            assert {tool.name for tool in tools if tool.outputSchema} == names
            for tool in tools:
                if tool.name not in names:
                    continue
                for pack_id in (fixture_spec.id, "missing-pack"):
                    result = await session.call_tool(tool.name, {"spec_id": pack_id})
                    assert not result.isError, result.content
                    assert result.structuredContent == json.loads(result.content[0].text)
                    validate(result.structuredContent, tool.outputSchema)
                with pytest.raises(ValidationError):
                    validate({}, tool.outputSchema)

    asyncio.run(scenario())


def test_resources_export_offline_with_zero_writes(service, linked_spec, tmp_path, monkeypatch):
    spec = linked_spec.model_copy(update={"figures": [*linked_spec.figures, FigureSpec(
        id="tikz-probe", title="TikZ probe", kind="tikz", tikz=r"\draw (0,0) -- (1,1);",
    )], "content_hash": ""})
    service.put_spec(spec)
    index = {"subjects": {"hf": {"chapter": {"figures": []}}}}
    (service.store.home / "book-figures.json").write_text(json.dumps(index), encoding="utf-8")
    monkeypatch.setenv("DEEPTUTOR_REVIEW_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("DEEPTUTOR_REVIEW_READ_ONLY", "1")

    def deny(*args, **kwargs):
        pytest.fail("read_resource attempted a writable operation")

    monkeypatch.setattr(ReviewStore, "_transaction", deny)
    monkeypatch.setattr(ReviewStore, "_migrate", deny)
    monkeypatch.setattr(core_state, "load_core_progress", deny)
    monkeypatch.setattr(tikz, "render_tikz_document", deny)

    async def scenario():
        async with create_connected_server_and_client_session(build_server()) as session:
            resources = (await session.list_resources()).resources
            assert [(str(resource.uri), resource.mimeType) for resource in resources] == [
                ("review://book-figures", "application/json"),
            ]
            templates = (await session.list_resource_templates()).resourceTemplates
            assert [(template.uriTemplate, template.mimeType) for template in templates] == [
                ("review://spec/{spec_id}/export", "text/html"),
            ]
            manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))["mcp"]
            assert manifest["resources"] == {str(resource.uri): resource.mimeType for resource in resources}
            assert manifest["resource_templates"] == {template.uriTemplate: template.mimeType for template in templates}
            figures, = (await session.read_resource(AnyUrl("review://book-figures"))).contents
            assert figures.mimeType == "application/json" and json.loads(figures.text) == index
            page, = (await session.read_resource(AnyUrl(f"review://spec/{spec.id}/export"))).contents
            assert page.mimeType == "text/html" and "<!doctype html>" in page.text.lower()
            assert "Content-Security-Policy" in page.text and "default-src 'none'" in page.text
            _OfflinePage().feed(page.text)
            assert "TikZ rendering requires disk access" in page.text
            for uri in ("review://unknown", "review://book-figures?other=1",
                        "review://spec/missing-pack/export", f"review://spec/{spec.id}/export#fragment"):
                with pytest.raises(McpError) as exc:
                    await session.read_resource(AnyUrl(uri))
                assert exc.value.error.code == -32002

    before = service.store.db_path.read_bytes()
    asyncio.run(scenario())
    assert service.store.db_path.read_bytes() == before
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "core-home").exists()


def test_resource_rejects_large_and_rejected_packs(service, fixture_spec, monkeypatch):
    service.put_spec(fixture_spec)
    uri = AnyUrl(f"review://spec/{fixture_spec.id}/export")

    async def read_error():
        async with create_connected_server_and_client_session(build_server()) as session:
            with pytest.raises(McpError) as exc:
                await session.read_resource(uri)
            return exc.value.error.message

    monkeypatch.setattr(export, "MAX_BYTES", 100)
    assert "size limit" in asyncio.run(read_error())
    service.store.save_quality(fixture_spec, {"state": "rejected"}, {})
    assert "quality gates" in asyncio.run(read_error())


@pytest.mark.parametrize("entry", ["start", "poll"])
@pytest.mark.parametrize("outcome", ["done", "failed", "timeout"])
def test_generation_wait_returns_existing_status_payload(monkeypatch, entry, outcome):
    tool = agent_tools.ReviewGenerateTool()
    queued = {"job_id": "stub-job", "kb": "hf", "chapter": "Chapter A", "status": "queued"}
    calls = {"start": 0, "load": 0}

    def start(**kwargs):
        calls["start"] += 1
        return queued

    def load(job_id):
        assert job_id == "stub-job"
        calls["load"] += 1
        if calls["load"] == 1 or outcome == "timeout":
            return queued
        if outcome == "failed":
            return {**queued, "status": "failed", "error": "synthetic job failure"}
        return {**queued, "status": "done", "spec_id": "stub-pack", "qa_passed": True,
                "qa_count": 5, "counts": {"items": 5}}

    monkeypatch.setattr(jobs, "start_generation_job", start)
    monkeypatch.setattr(jobs, "load_job", load)
    monkeypatch.setattr(jobs, "is_stale", lambda job: False)
    monkeypatch.setattr(agent_tools, "_generate_module", lambda: SimpleNamespace(
        infer_chapters=lambda kb: [SimpleNamespace(name="Chapter A")],
    ))

    async def scenario():
        async with create_connected_server_and_client_session(build_server()) as session:
            generate = next(item for item in (await session.list_tools()).tools if item.name == "review_generate")
            wait = generate.inputSchema["properties"]["wait_seconds"]
            assert {key: wait[key] for key in ("type", "minimum", "maximum")} == {
                "type": "integer", "minimum": 0, "maximum": 60,
            }
            assert "wait_seconds" not in generate.inputSchema["required"]
            params = {"chapter": "Chapter A"} if entry == "start" else {"job_id": "stub-job"}
            response = await asyncio.wait_for(
                session.call_tool("review_generate", {"kb": "hf", "wait_seconds": 1, **params}),
                timeout=3,
            )
            assert not response.isError, response.content
            return json.loads(response.content[0].text)

    result = asyncio.run(scenario())
    assert calls["start"] == (entry == "start") and calls["load"] >= 2
    assert result["job_id"] == "stub-job"
    assert result["status"] == ("queued" if outcome == "timeout" else outcome)
    if outcome == "done":
        assert result["qa_passed"] and result["spec_id"] == "stub-pack"
    elif outcome == "failed":
        assert result["error"] == "synthetic job failure"
    else:
        assert "~20" in result["instruction"]
        if entry == "start":
            assert result["reused"] is False
    assert "wait_seconds" in {parameter.name for parameter in tool.get_definition().parameters}


@pytest.mark.parametrize("value", [-1, 61, True, 1.5, "2", None])
def test_generation_rejects_invalid_wait(value):
    result = asyncio.run(agent_tools.ReviewGenerateTool().execute(kb="hf", wait_seconds=value))
    assert json.loads(result.content)["error"] == "invalid_wait_seconds"


def test_quality_decisions_verify_reject_and_reset_preserve_pack_and_events(service, fixture_spec):
    service.store.put_spec(fixture_spec, assessment=({"state": "legacy"}, {}))
    original_hash = service.store.spec_meta(fixture_spec.id)["content_hash"]
    preview = service.today(fixture_spec.id)
    service.attempt(
        fixture_spec.id,
        agent_tools.AttemptInput(
            item_id=preview["items"][0]["item_id"],
            rating="good",
            session_id=preview["session_id"],
        ),
    )
    original_events = service.store.event_count(fixture_spec.id)
    assert original_events == 1

    verified = service.quality_decision(
        fixture_spec.id, decision="verify", note="人工检查通过", reviewer="reviewer-a",
    )
    assert verified["state"] == "verified"
    assert verified["reviewed_by"] == "reviewer-a"
    assert verified["review_note"] == "人工检查通过"
    assert verified["reviewed_at"] is not None
    assert service.today(fixture_spec.id)["items"]

    rejected = service.quality_decision(
        fixture_spec.id, decision="reject", note="证据不足", reviewer="reviewer-b",
    )
    assert rejected["state"] == "rejected"
    assert service.today(fixture_spec.id)["items"] == []
    _, rejected_page = service.spec_page(limit=50, offset=0)
    assert next(row for row in rejected_page if row["spec_id"] == fixture_spec.id)["quality_state"] == "rejected"
    with pytest.raises(Exception, match="QA gates"):
        service.export(fixture_spec.id)

    reset = service.quality_decision(
        fixture_spec.id, decision="reset", reviewer="reviewer-c",
    )
    assert reset["state"] == "legacy"
    assert reset["reviewed_by"] is None
    assert reset["reviewed_at"] is None
    assert reset["review_note"] is None
    with service.store._read_connection() as conn:
        decisions = conn.execute(
            "SELECT decision, reviewer, note FROM review_quality_decisions "
            "WHERE spec_id = ? ORDER BY ts, rowid",
            (fixture_spec.id,),
        ).fetchall()
    assert [tuple(row) for row in decisions] == [
        ("verify", "reviewer-a", "人工检查通过"),
        ("reject", "reviewer-b", "证据不足"),
        ("reset", "reviewer-c", None),
    ]
    assert service.store.spec_meta(fixture_spec.id)["content_hash"] == original_hash
    assert service.store.event_count(fixture_spec.id) == original_events


def test_quality_decision_reset_restores_assessed_draft(service, fixture_spec):
    service.put_spec(fixture_spec)
    assessed = service.quality(fixture_spec.id)["state"]
    assert assessed == "draft"
    assert service.today(fixture_spec.id)["items"] == []
    service.quality_decision(
        fixture_spec.id, decision="verify", note="human review", reviewer="reviewer",
    )
    assert service.today(fixture_spec.id)["items"]
    reviewed = service.quality(fixture_spec.id, refresh=True)
    assert reviewed["state"] == "verified"
    assert reviewed["reviewed_by"] == "reviewer"
    assert reviewed["review_note"] == "human review"
    reset = service.quality_decision(fixture_spec.id, decision="reset", reviewer="reviewer")
    assert reset["state"] == "draft"


def test_quality_decision_tool_contract_and_read_only_guard(service, fixture_spec, monkeypatch):
    from deeptutor.review.mcp_server import TOOL_ANNOTATIONS, TOOL_CLASSES

    tool = TOOL_CLASSES["review_quality_decision"]()
    definition = tool.get_definition()
    assert definition.name == "review_quality_decision"
    assert TOOL_ANNOTATIONS[definition.name] == {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
    service.store.put_spec(fixture_spec, assessment=({"state": "legacy"}, {}))

    async def call():
        return await tool.execute(spec_id=fixture_spec.id, decision="verify", note="checked")

    result = asyncio.run(call())
    assert result.success
    payload = json.loads(result.content)
    assert payload["quality_state"] == "verified"
    assert payload["review_note"] == "checked"

    monkeypatch.setenv("DEEPTUTOR_REVIEW_READ_ONLY", "1")
    blocked = asyncio.run(call())
    assert json.loads(blocked.content)["error"] == "read_only"
