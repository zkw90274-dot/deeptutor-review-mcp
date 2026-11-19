#!/usr/bin/env python
"""End-to-end smoke test: seed a pack, start the MCP server, call real tools.

    python examples/demo/smoke.py

Exits non-zero when the handshake, the tool listing or a tool call fails, so CI
proves the published entry point actually serves the documented surface.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

EXPECTED_TOOLS = (
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


def _seed(home: Path) -> str:
    os.environ["DEEPTUTOR_REVIEW_HOME"] = str(home)
    from deeptutor.review.fixture import build_fixture_spec
    from deeptutor.review.service import ReviewService

    return ReviewService().put_spec(build_fixture_spec())["spec_id"]


async def _run(home: Path, spec_id: str) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from pydantic import AnyUrl

    env = dict(os.environ)
    env["DEEPTUTOR_REVIEW_HOME"] = str(home)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "deeptutor.review.mcp_server"],
        env=env,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print("server   :", init.serverInfo.name, init.serverInfo.version)

            listed = await session.list_tools()
            names = tuple(tool.name for tool in listed.tools)
            print("tools    :", len(names))
            missing = [name for name in EXPECTED_TOOLS if name not in names]
            if missing or len(names) != len(EXPECTED_TOOLS):
                raise SystemExit(f"unexpected tools: {names}; missing: {missing}")

            specs = await session.call_tool("review_specs", {})
            payload = json.loads(specs.content[0].text)
            packs = payload.get("specs") or payload.get("packs") or []
            print("packs    :", [pack.get("id") for pack in packs])
            if not any(pack.get("id") == spec_id for pack in packs):
                raise SystemExit(f"seeded pack {spec_id!r} missing from review_specs")

            due = await session.call_tool("review_due", {"include_draft": True})
            overview = json.loads(due.content[0].text)
            print("due      :", overview.get("count"), "packs,", overview.get("totals", {}).get("queued"), "queued items")
            if not any(entry.get("spec_id") == spec_id for entry in overview.get("packs") or []):
                raise SystemExit(f"seeded pack {spec_id!r} missing from review_due")

            today = await session.call_tool("review_today", {"spec_id": spec_id, "include_draft": True})
            queue = json.loads(today.content[0].text)
            items = queue.get("items") or []
            print(
                "queue    :",
                len(items),
                "items, planned",
                queue.get("planned_minutes"),
                "min, budget",
                queue.get("budget"),
            )
            if not items:
                raise SystemExit("review_today returned an empty queue")

            item_args = {"spec_id": spec_id, "item_id": items[0]["item_id"], "session_id": queue["session_id"]}
            hint = await session.call_tool("review_hint", item_args)
            if hint.isError or not json.loads(hint.content[0].text).get("hint"):
                raise SystemExit("review_hint failed")
            attempt = await session.call_tool("review_attempt", {**item_args, "rating": "good", "hints_used": 0})
            recorded = json.loads(attempt.content[0].text)
            if attempt.isError or recorded.get("rating") != "hard" or "answer" in recorded:
                raise SystemExit("review_attempt did not apply the hint ledger and reveal policy")
            reveal = await session.call_tool("review_reveal", {"spec_id": spec_id, "event_id": recorded["event_id"]})
            if reveal.isError or not json.loads(reveal.content[0].text).get("answer"):
                raise SystemExit("review_reveal failed")
            print("feedback : hint -> hard -> reveal")
            links = await session.call_tool("review_link", {"action": "list", "spec_id": spec_id})
            listing = json.loads(links.content[0].text)
            if links.isError or listing.get("links") != [] or listing.get("total_concepts") != 2:
                raise SystemExit("review_link could not list the fixture pack's effective links")
            print("links    : fixture pack has no Mastery bindings")
            mixed = await session.call_tool("review_session_plan", {"include_draft": True, "limit": 2})
            plan = json.loads(mixed.content[0].text)
            if mixed.isError or not 0 < len(plan.get("items", [])) <= 2:
                raise SystemExit("review_session_plan could not preview the pack within the item limit")
            print("mixed    :", plan["selected"]["total"], "items within the shared budget")
            artifact_result = await session.call_tool("review_artifact_status", {"spec_id": spec_id})
            artifacts = json.loads(artifact_result.content[0].text)
            if artifact_result.isError or artifacts.get("count") != 1 or artifacts["items"][0]["status"] != "unknown":
                raise SystemExit("review_artifact_status did not report the fixture pack as unknown")
            if artifacts != artifact_result.structuredContent:
                raise SystemExit("review_artifact_status structured output differs from its text")
            print("artifact :", artifacts["items"][0]["status"])

            resource, = (await session.list_resources()).resources
            template, = (await session.list_resource_templates()).resourceTemplates
            if str(resource.uri) != "review://book-figures" or template.uriTemplate != "review://spec/{spec_id}/export":
                raise SystemExit("unexpected MCP resource listing")
            figures, = (await session.read_resource(resource.uri)).contents
            page, = (await session.read_resource(AnyUrl(template.uriTemplate.replace("{spec_id}", spec_id)))).contents
            if figures.mimeType != "application/json" or not isinstance(json.loads(figures.text), dict):
                raise SystemExit("invalid book figure index")
            if page.mimeType != "text/html" or not page.text.startswith("<!doctype html>"):
                raise SystemExit("invalid export resource")
            print("resources:", figures.mimeType, page.mimeType)

            print("OK")


def main() -> int:
    home = Path(tempfile.mkdtemp(prefix="deeptutor-review-smoke-"))
    spec_id = _seed(home)
    print("home     :", home)
    asyncio.run(_run(home, spec_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
