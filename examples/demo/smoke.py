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
)


def _seed(home: Path) -> str:
    os.environ["DEEPTUTOR_REVIEW_HOME"] = str(home)
    from deeptutor.review.fixture import build_fixture_spec
    from deeptutor.review.service import ReviewService

    return ReviewService().put_spec(build_fixture_spec())["spec_id"]


async def _run(home: Path, spec_id: str) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

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
            if missing:
                raise SystemExit(f"missing tools: {missing}")

            specs = await session.call_tool("review_specs", {})
            payload = json.loads(specs.content[0].text)
            packs = payload.get("specs") or payload.get("packs") or []
            print("packs    :", [pack.get("id") for pack in packs])
            if not any(pack.get("id") == spec_id for pack in packs):
                raise SystemExit(f"seeded pack {spec_id!r} missing from review_specs")

            due = await session.call_tool("review_due", {})
            overview = json.loads(due.content[0].text)
            print("due      :", overview.get("count"), "packs,", overview.get("totals", {}).get("queued"), "queued items")
            if not any(entry.get("spec_id") == spec_id for entry in overview.get("packs") or []):
                raise SystemExit(f"seeded pack {spec_id!r} missing from review_due")

            today = await session.call_tool("review_today", {"spec_id": spec_id})
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

            print("OK")


def main() -> int:
    home = Path(tempfile.mkdtemp(prefix="deeptutor-review-smoke-"))
    spec_id = _seed(home)
    print("home     :", home)
    asyncio.run(_run(home, spec_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
