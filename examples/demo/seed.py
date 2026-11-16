#!/usr/bin/env python
"""Seed a local review workspace with the built-in 3-question demo pack.

Run this once before wiring the MCP server, so the agent has a pack to read:

    python examples/demo/seed.py            # -> ./demo-review-home
    DEEPTUTOR_REVIEW_HOME=./demo-review-home deeptutor-review-mcp

The demo pack is the package's own Phase 0.5 fixture: 2 concepts, 3 items
(single choice, numeric, flashcard), no model call, no knowledge base needed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    home = Path(os.environ.get("DEEPTUTOR_REVIEW_HOME") or "./demo-review-home").resolve()
    home.mkdir(parents=True, exist_ok=True)
    os.environ["DEEPTUTOR_REVIEW_HOME"] = str(home)

    from deeptutor.review.fixture import build_fixture_spec
    from deeptutor.review.service import ReviewService

    result = ReviewService().put_spec(build_fixture_spec())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
    print("review home :", home)
    print("pack id     :", result["spec_id"])
    print("next        : DEEPTUTOR_REVIEW_HOME=%s deeptutor-review-mcp" % home)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
