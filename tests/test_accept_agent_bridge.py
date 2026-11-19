"""The agent-tool acceptance bridge must keep covering the full tool surface."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "accept_agent_bridge.py"


def test_agent_bridge_covers_all_tools() -> None:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=120,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    assert completed.returncode == 0, output
    from deeptutor.review.agent_tools import REVIEW_AGENT_TOOL_NAMES

    assert len(REVIEW_AGENT_TOOL_NAMES) == 20
    assert "every tool named per contract" in output
    assert "'review_hint': 4" in output
    assert "'review_generate': 8" in output
    assert "14/14 steps passed" in output
    # A renamed or dropped tool must break this test, not just the count.
    for name in REVIEW_AGENT_TOOL_NAMES:
        assert f"'{name}':" in output
