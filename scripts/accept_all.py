"""Unified acceptance entry point for the review studio (RS-1303).

Run with the project virtualenv:

    .venv/Scripts/python.exe scripts/accept_all.py

It chains, in order:

1. ``accept_phase05.py``            -- HTTP + export acceptance;
2. ``accept_book_bridge.py``        -- Book -> Review bridge acceptance;
3. ``accept_async_generation.py``    -- background job start/poll acceptance;
4. ``accept_agent_bridge.py``       -- agent-tool review flow;
5. ``build_golden_specs.py --check`` -- compare golden inputs without writing;
6. ``build_golden_specs.py``         -- export the golden specs.

Every step runs in a child process started with ``sys.executable`` so the
child uses the same interpreter. Each acceptance script explicitly selects
the checkout package. Child output is passed through to stdout/stderr so a
failure can be diagnosed in place; this script only adds a one-line
``[PASS]``/``[FAIL]`` marker per step plus a final summary.

Exit code: 0 only when every step exits 0; 1 otherwise.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
GOLDEN_SCRIPT = SCRIPTS / "build_golden_specs.py"
MISSING_HINT = "missing script (WS-A must deliver it): {path}"


def run_step(name: str, script: Path, args: list[str], timeout: float) -> bool:
    """Run one acceptance step and print a ``[PASS]``/``[FAIL]`` marker."""
    if not script.is_file():
        print(f"[FAIL] {name}: " + MISSING_HINT.format(path=script), flush=True)
        return False

    argv = [sys.executable, str(script), *args]
    started = time.time()
    try:
        # ``check=False`` keeps this script in charge of the exit code and of
        # the summary; the child's own output is inherited unchanged.
        completed = subprocess.run(argv, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - started
        print(f"[FAIL] {name}: timed out after {timeout:g}s ({elapsed:.1f}s)", flush=True)
        return False
    except OSError as exc:
        print(f"[FAIL] {name}: could not start child process: {exc}", flush=True)
        return False

    elapsed = time.time() - started
    if completed.returncode == 0:
        print(f"[PASS] {name} ({elapsed:.1f}s)", flush=True)
        return True
    print(
        f"[FAIL] {name}: exit code {completed.returncode} ({elapsed:.1f}s)",
        flush=True,
    )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="per-step timeout in seconds (default: 900)",
    )
    args = parser.parse_args()

    steps: list[tuple[str, Path, list[str]]] = [
        ("accept_phase05.py", SCRIPTS / "accept_phase05.py", []),
        ("accept_book_bridge.py", SCRIPTS / "accept_book_bridge.py", []),
        ("accept_async_generation.py", SCRIPTS / "accept_async_generation.py", []),
        ("accept_agent_bridge.py", SCRIPTS / "accept_agent_bridge.py", []),
        ("build_golden_specs.py --check", GOLDEN_SCRIPT, ["--check"]),
        ("build_golden_specs.py (export)", GOLDEN_SCRIPT, []),
    ]

    print("review-studio acceptance", flush=True)
    print(f"interpreter: {sys.executable}", flush=True)
    print(f"scripts: {SCRIPTS}", flush=True)

    results: list[tuple[str, bool]] = []
    for name, script, extra in steps:
        print(f"\n--- {name} ---", flush=True)
        results.append((name, run_step(name, script, extra, args.timeout)))

    failures = [name for name, passed in results if not passed]
    print("", flush=True)
    if failures:
        print(f"ACCEPTANCE FAILED ({len(failures)}/{len(results)} step(s) failed)", flush=True)
        for name in failures:
            print(f"  - {name}", flush=True)
        return 1
    print(f"ACCEPTANCE PASSED ({len(results)}/{len(results)} steps)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
