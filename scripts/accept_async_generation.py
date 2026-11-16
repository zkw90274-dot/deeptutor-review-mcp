"""Phase 32 acceptance: review_generate as a background job with polling.

Run with the project virtualenv:

    .venv/Scripts/python.exe scripts/accept_async_generation.py

Drives the real tool, the real job store and a real background thread; only
the generator module is replaced (a 1.2 s sleep plus the golden spec) so the
script never calls an LLM. Steps, per
the async-generation design:

    start -> poll while running -> chapter/job_id mutex -> poll done ->
    pack stored -> failing chapter -> unknown job -> path-shaped job id ->
    corrupt job file

Like the other acceptance scripts this is stdlib-only, runs against the
working tree (``src``), and uses an isolated
``DEEPTUTOR_REVIEW_HOME`` so the real ``data/`` tree is never touched. Exit
code is 0 only when every step passes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[1] / "src"
GOLDEN_JSON = Path(__file__).resolve().parents[1] / "tests" / "golden" / "golden-formula.json"

SPEC_ID = "golden-formula"

# ``deeptutor`` is a regular package in site-packages, so putting the checkout
# on ``sys.path`` does not merge the namespace portion: the installed wheel
# would win and the script would exercise stale code. Prepend the checkout to
# the package's own ``__path__`` instead, the same seam ``tests/conftest.py``
# uses.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_HOME = Path(tempfile.mkdtemp(prefix="review-async-accept-"))
os.environ["DEEPTUTOR_HOME"] = str(_HOME)
os.environ["DEEPTUTOR_REVIEW_HOME"] = str(_HOME / "review")
os.environ["AUTH_ENABLED"] = "false"

import deeptutor  # noqa: E402

_CHECKOUT_PACKAGE = SRC / "deeptutor"
if str(_CHECKOUT_PACKAGE) not in deeptutor.__path__:
    deeptutor.__path__.insert(0, str(_CHECKOUT_PACKAGE))

REPORT: dict = {"home": str(_HOME), "steps": []}
FAILURES: list[str] = []


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)
        raise AssertionError(message)


def record(step: str, detail: str) -> None:
    REPORT["steps"].append({"step": step, "detail": detail})
    print(f"  [PASS] {step}: {detail}")


def _install_fake_generator() -> None:
    """Replace only the LLM-facing module; job store and thread stay real."""
    spec = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))
    module = types.ModuleType("deeptutor.review.generate")

    def infer_chapters(kb_name: str):
        return [
            types.SimpleNamespace(name="第1章_绪论", doc_count=3),
            types.SimpleNamespace(name="boom", doc_count=1),
        ]

    def generate_spec_from_kb(**kwargs):
        if kwargs["chapter"] == "boom":
            raise RuntimeError("generator exploded for the acceptance run")
        time.sleep(1.2)  # long enough to observe a running poll

        class _Result:
            pass

        result = _Result()
        result.spec = dict(spec)
        result.qa_count = len(spec["items"])
        result.skipped = 0
        result.warnings = []
        return result

    module.infer_chapters = infer_chapters
    module.generate_spec_from_kb = generate_spec_from_kb
    sys.modules["deeptutor.review.generate"] = module


def _poll(tool, job_id: str) -> dict:
    result = asyncio.run(tool.execute(kb="hf", job_id=job_id))
    assert_true(result.success, f"poll {job_id}: success is False")
    payload = json.loads(result.content)
    assert_true("instruction" in payload, f"poll {job_id}: missing instruction")
    return payload


def _wait_for(tool, job_id: str, statuses: set[str], timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = _poll(tool, job_id)
        if payload["status"] in statuses:
            return payload
        time.sleep(0.25)
    raise AssertionError(
        f"job {job_id} never reached {sorted(statuses)} within {timeout}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="keep the temp home")
    args = parser.parse_args()

    _install_fake_generator()

    from deeptutor.review.agent_tools import ReviewGenerateTool
    from deeptutor.review.jobs import jobs_dir
    from deeptutor.review.service import ReviewService

    tool = ReviewGenerateTool()
    print(f"home: {_HOME}")

    # 1. Start: the call returns a job id in seconds, not the pack.
    started = time.time()
    result = asyncio.run(
        tool.execute(kb="hf", chapter="第1章_绪论", num_questions=6)
    )
    assert_true(result.success, "start: success is False")
    start = json.loads(result.content)
    elapsed = time.time() - started
    assert_true(bool(start.get("job_id")), "start: missing job_id")
    assert_true(
        start.get("status") in ("queued", "running"),
        f"start: unexpected status {start.get('status')!r}",
    )
    assert_true(elapsed < 5.0, f"start: took {elapsed:.1f}s, expected seconds")
    assert_true(
        "job_id" in start["instruction"], "start: instruction does not say to poll"
    )
    job_id = start["job_id"]
    assert_true(
        (jobs_dir() / f"{job_id}.json").is_file(), "start: job file missing"
    )
    record("start", f"{job_id} in {elapsed:.2f}s, status {start['status']}")

    # 2. Poll while the background thread runs.
    running = _poll(tool, job_id)
    assert_true(
        running["status"] in ("queued", "running"),
        f"poll while running: status {running['status']!r}",
    )
    record("poll-running", running["instruction"][:60])

    # 3. chapter and job_id together are rejected before anything runs.
    mutex = json.loads(
        asyncio.run(
            tool.execute(kb="hf", job_id=job_id, chapter="第1章_绪论")
        ).content
    )
    assert_true("not both" in mutex.get("error", ""), "mutex: error text missing")
    record("mutex", mutex["error"][:50])

    # 4. Poll until done; the pack is stored through the real QA path.
    done = _wait_for(tool, job_id, {"done", "failed"})
    assert_true(done["status"] == "done", f"done: status {done['status']!r}")
    assert_true(done.get("spec_id") == SPEC_ID, f"done: spec_id {done.get('spec_id')!r}")
    assert_true(done.get("qa_passed") is True, "done: qa_passed is not True")
    assert_true(done.get("qa_count") == done.get("counts", {}).get("items"), "done: question count disagrees with stored items")
    assert_true("review_today" in done["instruction"], "done: no next-step hint")
    stored = {meta["spec_id"] for meta in ReviewService().list_specs()}
    assert_true(SPEC_ID in stored, f"store: {SPEC_ID!r} missing from {stored}")
    record("done", f"spec {SPEC_ID} stored, qa passed")

    # 5. A generator failure lands in the job record, not in the tool call.
    failed_start = json.loads(
        asyncio.run(tool.execute(kb="hf", chapter="boom", num_questions=5)).content
    )
    failed = _wait_for(tool, failed_start["job_id"], {"done", "failed"})
    assert_true(failed["status"] == "failed", f"failed: status {failed['status']!r}")
    assert_true(
        "RuntimeError" in (failed.get("error") or ""),
        "failed: error text does not carry the exception",
    )
    record("failed-job", failed["error"][:60])

    # 6. Unknown and path-shaped ids are soft misses, never file access.
    unknown = json.loads(
        asyncio.run(tool.execute(kb="hf", job_id="gen-9999999999-ffff")).content
    )
    assert_true(
        "unknown generation job" in unknown.get("error", ""), "unknown: error text"
    )
    hostile = json.loads(
        asyncio.run(tool.execute(kb="hf", job_id="../../etc/passwd")).content
    )
    assert_true(
        "unknown generation job" in hostile.get("error", ""),
        "hostile id: should read as unknown, not escape the jobs dir",
    )
    record("unknown-and-hostile", "both read as unknown generation job")

    # 7. A corrupt job file reads as unknown, not as a crash.
    corrupt_id = "gen-0000000000-dead"
    (jobs_dir() / f"{corrupt_id}.json").write_text("{not json", encoding="utf-8")
    corrupt = json.loads(
        asyncio.run(tool.execute(kb="hf", job_id=corrupt_id)).content
    )
    assert_true(
        "unknown generation job" in corrupt.get("error", ""), "corrupt: error text"
    )
    record("corrupt-file", "reads as unknown generation job")

    print()
    if FAILURES:
        print(f"ACCEPTANCE FAILED ({len(FAILURES)} failure(s))")
        return 1
    print(f"ACCEPTANCE PASSED ({len(REPORT['steps'])} steps)")

    if not args.keep:
        shutil.rmtree(_HOME, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
