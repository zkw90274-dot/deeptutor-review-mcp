"""Agent tool-call acceptance: the full ten-tool review surface, no HTTP.

Run with the project virtualenv:

    .venv/Scripts/python.exe scripts/accept_agent_bridge.py

Unlike ``accept_phase05.py`` this drives no server: it instantiates the ten
review agent tools directly (the same classes the manifest advertises) and
calls them in the order a real agent would, per the frozen contract in
``docs/maintainer-notes.md`` §5:

    review_specs -> review_today -> review_hint(1, 2) -> review_attempt(good)
    -> replay -> review_attempt(again) -> review_weakness
    -> review_export_link -> review_generate(chapters, start, poll)

``review_generate`` runs against an injected fake generator and job store so
the tool wiring (chapter preflight, ``replace`` pass-through, polling) is
exercised without an LLM or a multi-minute job; the real pipeline stays in
``accept_async_generation.py``. The run uses an isolated
``DEEPTUTOR_REVIEW_HOME`` / ``DEEPTUTOR_HOME`` so the real ``data/`` tree is
never touched; the script refuses to continue if the resolved review home is
outside the temporary directory. The golden spec is loaded from
``tests/golden/golden-formula.json`` (built by
``scripts/build_golden_specs.py --write``) and ingested through the real
``ReviewService.put_spec`` QA path.

Like the other acceptance scripts this is stdlib-only and imports the working
tree (``src``) rather than the installed wheel; the
module paths used are printed up front. Exit code is 0 only when every step
passes.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[1] / "src"
GOLDEN_JSON = Path(__file__).resolve().parents[1] / "tests" / "golden" / "golden-formula.json"

SPEC_ID = "golden-formula"
SUBMISSION_GOOD = "ws29-accept-good"
SUBMISSION_AGAIN = "ws29-accept-again"

# ``deeptutor`` is a regular package in site-packages, so putting the checkout
# on ``sys.path`` does not merge the namespace portion: the installed wheel
# would win and the script would exercise stale code. Prepend the checkout to
# the package's own ``__path__`` instead, the same seam ``tests/conftest.py``
# and ``scripts/build_golden_specs.py`` use.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The review home must be decided before anything constructs a store; do the
# env setup before the first ``deeptutor`` import below reads settings.
_HOME = Path(tempfile.mkdtemp(prefix="review-agent-accept-"))
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


def parse_payload(result, *, tool: str) -> dict:
    """Assert a ToolResult carries the frozen contract: JSON content plus the
    ``review_xxx`` metadata mirror, and return the parsed payload."""
    assert_true(result.success, f"{tool}: success is False: {result.content[:200]}")
    try:
        payload = json.loads(result.content)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{tool}: content is not valid JSON: {exc}") from exc
    assert_true(isinstance(payload, dict), f"{tool}: payload is not an object")
    assert_true("instruction" in payload, f"{tool}: payload missing instruction")
    meta_key = tool  # metadata mirrors the payload under the tool name
    assert_true(
        isinstance(result.metadata, dict) and meta_key in result.metadata,
        f"{tool}: metadata missing {meta_key!r} mirror",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="keep the temporary review home")
    args = parser.parse_args()
    started_all = time.time()
    print(f"home       : {_HOME}")
    print(f"checkout   : {SRC}")

    def step(name: str, fn):
        started = time.time()
        try:
            detail = fn()
            REPORT["steps"].append(
                {"name": name, "passed": True, "seconds": round(time.time() - started, 2), "detail": detail}
            )
            print(f"[PASS] {name} ({round(time.time() - started, 2)}s) {detail}")
            return detail
        except AssertionError as exc:
            REPORT["steps"].append({"name": name, "passed": False, "error": str(exc)})
            print(f"[FAIL] {name}: {exc}")
            return None

    # -- imports ---------------------------------------------------------
    try:
        from deeptutor.core.tool_protocol import BaseTool  # noqa: F401
        from deeptutor.review.agent_tools import (
            ReviewAttemptTool,
            ReviewBookFiguresTool,
            ReviewExportLinkTool,
            ReviewExportTreeTool,
            ReviewGenerateTool,
            ReviewHintTool,
            ReviewPlanTool,
            ReviewSpecsTool,
            ReviewTodayTool,
            ReviewWeaknessTool,
        )
    except ImportError as exc:
        print(f"[FAIL] import deeptutor.review.agent_tools: {exc}")
        print(
            "ACCEPTANCE FAILED - the Phase 29 WS-A deliverable "
            "src/deeptutor/review/agent_tools.py is missing or does not export "
            "the ten frozen tool classes yet."
        )
        shutil.rmtree(_HOME, ignore_errors=True)
        return 2

    tools = {
        "review_specs": ReviewSpecsTool(),
        "review_today": ReviewTodayTool(),
        "review_attempt": ReviewAttemptTool(),
        "review_weakness": ReviewWeaknessTool(),
        "review_export_link": ReviewExportLinkTool(),
        "review_export_tree": ReviewExportTreeTool(),
        "review_book_figures": ReviewBookFiguresTool(),
        "review_hint": ReviewHintTool(),
        "review_generate": ReviewGenerateTool(),
        "review_plan": ReviewPlanTool(),
    }

    def run(tool: str, **kwargs) -> dict:
        result = asyncio.run(tools[tool].execute(**kwargs))
        return parse_payload(result, tool=tool)

    def verify_definitions() -> dict:
        detail: dict = {}
        for name, tool in tools.items():
            definition = tool.get_definition()
            assert_true(isinstance(tool, BaseTool), f"{name}: not a BaseTool subclass")
            assert_true(definition.name == name, f"{name}: definition.name={definition.name!r}")
            assert_true(bool(definition.description), f"{name}: empty description")
            detail[name] = len(definition.parameters)
        return detail

    def seed_spec() -> dict:
        from deeptutor.review.service import ReviewService

        assert_true(GOLDEN_JSON.exists(), f"missing {GOLDEN_JSON}; run scripts/build_golden_specs.py --write")
        payload = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))
        assert_true(payload.get("id") == SPEC_ID, f"{GOLDEN_JSON.name}: id={payload.get('id')!r}")
        summary = ReviewService().put_spec(payload)
        # Red line: the isolated home really owns the storage.
        store = ReviewService().store
        home = Path(store.home).resolve()
        assert_true(
            str(home).startswith(str(_HOME.resolve())),
            f"review home escaped the temporary directory: {home}",
        )
        assert_true(summary["created"] is True, "put_spec should report created=True")
        return {"items": summary["counts"]["items"], "db": str(store.db_path)}

    state: dict = {}

    def step_specs() -> dict:
        payload = run("review_specs")
        specs = payload.get("specs")
        assert_true(isinstance(specs, list) and specs, "review_specs: empty specs list")
        ids = {row.get("id") for row in specs}
        assert_true(SPEC_ID in ids, f"review_specs: {SPEC_ID} not listed (got {sorted(ids)})")
        row = next(row for row in specs if row.get("id") == SPEC_ID)
        assert_true(bool(row.get("title")), "review_specs: spec row missing title")
        return {"specs": len(specs)}

    def step_today() -> dict:
        payload = run("review_today", spec_id=SPEC_ID)
        items = payload.get("items")
        assert_true(isinstance(items, list) and items, "review_today: empty items for a fresh spec")
        first = items[0]
        item_id = first.get("item_id") or first.get("id")
        assert_true(bool(item_id), f"review_today: first item has no id: {first}")
        assert_true(bool(first.get("kind")), "review_today: item missing kind")
        assert_true(bool(first.get("stem")), "review_today: item missing stem")
        state["item_id"] = item_id
        state["concept_id"] = first.get("concept_id")
        return {"item_id": item_id, "items": len(items)}

    def step_attempt_good() -> dict:
        payload = run(
            "review_attempt",
            spec_id=SPEC_ID,
            item_id=state["item_id"],
            rating="good",
            submission_id=SUBMISSION_GOOD,
        )
        # id relay: the tool must echo the very id it got from review_today.
        assert_true(payload.get("item_id") == state["item_id"], "review_attempt: item_id mismatch")
        assert_true(payload.get("event_id"), "review_attempt: missing event_id")
        assert_true(payload.get("created") is True, "review_attempt: first call should be created=True")
        assert_true("mastery_projection" in payload, "review_attempt: missing mastery_projection")
        assert_true("next" in payload, "review_attempt: missing next")
        state["event_id"] = payload.get("event_id")
        return {"event_id": state["event_id"]}

    def step_replay() -> dict:
        payload = run(
            "review_attempt",
            spec_id=SPEC_ID,
            item_id=state["item_id"],
            rating="good",
            submission_id=SUBMISSION_GOOD,
        )
        assert_true(payload.get("created") is False, "replay: created should be False")
        assert_true(
            payload.get("event_id") == state["event_id"],
            "replay: event_id changed for the same submission_id",
        )
        return {"event_id": payload.get("event_id"), "created": payload.get("created")}

    def step_attempt_again() -> dict:
        payload = run(
            "review_attempt",
            spec_id=SPEC_ID,
            item_id=state["item_id"],
            rating="again",
            submission_id=SUBMISSION_AGAIN,
        )
        assert_true(payload.get("event_id"), "review_attempt(again): missing event_id")
        assert_true(payload.get("created") is True, "review_attempt(again): should be created=True")
        assert_true(payload.get("event_id") != state["event_id"], "again rating reused the good event")
        return {"event_id": payload.get("event_id")}

    def step_description_guard() -> dict:
        description = tools["review_attempt"].get_definition().description
        lowered = description.lower()
        assert_true("self-assess" in lowered, "description missing 'self-assess'")
        assert_true("never grade" in lowered, "description missing 'never grade'")
        return {"description": description}

    def step_weakness() -> dict:
        payload = run("review_weakness", spec_id=SPEC_ID)
        rows = payload.get("weak_concepts")
        assert_true(isinstance(rows, list) and rows, "review_weakness: empty weak_concepts")
        item_id = state["item_id"]
        assert_true(
            any(item_id in row.get("item_ids", []) for row in rows),
            f"review_weakness: attempted item {item_id!r} not covered by any row",
        )
        row = next(row for row in rows if item_id in row.get("item_ids", []))
        assert_true(isinstance(row.get("weakness"), (int, float)), "review_weakness: weakness not numeric")
        assert_true(bool(row.get("concept_name")), "review_weakness: row missing concept_name")
        assert_true(
            row.get("evidence_count", 0) >= 1,
            "review_weakness: evidence_count should reflect the two recorded attempts",
        )
        return {"weak_concepts": len(rows), "concept": row.get("concept_name")}

    def step_export_link() -> dict:
        payload = run("review_export_link", spec_id=SPEC_ID)
        url = payload.get("url") or payload.get("export_url") or payload.get("link")
        expected = f"/api/review/{SPEC_ID}/export"
        assert_true(isinstance(url, str), "review_export_link: no url field")
        assert_true(url.endswith(expected), f"review_export_link: {url!r} does not end with {expected!r}")
        assert_true("://" not in url, f"review_export_link: expected a relative URL, got {url!r}")
        return {"url": url}

    def step_soft_failure() -> dict:
        # §5 frozen convention: unknown resource -> success=True with an
        # ``error`` field plus an ``instruction`` redirect; only internal
        # failures are success=False.
        payload = run("review_today", spec_id="ws29-no-such-spec")
        assert_true(payload.get("error"), "soft failure: missing error field")
        assert_true(
            "review_specs" in str(payload.get("instruction", "")),
            "soft failure: instruction should redirect to review_specs",
        )
        return {"error": payload["error"]}

    def step_hint() -> dict:
        golden = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))
        item = next(
            (row for row in golden.get("items", []) if row.get("id") == state["item_id"]),
            None,
        )
        correct = ""
        if item and item.get("kind") == "single_choice":
            index = item["answer"]["choice_index"] or 0
            correct = item["options"][index]["text"]
        levels: dict = {}
        for level in (1, 2):
            payload = run(
                "review_hint", spec_id=SPEC_ID, item_id=state["item_id"], level=level
            )
            hint = payload.get("hint") or ""
            assert_true(bool(hint), f"review_hint L{level}: empty hint")
            assert_true(
                payload.get("level") == level,
                f"review_hint L{level}: level echo is {payload.get('level')!r}",
            )
            assert_true(
                isinstance(payload.get("fallback"), bool),
                f"review_hint L{level}: fallback is not a boolean",
            )
            if correct:
                assert_true(
                    correct not in hint,
                    f"review_hint L{level}: hint repeats the correct option",
                )
            levels[level] = {"chars": len(hint), "fallback": payload["fallback"]}
        return levels

    def step_generate() -> dict:
        from types import SimpleNamespace

        import deeptutor.review.agent_tools as agent_tools_mod

        calls: dict = {}
        chapter_name = "第1章_验收夹具"

        class FakeGenerator:
            @staticmethod
            def infer_chapters(kb_name):
                calls["kb"] = kb_name
                return [SimpleNamespace(name=chapter_name, doc_count=1)]

        class FakeJobs:
            @staticmethod
            def start_generation_job(**kwargs):
                calls["start"] = kwargs
                return {"job_id": "gen-ws29-accept", "status": "queued"}

            @staticmethod
            def load_job(job_id):
                start = calls["start"]
                return {
                    "job_id": job_id,
                    "kb": start["kb"],
                    "chapter": start["chapter"],
                    "status": "done",
                    "spec_id": "ws29-generated",
                    "replace": start.get("replace", False),
                    "qa_count": 3,
                    "skipped": 0,
                    "counts": {"items": 3},
                    "qa_passed": True,
                    "warnings": [],
                    "error": None,
                    "created_at": time.time(),
                }

            @staticmethod
            def is_stale(job):
                return False

        original_generate = agent_tools_mod._generate_module
        original_jobs = agent_tools_mod._jobs_module
        agent_tools_mod._generate_module = lambda: FakeGenerator
        agent_tools_mod._jobs_module = lambda: FakeJobs
        try:
            listed = run("review_generate", kb="ws29-fixture-kb")
            assert_true(
                listed.get("mode") == "chapters",
                "review_generate: listing mode missing",
            )
            assert_true(
                [row.get("name") for row in listed.get("chapters", [])] == [chapter_name],
                "review_generate: chapter listing mismatch",
            )
            missing = run("review_generate", kb="ws29-fixture-kb", chapter="不存在的章")
            assert_true(missing.get("error"), "review_generate: unknown chapter should soft-fail")
            assert_true("start" not in calls, "review_generate: unknown chapter started a job")
            started = run(
                "review_generate",
                kb="ws29-fixture-kb",
                chapter=chapter_name,
                num_questions=3,
                # A truthy transport value must not opt into replacing a pack.
                replace="true",
            )
            assert_true(started.get("status") == "queued", "review_generate: start status")
            assert_true(bool(started.get("job_id")), "review_generate: no job_id")
            assert_true(
                calls["start"].get("replace") is False,
                "review_generate: non-literal replace must stay False",
            )
            polled = run("review_generate", kb="ws29-fixture-kb", job_id=started["job_id"])
            assert_true(polled.get("status") == "done", "review_generate: poll status")
            assert_true(
                polled.get("spec_id") == "ws29-generated",
                "review_generate: poll spec_id mismatch",
            )
            assert_true(polled.get("replace") is False, "review_generate: poll replace mirror")
        finally:
            agent_tools_mod._generate_module = original_generate
            agent_tools_mod._jobs_module = original_jobs
        return {"chapters": 1, "job_id": started["job_id"], "status": polled["status"]}

    step("0. definitions: ten tools named per contract", verify_definitions)
    if not step("1. golden spec ingested through put_spec (isolated home)", seed_spec):
        print("ACCEPTANCE FAILED — could not seed the golden spec.")
        return 1
    step("2. review_specs lists the golden spec", step_specs)
    if not step("3. review_today returns items", step_today):
        print("ACCEPTANCE FAILED — no item_id to relay into review_attempt.")
        return 1
    step("4. review_hint returns graded hints without the answer", step_hint)
    step("5. review_attempt records the learner's 'good' self-rating", step_attempt_good)
    step("6. same submission_id replays idempotently", step_replay)
    step("7. review_attempt records an 'again' self-rating", step_attempt_again)
    step("8. description guards: self-assess / never grade", step_description_guard)
    step("9. review_weakness ranks concepts after attempts", step_weakness)
    step("10. review_export_link returns the relative export URL", step_export_link)
    step("11. unknown spec soft-fails with error + redirect", step_soft_failure)
    step("12. review_generate: chapters -> start -> poll (fake pipeline)", step_generate)

    passed = len([s for s in REPORT["steps"] if s["passed"]])
    total = len(REPORT["steps"])
    print(f"\n{passed}/{total} steps passed in {round(time.time() - started_all, 2)}s")
    if FAILURES:
        print("ACCEPTANCE FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("ACCEPTANCE PASSED")
    if not args.keep:
        shutil.rmtree(_HOME, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
