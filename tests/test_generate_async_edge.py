"""Adversarial edge cases for the Phase 32 async generation job.

These tests attack the two seams a model can reach: the on-disk job store
(``<review_home>/jobs/*.json``) and the poll path of ``review_generate``. The
job file is written by another process and may be corrupt, truncated, or named
by a hostile ``job_id``; the poller must degrade to "no such job" rather than
raise, hang, or touch a file outside the jobs directory.

Unlike ``tests/test_agent_tools.py``, the real ``deeptutor.review.jobs`` module
is used here: faking the store would hide exactly the parsing and path
validation these tests exist to exercise.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

from deeptutor.core.tool_protocol import ToolResult
from deeptutor.review import jobs
from deeptutor.review.agent_tools import ReviewGenerateTool

JOB_ID = "gen-1-abcd"

#: Every key ``create_job`` writes. A reader must see all of them or ``None``;
#: any other key set would be the partial read that ``os.replace`` prevents.
FULL_FIELDS = frozenset(
    {
        "job_id",
        "kb",
        "chapter",
        "num_questions",
        "question_types",
        "difficulty",
        "replace",
        "status",
        "created_at",
        "updated_at",
        "spec_id",
        "qa_count",
        "skipped",
        "qa_passed",
        "counts",
        "warnings",
        "error",
    }
)


@pytest.fixture()
def review_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "review"
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(home))
    return home


def _jobs_dir(review_home: Path) -> Path:
    return review_home / "jobs"


def _record(*, job_id: str = JOB_ID, **overrides: Any) -> dict[str, Any]:
    now = time.time()
    record: dict[str, Any] = {
        "job_id": job_id,
        "kb": "hf",
        "chapter": "第1章",
        "num_questions": 6,
        "question_types": None,
        "difficulty": "medium",
        "status": "running",
        "created_at": now,
        "updated_at": now,
        "spec_id": None,
        "qa_count": None,
        "skipped": None,
        "qa_passed": None,
        "counts": None,
        "error": None,
    }
    record.update(overrides)
    return record


def _write_raw(review_home: Path, job_id: str, text: str) -> Path:
    directory = _jobs_dir(review_home)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{job_id}.json"
    path.write_text(text, encoding="utf-8")
    return path


def _write_job(review_home: Path, payload: dict[str, Any]) -> Path:
    return _write_raw(
        review_home,
        payload["job_id"],
        json.dumps(payload, ensure_ascii=False),
    )


def _poll(job_id: str, kb: str = "hf") -> dict[str, Any]:
    """Drive the real tool through the poll branch and freeze the soft shape."""
    result = asyncio.run(
        ReviewGenerateTool().execute(kb=kb, job_id=job_id)
    )
    assert result.success is True, result.content
    payload = json.loads(result.content)
    assert isinstance(payload, dict)
    assert result.metadata["review_generate"] == payload
    assert payload["instruction"]
    return payload


def _files_outside(review_home: Path) -> set[Path]:
    """Every file under the tmp root except those inside the jobs directory."""
    root = review_home.parent
    jobs_root = _jobs_dir(review_home)
    return {
        path
        for path in root.rglob("*")
        if path.is_file() and jobs_root not in path.parents
    }


# -- 1. corrupt / malformed job files on disk -------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(
            '{"job_id": "gen-1-abcd", "status": "running"', id="truncated-json"
        ),
        pytest.param("[]", id="non-dict-json"),
        pytest.param(
            json.dumps({"job_id": "gen-9-ffff", "status": "running"}),
            id="inner-id-disagrees-with-filename",
        ),
        pytest.param("   \n\t  ", id="whitespace-only"),
        pytest.param("null", id="json-null"),
        pytest.param(
            '{"job_id": "gen-1-abcd", "status": "running"} trailing',
            id="valid-object-plus-trailing-garbage",
        ),
    ],
)
def test_corrupt_job_file_is_the_unknown_job_soft_failure(
    review_home: Path, raw: str
) -> None:
    _write_raw(review_home, JOB_ID, raw)

    assert jobs.load_job(JOB_ID) is None

    payload = _poll(JOB_ID)
    assert payload["error"] == f"unknown generation job: {JOB_ID!r}"
    assert payload["job_id"] == JOB_ID


def test_missing_job_file_is_the_unknown_job_soft_failure(review_home: Path) -> None:
    payload = _poll(JOB_ID)
    assert payload["error"] == f"unknown generation job: {JOB_ID!r}"
    assert "review_specs" in payload["instruction"]


# -- 2. job_id path attacks -------------------------------------------------


@pytest.mark.parametrize(
    "job_id",
    [
        pytest.param("../../etc", id="parent-traversal"),
        pytest.param("gen-1-abcd/../x", id="embedded-traversal"),
        pytest.param("gen-1-ABCD", id="uppercase-hex"),
        pytest.param("gen-1-" + "a" * 4096, id="very-long"),
        pytest.param("gen-１２-abcd", id="fullwidth-digits"),
        pytest.param("gen--abcd", id="empty-epoch"),
        pytest.param("gen-1-abcd.json", id="extension-smuggled"),
    ],
)
def test_poll_rejects_ids_that_cannot_be_path_components(
    review_home: Path, job_id: str
) -> None:
    before = _files_outside(review_home)

    # Rejected before a Path is ever joined, so no directory is created and no
    # lookup can escape ``<review_home>/jobs``.
    assert jobs.load_job(job_id) is None

    payload = _poll(job_id)
    assert payload["error"] == f"unknown generation job: {job_id!r}"
    assert payload["job_id"] == job_id
    assert _files_outside(review_home) == before

    # The obvious traversal target must not exist anywhere near the store.
    assert not (review_home.parent / "etc.json").exists()
    assert not (review_home / "x.json").exists()


def test_load_job_rejects_ids_that_regex_anchors_would_let_through(
    review_home: Path,
) -> None:
    """``fullmatch`` must reject a trailing newline and a trailing space."""
    assert jobs.load_job("gen-1-abcd\n") is None
    assert jobs.load_job("gen-1-abcd ") is None
    assert jobs.load_job("gen-1-abcd\t") is None


# -- 3. stale boundaries ----------------------------------------------------


def test_is_stale_is_strictly_greater_than_the_threshold() -> None:
    now = 1_800_000_000.0
    exact = _record(updated_at=now - jobs.JOB_STALE_SECONDS)
    one_past = _record(updated_at=now - jobs.JOB_STALE_SECONDS - 1.0)

    assert jobs.is_stale(exact, now=now) is False
    assert jobs.is_stale(one_past, now=now) is True

    # A queued record can only be a zombie: nothing will ever start it.
    zombie = _record(status="queued", updated_at=now - 10_000.0)
    assert jobs.is_stale(zombie, now=now) is True

    # Terminal statuses never go stale, however old the stamp.
    assert jobs.is_stale(_record(status="done", updated_at=0.0), now=now) is False
    assert jobs.is_stale(_record(status="failed", updated_at=0.0), now=now) is False


def test_poll_tool_uses_the_same_stale_boundary(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed = time.time()
    monkeypatch.setattr(
        jobs, "time", types.SimpleNamespace(time=lambda: fixed)
    )

    at_boundary = _record(
        created_at=fixed - 100.0,
        updated_at=fixed - jobs.JOB_STALE_SECONDS,
    )
    _write_job(review_home, at_boundary)
    payload = _poll(JOB_ID)
    assert "stale" not in payload
    assert payload["status"] == "running"
    assert "Still generating" in payload["instruction"]

    one_second_past = _record(
        created_at=fixed - 100.0,
        updated_at=fixed - jobs.JOB_STALE_SECONDS - 1.0,
    )
    _write_job(review_home, one_second_past)
    payload = _poll(JOB_ID)
    assert payload["stale"] is True
    assert "has not updated for over an hour" in payload["instruction"]


def test_poll_tool_reports_a_queued_zombie_as_stale(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed = time.time()
    monkeypatch.setattr(
        jobs, "time", types.SimpleNamespace(time=lambda: fixed)
    )
    _write_job(
        review_home,
        _record(
            status="queued",
            created_at=fixed - 20_000.0,
            updated_at=fixed - 10_000.0,
        ),
    )

    payload = _poll(JOB_ID)
    assert payload["status"] == "queued"
    assert payload["stale"] is True
    assert "has not updated" in payload["instruction"]


# -- 4. concurrent writers never expose torn JSON ---------------------------


def test_concurrent_heartbeat_never_exposes_a_partial_record(
    review_home: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    threads: list[threading.Thread] = []

    def blocking_runner(job: dict[str, Any]) -> None:
        threads.append(threading.current_thread())
        entered.set()
        # The main thread rewrites a second job while this one is open.
        assert release.wait(timeout=10), "the test never released the runner"

    first = jobs.start_generation_job(
        kb="kb-x",
        chapter="第1章",
        num_questions=6,
        question_types=None,
        difficulty="medium",
        runner=blocking_runner,
    )
    assert entered.wait(timeout=10), "the runner never started"

    second = jobs.create_job(
        kb="kb-x",
        chapter="第2章",
        num_questions=6,
        question_types=None,
        difficulty="medium",
    )
    assert first["job_id"] != second["job_id"]

    try:
        for _ in range(200):
            jobs.save_job(second)
            loaded = jobs.load_job(first["job_id"])
            assert loaded is None or set(loaded) == FULL_FIELDS
            assert loaded is None or loaded["job_id"] == first["job_id"]

            scratch = jobs.load_job(second["job_id"])
            assert scratch is not None and set(scratch) == FULL_FIELDS
            assert scratch["job_id"] == second["job_id"]
    finally:
        release.set()
        threads[0].join(timeout=10)

    assert not threads[0].is_alive(), "the runner thread outlived the test"

    # Atomic writes leave no temporary file behind.
    leftovers = [
        path
        for path in _jobs_dir(review_home).iterdir()
        if path.name.endswith(".tmp")
    ]
    assert leftovers == []


# -- 5. two jobs for the same chapter ---------------------------------------


def _wait_for_status(
    job_id: str, status: str, *, timeout: float = 10.0
) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = jobs.load_job(job_id)
        if record is not None and record["status"] == status:
            return record
        time.sleep(0.01)
    raise AssertionError(f"{job_id} never reached {status!r}")


def test_two_jobs_for_the_same_chapter_stay_independent(
    review_home: Path,
) -> None:
    threads: list[threading.Thread] = []

    def runner_for(spec_id: str):
        def runner(job: dict[str, Any]) -> None:
            threads.append(threading.current_thread())
            job["status"] = "done"
            job["spec_id"] = spec_id
            job["qa_count"] = 6
            job["qa_passed"] = True
            job["counts"] = {"items": 6}
            jobs.save_job(job)

        return runner

    first = jobs.start_generation_job(
        kb="kb-x",
        chapter="第1章_绪论",
        num_questions=6,
        question_types=["choice"],
        difficulty="medium",
        runner=runner_for("kb-aaaa1111-1"),
    )
    second = jobs.start_generation_job(
        kb="kb-x",
        chapter="第1章_绪论",
        num_questions=6,
        question_types=["written"],
        difficulty="easy",
        runner=runner_for("kb-bbbb2222-2"),
    )

    # A 4-hex id under the same epoch second could collide; the store must not
    # silently overwrite one job with the other.
    assert first["job_id"] != second["job_id"]

    stored_first = _wait_for_status(first["job_id"], "done")
    stored_second = _wait_for_status(second["job_id"], "done")

    assert stored_first["spec_id"] == "kb-aaaa1111-1"
    assert stored_second["spec_id"] == "kb-bbbb2222-2"
    assert stored_first["question_types"] == ["choice"]
    assert stored_second["question_types"] == ["written"]

    # The poll tool reads the same distinct records.
    assert _poll(first["job_id"], kb="kb-x")["spec_id"] == "kb-aaaa1111-1"
    assert _poll(second["job_id"], kb="kb-x")["spec_id"] == "kb-bbbb2222-2"

    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()


# -- 6. create_job preserves question types verbatim ------------------------


def test_create_job_preserves_duplicate_question_types(review_home: Path) -> None:
    job = jobs.create_job(
        kb="kb-x",
        chapter="第1章",
        num_questions=6,
        question_types=["choice", "choice", "written"],
        difficulty="medium",
    )

    assert job["question_types"] == ["choice", "choice", "written"]
    stored = jobs.load_job(job["job_id"])
    assert stored is not None
    assert stored["question_types"] == ["choice", "choice", "written"]


def test_create_job_stores_none_question_types_as_none(review_home: Path) -> None:
    job = jobs.create_job(
        kb="kb-x",
        chapter="第1章",
        num_questions=6,
        question_types=None,
        difficulty="medium",
    )

    assert job["question_types"] is None
    stored = jobs.load_job(job["job_id"])
    assert stored is not None
    assert stored["question_types"] is None


# -- 7. clock skew ----------------------------------------------------------


def test_poll_survives_created_at_in_the_future(review_home: Path) -> None:
    now = time.time()
    _write_job(
        review_home,
        _record(
            status="running",
            created_at=now + 3600.0,
            updated_at=now,
        ),
    )

    payload = _poll(JOB_ID)
    assert payload["status"] == "running"
    assert isinstance(payload["elapsed_seconds"], (int, float))
    assert not isinstance(payload["elapsed_seconds"], bool)
    # Clock skew must not surface a negative age: the tool clamps to zero.
    assert payload["elapsed_seconds"] == 0
    assert "Still generating" in payload["instruction"]


# -- 8. malformed-but-parseable records -------------------------------------


@pytest.mark.parametrize(
    "created_at",
    [
        pytest.param("not-a-number", id="string"),
        pytest.param([1], id="list"),
        pytest.param({"epoch": 1}, id="object"),
        pytest.param(float("inf"), id="infinity"),
        pytest.param(float("nan"), id="nan"),
    ],
)
def test_poll_degrades_instead_of_raising_on_a_malformed_created_at(
    review_home: Path, created_at: Any
) -> None:
    """A file can be valid JSON and still carry an unusable ``created_at``.

    ``load_job`` only rejects non-dict payloads and id mismatches, so the
    timestamp reaches ``float(...)`` / ``round(...)`` in ``_poll_job``. The
    tool contract is that a bad record never escapes as an exception; either
    degradation is acceptable — a soft payload or an actionable failure text.
    """
    _write_job(
        review_home,
        _record(created_at=created_at, updated_at=time.time()),
    )

    # This call is the assertion: the defect raises here.
    result = asyncio.run(ReviewGenerateTool().execute(kb="hf", job_id=JOB_ID))
    assert isinstance(result, ToolResult)
    if result.success:
        payload = json.loads(result.content)
        assert payload["instruction"]
    else:
        assert "review_generate failed" in result.content


# -- 9. unknown status ------------------------------------------------------


def test_unknown_status_falls_back_to_an_honest_instruction(
    review_home: Path,
) -> None:
    _write_job(review_home, _record(status="paused"))

    payload = _poll(JOB_ID)
    assert payload["status"] == "paused"
    # An unrecognized status must not be reported as a failed generation.
    assert payload["instruction"].startswith(
        "This job record carries an unrecognized status"
    )
    assert "review_specs" in payload["instruction"]
