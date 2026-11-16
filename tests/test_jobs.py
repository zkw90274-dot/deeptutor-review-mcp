"""Job store and runner tests (Phase 32).

The job file is the only channel a second process can see, so these tests
assert on the bytes under ``<review_home>/jobs`` rather than on the in-memory
dict. Generation itself is never executed: the generator module and the review
service are injected at the same lazy-import seam the runner uses.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from deeptutor.review import jobs
from deeptutor.review import service as service_module
from deeptutor.review.models import GateResult, QualityReport
from deeptutor.review.service import QualityGateFailed

CONTRACT_FIELDS = (
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
)


@pytest.fixture()
def review_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "review"
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(home))
    return home


def _read(review_home: Path, job_id: str) -> dict:
    path = review_home / "jobs" / f"{job_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _install_generate(monkeypatch: pytest.MonkeyPatch, **attributes) -> types.ModuleType:
    """Replace the generator module the runner imports on demand."""
    module = types.ModuleType("deeptutor.review.generate")
    for name, value in attributes.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "deeptutor.review.generate", module)
    return module


def _install_service(
    monkeypatch: pytest.MonkeyPatch, *, response: dict
) -> list:
    """Replace ``ReviewService`` while keeping the real gate exception."""
    calls: list = []

    class FakeService:
        def put_spec(
            self,
            spec,
            *,
            overwrite: bool = True,
            allow_history_rewrite: bool = False,
        ):
            calls.append((spec, overwrite, allow_history_rewrite))
            return response

    monkeypatch.setattr(service_module, "ReviewService", FakeService)
    return calls


def test_create_job_writes_queued_record(review_home: Path) -> None:
    job = jobs.create_job(
        kb="kb-x",
        chapter="第1章_绪论",
        num_questions=6,
        question_types=["mcq", "short"],
        difficulty="medium",
    )

    assert tuple(job) == CONTRACT_FIELDS
    assert job["job_id"].startswith("gen-")
    assert job["status"] == "queued"
    assert job["created_at"] <= job["updated_at"]
    assert job["question_types"] == ["mcq", "short"]
    for name in ("spec_id", "qa_count", "skipped", "qa_passed", "counts", "error"):
        assert job[name] is None
    assert _read(review_home, job["job_id"]) == job


def test_save_job_refreshes_updated_at(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = jobs.create_job(
        kb="kb-x",
        chapter="第1章",
        num_questions=4,
        question_types=None,
        difficulty="easy",
    )
    original = job["updated_at"]
    monkeypatch.setattr(jobs, "time", types.SimpleNamespace(time=lambda: original + 5.0))

    jobs.save_job(job)

    assert job["updated_at"] == original + 5.0
    assert _read(review_home, job["job_id"])["updated_at"] == original + 5.0


def test_load_job_roundtrips_the_contract_fields(review_home: Path) -> None:
    job = jobs.create_job(
        kb="kb-x",
        chapter="实验一_示波器",
        num_questions=3,
        question_types=["mcq"],
        difficulty="hard",
    )

    loaded = jobs.load_job(job["job_id"])

    assert loaded == job
    assert set(loaded) == set(CONTRACT_FIELDS)


def test_load_job_unknown_id_returns_none(review_home: Path) -> None:
    assert jobs.load_job("gen-999-dead") is None


def test_load_job_corrupt_json_returns_none(review_home: Path) -> None:
    path = jobs.jobs_dir() / "gen-123-abcd.json"
    path.write_text('{"job_id": "gen-123-abcd", "status": ', encoding="utf-8")

    assert jobs.load_job("gen-123-abcd") is None


def test_load_job_rejects_ids_outside_the_jobs_directory(review_home: Path) -> None:
    for hostile in ("../evil", "..\\evil", "gen-1-ABC1", "gen-1-abcg", "gen-1-abc1.json", ""):
        assert jobs.load_job(hostile) is None

    assert list(jobs.jobs_dir().iterdir()) == []


def test_start_generation_job_returns_queued_and_runs_runner(review_home: Path) -> None:
    finished = threading.Event()
    seen: list[str] = []

    def runner(job: dict) -> None:
        seen.append(job["job_id"])
        finished.set()

    job = jobs.start_generation_job(
        kb="kb-x",
        chapter="第1章",
        num_questions=2,
        question_types=["mcq"],
        difficulty="easy",
        runner=runner,
    )

    assert job["status"] == "queued"
    assert finished.wait(5.0), "the injected runner never ran"
    assert seen == [job["job_id"]]
    assert _read(review_home, job["job_id"])["status"] == "queued"


def test_identical_start_reuses_the_active_job(review_home: Path) -> None:
    seen: list[str] = []

    def runner(job: dict) -> None:
        seen.append(job["job_id"])

    first = jobs.start_generation_job(
        kb="kb-x",
        chapter="第1章",
        num_questions=2,
        question_types=["mcq"],
        difficulty="easy",
        runner=runner,
    )
    second = jobs.start_generation_job(
        kb="kb-x",
        chapter="第1章",
        num_questions=2,
        question_types=["mcq"],
        difficulty="easy",
        runner=runner,
    )

    assert second["job_id"] == first["job_id"]
    assert second["reused"] is True
    assert len(list(jobs.jobs_dir().glob("gen-*.json"))) == 1


def test_different_options_start_a_separate_job(review_home: Path) -> None:
    def runner(job: dict) -> None:
        pass

    first = jobs.start_generation_job(
        kb="kb-x",
        chapter="第1章",
        num_questions=2,
        question_types=["mcq"],
        difficulty="easy",
        runner=runner,
    )
    second = jobs.start_generation_job(
        kb="kb-x",
        chapter="第1章",
        num_questions=4,
        question_types=["mcq"],
        difficulty="hard",
        runner=runner,
    )

    assert second["job_id"] != first["job_id"]
    assert not second.get("reused")
    assert len(list(jobs.jobs_dir().glob("gen-*.json"))) == 2


def test_run_generation_records_done_with_put_spec_fields(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = object()
    generate_calls: list[dict] = []

    def fake_generate(**kwargs):
        generate_calls.append(kwargs)
        return types.SimpleNamespace(
            spec=sentinel, qa_count=6, skipped=1, warnings=["one item skipped"]
        )

    _install_generate(monkeypatch, generate_spec_from_kb=fake_generate)
    stored_specs = _install_service(
        monkeypatch,
        response={
            "spec_id": "kb-x-1",
            "created": True,
            "quality": {"passed": True},
            "counts": {"items": 6},
        },
    )
    job = jobs.create_job(
        kb="kb-x",
        chapter="第1章_绪论",
        num_questions=6,
        question_types=["mcq", "short"],
        difficulty="medium",
    )

    jobs._run_generation(job)

    stored = _read(review_home, job["job_id"])
    assert stored["status"] == "done"
    assert stored["spec_id"] == "kb-x-1"
    assert stored["qa_count"] == 6
    assert stored["skipped"] == 1
    assert stored["qa_passed"] is True
    assert stored["counts"] == {"items": 6}
    assert stored["warnings"] == ["one item skipped"]
    assert stored["error"] is None
    assert stored_specs == [(sentinel, False, False)]
    assert generate_calls == [
        {
            "kb_name": "kb-x",
            "chapter": "第1章_绪论",
            "num_questions": 6,
            "question_types": ["mcq", "short"],
            "difficulty": "medium",
            "llm_complete": jobs._complete_concept_notes,
        }
    ]


def _install_service_with_history(
    monkeypatch: pytest.MonkeyPatch, *, counts: dict
) -> list:
    calls: list = []

    class FakeStore:
        def spec_meta(self, spec_id: str) -> dict:
            return {"spec_id": spec_id}

        def history_counts(self, spec_id: str) -> dict:
            return dict(counts)

    class FakeService:
        store = FakeStore()

        def put_spec(
            self,
            spec,
            *,
            overwrite: bool = True,
            allow_history_rewrite: bool = False,
        ):
            calls.append((overwrite, allow_history_rewrite))
            return {
                "spec_id": "kb-x-1",
                "created": False,
                "quality": {"passed": True},
                "counts": {"items": 1},
            }

    monkeypatch.setattr(service_module, "ReviewService", FakeService)
    return calls


def _install_spec_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_generate(
        monkeypatch,
        generate_spec_from_kb=lambda **kwargs: types.SimpleNamespace(
            spec=types.SimpleNamespace(id="kb-x-1"),
            qa_count=1,
            skipped=0,
            warnings=[],
        ),
    )


def test_run_generation_refuses_to_replace_an_existing_pack(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_spec_stub(monkeypatch)
    calls = _install_service_with_history(
        monkeypatch, counts={"events": 1, "states": 1, "sessions": 1}
    )
    job = jobs.create_job(
        kb="kb-x",
        chapter="第1章_绪论",
        num_questions=1,
        question_types=None,
        difficulty="medium",
    )

    jobs._run_generation(job)

    stored = _read(review_home, job["job_id"])
    assert stored["status"] == "failed"
    assert "replace=true" in stored["error"]
    assert calls == []


def test_run_generation_replace_forwards_the_history_override(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_spec_stub(monkeypatch)
    calls = _install_service_with_history(
        monkeypatch, counts={"events": 2, "states": 2, "sessions": 1}
    )
    job = jobs.create_job(
        kb="kb-x",
        chapter="第1章_绪论",
        num_questions=1,
        question_types=None,
        difficulty="medium",
        replace=True,
    )

    jobs._run_generation(job)

    stored = _read(review_home, job["job_id"])
    assert stored["status"] == "done"
    assert calls == [(True, True)]
    assert any("2 review event(s)" in warning for warning in stored["warnings"])


def test_run_generation_fails_when_no_spec(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_generate(
        monkeypatch,
        generate_spec_from_kb=lambda **kwargs: types.SimpleNamespace(
            spec=None, qa_count=0, skipped=0
        ),
    )
    job = jobs.create_job(
        kb="kb-x",
        chapter="第9章_空章",
        num_questions=6,
        question_types=None,
        difficulty="medium",
    )

    jobs._run_generation(job)

    stored = _read(review_home, job["job_id"])
    assert stored["status"] == "failed"
    assert "no questions could be generated" in stored["error"]
    assert "第9章_空章" in stored["error"]


def test_run_generation_fails_on_quality_gate(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = QualityReport(
        passed=False,
        gates=[GateResult(gate="G3", passed=False, message="too few items")],
    )

    def fake_generate(**kwargs):
        raise QualityGateFailed(report)

    _install_generate(monkeypatch, generate_spec_from_kb=fake_generate)
    job = jobs.create_job(
        kb="kb-x",
        chapter="第1章",
        num_questions=6,
        question_types=None,
        difficulty="medium",
    )

    jobs._run_generation(job)

    stored = _read(review_home, job["job_id"])
    assert stored["status"] == "failed"
    assert "quality gates failed" in stored["error"]
    assert "G3" in stored["error"]
    assert "too few items" in stored["error"]
    assert len(stored["error"]) <= 300


def test_run_generation_fails_on_unexpected_exception(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_generate(**kwargs):
        raise RuntimeError("provider unreachable")

    _install_generate(monkeypatch, generate_spec_from_kb=fake_generate)
    job = jobs.create_job(
        kb="kb-x",
        chapter="第1章",
        num_questions=6,
        question_types=None,
        difficulty="medium",
    )

    jobs._run_generation(job)

    stored = _read(review_home, job["job_id"])
    assert stored["status"] == "failed"
    assert stored["error"] == "RuntimeError: provider unreachable"


def test_is_stale_only_for_active_jobs() -> None:
    now = 1_800_000_000.0

    fresh = {"status": "running", "updated_at": now - 60.0}
    assert jobs.is_stale(fresh, now=now) is False

    old = {"status": "running", "updated_at": now - jobs.JOB_STALE_SECONDS - 1.0}
    assert jobs.is_stale(old, now=now) is True

    old_but_done = {"status": "done", "updated_at": now - 10_000.0}
    assert jobs.is_stale(old_but_done, now=now) is False


def test_is_stale_covers_old_queued_jobs() -> None:
    now = 1_800_000_000.0

    fresh = {"status": "queued", "updated_at": now - 60.0}
    assert jobs.is_stale(fresh, now=now) is False

    old = {"status": "queued", "updated_at": now - jobs.JOB_STALE_SECONDS - 1.0}
    assert jobs.is_stale(old, now=now) is True


def test_is_stale_treats_nan_and_infinity_as_not_alive() -> None:
    now = 1_800_000_000.0

    for poisoned in (float("nan"), float("inf"), float("-inf")):
        job = {"status": "running", "updated_at": poisoned}
        assert jobs.is_stale(job, now=now) is True, poisoned


# -- Windows replace/read race (regression from the Phase 32 acceptance run) --


def test_save_job_survives_a_concurrent_reader(review_home: Path) -> None:
    """A poller holding the file open must not kill the writer.

    On Windows ``os.replace`` fails with WinError 5 while a reader keeps the
    target open, so ``save_job`` retries across the read window instead of
    raising and killing the generation thread.
    """
    job = jobs.create_job(
        kb="hf", chapter="第1章", num_questions=6, question_types=None, difficulty=""
    )
    target = review_home / "jobs" / f"{job['job_id']}.json"
    assert target.is_file()

    errors: list[BaseException] = []

    def writer() -> None:
        try:
            job["status"] = "running"
            jobs.save_job(job)
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
            errors.append(exc)

    handle = open(target, "rb")  # stands in for a poller mid-read
    thread = threading.Thread(target=writer)
    thread.start()
    time.sleep(0.05)  # let os.replace collide and enter its retry loop
    handle.close()
    thread.join(timeout=5.0)

    assert thread.is_alive() is False
    assert errors == []
    assert jobs.load_job(job["job_id"])["status"] == "running"


def test_concurrent_polls_never_see_a_phantom_missing_job(review_home: Path) -> None:
    """Readers never observe ``None`` or a torn record while the writer runs."""
    job = jobs.create_job(
        kb="hf", chapter="第1章", num_questions=6, question_types=None, difficulty=""
    )
    stop = threading.Event()
    missing: list[int] = []
    errors: list[BaseException] = []
    read_count = [0]
    first_read = threading.Event()

    def reader() -> None:
        try:
            while not stop.is_set():
                record = jobs.load_job(job["job_id"])
                read_count[0] += 1
                first_read.set()
                if record is None:
                    missing.append(1)
                time.sleep(0.0005)
        except BaseException as exc:
            errors.append(exc)
            first_read.set()

    def writer() -> None:
        try:
            for counter in range(60):
                job["qa_count"] = counter
                jobs.save_job(job)
                time.sleep(0.001)
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
            errors.append(exc)

    reader_thread = threading.Thread(target=reader)
    writer_thread = threading.Thread(target=writer)
    reader_thread.start()
    assert first_read.wait(timeout=5)
    writer_thread.start()
    writer_thread.join(timeout=10.0)
    stop.set()
    reader_thread.join(timeout=5.0)

    assert writer_thread.is_alive() is False
    assert reader_thread.is_alive() is False
    assert errors == []
    assert missing == []
    assert read_count[0] > 0


def test_load_job_retries_a_transient_read_error(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read that collides with a replace is retried, not reported missing."""
    job = jobs.create_job(
        kb="hf", chapter="第1章", num_questions=6, question_types=None, difficulty=""
    )
    real_read_text = Path.read_text
    target_name = f"{job['job_id']}.json"
    failures = {"count": 0}

    def flaky_read_text(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self.name == target_name and failures["count"] < 2:
            failures["count"] += 1
            raise PermissionError("the replace window")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)
    record = jobs.load_job(job["job_id"])

    assert failures["count"] == 2
    assert record is not None
    assert record["job_id"] == job["job_id"]


def test_runner_stops_quietly_when_the_store_stays_unwritable(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persistent write failure must not crash the thread or fake progress."""
    job = jobs.create_job(
        kb="hf", chapter="第1章", num_questions=6, question_types=None, difficulty=""
    )
    calls = {"count": 0}

    def always_locked(_job: dict) -> None:
        calls["count"] += 1
        raise PermissionError("the store never releases")

    monkeypatch.setattr(jobs, "save_job", always_locked)
    jobs._run_generation(job)  # must return, not raise

    assert calls["count"] == 1
    assert job["status"] == "running"  # in-memory only; never persisted
    assert jobs.load_job(job["job_id"])["status"] == "queued"


def test_save_job_gives_up_after_the_retry_window_without_litter(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = jobs.create_job(
        kb="hf", chapter="第1章", num_questions=6, question_types=None, difficulty=""
    )

    def always_locked(_src: str, _dst: str) -> None:
        raise PermissionError("never releases")

    monkeypatch.setattr(jobs.os, "replace", always_locked)
    with pytest.raises(OSError):
        jobs.save_job(job)

    assert list((review_home / "jobs").glob(".*.tmp")) == []


def test_run_generation_retries_once_on_a_network_error(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transport failure gets one more attempt; the second result stands."""
    calls = {"count": 0}

    class APITimeoutError(RuntimeError):
        pass

    def flaky_generate(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise APITimeoutError("the provider timed out")
        return types.SimpleNamespace(spec=object(), qa_count=6, skipped=0, warnings=[])

    _install_generate(monkeypatch, generate_spec_from_kb=flaky_generate)
    _install_service(
        monkeypatch,
        response={
            "spec_id": "kb-x-1",
            "created": True,
            "quality": {"passed": True},
            "counts": {"items": 6},
        },
    )
    job = jobs.create_job(
        kb="kb-x",
        chapter="第1章",
        num_questions=6,
        question_types=None,
        difficulty="",
    )

    jobs._run_generation(job)

    assert calls["count"] == 2
    assert _read(review_home, job["job_id"])["status"] == "done"


def test_run_generation_does_not_retry_other_errors(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0}

    def broken_generate(**kwargs):
        calls["count"] += 1
        raise ValueError("the chapter is wrong")

    _install_generate(monkeypatch, generate_spec_from_kb=broken_generate)
    job = jobs.create_job(
        kb="kb-x",
        chapter="第1章",
        num_questions=6,
        question_types=None,
        difficulty="",
    )

    jobs._run_generation(job)

    assert calls["count"] == 1
    stored = _read(review_home, job["job_id"])
    assert stored["status"] == "failed"
    assert "ValueError" in stored["error"]


def test_start_generation_job_copies_the_context_into_the_thread(
    review_home: Path,
) -> None:
    """The request's ContextVars (the user scope) must reach the runner.

    A bare thread would reset them, resolving review_home and the KB root to
    the default scope and silently writing the pack into the wrong store.
    """
    import contextvars

    scope = contextvars.ContextVar("scope")
    scope.set("ws-user-42")
    seen: list[str | None] = []
    finished = threading.Event()

    def runner(_job: dict) -> None:
        seen.append(scope.get(None))
        finished.set()

    jobs.start_generation_job(
        kb="hf",
        chapter="第1章",
        num_questions=6,
        question_types=None,
        difficulty="",
        runner=runner,
    )

    assert finished.wait(timeout=5.0)
    assert seen == ["ws-user-42"]


def test_load_job_raises_on_a_persistent_store_error(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store that stays unreadable is an outage, not a missing job."""
    job = jobs.create_job(
        kb="hf", chapter="第1章", num_questions=6, question_types=None, difficulty=""
    )

    def always_locked(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise PermissionError("held forever")

    monkeypatch.setattr(Path, "read_text", always_locked)
    with pytest.raises(OSError):
        jobs.load_job(job["job_id"])
