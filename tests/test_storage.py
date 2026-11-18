from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
from pydantic import ValidationError

from deeptutor.learning.models import RepetitionState
from deeptutor.review.adapter import initial_state, replay_state, scheduler_for
from deeptutor.review.fixture import build_fixture_spec
from deeptutor.review.service import AttemptInput, ReviewService
from deeptutor.review.storage import (
    ReviewStore,
    StateTransition,
    ReviewStoreError,
    ReviewStoreIOError,
    SessionConflict,
    SpecHistoryConflict,
    SubmissionConflict,
)

NOW = 1_800_000_000.0


def test_put_spec_is_idempotent(service: ReviewService) -> None:
    spec = build_fixture_spec()
    first = service.put_spec(spec)
    second = service.put_spec(spec)
    assert first["created"] is True
    assert second["created"] is False
    assert first["content_hash"] == second["content_hash"]
    assert service.store.spec_count() == 1


def _changed_payload(spec, title: str) -> dict:
    payload = spec.model_dump(mode="json")
    payload["title"] = title
    payload["content_hash"] = ""
    return payload


def _attempt(service: ReviewService, spec, submission_id: str = "history-1") -> None:
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="good",
            session_id="history",
            submission_id=submission_id,
        ),
        now=NOW,
    )


def test_content_rewrite_of_a_pack_with_history_needs_explicit_flag(
    service: ReviewService,
) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    _attempt(service, spec)

    with pytest.raises(SpecHistoryConflict) as excinfo:
        service.put_spec(_changed_payload(spec, "改标题"))
    assert excinfo.value.counts["events"] == 1
    assert service.store.get_spec(spec.id).title == spec.title

    service.put_spec(
        _changed_payload(spec, "改标题"), allow_history_rewrite=True
    )
    assert service.store.get_spec(spec.id).title == "改标题"
    assert service.store.event_count(spec.id) == 1


def test_same_content_refresh_keeps_history_without_flag(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    _attempt(service, spec)
    refreshed = service.put_spec(spec)
    assert refreshed["created"] is False
    assert service.store.event_count(spec.id) == 1


def test_content_rewrite_without_history_is_unchanged(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    result = service.put_spec(_changed_payload(spec, "改标题"))
    assert result["created"] is False
    assert service.store.get_spec(spec.id).title == "改标题"


def test_phase_05_spec_json_without_linkage_fields_still_loads(
    service: ReviewService,
) -> None:
    """An old stored row has no additive fields; its old hash must still verify."""
    from deeptutor.review.models import ReviewSpec

    spec = build_fixture_spec()
    old_payload = spec.model_dump(mode="json")
    old_payload.pop("mastery_path_id", None)
    for concept in old_payload["concepts"]:
        concept.pop("knowledge_point_id", None)
    old_payload["content_hash"] = ""
    old = ReviewSpec.model_validate(old_payload)
    old_payload["content_hash"] = old.content_hash

    conn = sqlite3.connect(service.store.home / "review.sqlite3")
    try:
        conn.execute(
            "INSERT INTO review_specs(spec_id, content_hash, created_at, updated_at, spec_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                spec.id,
                old.content_hash,
                NOW,
                NOW,
                json.dumps(old_payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    loaded = service.store.get_spec(spec.id)
    assert loaded.content_hash == old.content_hash
    assert loaded.mastery_path_id == ""
    assert all(concept.knowledge_point_id == "" for concept in loaded.concepts)


def test_put_spec_revalidates_model_copies(service: ReviewService) -> None:
    spec = build_fixture_spec()
    altered = spec.model_copy(update={"title": "changed without revalidation"})
    with pytest.raises(ValidationError):
        service.put_spec(altered)
    assert service.store.spec_count() == 0


def test_spec_json_is_written_atomically(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    path = service.store.home / f"review_{spec.id}" / "spec.json"
    assert path.is_file()
    assert spec.content_hash in path.read_text(encoding="utf-8")
    assert not list(path.parent.glob("*.tmp"))


def test_spec_snapshot_write_failure_does_not_commit_db(
    service: ReviewService, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ReviewStore, "_write_atomic", staticmethod(fail))
    with pytest.raises(ReviewStoreIOError):
        service.put_spec(build_fixture_spec())
    assert service.store.spec_count() == 0


def test_atomic_write_fsyncs_before_replacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[int] = []
    real_fsync = os.fsync

    def spy(descriptor: int) -> None:
        synced.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", spy)
    target = tmp_path / "spec.json"
    ReviewStore._write_atomic(target, "{}\n")
    # The file is flushed before the rename; POSIX also fsyncs the directory
    # so the rename itself survives a power loss.
    assert len(synced) == (2 if os.name == "posix" else 1)
    assert target.read_text(encoding="utf-8") == "{}\n"


def test_snapshot_is_written_under_the_cross_process_write_lock(
    service: ReviewService, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ``BEGIN IMMEDIATE`` is what serialises writers across processes. The
    # snapshot must be written while that lock is held, otherwise a second
    # process could replace the file between the row check and the commit.
    real_write = ReviewStore._write_atomic
    observed: list[str] = []

    def probing_write(path: Path, text: str) -> None:
        other = sqlite3.connect(
            service.store.db_path, timeout=0.2, isolation_level=None
        )
        try:
            with pytest.raises(sqlite3.OperationalError):
                other.execute("BEGIN IMMEDIATE")
        finally:
            other.close()
        observed.append(path.name)
        real_write(path, text)

    monkeypatch.setattr(ReviewStore, "_write_atomic", staticmethod(probing_write))
    service.put_spec(build_fixture_spec())
    assert observed == ["spec.json"]


def test_event_history_is_bounded(service: ReviewService, monkeypatch: pytest.MonkeyPatch) -> None:
    import deeptutor.review.storage as storage

    monkeypatch.setattr(storage, "MAX_EVENTS_PER_CONCEPT", 1)
    spec = build_fixture_spec()
    service.put_spec(spec)
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="good",
            session_id="bounded",
            submission_id="bounded-1",
        ),
        now=NOW,
    )
    with pytest.raises(ReviewStoreError):
        service.attempt(
            spec.id,
            AttemptInput(
                item_id="item-resonance-numeric",
                rating="good",
                session_id="bounded",
                submission_id="bounded-2",
            ),
            now=NOW + 1,
        )
    assert service.store.event_count(spec.id) == 1


def test_events_are_append_only(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="good",
            session_id="append-only",
            submission_id="sub-append",
        ),
        now=NOW,
    )
    with sqlite3.connect(service.store.db_path) as conn:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("UPDATE review_events SET rating = 'again'")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM review_events")


def test_same_submission_is_idempotent(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    request = AttemptInput(
        item_id="item-resonance-choice",
        rating="good",
        session_id="idem",
        submission_id="sub-idem",
        answer=1,
    )
    first = service.attempt(spec.id, request, now=NOW)
    second = service.attempt(spec.id, request, now=NOW + 5)
    assert first["created"] is True
    assert second["created"] is False
    assert service.store.event_count(spec.id) == 1
    assert second["next_due_ts"] == pytest.approx(first["next_due_ts"])


def test_idempotency_ignores_elapsed_time(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    base = {
        "item_id": "item-resonance-choice",
        "rating": "good",
        "session_id": "timing",
        "submission_id": "sub-timing",
    }
    first = service.attempt(spec.id, AttemptInput(**base, elapsed_ms=100), now=NOW)
    second = service.attempt(spec.id, AttemptInput(**base, elapsed_ms=5000), now=NOW + 3)
    assert first["created"] is True
    assert second["created"] is False
    assert service.store.event_count(spec.id) == 1


def test_same_submission_with_different_payload_conflicts(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="good",
            session_id="conflict",
            submission_id="sub-conflict",
        ),
        now=NOW,
    )
    with pytest.raises(SubmissionConflict):
        service.attempt(
            spec.id,
            AttemptInput(
                item_id="item-resonance-choice",
                rating="again",
                session_id="conflict",
                submission_id="sub-conflict",
            ),
            now=NOW + 1,
        )
    assert service.store.event_count(spec.id) == 1


def test_state_version_increments(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    concept = spec.concepts[0]
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="good",
            session_id="version",
            submission_id="sub-v1",
        ),
        now=NOW,
    )
    first = service.store.get_state_row(spec.id, concept.key)
    assert first is not None and first["version"] == 1
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-numeric",
            rating="hard",
            session_id="version",
            submission_id="sub-v2",
        ),
        now=NOW + 1,
    )
    second = service.store.get_state_row(spec.id, concept.key)
    assert second is not None and second["version"] == 2


def test_replay_matches_materialized_state(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    concept = spec.concepts[0]
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="good",
            session_id="replay",
            submission_id="sub-replay-1",
        ),
        now=NOW,
    )
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-numeric",
            rating="again",
            session_id="replay",
            submission_id="sub-replay-2",
        ),
        now=NOW + 10,
    )
    row = service.store.get_state_row(spec.id, concept.key)
    assert row is not None
    materialized = RepetitionState.model_validate_json(row["state_json"])
    replayed = replay_state(
        concept,
        service.store.events_for_concept(spec.id, concept.key),
        now=NOW + 10,
        scheduler=scheduler_for(spec),
    )
    assert replayed.next_review_at == pytest.approx(materialized.next_review_at)
    assert replayed.review_count == materialized.review_count
    assert replayed.lapse_count == materialized.lapse_count


def test_replay_uses_append_order_not_wall_clock(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    concept = spec.concepts[0]
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="good",
            session_id="clock",
            submission_id="clock-1",
        ),
        now=NOW + 100,
    )
    # A backwards clock adjustment must not reorder the event log: the second
    # append is still the second state transition.
    service.attempt(
        spec.id,
        AttemptInput(
            item_id="item-resonance-numeric",
            rating="again",
            session_id="clock",
            submission_id="clock-2",
        ),
        now=NOW,
    )
    row = service.store.get_state_row(spec.id, concept.key)
    assert row is not None
    materialized = RepetitionState.model_validate_json(row["state_json"])
    replayed = replay_state(
        concept,
        service.store.events_for_concept(spec.id, concept.key),
        now=NOW + 100,
        scheduler=scheduler_for(spec),
    )
    assert replayed.next_review_at == pytest.approx(materialized.next_review_at)
    assert replayed.lapse_count == materialized.lapse_count


def test_session_belongs_to_one_spec(store: ReviewStore) -> None:
    store.ensure_session("spec-a", "shared-session", 10, now=NOW)
    with pytest.raises(SessionConflict):
        store.ensure_session("spec-b", "shared-session", 10, now=NOW)


def test_open_session_is_atomic_across_store_instances(store: ReviewStore) -> None:
    import threading

    # Separate stores stand in for separate workers: each call opens its own
    # connection, so only the database write lock can serialise them.
    stores = [ReviewStore(store.home) for _ in range(2)]
    barrier = threading.Barrier(6)
    results: list[tuple[str, bool]] = []
    lock = threading.Lock()

    def join_or_create(index: int) -> None:
        barrier.wait()
        row, created = stores[index % 2].open_session(
            "spec-race", f"candidate-{index}", 10, now=NOW
        )
        with lock:
            results.append((str(row["session_id"]), created))

    threads = [threading.Thread(target=join_or_create, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == len(threads)
    sessions = {session_id for session_id, _ in results}
    assert len(sessions) == 1
    assert sum(1 for _, created in results if created) == 1
    open_row = store.latest_open_session("spec-race", now=NOW)
    assert open_row is not None and str(open_row["session_id"]) in sessions


def test_open_session_rolls_over_an_expired_sitting(store: ReviewStore) -> None:
    from deeptutor.review.storage import MAX_OPEN_SESSION_SECONDS

    store.ensure_session("spec-roll", "old-sitting", 10, now=NOW)
    row, created = store.open_session(
        "spec-roll", "fresh-sitting", 10, now=NOW + MAX_OPEN_SESSION_SECONDS + 1
    )
    assert created is True
    assert row["session_id"] == "fresh-sitting"
    old = store.get_session("old-sitting")
    assert old is not None
    assert old["ended_at"] == NOW + MAX_OPEN_SESSION_SECONDS


def test_complete_session_is_idempotent(service: ReviewService) -> None:
    spec = build_fixture_spec()
    service.put_spec(spec)
    service.today(spec.id, session_id="complete", now=NOW)
    first = service.complete(spec.id, "complete", now=NOW + 60)
    second = service.complete(spec.id, "complete", now=NOW + 120)
    assert first["ended_at"] == second["ended_at"]


def test_newer_database_schema_is_refused(tmp_path: Path) -> None:
    home = tmp_path / "review"
    store = ReviewStore(home)
    conn = sqlite3.connect(store.db_path)
    try:
        conn.execute("ALTER TABLE review_events DROP COLUMN learner_quote")
        conn.execute(
            "INSERT INTO review_schema_migrations (version, applied_at) VALUES (999, 0)"
        )
        conn.commit()
        before = list(conn.iterdump())
    finally:
        conn.close()
    with pytest.raises(ReviewStoreError, match="newer"):
        ReviewStore(home)
    with closing(sqlite3.connect(store.db_path)) as conn:
        assert list(conn.iterdump()) == before


def test_v4_migration_is_serialized_across_processes(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "review")
    with closing(sqlite3.connect(store.db_path)) as conn, conn:
        conn.execute("ALTER TABLE review_events DROP COLUMN learner_quote")
        conn.execute("UPDATE review_schema_migrations SET version = 4")

    worker = """
import sys
import time
from pathlib import Path

import deeptutor
deeptutor.__path__.insert(0, sys.argv[1])
from deeptutor.review.storage import ReviewStore

signal = Path(sys.argv[3])
original_probe = ReviewStore._needs_migration

def synchronized_probe(store):
    needs_migration = original_probe(store)
    signal.with_suffix('.ready').touch()
    deadline = time.monotonic() + 15
    while not signal.with_name('start').exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert signal.with_name('start').exists()
    return needs_migration

ReviewStore._needs_migration = synchronized_probe
ReviewStore(sys.argv[2])
"""
    import time

    source = Path(__file__).resolve().parents[1] / "src" / "deeptutor"
    processes = [
        subprocess.Popen(
            [sys.executable, "-B", "-c", worker, str(source), str(store.home), str(tmp_path / f"worker-{index}")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for index in range(6)
    ]
    try:
        deadline = time.monotonic() + 12
        while len(list(tmp_path.glob("*.ready"))) < len(processes) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(list(tmp_path.glob("*.ready"))) == len(processes)
        (tmp_path / "start").touch()
        results = [process.communicate(timeout=15) for process in processes]
        assert [process.returncode for process in processes] == [0] * len(processes), results
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    with closing(sqlite3.connect(store.db_path)) as conn:
        assert conn.execute("SELECT version FROM review_schema_migrations ORDER BY version").fetchall() == [(4,), (5,)]
        assert [column[1] for column in conn.execute("PRAGMA table_info(review_events)")].count("learner_quote") == 1
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_v2_database_migrates_in_place_without_losing_events(tmp_path: Path) -> None:
    # RS-202: a store written before the v3 columns existed must open, gain the
    # columns with their defaults, and keep every event and spec row.
    home = tmp_path / "review"
    store = ReviewStore(home)
    spec = build_fixture_spec()
    store.put_spec(spec)
    conn = sqlite3.connect(store.db_path)
    try:
        conn.execute(
            "INSERT INTO review_events (event_id, submission_id, spec_id, item_id, "
            "concept_key, session_id, ts, rating, result, quality, "
            "scheduled_interval_days, predicted_retrievability, app_version, "
            "schema_version, payload_hash) "
            "VALUES ('legacy-event', 'legacy-submission', ?, ?, ?, 'legacy-sitting', "
            "?, 'good', 'correct', 0.9, 3.0, 0.8, '1.7.2', '1.0', 'legacy-hash')",
            (spec.id, spec.items[0].id, spec.concepts[0].key, NOW),
        )
        # Recreate the v2 shape: the v3 columns are not there yet.
        conn.execute("ALTER TABLE review_events DROP COLUMN hints_used")
        conn.execute("ALTER TABLE review_events DROP COLUMN confidence")
        conn.execute("DELETE FROM review_schema_migrations WHERE version >= 3")
        conn.execute(
            "INSERT INTO review_schema_migrations (version, applied_at) VALUES (2, 0)"
        )
        conn.commit()
    finally:
        conn.close()

    reopened = ReviewStore(home)
    with reopened._connect() as conn:
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(review_events)")
        }
        legacy = conn.execute(
            "SELECT * FROM review_events WHERE event_id = 'legacy-event'"
        ).fetchone()
    assert {"hints_used", "confidence"} <= columns
    assert legacy is not None
    assert legacy["hints_used"] == 0
    assert legacy["confidence"] is None
    assert reopened.spec_count() == 1
    assert [event["event_id"] for event in reopened.events_for_spec(spec.id)] == [
        "legacy-event"
    ]


def test_unsafe_spec_id_is_rejected(service: ReviewService, tmp_path: Path) -> None:
    with pytest.raises(Exception):
        service.put_spec({"id": "../escape", "title": "x"})
    # Colon is valid in a URL but invalid in a Windows path component.
    with pytest.raises(Exception):
        service.put_spec({"id": "review:1", "title": "x"})


def test_last_activity_uses_events_not_views(store) -> None:
    from deeptutor.review.fixture import build_fixture_spec

    spec = build_fixture_spec()
    assert store.last_activity_ts(spec.id) is None
    store.record_views(spec.id, "views-only", ["item-resonance-choice"], now=100.0)
    assert store.last_activity_ts(spec.id) is None


def test_events_for_spec_honours_since_and_limit(store: ReviewStore) -> None:
    spec = build_fixture_spec()
    store.put_spec(spec)
    concept_key = spec.concepts[0].key

    def add(submission: str, ts: float) -> None:
        store.append_event(
            submission_id=submission,
            spec_id=spec.id,
            item_id=spec.items[0].id,
            concept_key=concept_key,
            session_id="s1",
            ts=ts,
            rating="good",
            result="correct",
            quality=1.0,
            elapsed_ms=0,
            payload_hash=submission,
            app_version="test",
            transition=lambda _state: StateTransition(
                state=initial_state(spec.concepts[0], now=ts),
                predicted_retrievability=0.9,
                scheduled_interval_days=1.0,
            ),
        )

    add("e1", 1_000.0)
    add("e2", 2_000.0)
    add("e3", 3_000.0)

    assert [row["submission_id"] for row in store.events_for_spec(spec.id)] == [
        "e1",
        "e2",
        "e3",
    ]
    assert [row["submission_id"] for row in store.events_for_spec(spec.id, since=2_000.0)] == [
        "e2",
        "e3",
    ]
    # A cap keeps the newest rows, then restores oldest-first order: dropping
    # the newest events would blind every recent-window aggregate.
    assert [row["submission_id"] for row in store.events_for_spec(spec.id, limit=2)] == [
        "e2",
        "e3",
    ]


def test_sessions_for_spec_includes_a_session_that_ended_in_window(
    store: ReviewStore,
) -> None:
    spec = build_fixture_spec()
    store.put_spec(spec)
    # Started long before the window, finished inside it: its answers are in the
    # window, so excluding it would divide one window by another.
    store.ensure_session(spec.id, "old-but-finished", 10, now=1_000.0)
    store.complete_session("old-but-finished", now=5_000.0)
    # Fully outside the window on both ends.
    store.ensure_session(spec.id, "finished-before", 10, now=100.0)
    store.complete_session("finished-before", now=200.0)
    # Still open.
    store.ensure_session(spec.id, "open", 10, now=1_500.0)

    since = 4_000.0
    ids = [row["session_id"] for row in store.sessions_for_spec(spec.id, since=since)]

    assert "old-but-finished" in ids
    assert "open" in ids
    assert "finished-before" not in ids
