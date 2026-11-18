"""SQLite persistence for Review Studio.

Design rules enforced here, not in callers:

* ``review_events`` is append-only. Database triggers reject UPDATE/DELETE,
  so a bug cannot rewrite history.
* ``review_state`` is a materialized view with a monotonically increasing
  ``version``. Writers run inside ``BEGIN IMMEDIATE`` so check-and-set is
  atomic across processes.
* A repeated ``submission_id`` with the same payload returns the original
  event; a different payload is a conflict, never a silent overwrite.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from deeptutor.learning.models import RepetitionState

from deeptutor.review.models import ReviewSpec, canonical_json

SCHEMA_VERSION = 5
SQLITE_TIMEOUT_SECONDS = 5.0
MAX_EVENTS_PER_CONCEPT = 10_000
#: An unfinished session older than this is closed automatically. Chat-only
#: clients have no "complete session" command, and a sitting that spans weeks
#: makes completed/session counts meaningless. Three days matches the queue's
#: recovery horizon, so a deliberate multi-day resume still stays in one
#: sitting.
MAX_OPEN_SESSION_SECONDS = 72 * 60 * 60
#: Gaps longer than this between two session activities count as idle time,
#: not study time. Completion reports the sum of the capped gaps.
ACTIVE_GAP_SECONDS = 5 * 60
# Colons are deliberately excluded: they are legal in a URL but invalid in a
# Windows filename, and both spec ids and session ids become path components.
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

#: Process-local write locks keyed by ``(home, spec_id)``. ``ReviewService`` is
#: constructed per request, so the lock must live at module scope to serialise
#: concurrent writers within one process.
_SPEC_WRITE_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_SPEC_WRITE_LOCKS_GUARD = threading.Lock()


def _spec_write_lock(home: Path, spec_id: str) -> threading.Lock:
    key = (str(home.resolve()), spec_id)
    with _SPEC_WRITE_LOCKS_GUARD:
        lock = _SPEC_WRITE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SPEC_WRITE_LOCKS[key] = lock
        return lock


def _history_counts(conn: sqlite3.Connection, spec_id: str) -> dict[str, int]:
    row = conn.execute(
        "SELECT (SELECT COUNT(*) FROM review_events WHERE spec_id = ?) AS events, "
        "(SELECT COUNT(*) FROM review_state WHERE spec_id = ?) AS states, "
        "(SELECT COUNT(*) FROM review_sessions WHERE spec_id = ?) AS sessions",
        (spec_id, spec_id, spec_id),
    ).fetchone()
    return {key: int(row[key]) for key in ("events", "states", "sessions")}


class ReviewStoreError(RuntimeError):
    pass


class ReviewStoreIOError(ReviewStoreError):
    pass


class ReadOnlyStoreUnavailable(ReviewStoreIOError):
    pass


class SpecNotFound(ReviewStoreError):
    pass


class SpecIntegrityError(ReviewStoreError):
    """A row's JSON describes a different spec than the row key."""


class SpecExists(ReviewStoreError):
    """A non-overwriting write hit a spec id that is already stored."""

    def __init__(self, spec_id: str) -> None:
        super().__init__(f"spec {spec_id!r} already exists")
        self.spec_id = spec_id


class SpecHistoryConflict(ReviewStoreError):
    """A content-changing overwrite would strand the pack's review history."""

    def __init__(
        self,
        spec_id: str,
        *,
        events: int,
        states: int,
        sessions: int,
    ) -> None:
        super().__init__(
            f"spec {spec_id!r} already has {events} review event(s), "
            f"{states} state row(s) and {sessions} session(s); "
            "rewriting its content requires an explicit history rewrite"
        )
        self.spec_id = spec_id
        self.counts = {"events": events, "states": states, "sessions": sessions}


class SessionNotFound(ReviewStoreError):
    pass


class SessionConflict(ReviewStoreError):
    pass


class SubmissionConflict(ReviewStoreError):
    def __init__(self, message: str, existing: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.existing = existing


class VersionConflict(ReviewStoreError):
    pass


@dataclass(frozen=True)
class StateTransition:
    state: RepetitionState
    predicted_retrievability: float
    scheduled_interval_days: float


TransitionFn = Callable[[str | None], StateTransition]


def review_home(explicit: str | Path | None = None) -> Path:
    """Resolve the review root.

    ``DEEPTUTOR_REVIEW_HOME`` exists so tests and acceptance runs never write
    into a learner's real workspace. Without the override the path comes from
    the core per-user path service (admin fallback outside a request).
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    override = os.environ.get("DEEPTUTOR_REVIEW_HOME")
    if override:
        return Path(override).expanduser().resolve()
    from deeptutor.services.path_service import get_path_service

    return (get_path_service().get_workspace_dir() / "review").resolve()


def _safe_component(value: str) -> str:
    if not _SAFE_ID.match(value):
        raise ValueError(f"unsafe id for a filesystem path: {value!r}")
    # Windows ignores trailing dots/spaces, so "abc" and "abc." would resolve
    # to the same directory and overwrite each other's snapshot.
    if value.endswith(".") or value.endswith(" "):
        raise ValueError(f"unsafe id for a filesystem path: {value!r}")
    return value


def _plan_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    plan = json.loads(row["plan_json"])
    plan.setdefault("plan_id", row["plan_id"])
    plan.setdefault("subject", row["subject"])
    plan.setdefault("status", row["status"])
    plan["created_at"] = row["created_at"]
    plan["updated_at"] = row["updated_at"]
    return plan


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class ReviewStore:
    def __init__(self, home: str | Path | None = None, *, read_only: bool = False) -> None:
        self.home = review_home(home)
        self.read_only = read_only
        self.db_path = self.home / "review.sqlite3"
        if read_only:
            if not self.db_path.is_file():
                raise ReadOnlyStoreUnavailable("review database is missing")
            if self._needs_migration():
                raise ReadOnlyStoreUnavailable("review database needs a schema migration")
            return
        self.home.mkdir(parents=True, exist_ok=True)
        # Checking the schema version is read-only. Re-running migrations on
        # every request would take the single SQLite write lock just to read.
        if self._needs_migration():
            self._migrate()

    def _needs_migration(self) -> bool:
        if not self.db_path.exists():
            return True
        try:
            with self._read_connection() as conn:
                return self._schema_version(conn) != SCHEMA_VERSION
        except sqlite3.Error as exc:
            raise ReviewStoreIOError("review store schema could not be read") from exc

    @staticmethod
    def _schema_version(conn: sqlite3.Connection) -> int | None:
        migration_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='review_schema_migrations'"
        ).fetchone()
        if migration_table is None:
            return None
        (version,) = conn.execute("SELECT MAX(version) FROM review_schema_migrations").fetchone()
        if version is not None and version > SCHEMA_VERSION:
            raise ReviewStoreError(
                f"review database schema {version} is newer than this package "
                f"({SCHEMA_VERSION}); refusing to downgrade"
            )
        return version

    # -- connection ---------------------------------------------------------
    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(
            self.db_path.as_uri() + "?mode=ro", uri=True, timeout=SQLITE_TIMEOUT_SECONDS
        )) as conn:
            conn.row_factory = sqlite3.Row
            yield conn

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            with self._read_connection() as conn:
                yield conn
            return
        with closing(sqlite3.connect(
            self.db_path, timeout=SQLITE_TIMEOUT_SECONDS, isolation_level=None
        )) as conn:
            conn.row_factory = sqlite3.Row
            self._enable_wal(conn)
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA recursive_triggers=ON")
            yield conn

    @staticmethod
    def _enable_wal(conn: sqlite3.Connection) -> None:
        deadline = time.monotonic() + SQLITE_TIMEOUT_SECONDS
        while True:
            try:
                if conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal":
                    return
                conn.execute("PRAGMA journal_mode=WAL").fetchone()
                return
            except sqlite3.OperationalError as exc:
                if (
                    exc.sqlite_errorcode not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
                    or time.monotonic() >= deadline
                ):
                    raise
                time.sleep(0.02)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def _migrate(self) -> None:
        try:
            self._migrate_locked()
        except sqlite3.OperationalError as exc:
            # A read-only replica raises here: the store is unavailable, not
            # conflicting, so raise the IO variant the API maps to 503.
            raise ReviewStoreIOError(
                f"review store cannot be migrated read-write: {exc}"
            ) from exc

    def _migrate_locked(self) -> None:
        with closing(sqlite3.connect(self.db_path, timeout=SQLITE_TIMEOUT_SECONDS)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            if self._schema_version(conn) == SCHEMA_VERSION:
                return
            ddl = """
                CREATE TABLE IF NOT EXISTS review_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS review_specs (
                    spec_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    spec_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS review_events (
                    event_id TEXT PRIMARY KEY,
                    submission_id TEXT NOT NULL UNIQUE,
                    spec_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    concept_key TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    ts REAL NOT NULL,
                    rating TEXT NOT NULL CHECK (rating IN ('again', 'hard', 'good')),
                    result TEXT NOT NULL,
                    quality REAL NOT NULL,
                    elapsed_ms INTEGER NOT NULL DEFAULT 0,
                    scheduled_interval_days REAL NOT NULL,
                    predicted_retrievability REAL NOT NULL,
                    source TEXT NOT NULL DEFAULT 'review',
                    app_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    payload_hash TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_review_events_spec_concept
                    ON review_events (spec_id, concept_key, ts);
                CREATE INDEX IF NOT EXISTS idx_review_events_session
                    ON review_events (session_id, ts);

                -- Write-side CAS cache only: the insert transaction reads the
                -- previous state to seed the transition. Every read surface
                -- (today/weakness/metrics/export) replays review_events, which
                -- is the source of truth; this table must never be consulted
                -- for a user-visible answer.
                CREATE TABLE IF NOT EXISTS review_state (
                    spec_id TEXT NOT NULL,
                    concept_key TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    next_review_at REAL NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (spec_id, concept_key)
                );

                CREATE TABLE IF NOT EXISTS review_sessions (
                    session_id TEXT PRIMARY KEY,
                    spec_id TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    planned_minutes INTEGER NOT NULL DEFAULT 0
                );

                -- The metrics endpoint reads sessions by spec and window; without
                -- this index every call scans the whole table.
                CREATE INDEX IF NOT EXISTS idx_review_sessions_spec
                    ON review_sessions (spec_id, started_at);

                -- Study plans are deliberately mutable: the end date and the
                -- daily budget change while the learner is mid-course, which
                -- is exactly what ReviewSpec.plan (part of content_hash) cannot
                -- do without a history conflict. The JSON keeps the scope and
                -- the derived schedule; the rows below keep execution state.
                CREATE TABLE IF NOT EXISTS review_plans (
                    plan_id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('active','completed','cancelled')),
                    plan_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS review_daily_tasks (
                    plan_id TEXT NOT NULL,
                    local_date TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    kind TEXT NOT NULL CHECK (kind IN ('study','review','milestone')),
                    title TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    chapter TEXT NOT NULL DEFAULT '',
                    minutes INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL CHECK (status IN ('todo','done','skipped')),
                    calendar_event_id TEXT,
                    kanban_card_id TEXT,
                    completed_at REAL,
                    self_note TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (plan_id, local_date, seq)
                );

                CREATE INDEX IF NOT EXISTS idx_review_daily_tasks_date
                    ON review_daily_tasks (local_date, plan_id);

                CREATE TABLE IF NOT EXISTS review_notes (
                    note_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    chapter TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL,
                    body TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL CHECK (status IN ('draft','confirmed')),
                    document_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_review_notes_plan
                    ON review_notes (plan_id, chapter);

                -- Display metadata, not a review event. It lets a concept with
                -- more items than the daily slice rotate instead of starving
                -- the tail forever.
                CREATE TABLE IF NOT EXISTS review_views (
                    spec_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    shown_at REAL NOT NULL,
                    PRIMARY KEY (spec_id, session_id, item_id)
                );
                CREATE INDEX IF NOT EXISTS idx_review_views_spec_item
                    ON review_views (spec_id, item_id);

                CREATE TRIGGER IF NOT EXISTS review_events_no_update
                BEFORE UPDATE ON review_events
                BEGIN
                    SELECT RAISE(ABORT, 'review_events is append-only');
                END;

                CREATE TRIGGER IF NOT EXISTS review_events_no_delete
                BEFORE DELETE ON review_events
                BEGIN
                    SELECT RAISE(ABORT, 'review_events is append-only');
                END;
                """
            statement = ""
            for line in ddl.splitlines(keepends=True):
                statement += line
                if sqlite3.complete_statement(statement):
                    conn.execute(statement)
                    statement = ""
            # v3 columns: hint usage and learner confidence ride on the event
            # row. ALTER TABLE keeps existing databases readable; the check
            # makes the migration idempotent.
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(review_events)")
            }
            if "hints_used" not in columns:
                conn.execute(
                    "ALTER TABLE review_events ADD COLUMN hints_used INTEGER NOT NULL DEFAULT 0"
                )
            if "confidence" not in columns:
                conn.execute("ALTER TABLE review_events ADD COLUMN confidence REAL")
            # v5: the learner's own words behind the rating. Nullable on
            # purpose — older events have none, and a tool call that omits it
            # is not "empty", it is "not recorded". The column makes the
            # "never grade for the learner" rule auditable after the fact
            # instead of only promised in the prompt.
            if "learner_quote" not in columns:
                conn.execute("ALTER TABLE review_events ADD COLUMN learner_quote TEXT")
            conn.execute(
                "INSERT OR IGNORE INTO review_schema_migrations (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, time.time()),
            )

    # -- specs --------------------------------------------------------------
    def _spec_file(self, spec_id: str) -> Path:
        directory = self.home / f"review_{_safe_component(spec_id)}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "spec.json"

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        # ``mkstemp`` gives every writer its own temporary name. A fixed
        # ``spec.json.tmp`` makes two concurrent writers race on the same
        # file: one ``os.replace`` wins, the other raises FileNotFoundError.
        descriptor, temp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=path.name + "-", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                # Without this the rename can land while the bytes are still
                # only in the page cache: a crash then leaves an empty file
                # that the next read fails to parse.
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        if os.name == "posix":
            # Make the rename itself durable. Windows has no equivalent that
            # ``os.replace`` exposes; NTFS journals the metadata on its own.
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

    def put_spec(
        self,
        spec: ReviewSpec,
        *,
        overwrite: bool = True,
        allow_history_rewrite: bool = False,
    ) -> tuple[ReviewSpec, bool]:
        """Store a spec. Returns ``(stored_spec, created)``.

        Same content hash is idempotent; the derived quality report may still
        be refreshed because it is not part of the hash. ``overwrite=False``
        rejects an id that already has a row instead of replacing it; the row
        check and the write share the per-spec lock so a concurrent
        non-overwriting writer cannot slip between them.

        A content-changing overwrite of a spec that already has events or
        materialized state raises ``SpecHistoryConflict`` unless the caller
        passes ``allow_history_rewrite=True``: item/concept keys in existing
        history would otherwise point at a pack that no longer contains them.
        Sessions carry no such keys and do not block the rewrite.
        """
        stored = spec
        payload = canonical_json(stored)
        # File and DB row form one logical record: serialise writers per
        # spec so an interleaved pair cannot leave the snapshot describing
        # one version while the authoritative row describes another. The
        # process-local lock covers one service process; ``BEGIN IMMEDIATE``
        # (taken before the snapshot write) extends that to every process
        # sharing the database, because SQLite's write lock is cross-process.
        with _spec_write_lock(self.home, stored.id):
            moment = time.time()
            with self._transaction() as conn:
                row = conn.execute(
                    "SELECT content_hash, created_at FROM review_specs WHERE spec_id = ?",
                    (stored.id,),
                ).fetchone()
                if row is not None and not overwrite:
                    # Checked before the snapshot write: a rejected spec must
                    # not leave a file that no row accounts for.
                    raise SpecExists(stored.id)
                if (
                    row is not None
                    and stored.content_hash != row["content_hash"]
                    and not allow_history_rewrite
                ):
                    counts = _history_counts(conn, stored.id)
                    # Sessions carry no item/concept keys: they stay meaningful
                    # across a content rewrite. Only events and the state rows
                    # derived from them can be stranded.
                    if counts["events"] or counts["states"]:
                        raise SpecHistoryConflict(stored.id, **counts)
                # Write the human-readable snapshot before the authoritative
                # DB row: if this fails, the transaction rolls back and the
                # spec is simply not stored; never a DB row that claims a file
                # exists when it does not. Do not leak the absolute path in
                # the error message.
                try:
                    self._write_atomic(self._spec_file(stored.id), payload)
                except OSError as exc:
                    raise ReviewStoreIOError(
                        "could not write the spec snapshot"
                    ) from exc
                created = row is None
                if row is None:
                    conn.execute(
                        "INSERT INTO review_specs (spec_id, content_hash, created_at, updated_at, spec_json) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (stored.id, stored.content_hash, moment, moment, payload),
                    )
                else:
                    conn.execute(
                        "UPDATE review_specs SET content_hash = ?, updated_at = ?, spec_json = ? "
                        "WHERE spec_id = ?",
                        (stored.content_hash, moment, payload, stored.id),
                    )
        return stored, created

    def history_counts(self, spec_id: str) -> dict[str, int]:
        """Rows that reference this spec: events, materialized states, sessions."""
        with self._connect() as conn:
            return _history_counts(conn, spec_id)

    def get_spec(self, spec_id: str) -> ReviewSpec:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT spec_json FROM review_specs WHERE spec_id = ?", (spec_id,)
            ).fetchone()
        if row is None:
            raise SpecNotFound(spec_id)
        return self.decode_spec(spec_id, row["spec_json"])

    @staticmethod
    def decode_spec(spec_id: str, payload: str) -> ReviewSpec:
        spec = ReviewSpec.model_validate_json(payload)
        if spec.id != spec_id:
            # The row key is what every read/write path uses; an embedded id
            # that disagrees would make ``today`` write into a different pack.
            raise SpecIntegrityError(
                f"stored spec id {spec.id!r} does not match row key {spec_id!r}"
            )
        return spec

    def spec_page(self, *, limit: int, offset: int) -> tuple[int, list[dict[str, Any]]]:
        with self._read_connection() as conn:
            conn.execute("BEGIN")
            count = conn.execute("SELECT COUNT(*) FROM review_specs").fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM review_specs ORDER BY updated_at DESC, spec_id LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return count, [dict(row) for row in rows]

    @contextmanager
    def spec_rows(self) -> Iterator[Iterator[sqlite3.Row]]:
        with self._read_connection() as conn:
            yield iter(conn.execute("SELECT spec_id, spec_json FROM review_specs ORDER BY spec_id"))

    def spec_meta(self, spec_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT spec_id, content_hash, created_at, updated_at FROM review_specs "
                "WHERE spec_id = ?",
                (spec_id,),
            ).fetchone()
        if row is None:
            raise SpecNotFound(spec_id)
        return dict(row)

    def list_specs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT spec_id, content_hash, created_at, updated_at "
                "FROM review_specs ORDER BY created_at, spec_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def spec_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM review_specs").fetchone()
        return int(row["n"])

    # -- state --------------------------------------------------------------
    def get_state_row(self, spec_id: str, concept_key: str) -> dict[str, Any] | None:
        """Write-side predecessor for CAS; not a read-model API.

        Read surfaces replay ``review_events`` instead, so a stale row here
        cannot surface to the learner.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM review_state WHERE spec_id = ? AND concept_key = ?",
                (spec_id, concept_key),
            ).fetchone()
        return _row_dict(row)

    # -- events -------------------------------------------------------------
    def events_for_concept(self, spec_id: str, concept_key: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM review_events WHERE spec_id = ? AND concept_key = ? "
                "ORDER BY rowid",
                (spec_id, concept_key),
            ).fetchall()
        return [dict(row) for row in rows]

    @contextmanager
    def replay_events(self, spec_id: str) -> Iterator[Iterator[sqlite3.Row]]:
        with self._read_connection() as conn:
            yield iter(conn.execute(
                "SELECT concept_key, item_id, ts, result, quality, session_id "
                "FROM review_events WHERE spec_id = ? ORDER BY concept_key, rowid",
                (spec_id,),
            ))

    def item_activity(self, spec_id: str) -> dict[tuple[str, str], float]:
        with self._read_connection() as conn:
            rows = conn.execute(
                "SELECT concept_key, item_id, MAX(ts) FROM review_events "
                "WHERE spec_id = ? GROUP BY concept_key, item_id",
                (spec_id,),
            )
            return {(key, item): float(timestamp) for key, item, timestamp in rows}

    def event_by_submission(self, submission_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM review_events WHERE submission_id = ?", (submission_id,)
            ).fetchone()
        return _row_dict(row)

    def events_for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM review_events WHERE session_id = ? ORDER BY rowid",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def events_for_spec(
        self,
        spec_id: str,
        *,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """All events for one spec, oldest first.

        Read-only. ``MAX_EVENTS_PER_CONCEPT`` caps per-concept history, but one
        spec may hold up to ``MAX_CONTAINER_ITEMS`` concepts, so the theoretical
        worst case is tens of millions of rows — an aggregate endpoint must pass
        ``limit`` rather than load the table. ``since`` is pushed into SQL for
        the same reason.
        """
        query = "SELECT * FROM review_events WHERE spec_id = ?"
        params: list[Any] = [spec_id]
        if since is not None:
            query += " AND ts >= ?"
            params.append(float(since))
        if limit is None:
            query += " ORDER BY rowid"
        else:
            # A cap keeps the *newest* rows by event time: the oldest 50k
            # would drop exactly the events a recent-window aggregate is
            # about, and rowid order disagrees with time after a clock jump or
            # an import replay. Fetch newest first, then restore oldest-first.
            query += " ORDER BY ts DESC, rowid DESC LIMIT ?"
            params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        if limit is not None:
            rows = list(reversed(rows))
        return [dict(row) for row in rows]

    def sessions_for_spec(
        self, spec_id: str, *, since: float | None = None
    ) -> list[dict[str, Any]]:
        """Sessions that were active inside the window, oldest first.

        ``started_at >= since`` alone would drop a long session that began before
        the window and finished inside it, while its answers still count in the
        matching event query — the completion rate would then divide one window's
        completions by another window's session count. An unfinished session is
        always included: it is active by definition.
        """
        query = "SELECT * FROM review_sessions WHERE spec_id = ?"
        params: list[Any] = [spec_id]
        if since is not None:
            query += " AND (ended_at IS NULL OR ended_at >= ? OR started_at >= ?)"
            params.extend([float(since), float(since)])
        query += " ORDER BY started_at, session_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def last_activity_ts(self, spec_id: str) -> float | None:
        """Last real answer timestamp; views and sessions do not count."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(ts) AS last_ts FROM review_events WHERE spec_id = ?",
                (spec_id,),
            ).fetchone()
        value = row["last_ts"] if row is not None else None
        return float(value) if value is not None else None

    def last_view_map(self, spec_id: str) -> dict[str, float]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT item_id, MAX(shown_at) AS shown_at FROM review_views "
                "WHERE spec_id = ? GROUP BY item_id",
                (spec_id,),
            ).fetchall()
        return {str(row["item_id"]): float(row["shown_at"]) for row in rows}

    def record_views(
        self, spec_id: str, session_id: str, item_ids: list[str], *, now: float
    ) -> None:
        unique = list(dict.fromkeys(item_ids))
        if not unique:
            return
        with self._transaction() as conn:
            conn.executemany(
                "INSERT INTO review_views (spec_id, session_id, item_id, shown_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(spec_id, session_id, item_id) "
                "DO UPDATE SET shown_at = excluded.shown_at",
                [(spec_id, session_id, item_id, now) for item_id in unique],
            )

    def event_count(self, spec_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM review_events WHERE spec_id = ?", (spec_id,)
            ).fetchone()
        return int(row["n"])

    def append_event(
        self,
        *,
        submission_id: str,
        spec_id: str,
        item_id: str,
        concept_key: str,
        session_id: str,
        ts: float,
        rating: str,
        result: str,
        quality: float,
        elapsed_ms: int,
        payload_hash: str,
        app_version: str,
        transition: TransitionFn,
        hints_used: int = 0,
        confidence: float | None = None,
        learner_quote: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Append one event and advance the materialized state atomically.

        ``transition`` receives the current state JSON (or ``None``) and returns
        the core scheduler's new state. It runs inside the write transaction,
        so two concurrent submissions for the same concept cannot both read
        the same version and overwrite each other.
        """
        with self._transaction() as conn:
            existing = conn.execute(
                "SELECT * FROM review_events WHERE submission_id = ?", (submission_id,)
            ).fetchone()
            if existing is not None:
                if existing["payload_hash"] == payload_hash:
                    return dict(existing), False
                raise SubmissionConflict(
                    "submission_id was already used with a different payload",
                    dict(existing),
                )

            # Re-check inside the write transaction: ``ensure_session`` ran in
            # an earlier transaction, so a concurrent ``complete_session`` can
            # slip between the two. An event must not land in a session that
            # has already been closed.
            session_row = conn.execute(
                "SELECT ended_at FROM review_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session_row is not None and session_row["ended_at"] is not None:
                raise SessionConflict(f"session {session_id} has already ended")

            history_count = conn.execute(
                "SELECT COUNT(*) AS n FROM review_events "
                "WHERE spec_id = ? AND concept_key = ?",
                (spec_id, concept_key),
            ).fetchone()
            if int(history_count["n"]) >= MAX_EVENTS_PER_CONCEPT:
                raise ReviewStoreError(
                    f"event history limit reached for {spec_id}/{concept_key}"
                )

            state_row = conn.execute(
                "SELECT * FROM review_state WHERE spec_id = ? AND concept_key = ?",
                (spec_id, concept_key),
            ).fetchone()
            started = transition(None if state_row is None else str(state_row["state_json"]))
            state_json = canonical_json(started.state)
            moment = time.time()
            event_id = uuid.uuid4().hex

            conn.execute(
                "INSERT INTO review_events ("
                "event_id, submission_id, spec_id, item_id, concept_key, session_id, ts, "
                "rating, result, quality, elapsed_ms, scheduled_interval_days, "
                "predicted_retrievability, source, app_version, schema_version, payload_hash, "
                "hints_used, confidence, learner_quote"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    submission_id,
                    spec_id,
                    item_id,
                    concept_key,
                    session_id,
                    ts,
                    rating,
                    result,
                    quality,
                    int(elapsed_ms),
                    started.scheduled_interval_days,
                    started.predicted_retrievability,
                    "review",
                    app_version,
                    "1.0",
                    payload_hash,
                    max(0, int(hints_used)),
                    confidence,
                    learner_quote,
                ),
            )

            if state_row is None:
                conn.execute(
                    "INSERT INTO review_state (spec_id, concept_key, version, next_review_at, "
                    "state_json, updated_at) VALUES (?, ?, 1, ?, ?, ?)",
                    (spec_id, concept_key, started.state.next_review_at, state_json, moment),
                )
            else:
                updated = conn.execute(
                    "UPDATE review_state SET version = version + 1, next_review_at = ?, "
                    "state_json = ?, updated_at = ? "
                    "WHERE spec_id = ? AND concept_key = ? AND version = ?",
                    (
                        started.state.next_review_at,
                        state_json,
                        moment,
                        spec_id,
                        concept_key,
                        int(state_row["version"]),
                    ),
                )
                if updated.rowcount != 1:
                    raise VersionConflict(
                        f"state version moved for {spec_id}/{concept_key}"
                    )

            row = conn.execute(
                "SELECT * FROM review_events WHERE event_id = ?", (event_id,)
            ).fetchone()
            return dict(row), True

    # -- sessions -----------------------------------------------------------
    def latest_open_session(
        self, spec_id: str, *, now: float | None = None
    ) -> dict[str, Any] | None:
        """The most recent unfinished session for a spec, if any.

        ``today`` reuses it instead of minting a session per call: otherwise
        every "what should I review?" inflates the session count and depresses
        the completion rate even though the learner had one sitting.

        A session older than ``MAX_OPEN_SESSION_SECONDS`` is closed first. The
        MCP surface has no "finish session" tool, so without this a learner in
        a chat-only deployment would stay in one session forever and every
        answer for weeks would be counted as a single sitting.
        """
        moment = time.time() if now is None else now
        cutoff = moment - MAX_OPEN_SESSION_SECONDS
        with self._transaction() as conn:
            conn.execute(
                "UPDATE review_sessions SET ended_at = started_at + ? "
                "WHERE spec_id = ? AND ended_at IS NULL AND started_at < ?",
                (MAX_OPEN_SESSION_SECONDS, spec_id, cutoff),
            )
            row = conn.execute(
                "SELECT * FROM review_sessions WHERE spec_id = ? AND ended_at IS NULL "
                "ORDER BY started_at DESC, rowid DESC LIMIT 1",
                (spec_id,),
            ).fetchone()
        return _row_dict(row)

    def open_session(
        self, spec_id: str, candidate_id: str, planned_minutes: int, *, now: float
    ) -> tuple[dict[str, Any], bool]:
        """Join the spec's open sitting, or create ``candidate_id``.

        Reading the open session and inserting a new one must share a single
        ``BEGIN IMMEDIATE``: split across two transactions, two workers can
        both see "no open session" and mint two sittings for the same learner.
        The write lock makes the loser join the winner's row instead.
        """
        _safe_component(candidate_id)
        cutoff = now - MAX_OPEN_SESSION_SECONDS
        with self._transaction() as conn:
            conn.execute(
                "UPDATE review_sessions SET ended_at = started_at + ? "
                "WHERE spec_id = ? AND ended_at IS NULL AND started_at < ?",
                (MAX_OPEN_SESSION_SECONDS, spec_id, cutoff),
            )
            row = conn.execute(
                "SELECT * FROM review_sessions WHERE spec_id = ? AND ended_at IS NULL "
                "ORDER BY started_at DESC, rowid DESC LIMIT 1",
                (spec_id,),
            ).fetchone()
            if row is not None:
                return dict(row), False
            conn.execute(
                "INSERT INTO review_sessions (session_id, spec_id, started_at, ended_at, "
                "planned_minutes) VALUES (?, ?, ?, NULL, ?)",
                (candidate_id, spec_id, now, int(planned_minutes)),
            )
            created = conn.execute(
                "SELECT * FROM review_sessions WHERE session_id = ?", (candidate_id,)
            ).fetchone()
            return dict(created), True

    def ensure_session(
        self, spec_id: str, session_id: str, planned_minutes: int, *, now: float
    ) -> dict[str, Any]:
        _safe_component(session_id)
        expired = False
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM review_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO review_sessions (session_id, spec_id, started_at, ended_at, "
                    "planned_minutes) VALUES (?, ?, ?, NULL, ?)",
                    (session_id, spec_id, now, int(planned_minutes)),
                )
            elif row["spec_id"] != spec_id:
                raise SessionConflict(
                    f"session {session_id} belongs to a different spec"
                )
            elif row["ended_at"] is not None:
                # A finished session is history. Continuing to write answers
                # into it would mix two sittings and make "same-session retest"
                # meaningless; the caller must start a new session id.
                raise SessionConflict(f"session {session_id} has already ended")
            elif now - float(row["started_at"]) > MAX_OPEN_SESSION_SECONDS:
                # Explicit session ids must obey the same rollover as
                # ``latest_open_session``. The close must commit, so raise
                # only after the transaction block.
                conn.execute(
                    "UPDATE review_sessions SET ended_at = started_at + ? "
                    "WHERE session_id = ?",
                    (MAX_OPEN_SESSION_SECONDS, session_id),
                )
                expired = True
            row = conn.execute(
                "SELECT * FROM review_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if expired:
            raise SessionConflict(f"session {session_id} has expired")
        return dict(row)

    def session_activity(self, session_id: str) -> list[float]:
        """Timestamps of everything the learner did inside one session.

        Events (answers) and views (cards shown) together describe the active
        window; ``complete`` sums the gaps capped at ``ACTIVE_GAP_SECONDS`` so
        idle time between sittings does not inflate ``actual_minutes``.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts AS at FROM review_events WHERE session_id = ? "
                "UNION ALL "
                "SELECT shown_at AS at FROM review_views WHERE session_id = ? "
                "ORDER BY at",
                (session_id, session_id),
            ).fetchall()
        return [float(row["at"]) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM review_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return _row_dict(row)

    def complete_session(self, session_id: str, *, now: float) -> dict[str, Any] | None:
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM review_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            if row["ended_at"] is None:
                conn.execute(
                    "UPDATE review_sessions SET ended_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
                # One sitting per spec: a completed session must not fall back
                # to an older still-open row on the next ``today``. Legacy
                # databases can hold several open rows, so close them here.
                conn.execute(
                    "UPDATE review_sessions SET ended_at = ? "
                    "WHERE spec_id = ? AND ended_at IS NULL AND session_id <> ?",
                    (now, row["spec_id"], session_id),
                )
            row = conn.execute(
                "SELECT * FROM review_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            return dict(row)

    # --- study plans ---------------------------------------------------

    def create_plan(self, plan: dict[str, Any], *, now: float) -> dict[str, Any]:
        plan_id = str(plan["plan_id"])
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO review_plans "
                "(plan_id, subject, status, plan_json, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (plan_id, plan["subject"], plan["status"], canonical_json(plan), now, now),
            )
        stored = self.get_plan(plan_id)
        assert stored is not None
        return stored

    def get_plan(self, plan_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM review_plans WHERE plan_id = ?", (plan_id,)
            ).fetchone()
        return _plan_dict(row)

    def list_plans(self, *, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM review_plans"
        params: tuple[Any, ...] = ()
        if status:
            sql += " WHERE status = ?"
            params = (status,)
        with self._connect() as conn:
            rows = conn.execute(sql + " ORDER BY created_at DESC", params).fetchall()
        return [_plan_dict(row) for row in rows]

    def update_plan(
        self, plan_id: str, patch: dict[str, Any], *, now: float
    ) -> dict[str, Any] | None:
        current = self.get_plan(plan_id)
        if current is None:
            return None
        merged = {**current, **patch, "plan_id": plan_id, "updated_at": now}
        with self._transaction() as conn:
            conn.execute(
                "UPDATE review_plans SET subject = ?, status = ?, plan_json = ?, "
                "updated_at = ? WHERE plan_id = ?",
                (
                    merged["subject"],
                    merged["status"],
                    canonical_json(merged),
                    now,
                    plan_id,
                ),
            )
        return merged

    def replace_tasks(
        self,
        plan_id: str,
        tasks: list[dict[str, Any]],
        *,
        from_date: str | None = None,
    ) -> None:
        """Replace the plan's schedule, optionally only from ``from_date``.

        Replanning keeps past rows (and their completion state) intact; only
        today and later are rebuilt.
        """
        with self._transaction() as conn:
            if from_date:
                conn.execute(
                    "DELETE FROM review_daily_tasks "
                    "WHERE plan_id = ? AND local_date >= ?",
                    (plan_id, from_date),
                )
            else:
                conn.execute(
                    "DELETE FROM review_daily_tasks WHERE plan_id = ?", (plan_id,)
                )
            conn.executemany(
                "INSERT INTO review_daily_tasks "
                "(plan_id, local_date, seq, kind, title, detail, chapter, minutes, "
                "status, calendar_event_id, kanban_card_id, completed_at, self_note) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        plan_id,
                        task["local_date"],
                        int(task["seq"]),
                        task["kind"],
                        task["title"],
                        task.get("detail", ""),
                        task.get("chapter", ""),
                        int(task.get("minutes", 0)),
                        task.get("status", "todo"),
                        task.get("calendar_event_id"),
                        task.get("kanban_card_id"),
                        task.get("completed_at"),
                        task.get("self_note", ""),
                    )
                    for task in tasks
                ],
            )

    def delete_incomplete_before(self, plan_id: str, before: str) -> int:
        """Drop past rows a replan has superseded; completed work stays."""
        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM review_daily_tasks "
                "WHERE plan_id = ? AND local_date < ? AND status <> 'done'",
                (plan_id, before),
            )
            return cursor.rowcount

    def list_tasks(
        self,
        plan_id: str | None = None,
        *,
        local_date: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if plan_id:
            clauses.append("plan_id = ?")
            params.append(plan_id)
        if local_date:
            clauses.append("local_date = ?")
            params.append(local_date)
        if since:
            clauses.append("local_date >= ?")
            params.append(since)
        sql = "SELECT * FROM review_daily_tasks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self._connect() as conn:
            rows = conn.execute(sql + " ORDER BY local_date, seq", params).fetchall()
        return [dict(row) for row in rows]

    def update_task(
        self,
        plan_id: str,
        local_date: str,
        seq: int,
        patch: dict[str, Any],
    ) -> dict[str, Any] | None:
        allowed = {
            "status",
            "completed_at",
            "self_note",
            "calendar_event_id",
            "kanban_card_id",
            "minutes",
            "title",
            "detail",
        }
        fields = {key: value for key, value in patch.items() if key in allowed}
        if fields:
            assignments = ", ".join(f"{key} = ?" for key in fields)
            with self._transaction() as conn:
                conn.execute(
                    f"UPDATE review_daily_tasks SET {assignments} "
                    "WHERE plan_id = ? AND local_date = ? AND seq = ?",
                    (*fields.values(), plan_id, local_date, seq),
                )
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM review_daily_tasks "
                "WHERE plan_id = ? AND local_date = ? AND seq = ?",
                (plan_id, local_date, seq),
            ).fetchone()
        return dict(row) if row is not None else None

    def move_day(self, plan_id: str, from_date: str, to_date: str) -> int:
        """Move every task of one day to another date, preserving state.

        Used when the learner drags the day's calendar block. The caller must
        ensure ``to_date`` is free: merging two days would silently double the
        workload and confuse the completion state.
        """
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM review_daily_tasks "
                "WHERE plan_id = ? AND local_date = ? ORDER BY seq",
                (plan_id, from_date),
            ).fetchall()
            if not rows:
                return 0
            for row in rows:
                conn.execute(
                    "INSERT INTO review_daily_tasks "
                    "(plan_id, local_date, seq, kind, title, detail, chapter, minutes, "
                    "status, calendar_event_id, kanban_card_id, completed_at, self_note) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        plan_id,
                        to_date,
                        row["seq"],
                        row["kind"],
                        row["title"],
                        row["detail"],
                        row["chapter"],
                        row["minutes"],
                        row["status"],
                        row["calendar_event_id"],
                        row["kanban_card_id"],
                        row["completed_at"],
                        row["self_note"],
                    ),
                )
            conn.execute(
                "DELETE FROM review_daily_tasks WHERE plan_id = ? AND local_date = ?",
                (plan_id, from_date),
            )
        return len(rows)

    def create_note(self, note: dict[str, Any], *, now: float) -> dict[str, Any]:
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO review_notes "
                "(note_id, plan_id, chapter, title, body, status, document_id, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    note["note_id"],
                    note["plan_id"],
                    note.get("chapter", ""),
                    note["title"],
                    note.get("body", ""),
                    note.get("status", "draft"),
                    note.get("document_id"),
                    now,
                    now,
                ),
            )
        stored = self.get_note(note["note_id"])
        assert stored is not None
        return stored

    def get_note(self, note_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM review_notes WHERE note_id = ?", (note_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_notes(
        self, plan_id: str | None = None, *, status: str | None = None
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if plan_id:
            clauses.append("plan_id = ?")
            params.append(plan_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        sql = "SELECT * FROM review_notes"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self._connect() as conn:
            rows = conn.execute(sql + " ORDER BY created_at", params).fetchall()
        return [dict(row) for row in rows]

    def update_note(
        self, note_id: str, patch: dict[str, Any], *, now: float
    ) -> dict[str, Any] | None:
        current = self.get_note(note_id)
        if current is None:
            return None
        merged = {**current, **patch, "note_id": note_id, "updated_at": now}
        with self._transaction() as conn:
            conn.execute(
                "UPDATE review_notes SET chapter = ?, title = ?, body = ?, "
                "status = ?, document_id = ?, updated_at = ? WHERE note_id = ?",
                (
                    merged.get("chapter", ""),
                    merged["title"],
                    merged.get("body", ""),
                    merged["status"],
                    merged.get("document_id"),
                    now,
                    note_id,
                ),
            )
        return merged
