"""Phase 27 WS-A acceptance: Book -> Review bridge, end to end.

Run with the project virtualenv:

    .venv/Scripts/python.exe scripts/accept_book_bridge.py

The script writes a minimal but real-shaped Book through the public
``BookStorage`` API into a temporary runtime home, imports it through the real
``load_book_pages`` / ``build_spec_from_pages`` path, stores it with
``ReviewService.put_spec``, and drives ``POST /{spec_id}/book-attempt`` against
a real uvicorn process on an ephemeral port.

Red line: ``BookStorage`` receives an explicit ``PathService`` rooted in the
temporary home, and the script asserts that every resolved runtime path (books,
review store, chat history db, mastery store) stays below it. It refuses to run
otherwise, so the real ``data/`` tree is never touched.

Like ``accept_phase05.py`` this is stdlib-only and imports the working tree
(``src``) rather than the installed wheel, so it
validates the code being shipped; the module path used is printed up front.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[1] / "src"
ARTIFACTS = Path(__file__).resolve().parents[1] / "dist" / "accept_book_bridge"

BOOK_ID = "bk-ws27-accept"
PATH_ID = "path-ws27-accept"
KP_ID = "kp-ws27-resonance"
PAGE_MAIN = "pg-ws27-1"
PAGE_MULTI = "pg-ws27-2"
SPEC_ID = "book-ws27-accept"
LINKED_SPEC_ID = "book-ws27-accept-linked"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request(
    method: str,
    url: str,
    payload: dict | None = None,
    *,
    timeout: float = 30.0,
) -> tuple[int, dict | bytes, dict[str, str]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read()
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return response.status, json.loads(body.decode("utf-8")), dict(response.headers)
            return response.status, body, dict(response.headers)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            parsed = json.loads(body.decode("utf-8"))
        except Exception:
            parsed = body
        return exc.code, parsed, dict(exc.headers)


class Server:
    """uvicorn process running the working-tree Review router."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log_path = home / "server.log"
        self.launcher = home / "_accept_launcher.py"
        self.proc: subprocess.Popen | None = None

    def _write_launcher(self) -> None:
        # ``deeptutor`` is a regular package in site-packages, so PYTHONPATH
        # alone cannot expose the checkout's ``review/`` and ``modules/``
        # portions. Prepending to ``__path__`` is the same seam conftest.py
        # uses, and it keeps the server on the working tree.
        self.launcher.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(SRC)!r})\n"
            "import deeptutor\n"
            f"if {str(SRC / 'deeptutor')!r} not in deeptutor.__path__:\n"
            f"    deeptutor.__path__.insert(0, {str(SRC / 'deeptutor')!r})\n"
            "import deeptutor.review.api as review_api\n"
            "print('review api:', review_api.__file__, flush=True)\n"
            "import uvicorn\n"
            "from deeptutor.api.main import app\n"
            "uvicorn.run(app, host='127.0.0.1', port=int(sys.argv[1]), log_level='warning')\n",
            encoding="utf-8",
        )

    def start(self) -> None:
        self._write_launcher()
        env = os.environ.copy()
        env.update(
            {
                # Both roots live in the temporary home, so the core settings,
                # mastery store and review store all stay isolated.
                "DEEPTUTOR_HOME": str(self.home),
                "DEEPTUTOR_REVIEW_HOME": str(self.home / "review"),
                "AUTH_ENABLED": "false",
                "PYTHONUNBUFFERED": "1",
            }
        )
        log = self.log_path.open("ab")
        try:
            self.proc = subprocess.Popen(
                [sys.executable, str(self.launcher), str(self.port)],
                cwd=str(ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        finally:
            log.close()
        deadline = time.time() + 60
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    "server exited during startup; see " + str(self.log_path)
                )
            try:
                status, _, _ = request("GET", self.base + "/health/live", timeout=2)
                if status == 200:
                    return
            except Exception:
                time.sleep(0.3)
        raise RuntimeError("server did not become healthy; see " + str(self.log_path))

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.proc = None


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class CoreUnavailable(RuntimeError):
    """The installed core cannot be exercised in this environment.

    Only this class is allowed to turn a step into SKIP. A genuine bridge
    defect must never be downgraded to a skip.
    """


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _build_book(storage) -> None:
    """One minimal book with the shapes the importer must handle."""
    from deeptutor.book.models import Block, BlockType, Book, Page

    storage.save_book(
        Book(
            id=BOOK_ID,
            title="WS27 Book 桥验收",
            language="zh",
            knowledge_bases=["kb-ws27"],
        )
    )
    storage.save_page(
        Page(
            id=PAGE_MAIN,
            book_id=BOOK_ID,
            title="第一章 串联谐振",
            order=0,
            blocks=[
                Block(
                    id="blk-text-1",
                    type=BlockType.TEXT,
                    payload={"body": "串联谐振时感抗与容抗相互抵消，回路阻抗最小。"},
                ),
                Block(
                    id="blk-quiz-1",
                    type=BlockType.QUIZ,
                    payload={
                        "questions": [
                            {
                                "question_id": "q1",
                                "review_item_id": "item-book-linked",
                                "knowledge_point_id": KP_ID,
                                "question": "串联谐振时回路阻抗为什么最小？",
                                "question_type": "choice",
                                "options": {
                                    "A": "电阻被短路",
                                    "B": "感抗与容抗相互抵消",
                                },
                                "correct_answer": "B",
                                "explanation": "B. 感抗与容抗相互抵消。",
                                "difficulty": "easy",
                            },
                            {
                                "question_id": "q2",
                                "question": "谐振频率由什么决定？",
                                "question_type": "choice",
                                "options": {"A": "电感与电容", "B": "电源电压"},
                                "correct_answer": "A",
                                "explanation": "A. 由电感与电容共同决定。",
                                "difficulty": "medium",
                            },
                        ]
                    },
                ),
                Block(
                    id="blk-cards-1",
                    type=BlockType.FLASH_CARDS,
                    payload={
                        "cards": [
                            {
                                "front": "品质因数 Q 的定义？",
                                "back": "Q = ω0 L / R",
                                "hint": "串联回路",
                            }
                        ]
                    },
                ),
            ],
        )
    )
    storage.save_page(
        Page(
            id=PAGE_MULTI,
            book_id=BOOK_ID,
            title="第二章 多选练习",
            order=1,
            blocks=[
                Block(
                    id="blk-text-2",
                    type=BlockType.TEXT,
                    payload={"body": "多选题目用于验证降级行为。"},
                ),
                Block(
                    id="blk-quiz-2",
                    type=BlockType.QUIZ,
                    payload={
                        "questions": [
                            {
                                "question_id": "q3",
                                "question": "以下哪些属于谐振特征？",
                                "question_type": "multiple_choice",
                                "options": {
                                    "A": "阻抗最小",
                                    "B": "电流最大",
                                    "C": "电压为零",
                                },
                                "correct_answer": "AB",
                                "explanation": "A、B 正确。",
                                "difficulty": "medium",
                            }
                        ]
                    },
                ),
            ],
        )
    )


def seed_mastery_path() -> None:
    """The real core aggregate the projection step writes into."""
    from deeptutor.learning.models import (
        KnowledgePoint,
        LearningModule,
        LearningProgress,
    )
    from deeptutor.learning.storage import LearningStore

    LearningStore().save(
        LearningProgress(
            book_id=PATH_ID,
            name="WS27 验收路径",
            desired_retention=0.9,
            modules=[
                LearningModule(
                    id="module-ws27",
                    name="第2章 谐振回路",
                    order=1,
                    knowledge_points=[
                        KnowledgePoint(
                            id=KP_ID,
                            name="串联谐振",
                            type="concept",
                            module_id="module-ws27",
                        )
                    ],
                )
            ],
        )
    )


def core_progress():
    from deeptutor.learning.storage import LearningStore

    return LearningStore().load(PATH_ID)


def count_events(review_home: Path) -> int:
    db = review_home / "review.sqlite3"
    conn = sqlite3.connect(db)
    try:
        row = conn.execute("SELECT COUNT(*) FROM review_events").fetchone()
        return int(row[0])
    finally:
        conn.close()


def event_rows(review_home: Path) -> list[tuple[str, str, str]]:
    db = review_home / "review.sqlite3"
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT item_id, rating, result FROM review_events ORDER BY rowid"
        ).fetchall()
        return [(str(item), str(rating), str(result)) for item, rating, result in rows]
    finally:
        conn.close()


def _direct_core_record():
    """Step 5: exercise the core projection entry point without monkeypatching.

    Returns a small evidence payload. Raises :class:`CoreUnavailable` when the
    core cannot be imported or its store cannot be opened, which the caller
    reports as SKIP; every other failure is a real defect and stays a FAIL.
    """
    try:
        from deeptutor.learning.assessment import AssessmentRecord, record_assessment
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise CoreUnavailable(f"core assessment module unavailable: {exc}") from exc

    try:
        before = core_progress()
    except (OSError, ValueError) as exc:
        raise CoreUnavailable(f"core mastery store unavailable: {exc}") from exc
    if before is None:
        raise RuntimeError(f"mastery path {PATH_ID!r} was not seeded")
    before_evidence = len(before.learning_evidence)
    before_count = before.repetition_states.get(KP_ID)
    before_reviews = int(before_count.review_count) if before_count else 0

    record = AssessmentRecord(
        origin_type="external_import",
        origin_ref=SPEC_ID,
        turn_id="accept-core-session",
        question_id="item-book-linked",
        source="book",
        assessment_type="review",
        result="correct",
        is_correct=True,
        question="串联谐振时回路阻抗为什么最小？",
        question_type="single_choice",
        options={"A": "电阻被短路", "B": "感抗与容抗相互抵消"},
        user_answer="B. 感抗与容抗相互抵消",
        correct_answer="B. 感抗与容抗相互抵消",
        explanation="B. 感抗与容抗相互抵消。",
        mastery_path_id=PATH_ID,
        knowledge_point_id=KP_ID,
        attempt_id="accept-core-direct-1",
    )
    outcome = asyncio.run(record_assessment(record))
    after = core_progress()
    after_state = after.repetition_states.get(KP_ID)
    assert_true(outcome.attempt_recorded, f"core did not record the attempt: {outcome}")
    assert_true(
        len(after.learning_evidence) == before_evidence + 1,
        f"core evidence did not grow: {before_evidence} -> {len(after.learning_evidence)}",
    )
    assert_true(after_state is not None, "core repetition state missing after assessment")
    assert_true(
        int(after_state.review_count) == before_reviews + 1,
        f"core review_count did not advance: {before_reviews} -> {after_state.review_count}",
    )

    # Same attempt_id must not double-count: the core owns idempotency.
    replay = asyncio.run(record_assessment(record))
    replay_progress = core_progress()
    assert_true(not replay.attempt_recorded, f"replayed attempt was recorded again: {replay}")
    assert_true(
        len(replay_progress.learning_evidence) == before_evidence + 1,
        "replayed assessment duplicated core evidence",
    )
    assert_true(
        int(replay_progress.repetition_states[KP_ID].review_count)
        == int(after_state.review_count),
        "replayed assessment moved the core review_count",
    )
    return {
        "attempt_recorded": outcome.attempt_recorded,
        "diagnostics": list(outcome.diagnostics),
        "evidence_before": before_evidence,
        "evidence_after": len(after.learning_evidence),
        "review_count_before": before_reviews,
        "review_count_after": int(after_state.review_count),
        "replay_recorded": replay.attempt_recorded,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="keep the temporary runtime home")
    parser.add_argument("--allow-skip", action="store_true", help="explicitly permit unavailable core steps without claiming full acceptance")
    args = parser.parse_args()

    home = Path(tempfile.mkdtemp(prefix="book-bridge-accept-"))
    # Both roots must be set before ``deeptutor`` is imported: the runtime home
    # and the admin workspace root are resolved at import time.
    os.environ["DEEPTUTOR_HOME"] = str(home)
    os.environ["DEEPTUTOR_REVIEW_HOME"] = str(home / "review")
    os.environ["AUTH_ENABLED"] = "false"

    # Import the working tree (see conftest.py for why __path__, not sys.path).
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import deeptutor

    if str(SRC / "deeptutor") not in deeptutor.__path__:
        deeptutor.__path__.insert(0, str(SRC / "deeptutor"))

    from deeptutor.review import api as review_api
    from deeptutor.review.book_import import build_spec_from_pages, load_book_pages
    from deeptutor.review.qa import run_gates
    from deeptutor.review.queue import concept_state_map
    from deeptutor.review.service import QualityGateFailed, ReviewService
    from deeptutor.review.storage import review_home
    from deeptutor.services.path_service import PathService

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "home": str(home),
        "book_id": BOOK_ID,
        "review_module": review_api.__file__,
        "steps": [],
    }

    def step(name: str, fn):
        started = time.time()
        try:
            detail = fn()
            report["steps"].append(
                {
                    "name": name,
                    "passed": True,
                    "seconds": round(time.time() - started, 2),
                    "detail": detail,
                }
            )
            print(f"[PASS] {name}")
            return detail
        except Exception as exc:
            report["steps"].append(
                {
                    "name": name,
                    "passed": False,
                    "seconds": round(time.time() - started, 2),
                    "error": repr(exc),
                }
            )
            print(f"[FAIL] {name}: {exc}")
            raise

    def skip(name: str, reason: str) -> None:
        report["steps"].append(
            {"name": name, "passed": None, "skipped": True, "reason": reason}
        )
        print(f"[SKIP] {name}: {reason}")

    def step_optional(name: str, fn):
        """Run a core-dependent step; SKIP only when the core is unavailable."""
        try:
            return step(name, fn)
        except CoreUnavailable as exc:
            if not args.allow_skip:
                raise
            report["steps"] = [row for row in report["steps"] if row["name"] != name]
            skip(name, f"core unavailable: {exc}")
            return None

    server = Server(home)
    session_id = "accept-book-session"

    try:
        print(f"runtime home: {home}")
        print(f"review module: {review_api.__file__}")

        # ── 1. real Book through BookStorage, explicit temp PathService ──
        book_paths: dict = {}

        def step1():
            from deeptutor.book.storage import BookStorage

            PathService.reset_instance()
            path_service = PathService(workspace_root=home / "data")
            book_dir = path_service.get_book_dir()
            review_dir = review_home()
            chat_db = path_service.get_chat_history_db()
            for label, path in (
                ("book dir", book_dir),
                ("review home", review_dir),
                ("chat history db", chat_db),
            ):
                assert_true(
                    _under(path, home),
                    f"{label} escaped the temporary home: {path}",
                )

            storage = BookStorage(path_service=path_service)
            _build_book(storage)

            # Re-read through the same public API: the assertion is about what
            # was persisted, not about the objects still in memory.
            assert_true(
                BOOK_ID in storage.list_book_ids(),
                f"book {BOOK_ID!r} not listed after save: {storage.list_book_ids()}",
            )
            stored_book = storage.load_book(BOOK_ID)
            assert_true(stored_book is not None, "book manifest did not round-trip")
            stored_pages = storage.list_pages(BOOK_ID)
            assert_true(
                [page.id for page in stored_pages] == [PAGE_MAIN, PAGE_MULTI],
                f"pages did not round-trip in order: {[p.id for p in stored_pages]}",
            )
            assert_true(
                (path_service.get_book_root(BOOK_ID) / "manifest.json").exists(),
                "manifest.json missing on disk",
            )
            book_paths.update(
                {
                    "book_root": str(path_service.get_book_root(BOOK_ID)),
                    "book_dir": str(book_dir),
                    "pages": [page.id for page in stored_pages],
                }
            )
            return book_paths

        step("1. BookStorage writes a real Book under the temp PathService", step1)

        # Core fixture for steps 5/5b: the projection target must be a real
        # saved mastery path, otherwise the core drops the linkage. This is
        # setup, not a step, so it is only recorded in the report.
        from deeptutor.learning.storage import LearningStore

        PathService.reset_instance()
        mastery_db = LearningStore().db_path
        assert_true(
            _under(mastery_db, home),
            f"mastery store escaped the temporary home: {mastery_db}",
        )
        seed_mastery_path()
        seeded = core_progress()
        assert_true(seeded is not None and seeded.book_id == PATH_ID, "mastery path seeding failed")
        report["mastery_path"] = PATH_ID
        report["mastery_db"] = str(mastery_db)

        # ── 2. import through the real storage API into ReviewService ──
        def step2():
            book, pages = load_book_pages(BOOK_ID)
            spec = build_spec_from_pages(book, pages, spec_id=SPEC_ID)
            assert_true(
                spec.mastery_path_id == "",
                "a spec without an explicit path must stay review-only",
            )
            gates = run_gates(spec)
            if not gates.passed:
                raise QualityGateFailed(gates)
            stored = ReviewService().put_spec(spec)
            assert_true(stored["created"] is True, f"first put_spec was not created: {stored}")
            assert_true(
                stored["quality"]["passed"] is True,
                f"stored spec failed its gates: {stored['quality']}",
            )
            meta = ReviewService().spec_meta(SPEC_ID)
            assert_true(
                meta["counts"] == {"concepts": 1, "sections": 1, "items": 3},
                f"unexpected stored counts: {meta['counts']}",
            )
            return stored["counts"] | {"content_hash": stored["content_hash"]}

        step("2. load_book_pages -> build_spec_from_pages -> put_spec", step2)

        # ── 3. import shape and determinism ──
        def step3():
            book, pages = load_book_pages(BOOK_ID)
            first = build_spec_from_pages(book, pages, spec_id=SPEC_ID)
            second = build_spec_from_pages(book, pages, spec_id=SPEC_ID)

            # Page 2 carries only a multiple-choice question. The importer's
            # documented rule is exact-label matching, so an answer like "AB"
            # matches no single option and the question is skipped; a page with
            # no importable item yields no concept and no section.
            multi_questions = [
                question
                for block in pages[1].blocks
                if block.type.value == "quiz"
                for question in block.payload.get("questions", [])
            ]
            assert_true(
                [q["question_id"] for q in multi_questions] == ["q3"],
                f"fixture drift: page 2 questions are {multi_questions}",
            )
            assert_true(
                all("谐振特征" not in item.stem for item in first.items),
                "multi-select question must not become a gradable item",
            )
            assert_true(
                [concept.id for concept in first.concepts] == [f"concept-{PAGE_MAIN}"],
                f"expected exactly one concept for the importable page, got "
                f"{[concept.id for concept in first.concepts]}",
            )
            assert_true(
                [section.id for section in first.sections] == [f"sec-{PAGE_MAIN}"],
                f"unexpected sections: {[section.id for section in first.sections]}",
            )
            assert_true(
                first.concepts[0].knowledge_point_id == "",
                "review-only import must not carry a knowledge_point_id",
            )
            item_ids = [item.id for item in first.items]
            assert_true(
                item_ids == ["item-book-linked", "item-blk-quiz-1-1", "item-blk-cards-1-card-0"],
                f"unexpected item ids: {item_ids}",
            )
            linked_item = next(item for item in first.items if item.id == "item-book-linked")
            assert_true(
                linked_item.kind == "single_choice",
                f"linked question kind changed: {linked_item.kind}",
            )
            assert_true(
                [option.text for option in linked_item.options]
                == ["电阻被短路", "感抗与容抗相互抵消"],
                "options did not survive the import in Book order",
            )
            assert_true(
                [item.id for item in second.items] == item_ids,
                "item ids are not reproducible across two imports",
            )
            assert_true(
                first.content_hash == second.content_hash,
                "two imports of the same Book produced different content hashes",
            )
            assert_true(
                len({item.id for item in first.items}) == len(first.items),
                "duplicate item ids in one import",
            )
            return {
                "items": item_ids,
                "concepts": [concept.id for concept in first.concepts],
                "skipped_multi_select": "q3",
                "content_hash": first.content_hash,
            }

        step("3. import shape: linkage, stable ids, multi-select degradation", step3)

        # ── 4. bridge attempts: good/again, idempotency, queue state ──
        server.start()

        def step4():
            review_dir = home / "review"
            before = count_events(review_dir)
            assert_true(before == 0, f"review event log was not empty: {before}")

            status, today, _ = request(
                "GET", f"{server.base}/api/review/{SPEC_ID}/today?session_id={session_id}"
            )
            assert_true(status == 200, f"today status {status}: {today}")
            assert_true(
                today["counts"]["new"] == 3,
                f"expected 3 new items, got {today['counts']}",
            )

            status, good, _ = request(
                "POST",
                f"{server.base}/api/review/{SPEC_ID}/book-attempt",
                {
                    "item_id": "item-book-linked",
                    "is_correct": True,
                    "session_id": session_id,
                    "submission_id": "accept-book-good",
                    "answer": 1,
                },
            )
            assert_true(status == 200, f"good attempt status {status}: {good}")
            assert_true(good["rating"] == "good", f"is_correct=True -> {good['rating']}")
            assert_true(good["result"] == "correct", f"good result: {good['result']}")
            assert_true(good["created"] is True, "first attempt did not create an event")
            # Review-only spec: the bridge must not invent a core linkage.
            projection = good.get("mastery_projection") or {}
            assert_true(
                projection.get("status") == "skipped"
                and "no_mastery_linkage" in (projection.get("diagnostics") or []),
                f"unlinked spec projected into the core: {projection}",
            )
            assert_true(
                good.get("state_source") == "review",
                f"unlinked spec must read Review state: {good.get('state_source')}",
            )

            status, wrong, _ = request(
                "POST",
                f"{server.base}/api/review/{SPEC_ID}/book-attempt",
                {
                    "item_id": "item-blk-quiz-1-1",
                    "is_correct": False,
                    "session_id": session_id,
                    "submission_id": "accept-book-wrong",
                    "answer": 1,
                },
            )
            assert_true(status == 200, f"wrong attempt status {status}: {wrong}")
            assert_true(wrong["rating"] == "again", f"is_correct=False -> {wrong['rating']}")
            assert_true(wrong["result"] == "incorrect", f"wrong result: {wrong['result']}")
            assert_true(wrong["created"] is True, "second attempt did not create an event")
            assert_true(
                wrong["lapse_count"] >= 1, f"again must increment lapse_count: {wrong}"
            )
            assert_true(
                wrong["scheduled_interval_days"] < 1.0,
                f"failure interval should be short: {wrong['scheduled_interval_days']}",
            )

            rows = event_rows(review_dir)
            assert_true(
                rows
                == [
                    ("item-book-linked", "good", "correct"),
                    ("item-blk-quiz-1-1", "again", "incorrect"),
                ],
                f"event log does not hold good/again exactly once: {rows}",
            )

            service = ReviewService()
            states = concept_state_map(
                service.store.get_spec(SPEC_ID), service.store, now=time.time()
            )
            state = states[f"concept-{PAGE_MAIN}"]
            assert_true(
                state.review_count == 2 and state.lapse_count == 1,
                f"concept state did not absorb both answers: "
                f"review={state.review_count} lapse={state.lapse_count}",
            )

            status, again, _ = request(
                "GET", f"{server.base}/api/review/{SPEC_ID}/today?session_id={session_id}"
            )
            assert_true(status == 200, f"today after status {status}: {again}")
            by_id = {row["item_id"]: row for row in again["items"]}
            assert_true(
                by_id.get("item-blk-quiz-1-1", {}).get("status") == "repeat",
                f"the failed card must be offered again: {by_id}",
            )

            # Repeated submission_id with the same payload is idempotent.
            status, duplicate, _ = request(
                "POST",
                f"{server.base}/api/review/{SPEC_ID}/book-attempt",
                {
                    "item_id": "item-book-linked",
                    "is_correct": True,
                    "session_id": session_id,
                    "submission_id": "accept-book-good",
                    "answer": 1,
                },
            )
            assert_true(status == 200, f"duplicate status {status}: {duplicate}")
            assert_true(duplicate["created"] is False, "duplicate created a second event")
            assert_true(
                duplicate["event_id"] == good["event_id"],
                "duplicate returned a different event id",
            )
            assert_true(count_events(review_dir) == 2, "duplicate changed the event count")

            # Same submission_id with a different payload is a conflict.
            status, conflict, _ = request(
                "POST",
                f"{server.base}/api/review/{SPEC_ID}/book-attempt",
                {
                    "item_id": "item-book-linked",
                    "is_correct": False,
                    "session_id": session_id,
                    "submission_id": "accept-book-good",
                    "answer": 0,
                },
            )
            assert_true(status == 409, f"payload conflict status {status}: {conflict}")
            assert_true(
                (conflict.get("detail") or {}).get("code") == "submission_conflict",
                f"conflict body: {conflict}",
            )
            assert_true(
                count_events(review_dir) == 2, "conflicting submission changed the event count"
            )
            return {
                "events": rows,
                "review_count": state.review_count,
                "lapse_count": state.lapse_count,
                "repeat_items": [
                    row["item_id"] for row in again["items"] if row["status"] == "repeat"
                ],
                "duplicate_created": duplicate["created"],
            }

        step("4. /book-attempt maps correctness, replays idempotently, updates Today", step4)

        # ── 5. core projection path (real core, no monkeypatching) ──
        core_detail = step_optional(
            "5. core record_assessment projection (direct, real core)",
            _direct_core_record,
        )

        # 5b: the bridge itself must project when the spec carries a path.
        def step5b():
            book, pages = load_book_pages(BOOK_ID)
            linked_spec = build_spec_from_pages(
                book, pages, spec_id=LINKED_SPEC_ID, mastery_path_id=PATH_ID
            )
            assert_true(
                linked_spec.mastery_path_id == PATH_ID,
                f"linked spec lost its mastery path: {linked_spec.mastery_path_id}",
            )
            assert_true(
                linked_spec.concepts[0].knowledge_point_id == KP_ID,
                f"linked concept lost its objective: {linked_spec.concepts[0].knowledge_point_id}",
            )
            stored = ReviewService().put_spec(linked_spec)
            assert_true(stored["created"] is True, f"linked put_spec: {stored}")

            try:
                before = core_progress()
            except (OSError, ValueError) as exc:
                raise CoreUnavailable(f"core mastery store unavailable: {exc}") from exc
            if before is None:
                raise RuntimeError(f"mastery path {PATH_ID!r} was not seeded")
            before_evidence = len(before.learning_evidence)
            before_state = before.repetition_states[KP_ID]
            before_reviews = int(before_state.review_count)

            status, body, _ = request(
                "POST",
                f"{server.base}/api/review/{LINKED_SPEC_ID}/book-attempt",
                {
                    "item_id": "item-book-linked",
                    "is_correct": True,
                    # Sessions are spec-scoped by the store (409 otherwise), so
                    # the linked spec gets its own.
                    "session_id": f"{session_id}-linked",
                    "submission_id": "accept-book-linked",
                    "answer": 1,
                },
            )
            assert_true(status == 200, f"linked attempt status {status}: {body}")
            projection = body.get("mastery_projection") or {}
            assert_true(
                projection.get("status") == "applied",
                f"linked bridge did not project: {projection}",
            )
            assert_true(
                body.get("state_source") == "core",
                f"linked attempt must read core state: {body.get('state_source')}",
            )
            after = core_progress()
            assert_true(
                len(after.learning_evidence) == before_evidence + 1,
                f"core evidence did not grow: {before_evidence} -> {len(after.learning_evidence)}",
            )
            after_state = after.repetition_states[KP_ID]
            assert_true(
                int(after_state.review_count) == before_reviews + 1,
                f"core review_count did not advance: {before_reviews} -> {after_state.review_count}",
            )
            return {
                "projection": projection,
                "state_source": body.get("state_source"),
                "evidence_before": before_evidence,
                "evidence_after": len(after.learning_evidence),
                "review_count_before": before_reviews,
                "review_count_after": int(after_state.review_count),
            }

        step_optional("5b. linked spec: /book-attempt projects into the core", step5b)

        def step5c():
            before = core_progress()
            before_evidence = len(before.learning_evidence)
            payload = {
                "item_id": "item-book-linked", "rating": "hard", "answer": 1,
                "session_id": f"{session_id}-linked", "submission_id": "accept-direct-linked",
            }
            status, body, _ = request("POST", f"{server.base}/api/review/{LINKED_SPEC_ID}/attempt", payload)
            assert_true(status == 200, f"direct attempt status {status}: {body}")
            assert_true(body.get("mastery_projection", {}).get("status") == "applied", f"direct projection failed: {body}")
            after = core_progress()
            assert_true(len(after.learning_evidence) == before_evidence + 1, "direct attempt did not add core evidence")
            status, replay, _ = request("POST", f"{server.base}/api/review/{LINKED_SPEC_ID}/attempt", payload)
            assert_true(status == 200 and replay.get("created") is False, "direct attempt replay was not idempotent")
            assert_true(len(core_progress().learning_evidence) == before_evidence + 1, "replay duplicated core evidence")
            return {"evidence_before": before_evidence, "evidence_after": len(after.learning_evidence), "replay_created": replay["created"]}

        if not any(row.get("skipped") for row in report["steps"]):
            step("5c. /attempt writes real mastery evidence and replays once", step5c)

        skipped = [row for row in report["steps"] if row.get("skipped")]
        report["passed"] = not skipped
        report["skipped"] = [row["name"] for row in skipped]
        print("\nACCEPTANCE INCOMPLETE" + f" ({len(skipped)} step(s) skipped)" if skipped else "\nACCEPTANCE PASSED")
        print(f"artifact: {ARTIFACTS / 'report.json'}")
        print(f"runtime home: {home}")
        if core_detail is not None:
            print(f"core projection: {core_detail}")
        return 0
    except Exception:
        report["passed"] = False
        print("\nACCEPTANCE FAILED")
        print(f"runtime home: {home}")
        if server.log_path.exists():
            tail = server.log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            print("--- server log tail ---")
            print(tail)
        return 1
    finally:
        server.stop()
        (ARTIFACTS / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not args.keep:
            # Windows may keep a handle for a moment after the child exits;
            # retry briefly instead of reporting a cleanup that did not happen.
            for _ in range(20):
                try:
                    shutil.rmtree(home)
                    break
                except OSError:
                    time.sleep(0.25)
            report["home"] = "<removed>" if not home.exists() else str(home)
            (ARTIFACTS / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    raise SystemExit(main())
