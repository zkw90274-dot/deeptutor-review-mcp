"""Phase 0.5 acceptance: five steps against a real HTTP server.

Run with the project virtualenv:

    .venv/Scripts/python.exe scripts/accept_phase05.py

The script starts its own uvicorn process on an ephemeral port and uses an
isolated ``DEEPTUTOR_REVIEW_HOME``, so it never touches the real workspace.
It is intentionally stdlib-only: no requests/httpx dependency, no browser.
"""

from __future__ import annotations

import argparse
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
ARTIFACTS = Path(__file__).resolve().parents[1] / "dist" / "accept_phase05"
SPEC_ID = "fixture-hf-resonance"


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
    def __init__(self, home: Path) -> None:
        self.home = home
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log_path = home / "server.log"
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "DEEPTUTOR_REVIEW_HOME": str(self.home),
                # Keep core settings/logs inside the temporary home too.
                "DEEPTUTOR_HOME": str(self.home),
                "AUTH_ENABLED": "false",
                "PYTHONUNBUFFERED": "1",
            }
        )
        log = self.log_path.open("ab")
        try:
            self.proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import sys; import deeptutor; "
                    f"deeptutor.__path__.insert(0, {str(SRC / 'deeptutor')!r}); "
                    "import deeptutor.review.api as review_api; "
                    "print('review api:', review_api.__file__, flush=True); "
                    "import uvicorn; uvicorn.main()",
                    "deeptutor.api.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(self.port),
                    "--log-level",
                    "warning",
                ],
                cwd=str(ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        finally:
            # The child owns its duplicated handle; keeping this one open
            # prevents removing the temporary home on Windows.
            log.close()
        deadline = time.time() + 45
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited during startup; see " + str(self.log_path))
            try:
                status, body, _ = request("GET", self.base + "/health/live", timeout=2)
                if status == 200:
                    for line in self.log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                        if line.startswith("review api:"):
                            print(line, flush=True)
                    return
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            time.sleep(0.35)
        raise RuntimeError("server did not become healthy; see " + str(self.log_path))

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.proc = None

    def restart(self) -> None:
        self.stop()
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.start()


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def count_events(home: Path) -> int:
    db = home / "review.sqlite3"
    conn = sqlite3.connect(db)
    try:
        row = conn.execute("SELECT COUNT(*) FROM review_events").fetchone()
        return int(row[0])
    finally:
        conn.close()


def build_linked_spec():
    """Fixture spec with an explicit mastery path and two objectives."""
    from deeptutor.review.fixture import build_fixture_spec
    from deeptutor.review.models import ReviewSpec

    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    payload["mastery_path_id"] = SPEC_ID
    knowledge_points = {
        "c-series-resonance": "kp-series-resonance",
        "c-quality-factor": "kp-quality-factor",
    }
    for concept in payload["concepts"]:
        concept["knowledge_point_id"] = knowledge_points.get(concept["id"], "")
    # One prerequisite edge so the export must also produce a concept graph.
    for concept in payload["concepts"]:
        if concept["id"] == "c-quality-factor":
            concept["prerequisites"] = ["c-series-resonance"]
    return ReviewSpec.model_validate(payload)


def seed_mastery_path() -> float:
    """Create the real core aggregate the Review projection must update.

    ``kp-quality-factor`` starts with a pre-existing core retention state to
    prove that linked Today/attempt views read the core, not only Review's
    local event log.
    """
    from deeptutor.learning.models import (
        KnowledgePoint,
        LearningModule,
        LearningProgress,
        RepetitionState,
    )
    from deeptutor.learning.storage import LearningStore

    progress = LearningProgress(
        book_id=SPEC_ID,
        name="Review Studio 验收路径",
        desired_retention=0.9,
        modules=[
            LearningModule(
                id="module-hf-2",
                name="第2章 谐振回路",
                order=1,
                knowledge_points=[
                    KnowledgePoint(
                        id="kp-series-resonance",
                        name="串联谐振",
                        type="concept",
                        module_id="module-hf-2",
                    ),
                    KnowledgePoint(
                        id="kp-quality-factor",
                        name="品质因数",
                        type="concept",
                        module_id="module-hf-2",
                    ),
                ],
            )
        ],
    )
    seeded_due = time.time() - 3600.0
    progress.repetition_states["kp-quality-factor"] = RepetitionState(
        next_review_at=seeded_due,
        review_count=3,
        lapse_count=0,
        stability=5.0,
        last_review_at=seeded_due - 86400.0,
        desired_retention=0.9,
    )
    LearningStore().save(progress)
    return seeded_due


def core_progress():
    from deeptutor.learning.storage import LearningStore

    return LearningStore().load(SPEC_ID)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="keep the temporary review home")
    args = parser.parse_args()

    home = Path(tempfile.mkdtemp(prefix="review-accept-"))
    # The acceptance process must use the same runtime home as the server so
    # LearningStore and review storage resolve to the temporary directory.
    os.environ["DEEPTUTOR_HOME"] = str(home)
    os.environ["DEEPTUTOR_REVIEW_HOME"] = str(home / "review")
    import deeptutor

    deeptutor.__path__.insert(0, str(SRC / "deeptutor"))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report: dict = {"home": str(home), "steps": []}
    server = Server(home)
    payload: dict = {}
    session_id = ""
    now = time.time()

    def step(name: str, fn):
        started = time.time()
        try:
            detail = fn()
            report["steps"].append(
                {"name": name, "passed": True, "seconds": round(time.time() - started, 2), "detail": detail}
            )
            print(f"[PASS] {name}")
            return detail
        except Exception as exc:
            report["steps"].append(
                {"name": name, "passed": False, "seconds": round(time.time() - started, 2), "error": repr(exc)}
            )
            print(f"[FAIL] {name}: {exc}")
            raise

    try:
        report["mastery_path"] = SPEC_ID
        report["seed_due"] = seed_mastery_path()
        spec = build_linked_spec()
        payload = spec.model_dump(mode="json")
        server.start()

        def step1():
            nonlocal session_id
            status, body, _ = request("PUT", f"{server.base}/api/review/specs/{SPEC_ID}", payload)
            assert_true(status in (200, 201), f"PUT status {status}: {body}")
            status, today, _ = request("GET", f"{server.base}/api/review/{SPEC_ID}/today")
            assert_true(status == 200, f"today status {status}: {today}")
            items = today["items"]
            assert_true(len(items) == 3, f"expected 3 items, got {len(items)}")
            for item in items:
                assert_true(bool(item.get("due_at")), "missing due_at")
                assert_true(
                    isinstance(item.get("predicted_retrievability"), (int, float)),
                    "missing predicted_retrievability",
                )
            quality = next(
                item for item in items if item["concept_id"] == "c-quality-factor"
            )
            assert_true(
                quality["state_source"] == "core",
                f"quality must use core state: {quality}",
            )
            assert_true(
                abs(quality["due_at_ts"] - report["seed_due"]) <= 1e-6,
                "Today did not use the seeded core due date",
            )
            session_id = today["session_id"]
            return {
                "items": len(items),
                "session_id": session_id,
                "tomorrow": today["tomorrow"],
                "quality_state_source": quality["state_source"],
                "quality_due": quality["due_at_ts"],
            }

        step("1. seed spec -> today returns 3 items", step1)

        def step2():
            status, body, _ = request(
                "POST",
                f"{server.base}/api/review/{SPEC_ID}/attempt",
                {
                    "item_id": "item-resonance-choice",
                    "rating": "good",
                    "session_id": session_id,
                    "submission_id": "accept-good",
                    "answer": 1,
                },
            )
            assert_true(status == 200, f"attempt good status {status}: {body}")
            assert_true(body["result"] == "correct", "good must map to correct")
            assert_true(body["next_due_ts"] > now, "good must move due date forward")
            assert_true(
                body.get("state_source") == "core",
                f"linked attempt must read core state: {body}",
            )
            projection = body.get("mastery_projection") or {}
            assert_true(
                projection.get("status") == "applied",
                f"mastery projection status: {projection}",
            )
            assert_true(
                projection.get("due_date_consistent") is True,
                f"review/core due dates differ: {projection}",
            )
            progress = core_progress()
            assert_true(progress is not None, "core mastery path missing")
            evidence_ids = [item.evidence_id for item in progress.learning_evidence]
            assert_true(
                f"review:{body['event_id']}" in evidence_ids,
                f"core evidence missing review event: {evidence_ids}",
            )
            state = progress.repetition_states.get("kp-series-resonance")
            assert_true(state is not None, "core repetition state missing")
            assert_true(
                abs(state.next_review_at - body["next_due_ts"]) <= 1.0,
                "review and core due timestamps disagree",
            )

            # Hard means "remembered with effort": the core evidence must be
            # correct (not partial) while carrying quality 0.55.
            status, hard, _ = request(
                "POST",
                f"{server.base}/api/review/{SPEC_ID}/attempt",
                {
                    "item_id": "item-resonance-numeric",
                    "rating": "hard",
                    "session_id": session_id,
                    "submission_id": "accept-hard",
                    "answer": 1.5,
                },
            )
            assert_true(status == 200, f"attempt hard status {status}: {hard}")
            hard_projection = hard.get("mastery_projection") or {}
            assert_true(
                hard_projection.get("status") == "applied",
                f"hard projection status: {hard_projection}",
            )
            # The core already held this objective's state from the previous
            # good rating, so a local-vs-core due comparison is expected to be
            # inapplicable; the authoritative due date is the core one.
            assert_true(
                "core_state_precedes_review" in (hard_projection.get("diagnostics") or []),
                f"hard should report prior core state: {hard_projection}",
            )
            hard_progress = core_progress()
            hard_core = hard_progress.repetition_states.get("kp-series-resonance")
            assert_true(
                abs(hard_core.next_review_at - hard["next_due_ts"]) <= 1.0,
                "hard response did not use the core due date",
            )
            assert_true(
                hard["state_source"] == "core",
                f"hard response must read core state: {hard}",
            )
            progress = core_progress()
            hard_evidence = [
                item
                for item in progress.learning_evidence
                if item.evidence_id == f"review:{hard['event_id']}"
            ]
            assert_true(len(hard_evidence) == 1, "hard core evidence missing")
            assert_true(hard_evidence[0].result == "correct", "hard must be correct")
            assert_true(
                abs(float(hard_evidence[0].quality or 0.0) - 0.55) < 1e-9,
                "hard core quality must be 0.55",
            )
            hard_state = progress.repetition_states.get("kp-series-resonance")
            assert_true(hard_state is not None, "hard repetition state missing")
            assert_true(
                abs(hard_state.next_review_at - hard["next_due_ts"]) <= 1.0,
                "hard review and core due timestamps disagree",
            )
            return {
                "interval_days": body["scheduled_interval_days"],
                "next_due": body["next_due"],
                "correct": body["correct"],
                "mastery_projection": projection["status"],
                "core_evidence": len(progress.learning_evidence),
                "hard_interval_days": hard["scheduled_interval_days"],
                "hard_projection": hard_projection["status"],
            }

        step("2. rate A good -> next_due > now", step2)

        def step3():
            status, body, _ = request(
                "POST",
                f"{server.base}/api/review/{SPEC_ID}/attempt",
                {
                    "item_id": "item-quality-flashcard",
                    "rating": "again",
                    "session_id": session_id,
                    "submission_id": "accept-again",
                    "answer": "不会",
                },
            )
            assert_true(status == 200, f"attempt again status {status}: {body}")
            assert_true(body["result"] == "incorrect", "again must map to incorrect")
            assert_true(body["lapse_count"] >= 1, "again must increment lapse_count")
            assert_true(
                body["scheduled_interval_days"] < 1.0,
                f"failure interval should be short, got {body['scheduled_interval_days']}",
            )
            projection = body.get("mastery_projection") or {}
            assert_true(
                projection.get("status") == "applied",
                f"mastery projection status: {projection}",
            )
            progress = core_progress()
            state = progress.repetition_states.get("kp-quality-factor") if progress else None
            assert_true(state is not None, "core repetition state for B missing")
            assert_true(state.lapse_count >= 1, "again must propagate lapse to core")
            assert_true(
                body.get("state_source") == "core",
                f"linked attempt must read core state: {body}",
            )
            assert_true(
                body["review_count"] == state.review_count,
                "attempt response did not overlay core review_count",
            )
            assert_true(
                any(
                    item.evidence_id == f"review:{body['event_id']}"
                    and item.result == "incorrect"
                    for item in progress.learning_evidence
                ),
                "core evidence missing incorrect record",
            )
            return {
                "interval_days": body["scheduled_interval_days"],
                "lapse_count": body["lapse_count"],
                "next_due": body["next_due"],
                "mastery_projection": projection["status"],
                "core_lapse_count": state.lapse_count,
                "review_count": body["review_count"],
                "state_source": body["state_source"],
            }

        step("3. rate B again -> short interval + lapse", step3)

        def step4():
            before = count_events(home)
            before_evidence = len(core_progress().learning_evidence)
            server.restart()
            status, today, _ = request("GET", f"{server.base}/api/review/{SPEC_ID}/today")
            assert_true(status == 200, f"today after restart status {status}: {today}")
            for row in today["items"]:
                assert_true(
                    row["state_source"] == "core",
                    f"Today lost the core state after restart: {row}",
                )
            status, body, _ = request(
                "POST",
                f"{server.base}/api/review/{SPEC_ID}/attempt",
                {
                    "item_id": "item-resonance-choice",
                    "rating": "good",
                    "session_id": session_id,
                    "submission_id": "accept-good",
                    "answer": 1,
                },
            )
            assert_true(status == 200, f"duplicate after restart status {status}: {body}")
            assert_true(body["created"] is False, "duplicate submission must not create an event")
            assert_true(
                (body.get("mastery_projection") or {}).get("status") == "duplicate",
                f"duplicate projection status: {body.get('mastery_projection')}",
            )
            state = core_progress().repetition_states["kp-series-resonance"]
            assert_true(
                body.get("state_source") == "core",
                f"duplicate must still read core state: {body}",
            )
            assert_true(
                body["review_count"] == state.review_count,
                "duplicate response did not overlay core review_count",
            )
            after = count_events(home)
            after_evidence = len(core_progress().learning_evidence)
            assert_true(before == after == 3, f"event count changed: {before} -> {after}")
            assert_true(
                before_evidence == after_evidence == 3,
                f"core evidence count changed: {before_evidence} -> {after_evidence}",
            )
            return {
                "events_before": before,
                "events_after": after,
                "duplicate_created": body["created"],
                "core_evidence_before": before_evidence,
                "core_evidence_after": after_evidence,
            }

        step("4. restart -> state survives; duplicate is idempotent", step4)

        def step5():
            status, html, headers = request("POST", f"{server.base}/api/review/{SPEC_ID}/export")
            assert_true(status == 200, f"export status {status}: {html[:200] if isinstance(html, bytes) else html}")
            text = html.decode("utf-8") if isinstance(html, bytes) else str(html)
            assert_true(len(text.encode("utf-8")) <= 8_000_000, "export exceeds 8 MB")
            assert_true("MathJax" in text and "mjx-container" in text, "MathJax missing")
            assert_true(
                'src="http' not in text and "url(http" not in text,
                "export loads an external resource",
            )
            for needle in ("串联谐振发生时", "还是不会", "有点模糊", "会了", "第2章 2.1 串联谐振"):
                assert_true(needle in text, f"export missing {needle!r}")
            out = ARTIFACTS / "review.html"
            out.write_text(text, encoding="utf-8", newline="\n")
            (ARTIFACTS / "spec.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return {"bytes": len(text.encode("utf-8")), "artifact": str(out)}

        step("5. export offline HTML with answers/sources/ratings", step5)

        def step6():
            status, body, _ = request("GET", f"{server.base}/api/review/{SPEC_ID}/metrics?days=7")
            assert_true(status == 200, f"metrics status {status}: {body}")
            assert_true(body["spec_id"] == SPEC_ID, "metrics spec mismatch")
            assert_true(len(body["retention"]) == 10, "expected 10 retention buckets")
            assert_true(isinstance(body["completion"], dict), "completion missing")
            assert_true(
                body["completion"]["attempts"] >= 2, "expected the graded attempts"
            )
            # The whole point of the threshold: a thin bucket must not claim a rate.
            for bucket in body["retention"]:
                if bucket["samples"] < 5:
                    assert_true(
                        bucket["rate"] is None and bucket["status"] == "insufficient",
                        "thin bucket reported a rate",
                    )
            return {
                "retention_sampled": sum(1 for b in body["retention"] if b["samples"]),
                "attempts": body["completion"]["attempts"],
                "sessions": body["completion"]["sessions"],
            }

        step("6. metrics endpoint reports windowed calibration", step6)

        report["passed"] = True
        print("\nACCEPTANCE PASSED")
        print(f"artifact: {ARTIFACTS / 'review.html'}")
        print(f"review home: {home}")
        return 0
    except Exception:
        report["passed"] = False
        print("\nACCEPTANCE FAILED")
        print(f"review home: {home}")
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
            if home.exists():
                print(f"warning: temporary home was not removed: {home}")
            (ARTIFACTS / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    raise SystemExit(main())
