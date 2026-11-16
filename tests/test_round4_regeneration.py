"""Regeneration must not silently replace an existing pack.

Regenerating the same chapter resolves to the same deterministic spec id, so
a plain retry used to overwrite the pack that the learner's events, sessions
and mastery state point at. The job now fails unless the caller passes
``replace=true``, and the store re-checks under its per-spec lock.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

from deeptutor.review import jobs
from deeptutor.review import service as service_module
from deeptutor.review.agent_tools import ReviewGenerateTool
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.models import ReviewSpec
from deeptutor.review.service import ReviewService
from deeptutor.review.storage import ReviewStore, SpecExists, SpecNotFound


@pytest.fixture()
def review_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "review"
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(home))
    monkeypatch.setenv("DEEPTUTOR_HOME", str(tmp_path / "core-home"))
    return home


def _rewrite(spec: ReviewSpec, **updates: object) -> ReviewSpec:
    payload = spec.model_dump(mode="json")
    payload.update(updates)
    payload["content_hash"] = ""
    return ReviewSpec.model_validate(payload)


def _rename(spec: ReviewSpec, spec_id: str) -> ReviewSpec:
    """Move a spec and every id that embeds it onto a new id."""
    payload = spec.model_dump(mode="json")
    payload["id"] = spec_id
    payload["content_hash"] = ""
    prefix = f"{spec.id}-"
    for item in payload["items"]:
        if item["id"].startswith(prefix):
            item["id"] = f"{spec_id}-{item['id'][len(prefix):]}"
        if item.get("variant_of") and item["variant_of"].startswith(prefix):
            item["variant_of"] = f"{spec_id}-{item['variant_of'][len(prefix):]}"
    for section in payload["sections"]:
        for block in section["blocks"]:
            if block.get("item_id") and block["item_id"].startswith(prefix):
                block["item_id"] = f"{spec_id}-{block['item_id'][len(prefix):]}"
    return ReviewSpec.model_validate(payload)


def _install_generate(
    monkeypatch: pytest.MonkeyPatch,
    spec: ReviewSpec,
    *,
    chapter: str = "第2章_谐振",
    legacy_id: str | None = None,
) -> None:
    module = types.ModuleType("deeptutor.review.generate")
    result = types.SimpleNamespace(
        spec=spec,
        qa_count=len(spec.items),
        skipped=0,
        warnings=[],
        kb_name="hf",
        chapter=chapter,
    )
    module.generate_spec_from_kb = lambda **kwargs: result
    if legacy_id is not None:
        module.legacy_default_spec_id = lambda kb, chapter: legacy_id
    monkeypatch.setitem(sys.modules, "deeptutor.review.generate", module)


# -- store ------------------------------------------------------------------


def test_store_without_overwrite_keeps_the_stored_pack(review_home: Path) -> None:
    store = ReviewStore(review_home)
    original = build_fixture_spec()
    store.put_spec(original)
    before = store.spec_meta(SPEC_ID)
    replacement = _rewrite(original, title="replacement title")

    with pytest.raises(SpecExists):
        store.put_spec(replacement, overwrite=False)

    assert store.spec_meta(SPEC_ID) == before
    assert store.get_spec(SPEC_ID).title == original.title
    snapshot = review_home / f"review_{SPEC_ID}" / "spec.json"
    assert json.loads(snapshot.read_text(encoding="utf-8"))["title"] == original.title


def test_store_overwrite_replaces_the_pack(review_home: Path) -> None:
    store = ReviewStore(review_home)
    original = build_fixture_spec()
    store.put_spec(original)
    replacement = _rewrite(original, title="replacement title")

    stored, created = store.put_spec(replacement, overwrite=True)

    assert created is False
    assert stored.id == SPEC_ID
    assert store.get_spec(SPEC_ID).title == "replacement title"


def test_service_threads_overwrite_to_the_store(review_home: Path) -> None:
    service = ReviewService(ReviewStore(review_home))
    original = build_fixture_spec()
    service.put_spec(original)
    replacement = _rewrite(original, title="replacement title")

    with pytest.raises(SpecExists):
        service.put_spec(replacement, overwrite=False)

    assert service.store.get_spec(SPEC_ID).title == original.title


def test_api_maps_spec_exists_to_409() -> None:
    from fastapi import HTTPException

    from deeptutor.review.api import _handle

    mapped = _handle(SpecExists("kb-x-1"))

    assert isinstance(mapped, HTTPException)
    assert mapped.status_code == 409
    assert mapped.detail == {
        "code": "spec_exists",
        "message": "spec 'kb-x-1' already exists",
    }


# -- job --------------------------------------------------------------------


def test_generation_refuses_to_replace_an_existing_pack(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ReviewService(ReviewStore(review_home))
    original = build_fixture_spec()
    service.put_spec(original)
    before = service.store.spec_meta(SPEC_ID)
    _install_generate(monkeypatch, _rewrite(original, title="replacement title"))

    job = jobs.create_job(
        kb="hf",
        chapter="第2章_谐振",
        num_questions=3,
        question_types=None,
        difficulty="",
    )
    jobs._run_generation(job)

    stored = jobs.load_job(job["job_id"])
    assert stored is not None
    assert stored["status"] == "failed"
    assert stored["spec_id"] == SPEC_ID
    assert "already exists" in stored["error"]
    assert "replace=true" in stored["error"]
    assert service.store.spec_meta(SPEC_ID) == before
    assert service.store.get_spec(SPEC_ID).title == original.title


def test_generation_with_replace_true_overwrites_the_pack(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ReviewService(ReviewStore(review_home))
    original = build_fixture_spec()
    service.put_spec(original)
    _install_generate(monkeypatch, _rewrite(original, title="replacement title"))

    job = jobs.create_job(
        kb="hf",
        chapter="第2章_谐振",
        num_questions=3,
        question_types=None,
        difficulty="",
        replace=True,
    )
    jobs._run_generation(job)

    stored = jobs.load_job(job["job_id"])
    assert stored is not None
    assert stored["status"] == "done"
    assert stored["spec_id"] == SPEC_ID
    assert stored["error"] is None
    assert service.store.get_spec(SPEC_ID).title == "replacement title"


def test_generation_without_a_collision_still_stores(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = build_fixture_spec()
    _install_generate(monkeypatch, spec)

    job = jobs.create_job(
        kb="hf",
        chapter="第2章_谐振",
        num_questions=3,
        question_types=None,
        difficulty="",
    )
    jobs._run_generation(job)

    stored = jobs.load_job(job["job_id"])
    assert stored is not None
    assert stored["status"] == "done"
    assert stored["spec_id"] == SPEC_ID


def test_reused_legacy_id_is_also_refused(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_id = "kb-hf-chapter2"
    store = ReviewStore(review_home)
    legacy = _rename(build_fixture_spec(), legacy_id)
    payload = legacy.model_dump(mode="json")
    payload["sources"]["kb:hf·第2章_谐振"] = {
        "id": "kb:hf·第2章_谐振",
        "kind": "book_page",
        "title": "第2章_谐振",
        "kb": "hf",
        "ref": "chapter2",
        "snippet": "legacy identity",
    }
    payload["content_hash"] = ""
    legacy = ReviewSpec.model_validate(payload)
    store.put_spec(legacy)
    before = store.spec_meta(legacy_id)

    _install_generate(
        monkeypatch,
        _rewrite(build_fixture_spec(), title="replacement title"),
        legacy_id=legacy_id,
    )
    job = jobs.create_job(
        kb="hf",
        chapter="第2章_谐振",
        num_questions=3,
        question_types=None,
        difficulty="",
    )
    jobs._run_generation(job)

    stored = jobs.load_job(job["job_id"])
    assert stored is not None
    assert stored["status"] == "failed"
    assert stored["spec_id"] == legacy_id
    assert "already exists" in stored["error"]
    assert store.spec_meta(legacy_id) == before


def test_store_conflict_inside_the_job_is_reported_clearly(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = build_fixture_spec()
    _install_generate(monkeypatch, spec)

    class RacingService:
        class store:
            @staticmethod
            def spec_meta(spec_id: str) -> dict:
                raise SpecNotFound(spec_id)

        def put_spec(
            self,
            payload,
            *,
            overwrite: bool = True,
            allow_history_rewrite: bool = False,
        ) -> dict:
            raise SpecExists(payload.id)

    monkeypatch.setattr(service_module, "ReviewService", RacingService)
    job = jobs.create_job(
        kb="hf",
        chapter="第2章_谐振",
        num_questions=3,
        question_types=None,
        difficulty="",
    )
    jobs._run_generation(job)

    stored = jobs.load_job(job["job_id"])
    assert stored is not None
    assert stored["status"] == "failed"
    assert stored["spec_id"] == SPEC_ID
    assert "already exists" in stored["error"]
    assert "replace=true" in stored["error"]


# -- tool -------------------------------------------------------------------


def _install_fake_jobs(monkeypatch: pytest.MonkeyPatch) -> dict:
    seen: dict = {}
    module = types.ModuleType("deeptutor.review.jobs")

    def fake_start(**kwargs):
        seen.update(kwargs)
        return {"job_id": "gen-1-abcd", "status": "queued"}

    module.start_generation_job = fake_start
    monkeypatch.setitem(sys.modules, "deeptutor.review.jobs", module)
    return seen


def _install_chapter_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("deeptutor.review.generate")
    module.infer_chapters = lambda kb: [
        types.SimpleNamespace(name="第1章_绪论", doc_count=1)
    ]
    monkeypatch.setitem(sys.modules, "deeptutor.review.generate", module)


def _call(**kwargs) -> dict:
    result = asyncio.run(ReviewGenerateTool().execute(**kwargs))
    assert result.success is True, result.content
    return json.loads(result.content)


def test_tool_requires_a_literal_boolean_to_replace(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_chapter_generator(monkeypatch)
    seen = _install_fake_jobs(monkeypatch)

    _call(kb="hf", chapter="第1章_绪论")
    assert seen["replace"] is False

    _call(kb="hf", chapter="第1章_绪论", replace="true")
    assert seen["replace"] is False

    _call(kb="hf", chapter="第1章_绪论", replace=1)
    assert seen["replace"] is False

    _call(kb="hf", chapter="第1章_绪论", replace=True)
    assert seen["replace"] is True


def test_poll_of_a_collision_failure_tells_the_model_to_ask_first(
    review_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = jobs.create_job(
        kb="hf",
        chapter="第1章",
        num_questions=3,
        question_types=None,
        difficulty="",
    )
    job.update(
        status="failed",
        spec_id=SPEC_ID,
        error=(
            f"pack {SPEC_ID!r} already exists; start the generation again "
            "with replace=true only after the learner explicitly asks to "
            "replace it"
        ),
    )
    jobs.save_job(job)

    payload = _call(kb="hf", job_id=job["job_id"])

    assert payload["status"] == "failed"
    assert payload["replace"] is False
    assert "already exists" in payload["error"]
    assert "ask whether to keep" in payload["instruction"]
    assert "replace=true" in payload["instruction"]
