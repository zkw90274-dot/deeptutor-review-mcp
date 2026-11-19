from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3

import pytest

from deeptutor.review.mcp_server import TOOL_CLASSES
from deeptutor.review.models import ReviewSpec
from deeptutor.review.storage import ReviewStore, SpecExists


CHAPTER = "第2章_谐振"
SUBJECT = "测试科目"
ORIGINAL = "谐振频率为５０ Hz。\r\n".encode()
DRIFTED = "谐振频率为50 Hz。\n".encode()


@pytest.fixture(autouse=True)
def _isolate_manifest(monkeypatch):
    monkeypatch.delenv("DEEPTUTOR_REVIEW_BOOK_MANIFEST", raising=False)


@pytest.fixture()
def chapter_pack(service, fixture_spec, tmp_path, monkeypatch):
    from deeptutor.review import generate

    root = tmp_path / "knowledge-bases"
    chapter = root / "hf" / "raw" / f"001_{CHAPTER}.md"
    chapter.parent.mkdir(parents=True)
    chapter.write_bytes(ORIGINAL)
    chapter.with_name("002_第3章_滤波.md").write_text("other chapter", encoding="utf-8")
    monkeypatch.setattr(generate, "_knowledge_bases_root", lambda: root)
    payload = fixture_spec.model_dump(mode="json")
    payload.update(title=f"{SUBJECT} · {CHAPTER} 复习包", content_hash="")
    for source in payload["sources"].values():
        source.update(kb="hf", ref=CHAPTER)
    return ReviewSpec.model_validate(payload), chapter


def _fingerprint(raw):
    return hashlib.sha256(f"hf{CHAPTER}".encode() + raw).hexdigest()


def _artifact(service, spec_id):
    with service.store._read_connection() as conn:
        return dict(conn.execute("SELECT * FROM review_artifacts WHERE spec_id = ?", (spec_id,)).fetchone())


def _call(**kwargs):
    result = asyncio.run(TOOL_CLASSES["review_artifact_status"]().execute(**kwargs))
    return result, json.loads(result.content)


def _status(**kwargs):
    result, payload = _call(**kwargs)
    assert result.success, payload
    return payload["items"]


def _manifest(path, *books):
    path.write_text(json.dumps({"schema_version": 1, "books": books}), encoding="utf-8")


def _book(fingerprint, **kwargs):
    return {"subject": SUBJECT, "chapter": CHAPTER, "fingerprint": fingerprint,
            "generated_at": "2026-11-19T12:00:00Z", **kwargs}


def test_put_and_review_correct_store_fingerprint(service, chapter_pack):
    spec, chapter = chapter_pack
    first = service.put_spec(spec)
    baseline = _artifact(service, spec.id)
    assert baseline["chapter_fingerprint"] == _fingerprint(ORIGINAL)
    assert baseline["fingerprint_basis"] == "raw_chapter"
    assert baseline["content_hash"] == first["content_hash"]
    assert 0 < baseline["generated_at"] <= baseline["updated_at"]
    service.put_spec(spec)
    assert _artifact(service, spec.id)["chapter_fingerprint"] == baseline["chapter_fingerprint"]

    chapter.write_bytes(DRIFTED)
    with pytest.raises(SpecExists):
        service.put_spec(spec, overwrite=False)
    assert _artifact(service, spec.id)["chapter_fingerprint"] == baseline["chapter_fingerprint"]
    item = spec.items[0]
    result = asyncio.run(TOOL_CLASSES["review_correct"]().execute(
        spec_id=spec.id, item_id=item.id, patch={"stem": item.stem + " 请说明理由。"},
    ))
    assert result.success and json.loads(result.content).get("revision") == 1, result.content
    corrected = _artifact(service, spec.id)
    assert corrected["chapter_fingerprint"] == _fingerprint(DRIFTED)
    assert corrected["content_hash"] == service.store.get_spec(spec.id).content_hash != first["content_hash"]
    assert corrected["generated_at"] == baseline["generated_at"]
    assert corrected["updated_at"] >= baseline["updated_at"]
    assert service.store.get_spec(spec.id).concepts == spec.concepts


@pytest.mark.parametrize("source_state", ["missing", "unreadable", "invalid_utf8", "empty"])
def test_unavailable_source_records_unknown(service, chapter_pack, monkeypatch, source_state):
    spec, chapter = chapter_pack
    if source_state == "missing":
        chapter.unlink()
    elif source_state == "unreadable":
        original = type(chapter).open

        def deny_chapter(path, *args, **kwargs):
            if path == chapter:
                raise PermissionError("synthetic denial")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(type(chapter), "open", deny_chapter)
    else:
        chapter.write_bytes(b"\xff" if source_state == "invalid_utf8" else b"")
    service.put_spec(spec)
    record = _artifact(service, spec.id)
    assert record["fingerprint_basis"] == "unknown"
    assert record["chapter_fingerprint"] is None
    _manifest(service.store.home / "book-manifest.json", _book(_fingerprint(ORIGINAL)))
    assert _status()[0]["status"] == "unknown"


def test_status_prefers_id_then_falls_back_to_subject_chapter(service, chapter_pack, tmp_path, monkeypatch):
    spec, _ = chapter_pack
    service.put_spec(spec)
    other = spec.model_copy(update={"id": "separate-pack", "content_hash": ""})
    service.put_spec(other)
    default = service.store.home / "book-manifest.json"
    override = tmp_path / "override.json"
    _manifest(default, _book(_fingerprint(DRIFTED)))
    assert _status(spec_id=spec.id)[0]["status"] == "drift"
    _manifest(override, _book(_fingerprint(DRIFTED)), {
        **_book(_fingerprint(ORIGINAL), spec_id=spec.id), "subject": "alias", "chapter": "alias",
    })
    monkeypatch.setenv("DEEPTUTOR_REVIEW_BOOK_MANIFEST", str(override))
    rows = {row["spec_id"]: row for row in _status(subject=SUBJECT)}
    assert rows[spec.id] == {
        "spec_id": spec.id, "title": spec.title, "chapter_fingerprint": _fingerprint(ORIGINAL),
        "content_hash": spec.content_hash, "book_fingerprint": _fingerprint(ORIGINAL),
        "status": "match", "reason": "fingerprints_match",
    }
    assert rows[other.id]["status"] == "drift"
    _manifest(override, _book(_fingerprint(DRIFTED)), _book(_fingerprint(ORIGINAL)))
    assert _status(spec_id=spec.id)[0]["reason"] == "book_entry_ambiguous"
    _manifest(override, {**_book(_fingerprint(ORIGINAL)), "subject": "other"})
    assert _status(spec_id=spec.id)[0]["reason"] == "book_entry_missing"
    assert _status(spec_id=spec.id, subject="other") == []
    assert _status(spec_id="missing-pack") == []


@pytest.mark.parametrize("manifest", [None, "broken JSON", '{"schema_version":2,"books":[]}',
                                      '{"schema_version":1,"books":[{"fingerprint":"fake"}]}'])
def test_manifest_unavailable_is_read_only(service, chapter_pack, monkeypatch, manifest):
    spec, chapter = chapter_pack
    service.put_spec(spec)
    manifest_path = service.store.home / "book-manifest.json"
    if manifest is not None:
        manifest_path.write_text(manifest, encoding="utf-8")
    def persisted_files():
        return {path: path.read_bytes() for path in service.store.home.rglob("*")
                if path.is_file() and path.name not in {"review.sqlite3-shm", "review.sqlite3-wal"}}

    before = persisted_files()

    def deny_writes(*args, **kwargs):
        pytest.fail("artifact status attempted a write transaction")

    monkeypatch.setattr(ReviewStore, "_transaction", deny_writes)
    chapter.unlink()
    row, = _status()
    assert row["status"] == "unknown"
    assert row["reason"] == ("book_manifest_missing" if manifest is None else "book_manifest_invalid")
    assert row["chapter_fingerprint"] == _fingerprint(ORIGINAL)
    assert row["book_fingerprint"] is None
    assert persisted_files() == before
    wal = service.store.db_path.with_suffix(".sqlite3-wal")
    assert not wal.exists() or wal.stat().st_size == 0


def test_v7_migration_preserves_data_without_inventing_baseline(service, chapter_pack):
    spec, _ = chapter_pack
    service.put_spec(spec)
    with sqlite3.connect(service.store.db_path) as conn:
        conn.execute("DROP TABLE review_artifacts")
        conn.execute("UPDATE review_schema_migrations SET version = 7 WHERE version = 9")
    before = service.store.db_path.read_bytes()
    result, payload = _call(spec_id=spec.id)
    assert result.success and payload["error"] == "review_store_unavailable"
    assert service.store.db_path.read_bytes() == before
    migrated = ReviewStore(service.store.home)
    assert migrated.get_spec(spec.id).content_hash == spec.content_hash
    _manifest(migrated.home / "book-manifest.json", _book(_fingerprint(ORIGINAL)))
    assert _status()[0]["reason"] == "chapter_fingerprint_unavailable"
    with migrated._read_connection() as conn:
        assert conn.execute("SELECT MAX(version) FROM review_schema_migrations").fetchone()[0] == 9
        assert conn.execute("SELECT COUNT(*) FROM review_artifacts").fetchone()[0] == 0


@pytest.mark.parametrize("kwargs", [{"spec_id": 7}, {"spec_id": "../pack"}, {"spec_id": "  "},
                                    {"subject": []}, {"subject": "  "}])
def test_artifact_status_rejects_invalid_filters(kwargs):
    result, payload = _call(**kwargs)
    assert result.success
    assert payload["error"] == f"invalid_{next(iter(kwargs))}"
