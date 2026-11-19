from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from deeptutor.review.export_tree import split_pack_title
from deeptutor.review.models import ReviewSpec
from deeptutor.review.provenance import chapter_paths, source_chapter
from deeptutor.review.storage import ReviewStore, SpecIntegrityError


def chapter_fingerprint(spec: ReviewSpec) -> str | None:
    chapters = {
        key
        for item in spec.items
        for source_id in item.source_ids
        if (key := source_chapter(spec, source_id)) is not None
    }
    if len(chapters) != 1:
        return None
    (kb, chapter), = chapters
    digest = hashlib.sha256((kb + chapter).encode("utf-8"))
    has_text = False
    try:
        for path in chapter_paths(kb, chapter):
            raw = path.read_bytes()
            if raw.decode("utf-8").strip():
                has_text = True
            digest.update(raw)
    except (OSError, ValueError):
        return None
    return digest.hexdigest() if has_text else None


def _manifest_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("manifest fields must be non-empty strings")
    return value.strip()


class BookManifest:
    def __init__(self, home: Path) -> None:
        self.by_id: dict[str, list[str]] = defaultdict(list)
        self.by_chapter: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.error = ""
        override = os.getenv("DEEPTUTOR_REVIEW_BOOK_MANIFEST", "").strip()
        try:
            path = Path(override).expanduser() if override else home / "book-manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int:
                raise ValueError("manifest schema_version must be an integer")
            if manifest["schema_version"] != 1 or not isinstance(manifest.get("books"), list):
                raise ValueError("expected a version 1 manifest with a books list")
            for book in manifest["books"]:
                if not isinstance(book, dict):
                    raise ValueError("manifest entries must be objects")
                subject = _manifest_text(book.get("subject"))
                chapter = _manifest_text(book.get("chapter"))
                fingerprint = _manifest_text(book.get("fingerprint")).lower()
                if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
                    raise ValueError("expected a SHA-256 fingerprint")
                generated_at = book.get("generated_at")
                if type(generated_at) in (int, float):
                    if not math.isfinite(generated_at):
                        raise ValueError("generated_at must be finite")
                else:
                    _manifest_text(generated_at)
                if "spec_id" in book:
                    self.by_id[_manifest_text(book["spec_id"])].append(fingerprint)
                self.by_chapter[(subject, chapter)].append(fingerprint)
        except FileNotFoundError:
            self.error = "book_manifest_missing"
        except OSError:
            self.error = "book_manifest_unreadable"
        except (ValueError, RuntimeError, OverflowError):
            self.error = "book_manifest_invalid"

    def match(self, spec_id: str, subject: str, chapter: str) -> tuple[str | None, str]:
        if self.error:
            return None, self.error
        matches = self.by_id.get(spec_id) or self.by_chapter.get((subject, chapter), [])
        if not matches:
            return None, "book_entry_missing"
        if len(matches) > 1:
            return None, "book_entry_ambiguous"
        return matches[0], ""


def artifact_status(
    store: ReviewStore, *, spec_id: str | None, subject: str | None,
) -> dict[str, Any]:
    manifest = BookManifest(store.home)
    items = []
    omitted = 0
    with store.artifact_rows(spec_id) as rows:
        for row in rows:
            try:
                spec = store.decode_spec(row["spec_id"], row["spec_json"])
            except (ValueError, SpecIntegrityError):
                omitted += 1
                continue
            pack_subject, chapter = split_pack_title(spec.title)
            if subject is not None and subject != pack_subject:
                continue
            book_hash, reason = manifest.match(spec.id, pack_subject, chapter)
            chapter_hash = row["chapter_fingerprint"]
            status = "unknown"
            if not chapter_hash:
                reason = "chapter_fingerprint_unavailable"
            elif book_hash is not None:
                status = "match" if chapter_hash == book_hash else "drift"
                reason = "fingerprints_match" if status == "match" else "fingerprints_differ"
            items.append({
                "spec_id": spec.id,
                "title": spec.title,
                "chapter_fingerprint": chapter_hash,
                "content_hash": row["content_hash"],
                "book_fingerprint": book_hash,
                "status": status,
                "reason": reason,
            })
    return {"items": items, "count": len(items), "omitted": omitted}
