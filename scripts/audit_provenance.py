from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from deeptutor.review.models import ReviewSpec
from deeptutor.review.provenance import assess
from deeptutor.review.qa import run_gates


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit source support in a Review Studio database snapshot.")
    parser.add_argument("--home", required=True, type=Path, help="Snapshot directory containing review.sqlite3")
    parser.add_argument("--spec", required=True, help="Exact review pack ID")
    parser.add_argument("--kb-root", type=Path, help="Knowledge-base root (defaults to the configured DeepTutor root)")
    args = parser.parse_args()
    path = (args.home / "review.sqlite3").resolve()
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as database:
            database.execute("PRAGMA query_only=ON")
            row = database.execute("SELECT spec_json FROM review_specs WHERE spec_id = ?", (args.spec,)).fetchone()
    except sqlite3.Error as exc:
        parser.error(f"could not read snapshot ({type(exc).__name__})")
    if row is None:
        parser.error("spec_id not found in the snapshot")
    spec = ReviewSpec.model_validate_json(row[0])
    if spec.id != args.spec:
        parser.error("stored pack ID differs from its database key")
    report, anchors = assess(spec, run_gates(spec), kb_root=args.kb_root)
    counts = Counter(report["support"].values())
    print(json.dumps({
        "spec_id": spec.id,
        "title": spec.title,
        "content_hash": spec.content_hash,
        "state": report["state"],
        "counts": {level: counts[level] for level in ("exact", "term", "weak", "none")},
        "items": [{"item_id": item.id, **anchors[item.id]} for item in spec.items],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
