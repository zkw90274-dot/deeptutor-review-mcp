#!/usr/bin/env python
"""Index the diagrams every living book carries, section by section.

Usage:
    python scripts/build-book-figures.py            # write <review home>/book-figures.json
    python scripts/build-book-figures.py --dry-run  # report only
    python scripts/build-book-figures.py -v         # one line per section with a figure

A living book embeds its diagrams as inline SVG: the build reads figures/*.svg
and the page carries <figure data-fig="name"><figcaption>...</figcaption></figure>
inside the section that needs it. That makes the diagrams invisible to a plan:
a card names section 2.1 系统总线, and the learner has no way to know 图 2-1 ..
2-3 sit there. This script reads the built books and records which section owns
which figure, so the review tools can point at the diagram without opening the
HTML.

Inputs (deployment data, both git-ignored):
    <review home>/books.json          subject -> {"html": <built book path>}
    <review home>/book-sections.json  chapter -> note -> [{no, title, anchor}]

Output, consumed by deeptutor.review.materials.load_book_figures():
    {"version": 1, "generated_at": "...",
     "books": {"微机": {"2.1": {"title": "系统总线", "anchor": "s2-1",
                                "figures": [{"label": "图 2-1", "caption": "..."}]}}},
     "chapters": {"第2讲": {"2.1": [{"label": "图 2-1", "caption": "..."}]}}}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SECTION = re.compile(r'<section class="section"([^>]*)>(.*?)</section>', re.S)
FIGURE = re.compile(r'<figure data-fig="([^"]*)"[^>]*>(.*?)</figure>', re.S)
CAPTION = re.compile(r"<figcaption>(.*?)</figcaption>", re.S)
ATTR = re.compile(r'([\w-]+)="([^"]*)"')
TAG = re.compile(r"<[^>]+>")
NUMBER = re.compile(r"^\s*图\s*(\d+)\s*[-–—]\s*(\d+)")
WHITESPACE = re.compile(r"\s+")


def review_home() -> Path:
    import os

    root = os.environ.get("DEEPTUTOR_REVIEW_HOME")
    if root:
        return Path(root).expanduser()
    return REPO_ROOT / "data" / "user" / "workspace" / "review"


def _clean(text: str) -> str:
    return WHITESPACE.sub(" ", unescape(TAG.sub("", text))).strip()


def _label(caption: str) -> str:
    """The number the book itself prints (图 2-1), or "" when it prints none."""
    match = NUMBER.match(caption)
    return f"图 {match.group(1)}-{match.group(2)}" if match else ""


def read_book(path: Path) -> dict[str, dict[str, object]]:
    """Section number -> {title, anchor, figures} for one built book."""
    html = path.read_text(encoding="utf-8")
    out: dict[str, dict[str, object]] = {}
    for attrs, body in SECTION.findall(html):
        parsed = dict(ATTR.findall(attrs))
        number = str(parsed.get("data-no") or "").strip()
        if not number or str(parsed.get("data-appendix") or "") == "1":
            continue
        figures = []
        for _name, block in FIGURE.findall(body):
            caption = CAPTION.search(block)
            text = _clean(caption.group(1)) if caption else ""
            if text:
                figures.append({"label": _label(text), "caption": text})
        if not figures:
            continue
        out[number] = {
            "title": _clean(str(parsed.get("data-title") or "")),
            "anchor": str(parsed.get("id") or ""),
            "figures": figures,
        }
    return out


def build(home: Path, verbose: bool = False) -> dict[str, object]:
    books_path = home / "books.json"
    sections_path = home / "book-sections.json"
    books = json.loads(books_path.read_text(encoding="utf-8"))
    sections = {}
    if sections_path.exists():
        payload = json.loads(sections_path.read_text(encoding="utf-8"))
        sections = payload.get("chapters") or {}

    per_book: dict[str, dict[str, object]] = {}
    for subject, info in books.items():
        rel = str((info or {}).get("html") or "").strip()
        if not rel:
            continue
        path = Path(rel)
        if not path.is_absolute():
            path = REPO_ROOT / rel
        if not path.exists():
            print(f"warn: {subject}: missing {path}", file=sys.stderr)
            continue
        found = read_book(path)
        if found:
            per_book[subject] = found
        if verbose:
            for number, entry in found.items():
                print(f"{subject:<8} {number:<6} {len(entry['figures'])} figure(s)  {entry['title']}")

    # chapter -> section number -> figures: join the planned outline onto the
    # book by section number, and only keep a hit whose title also matches, so
    # the same number in another book cannot claim it.
    chapters: dict[str, dict[str, list[dict[str, str]]]] = {}
    for chapter, notes in sections.items():
        rows: dict[str, list[dict[str, str]]] = {}
        for info in (notes or {}).values():
            entries = info if isinstance(info, list) else [info]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                number = str(entry.get("no") or "").strip()
                title = str(entry.get("title") or "").strip()
                if not number:
                    continue
                for found in per_book.values():
                    hit = found.get(number)
                    if hit and str(hit["title"]) == title and number not in rows:
                        rows[number] = [
                            {"label": figure["label"], "caption": figure["caption"]}
                            for figure in hit["figures"]
                        ]
        if rows:
            chapters[str(chapter)] = rows

    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "books": per_book,
        "chapters": chapters,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-home", default=None, help="review home (default: <workspace>/review)")
    parser.add_argument("--out", default=None, help="output file (default: <review home>/book-figures.json)")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    home = Path(args.review_home).expanduser() if args.review_home else review_home()
    payload = build(home, verbose=args.verbose)
    books = payload["books"]
    chapters = payload["chapters"]
    total = sum(len(entry["figures"]) for found in books.values() for entry in found.values())
    mapped = sum(len(rows) for rows in chapters.values())
    print(f"books: {len(books)}  sections with figures: "
          f"{sum(len(found) for found in books.values())}  figures: {total}")
    print(f"plan chapters mapped: {len(chapters)}  sections: {mapped}")
    for subject, found in books.items():
        print(f"  {subject}: {len(found)} sections / "
              f"{sum(len(entry['figures']) for entry in found.values())} figures")
    if args.dry_run:
        return 0
    out = Path(args.out).expanduser() if args.out else home / "book-figures.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
