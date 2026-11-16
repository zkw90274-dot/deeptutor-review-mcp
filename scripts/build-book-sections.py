#!/usr/bin/env python
"""Map every planned note onto the living-book section that covers it.

Usage:
    python scripts/build-book-sections.py            # write <review home>/book-sections.json
    python scripts/build-book-sections.py --dry-run  # report the matches only
    python scripts/build-book-sections.py -v         # one line per note
    python scripts/build-book-sections.py --from-plan --subject <科目>
        # a lecture note covers several sections: a task labelled §7.–§10.
        # claims the sections those two anchors land on, and the remaining
        # parts of the chapter share the rest by task minutes.

A plan is built from knowledge-base files, which number material their own way
(MC1-2-1, 第10讲_课程重点). The learner reads the living book, whose sections
are numbered 1.7, 2.3, ...  This script scores each planned note against the
sections of the book chapter that the plan chapter maps to (book-map.json),
using shared character trigrams, and keeps a match only when it is clear:
coverage of the note >= 8%, best section 1.4x the runner-up, and a section
long enough to score. Everything else is left out on purpose, so the card
keeps naming the note itself.

Output, consumed by deeptutor.review.materials.load_book_sections():
    {"version": 1, "generated_at": "...", "chapters": {
        "第1章_离散时间信号的由来": {"MC1-2-1-...": {"no": "1.7",
            "title": "离散时间系统：线性与移不变", "anchor": "s01-7"}}}}

Deployment data stays local: <review home>/book-sections.json and
<review home>/books.json are git-ignored, so the published package needs none.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SECTION_TAG = re.compile(r'<section class="section"([^>]*)>', re.S)
ATTR = re.compile(r'([\w-]+)="([^"]*)"')
NOISE = re.compile(r"<script.*?</script>|<style.*?</style>", re.S)
TAG = re.compile(r"<[^>]+>")
DEFAULT_MIN_SCORE = 0.08
DEFAULT_MIN_RATIO = 1.4
DEFAULT_MIN_SECTION_GRAMS = 150
DEFAULT_MAX_DOC_RATIO = 3.0
DEFAULT_MAX_SECTIONS = 2
DEFAULT_MAX_PER_SECTION = 3
DOC_CHARS = 8000


def _doc_base(label: str) -> str:
    """The document a plan label was sliced from (第7讲_课程重点 §3 → 第7讲_课程重点)."""
    base = re.split(r"s*§", str(label), maxsplit=1)[0]
    return re.sub(r"[（(][一二三四五六七八九十d]+[）)]$", "", base).strip()


def trigrams(text: str) -> set[str]:
    body = re.sub(r"[^\w\u4e00-\u9fff]+", "", text.lower())
    if len(body) < 3:
        return set()
    return {body[index : index + 3] for index in range(len(body) - 2)}


def chapter_number(value: str) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    return str(int(digits)) if digits else ""


def read_book(path: Path) -> list[dict[str, object]]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    marks = list(SECTION_TAG.finditer(html))
    sections: list[dict[str, object]] = []
    for index, mark in enumerate(marks):
        attrs = dict(ATTR.findall(mark.group(1)))
        end = marks[index + 1].start() if index + 1 < len(marks) else len(html)
        body = NOISE.sub(" ", html[mark.end() : end])
        text = unescape(TAG.sub(" ", body))
        sections.append(
            {
                "anchor": attrs.get("id", "").strip(),
                "ch": chapter_number(attrs.get("data-ch", "")),
                "no": attrs.get("data-no", "").strip(),
                "title": attrs.get("data-title", "").strip(),
                "grams": trigrams(text),
            }
        )
    return sections


def note_sources(subject: str, chapter: str, labels: list[str]) -> dict[str, str]:
    """Plan note label -> raw file name for one chapter."""
    from deeptutor.review.generate import chapter_sources
    from deeptutor.review.materials import short_title

    try:
        names = chapter_sources(subject, chapter, limit=500)
    except Exception:  # noqa: BLE001 - a missing KB only leaves notes unmapped
        return {}
    bases = {name: short_title(name, chapter) for name in names}
    found: dict[str, str] = {}
    for label in labels:
        best = ""
        for name, base in bases.items():
            if not base:
                continue
            if label == base or label.startswith(base + "（") or label.startswith(base + " §"):
                if not best or len(base) > len(bases[best]):
                    best = name
        if best:
            found[label] = best
    return found


def pick(
    doc_grams: set[str],
    pool: list[dict[str, object]],
    *,
    min_score: float,
    min_ratio: float,
    min_section_grams: int,
    max_doc_ratio: float,
    max_sections: int,
) -> tuple[dict[str, object] | None, float, float]:
    if not doc_grams:
        return None, 0.0, 0.0
    scored: list[tuple[float, dict[str, object]]] = []
    for section in pool:
        grams = section["grams"]
        if len(grams) < min_section_grams:
            continue
        scored.append((len(doc_grams & grams) / len(doc_grams), section))
    if not scored:
        return None, 0.0, 0.0
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < min_score:
        return None, best_score, second
    if second and best_score < min_ratio * second:
        return None, best_score, second
    # A lecture note that summarises half the chapter matches several sections
    # about equally well; picking the top one would send the learner to the
    # wrong place. Such a note stays unnamed and the card keeps the chapter
    # pointer instead.
    if sum(1 for score, _ in scored if score >= 0.5 * best_score) > max_sections:
        return None, best_score, second
    # A note that is an entire lecture covers many sections at once; its best
    # section is whichever one happens to share the most text, not the place
    # to read it. Only a note no larger than a couple of sections is specific
    # enough to name one.
    if len(doc_grams) > max_doc_ratio * len(best["grams"]):
        return None, best_score, second
    return best, best_score, second


SKIP_LABELS = ("复核", "README", "核对报告")
RANGE = re.compile("§\\s*(\\d+)")


def kb_sections(text: str) -> list[dict[str, object]]:
    """Split a knowledge-base note on its «## N. title» headings."""
    out: list[dict[str, object]] = []
    for part in re.split(r"^##\s+", text, flags=re.M)[1:]:
        lines = part.split("\n", 1)
        label = lines[0].strip()
        body = lines[1] if len(lines) > 1 else ""
        number = re.match(r"(\d+)\s*[.、]", label)
        out.append(
            {
                "number": int(number.group(1)) if number else -1,
                "label": label,
                "grams": trigrams(body[:DOC_CHARS]),
            }
        )
    return out


def align_pool(pool: list[dict[str, object]], kb: list[dict[str, object]]) -> list[int]:
    """Monotone DP: the book section each knowledge-base section lands on."""
    rows, cols = len(pool), len(kb)
    if not rows or not cols:
        return [0] * cols
    score = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        left = pool[i]["grams"]
        for j in range(cols):
            right = kb[j]["grams"]
            score[i][j] = len(left & right) / max(1, len(left))
    best = [[0.0] * cols for _ in range(rows)]
    back = [[-1] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            previous, arg = 0.0, -1
            if i > 0:
                for earlier in range(j):
                    if best[i - 1][earlier] > previous:
                        previous, arg = best[i - 1][earlier], earlier
            best[i][j] = previous + score[i][j]
            back[i][j] = arg
    j = max(range(cols), key=lambda value: best[rows - 1][value])
    pairs: list[tuple[int, int]] = []
    for i in range(rows - 1, -1, -1):
        pairs.append((i, j))
        j = back[i][j]
    pairs.reverse()
    landing = [0] * cols
    filled = 0
    for i, j in pairs:
        while filled <= j:
            landing[filled] = i
            filled += 1
    while filled < cols:
        landing[filled] = rows - 1
        filled += 1
    return landing


def split_by_minutes(
    minutes: list[int], pool: list[dict[str, object]], weights: list[int]
) -> list[list[dict[str, object]]]:
    """Hand out consecutive sections so each task's share matches its minutes."""
    total_minutes = sum(minutes) or 1
    total_weight = sum(weights) or 1
    out: list[list[dict[str, object]]] = [[] for _ in minutes]
    acc_minutes = 0
    acc_weight = 0
    index = 0
    for position, task_minutes in enumerate(minutes):
        acc_minutes += task_minutes
        share = acc_minutes / total_minutes
        while index < len(pool):
            out[position].append(pool[index])
            acc_weight += weights[index]
            index += 1
            if acc_weight / total_weight >= share - 1e-9:
                break
    for position in range(len(out) - 1, 0, -1):
        if not out[position] and out[position - 1]:
            out[position] = [out[position - 1].pop()]
    while index < len(pool):
        out[-1].append(pool[index])
        index += 1
    return out


def plan_splits(
    subject: str,
    chapter: str,
    db_path: Path,
    plan_id: str,
    pool: list[dict[str, object]],
) -> dict[str, list[dict[str, str]]]:
    """Per plan task, the book sections it should send the learner to."""
    from deeptutor.review.generate import chapter_sources
    from deeptutor.review.materials import _raw_root

    try:
        names = chapter_sources(subject, chapter, limit=500)
    except Exception:  # noqa: BLE001 - a missing KB just leaves the chapter alone
        return {}
    note = ""
    for name in names:
        if "课程重点" in name:
            try:
                note = (_raw_root(subject) / name).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            break
    if not note:
        return {}
    kb = kb_sections(note)
    if not kb:
        return {}
    landing = align_pool(pool, kb)
    weights = [len(item["grams"]) or 1 for item in kb]
    weights += [1] * (len(pool) - len(weights))
    weights = weights[: len(pool)]

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT title, minutes, chapter FROM review_daily_tasks WHERE plan_id = ?"
        " AND kind = 'study' ORDER BY local_date, seq",
        (plan_id,),
    ).fetchall()
    conn.close()
    entries: list[tuple[str, int]] = []
    for title, minutes, task_chapter in rows:
        name = (task_chapter or str(title).split(" · ")[0]).strip()
        if name != chapter:
            continue
        entries.append((str(title).split(" · ", 1)[-1].strip(), int(minutes or 0)))
    if not entries:
        return {}

    spans: dict[int, list[dict[str, object]]] = {}
    plain: list[int] = []
    for index, (doc, _minutes) in enumerate(entries):
        if any(word in doc for word in SKIP_LABELS):
            continue
        numbers = [int(mark.group(1)) for mark in RANGE.finditer(doc)]
        if len(numbers) >= 2:
            low = landing[min(numbers)]
            high = landing[max(numbers)]
            spans[index] = pool[low : high + 1]
        else:
            plain.append(index)
    if plain:
        slices = split_by_minutes([entries[i][1] for i in plain], pool, weights)
        for position, index in enumerate(plain):
            spans[index] = slices[position]

    kept: dict[str, list[dict[str, str]]] = {}
    for index, (doc, _minutes) in enumerate(entries):
        picked = spans.get(index) or []
        if not picked or doc in kept:
            continue
        kept[doc] = [
            {"no": str(item["no"]), "title": str(item["title"]), "anchor": str(item["anchor"])}
            for item in picked
        ]
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description="Map planned notes to living-book sections")
    parser.add_argument("--home", default=None, help="review home (default: resolved by the package)")
    parser.add_argument("--books", default=None, help="books.json (default: <review home>/books.json)")
    parser.add_argument("--out", default=None, help="output (default: <review home>/book-sections.json)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--min-ratio", type=float, default=DEFAULT_MIN_RATIO)
    parser.add_argument("--min-section-grams", type=int, default=DEFAULT_MIN_SECTION_GRAMS)
    parser.add_argument("--max-doc-ratio", type=float, default=DEFAULT_MAX_DOC_RATIO)
    parser.add_argument("--max-sections", type=int, default=DEFAULT_MAX_SECTIONS)
    parser.add_argument("--max-per-section", type=int, default=DEFAULT_MAX_PER_SECTION)
    parser.add_argument(
        "--from-plan",
        action="store_true",
        help="split each chapter by the plan's own §-ranges and minutes (list values)",
    )
    parser.add_argument("--subject", default="", help="only this subject (with --from-plan)")
    args = parser.parse_args()
    if args.home:
        os.environ["DEEPTUTOR_REVIEW_HOME"] = str(Path(args.home).resolve())

    from deeptutor.review.materials import _raw_root, load_book_map
    from deeptutor.review.storage import review_home

    home = review_home()
    books_path = Path(args.books) if args.books else home / "books.json"
    out_path = Path(args.out) if args.out else home / "book-sections.json"
    try:
        books_cfg = json.loads(books_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"cannot read {books_path}: {error}", file=sys.stderr)
        return 2
    book_map = load_book_map()

    db = home / "review.sqlite3"
    conn = sqlite3.connect(db)
    plans = conn.execute(
        "SELECT plan_id, subject, plan_json FROM review_plans WHERE status = 'active'"
    ).fetchall()
    labels: dict[tuple[str, str], list[str]] = {}
    for plan_id, _subject, plan_json in plans:
        for row in conn.execute(
            "SELECT DISTINCT chapter, title FROM review_daily_tasks "
            "WHERE plan_id = ? AND kind = 'study'",
            (plan_id,),
        ):
            chapter, title = row
            chapter = (chapter or str(title).split(" · ")[0]).strip()
            doc = str(title).split(" · ", 1)[-1].strip()
            bucket = labels.setdefault((plan_id, chapter), [])
            if doc and doc not in bucket:
                bucket.append(doc)
    conn.close()

    books: dict[str, list[dict[str, object]]] = {}
    chapters: dict[str, dict[str, dict[str, str]]] = {}
    total = matched = 0
    for plan_id, subject, plan_json in plans:
        try:
            chapters_of_plan = json.loads(plan_json).get("chapters") or []
        except ValueError:
            chapters_of_plan = []
        for chapter in chapters_of_plan:
            plan_labels = labels.get((plan_id, str(chapter)))
            if not plan_labels:
                continue
            targets = {
                chapter_number(anchor)
                for _label, anchor in book_map.get(str(chapter), [])
                if chapter_number(anchor)
            }
            if not targets:
                continue
            entry = books_cfg.get(str(subject))
            html_path = Path(entry.get("html", "")) if isinstance(entry, dict) else Path(str(entry or ""))
            if not html_path.is_absolute():
                html_path = REPO_ROOT / html_path
            if not html_path.is_file():
                print(f"missing book html for {subject}: {html_path}", file=sys.stderr)
                continue
            if str(subject) not in books:
                books[str(subject)] = read_book(html_path)
            pool = [s for s in books[str(subject)] if s["ch"] in targets and s["no"] and s["title"]]
            if not pool:
                print(f"no sections for {subject} / {chapter} (chapters {sorted(targets)})", file=sys.stderr)
                continue
            if args.subject and str(subject) != args.subject:
                continue
            if args.from_plan:
                kept_parts = plan_splits(
                    str(subject), str(chapter), db, str(plan_id), pool
                )
                if kept_parts:
                    chapters[str(chapter)] = kept_parts
                print(f"{subject:8s} {chapter:44s} {len(kept_parts)} note parts")
                continue
            sources = note_sources(str(subject), str(chapter), plan_labels)
            kept: dict[str, dict[str, str]] = {}
            for label in plan_labels:
                total += 1
                name = sources.get(label)
                if not name:
                    if args.verbose:
                        print(f"  - {chapter} | {label} | no source file")
                    continue
                try:
                    text = (_raw_root(str(subject)) / name).read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                section, score, second = pick(
                    trigrams(text[:DOC_CHARS]),
                    pool,
                    min_score=args.min_score,
                    min_ratio=args.min_ratio,
                    min_section_grams=args.min_section_grams,
                    max_doc_ratio=args.max_doc_ratio,
                    max_sections=args.max_sections,
                )
                if section is None:
                    if args.verbose:
                        print(f"  - {chapter} | {label} | best {score:.2f} vs {second:.2f}")
                    continue
                kept[label] = {
                    "no": str(section["no"]),
                    "title": str(section["title"]),
                    "anchor": str(section["anchor"]),
                }
                matched += 1
                if args.verbose:
                    print(
                        f"  + {chapter} | {label} -> {section['no']} {section['title']}"
                        f" ({score:.2f} vs {second:.2f})"
                    )
            # One lecture sliced into parts lands every slice on the same
            # section when the match is really "this note covers the chapter"
            # (第7讲_课程重点 乘除法/移位/转移 all became 7.1). That only
            # counts when the crowded notes come from one document; distinct
            # notes on one section are a normal day in the book.
            groups: dict[str, list[str]] = {}
            for label, item in kept.items():
                groups.setdefault(item["anchor"], []).append(label)
            crowded = max(groups.items(), key=lambda pair: len(pair[1]), default=("", []))
            bases = {_doc_base(label) for label in crowded[1]}
            if len(crowded[1]) >= args.max_per_section and len(bases) == 1:
                print(
                    f"{subject:8s} {chapter:44s} dropped: {len(crowded[1])} slices "
                    f"of {next(iter(bases))} on {crowded[0]}",
                    file=sys.stderr,
                )
                kept = {}
            if kept:
                chapters[str(chapter)] = kept
            print(f"{subject:8s} {chapter:44s} {len(kept)}/{len(plan_labels)} mapped")

    # A run over one subject must not erase the chapters mapped for the others.
    existing: dict[str, object] = {}
    if out_path.is_file():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8")).get("chapters") or {}
        except (OSError, ValueError):
            existing = {}
    existing.update(chapters)
    chapters = existing  # type: ignore[assignment]
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "generated by scripts/build-book-sections.py; deployment data",
        "chapters": chapters,
    }
    print(f"\n{matched}/{total} planned notes mapped across {len(chapters)} chapters")
    if args.dry_run:
        print(f"dry run: would write {out_path}")
        return 0
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
