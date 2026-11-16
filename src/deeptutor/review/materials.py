"""Reading parts: a chapter's raw notes split where the learner can stop.

A plan used to cut a chapter into equal minute slices and call them
"第 1/18 段", which told nobody what to open. This module walks the chapter's
own raw documents instead: every part is one document, or a section range of
one, named the way the file names it (第10讲_课程重点 §一–§三). The planner
packs those parts into days, so a daily task always points at real text.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from deeptutor.review.study_plan import ChapterWork, Part

#: Reading speed for Chinese technical notes (characters per minute).
CHARS_PER_MINUTE = 500
#: A doc shorter than this never becomes its own day.
_MIN_PART_MINUTES = 3
#: Label length guard, so a kanban line stays readable.
_MAX_TITLE_CHARS = 26

_ORDINAL = re.compile(r"^\d{3,4}_")
_EXTENSION = re.compile(r"\.(md|markdown|txt)$", re.IGNORECASE)
_CHAPTER_PREFIX = re.compile(r"^第[0-9一二三四五六七八九十百]+章_")
_PAGE_HEADING = re.compile(r"^第\s*[0-9]+\s*页")
_SUBJECT_TOKENS = (
    "数字信号处理",
    "微机原理及应用",
    "电磁场与微波技术",
    "高频电子线路",
    "信号与系统",
    "模拟电子技术",
    "数字电子技术",
    "VHDL",
)

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


_LECTURE_PREFIX = re.compile(r"^(第[0-9一二三四五六七八九十百]+讲)_")


def short_title(name: str, chapter: str = "") -> str:
    """The note's own name: 016_第2章__第1讲_课程重点.md -> 第1讲_课程重点.

    A chapter that is itself 「第2讲」 drops the repeated prefix, so the line
    reads 「第2讲 · 课程重点」 instead of 「第2讲 · 第2讲_课程重点」.
    """
    stem = _EXTENSION.sub("", name)
    stem = _ORDINAL.sub("", stem)
    if "__" in stem:
        stem = stem.rsplit("__", 1)[-1]
    for token in _SUBJECT_TOKENS:
        if stem.startswith(token + "_"):
            stem = stem[len(token) + 1 :]
            break
    stem = _CHAPTER_PREFIX.sub("", stem)
    lecture = _LECTURE_PREFIX.match(stem)
    if lecture and chapter.startswith(lecture.group(1)):
        stem = stem[len(lecture.group(1)) :]
    stem = stem.strip(" _")
    if "…" in stem:
        # The source name was already abbreviated; keep the head, mark the rest.
        head = stem.split("…", 1)[0].rstrip(" _、，,")
        if head:
            stem = f"{head}等"
    if len(stem) > _MAX_TITLE_CHARS:
        head = stem[:_MAX_TITLE_CHARS]
        cut = max(head.rfind("_"), head.rfind("、"), head.rfind("，"), head.rfind(" "))
        if cut >= 8:
            head = head[:cut]
        stem = head.rstrip(" _、，,") + "…"
    return stem or name


_VERSION_TAG = re.compile(r"_v(\d+)_")


def source_tag(name: str) -> str:
    """A short disambiguator for two notes that share a title.

    002_课程PPT要点_v1_备份__01_....md and 030_课程PPT要点_v2_备份__01_....
    are different revisions of one deck; their titles collide, so the plan
    says （v1）/（v2） rather than （2）/（3）.
    """
    stem = _ORDINAL.sub("", _EXTENSION.sub("", name))
    prefix = stem.split("__", 1)[0] if "__" in stem else ""
    version = _VERSION_TAG.search(prefix)
    if version:
        return f"v{version.group(1)}"
    return ""


def _clean_heading(text: str) -> str:
    title = text.strip().lstrip("#").strip()
    title = re.sub(r"\s+", " ", title)
    if len(title) > 12:
        title = title[:11] + "…"
    return title


def _heading_base(title: str) -> str:
    """Drop a parenthetical aside, the part that usually carries page refs."""
    return title.split("（", 1)[0].strip(" 、，,。:：-")


def md_sections(text: str) -> list[tuple[str, int]]:
    """Level-two headings with the characters under each (## only).

    Course notes put the teaching structure at level two (## 三、核心推导);
    level three carries sub-points, which would shatter a day into fragments.
    """
    marks: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.startswith("## ") and not _PAGE_HEADING.match(line[3:].strip()):
            marks.append((offset, _clean_heading(line)))
        offset += len(line)
    if not marks:
        return []
    bases = [_heading_base(title) for _, title in marks]
    counts: dict[str, int] = {}
    for base in bases:
        counts[base] = counts.get(base, 0) + 1
    sections: list[tuple[str, int]] = []
    for index, (start, title) in enumerate(marks):
        end = marks[index + 1][0] if index + 1 < len(marks) else len(text)
        body = re.sub(r"\s", "", text[start:end])
        # Unique asides are noise; when two sections share a base, keep it whole.
        label = bases[index] if bases[index] and counts[bases[index]] == 1 else title
        if body:
            sections.append((label, len(body)))
    return sections


def _doc_text(path: Path) -> tuple[int, list[tuple[str, int]]]:
    if path.suffix.lower() in TEXT_SUFFIXES:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return path.stat().st_size // 2, []
        return len(text), md_sections(text)
    # A scan or a PDF has no text layer here; half the bytes is the old estimate.
    return path.stat().st_size // 2, []


def _split_by_sections(
    title: str, minutes: int, sections: list[tuple[str, int]], target: int
) -> list[Part] | None:
    total = sum(chars for _, chars in sections)
    if not sections or total <= 0:
        return None
    ranges: list[list[int]] = []
    start = 0
    budget = 0.0
    for index, (_name, chars) in enumerate(sections):
        budget += chars / total * minutes
        if budget >= target or index == len(sections) - 1:
            ranges.append([start, index, max(1, round(budget))])
            start = index + 1
            budget = 0.0
    # A 1-2 minute tail is noise on the board; fold it into its neighbour.
    merged: list[list[int]] = []
    for item in ranges:
        if merged and item[2] < _MIN_PART_MINUTES:
            merged[-1][1] = item[1]
            merged[-1][2] += item[2]
        else:
            merged.append(item)
    if len(merged) > 1 and merged[0][2] < _MIN_PART_MINUTES:
        merged[1][0] = merged[0][0]
        merged[1][2] += merged[0][2]
        merged.pop(0)
    parts: list[Part] = []
    for first, last, minutes_here in merged:
        if first == last:
            label = f"{title} {sections[first][0]}"
        else:
            label = f"{title} §{sections[first][0]}–§{sections[last][0]}"
        parts.append(Part(label=label, minutes=minutes_here))
    return parts or None


def attach_parts(
    subject: str, work: list[ChapterWork], minutes_per_day: int
) -> list[ChapterWork]:
    """Give every chapter the parts its own notes break into."""
    out: list[ChapterWork] = []
    for item in work:
        try:
            parts = chapter_parts(subject, item.name, item.minutes, minutes_per_day)
        except Exception:  # noqa: BLE001 - a missing KB must not block planning
            parts = []
        out.append(replace(item, parts=tuple(parts)))
    return out


def chapter_parts(
    kb_name: str, chapter: str, minutes: int, per_day: int
) -> list[Part]:
    """Name the reading for a chapter as parts a learner can finish in a day.

    Minutes come from the chapter estimate the planner already uses; they are
    shared out by each document's share of the chapter characters, so the
    plan total does not change - only its vocabulary.
    """
    from deeptutor.review.generate import chapter_sources

    try:
        names = chapter_sources(kb_name, chapter, limit=500)
    except Exception:  # noqa: BLE001 - a missing KB must not block the plan
        return []
    if not names:
        return []
    root = _raw_root(kb_name)
    docs: list[tuple[str, int, list[tuple[str, int]]]] = []
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        chars, sections = _doc_text(path)
        if chars > 0:
            docs.append((name, chars, sections))
    total = sum(chars for _, chars, _ in docs)
    if total <= 0:
        return []

    titles = {name: short_title(name, chapter) for name, _, _ in docs}
    clashes = {
        title
        for title in titles.values()
        if sum(1 for other in titles.values() if other == title) > 1
    }
    bases: dict[str, str] = {}
    for name, title in titles.items():
        tag = source_tag(name) if title in clashes else ""
        bases[name] = f"{title}（{tag}）" if tag else title

    parts: list[Part] = []
    taken = 0
    for position, (name, chars, sections) in enumerate(docs):
        share = chars * minutes / total
        doc_minutes = max(_MIN_PART_MINUTES, int(round(share)))
        if position == len(docs) - 1:
            doc_minutes = max(_MIN_PART_MINUTES, minutes - taken)
        taken += doc_minutes
        title = bases[name]
        if doc_minutes <= max(per_day, _MIN_PART_MINUTES):
            parts.append(Part(label=title, minutes=doc_minutes, source=name))
            continue
        # Chunk count, then the size of one chunk in minutes - not the count in
        # minutes, which used to slice a 46-minute note every two minutes.
        chunks = max(1, math.ceil(doc_minutes / per_day))
        target = max(1, math.ceil(doc_minutes / chunks))
        split = _split_by_sections(title, doc_minutes, sections, target)
        if split:
            parts.extend(
                Part(label=p.label, minutes=p.minutes, source=name) for p in split
            )
            continue
        # No level-two headings: keep the document whole and let the planner
        # split it across days as （上）/（下）. Pre-cutting it here turned into
        # labels like （2/2）（下）.
        parts.append(Part(label=title, minutes=doc_minutes, source=name))
    seen: dict[str, int] = {}
    unique: list[Part] = []
    for part in parts:
        if part.minutes <= 0:
            continue
        count = seen.get(part.label, 0) + 1
        seen[part.label] = count
        if count > 1:
            unique.append(replace(part, label=f"{part.label}（{count}）"))
        else:
            unique.append(part)
    return unique


#: Local map from a plan chapter to the living-book chapter that covers it.
BOOK_MAP_FILENAME = "book-map.json"


def load_book_map() -> dict[str, list[tuple[str, str]]]:
    """Chapter -> [(book chapter label, anchor)] from the review home.

    Plans are built from knowledge-base documents, while the learner reads the
    living book, whose chapters are numbered and named differently (a note
    called ``第2章_作业`` is read in the book's ``2 · 传输线基本理论``). The map
    is deployment data (``<review home>/book-map.json``), so the published
    package stays free of any one course outline; with no map the day note
    simply gains no pointer line.
    """
    from deeptutor.review.storage import review_home

    try:
        payload = json.loads(
            (review_home() / BOOK_MAP_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    mapping: dict[str, list[tuple[str, str]]] = {}
    for chapter, value in payload.items():
        entries: list[tuple[str, str]] = []
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str):
                label, _, anchor = item.partition("#")
            elif isinstance(item, dict):
                label = str(item.get("label", ""))
                anchor = str(item.get("anchor", ""))
            else:
                continue
            if label.strip():
                entries.append((label.strip(), anchor.strip()))
        if entries:
            mapping[str(chapter)] = entries
    return mapping


def book_pointer(
    chapter: str, mapping: dict[str, list[tuple[str, str]]] | None = None
) -> str:
    """``活书：第 2 章 · 选频网络（#ch2）`` for a plan chapter, else ``""``."""
    table = load_book_map() if mapping is None else mapping
    entries = table.get(str(chapter or "").strip())
    if not entries:
        return ""
    labels = [f"{label}（#{anchor}）" if anchor else label for label, anchor in entries]
    return "活书：" + "、".join(labels)


#: Local map from a planned note to the living-book section that covers it.
BOOK_SECTIONS_FILENAME = "book-sections.json"


def load_book_sections() -> dict[str, dict[str, dict[str, str]]]:
    """Chapter -> note label -> {no, title, anchor} from the review home.

    A plan is built from knowledge-base notes, and one note is rarely one book
    section: the note MC1-2-1 is read in section 1.7. The map is deployment
    data (<review home>/book-sections.json) produced by scanning the book, so
    the published package still carries no course outline; without it the day
    note falls back to naming the notes themselves.
    """
    from deeptutor.review.storage import review_home

    try:
        payload = json.loads(
            (review_home() / BOOK_SECTIONS_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}
    chapters = payload.get("chapters") if isinstance(payload, dict) else None
    if not isinstance(chapters, dict):
        return {}
    table: dict[str, dict[str, list[dict[str, str]]]] = {}
    for chapter, docs in chapters.items():
        if not isinstance(docs, dict):
            continue
        entries: dict[str, list[dict[str, str]]] = {}
        for doc, info in docs.items():
            # One note may be read across several book sections (a lecture note
            # covers 5–10 of them), so a value is a list; older maps stored one
            # section object and are still accepted.
            raw = info if isinstance(info, list) else [info]
            sections: list[dict[str, str]] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                number = str(item.get("no") or "").strip()
                title = str(item.get("title") or "").strip()
                if number and title:
                    sections.append(
                        {
                            "no": number,
                            "title": title,
                            "anchor": str(item.get("anchor") or "").strip(),
                        }
                    )
            if sections:
                entries[str(doc)] = sections
        if entries:
            table[str(chapter)] = entries
    return table


def book_sections(
    chapter: str, doc: str, mapping: dict[str, dict[str, list[dict[str, str]]]] | None = None
) -> list[dict[str, str]]:
    """The book sections a note is read in (empty when unmapped)."""
    table = load_book_sections() if mapping is None else mapping
    return list(table.get(str(chapter or "").strip(), {}).get(str(doc or "").strip()) or [])


def book_section(
    chapter: str, doc: str, mapping: dict[str, dict[str, list[dict[str, str]]]] | None = None
) -> dict[str, str] | None:
    """The first book section a note is read in, or None when unmapped."""
    entries = book_sections(chapter, doc, mapping)
    return entries[0] if entries else None


def section_lines(
    chapter: str, doc: str, mapping: dict[str, dict[str, list[dict[str, str]]]] | None = None
) -> list[str]:
    """Read as: ['1.7 离散时间系统：线性与移不变', ...] (empty when unmapped)."""
    return [
        f"{entry['no']} {_display_title(entry['title'])}"
        for entry in book_sections(chapter, doc, mapping)
    ]


def _display_title(title: str) -> str:
    """A book section title without the markdown marks its source carries."""
    return re.sub("[" + chr(96) + "*]+", "", str(title or "")).strip()


def section_line(
    chapter: str, doc: str, mapping: dict[str, dict[str, list[dict[str, str]]]] | None = None
) -> str:
    """Read as: 1.7 离散时间系统：线性与移不变 (empty when unmapped)."""
    lines = section_lines(chapter, doc, mapping)
    return lines[0] if lines else ""


#: Local index of the diagrams each living book prints, section by section.
BOOK_FIGURES_FILENAME = "book-figures.json"


def load_book_figures() -> dict[str, Any]:
    """The diagram index (<review home>/book-figures.json), empty when absent.

    A book draws its figures as inline SVG inside the section that needs them,
    so neither a plan nor a card can see them. The index is deployment data
    produced by scripts/build-book-figures.py; the published package ships
    without it and every caller must keep working when it is missing.
    """
    from deeptutor.review.storage import review_home

    try:
        payload = json.loads(
            (review_home() / BOOK_FIGURES_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def book_figures(
    subject: str = "", index: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """Section number -> {title, anchor, figures} for one subject's book.

    Without a subject every book is merged, which is only useful for a count:
    section 2.1 means something different in each book.
    """
    table = load_book_figures() if index is None else index
    books = table.get("books") if isinstance(table, dict) else None
    if not isinstance(books, dict):
        return {}
    if subject:
        found = books.get(str(subject or "").strip())
        return found if isinstance(found, dict) else {}
    merged: dict[str, dict[str, Any]] = {}
    for title, found in books.items():
        if not isinstance(found, dict):
            continue
        for number, entry in found.items():
            merged.setdefault(str(number), dict(entry, subject=str(title)))
    return merged


def chapter_figures(
    chapter: str, index: dict[str, Any] | None = None
) -> dict[str, list[dict[str, str]]]:
    """Section number -> figures for one planned chapter (empty when unmapped)."""
    table = load_book_figures() if index is None else index
    chapters = table.get("chapters") if isinstance(table, dict) else None
    if not isinstance(chapters, dict):
        return {}
    rows = chapters.get(str(chapter or "").strip())
    return rows if isinstance(rows, dict) else {}


def _raw_root(kb_name: str) -> Path:
    from deeptutor.review.generate import _knowledge_bases_root

    return _knowledge_bases_root() / kb_name / "raw"
