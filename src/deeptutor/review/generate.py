"""Knowledge-base -> ReviewSpec generation bridge (Phase 31 WS-1).

The existing question pipeline already produces a quiz for a topic from a
knowledge base. This module turns that quiz into the strict ReviewSpec model
instead of inventing a second generation path:

* :func:`infer_chapters` reads the KB's ``raw/`` filenames and groups them by
  ``第N章_*`` / ``实验N_*`` markers, so a caller can offer a chapter list.
* :func:`generate_spec_from_kb` calls ``AgentCoordinator.generate_from_topic``
  for one chapter and converts each ``qa_pair`` into a review item.

Conversion never calls the QA gates and never writes to storage. The tool layer
owns ``ReviewService.put_spec`` (which runs the gates); this module only has to
produce a spec that can pass them. A single malformed question is counted in
``skipped`` instead of failing the whole pack, but an LLM/transport failure is
allowed to propagate so the caller can distinguish "model failed" from "model
returned nothing usable".
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from deeptutor.review.balance import balance_items, pack_offset
from deeptutor.review.book_import import _explanation_with_answer
from deeptutor.review.concept_notes import compare_markdown, enrich_concepts
from deeptutor.review.qa import _MATH_PATTERN, parameter_infeasibility
from deeptutor.review.symbols import (
    MAX_SYMBOL_LINES,
    SYMBOL_HEADING,
    normalize_symbol_callout,
)
from deeptutor.review.models import (
    Answer,
    CalloutBlock,
    Concept,
    Item,
    ItemBlock,
    Option,
    ProseBlock,
    ReviewSpec,
    Section,
    SourceRef,
    concept_key,
    normalize_text,
)

__all__ = [
    "Chapter",
    "GenerationResult",
    "generate_spec_from_kb",
    "infer_chapters",
]

#: The answer panel's default self-check when the optional notes call did not
#: return a subject-specific one. It is deliberately quantitative-neutral so it
#: works for engineering and non-engineering packs alike.
DEFAULT_SELF_CHECK = (
    "对照题干条件与概念边界，确认结论没有超出适用范围；有数值时再核对单位与量纲。"
)


def _symbols_callout(
    concepts: list[Concept], extras: dict[str, dict[str, Any]]
) -> str:
    """A safe, pre-answer section note: symbols, meanings and units only."""
    parts: list[str] = []
    seen: set[str] = set()

    def add(entry: str) -> None:
        entry = entry.strip().strip("；;")
        if entry and entry not in seen:
            seen.add(entry)
            parts.append(entry)

    for concept in concepts:
        if len(parts) >= MAX_SYMBOL_LINES:
            break
        symbols = str((extras.get(concept.id) or {}).get("symbols") or "").strip()
        if symbols:
            # The note may repeat the same f_s/f_0/f_I across concepts; dedupe
            # individual symbol entries, not whole per-concept blocks.
            for entry in re.split(r"[；;\n]", symbols):
                add(entry)
                if len(parts) >= MAX_SYMBOL_LINES:
                    break
            continue
        # Manual/legacy specs may carry formula metadata but no LLM notes.
        for formula in concept.formulas:
            for symbol, meaning in (formula.variables or {}).items():
                add(f"{symbol} — {meaning}")
                if len(parts) >= MAX_SYMBOL_LINES:
                    break
    if not parts:
        return ""
    # The notes model sometimes answers with a paragraph; the normaliser keeps
    # only the definitions so the callout is a list rather than a wall of text.
    return normalize_symbol_callout(f"{SYMBOL_HEADING}：" + "；".join(parts))


# Pipeline question types. Review kinds use longer names, so accept both.
_PIPELINE_QUESTION_TYPES = ("choice", "written")
_QUESTION_TYPE_ALIASES = {
    "single_choice": "choice",
    "short_answer": "written",
}

_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
_MAX_WARNINGS = 20

_UNSAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")
# Chapters use Arabic numerals (``第1章_绪论``). Experiments appear as both
# ``实验一`` and ``实验1`` in real KBs, so both numeral forms are accepted.
_CHAPTER_RE = re.compile(
    r"第\s*(?P<number>\d+|[零一二三四五六七八九十百千万]+)\s*章[_\-\s]*(?P<title>[^_]+)"
)
_EXPERIMENT_RE = re.compile(
    r"实验\s*(?P<number>[0-9]+|[一二三四五六七八九十百千零〇两]+)"
    r"[_\-\s]*(?P<title>[^_]+)"
)
# Some courses never use ``第N章``: 微机 is filed as ``第N讲``, and 信号 as
# ``chN`` courseware. Without these markers the whole KB collapses into 其他
# and review_generate has nothing selectable.
_LECTURE_RE = re.compile(
    r"第\s*(?P<number>\d+|[零一二三四五六七八九十百千万]+)\s*"
    r"(?:讲|课|次课)[_\-\s]*(?P<title>[^_]*)"
)
# ``课件__ch1-1.1`` / ``CH3-2_周期信号的频谱分析``; the lookbehind keeps
# ordinary words containing ``ch`` out of the match.
_COURSEWARE_RE = re.compile(r"(?i)(?<![a-z])ch\s*(?P<number>\d+)")
# Titles that describe the document type rather than its topic. They would
# turn ``第1讲_课程重点`` into a meaningless chapter name.
_GENERIC_TITLES = frozenset(
    {"课程重点", "课堂重点", "复核报告", "核对报告", "课件", "讲义", "README"}
)
# DSP exports prefix the real title with an internal media code
# (``MC1-1-1离散时间信号的由来``); the code is not part of the topic.
_INTERNAL_TITLE_CODE = re.compile(r"(?i)^MC\d+(?:-\d+){1,3}")
# ``第4章_第四章``: the "title" only restates the chapter number.
_CHAPTER_NUMERAL_RE = re.compile(
    r"^第\s*(?:\d+|[零一二三四五六七八九十百千万]+)\s*章$"
)
# Ingest prefixes every raw document with an ordinal (``014_...``). Stripping
# it is what makes the chapter marker a true prefix for the lecture notes.
_INGEST_PREFIX = re.compile(r"^\d+_")
_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}

OTHER_CHAPTER = "其他"

# The question pipeline emits ``true``/``false`` for 判断题 written answers.
# A Chinese pack must not show a bare English boolean as the answer, and the
# appended "（正确答案：…）" suffix has to carry the same word the learner sees.
_BOOLEAN_ANSWERS = {"true": "正确", "false": "错误"}


@dataclass(frozen=True)
class Chapter:
    """One inferred chapter and how many raw documents it owns."""

    name: str
    doc_count: int


@dataclass(frozen=True)
class GenerationResult:
    """Outcome of one chapter generation.

    ``qa_count`` counts items that made it into ``spec``; ``skipped`` counts
    raw qa_pairs that could not be converted. ``spec`` is ``None`` only when
    the model returned no usable question at all.
    """

    spec: ReviewSpec | None
    chapter: str
    kb_name: str
    qa_count: int
    skipped: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Group:
    kind: str  # "chapter" | "experiment" | "lecture" | "courseware"
    number: int
    number_text: str
    title: str = ""
    doc_count: int = 0

    @property
    def name(self) -> str:
        # Chapters and lectures are canonicalised to Arabic numerals (the
        # plan's example is ``第1章_绪论``), while experiments keep the source's
        # numeral form so ``实验一`` is not renamed to ``实验1``.
        if self.kind == "chapter":
            stem = f"第{self.number}章"
        elif self.kind == "lecture":
            stem = f"第{self.number}讲"
        elif self.kind == "courseware":
            stem = f"ch{self.number}"
        else:
            stem = f"实验{self.number_text}"
        return f"{stem}_{self.title}" if self.title else stem


@dataclass(frozen=True)
class _ParsedQa:
    """A qa_pair reduced to the values the ReviewSpec needs."""

    stem: str
    kind: str  # "single_choice" | "short_answer"
    options: list[Option]
    answer: Answer
    explanation: str
    difficulty: str
    concept_name: str


def _knowledge_bases_root() -> Path:
    """Resolve the KB root exactly the way the core runtime does.

    Imported lazily so ``infer_chapters`` stays cheap and tests can replace
    this one seam without booting the runtime's path service.
    """

    from deeptutor.services.path_service import get_path_service

    return get_path_service().get_knowledge_bases_root()


def _validated_kb_name(kb_name: str) -> str:
    name = str(kb_name or "").strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"invalid knowledge base name: {kb_name!r}")
    if Path(name).name != name:
        raise ValueError(f"invalid knowledge base name: {kb_name!r}")
    return name


_STALE_INDEX_WARNING = (
    "知识库索引可能已过期：raw 中最新的文件改动时间 {newest} 晚于上次索引时间 "
    "{indexed}；检索仍可能返回旧内容，建议重建索引后再生成。"
)


def index_staleness_warning(kb_name: str) -> str | None:
    """Warn when ``raw/`` changed after the KB's last successful index build.

    The core only raises ``needs_reindex`` on an *embedding-signature* mismatch;
    editing a ``raw/`` file never marks the KB. Because generation retrieves
    through the RAG index, a corrected source keeps coming back as the old text
    until somebody rebuilds — and incremental re-adds do not help (they append
    new chunks without deleting the old ones for that file).

    Returns ``None`` when the KB has no ``metadata.json``, no
    ``last_indexed_at``, or nothing to compare, so a KB that was never indexed
    through the tracked path is not reported as stale.
    """

    name = _validated_kb_name(kb_name)
    try:
        kb_dir = _knowledge_bases_root() / name
        raw = kb_dir / "raw"
        metadata_file = kb_dir / "metadata.json"
        if not raw.is_dir() or not metadata_file.is_file():
            return None
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        indexed_at = str(metadata.get("last_indexed_at") or "").strip()
        if not indexed_at:
            return None
        indexed_dt = datetime.fromisoformat(indexed_at)
        newest = max(
            (entry.stat().st_mtime for entry in raw.iterdir() if entry.is_file()),
            default=0.0,
        )
    except Exception:
        # Advisory only: an unresolvable workspace or an unreadable metadata
        # file is not evidence of staleness, and this warning must never turn
        # a working generation into a failure.
        return None
    if not newest:
        return None
    newest_dt = datetime.fromtimestamp(newest)
    if newest_dt <= indexed_dt:
        return None
    return _STALE_INDEX_WARNING.format(
        newest=newest_dt.strftime("%Y-%m-%d %H:%M"),
        indexed=indexed_dt.strftime("%Y-%m-%d %H:%M"),
    )


def _chinese_number(text: str) -> int | None:
    """Parse the numeral forms used in KB filenames (一..九百九十九)."""

    if text.isdigit():
        return int(text)
    if not text or any(char not in _CN_DIGITS and char not in _CN_UNITS for char in text):
        return None
    total = 0
    current = 0
    for char in text:
        if char in _CN_DIGITS:
            current = _CN_DIGITS[char]
        else:
            unit = _CN_UNITS[char]
            total += (current or 1) * unit
            current = 0
    return total + current


def _clean_group_title(title: str) -> str:
    """Drop document-type titles and internal codes from a group name."""
    title = _INTERNAL_TITLE_CODE.sub("", title.strip("_- \t")).strip("_- \t")
    if not title or title in _GENERIC_TITLES:
        return ""
    # ``第1章_第1讲`` names the lecture, not the chapter's topic;
    # ``第4章_第四章`` only restates the chapter number.
    if _LECTURE_RE.fullmatch(title) or _CHAPTER_NUMERAL_RE.fullmatch(title):
        return ""
    return title


def _courseware_title(stem: str, match: re.Match[str]) -> str:
    """The first CJK topic after ``chN``; section numbers are not topics."""
    for part in stem[match.end() :].split("_"):
        part = part.strip("_- \t")
        if part and any("一" <= char <= "鿿" for char in part):
            return _clean_group_title(part)
    return ""


def _courseware_marker(stem: str) -> tuple[str, int, str, str, bool] | None:
    match = _COURSEWARE_RE.search(stem)
    if match is None:
        return None
    number = int(match.group("number"))
    return "courseware", number, str(number), _courseware_title(stem, match), True


def _chapter_marker(stem: str) -> tuple[str, int, str, str, bool] | None:
    """Return ``(kind, number, number_text, title, primary)`` for the marker.

    Chapters and experiments keep their existing precedence; a file with no
    chapter marker falls back to a lecture (``第N讲``) and then to courseware
    (``chN``). ``primary`` marks a chapter marker that sits at the start of
    the stem once the ingest ordinal is stripped (the lecture-note files). A
    later marker in the same file (``学习资料__第1章_绪论``) is secondary.
    """

    match = _CHAPTER_RE.search(stem)
    kind = "chapter"
    if match is None:
        match = _EXPERIMENT_RE.search(stem)
        kind = "experiment"
    if match is None:
        match = _LECTURE_RE.search(stem)
        kind = "lecture"
    if match is None:
        return _courseware_marker(stem)
    number_text = match.group("number")
    number = _chinese_number(number_text)
    if number is None:
        return None
    title = _clean_group_title(match.group("title"))
    if kind != "chapter":
        return kind, number, number_text, title, True
    primary = _CHAPTER_RE.match(_INGEST_PREFIX.sub("", stem, count=1)) is not None
    return kind, number, number_text, title, primary


def chapter_workloads(kb_name: str) -> dict[str, int]:
    """Approximate reading effort per inferred chapter, in characters.

    A planner needs *some* size signal before the chapter has a review pack.
    The only honest one available offline is the raw material itself: text
    files count their characters, binary files (PDF/scans) count half their
    byte size. It is deliberately a rough estimate; the tool exposes it so the
    learner can override the daily budget.
    """

    name = _validated_kb_name(kb_name)
    raw = _knowledge_bases_root() / name / "raw"
    if not raw.is_dir():
        raise FileNotFoundError(f"knowledge base not found: {name}")
    totals: dict[tuple[str, int], int] = {}
    for path in raw.iterdir():
        if not path.is_file():
            continue
        marker = _chapter_marker(path.stem)
        if marker is None:
            continue
        kind, number, _number_text, _title, _primary = marker
        if path.suffix.lower() in {".md", ".txt", ".markdown"}:
            try:
                size = len(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                size = path.stat().st_size // 2
        else:
            size = path.stat().st_size // 2
        key = (kind, number)
        totals[key] = totals.get(key, 0) + size

    result: dict[str, int] = {}
    for chapter in infer_chapters(kb_name):
        marker = _chapter_marker(chapter.name)
        if marker is None:
            continue
        kind, number, _number_text, _title, _primary = marker
        result[chapter.name] = totals.get((kind, number), 0)
    return result


def chapter_sources(kb_name: str, chapter: str, *, limit: int = 6) -> list[str]:
    """Raw file names that belong to one inferred chapter.

    A study task that says "read the chapter" is not actionable when ``raw/``
    holds a dozen files; the planner lists the first few so the learner knows
    what to open. Ordering is by file name, which follows the ingest order.
    """

    name = _validated_kb_name(kb_name)
    raw = _knowledge_bases_root() / name / "raw"
    if not raw.is_dir():
        raise FileNotFoundError(f"knowledge base not found: {name}")
    marker = _chapter_marker(chapter)
    if marker is None:
        return []
    kind, number = marker[0], marker[1]
    names: list[str] = []
    for path in sorted(raw.iterdir(), key=lambda entry: entry.name):
        if not path.is_file():
            continue
        found = _chapter_marker(path.stem)
        if found is not None and found[0] == kind and found[1] == number:
            names.append(path.name)
    return names[:limit]


def infer_chapters(kb_name: str) -> list[Chapter]:
    """Group ``<kb>/raw`` document names by their chapter marker.

    The extension is stripped before matching, so ``第1章_绪论.md`` and
    ``第1章_绪论`` land in one group. Every file type counts — a scanned
    answer sheet PDF belongs to its experiment as much as the markdown notes.

    For chapters, the lecture notes win: when a chapter number already has
    primary files (marker directly after the ingest ordinal), secondary files
    that restate the same chapter (``学习资料__第N章_...``) are dropped instead
    of double-counting it. Secondary markers are still used when they are the
    only source of a chapter.

    Not every course uses chapters. Files without a chapter marker fall back
    to a lecture marker (``第N讲``) and then to courseware (``chN``), so 微机
    and 信号 expose selectable groups instead of one giant ``其他`` bucket.
    A chapter marker still wins when both are present, which keeps the 高频 /
    电磁场 grouping unchanged. Unmatched documents are counted under ``其他``.
    """

    name = _validated_kb_name(kb_name)
    raw = _knowledge_bases_root() / name / "raw"
    if not raw.is_dir():
        raise FileNotFoundError(f"knowledge base not found: {name}")

    chapter_primary: dict[int, list[str]] = {}
    chapter_secondary: dict[int, list[str]] = {}
    experiments: dict[int, _Group] = {}
    lectures: dict[int, _Group] = {}
    courseware: dict[int, _Group] = {}
    other = 0
    groups_by_kind = {
        "experiment": experiments,
        "lecture": lectures,
        "courseware": courseware,
    }

    for path in sorted(raw.iterdir(), key=lambda entry: entry.name):
        if not path.is_file():
            continue
        marker = _chapter_marker(path.stem)
        if marker is None:
            other += 1
            continue
        kind, number, number_text, title, primary = marker
        if kind != "chapter":
            # Experiments, lectures and courseware are one-document-per-group
            # series: every file belongs to its marker and no primary/secondary
            # distinction is needed.
            bucket_by_kind = groups_by_kind[kind]
            group = bucket_by_kind.get(number)
            if group is None:
                bucket_by_kind[number] = _Group(
                    kind=kind,
                    number=number,
                    number_text=number_text,
                    title=title,
                    doc_count=1,
                )
            else:
                group.doc_count += 1
                if not group.title and title:
                    group.title = title
            continue
        bucket = chapter_primary if primary else chapter_secondary
        bucket.setdefault(number, []).append(title)

    groups: list[_Group] = []
    for number in sorted(set(chapter_primary) | set(chapter_secondary)):
        if chapter_primary.get(number):
            entries = chapter_primary[number]
        else:
            entries = chapter_secondary.get(number, [])
        groups.append(
            _Group(
                kind="chapter",
                number=number,
                number_text=str(number),
                # A chapter may be filed with one lecture-only title and one
                # real topic (``第1章_第1讲`` + ``第1章_MC1-1-1离散时间…``);
                # prefer the first title that actually names the topic.
                title=next((entry for entry in entries if entry), ""),
                doc_count=len(entries),
            )
        )
    groups.extend(sorted(experiments.values(), key=lambda item: item.number))
    groups.extend(sorted(lectures.values(), key=lambda item: item.number))
    groups.extend(sorted(courseware.values(), key=lambda item: item.number))
    chapters = [Chapter(name=group.name, doc_count=group.doc_count) for group in groups]
    if other:
        chapters.append(Chapter(name=OTHER_CHAPTER, doc_count=other))
    return chapters


def _slug(text: object, max_length: int = 60) -> str:
    """ASCII id fragment; empty when the text has no ASCII characters."""

    cleaned = _UNSAFE_ID.sub("-", str(text or "")).strip("-._")
    return cleaned[:max_length].strip("-._")


def _digest(text: object, length: int = 10) -> str:
    return hashlib.sha1(normalize_text(str(text or "")).encode("utf-8")).hexdigest()[:length]


def legacy_default_spec_id(kb_name: str, chapter: str) -> str:
    """The spec id used before Phase 35 (no collision digest).

    Kept so a regeneration can reuse the id that already holds the learner's
    events and sessions instead of creating a duplicate pack.
    """
    kb_part = _slug(kb_name, 32) or _digest(kb_name)
    chapter_part = _slug(chapter, 60) or _digest(chapter)
    return f"kb-{kb_part}-{chapter_part}"


def _default_spec_id(kb_name: str, chapter: str) -> str:
    kb_part = _slug(kb_name, 32) or _digest(kb_name)
    chapter_part = _slug(chapter, 60) or _digest(chapter)
    # ASCII slugging can map different names onto the same fragment (spaces,
    # punctuation, case). The digest of the exact pair keeps those packs
    # distinct while remaining stable for an exact regeneration of the same
    # chapter.
    identity = hashlib.sha1(
        f"{kb_name}\x00{chapter}".encode("utf-8")
    ).hexdigest()[:6]
    return f"kb-{kb_part}-{chapter_part}-{identity}"


def _concept_id(name: str) -> str:
    """Stable, collision-free concept id.

    A plain slug is not enough: different names can slug to the same text and
    a pure-CJK name slugs to nothing. The short digest keeps identity stable
    across regenerations without depending on declaration order.
    """

    slug = _slug(name, 40)
    digest = _digest(name, 8)
    return f"c-{slug}-{digest}" if slug else f"c-{digest}"


def _label_sort_key(label: str) -> tuple[tuple[int, Any], ...]:
    """Natural order for option labels (``A``, ``B`` / ``1``, ``2``, ``10``)."""

    parts = [part for part in re.split(r"(\d+)", str(label)) if part]
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower()) for part in parts)


_RAW_COMPARISON = re.compile(r"(?<!\\)[<>]")
_RAW_PIPE = re.compile(r"(?<!\\)\|")


def _escape_math(text: str) -> str:
    """Escape raw ``<``, ``>`` and ``|`` inside math segments (G3 requirement).

    Reuses the gate's own segment pattern so the fixer and the check cannot
    drift apart. Text outside math is left alone — prose may legitimately
    contain those characters and the exporter escapes HTML at render time.
    """

    def fix(match: re.Match[str]) -> str:
        segment = match.group(0)
        segment = _RAW_COMPARISON.sub(
            lambda m: r"\lt{}" if m.group(0) == "<" else r"\gt{}", segment
        )
        return _RAW_PIPE.sub(lambda _m: r"\vert{}", segment)

    return _MATH_PATTERN.sub(fix, text)


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
_FENCE_LINE_RE = re.compile(r"^\s*(?:```+|~~~+)")


def _plain_markdown(text: str) -> str:
    """Normalise the Markdown the model emits without destroying structure.

    ``**bold**`` is still flattened, but pipe tables and fenced code blocks
    are kept verbatim: the export page now renders them as a real table and a
    monospace block (微机 instruction tables, DSP pseudo-code). Fenced code
    is also what protects ASCII diagrams from whitespace collapsing.
    """
    lines = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if _FENCE_LINE_RE.match(line):
            in_fence = not in_fence
            lines.append(line)
            continue
        if in_fence:
            lines.append(line)
            continue
        if _TABLE_SEP_RE.match(stripped):
            lines.append(stripped)
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            lines.append(_BOLD_RE.sub(lambda m: m.group(1) or m.group(2), stripped))
            continue
        lines.append(_BOLD_RE.sub(lambda m: m.group(1) or m.group(2), line))
    return "\n".join(lines)


def _transform_outside_fences(text: str, transform: Callable[[str], str]) -> str:
    """Apply *transform* to prose only; fenced code passes through verbatim.

    ``_normalize_identifiers`` would otherwise wrap ``V_CC`` inside an
    assembly listing in ``$…$``, and ``_escape_math`` would rewrite ``<`` and
    ``|`` in code. Code is rendered as plain text, so neither pass applies.
    """
    parts: list[str] = []
    buffer: list[str] = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        if _FENCE_LINE_RE.match(line):
            parts.append("".join(buffer) if in_fence else transform("".join(buffer)))
            buffer = []
            parts.append(line)
            in_fence = not in_fence
            continue
        buffer.append(line)
    parts.append("".join(buffer) if in_fence else transform("".join(buffer)))
    return "".join(parts)


_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9\\./])"
    r"([A-Za-zθαβγδεζηικλμνξοπρστυφχψω]{1,2})"
    # The braced and bare forms are separate alternatives on purpose. With an
    # optional ``\}`` a bare ``f_m`` inside ``\frac{..f_m}{F}`` swallowed the
    # brace that closed \frac's first argument, producing invalid LaTeX.
    r"(?:_\{([A-Za-z0-9,.]{1,10})\}|_([A-Za-z0-9,.]{1,10}))"
    r"(?:\^\{([A-Za-z0-9+\-]{1,6})\}|\^([A-Za-z0-9+\-]{1,6}))?"
)
_GREEK = {
    "α": "\\alpha",
    "β": "\\beta",
    "γ": "\\gamma",
    "δ": "\\delta",
    "ε": "\\varepsilon",
    "ζ": "\\zeta",
    "η": "\\eta",
    "θ": "\\theta",
    "ι": "\\iota",
    "κ": "\\kappa",
    "λ": "\\lambda",
    "μ": "\\mu",
    "ν": "\\nu",
    "ξ": "\\xi",
    "ο": "o",
    "π": "\\pi",
    "ρ": "\\rho",
    "σ": "\\sigma",
    "τ": "\\tau",
    "υ": "\\upsilon",
    "φ": "\\varphi",
    "χ": "\\chi",
    "ψ": "\\psi",
    "ω": "\\omega",
}


_CJK = re.compile(r"[一-鿿]")
# Only commands that cannot occur in prose or a Windows path. ``\Users`` and
# ``\knowledge_bases`` match a generic ``\[A-Za-z]{2,}`` and would otherwise be
# wrapped as math.
_FORMULA_COMMAND = re.compile(
    r"\\(?:frac|dfrac|tfrac|sqrt|sum|int|prod|lim|times|cdot|approx|leq|geq|neq"
    r"|pm|mp|infty|partial|nabla|overline|underline|hat|bar|vec|text|mathrm"
    r"|alpha|beta|gamma|delta|Delta|theta|Theta|lambda|Lambda|mu|sigma|Sigma"
    r"|phi|Phi|psi|Psi|rho|tau|pi|eta|zeta|xi|nu|kappa|chi|omega|Omega)\b"
)


def _looks_like_bare_formula(text: str) -> bool:
    """True when the whole segment is a formula the model forgot to delimit.

    Wrapping identifier by identifier inside ``\\frac{\\Delta f_m}{F}`` would
    nest ``$…$`` inside the command's argument, which MathJax renders as
    broken math. A segment with a LaTeX command, no CJK prose and balanced
    braces is a single formula, so it gets one pair of delimiters.
    """

    stripped = text.strip()
    if not stripped or "$" in stripped:
        return False
    if not _FORMULA_COMMAND.search(stripped) or _CJK.search(stripped):
        return False
    return stripped.count("{") == stripped.count("}")


def _identifier_math(text: str) -> str:
    if _looks_like_bare_formula(text):
        return f"${text.strip()}$"

    def replace(match: re.Match[str]) -> str:
        base = _GREEK.get(match.group(1), match.group(1))
        subscript = match.group(2) or match.group(3)
        superscript = match.group(4) or match.group(5)
        math = f"{base}_{{{subscript}}}"
        if superscript:
            math += f"^{{{superscript}}}"
        return f"${math}$"

    return _IDENTIFIER.sub(replace, text)


def _normalize_identifiers(text: str) -> str:
    """Wrap bare variable notation (``V_CC``, ``I_c1m^2``) in math delimiters.

    The model sometimes writes identifiers without ``$…$``; the export page
    renders text nodes literally, so the learner sees raw underscores. Only
    regions outside existing math are rewritten, and only short (1-2 char)
    bases, which keeps prose and file names untouched.
    """
    parts: list[str] = []
    last = 0
    for match in _MATH_PATTERN.finditer(text):
        parts.append(_identifier_math(text[last : match.start()]))
        parts.append(match.group(0))
        last = match.end()
    parts.append(_identifier_math(text[last:]))
    return "".join(parts)


def _clean_text(text: object) -> str:
    """Apply the full normalization chain to one free-text spec field.

    Order matters: Markdown is normalised first, raw math symbols are escaped
    second (segment-wise), and bare identifiers are wrapped last so the new
    ``$…$`` segments are never re-escaped. Fenced code is excluded from the
    last two passes: it is not math and must keep its literal identifiers and
    comparison characters. Every field that reaches G3 must go through this —
    the gate scans all strings, not just stems.
    """
    plain = _plain_markdown(str(text or "").strip())
    return _transform_outside_fences(
        plain, lambda part: _normalize_identifiers(_escape_math(part))
    )


def _option_items(raw: object) -> list[tuple[str, str]]:
    """Sorted ``(label, text)`` pairs; empty when the options are malformed."""

    if not isinstance(raw, dict):
        return []
    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, text in raw.items():
        clean_label = str(label).strip()
        clean_text = _clean_text(text)
        if not clean_label or not clean_text:
            return []
        key = clean_label.upper()
        if key in seen:
            return []
        seen.add(key)
        items.append((clean_label, clean_text))
    items.sort(key=lambda pair: _label_sort_key(pair[0]))
    return items


def _match_choice(
    items: list[tuple[str, str]], correct_answer: str
) -> tuple[int, str, str] | None:
    """Map a reference answer to ``(index, label, text)`` in sorted options."""

    answer = str(correct_answer or "").strip()
    if not answer:
        return None
    letter = answer.upper().rstrip(".、．)）").strip()
    if len(letter) == 1 and "A" <= letter <= "Z":
        for index, (label, text) in enumerate(items):
            if label.upper() == letter:
                return index, label, text
        return None
    for index, (label, text) in enumerate(items):
        if answer in {label, text, f"{label}. {text}", f"{label}、{text}"}:
            return index, label, text
    return None


def _difficulty(value: object, fallback: str) -> str:
    for candidate in (value, fallback):
        text = str(candidate or "").strip().lower()
        if text in _DIFFICULTIES:
            return text
    return "medium"


def _normalize_question_types(
    question_types: Iterable[str] | None, warnings: list[str]
) -> list[str] | None:
    if question_types is None:
        return None
    picked: list[str] = []
    for entry in question_types:
        name = str(entry or "").strip().lower()
        name = _QUESTION_TYPE_ALIASES.get(name, name)
        if name in _PIPELINE_QUESTION_TYPES and name not in picked:
            picked.append(name)
        elif name:
            _warn(warnings, f"ignored unsupported question type: {entry!r}")
    return picked or None


def _warn(warnings: list[str], message: str) -> None:
    if len(warnings) < _MAX_WARNINGS:
        warnings.append(message)
    elif len(warnings) == _MAX_WARNINGS:
        warnings.append("further warnings omitted")


def _parse_qa(
    qa: dict[str, Any], *, chapter: str, requested_difficulty: str
) -> tuple[_ParsedQa | None, str]:
    """Convert one pipeline qa_pair; return ``(parsed, reason)``."""

    stem = _clean_text(qa.get("question"))
    if not stem:
        return None, "empty question stem"

    correct = _clean_text(qa.get("correct_answer"))
    explanation = _clean_text(qa.get("explanation"))
    # The pipeline marks a failed single question with this literal prefix and
    # "N/A" everywhere else; such a placeholder must never enter a pack.
    if stem.startswith("[Generation failed]") or correct == "N/A":
        return None, "generation failed placeholder"
    question_type = str(qa.get("question_type") or "").strip().lower()
    difficulty = _difficulty(qa.get("difficulty"), requested_difficulty)
    # ``concentration`` becomes the concept name, which G3 scans like any
    # other string: ``$f_s<f_p$`` would reject the whole pack at storage.
    concept_name = _clean_text(qa.get("concentration")) or _clean_text(chapter)

    if question_type != "written":
        raw_options = qa.get("options")
        items = _option_items(raw_options)
        if len(items) >= 2:
            match = _match_choice(items, correct)
            if match is None:
                # Degrading a choice item to short_answer would hand the
                # learner the raw option letter as the answer key and show
                # the question as a fill-in. Skip it instead.
                return None, "choice answer did not match any option"
            index, label, text = match
            return (
                _ParsedQa(
                    stem=stem,
                    kind="single_choice",
                    options=[Option(label=label, text=text) for label, text in items],
                    answer=Answer(choice_index=index),
                    # G4 looks for the option text, not the label; the
                    # fallback appends "B. <text>" when it is absent.
                    explanation=_explanation_with_answer(
                        explanation, f"{label}. {text}", needle=text
                    ),
                    difficulty=difficulty,
                    concept_name=concept_name,
                ),
                "",
            )
        if raw_options:
            return None, "choice options could not be parsed"
        if re.fullmatch(r"[A-Za-z]", correct or ""):
            return None, "choice answer has no parseable options"

    # Written questions become short answers; choice items above never fall
    # through with an option letter as the reference answer.
    if not correct:
        return None, "no usable reference answer"
    correct = _BOOLEAN_ANSWERS.get(correct.strip().lower(), correct)
    return (
        _ParsedQa(
            stem=stem,
            kind="short_answer",
            options=[],
            answer=Answer(text=correct),
            explanation=_explanation_with_answer(explanation, correct),
            difficulty=difficulty,
            concept_name=concept_name,
        ),
        "",
    )


def _run_generate(coordinator: Any, **kwargs: Any) -> dict[str, Any]:
    """Run the pipeline coroutine from this synchronous entry point.

    The tool layer calls this function through ``asyncio.to_thread``, so no
    loop is running in this thread. ``asyncio.run`` also gives the pipeline a
    fresh loop for its StreamBus.
    """

    result = asyncio.run(coordinator.generate_from_topic(**kwargs))
    if not isinstance(result, dict):
        raise TypeError(f"generate_from_topic returned {type(result).__name__}, expected dict")
    return result


def generate_spec_from_kb(
    *,
    kb_name: str,
    chapter: str,
    num_questions: int = 12,
    question_types: list[str] | None = None,
    difficulty: str = "",
    spec_id: str | None = None,
    now: float | None = None,
    llm_complete: Callable[[str, str], str] | None = None,
) -> GenerationResult:
    """Generate one chapter's review pack from a knowledge base.

    ``chapter`` must be one of :func:`infer_chapters`' names; an unknown KB
    raises ``FileNotFoundError`` and an unmatched chapter raises
    ``ValueError`` so the tool layer can ask the learner for a valid name.
    LLM/transport errors propagate unchanged.

    ``now`` is accepted for interface compatibility; ReviewSpec v1 has no
    generation timestamp field, so nothing reads it yet.

    ``llm_complete`` is an optional synchronous ``(prompt, system_prompt) ->
    text`` hook. When given, one extra call fills each concept's one-line
    summary and definition, which ``review_hint`` uses at levels 1 and 2.
    Failures only add warnings; the pack itself never depends on this call.
    Direct callers (tests, scripts) stay offline by default; the job runner
    passes the core LLM client.

    If you store the result yourself (rather than through
    :func:`deeptutor.review.jobs.start_generation_job`), pass
    ``spec_id=legacy_default_spec_id(kb_name, chapter)`` when an older pack
    may already hold that chapter's review history. The job runner performs
    that lookup and identity check automatically; a bare call always mints the
    new-format id.
    """

    name = _validated_kb_name(kb_name)
    chapter = str(chapter or "").strip()
    if not chapter:
        raise ValueError("chapter is required")
    # Resolve to the canonical group name. The tool layer collapses
    # whitespace for display, so a name copied back verbatim from the chapter
    # list can differ from the raw group name (internal double spaces,
    # U+3000). Matching on the collapsed form and continuing with the raw
    # form keeps the listing, this validation and the deterministic spec_id
    # on one spelling.
    known = {entry.name for entry in infer_chapters(name)}
    canonical = {" ".join(entry.split()): entry for entry in known}
    resolved = canonical.get(" ".join(chapter.split()))
    if resolved is None:
        raise ValueError(f"chapter {chapter!r} not found in knowledge base {name!r}")
    chapter = resolved

    warnings: list[str] = []
    staleness = index_staleness_warning(name)
    if staleness:
        _warn(warnings, staleness)
    pipeline_types = _normalize_question_types(question_types, warnings)
    resolved_spec_id = str(spec_id or "").strip() or _default_spec_id(name, chapter)
    count = max(1, int(num_questions or 1))

    output_dir = tempfile.mkdtemp(prefix="deeptutor-review-kb-")
    try:
        from deeptutor.agents.question.coordinator import AgentCoordinator

        coordinator = AgentCoordinator(
            kb_name=name,
            language="zh",
            output_dir=output_dir,
        )
        payload = _run_generate(
            coordinator,
            user_topic=chapter,
            num_questions=count,
            difficulty=str(difficulty or "").strip(),
            question_types=pipeline_types,
        )
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)

    results = payload.get("results") or []
    if not isinstance(results, list):
        results = []
        _warn(warnings, "generator returned a non-list results field")
    if payload.get("success") is False:
        _warn(
            warnings,
            str(payload.get("error") or "generation reported a partial failure"),
        )

    # ``chapter`` itself is raw KB input (a filename fragment) and becomes the
    # section title, the spec title and the fallback concept name. The spec id
    # keeps the raw spelling so regenerating an existing pack is unchanged.
    chapter_text = _clean_text(chapter)

    source_id = f"kb:{name}·{chapter_text}"
    source = SourceRef(
        id=source_id,
        kind="doc",
        title=name,
        kb=name,
        ref=chapter_text,
    )

    concepts: dict[str, Concept] = {}
    concept_order: list[str] = []
    items: list[Item] = []
    skipped = 0

    for index, entry in enumerate(results):
        qa = entry.get("qa_pair") if isinstance(entry, dict) else None
        if not isinstance(qa, dict):
            skipped += 1
            _warn(warnings, f"qa pair {index + 1}: missing qa_pair payload")
            continue
        parsed, reason = _parse_qa(
            qa, chapter=chapter, requested_difficulty=difficulty
        )
        if parsed is None:
            skipped += 1
            _warn(warnings, f"qa pair {index + 1} skipped: {reason}")
            continue

        concept_lookup = normalize_text(parsed.concept_name)
        concept = concepts.get(concept_lookup)
        if concept is None:
            concept = Concept(
                id=_concept_id(parsed.concept_name),
                key=concept_key(parsed.concept_name, "", [name]),
                name=parsed.concept_name,
                difficulty=parsed.difficulty,
                source_ids=[source_id],
            )
            concepts[concept_lookup] = concept
            concept_order.append(concept_lookup)
        # Use the concept object's id, never a fresh id for the same lookup:
        # "RC Circuit" and "rc circuit" normalize to one concept, but a
        # recomputed id would differ and the item would reference a concept
        # that never entered the pack (whole-generation ValidationError).
        concept_id = concept.id

        try:
            item = Item(
                id=f"{resolved_spec_id}-q{index + 1:03d}",
                concept_ids=[concept_id],
                kind=parsed.kind,
                stem=parsed.stem,
                options=parsed.options,
                answer=parsed.answer,
                explanation=parsed.explanation,
                difficulty=parsed.difficulty,
                source_ids=[source_id],
            )
        except Exception as exc:  # noqa: BLE001 - one bad item must not kill the pack
            skipped += 1
            _warn(warnings, f"qa pair {index + 1} skipped: {type(exc).__name__}: {exc}")
            continue
        infeasible = parameter_infeasibility(item)
        if infeasible:
            skipped += 1
            _warn(warnings, f"qa pair {index + 1} skipped: {infeasible}")
            continue
        items.append(item)

    if not items:
        _warn(warnings, "no usable questions were generated")
        return GenerationResult(
            spec=None,
            chapter=chapter,
            kb_name=name,
            qa_count=0,
            skipped=skipped,
            warnings=warnings,
        )

    # The pipeline returns the correct option first almost every time, so a pack
    # can arrive with 10 of 13 answers on "A" — beatable without reading. Rotate
    # the options per pack; the helper also remaps the letters the explanation
    # uses to discuss the distractors ("B 误用线性关系 …").
    items, _ = balance_items(items, offset=pack_offset(resolved_spec_id))

    used_concepts = {concept_id for item in items for concept_id in item.concept_ids}
    concept_list = [concepts[key] for key in concept_order if concepts[key].id in used_concepts]

    extras: dict[str, dict[str, Any]] = {}
    if llm_complete is not None:
        items_by_concept: dict[str, list[Item]] = {}
        for item in items:
            for concept_id in item.concept_ids:
                items_by_concept.setdefault(concept_id, []).append(item)
        enrich_concepts(
            concept_list,
            items_by_concept,
            chapter=chapter_text,
            complete=llm_complete,
            clean=_clean_text,
            warn=lambda message: _warn(warnings, message),
            extras=extras,
        )

    # Structured study aids, not an ever-growing explanation string. They are
    # rendered only inside the answer panel, after the attempt; review_today's
    # whitelist never exposes them.
    for item in items:
        note = extras.get(item.concept_ids[0], {}) if item.concept_ids else {}
        item.self_check = (
            str(note.get("self_check") or "").strip() or DEFAULT_SELF_CHECK
        )
        item.pitfalls = [str(value) for value in (note.get("pitfalls") or [])]
        item.assumptions = [
            str(value) for value in (note.get("assumptions") or [])
        ]
        # A comparison table is attached only when the notes model returned a
        # valid one; an empty value keeps the item's content hash unchanged.
        item.study_table = compare_markdown(note.get("compare"))

    summary = (
        f"本章共 {len(items)} 道复习题，覆盖 {len(concept_list)} 个知识点。"
    )
    blocks: list[Any] = [ProseBlock(text=summary)]
    symbols = _symbols_callout(concept_list, extras)
    if symbols:
        blocks.append(CalloutBlock(tone="info", text=symbols))
    blocks.extend(ItemBlock(item_id=item.id) for item in items)
    section = Section(
        id=f"sec-{resolved_spec_id}",
        title=chapter_text,
        concept_ids=[concept.id for concept in concept_list],
        order=0,
        blocks=blocks,
    )
    spec = ReviewSpec(
        id=resolved_spec_id,
        kb_ids=[name],
        language="zh",
        title=_clean_text(f"{name} · {chapter_text} 复习包"),
        concepts=concept_list,
        sections=[section],
        items=items,
        sources={source_id: source},
    )
    return GenerationResult(
        spec=spec,
        chapter=chapter,
        kb_name=name,
        qa_count=len(items),
        skipped=skipped,
        warnings=warnings,
    )
