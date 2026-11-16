"""Deterministic content-fidelity audit for stored Review Studio packs.

Run with the project virtualenv:

    .venv/Scripts/python.exe scripts/audit_content_fidelity.py

The audit never calls an LLM. It reads the specs already stored in
``<review home>/review.sqlite3`` and applies only checks a machine can prove:

1. structural     -- non-empty stem; choice items have >= 2 options with
                     unique labels; written items have no options; no
                     ``[Generation failed]`` / ``N/A`` placeholders.
2. answer key     -- choice ``answer.choice_index`` points at a real option;
                     written items carry a non-empty reference answer
                     (``answer.text``; numeric items use ``answer.value``).
3. explanation    -- non-empty, and contains the correct option's text or the
                     written answer (same predicate as
                     ``book_import._explanation_with_answer`` / gate G4).
4. math hygiene   -- every ``$...$`` / ``$$...$$`` pair is balanced, and the
                     math segments matched by ``qa._MATH_PATTERN`` contain no
                     raw ``<``, ``>`` or ``|`` (use ``\\lt`` / ``\\gt`` /
                     ``\\vert{}``).
5. markdown leak  -- no *unbalanced* ``*`` markers outside fenced code and math
                     in the stem, options or explanation. Pipe tables and
                     balanced ``*em*``/``**strong**`` are rendered as real
                     blocks since P5 and the emphasis pass; ``__`` has no
                     renderer and is still flagged whenever it appears.
6. grounding      -- coarse character-bigram overlap between stem+explanation
                     and the chapter's raw KB documents, resolved through the
                     same path/marker logic as ``generate.infer_chapters``.
                     A low score is flagged as "weak grounding"; it is lexical
                     evidence only and never proves fabrication.
7. duplicates     -- identical or near-identical normalized stems in one spec.

Only categories 1-4 are blocking. Grounding, markdown leakage and duplicate
flags are reported but do not change the exit code. Exit code is 0 when every
selected spec passes the blocking checks and 1 otherwise.

Paths default to the real review home (``DEEPTUTOR_REVIEW_HOME`` or the core
path service). ``--home`` points the audit at another review home (tests).
The machine-readable report is written to ``dist/content-audit-<timestamp>.json``
unless ``--output`` overrides it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Working-tree imports
# ---------------------------------------------------------------------------
# The product helpers are optional: every one of them has a local fallback so
# the audit still runs if the module layout drifts.

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_IMPORT_NOTES: list[str] = []
_RUNTIME_NOTES: list[str] = []

# ``deeptutor`` is a regular package in site-packages, so a sys.path entry
# alone does not merge the checkout's namespace portion: the installed wheel
# would win and the audit would import stale product helpers. Prepend the
# checkout to the package's own ``__path__``, the seam used by
# ``tests/conftest.py`` and the acceptance scripts.
try:
    import deeptutor as _deeptutor

    _CHECKOUT_PACKAGE = SRC / "deeptutor"
    if str(_CHECKOUT_PACKAGE) not in _deeptutor.__path__:
        _deeptutor.__path__.insert(0, str(_CHECKOUT_PACKAGE))
except Exception as exc:  # pragma: no cover - exercised only on layout drift
    _IMPORT_NOTES.append(
        f"checkout not prepended to deeptutor.__path__ ({type(exc).__name__})"
    )

_QA_MATH_PATTERN = re.compile(
    r"\$\$(?P<double>.+?)\$\$|\$(?P<single>.+?)\$"
    r"|\\\((?P<paren>.+?)\\\)|\\\[(?P<bracket>.+?)\\\]",
    re.DOTALL,
)
_QA_RAW_COMPARISON = re.compile(r"(?<!\\)[<>]")
_QA_RAW_PIPE = re.compile(r"(?<!\\)\|")

_answer_coverage = None
_qa_mentions_number = None
_ANSWER_COVERAGE_MIN = 0.6

try:
    from deeptutor.review.qa import (  # type: ignore[attr-defined]
        _MATH_PATTERN as _QA_MATH_PATTERN,
        _RAW_COMPARISON as _QA_RAW_COMPARISON,
        _RAW_PIPE as _QA_RAW_PIPE,
        _mentions_number as _qa_mentions_number,
        answer_coverage as _answer_coverage,
        ANSWER_COVERAGE_MIN as _ANSWER_COVERAGE_MIN,
    )
except Exception as exc:  # pragma: no cover - exercised only on layout drift
    _IMPORT_NOTES.append(
        f"qa math patterns reimplemented verbatim ({type(exc).__name__})"
    )

_explanation_with_answer = None
try:
    from deeptutor.review.book_import import (  # type: ignore[attr-defined]
        _explanation_with_answer as _explanation_with_answer,
    )
except Exception as exc:  # pragma: no cover - exercised only on layout drift
    _IMPORT_NOTES.append(
        "explanation predicate reimplemented "
        f"(book_import._explanation_with_answer unavailable: {type(exc).__name__})"
    )

_knowledge_bases_root = None
_chapter_marker = None
_validated_kb_name = None
try:
    from deeptutor.review.generate import (  # type: ignore[attr-defined]
        _chapter_marker as _chapter_marker,
        _knowledge_bases_root as _knowledge_bases_root,
        _validated_kb_name as _validated_kb_name,
    )
except Exception as exc:  # pragma: no cover - exercised only on layout drift
    _IMPORT_NOTES.append(
        "KB path/marker helpers reimplemented coarsely "
        f"(generate helpers unavailable: {type(exc).__name__})"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIT_VERSION = "1"

# Bigram overlap below this is reported as "weak grounding". The value is
# deliberately conservative: a correct paraphrase can share few two-character
# sequences with the source, so a flag is a prompt to look, not a verdict.
DEFAULT_GROUNDING_THRESHOLD = 0.35

# Near-duplicate stems are those whose SequenceMatcher ratio reaches this.
# Identical normalized stems are always reported regardless of length.
NEAR_DUPLICATE_RATIO = 0.90
NEAR_DUPLICATE_MIN_CHARS = 6
MAX_DUPLICATE_PAIRS = 50
MAX_DETAIL_LINES = 30

# Bounds keep one huge document from dominating memory/time. They are part of
# the deterministic contract: files are read in sorted order and truncated.
MAX_DOC_CHARS = 300_000
MAX_CHAPTER_CHARS = 1_200_000
TEXT_SUFFIXES = frozenset(
    {".md", ".markdown", ".txt", ".tex", ".csv", ".json", ".html", ".htm", ".rst"}
)

KNOWN_KINDS = frozenset({"single_choice", "numeric", "short_answer", "flashcard"})
WRITTEN_KINDS = frozenset({"numeric", "short_answer", "flashcard"})
OTHER_CHAPTER = "其他"

_BOLD_UNDERSCORE_RE = re.compile(r"__(.+?)__")
#: Inline/display math, removed before looking for emphasis markers: ``a*b``
#: inside a formula is multiplication, not an unpaired delimiter.
_INLINE_MATH_RE = re.compile(
    r"\$\$[\s\S]+?\$\$|\$[^$\n]+?\$|\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\]"
)
_DOLLAR_TOKEN_RE = re.compile(r"(?<!\\)(\${1,2})")
_PAREN_OPEN_RE = re.compile(r"(?<!\\)\\\(")
_PAREN_CLOSE_RE = re.compile(r"(?<!\\)\\\)")
_BRACKET_OPEN_RE = re.compile(r"(?<!\\)\\\[")
_BRACKET_CLOSE_RE = re.compile(r"(?<!\\)\\\]")
#: ``$f_{0}$.7,n`` — the old identifier regex split ``f_{0.7,n}`` in two.
_SPLIT_SUBSCRIPT_RE = re.compile(r"\$[^$\n]{1,60}\$\.[0-9]")
_INGEST_PREFIX = re.compile(r"^\d+_")
_WHITESPACE = re.compile(r"\s+")

BLOCKING_CATEGORIES = ("structural", "answer", "explanation", "math")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _collapse(text: object) -> str:
    return _WHITESPACE.sub(" ", str(text or "")).strip()


def _normalized_stem(text: object) -> str:
    """NFKC + lowercase + collapsed whitespace, trailing sentence punctuation removed."""
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = _WHITESPACE.sub(" ", value).strip()
    return value.rstrip("。．.!！?？;；")


def _bigrams(text: object) -> set[str]:
    """Character bigrams over content characters (no whitespace/punctuation)."""
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    chars = [
        char
        for char in value
        if not char.isspace()
        and not unicodedata.category(char).startswith(("P", "Z"))
    ]
    return {first + second for first, second in zip(chars, chars[1:])}


def _explanation_contains(explanation: str, answer: str, needle: str) -> bool:
    """Verbatim containment, using the importer's own predicate.

    ``_explanation_with_answer`` returns the explanation unchanged when the
    needle is already present; comparing the result to the stripped input
    reuses that predicate without duplicating the containment rule.

    This is only the fast path for the choice branch. G4 accepts a paraphrase
    for written answers (bigram coverage) and reads units for numeric ones, so
    both of those branches fall back to the gate's own helpers below.
    """
    text = explanation.strip()
    if not text or not needle.strip():
        return False
    if _explanation_with_answer is not None:
        return (
            _explanation_with_answer(text, answer, needle=needle) == text
        )
    return needle.strip() in text


# ---------------------------------------------------------------------------
# Math / markdown hygiene
# ---------------------------------------------------------------------------


def _math_issues(text: str) -> list[str]:
    """Return deterministic math-delimiter issues for one text field."""
    issues: list[str] = []

    tokens = [match.group(0) for match in _DOLLAR_TOKEN_RE.finditer(text)]
    if len(tokens) % 2:
        issues.append("unbalanced $ delimiters (odd count)")
    else:
        for index in range(0, len(tokens), 2):
            if tokens[index] != tokens[index + 1]:
                issues.append("mixed $ and $$ delimiters")
                break

    for opening, closing, label in (
        (_PAREN_OPEN_RE, _PAREN_CLOSE_RE, r"\(...\)"),
        (_BRACKET_OPEN_RE, _BRACKET_CLOSE_RE, r"\[...\]"),
    ):
        if len(opening.findall(text)) != len(closing.findall(text)):
            issues.append(f"unbalanced {label} delimiters")

    if _SPLIT_SUBSCRIPT_RE.search(text):
        issues.append("split subscript: $...$ is followed by .digit (rejoin it)")

    for match in _QA_MATH_PATTERN.finditer(text):
        segment = next(group for group in match.groups() if group is not None)
        if _QA_RAW_COMPARISON.search(segment):
            issues.append(r"raw < or > inside math (use \lt / \gt)")
        if _QA_RAW_PIPE.search(segment):
            issues.append(r"raw | inside math (use \vert{})")

    return list(dict.fromkeys(issues))


#: ``a', 'b`` or a trailing ``']`` — the shape a list literal leaves behind
#: once its opening ``['`` has been stripped.
_LIST_REPR_LEAK = re.compile(r"['\"]\s*,\s*['\"]|['\"]\]")


def _callout_issues(text: str) -> list[str]:
    """Callout text must not carry a Python/JSON list literal.

    The notes model sometimes answers ``['a', 'b']`` for the symbol note. The
    normaliser unpacks that, but a regression there reaches the learner as
    quotes and a stray bracket in the pre-answer card, which no item-level
    check would see.
    """

    for line in text.splitlines():
        if _LIST_REPR_LEAK.search(line):
            return ["callout contains a list literal"]
    return []


def _markdown_issues(text: str) -> list[str]:
    """Markers that would still reach the learner verbatim.

    Pipe tables and fenced code are real blocks since P5, and the export now
    resolves balanced ``*em*`` / ``**strong**`` into ``<em>``/``<strong>``. What
    is left is the *unbalanced* case: an odd asterisk count outside math and
    fences cannot pair up, so the star lands on the page. ``__`` has no renderer
    at all and stays flagged whenever it appears.
    """
    prose: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if not in_fence:
            prose.append(line)
    visible = _INLINE_MATH_RE.sub("", "\n".join(prose))
    issues: list[str] = []
    if visible.count("*") % 2 == 1:
        issues.append("unbalanced '*' marker")
    if _BOLD_UNDERSCORE_RE.search(visible):
        issues.append("bold marker '__'")
    return issues


def _is_placeholder(value: object) -> bool:
    text = str(value or "").strip()
    lowered = text.lower()
    return "[generation failed]" in lowered or text == "N/A"


# ---------------------------------------------------------------------------
# Knowledge-base grounding
# ---------------------------------------------------------------------------


def _fallback_kb_root() -> Path:
    try:
        from deeptutor.services.path_service import get_path_service

        return Path(get_path_service().get_knowledge_bases_root())
    except Exception:
        pass
    home = Path.home()
    data_home = os.environ.get("DEEPTUTOR_HOME")
    base = Path(data_home) if data_home else home / ".deeptutor"
    return base / "data" / "knowledge_bases"


def _kb_root() -> Path:
    """Resolve the KB root like ``generate.infer_chapters``, then guard cwd.

    The core path service roots data at the invocation directory. Running the
    audit by absolute path from another cwd can therefore point at a
    non-existent KB tree, which would silently turn every grounding check into
    "skipped". When the resolved root does not exist, the repo's own data tree
    is used and the substitution is recorded in the JSON report.
    """
    candidate: Path | None = None
    if _knowledge_bases_root is not None:
        try:
            candidate = Path(_knowledge_bases_root())
        except Exception as exc:  # pragma: no cover - defensive
            _IMPORT_NOTES.append(
                f"KB root fallback used ({type(exc).__name__} from path service)"
            )
    if candidate is None:
        candidate = _fallback_kb_root()
    if candidate.is_dir():
        return candidate
    repo_root = ROOT / "data" / "knowledge_bases"
    if repo_root.is_dir():
        _RUNTIME_NOTES.append(
            "KB root resolved to a missing directory "
            f"({candidate}); using repo root {repo_root}"
        )
        return repo_root
    return candidate


def _safe_kb_name(name: object) -> str | None:
    value = str(name or "").strip()
    if _validated_kb_name is not None:
        try:
            return _validated_kb_name(value)
        except Exception:
            return None
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        return None
    if Path(value).name != value:
        return None
    return value


def _canonical_chapter_name(stem: str) -> tuple[str, bool] | None:
    """``(canonical name, primary)`` using generate._chapter_marker.

    ``None`` means the filename has no chapter/experiment marker. Only used
    when the product helper imported successfully.
    """
    if _chapter_marker is None:
        return None
    marker = _chapter_marker(stem)
    if marker is None:
        return None
    kind, number, number_text, title, primary = marker
    # P4 added lecture and courseware markers to the product resolver; the
    # audit must canonicalise them the same way, or a 微机/信号 pack falls
    # through to ``实验N`` and grounding silently finds no source documents.
    if kind == "chapter":
        base = f"第{number}章"
    elif kind == "lecture":
        base = f"第{number}讲"
    elif kind == "courseware":
        base = f"ch{number}"
    else:
        base = f"实验{number_text}"
    name = f"{base}_{title}" if title else base
    return name, primary


class _KbIndex:
    """Raw documents for one KB, split by canonical chapter name."""

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.primary: dict[str, list[Path]] = {}
        self.secondary: dict[str, list[Path]] = {}
        self.unmarked: list[Path] = []
        self._scan()

    def _scan(self) -> None:
        for path in sorted(self.raw_dir.iterdir(), key=lambda entry: entry.name):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            resolved = _canonical_chapter_name(path.stem)
            if resolved is None:
                # No chapter/experiment marker (or the marker helper is
                # unavailable): the file only belongs to the "其他" bucket.
                self.unmarked.append(path)
                continue
            name, primary = resolved
            bucket = self.primary if primary else self.secondary
            bucket.setdefault(name, []).append(path)

    def documents(self, chapter_ref: str) -> list[Path]:
        key = _collapse(chapter_ref)
        if key == OTHER_CHAPTER:
            return list(self.unmarked)
        if _chapter_marker is None:
            # Coarse fallback: every document whose stem contains the marker.
            needle = key.replace(" ", "")
            return [
                path
                for path in sorted(self.raw_dir.iterdir(), key=lambda e: e.name)
                if path.is_file()
                and path.suffix.lower() in TEXT_SUFFIXES
                and needle
                and needle in _INGEST_PREFIX.sub("", path.stem, count=1).replace(" ", "")
            ]
        by_name = {_collapse(name): paths for name, paths in self.primary.items()}
        if key in by_name:
            return list(by_name[key])
        by_secondary = {
            _collapse(name): paths for name, paths in self.secondary.items()
        }
        return list(by_secondary.get(key, []))

    def chapter_text(self, chapter_ref: str) -> tuple[str, int]:
        """Concatenated text and the number of documents actually read."""
        chunks: list[str] = []
        total = 0
        read = 0
        for path in self.documents(chapter_ref):
            try:
                data = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            chunks.append(data[:MAX_DOC_CHARS])
            total += len(chunks[-1])
            read += 1
            if total >= MAX_CHAPTER_CHARS:
                break
        return "\n".join(chunks), read


class _KbCache:
    def __init__(self) -> None:
        self._indexes: dict[str, _KbIndex | None] = {}
        self._texts: dict[tuple[str, str], tuple[str, int] | None] = {}
        self._root: Path | None = None

    def root(self) -> Path:
        if self._root is None:
            self._root = _kb_root()
        return self._root

    def index(self, kb: str) -> _KbIndex | None:
        if kb not in self._indexes:
            safe = _safe_kb_name(kb)
            raw = self.root() / safe / "raw" if safe else None
            self._indexes[kb] = (
                _KbIndex(raw)
                if safe and raw is not None and raw.is_dir()
                else None
            )
        return self._indexes[kb]

    def chapter_text(self, kb: str, chapter: str) -> tuple[str, int] | None:
        key = (kb, _collapse(chapter))
        if key not in self._texts:
            index = self.index(kb)
            self._texts[key] = index.chapter_text(chapter) if index else None
        return self._texts[key]


def _grounding_score(item_text: str, document_text: str) -> float | None:
    item_bigrams = _bigrams(item_text)
    if not item_bigrams:
        return None
    document_bigrams = _bigrams(document_text)
    return round(len(item_bigrams & document_bigrams) / len(item_bigrams), 4)


def _source_scope(
    source: dict[str, Any], default_kb: str
) -> tuple[str, str]:
    """Resolve ``(kb, chapter)`` from a source's kb/ref/title.

    ``generate`` writes ``kb`` and ``ref`` directly, but hand-written specs
    sometimes carry the chapter in ``title`` (``高频 · 第3章_...``) and leave
    ``ref``/``kb`` empty. The fallback keeps grounding useful without guessing:
    the only inference is the explicit ``kb · chapter`` convention.
    """
    kb = str(source.get("kb") or "").strip()
    chapter = str(source.get("ref") or "").strip()
    title = str(source.get("title") or "").strip()
    if "·" in title:
        prefix, _, suffix = title.partition("·")
        if not kb:
            kb = prefix.strip()
        if not chapter:
            chapter = suffix.strip()
    if not chapter:
        chapter = title
    if not kb:
        kb = default_kb
    return kb, chapter


def _audit_grounding(
    item: dict[str, Any],
    sources: dict[str, Any],
    cache: _KbCache,
    threshold: float,
    default_kb: str = "",
) -> list[dict[str, Any]]:
    """Return one grounding entry per distinct KB source of the item."""
    source_ids = (
        item.get("source_ids")
        if isinstance(item.get("source_ids"), list)
        else []
    )
    item_text = f"{item.get('stem') or ''}\n{item.get('explanation') or ''}"
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for source_id in source_ids:
        source = sources.get(source_id) if isinstance(sources, dict) else None
        if not isinstance(source, dict):
            entries.append(
                {
                    "source_id": source_id,
                    "status": "skipped",
                    "reason": "source id not present in spec.sources",
                }
            )
            continue
        kind = str(source.get("kind") or "")
        kb, chapter = _source_scope(source, default_kb)
        if kind not in {"doc", "book_page"} or not kb or not chapter:
            entries.append(
                {
                    "source_id": source_id,
                    "status": "skipped",
                    "reason": "not a KB document source (kind/kb/ref)",
                }
            )
            continue
        key = (kb, _collapse(chapter))
        if key in seen:
            continue
        seen.add(key)

        index = cache.index(kb)
        if index is None:
            entries.append(
                {
                    "source_id": source_id,
                    "kb": kb,
                    "ref": chapter,
                    "status": "skipped",
                    "reason": "knowledge base raw directory not found",
                }
            )
            continue
        chapter_pair = cache.chapter_text(kb, chapter)
        if chapter_pair is None:
            entries.append(
                {
                    "source_id": source_id,
                    "kb": kb,
                    "ref": chapter,
                    "status": "skipped",
                    "reason": "no text documents matched the chapter",
                }
            )
            continue
        chapter_text, doc_count = chapter_pair
        if not chapter_text or doc_count == 0:
            entries.append(
                {
                    "source_id": source_id,
                    "kb": kb,
                    "ref": chapter,
                    "status": "skipped",
                    "reason": "no text documents matched the chapter",
                }
            )
            continue

        score = _grounding_score(item_text, chapter_text)
        if score is None:
            entries.append(
                {
                    "source_id": source_id,
                    "kb": kb,
                    "ref": chapter,
                    "doc_count": doc_count,
                    "status": "skipped",
                    "reason": "item too short for a bigram score",
                }
            )
            continue

        entries.append(
            {
                "source_id": source_id,
                "kb": kb,
                "ref": chapter,
                "doc_count": doc_count,
                "score": score,
                "status": "ok" if score >= threshold else "weak",
                "reason": (
                    "at or above threshold"
                    if score >= threshold
                    else f"below threshold {threshold:g}"
                ),
            }
        )

    return entries


# ---------------------------------------------------------------------------
# Item audit
# ---------------------------------------------------------------------------


def audit_item(
    item: dict[str, Any],
    index: int,
    sources: dict[str, Any],
    cache: _KbCache,
    threshold: float,
    default_kb: str = "",
) -> dict[str, Any]:
    item_id = str(item.get("id") or f"#{index}")
    failures: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    def fail(category: str, message: str) -> None:
        failures.append({"category": category, "item_id": item_id, "message": message})

    def warn(category: str, message: str) -> None:
        warnings.append({"category": category, "item_id": item_id, "message": message})

    kind = str(item.get("kind") or "")
    stem = str(item.get("stem") or "")
    explanation = str(item.get("explanation") or "")
    answer = item.get("answer") if isinstance(item.get("answer"), dict) else {}
    raw_options = item.get("options")
    if raw_options is None:
        raw_options = []
    if not isinstance(raw_options, list):
        fail("structural", "options is not a list")
        raw_options = []
    options = [option for option in raw_options if isinstance(option, dict)]
    if len(options) != len(raw_options):
        fail("structural", "an option is not an object")

    # --- 1. structural -------------------------------------------------
    if not stem.strip():
        fail("structural", "stem is empty")
    if _is_placeholder(stem):
        fail("structural", "stem contains a generation-failed / N/A placeholder")
    if kind not in KNOWN_KINDS:
        fail("structural", f"unknown item kind {kind!r}")

    labels: list[str] = []
    for option_index, option in enumerate(options):
        label = str(option.get("label") or "")
        text = str(option.get("text") or "")
        if not label.strip():
            fail("structural", f"options[{option_index}].label is empty")
        if _is_placeholder(label):
            fail(
                "structural",
                f"options[{option_index}].label is an N/A / generation-failed placeholder",
            )
        if not text.strip():
            fail("structural", f"options[{option_index}].text is empty")
        if _is_placeholder(text):
            fail(
                "structural",
                f"options[{option_index}].text is an N/A / generation-failed placeholder",
            )
        labels.append(label.strip().upper())
    if len(set(labels)) != len(labels):
        fail("structural", "option labels are not unique")

    if kind == "single_choice":
        if len(options) < 2:
            fail("structural", "single_choice has fewer than two options")
    elif kind in WRITTEN_KINDS and options:
        fail("structural", f"{kind} item must not carry options")

    if _is_placeholder(explanation):
        fail("structural", "explanation contains an N/A / generation-failed placeholder")
    if _is_placeholder(answer.get("text")):
        fail("structural", "answer.text is an N/A / generation-failed placeholder")

    # --- 2. answer key -------------------------------------------------
    choice_index = answer.get("choice_index")
    correct_text = ""
    if kind == "single_choice":
        valid_index = (
            isinstance(choice_index, int)
            and not isinstance(choice_index, bool)
            and 0 <= choice_index < len(options)
        )
        if not valid_index:
            fail(
                "answer",
                f"choice_index {choice_index!r} does not map to one of "
                f"{len(options)} options",
            )
        else:
            correct_text = str(options[choice_index].get("text") or "")
        if _is_placeholder(correct_text):
            fail("answer", "correct option text is an N/A / generation-failed placeholder")
    elif kind == "numeric":
        value = answer.get("value")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            fail("answer", f"numeric answer.value is missing or not finite: {value!r}")
    elif kind in WRITTEN_KINDS:
        text = str(answer.get("text") or "").strip()
        if not text:
            fail("answer", "written answer.text is empty")

    # --- 3. explanation coherence --------------------------------------
    if not explanation.strip():
        fail("explanation", "explanation is empty")
    elif kind == "single_choice" and correct_text:
        label = ""
        if isinstance(choice_index, int) and 0 <= choice_index < len(options):
            label = str(options[choice_index].get("label") or "")
        displayed = f"{label}. {correct_text}" if label else correct_text
        if not _explanation_contains(explanation, displayed, correct_text):
            fail(
                "explanation",
                "explanation does not contain the correct option's text",
            )
    elif kind == "numeric":
        value = answer.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            candidates = {str(value), f"{value:g}", f"{value:.3f}"}
            if not any(candidate in explanation for candidate in candidates):
                # G4 reads units ("500 mV" for 500); mirror that instead of
                # failing a correct explanation written with a unit.
                if _qa_mentions_number is None or not _qa_mentions_number(
                    explanation, value
                ):
                    fail("explanation", f"explanation does not contain {value:g}")
    elif kind in WRITTEN_KINDS:
        text = str(answer.get("text") or "").strip()
        if text and not _explanation_contains(explanation, text, text):
            # G4 accepts a high-coverage paraphrase ("得到固定的中频信号" for
            # "固定中频"); the audit must not be stricter than the gate it
            # mirrors, or natural prose shows up as a blocking failure.
            coverage = (
                _answer_coverage(explanation, text) if _answer_coverage else 0.0
            )
            if coverage <= _ANSWER_COVERAGE_MIN:
                fail("explanation", "explanation does not contain the written answer")

    # --- 4. math hygiene -----------------------------------------------
    fields: list[tuple[str, str]] = [
        ("stem", stem),
        ("explanation", explanation),
        ("answer.text", str(answer.get("text") or "")),
    ]
    for option_index, option in enumerate(options):
        fields.append(
            (f"options[{option_index}].text", str(option.get("text") or ""))
        )
    for field_name, text in fields:
        for issue in _math_issues(text):
            fail("math", f"{field_name}: {issue}")

    # --- 5. markdown leakage (non-blocking) ----------------------------
    for field_name, text in fields:
        for issue in _markdown_issues(text):
            warn("markdown", f"{field_name}: {issue}")

    # --- 6. grounding (non-blocking) -----------------------------------
    grounding = _audit_grounding(item, sources, cache, threshold, default_kb)
    for entry in grounding:
        if entry["status"] == "weak":
            warn(
                "grounding",
                "weak grounding: bigram score "
                f"{entry['score']:.4f} < {threshold:g} against "
                f"{entry.get('kb')} / {entry.get('ref')}",
            )
        elif entry["status"] == "skipped":
            warn("grounding", f"grounding skipped: {entry.get('reason')}")
    if not grounding:
        warn("grounding", "no source ids to ground against")

    return {
        "item_id": item_id,
        "kind": kind,
        "failures": failures,
        "warnings": warnings,
        "grounding": grounding,
    }


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


def find_duplicates(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Identical or near-identical normalized stems in one spec."""
    normalized = [
        (
            str(item.get("id") or f"#{index}"),
            _normalized_stem(item.get("stem")),
            str(item.get("variant_of") or ""),
        )
        for index, item in enumerate(items)
        if isinstance(item, dict)
    ]
    duplicates: list[dict[str, Any]] = []
    truncated = False
    for first in range(len(normalized)):
        for second in range(first + 1, len(normalized)):
            id_a, stem_a, variant_a = normalized[first]
            id_b, stem_b, variant_b = normalized[second]
            if not stem_a or not stem_b:
                continue
            if stem_a == stem_b:
                kind = "identical"
                ratio = 1.0
            else:
                shorter = min(len(stem_a), len(stem_b))
                if shorter < NEAR_DUPLICATE_MIN_CHARS:
                    continue
                ratio = SequenceMatcher(None, stem_a, stem_b).ratio()
                if ratio < NEAR_DUPLICATE_RATIO:
                    continue
                kind = "near"
            if len(duplicates) >= MAX_DUPLICATE_PAIRS:
                truncated = True
                continue
            duplicates.append(
                {
                    "a": id_a,
                    "b": id_b,
                    "kind": kind,
                    "ratio": round(ratio, 4),
                    "variant_pair": variant_a == id_b or variant_b == id_a,
                }
            )
    return duplicates, truncated


# ---------------------------------------------------------------------------
# Store access
# ---------------------------------------------------------------------------


def _resolve_home(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    try:
        from deeptutor.review.storage import review_home

        return review_home()
    except Exception:
        override = os.environ.get("DEEPTUTOR_REVIEW_HOME")
        if override:
            return Path(override).expanduser().resolve()
        return _fallback_kb_root().parent / "user" / "workspace" / "review"


def _read_specs(db_path: Path, spec_ids: list[str] | None) -> list[tuple[str, str, str]]:
    """Return ``(spec_id, content_hash, spec_json)`` rows in stable order."""
    try:
        uri = f"{db_path.as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
    except (ValueError, sqlite3.Error):
        connection = sqlite3.connect(str(db_path))
    try:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT spec_id, content_hash, spec_json FROM review_specs ORDER BY spec_id"
        ).fetchall()
    finally:
        connection.close()

    wanted = set(spec_ids or [])
    selected: list[tuple[str, str, str]] = []
    for row in rows:
        spec_id = str(row["spec_id"])
        if wanted and spec_id not in wanted:
            continue
        selected.append((spec_id, str(row["content_hash"] or ""), str(row["spec_json"])))
    if wanted:
        found = {item[0] for item in selected}
        missing = sorted(wanted - found)
        if missing:
            raise LookupError(f"spec id not found in store: {', '.join(missing)}")
    return selected


# ---------------------------------------------------------------------------
# Spec audit
# ---------------------------------------------------------------------------


def _finalize(report: dict[str, Any]) -> dict[str, Any]:
    """Fill the derived pass/fail fields on every audit_spec return path."""
    report["blocking_failures"] = sum(
        report["counts"][category] for category in BLOCKING_CATEGORIES
    )
    report["passed"] = report["blocking_failures"] == 0
    return report


def audit_spec(
    spec_id: str,
    content_hash: str,
    spec_json: str,
    cache: _KbCache,
    threshold: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "spec_id": spec_id,
        "content_hash": content_hash,
        "title": "",
        "item_count": 0,
        "counts": {
            "structural": 0,
            "answer": 0,
            "explanation": 0,
            "math": 0,
            "markdown": 0,
            "grounding_weak": 0,
            "duplicates": 0,
        },
        "failures": [],
        "warnings": [],
        "items": [],
        "duplicates": [],
        "duplicates_truncated": False,
        "parsed": False,
        "blocking_failures": 0,
        "passed": False,
    }

    try:
        spec = json.loads(spec_json)
    except json.JSONDecodeError as exc:
        report["failures"].append(
            {
                "category": "structural",
                "item_id": "",
                "message": f"spec_json is not valid JSON: {exc.msg}",
            }
        )
        report["counts"]["structural"] += 1
        return _finalize(report)

    if not isinstance(spec, dict):
        report["failures"].append(
            {
                "category": "structural",
                "item_id": "",
                "message": "spec_json is not an object",
            }
        )
        report["counts"]["structural"] += 1
        return _finalize(report)

    report["parsed"] = True
    report["title"] = str(spec.get("title") or "")
    stored_id = str(spec.get("id") or "")
    if stored_id and stored_id != spec_id:
        report["warnings"].append(
            {
                "category": "schema",
                "item_id": "",
                "message": f"column spec_id {spec_id!r} != spec.id {stored_id!r}",
            }
        )

    for section in spec.get("sections") or []:
        for block in section.get("blocks") or []:
            if not isinstance(block, dict) or block.get("kind") != "callout":
                continue
            for message in _callout_issues(str(block.get("text") or "")):
                report["failures"].append(
                    {"category": "markdown", "item_id": "", "message": message}
                )
                report["counts"]["markdown"] += 1

    items = spec.get("items")
    if not isinstance(items, list):
        report["failures"].append(
            {
                "category": "structural",
                "item_id": "",
                "message": "spec.items is not a list",
            }
        )
        report["counts"]["structural"] += 1
        return _finalize(report)
    if not items:
        report["failures"].append(
            {
                "category": "structural",
                "item_id": "",
                "message": "spec has no items",
            }
        )
        report["counts"]["structural"] += 1
        return _finalize(report)

    report["item_count"] = len(items)
    sources = spec.get("sources") if isinstance(spec.get("sources"), dict) else {}
    kb_ids = spec.get("kb_ids") if isinstance(spec.get("kb_ids"), list) else []
    # A single KB id is an unambiguous default for sources that omit ``kb``.
    default_kb = str(kb_ids[0]) if len(kb_ids) == 1 else ""

    item_reports: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            report["failures"].append(
                {
                    "category": "structural",
                    "item_id": f"#{index}",
                    "message": "item is not an object",
                }
            )
            report["counts"]["structural"] += 1
            continue
        item_report = audit_item(
            item, index, sources, cache, threshold, default_kb
        )
        item_reports.append(item_report)
        report["failures"].extend(item_report["failures"])
        report["warnings"].extend(item_report["warnings"])
        for failure in item_report["failures"]:
            if failure["category"] in report["counts"]:
                report["counts"][failure["category"]] += 1
        for warning in item_report["warnings"]:
            if warning["category"] == "markdown":
                report["counts"]["markdown"] += 1
            elif (
                warning["category"] == "grounding"
                and "weak grounding" in warning["message"]
            ):
                report["counts"]["grounding_weak"] += 1

    duplicates, truncated = find_duplicates(items)
    report["duplicates"] = duplicates
    report["duplicates_truncated"] = truncated
    report["counts"]["duplicates"] = len(duplicates)
    for pair in duplicates:
        report["warnings"].append(
            {
                "category": "duplicate",
                "item_id": pair["a"],
                "message": (
                    f"stem {pair['kind']} to {pair['b']} (ratio {pair['ratio']:.3f})"
                ),
            }
        )

    report["items"] = item_reports
    return _finalize(report)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_summary(report: dict[str, Any], verbose: bool = False) -> None:
    failures = report["failures"]
    warnings = report["warnings"]
    status = "[OK]" if report["passed"] else "[FAIL]"
    print(
        f"{status} {report['spec_id']}  "
        f"items={report['item_count']}  "
        + "  ".join(
            f"{category}={report['counts'][category]}"
            for category in BLOCKING_CATEGORIES
        )
        + f"  markdown={report['counts']['markdown']}"
        + f"  weak_grounding={report['counts']['grounding_weak']}"
        + f"  duplicates={report['counts']['duplicates']}"
    )
    if not report["parsed"]:
        return

    printed = 0
    for failure in failures:
        if printed >= MAX_DETAIL_LINES:
            break
        print(
            f"    FAIL [{failure['category']}] {failure['item_id']}: "
            f"{failure['message']}"
        )
        printed += 1
    for warning in warnings:
        if (
            not verbose
            and warning["category"] == "grounding"
            and "skipped" in warning["message"]
        ):
            continue
        if printed >= MAX_DETAIL_LINES:
            break
        print(
            f"    warn [{warning['category']}] {warning['item_id']}: "
            f"{warning['message']}"
        )
        printed += 1
    hidden_skips = sum(
        1
        for warning in warnings
        if not verbose
        and warning["category"] == "grounding"
        and "skipped" in warning["message"]
    )
    hidden = len(failures) + len(warnings) - printed - hidden_skips
    if hidden > 0:
        print(
            f"    ... {hidden} more finding(s); see the JSON report for the full list"
        )
    if hidden_skips:
        print(
            f"    ... {hidden_skips} grounding-skipped note(s); use --verbose to show"
        )
    if report["duplicates_truncated"]:
        print(
            f"    ... duplicate pairs truncated at {MAX_DUPLICATE_PAIRS}; "
            "see the JSON report"
        )


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_report(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic, LLM-free content-fidelity audit for stored "
            "Review Studio specs."
        )
    )
    parser.add_argument(
        "--home",
        default=None,
        help=(
            "review home containing review.sqlite3 "
            "(default: DEEPTUTOR_REVIEW_HOME or the core workspace)"
        ),
    )
    parser.add_argument(
        "--spec-id",
        action="append",
        dest="spec_ids",
        default=None,
        metavar="ID",
        help="audit only this spec id; repeat for multiple ids",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON report path (default: dist/content-audit-<timestamp>.json)",
    )
    parser.add_argument(
        "--grounding-threshold",
        type=float,
        default=DEFAULT_GROUNDING_THRESHOLD,
        help=(
            "bigram-overlap threshold below which grounding is flagged "
            f"(default: {DEFAULT_GROUNDING_THRESHOLD:g})"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also print grounding-skipped warnings",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = build_parser().parse_args(argv)
    if not 0.0 <= args.grounding_threshold <= 1.0:
        print("error: --grounding-threshold must be between 0 and 1", file=sys.stderr)
        return 1

    home = _resolve_home(args.home)
    db_path = home / "review.sqlite3"
    fallback_note = ""
    if (
        not db_path.is_file()
        and not args.home
        and not os.environ.get("DEEPTUTOR_REVIEW_HOME")
    ):
        # The core path service roots the workspace at the invocation
        # directory. Running this script by absolute path from another cwd
        # would otherwise look at a non-existent home; the repo store is the
        # one the project is actually using.
        repo_home = ROOT / "data" / "user" / "workspace" / "review"
        if (repo_home / "review.sqlite3").is_file():
            fallback_note = (
                f"note: cwd-resolved home has no store; using repo home {repo_home}"
            )
            home = repo_home
            db_path = home / "review.sqlite3"

    print("Content fidelity audit (deterministic, no LLM)")
    print(f"review home: {home}")
    print(f"stored specs: {db_path}")
    if fallback_note:
        print(fallback_note)
    if _IMPORT_NOTES:
        for note in _IMPORT_NOTES:
            print(f"note: {note}")

    if not db_path.is_file():
        print(f"error: store not found: {db_path}", file=sys.stderr)
        return 1

    try:
        rows = _read_specs(db_path, args.spec_ids)
    except (sqlite3.Error, LookupError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not rows:
        print("error: no specs matched", file=sys.stderr)
        return 1

    cache = _KbCache()
    reports = [
        audit_spec(spec_id, content_hash, spec_json, cache, args.grounding_threshold)
        for spec_id, content_hash, spec_json in rows
    ]

    print("")
    for report in reports:
        _print_summary(report, args.verbose)
    for note in _RUNTIME_NOTES:
        print(f"note: {note}")

    totals = {
        category: sum(report["counts"][category] for report in reports)
        for category in (
            "structural",
            "answer",
            "explanation",
            "math",
            "markdown",
            "grounding_weak",
            "duplicates",
        )
    }
    item_total = sum(report["item_count"] for report in reports)
    blocking = sum(totals[category] for category in BLOCKING_CATEGORIES)
    passed = blocking == 0

    print("")
    print(
        f"Summary: {len(reports)} spec(s), {item_total} item(s), "
        f"{blocking} blocking finding(s), "
        f"{totals['markdown']} markdown warning(s), "
        f"{totals['grounding_weak']} weak-grounding flag(s), "
        f"{totals['duplicates']} duplicate pair(s)"
    )
    print(
        "Grounding note: bigram overlap is coarse lexical evidence only; a low "
        "score is a prompt for human review, not proof of fabrication."
    )

    stamp = _timestamp()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else ROOT / "dist" / f"content-audit-{stamp}.json"
    )
    payload = {
        "audit_version": AUDIT_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "review_home": str(home),
        "db_path": str(db_path),
        "grounding_threshold": args.grounding_threshold,
        "near_duplicate_ratio": NEAR_DUPLICATE_RATIO,
        "imports": _IMPORT_NOTES,
        "runtime_notes": _RUNTIME_NOTES,
        "specs": reports,
        "totals": totals,
        "item_count": item_total,
        "blocking_failures": blocking,
        "passed": passed,
    }
    _write_report(payload, output)
    print(f"JSON report: {output}")
    print("Result: " + ("PASS (exit 0)" if passed else "FAIL (exit 1)"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
