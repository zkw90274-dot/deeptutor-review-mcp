"""Best-effort per-concept notes for generated review packs.

The question pipeline names each concept (``concentration``) but fills no
summary or definition, so ``review_hint`` has to fall back to a truncated name
at level 1 and a "recall the definition" prompt at level 2. One extra
completion call per pack fills ``summary_one_line`` and ``definition`` from the
questions the pack already contains.

The call is best-effort: a transport error, a malformed reply or a note that
gives an answer away leaves that concept name-only and records a warning, so a
pack is never lost because of this step. Accepted notes still pass the same
G3 math normalization and the same leak checks ``review_hint`` applies at read
time before they are stored.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from deeptutor.review.models import normalize_text
from deeptutor.review.qa import unbalanced_math_delimiters

Complete = Callable[[str, str], str]
Clean = Callable[[str], str]
Warn = Callable[[str], None]

#: Level-1 hints show the summary verbatim, so keep it to one short sentence.
NOTE_SUMMARY_CHARS = 120

#: Level 2 appends a formula after the definition; this leaves room under the
#: 800-character hint ceiling.
NOTE_DEFINITION_CHARS = 600

#: Static study aids attached to the answer panel / section. They are shown
#: only after the attempt (pitfalls/self-check) or contain no answer values
#: (symbols), so they stay short.
NOTE_SYMBOLS_CHARS = 240
NOTE_SELF_CHECK_CHARS = 160
NOTE_PITFALL_CHARS = 120
NOTE_ASSUMPTION_CHARS = 120
MAX_PITFALLS = 3
MAX_ASSUMPTIONS = 3
#: Comparison tables are rendered after the attempt; keep them small enough to
#: stay legible on a phone and to reject a model that dumps a whole lecture.
NOTE_COMPARE_TITLE_CHARS = 40
NOTE_COMPARE_CELL_CHARS = 40
MAX_COMPARE_COLUMNS = 4
MAX_COMPARE_ROWS = 6

#: One reply for a whole pack; larger packs are chunked so the model does not
#: have to hold a dozen concepts in one answer. Two (rather than eight) keeps
#: the expanded notes schema (study aids + compare table) inside the reply
#: budget: a four-concept reply already truncated at ~2.7k characters in a
#: real run, and a truncated tail is invalid JSON that loses the whole chunk.
_CHUNK = 2

_SYSTEM_PROMPT = (
    "你是复习资料编辑。根据某章已有的题目与解析，为每个知识点写提示与自检材料，"
    "用于学习者回忆、验算与纠错。\n"
    "只输出严格 JSON 数组，不要输出其他文字。每项格式：\n"
    '{"name": "<知识点名，逐字照抄>", "summary": "<一句话概括，不超过60字>", '
    '"definition": "<完整定义与适用条件，不超过300字>", '
    '"symbols": "<符号与单位，1-3 条；没有明确符号就留空>", '
    '"pitfalls": ["<本题常见错误，1-3 条；没有就留空>"], '
    '"self_check": "<一句可执行的验算/自检提示；没有就留空>", '
    '"assumptions": ["<本题成立所需的假设或适用条件，1-3 条；没有就留空>"], '
    '"compare": {"title": "<对照表标题；没有就留空>", '
    '"columns": ["<列名>", "<列名>"], "rows": [["<单元格>", "<单元格>"]]}}\n'
    "要求：\n"
    "1. summary/definition 只描述知识点本身，不得复述任何题目的答案、选项文字或具体数值结果；\n"
    "2. 不出现“正确答案是”“选 X”等字样；\n"
    "3. 数学表达式用 $...$ 包裹，不等式用 \\lt / \\gt，绝对值用 \\vert{}；\n"
    "4. 每个知识点恰好一项，name 与输入完全一致；\n"
    "5. symbols 只写符号、含义、单位，不写答案数值；pitfalls/self_check/assumptions 可以提示正确做法，但不要整段复述答案；\n"
    "6. compare 只在知识点本身包含 2-4 个可并列对象时给出（如三种传播方式、两类滤波器、四个寄存器）："
    "2-4 列、1-6 行、每格不超过 30 字；没有就输出空对象 {}。"
)


def _answer_brief(item: Any) -> str:
    answer = getattr(item, "answer", None)
    options = list(getattr(item, "options", None) or [])
    choice_index = getattr(answer, "choice_index", None)
    if isinstance(choice_index, int) and 0 <= choice_index < len(options):
        return str(getattr(options[choice_index], "text", "") or "")
    value = getattr(answer, "value", None)
    if value is not None:
        return str(value)
    return str(getattr(answer, "text", "") or "")


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return "" if unbalanced_math_delimiters(text) else text


def build_prompt(chapter: str, concepts: list[Any], items_by_concept: dict[str, list[Any]]) -> str:
    lines = [f"章节：{chapter}", "知识点及本章相关题目："]
    for index, concept in enumerate(concepts, start=1):
        lines.append(f"{index}. {concept.name}")
        for item in items_by_concept.get(concept.id, [])[:4]:
            stem = _clip(str(getattr(item, "stem", "") or ""), 200)
            answer = _clip(_answer_brief(item), 160)
            explanation = _clip(str(getattr(item, "explanation", "") or ""), 240)
            detail = f"题：{stem}"
            if answer:
                detail += f"（参考要点：{answer}"
                if explanation:
                    detail += f"；解析：{explanation}"
                detail += "）"
            elif explanation:
                detail += f"（解析：{explanation}）"
            lines.append(f"   - {detail}")
    return "\n".join(lines)


def _extract_json(raw: str) -> Any:
    text = str(raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start == -1 or end <= start:
            continue
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            continue
    raise ValueError("reply contains no JSON payload")


def _entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("concepts") or payload.get("notes") or []
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


def _note_leaks(text: str, item: Any) -> bool:
    # Deferred: agent_tools imports the service layer, and generate imports
    # this module. At call time both are fully loaded, so no cycle forms.
    from deeptutor.review.agent_tools import (
        _hint_leaks,
        _paraphrase_leaks,
        _written_hint_leaks,
    )

    return (
        _hint_leaks(text, item)
        or _written_hint_leaks(text, item)
        or _paraphrase_leaks(text, item)
    )


def _apply(
    concept: Any,
    entry: dict[str, Any],
    items: list[Any],
    *,
    clean: Clean,
    warn: Warn,
) -> bool:
    summary = _clip(clean(str(entry.get("summary") or "")), NOTE_SUMMARY_CHARS)
    definition = _clip(clean(str(entry.get("definition") or "")), NOTE_DEFINITION_CHARS)
    if summary and any(_note_leaks(summary, item) for item in items):
        warn(f"概念「{concept.name}」的一句话摘要可能泄露答案，已丢弃")
        summary = ""
    if definition and any(_note_leaks(definition, item) for item in items):
        warn(f"概念「{concept.name}」的定义可能泄露答案，已丢弃")
        definition = ""
    if summary and summary == definition:
        # Both rungs would read the same; keep the fuller definition.
        summary = ""
    if not summary and not definition:
        return False
    concept.summary_one_line = summary or concept.summary_one_line
    concept.definition = definition or concept.definition
    return True


def _compare_for(entry: dict[str, Any], clean: Clean) -> dict[str, Any] | None:
    """Validate the optional comparison table; any malformed shape is dropped."""
    raw = entry.get("compare")
    if not isinstance(raw, dict):
        return None
    columns_raw = raw.get("columns")
    rows_raw = raw.get("rows")
    if not isinstance(columns_raw, list) or not isinstance(rows_raw, list):
        return None
    if not 2 <= len(columns_raw) <= MAX_COMPARE_COLUMNS:
        return None
    columns = [
        _clip(clean(str(cell or "")), NOTE_COMPARE_CELL_CHARS) for cell in columns_raw
    ]
    if any(not cell for cell in columns):
        return None
    rows: list[list[str]] = []
    for row in rows_raw:
        if not isinstance(row, list) or len(row) != len(columns):
            return None
        cells = [
            _clip(clean(str(cell or "")), NOTE_COMPARE_CELL_CHARS) for cell in row
        ]
        if any(not cell for cell in cells):
            return None
        rows.append(cells)
        if len(rows) >= MAX_COMPARE_ROWS:
            break
    if not rows:
        return None
    title = _clip(clean(str(raw.get("title") or "")), NOTE_COMPARE_TITLE_CHARS)
    return {"title": title, "columns": columns, "rows": rows}


def compare_markdown(compare: dict[str, Any] | None) -> str:
    """Render a validated compare object as a Markdown table.

    The export page's table renderer consumes this string, so the table stays
    plain data (no HTML) and can be printed or copied like the rest of the
    answer. ``|`` inside a cell would split the row, so it becomes a
    full-width bar; newlines collapse to spaces.
    """
    if not compare:
        return ""
    columns = [str(cell) for cell in compare.get("columns") or []]
    rows = compare.get("rows") or []
    if len(columns) < 2 or not rows:
        return ""

    def cell(value: object) -> str:
        return (
            str(value or "")
            .replace("|", "｜")
            .replace("\r", " ")
            .replace("\n", " ")
            .strip()
        )

    lines: list[str] = []
    title = str(compare.get("title") or "").strip()
    if title:
        lines.extend([cell(title), ""])
    lines.append("| " + " | ".join(cell(value) for value in columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        if isinstance(row, list) and len(row) == len(columns):
            lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def _extras_for(
    concept: Any,
    entry: dict[str, Any],
    items: list[Any],
    *,
    clean: Clean,
    warn: Warn,
) -> dict[str, Any]:
    """Collect the study aids that do not fit Concept's schema.

    ``symbols`` is shown before the attempt, so it passes the same leak gate as
    the summary/definition. ``self_check`` and ``pitfalls`` are rendered inside
    the answer panel, after the learner has answered, so they may mention the
    correct approach without spoiling the question.
    """
    symbols = _clip(clean(str(entry.get("symbols") or "")), NOTE_SYMBOLS_CHARS)
    if symbols and any(_note_leaks(symbols, item) for item in items):
        warn(f"概念「{concept.name}」的符号与单位可能泄露答案，已丢弃")
        symbols = ""

    self_check = _clip(
        clean(str(entry.get("self_check") or "")), NOTE_SELF_CHECK_CHARS
    )

    def many(key: str, limit: int, chars: int) -> list[str]:
        raw = entry.get(key) or []
        if isinstance(raw, str):
            raw = [raw]
        values: list[str] = []
        if isinstance(raw, list):
            for item in raw:
                text = _clip(clean(str(item or "")), chars).rstrip("。；;")
                if text and text not in values:
                    values.append(text)
                if len(values) >= limit:
                    break
        return values

    pitfalls = many("pitfalls", MAX_PITFALLS, NOTE_PITFALL_CHARS)
    assumptions = many("assumptions", MAX_ASSUMPTIONS, NOTE_ASSUMPTION_CHARS)
    compare = _compare_for(entry, clean)

    extra: dict[str, Any] = {}
    if symbols:
        extra["symbols"] = symbols
    if self_check:
        extra["self_check"] = self_check
    if pitfalls:
        extra["pitfalls"] = pitfalls
    if assumptions:
        extra["assumptions"] = assumptions
    if compare:
        extra["compare"] = compare
    return extra


def enrich_concepts(
    concepts: list[Any],
    items_by_concept: dict[str, list[Any]],
    *,
    chapter: str,
    complete: Complete | None,
    clean: Clean,
    warn: Warn,
    extras: dict[str, dict[str, Any]] | None = None,
) -> int:
    """Fill concept notes in place; return how many concepts were enriched.

    ``extras`` is an optional collector for the section callout (symbols) and
    the answer-panel study aids (self-check/pitfalls). They are deliberately
    not added to :class:`~deeptutor.review.models.Concept`: ReviewSpec v1 stays
    unchanged, and the generated block/explanation carries the text.
    """

    if complete is None or not concepts:
        return 0
    expected = {normalize_text(concept.name): concept for concept in concepts}
    enriched = 0
    for offset in range(0, len(concepts), _CHUNK):
        chunk = concepts[offset : offset + _CHUNK]
        try:
            reply = complete(
                build_prompt(chapter, chunk, items_by_concept), _SYSTEM_PROMPT
            )
            payload = _extract_json(reply)
        except Exception as exc:  # noqa: BLE001 - notes are optional
            warn(f"概念摘要生成失败（{type(exc).__name__}），本包仍可复习")
            continue
        entries = _entries(payload)
        recognized = 0
        for entry in entries:
            name = normalize_text(str(entry.get("name") or ""))
            concept = expected.get(name)
            if concept is None:
                continue
            recognized += 1
            accepted = _apply(
                concept,
                entry,
                items_by_concept.get(concept.id, []),
                clean=clean,
                warn=warn,
            )
            if extras is not None:
                extra = _extras_for(
                    concept,
                    entry,
                    items_by_concept.get(concept.id, []),
                    clean=clean,
                    warn=warn,
                )
                if extra:
                    extras[concept.id] = extra
                    accepted = True
            if accepted:
                enriched += 1
        if not entries:
            warn("概念摘要为空，已跳过")
        elif not recognized:
            # A reply in some other shape (for example a question analysis
            # echo) must not fail silently: the learner would never know the
            # notes were skipped.
            warn("概念摘要格式不符（缺少 name/summary/definition），已跳过")
    return enriched
