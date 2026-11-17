"""Round 4: the hint-leak detector, tested at the holes the audit found.

Five contracts are pinned here, each at a form the round-3 gate let through:

* a choice hint that names the correct label in any common phrasing leaks,
  while naming a *distractor* label does not;
* a one-character answer leaks only after a reveal phrase ("答案是对的"),
  never from a bare occurrence of the character;
* value+unit pairs are compared in base units (500 mA == 0.5 A, 25% == 0.25,
  2 MHz == 2e6) and an answer tolerance is honoured;
* compaction no longer deletes every "的" (which fabricated 4-grams) and no
  longer casefolds (``mW``/``MW``, ``ω``/``Ω``);
* the paraphrase ratio subtracts stem and distractor material, so a short
  option is not saturated by a topical mention, and the two real generated
  packs in the learner's store still fall back.

The two real packs are read through a read-only SQLite URI and copied into the
test's temporary store; the real database is never opened for writing.

All tool calls run under ``asyncio.run`` because no async pytest plugin is
installed in this package, matching ``tests/test_hint_tool.py``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from deeptutor.core.tool_protocol import BaseTool, ToolResult
from deeptutor.review import agent_tools
from deeptutor.review.agent_tools import ReviewHintTool
from deeptutor.review.models import (
    Answer,
    Concept,
    Item,
    Option,
    ReviewPlan,
    ReviewSpec,
    SourceRef,
)
from deeptutor.review.service import ReviewService

PACK_ID = "round4-hints"
#: Real packs are deployment data, not fixtures. The ids stay out of the public
#: tree; a maintainer opts in with DEEPTUTOR_REAL_SPEC_IDS and
#: DEEPTUTOR_REVIEW_HOME.
REAL_SPEC_IDS = tuple(
    part.strip()
    for part in os.environ.get("DEEPTUTOR_REAL_SPEC_IDS", "").split(",")
    if part.strip()
)


def _require_real_database() -> Path:
    home = os.environ.get("DEEPTUTOR_REVIEW_HOME")
    database = Path(home).expanduser() / "review.sqlite3" if home else None
    if database is None or not database.exists():
        pytest.skip("set DEEPTUTOR_REVIEW_HOME to a store containing the real packs")
    return database


# -- helpers ----------------------------------------------------------------


def _run(tool: BaseTool, **kwargs: Any) -> ToolResult:
    return asyncio.run(tool.execute(**kwargs))


def _payload(result: ToolResult, *, tool: str) -> dict[str, Any]:
    assert result.success is True, result.content
    payload = json.loads(result.content)
    assert isinstance(payload, dict)
    assert "instruction" in payload and payload["instruction"]
    assert result.metadata[tool] == payload
    return payload


def _hint(spec_id: str, item_id: str, *, level: int) -> dict[str, Any]:
    return _payload(
        _run(ReviewHintTool(), spec_id=spec_id, item_id=item_id, level=level),
        tool="review_hint",
    )


def _choice_item(
    *,
    correct: int = 1,
    options: list[tuple[str, str]] | None = None,
    stem: str = "关于该回路，下列判断正确的是：",
) -> Item:
    pairs = options or [
        ("A", "因为甲条件成立"),
        ("B", "因为乙条件成立"),
        ("C", "因为丙条件成立"),
    ]
    return Item(
        id="item-choice",
        concept_ids=["c-choice"],
        kind="single_choice",
        stem=stem,
        options=[Option(label=label, text=text) for label, text in pairs],
        answer=Answer(choice_index=correct),
        explanation=pairs[correct][1],
    )


def _numeric_item(
    value: float, *, tolerance: float = 0.0, stem: str = "求该量的大小。"
) -> Item:
    return Item(
        id="item-numeric",
        concept_ids=["c-numeric"],
        kind="numeric",
        stem=stem,
        answer=Answer(value=value, tolerance=tolerance, text=""),
        explanation=f"结果为 {value:g}。",
    )


def _short_item(answer_text: str, *, stem: str = "请简述该判断。") -> Item:
    return Item(
        id="item-short",
        concept_ids=["c-short"],
        kind="short_answer",
        stem=stem,
        answer=Answer(text=answer_text),
        explanation=answer_text,
    )


def _pack(
    *,
    choice_summary: str = "",
    choice_name: str = "某概念",
    choice_definition: str = "一个与选项无关的概念说明。",
    numeric_definition: str = "",
    short_definition: str = "",
) -> ReviewSpec:
    """A minimal pack that passes every gate, one item per concept."""
    source = SourceRef(
        id="src-r4",
        kind="manual",
        title="第四轮来源材料",
        snippet="测试用来源片段，不含外部链接。",
    )
    concepts = [
        Concept(
            id="c-choice",
            name=choice_name,
            definition=choice_definition,
            summary_one_line=choice_summary,
            source_ids=["src-r4"],
            knowledge_type="concept",
            coverage="kb",
        )
    ]
    items = [
        Item(
            id="item-choice",
            concept_ids=["c-choice"],
            kind="single_choice",
            stem="关于该回路，下列判断正确的是：",
            options=[
                Option(label="A", text="因为甲条件成立"),
                Option(label="B", text="因为乙条件成立"),
                Option(label="C", text="因为丙条件成立"),
            ],
            answer=Answer(choice_index=1),
            explanation="因为乙条件成立",
            source_ids=["src-r4"],
        )
    ]
    if numeric_definition:
        concepts.append(
            Concept(
                id="c-numeric",
                name="某个数值概念",
                definition=numeric_definition,
                source_ids=["src-r4"],
                knowledge_type="concept",
                coverage="kb",
            )
        )
        items.append(
            Item(
                id="item-numeric",
                concept_ids=["c-numeric"],
                kind="numeric",
                stem="求该量的大小。",
                answer=Answer(value=0.5, tolerance=0.0, text=""),
                explanation="输出为 0.5 A。",
                source_ids=["src-r4"],
            )
        )
    if short_definition:
        concepts.append(
            Concept(
                id="c-short",
                name="某个判断概念",
                definition=short_definition,
                source_ids=["src-r4"],
                knowledge_type="concept",
                coverage="kb",
            )
        )
        items.append(
            Item(
                id="item-short",
                concept_ids=["c-short"],
                kind="short_answer",
                stem="请简述该判断。",
                answer=Answer(text="对"),
                explanation="答案是对的。",
                source_ids=["src-r4"],
            )
        )
    return ReviewSpec(
        id=PACK_ID,
        kb_ids=["test"],
        title="第四轮提示泄漏",
        plan=ReviewPlan(daily_minutes=10),
        concepts=concepts,
        items=items,
        sources={"src-r4": source},
    )


# -- 1. label forms ---------------------------------------------------------


@pytest.mark.parametrize(
    "hint",
    [
        "正确的是B",
        "正确的为B",
        "正确选项是B",
        "应选B",
        "应选择B",
        "选B",
        "选择B",
        "B is correct",
        "correct answer is B",
        "answer: B",
    ],
)
def test_the_correct_label_is_detected_in_every_common_phrasing(hint: str) -> None:
    assert agent_tools._hint_leaks(hint, _choice_item(correct=1)) is True


@pytest.mark.parametrize(
    "hint",
    ["选A", "答案：A", "A is correct", "correct answer is A", "正确的是C"],
)
def test_a_distractor_label_is_not_a_leak(hint: str) -> None:
    assert agent_tools._hint_leaks(hint, _choice_item(correct=1)) is False


def test_a_label_inside_a_longer_token_is_not_a_leak() -> None:
    """Only a standalone label token counts: "BC" never names option B."""
    assert agent_tools._hint_leaks("正确的是BC", _choice_item(correct=1)) is False


def test_a_choice_summary_naming_the_answer_label_falls_back_at_level_two(
    service: ReviewService,
) -> None:
    service.put_spec(
        _pack(choice_summary="这一问的结论正确的是B。", choice_definition="")
    )

    payload = _hint(PACK_ID, "item-choice", level=2)

    assert payload["fallback"] is True
    assert "正确的是B" not in payload["hint"]


def test_a_choice_summary_naming_a_distractor_label_is_kept_at_level_two(
    service: ReviewService,
) -> None:
    summary = "先排除选A的说法，再比较其余。"
    service.put_spec(_pack(choice_summary=summary, choice_definition=""))

    payload = _hint(PACK_ID, "item-choice", level=2)

    assert payload["fallback"] is False
    assert summary in payload["hint"]


# -- 2. one-character answers ---------------------------------------------


@pytest.mark.parametrize("hint", ["答案是对的", "正确答案为对", "应选对"])
def test_a_reveal_phrase_still_catches_a_one_character_answer(hint: str) -> None:
    assert agent_tools._written_hint_leaks(hint, _short_item("对")) is True


@pytest.mark.parametrize("hint", ["先判断对错再作答。", "对称与反对称要分清。"])
def test_a_bare_one_character_answer_is_not_a_leak(hint: str) -> None:
    assert agent_tools._written_hint_leaks(hint, _short_item("对")) is False


def test_a_choice_option_of_one_character_is_caught_by_the_reveal_phrase() -> None:
    item = _choice_item(correct=0, options=[("A", "对"), ("B", "错")])

    assert agent_tools._hint_leaks("答案是对的。", item) is True
    assert agent_tools._hint_leaks("先分清对与错。", item) is False


def test_a_one_character_answer_definition_falls_back(
    service: ReviewService,
) -> None:
    service.put_spec(_pack(short_definition="本题答案是对的。"))

    payload = _hint(PACK_ID, "item-short", level=2)

    assert payload["fallback"] is True
    assert "答案是对的" not in payload["hint"]


# -- 3. units and tolerance ------------------------------------------------


@pytest.mark.parametrize(
    "hint,value",
    [
        ("输出为 500 mA。", 0.5),
        ("占比为 25%。", 0.25),
        ("频率为 2 MHz。", 2e6),
        ("电容为 1 μF。", 1e-6),
        ("电阻为 1 kΩ。", 1000),
        ("输出为 2.5 kV。", 2500),
        ("功耗为 30 mW。", 0.03),
        ("时长为 3 ms。", 0.003),
    ],
)
def test_an_equivalent_unit_is_a_leak(hint: str, value: float) -> None:
    assert agent_tools._written_hint_leaks(hint, _numeric_item(value)) is True


@pytest.mark.parametrize(
    "hint,value",
    [
        ("输出为 5 A。", 0.5),
        ("频率为 2 kHz。", 2e6),
        ("电阻为 1 MΩ。", 1000),
        ("功耗为 30 MW。", 0.03),
    ],
)
def test_a_different_magnitude_is_not_a_leak(hint: str, value: float) -> None:
    assert agent_tools._written_hint_leaks(hint, _numeric_item(value)) is False


def test_a_hint_number_inside_the_answer_tolerance_is_a_leak() -> None:
    item = _numeric_item(1.6, tolerance=0.02)

    assert agent_tools._written_hint_leaks("约为 1.59。", item) is True
    assert agent_tools._written_hint_leaks("约为 1.5。", item) is False


def test_a_plain_number_outside_the_tolerance_is_still_not_a_leak() -> None:
    """Plain numbers keep their exact behavior when no tolerance is set."""
    item = _numeric_item(1.6, tolerance=0.0)

    assert agent_tools._written_hint_leaks("约为 1.59。", item) is False
    assert agent_tools._written_hint_leaks("结果为 1.6。", item) is True


def test_a_unit_equivalent_definition_falls_back(service: ReviewService) -> None:
    service.put_spec(_pack(numeric_definition="输出在 500 mA 时达到稳定。"))

    payload = _hint(PACK_ID, "item-numeric", level=2)

    assert payload["fallback"] is True
    assert "500 mA" not in payload["hint"]


# -- 4. compaction ---------------------------------------------------------


def test_compaction_keeps_the_possessive_particle() -> None:
    assert "的" in agent_tools._hint_compact("电路的响应")


def test_compaction_keeps_case_and_greek_distinctions() -> None:
    compact = agent_tools._hint_compact

    assert compact("1 mW") != compact("1 MW")
    assert compact("Ω") != compact("ω")


def test_deleting_the_particle_no_longer_fabricates_a_shared_phrase() -> None:
    """``电阻的值随…`` used to compact to ``电阻值随`` and hit the hint."""
    stem = "该量如何变化？"
    item = _choice_item(
        correct=0,
        options=[("A", "电阻的值随多少变化"), ("B", "无关")],
        stem=stem,
    )
    hint = "先看电阻值随温度怎样变再判断"

    assert agent_tools._paraphrase_leaks(hint, item) is False


# -- 5. paraphrase ratio ---------------------------------------------------


def test_a_short_option_is_not_saturated_by_a_stem_topic() -> None:
    """``容性`` is the stem's own word, so repeating it proves nothing."""
    stem = "该回路在某一频率下呈现容性还是感性？请判断。"
    item = _choice_item(
        correct=0, options=[("A", "呈容性"), ("B", "呈感性")], stem=stem
    )

    assert agent_tools._paraphrase_leaks("回路的容性或者感性由频率决定", item) is False


def test_a_paraphrased_normal_option_still_leaks() -> None:
    stem = "串联谐振发生时，回路阻抗为什么最小？"
    item = _choice_item(
        correct=1,
        options=[
            ("A", "因为电感电压为零"),
            ("B", "因为感抗与容抗相互抵消"),
            ("C", "因为电阻被短路"),
        ],
        stem=stem,
    )

    assert agent_tools._paraphrase_leaks("谐振时感抗与容抗相互抵消，回路呈纯电阻性。", item)


# -- 6. the real generated packs ------------------------------------------


def _real_spec(spec_id: str) -> ReviewSpec:
    database = _require_real_database()
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT spec_json FROM review_specs WHERE spec_id = ?", (spec_id,)
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        pytest.skip(f"{spec_id} is not in the local review store")
    return ReviewSpec.model_validate_json(row[0])


@pytest.mark.skipif(
    not REAL_SPEC_IDS,
    reason="set DEEPTUTOR_REAL_SPEC_IDS to run against real packs",
)
@pytest.mark.parametrize("spec_id", REAL_SPEC_IDS)
def test_a_real_generated_pack_still_falls_back(
    service: ReviewService, spec_id: str
) -> None:
    database = _require_real_database()
    spec = _real_spec(spec_id)
    item = spec.item_by_id(f"{spec_id}-q001")
    correct = item.options[item.answer.choice_index].text
    before = hashlib.sha256(database.read_bytes()).hexdigest()

    service.put_spec(spec)

    for level in (1, 2):
        payload = _hint(spec.id, item.id, level=level)
        assert payload["fallback"] is True, (spec_id, level, payload["hint"])
        assert correct not in payload["hint"]
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
