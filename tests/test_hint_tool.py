"""Phase 34: the ``review_hint`` deterministic hint ladder.

The hint must help without ever restating the answer. These tests drive the
tool exactly as a chat model would — model-shaped arguments through
``execute``, JSON body parsed back — and pin four contracts:

* level 1 names the concept without the statement, level 2 adds the definition/formula;
* a hint that would leak the answer falls back to source guidance;
* a missing pack / item / id is a *soft* failure, not an exception;
* the attempt flow treats a hinted ``good`` as ``hard`` and passes confidence
  through untouched.

No async test plugin is installed in this package, so every call runs under
``asyncio.run``, matching ``tests/test_agent_tools.py``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from deeptutor.core.tool_protocol import BaseTool, ToolResult
from deeptutor.review.agent_tools import ReviewAttemptTool, ReviewHintTool
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.models import ReviewSpec
from deeptutor.review.service import ReviewService

CHOICE_ITEM = "item-resonance-choice"
FLASHCARD_ITEM = "item-quality-flashcard"
RESONANCE_CONCEPT = "c-series-resonance"
NAME = "串联谐振"
SUMMARY = "谐振时 XL = XC，Z = R 最小，电流最大。"
DEFINITION = "串联回路中感抗与容抗相互抵消、回路呈纯电阻性的状态。"
SAFE_DEFINITION = "串联谐振回路的定义与适用条件需要结合元件参数分析。"
FORMULA_LATEX = r"f_0 = \frac{1}{2\pi\sqrt{LC}}"
CHOICE_ANSWER_TEXT = "因为感抗与容抗相互抵消"
FLASHCARD_ANSWER_TEXT = r"Q = \frac{1}{R}\sqrt{\frac{L}{C}}"


def _run(tool: BaseTool, **kwargs: Any) -> ToolResult:
    return asyncio.run(tool.execute(**kwargs))


def _payload(result: ToolResult, *, tool: str) -> dict[str, Any]:
    """Assert the frozen result shape, then return the parsed payload."""
    assert result.success is True, result.content
    payload = json.loads(result.content)
    assert isinstance(payload, dict)
    assert "instruction" in payload and payload["instruction"]
    assert result.metadata[tool] == payload
    return payload


@pytest.fixture()
def seeded(service: ReviewService) -> ReviewService:
    service.put_spec(build_fixture_spec())
    return service


def _hint(**kwargs: Any) -> dict[str, Any]:
    return _payload(_run(ReviewHintTool(), **kwargs), tool="review_hint")


def _attempt(**kwargs: Any) -> dict[str, Any]:
    return _payload(_run(ReviewAttemptTool(), **kwargs), tool="review_attempt")


def _variant(**updates: Any) -> Any:
    """The fixture pack with one concept's fields replaced.

    Mutating a validated model leaves ``content_hash`` stale, and ``put_spec``
    rejects the mismatch. Clearing it makes the revalidation recompute it.
    """
    spec = build_fixture_spec()
    for concept in spec.concepts:
        if concept.id == RESONANCE_CONCEPT:
            for field, value in updates.items():
                setattr(concept, field, value)
    return ReviewSpec.model_validate({**spec.model_dump(mode="json"), "content_hash": ""})


# -- level ladder -----------------------------------------------------------


def test_level_one_returns_the_concept_name(seeded: ReviewService) -> None:
    payload = _hint(spec_id=SPEC_ID, item_id=CHOICE_ITEM, level=1)
    assert payload["level"] == 1
    assert payload["fallback"] is False
    assert payload["hint"] == NAME
    assert SUMMARY not in payload["hint"]
    assert payload["source"] is None
    assert payload["item_id"] == CHOICE_ITEM


def test_level_two_returns_definition_and_formula(seeded: ReviewService) -> None:
    seeded.put_spec(_variant(definition=SAFE_DEFINITION))
    payload = _hint(spec_id=SPEC_ID, item_id=CHOICE_ITEM, level=2)
    assert payload["level"] == 2
    assert payload["fallback"] is False
    assert SAFE_DEFINITION in payload["hint"]
    assert FORMULA_LATEX in payload["hint"]


def test_level_out_of_range_is_clamped_to_the_ladder(seeded: ReviewService) -> None:
    seeded.put_spec(_variant(definition=SAFE_DEFINITION))
    low = _hint(spec_id=SPEC_ID, item_id=CHOICE_ITEM, level=0)
    high = _hint(spec_id=SPEC_ID, item_id=CHOICE_ITEM, level=99)
    assert low["level"] == 1
    assert low["hint"] == NAME
    assert high["level"] == 2
    assert SAFE_DEFINITION in high["hint"]


def test_authored_definition_that_answers_the_question_is_blocked(seeded: ReviewService) -> None:
    payload = _hint(spec_id=SPEC_ID, item_id=CHOICE_ITEM, level=2)
    assert payload["fallback"] is True
    assert DEFINITION not in payload["hint"]


# -- leak gate and fallback -------------------------------------------------


def test_definition_containing_the_correct_option_falls_back(
    service: ReviewService,
) -> None:
    """A definition that spells out the right option must not be read aloud."""
    service.put_spec(_variant(definition=f"{CHOICE_ANSWER_TEXT}，回路呈纯电阻性。"))
    result = _hint(spec_id=SPEC_ID, item_id=CHOICE_ITEM, level=2)
    assert result["fallback"] is True
    assert CHOICE_ANSWER_TEXT not in result["hint"]
    # A valid pack always carries a source for every item, so the fallback
    # anchors to it instead of the generic text.
    assert "第2章 2.1 串联谐振" in result["hint"]


def test_generated_style_concept_falls_back_to_its_name(service: ReviewService) -> None:
    """Generated packs carry only the concept name, so the ladder uses it.

    The name arrives only through the same leak gate: a name that restates
    the correct option still degrades to the source guidance.
    """
    name = "串联谐振回路的品质因数与通频带"
    service.put_spec(_variant(name=name, summary_one_line="", definition="", formulas=[]))
    level_one = _hint(spec_id=SPEC_ID, item_id=CHOICE_ITEM, level=1)
    assert level_one["fallback"] is False
    assert level_one["hint"] == name
    level_two = _hint(spec_id=SPEC_ID, item_id=CHOICE_ITEM, level=2)
    assert level_two["fallback"] is False
    # Level 2 must not repeat level 1 verbatim: with no definition in the
    # pack it points at the definition and its boundary instead.
    assert level_two["hint"] != level_one["hint"]
    assert name in level_two["hint"]
    assert "完整定义与适用条件" in level_two["hint"]


def test_generated_style_long_name_keeps_the_two_levels_apart(
    service: ReviewService,
) -> None:
    """A real generated name is a long sentence; level 1 must stay a brief."""
    name = (
        "串联谐振回路的品质因数与通频带，以及激励频率偏离时电流响应特性"
        "与选择性指标的计算方法及其与元件参数的关系，重点掌握品质因数与"
        "通频带的定义式、二者与回路元件参数的定量关系，并能据此分析选择性的变化趋势"
    )
    assert len(name) > 80
    service.put_spec(_variant(name=name, summary_one_line="", definition="", formulas=[]))
    level_one = _hint(spec_id=SPEC_ID, item_id=CHOICE_ITEM, level=1)
    level_two = _hint(spec_id=SPEC_ID, item_id=CHOICE_ITEM, level=2)
    assert level_one["fallback"] is False
    assert level_two["fallback"] is False
    assert len(level_one["hint"]) <= 80
    assert level_one["hint"].endswith("…")
    assert name in level_two["hint"]
    assert "完整定义与适用条件" in level_two["hint"]
    assert level_one["hint"] != level_two["hint"]


def test_hint_fallback_without_a_matching_source_is_generic() -> None:
    """Defensive branch: a pack whose source map lost the item's source."""
    from deeptutor.review import agent_tools

    spec = build_fixture_spec()
    item = spec.item_by_id(CHOICE_ITEM)
    spec.sources = {}
    hint, source = agent_tools._hint_fallback(spec, item)
    assert source is None
    assert hint == "先回顾该题涉及的概念要点后再作答（本包没有更细的来源锚点）。"


def test_fallback_anchors_to_the_items_source(service: ReviewService) -> None:
    """The fallback must point at the source material, not empty guidance."""
    service.put_spec(_variant(definition=f"{CHOICE_ANSWER_TEXT}，回路呈纯电阻性。"))
    payload = _hint(spec_id=SPEC_ID, item_id=CHOICE_ITEM, level=2)
    assert payload["fallback"] is True
    assert payload["source"] == {
        "title": "第2章 2.1 串联谐振",
        "ref": "chapter2/section-2.1",
    }
    assert "第2章 2.1 串联谐振" in payload["hint"]
    assert "chapter2/section-2.1" in payload["hint"]


def test_empty_summary_and_definition_fall_back_to_the_name(
    service: ReviewService,
) -> None:
    """Generated packs have only the concept name; it becomes the hint.

    The name is the last rung of the ladder, still under the leak gate.
    """
    spec = build_fixture_spec()
    name = spec.concept_by_id(RESONANCE_CONCEPT).name
    service.put_spec(
        _variant(summary_one_line="", definition="", formulas=[])
    )
    level_one = _hint(spec_id=SPEC_ID, item_id=CHOICE_ITEM, level=1)
    level_two = _hint(spec_id=SPEC_ID, item_id=CHOICE_ITEM, level=2)
    assert level_one["fallback"] is False
    assert level_two["fallback"] is False
    assert level_one["hint"] == name
    # With no definition, level 2 points at one instead of copying level 1.
    assert level_two["hint"] != level_one["hint"]
    assert name in level_two["hint"]


def test_written_answer_text_is_never_restated(seeded: ReviewService) -> None:
    """The formula in the definition is also the written answer, so the gate
    must fall back rather than dictate it."""
    payload = _hint(spec_id=SPEC_ID, item_id=FLASHCARD_ITEM, level=2)
    assert payload["fallback"] is True
    assert FLASHCARD_ANSWER_TEXT not in payload["hint"]


# -- soft failures ----------------------------------------------------------


def test_unknown_item_is_a_soft_failure(seeded: ReviewService) -> None:
    payload = _hint(spec_id=SPEC_ID, item_id="item-does-not-exist", level=1)
    assert payload["error"] == "unknown item: item-does-not-exist"
    assert payload["spec_id"] == SPEC_ID
    assert "review_today" in payload["instruction"]


def test_missing_item_id_is_a_soft_failure(seeded: ReviewService) -> None:
    payload = _hint(spec_id=SPEC_ID, level=1)
    assert payload["error"] == "item_id is required"
    assert "review_today" in payload["instruction"]


def test_unknown_spec_is_a_soft_failure(seeded: ReviewService) -> None:
    payload = _hint(spec_id="spec-that-does-not-exist", item_id=CHOICE_ITEM)
    assert payload["error"] == "spec not found: spec-that-does-not-exist"
    assert "review_specs" in payload["instruction"]


# -- attempt integration ----------------------------------------------------


def test_hinted_good_is_recorded_as_hard(seeded: ReviewService) -> None:
    payload = _attempt(
        spec_id=SPEC_ID,
        item_id=CHOICE_ITEM,
        rating="good",
        hints_used=1,
        submission_id="hint-attempt-1",
    )
    assert payload["rating"] == "hard"
    assert payload["requested_rating"] == "good"
    assert payload["hints_used"] == 1
    assert "hinted" in payload["instruction"]
    assert "unaided" in payload["instruction"]


def test_unaided_good_stays_good(seeded: ReviewService) -> None:
    payload = _attempt(
        spec_id=SPEC_ID,
        item_id=CHOICE_ITEM,
        rating="good",
        hints_used=0,
        submission_id="hint-attempt-2",
    )
    assert payload["rating"] == "good"
    assert payload["requested_rating"] == "good"
    assert payload["hints_used"] == 0


def test_confidence_is_passed_through(seeded: ReviewService) -> None:
    payload = _attempt(
        spec_id=SPEC_ID,
        item_id=CHOICE_ITEM,
        rating="good",
        confidence=80,
        submission_id="hint-confidence-1",
    )
    assert payload["confidence"] == 80.0


def test_non_numeric_confidence_is_a_soft_failure(seeded: ReviewService) -> None:
    payload = _attempt(
        spec_id=SPEC_ID,
        item_id=CHOICE_ITEM,
        rating="good",
        confidence="high",
    )
    assert payload["error"] == "confidence must be a number between 0 and 100"
    assert "confidence" in payload["instruction"]


def test_uncoercible_hints_used_is_a_soft_failure(seeded: ReviewService) -> None:
    payload = _attempt(
        spec_id=SPEC_ID,
        item_id=CHOICE_ITEM,
        rating="good",
        hints_used="two",
    )
    assert "hints_used must be an integer" in payload["error"]


def test_out_of_range_confidence_is_a_soft_failure(seeded: ReviewService) -> None:
    payload = _attempt(
        spec_id=SPEC_ID,
        item_id=CHOICE_ITEM,
        rating="good",
        confidence=150,
    )
    assert "between 0 and 100" in payload["error"]


def test_source_metadata_that_restates_the_answer_still_falls_back(
    service: ReviewService,
) -> None:
    """Source title/ref are authored content and go through the leak gate."""
    spec = _variant(
        name=f"考虑{CHOICE_ANSWER_TEXT}的情况",
        summary_one_line="",
        definition="",
        formulas=[],
    )
    payload = spec.model_dump(mode="json")
    payload["sources"]["src-hf-2-1"]["ref"] = CHOICE_ANSWER_TEXT
    payload["content_hash"] = ""
    service.put_spec(payload)

    result = _hint(spec_id=SPEC_ID, item_id=CHOICE_ITEM, level=1)
    assert result["fallback"] is True
    assert CHOICE_ANSWER_TEXT not in result["hint"]
    assert result["source"] is None
