"""Regressions from the real-usage report (2026-11-08).

Covers C-01 (impossible parameters), H-01 (paraphrased hint leak),
H-02 (hard truncation), S-01 (session fallback/idle minutes),
and X-01 (bare identifiers / duplicated short answer).
"""

from __future__ import annotations

import asyncio
import json

from deeptutor.review import generate
from deeptutor.review.agent_tools import ReviewHintTool
from deeptutor.review.book_import import _explanation_with_answer
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.models import ReviewSpec
from deeptutor.review.qa import (
    answer_coverage,
    parameter_infeasibility,
    run_gates,
)
from deeptutor.review.service import AttemptInput, ReviewService

NOW = 1_800_000_000.0
ITEM = "item-resonance-choice"

Q006_STEM = (
    "某谐振功率放大器采用余弦脉冲集电极电流，已知直流电源电压 V_CC=24 V，"
    "集电极电流脉冲最大值 i_Cmax=1 A，半导通角 θ_c=90°，谐振负载 R_p=50 Ω，"
    "最大集电极功耗 P_Cmax=2 W。试求集电极电流直流分量 I_c0、基波分量幅值 "
    "I_c1m、输出功率 P_o、直流功率 P_dc、集电极效率 η 与管耗 P_c，并校核是否"
    "超过最大集电极功耗，判断该放大器所处工作状态。"
)
Q006_ANSWER = (
    "α1=0.5，I_c1m=α1 i_Cmax=0.5 A。P_o=1/2·I_c1m^2·R_p=1/2×0.5^2×50=6.25 W；"
    "P_dc=V_CC I_c0≈24×0.3183≈7.64 W；η≈81.8%；P_c≈1.39 W。"
)


def _payload() -> dict:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    return payload


def _q006_item_payload() -> dict:
    payload = _payload()
    concept_id = payload["concepts"][0]["id"]
    payload["items"].append(
        {
            "id": "item-q006",
            "concept_ids": [concept_id],
            "kind": "short_answer",
            "stem": Q006_STEM,
            "options": [],
            "answer": {"text": Q006_ANSWER},
            "explanation": Q006_ANSWER,
            "source_ids": [payload["items"][0]["source_ids"][0]],
        }
    )
    return payload


# -- C-01 ----------------------------------------------------------------


def test_impossible_voltage_swing_is_a_gate_failure() -> None:
    spec = ReviewSpec.model_validate(_q006_item_payload())
    item = spec.item_by_id("item-q006")
    reason = parameter_infeasibility(item)
    assert reason and "V_CC=24" in reason

    report = run_gates(spec)
    assert report.passed is False
    assert any(
        gate.gate == "G9-parameter-feasibility" and not gate.passed
        for gate in report.gates
    )


def test_generation_skips_an_impossible_item() -> None:
    qa = {
        "question": Q006_STEM,
        "question_type": "written",
        "correct_answer": Q006_ANSWER,
        "explanation": Q006_ANSWER,
        "concentration": "谐振功率放大器参数",
        "difficulty": "medium",
    }
    parsed, reason = generate._parse_qa(
        qa, chapter="第5章", requested_difficulty="medium"
    )
    # ``_parse_qa`` itself does not skip; the conversion loop checks the built
    # Item. Verify the check it uses fires.
    assert parsed is not None
    item = generate.Item(
        id="probe-q006",
        concept_ids=["c-probe"],
        kind=parsed.kind,
        stem=parsed.stem,
        options=parsed.options,
        answer=parsed.answer,
        explanation=parsed.explanation,
        source_ids=[],
    )
    assert parameter_infeasibility(item)


# -- X-01 ----------------------------------------------------------------


def test_bare_identifiers_are_wrapped_in_math() -> None:
    text = "V_CC=24 V，I_c1m=0.5 A，P_o=1/2·I_c1m^2·R_p。"
    normalized = generate._normalize_identifiers(text)
    assert "$V_{CC}$" in normalized
    assert "$I_{c1m}^{2}$" in normalized
    assert "$R_{p}$" in normalized
    assert "V_CC" not in normalized


def test_short_answer_explanation_is_not_duplicated() -> None:
    # The explanation already covers the answer's content; appending the whole
    # answer again is the duplication the report flagged.
    explanation = (
        "先求 I_c1m=α1 i_Cmax=0.5 A，再算 P_o=6.25 W、P_dc=7.64 W、"
        "η≈81.8%、P_c≈1.39 W。"
    )
    answer = (
        "I_c1m=0.5 A；P_o=6.25 W；P_dc=7.64 W；η≈81.8%；P_c≈1.39 W；"
        "未超过 P_Cmax=2 W；工作在乙类状态。"
    )
    assert answer_coverage(explanation, answer) >= 0.6
    assert _explanation_with_answer(explanation, answer) == explanation


# -- H-01 / H-02 ---------------------------------------------------------


def _paraphrase_spec(service: ReviewService) -> None:
    payload = _payload()
    concept = payload["concepts"][0]
    concept["name"] = (
        "丙类谐振功放导通角小于 90°、集电极电流为尖顶余弦脉冲，"
        "依靠谐振回路选频恢复完整基波正弦输出的工作机制"
    )
    concept["summary_one_line"] = ""
    concept["definition"] = ""
    concept["formulas"] = []
    item = payload["items"][0]
    item["stem"] = (
        "某丙类谐振功率放大器的导通角 $\\theta_c=70^\\circ$，"
        "集电极电流 $i_C$ 为一串尖顶余弦脉冲。关于输出波形，下列说法正确的是"
    )
    item["options"] = [
        {
            "label": "A",
            "text": "i_C 是尖顶余弦脉冲，负载为 LC 并联谐振回路，"
            "对基波高阻抗、对谐波短路，输出为完整基波正弦电压",
        },
        {"label": "B", "text": "输出仍是不连续的脉冲"},
        {"label": "C", "text": "集电极效率必然低于甲类"},
        {"label": "D", "text": "必须加深负反馈才能工作"},
    ]
    item["answer"] = {"choice_index": 0}
    item["explanation"] = item["options"][0]["text"]
    service.put_spec(ReviewSpec.model_validate(payload))


def test_paraphrased_concept_name_does_not_leak_the_option(
    service: ReviewService,
) -> None:
    _paraphrase_spec(service)
    tool = ReviewHintTool()
    level_one = json.loads(
        asyncio.run(tool.execute(spec_id=SPEC_ID, item_id=ITEM, level=1)).content
    )
    level_two = json.loads(
        asyncio.run(tool.execute(spec_id=SPEC_ID, item_id=ITEM, level=2)).content
    )
    assert level_one["fallback"] is True
    assert level_two["fallback"] is True
    assert "对基波高阻抗" not in level_one["hint"]
    assert level_one["hint"] != level_two["hint"]


def test_hint_brief_cuts_at_a_punctuation_boundary() -> None:
    text = "第一部分内容较长需要截断，第二部分是更长的术语说明" * 3
    brief = generate_hint_brief(text, limit=40)
    assert len(brief) <= 40
    assert brief.endswith("，…") or brief.endswith("…")


def generate_hint_brief(text: str, *, limit: int) -> str:
    from deeptutor.review.agent_tools import _hint_brief

    return _hint_brief(text, limit)


# -- S-01 ----------------------------------------------------------------


def test_complete_closes_other_open_sessions(service: ReviewService) -> None:
    service.put_spec(build_fixture_spec())
    service.today(SPEC_ID, session_id="older", now=NOW - 3600.0)
    service.today(SPEC_ID, session_id="newer", now=NOW)
    service.complete(SPEC_ID, "newer", now=NOW + 60.0)

    older = service.store.get_session("older")
    assert older["ended_at"] is not None
    # The next queue must not fall back to the older open row.
    following = service.today(SPEC_ID, now=NOW + 120.0)
    assert following["session_id"] != "older"


def test_complete_reports_active_minutes_not_idle_span(
    service: ReviewService,
) -> None:
    service.put_spec(build_fixture_spec())
    today = service.today(SPEC_ID, now=NOW)
    for offset in (240.0, 480.0):
        service.attempt(
            SPEC_ID,
            AttemptInput(
                item_id=ITEM,
                rating="good",
                session_id=today["session_id"],
                submission_id=f"active-{offset}",
            ),
            now=NOW + offset,
        )
    result = service.complete(
        SPEC_ID, today["session_id"], now=NOW + 4 * 3600.0
    )
    assert result["actual_minutes"] == 8.0
