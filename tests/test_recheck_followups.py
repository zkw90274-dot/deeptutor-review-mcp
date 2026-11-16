from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from deeptutor.review import generate
from deeptutor.review.agent_tools import ReviewExportLinkTool, ReviewHintTool
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.models import ReviewSpec
from deeptutor.review.qa import run_gates
from deeptutor.review.service import ReviewService

ITEM_ID = "item-resonance-choice"
BAD_STEM = "某谐振功率放大器 V_CC=24 V，V_cm=25 V，求输出功率。"
BAD_ANSWER = "输出功率 P_o=6.25 W。"


def _spec(**item_updates: Any) -> ReviewSpec:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    payload["items"][0].update(item_updates)
    return ReviewSpec.model_validate(payload)


def _bad_spec() -> ReviewSpec:
    return _spec(
        kind="short_answer", stem=BAD_STEM, options=[],
        answer={"text": BAD_ANSWER}, explanation=BAD_ANSWER,
    )


@pytest.mark.parametrize("method", ["get", "post"])
def test_failed_export_returns_gate_details(client: Any, service: ReviewService, method: str) -> None:
    service.store.put_spec(_bad_spec())
    response = getattr(client, method)(f"/api/review/{SPEC_ID}/export")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "quality_gate_failed"
    assert any(
        gate["gate"] == "G9-parameter-feasibility" and not gate["passed"]
        for gate in detail["quality"]["gates"]
    )


def test_export_link_reports_failed_quality_without_a_dead_url(service: ReviewService) -> None:
    service.store.put_spec(_bad_spec())
    result = asyncio.run(ReviewExportLinkTool().execute(spec_id=SPEC_ID))
    payload = json.loads(result.content)
    assert payload["error"] == "quality_gate_failed"
    assert payload["quality"]["passed"] is False
    assert "url" not in payload
    assert "regenerat" in payload["instruction"].lower()


@pytest.mark.parametrize("padding", [1, 25])
@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("name", [
    "丙类谐振功放的导通角小于90°、集电极电流为尖顶余弦脉冲并依靠谐振回路选频恢复完整基波正弦输出的工作机制",
    "谐振功放频谱与基波正弦电压的关系及其在模拟电路中若干相关元件参数背景条件的分析方法",
])
def test_long_choice_options_cannot_dilute_a_conclusion_hint(
    service: ReviewService, padding: int, level: int, name: str,
) -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    payload["concepts"][0].update(
        name=name,
        summary_one_line="", definition="", formulas=[],
    )
    item = payload["items"][0]
    item.update(
        stem="丙类谐振功放中电流断续而负载电压连续，其原因是什么？",
        options=[
            {"label": "A", "text": "谐振回路对各次谐波近似短路，故输出为完整的基波正弦电压。" + "补充背景说明与测量条件。" * padding},
            {"label": "B", "text": "晶体管保持线性，回路不必滤波。"},
            {"label": "C", "text": "扼流圈把电流脉冲补齐。"},
        ],
        answer={"choice_index": 0},
    )
    item["explanation"] = item["options"][0]["text"]
    service.put_spec(ReviewSpec.model_validate(payload))
    result = asyncio.run(ReviewHintTool().execute(spec_id=SPEC_ID, item_id=ITEM_ID, level=level))
    hint = json.loads(result.content)
    assert hint["fallback"] is True
    assert "基波正弦" not in hint["hint"]
    assert "选频恢复" not in hint["hint"]


@pytest.mark.parametrize("level", [1, 2])
def test_shared_written_terms_keep_useful_formula_direction(service: ReviewService, level: int) -> None:
    name = "根据电压利用系数 ξ、波形系数 g1(θc) 计算输出功率、直流功率与集电极效率"
    answer = "由波形系数 g1(θc) 与电压利用系数 ξ 求集电极效率 η=0.5ξg1(θc)=76.5%，输出功率为9.18 W。"
    payload = _spec(
        kind="short_answer", stem="已知 ξ=0.9，g1(θc)=1.7，求集电极效率和输出功率。",
        options=[], answer={"text": answer}, explanation=answer,
    ).model_dump(mode="json")
    payload["content_hash"] = ""
    payload["concepts"][0].update(name=name, summary_one_line="", definition="", formulas=[])
    service.put_spec(ReviewSpec.model_validate(payload))
    result = asyncio.run(ReviewHintTool().execute(spec_id=SPEC_ID, item_id=ITEM_ID, level=level))
    hint = json.loads(result.content)
    assert hint["fallback"] is False
    assert "g1(θc)" in hint["hint"]
    assert "76.5" not in hint["hint"]
    assert "9.18" not in hint["hint"]


@pytest.mark.parametrize("name,answer", [
    ("集电极效率为76.5%，输出功率为9.18 W的计算", "集电极效率为76.5%，输出功率为9.18 W；直流功率为12 W。"),
    ("通过改变电源电压实现调幅，集电极调幅应选择过压状态", "集电极调幅应选择过压状态，并合理选择负载。"),
    ("集电极调幅工作在过压状态", "集电极调幅工作在过压状态，因为此时输出幅度随电源电压近似线性变化。"),
    ("The amplifier operates in overvoltage mode", "The amplifier operates in overvoltage mode because its output varies with the supply."),
    ("计算输出功率为9.18瓦", "输出功率为9.18瓦，直流功率为12瓦。"),
])
def test_written_partial_results_are_not_hints(service: ReviewService, name: str, answer: str) -> None:
    payload = _spec(
        kind="short_answer", stem="计算集电极效率与输出功率，并说明调幅工作状态。",
        options=[], answer={"text": answer}, explanation=answer,
    ).model_dump(mode="json")
    payload["content_hash"] = ""
    payload["concepts"][0].update(name=name, summary_one_line="", definition="", formulas=[])
    service.put_spec(ReviewSpec.model_validate(payload))
    result = asyncio.run(ReviewHintTool().execute(spec_id=SPEC_ID, item_id=ITEM_ID, level=1))
    hint = json.loads(result.content)
    assert hint["fallback"] is True
    assert "76.5" not in hint["hint"]
    assert "过压状态" not in hint["hint"]


@pytest.mark.parametrize("stem,answer", [
    ("判断 V_CC=24 V，V_cm=25 V 这组参数是否可能。", "不可能，电压幅值超过电源电压，这组参数不可行。"),
    ("若 V_CC=24 V，V_cm=25 V，电路会怎样？", "该假设不成立，不能据此按正常谐振功放计算输出功率。"),
    ("Can V_CC=24 V, V_cm=25 V be physically feasible?", "These parameters are not feasible because the voltage swing exceeds the supply."),
    ("判断 V_CC=24 V，V_cm=25 V 是否合理。", "不合理：电压幅值超过电源电压，这组参数不可行。"),
    ("判断 V_CC=24 V，V_cm=25 V 是否合理。", "否，电压幅值超过电源电压，这组参数不可行。"),
    ("Are V_CC=24 V, V_cm=25 V physically feasible?", "These parameters are not feasible because the voltage swing exceeds the supply."),
])
def test_parameter_counterexamples_remain_valid(stem: str, answer: str) -> None:
    spec = _spec(kind="short_answer", stem=stem, options=[], answer={"text": answer}, explanation=answer)
    report = run_gates(spec)
    assert report.passed, report.model_dump(mode="json")


def test_feasibility_question_with_wrong_positive_answer_stays_blocked() -> None:
    answer = "这组参数合理，可以得到输出功率6.25 W。"
    spec = _spec(
        kind="short_answer", stem="V_CC=24 V、V_cm=25 V 是否合理？",
        options=[], answer={"text": answer}, explanation=answer,
    )
    assert run_gates(spec).passed is False


@pytest.mark.parametrize("answer", [
    "不能说这组参数不合理，实际可以得到输出功率6.25 W。",
    "不可能没有输出，这组参数合理，输出功率为6.25 W。",
    "The result is not possible to measure, but these parameters are feasible.",
])
def test_unrelated_negative_words_do_not_exempt_impossible_parameters(answer: str) -> None:
    spec = _spec(
        kind="short_answer", stem="V_CC=24 V，V_cm=25 V 是否合理？",
        options=[], answer={"text": answer}, explanation=answer,
    )
    assert run_gates(spec).passed is False


@pytest.mark.parametrize("separator", ["、", ", ", "; ", " and "])
def test_parameter_assignments_stop_at_the_next_value(separator: str) -> None:
    spec = _spec(
        kind="short_answer", stem=f"V_CC=24 V{separator}V_cm=25 V。求输出功率。",
        options=[], answer={"text": BAD_ANSWER}, explanation=BAD_ANSWER,
    )
    assert run_gates(spec).passed is False


def test_generation_skips_bad_item_and_persists_survivors(
    monkeypatch: pytest.MonkeyPatch, service: ReviewService,
) -> None:
    from deeptutor.agents.question.coordinator import AgentCoordinator

    monkeypatch.setattr(generate, "infer_chapters", lambda name: [generate.Chapter("第5章", 1)])
    pairs = [
        {"question": BAD_STEM, "correct_answer": BAD_ANSWER, "concentration": "不可行计算"},
        {"question": "定义谐振。", "correct_answer": "电抗相互抵消。", "concentration": "谐振"},
        {"question": "定义品质因数。", "correct_answer": "衡量选择性的量。", "concentration": "品质因数"},
    ]

    async def pipeline(self: Any, **kwargs: Any) -> dict[str, Any]:
        return {"success": True, "results": [
            {"qa_pair": {**pair, "question_type": "written", "explanation": pair["correct_answer"]}}
            for pair in pairs
        ]}

    monkeypatch.setattr(AgentCoordinator, "generate_from_topic", pipeline)
    result = generate.generate_spec_from_kb(kb_name="hf", chapter="第5章", num_questions=3)
    assert result.skipped == 1
    assert result.qa_count == 2
    assert any("V_CC" in warning and "skipped" in warning for warning in result.warnings)
    assert result.spec is not None
    assert len(result.spec.concepts) == 2
    stored = service.put_spec(result.spec)
    assert stored["quality"]["passed"] is True
    html = service.export(result.spec.id)
    assert "不可行计算" not in html
    assert "定义谐振" in html
    assert "定义品质因数" in html
