from __future__ import annotations

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest

from deeptutor.review import generate, jobs
from deeptutor.review.agent_tools import ReviewGenerateTool, ReviewHintTool
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.models import ReviewSpec
from deeptutor.review.qa import answer_coverage, parameter_infeasibility, run_gates
from deeptutor.review.service import ReviewService


@pytest.mark.parametrize("answer,explanation", [("是", "答案是是"), ("对", "判断结果为对"), ("A", "正确答案为A"), ("5", "答案是5"), ("√", "答案为√")])
def test_short_literal_answers_survive_ingestion(service: ReviewService, answer: str, explanation: str) -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    payload["items"][0].update(kind="short_answer", options=[], answer={"text": answer}, explanation=explanation)
    result = service.put_spec(ReviewSpec.model_validate(payload))
    assert result["quality"]["passed"]
    assert answer_coverage(explanation, answer) == 1.0


@pytest.mark.parametrize("answer,correct,wrong", [
    ("9007199254740993", "计算结果为9007199254740993。", "计算结果为9007199254740992。"),
    ("0.1234567890123456789", "结果为0.1234567890123456789。", "结果为0.12345678901234568。"),
])
def test_text_numeric_answers_keep_decimal_precision(answer: str, correct: str, wrong: str) -> None:
    assert answer_coverage(correct, answer) == 1.0
    assert answer_coverage(wrong, answer) == 0.0


@pytest.mark.parametrize("explanation", ["输出不会导致增大。", "输出不会明显增大。", "该假设并不成立，不能说明增大。"])
def test_negative_clauses_cannot_claim_positive_answers(explanation: str) -> None:
    assert answer_coverage(explanation, "增大") < 0.6


@pytest.mark.parametrize("explanation", ["输出不得不增大。", "输出无不增大。", "输出不可不增大。", "输出未尝不增大。", "输出不能不增大。", "电压不变，电流增大。"])
def test_double_negatives_and_other_clauses_keep_positive_meaning(explanation: str) -> None:
    assert answer_coverage(explanation, "增大") >= 0.6


@pytest.mark.parametrize("text,value", [("输出为0.50 A", 0.5), ("输出为0.500 W", 0.5), ("输出为1.20 V", 1.2), ("输出为5e-1 A", 0.5)])
def test_numeric_g4_compares_values_not_trailing_zero_spelling(text: str, value: float) -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    payload["items"][0].update(kind="numeric", options=[], answer={"value": value}, explanation=text)
    report = run_gates(ReviewSpec.model_validate(payload))
    assert next(gate for gate in report.gates if gate.gate == "G4-answer-key").passed


@pytest.mark.parametrize("text,expected", [("$f_s<f<f_p$", r"$f_s\lt{}f\lt{}f_p$"), ("$V_{CC}>V_{cm}$", r"$V_{CC}\gt{}V_{cm}$"), (r"$x\lt{}y$", r"$x\lt{}y$")])
def test_generated_comparison_commands_have_tex_boundaries(text: str, expected: str) -> None:
    actual = generate._escape_math(text)
    assert actual == expected
    assert generate._escape_math(actual) == actual


def test_generation_pipeline_keeps_comparison_macros_renderable(service: ReviewService, monkeypatch) -> None:
    from deeptutor.agents.question.coordinator import AgentCoordinator

    monkeypatch.setattr(generate, "infer_chapters", lambda name: [generate.Chapter("第6章", 1)])

    async def fake_pipeline(self, **kwargs):
        return {"success": True, "results": [{"qa_pair": {
            "question_type": "written", "question": "分析 $f_s<f<f_p$ 与 $V_{CC}>V_{cm}$。",
            "correct_answer": "该频段呈感性。", "explanation": "该频段呈感性。", "concentration": "频段判别",
        }}]}

    monkeypatch.setattr(AgentCoordinator, "generate_from_topic", fake_pipeline)
    result = generate.generate_spec_from_kb(kb_name="hf", chapter="第6章", num_questions=1)
    assert result.spec is not None
    assert r"\lt{}f" in result.spec.items[0].stem
    assert r"\gt{}V" in result.spec.items[0].stem
    stored = service.put_spec(result.spec)
    assert stored["quality"]["passed"]
    html = service.export(result.spec.id)
    assert r"\\ltf" not in html and r"\\gtV" not in html


@pytest.mark.parametrize("name", [
    "叠加原理在非线性电路中失效的原因，以及非线性电路必然产生新频率这一结论（含 i=kv² 时多出相乘项 2kv₁v₂ 的说明）",
    "反馈型振荡器起振条件、平衡条件与稳定条件的区分（|A₀F|>1 与 AF=1、相位条件 φ_A+φ_F=2nπ 的含义）",
])
@pytest.mark.parametrize("field,level", [("name", 1), ("name", 2), ("summary_one_line", 2), ("definition", 2)])
def test_conclusion_formulas_are_not_choice_hints(service: ReviewService, name: str, field: str, level: int) -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    concept = payload["concepts"][0]
    concept.update(name="相关考点", summary_one_line="", definition="", formulas=[])
    concept[field] = name
    item = payload["items"][0]
    if "叠加" in name:
        item.update(stem=r"已知 $i=kv^2$，输入两频率不同的信号。下列正确的是？", options=[
            {"label": "A", "text": r"不满足叠加原理，输出多出 $2kv_{1}v_{2}$，并由此产生 $\omega_{1}\pm\omega_{2}$ 的新频率"},
            {"label": "B", "text": r"不满足叠加原理，但多出的 $2kv_{1}v_{2}$ 仅使幅度变大，不产生新频率"},
            {"label": "C", "text": r"满足叠加原理，输出为 $kv_{1}^{2}+kv_{2}^{2}$，不产生任何新频率"},
            {"label": "D", "text": "满足叠加原理，因为平方项可以先各自平方再相加，只含原频率"},
        ])
    else:
        item.update(stem="下列起振与平衡条件的配对正确的是？", options=[
            {"label": "A", "text": r"起振要求 $\vert{}A_0F\vert{}\gt1$ 且 $\varphi_A+\varphi_F=2n\pi$；达到平衡后应满足 $AF=1$ 且 $\varphi_A+\varphi_F=2n\pi$"},
            {"label": "B", "text": r"起振要求 $AF=1$ 且 $\varphi_A+\varphi_F=(2n+1)\pi$；达到平衡后应满足 $\vert{}A_0F\vert{}\gt1$"},
            {"label": "C", "text": r"起振与平衡都要求 $\vert{}A_0F\vert{}\gt1$，相位条件均为 $\varphi_A+\varphi_F=(2n+1)\pi$"},
            {"label": "D", "text": r"起振要求 $\vert{}A_0F\vert{}\lt1$ 且 $\varphi_A+\varphi_F=2n\pi$；达到平衡后应满足 $AF=1$ 且为负反馈"},
        ])
    item["answer"] = {"choice_index": 0}
    item["explanation"] = item["options"][0]["text"]
    service.put_spec(ReviewSpec.model_validate(payload))
    result = json.loads(asyncio.run(ReviewHintTool().execute(spec_id=SPEC_ID, item_id=item["id"], level=level)).content)
    assert result["fallback"] is True
    assert "2kv" not in result["hint"] and "AF=1" not in result["hint"]


def test_unrelated_conclusion_words_do_not_force_hint_fallback(service: ReviewService) -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    payload["concepts"][0].update(name="比较两个电阻的大小是否小于阈值", summary_one_line="", definition="", formulas=[])
    item = payload["items"][0]
    item.update(stem="分析串联电路的总电流变化。", options=[{"label": "A", "text": "增大"}, {"label": "B", "text": "减小"}], answer={"choice_index": 0}, explanation="总电流增大。")
    service.put_spec(ReviewSpec.model_validate(payload))
    result = json.loads(asyncio.run(ReviewHintTool().execute(spec_id=SPEC_ID, item_id=item["id"])).content)
    assert result["fallback"] is False


@pytest.mark.parametrize("stem", ["差动放大器 V_CC=5 V，V_cm=12 V，判断是否可行。", "某电路 E_c=5 V，V_cm=12 V，判断是否可行。"])
def test_g9_does_not_guess_unrelated_voltage_semantics(stem: str) -> None:
    assert parameter_infeasibility(SimpleNamespace(stem=stem, explanation="", answer=SimpleNamespace(text=""), options=[])) is None


def test_g9_understands_nested_math_wrappers() -> None:
    item = SimpleNamespace(stem=r"某谐振功放 $\mathrm{V_{CC}}=5 V$，$\mathrm{V_{cm}}=9 V$。", explanation="", answer=SimpleNamespace(text=""), options=[])
    assert parameter_infeasibility(item)


def test_scaled_product_is_not_itself_the_voltage_amplitude() -> None:
    item = SimpleNamespace(stem="谐振功放 V_CC=24 V，且 2×I_c1m×R_p=30 V。", explanation="", answer=SimpleNamespace(text=""), options=[])
    assert parameter_infeasibility(item) is None


@pytest.mark.parametrize("supply,blocked", [(12, True), (20, False)])
def test_explicit_twice_product_is_converted_to_amplitude(supply: int, blocked: bool) -> None:
    item = SimpleNamespace(stem=f"谐振功放 V_CC={supply} V，且 2×I_c1m×R_p=30 V。", explanation="", answer=SimpleNamespace(text=""), options=[])
    reason = parameter_infeasibility(item)
    assert bool(reason) is blocked
    if blocked:
        assert "V_cm=15" in reason


def test_failed_generation_keeps_skip_diagnostics(service: ReviewService) -> None:
    job = jobs.create_job(kb="hf", chapter="第5章", num_questions=5, question_types=["choice"], difficulty="")
    job.update(status="failed", error="no usable questions", skipped=5, warnings=["question skipped: invalid option answer"])
    jobs.save_job(job)
    result = json.loads(asyncio.run(ReviewGenerateTool().execute(kb="hf", job_id=job["job_id"])).content)
    assert result["skipped"] == 5
    assert result["warnings"] == job["warnings"]


def test_all_skipped_runner_and_poll_preserve_the_same_diagnostics(service: ReviewService, monkeypatch) -> None:
    job = jobs.create_job(kb="hf", chapter="第5章", num_questions=5, question_types=["choice"], difficulty="")
    monkeypatch.setattr(generate, "generate_spec_from_kb", lambda **kwargs: SimpleNamespace(spec=None, qa_count=0, skipped=5, warnings=["invalid answer option skipped"]))
    jobs._run_generation(job)
    stored = jobs.load_job(job["job_id"])
    assert stored["status"] == "failed"
    assert stored["qa_count"] == 0 and stored["skipped"] == 5
    result = json.loads(asyncio.run(ReviewGenerateTool().execute(kb="hf", job_id=job["job_id"])).content)
    assert result["warnings"] == stored["warnings"]
    assert result["skipped"] == 5


def test_locked_today_returns_service_unavailable(client, service: ReviewService) -> None:
    service.put_spec(build_fixture_spec())
    with service.store._connect() as connection:
        path = connection.execute("PRAGMA database_list").fetchone()[2]
    blocker = sqlite3.connect(path)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        response = client.get(f"/api/review/{SPEC_ID}/today")
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "storage_unavailable"
        assert "OperationalError" not in response.text
    finally:
        blocker.rollback()
        blocker.close()


def test_ipv6_local_frontend_is_allowed_and_lan_is_configurable(client, monkeypatch) -> None:
    assert client.get("/api/review/specs", headers={"Origin": "http://[::1]:3782"}).status_code == 200
    monkeypatch.setenv("CORS_ORIGINS", "http://192.168.1.5:3782")
    assert client.get("/api/review/specs", headers={"Origin": "http://192.168.1.5:3782"}).status_code == 200
    assert client.get("/api/review/specs", headers={"Origin": "https://evil.example"}).status_code == 403
