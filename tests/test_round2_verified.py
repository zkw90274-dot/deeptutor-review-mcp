from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from deeptutor.review.agent_tools import ReviewHintTool
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.models import ReviewSpec
from deeptutor.review.qa import answer_coverage, parameter_infeasibility, run_gates
from deeptutor.review.service import ReviewService
from deeptutor.review.tikz import TikzRenderError, _prepare_source, render_tikz_document


@pytest.mark.parametrize("stem", [
    r"$V_{\mathrm{CC}}=24$ V，$V_{cm}=25$ V",
    r"V_{\text{CC}}=24 V，V_{cm}=25 V",
    "谐振功放 E_c=24 V，V_cm=25 V", "V_CC＝24 V，V_cm＝25 V",
    "V_CC=24000 mV，V_cm=25 V", "V_CC=12V，V_cm=13V",
    "谐振功放电压利用系数 ξ=1.2。",
    "谐振功放 V_CC=24 V，V_cm=0.5×50=25 V。",
    "谐振功放 V_CC=24 V，V_cm=0.5*50=25 V。",
    "谐振功放 V_CC=24 V，V_cm=1/2*50=25 V。",
])
def test_g9_rejects_equivalent_impossible_parameters(stem: str) -> None:
    item = SimpleNamespace(stem=stem, explanation="计算功率。", answer=SimpleNamespace(text="计算功率。"), options=[])
    assert parameter_infeasibility(item)


@pytest.mark.parametrize("stem", [
    "二阶系统阻尼比ξ=1.2，分析响应。", "归一化频率ξ=1.5，分析频率响应。",
    "双电源放大器V_CC=-15 V，V_cm=10 V。", "差分对 V_CC=5 V，V_cm=12 V。",
    "谐振功放 V_CC=24 V，V_cm=21600 mV。",
])
def test_g9_does_not_reject_unrelated_semantics_or_valid_units(stem: str) -> None:
    item = SimpleNamespace(stem=stem, explanation="根据题设分析。", answer=SimpleNamespace(text="根据题设分析。"), options=[])
    assert parameter_infeasibility(item) is None


@pytest.mark.parametrize("answer,explanation", [
    ("可以", "不可以，该假设不成立。"), ("增大", "减小，不会增大。"),
    ("√", "输出与输入无关。"), ("—", "结果不确定。"),
])
def test_g4_rejects_negative_or_vacuous_agreement(answer: str, explanation: str) -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    payload["items"][0].update(kind="short_answer", options=[], answer={"text": answer}, explanation=explanation)
    spec = ReviewSpec.model_validate(payload)
    gate = next(gate for gate in run_gates(spec).gates if gate.gate == "G4-answer-key")
    assert gate.passed is False
    assert answer_coverage(explanation, answer) < 0.6


@pytest.mark.parametrize("answer,explanation", [("可以", "可以，该条件成立。"), ("不可以", "不可以，该条件不成立。"), ("增大", "该增益会增大。"), ("√", "判定为√。"), ("可以", "该条件不是不可以满足。"), ("增大", "该量并非不会增大。")])
def test_g4_keeps_actual_positive_agreement(answer: str, explanation: str) -> None:
    assert answer_coverage(explanation, answer) >= 0.6


@pytest.mark.parametrize("field", ["name", "summary_one_line", "definition"])
@pytest.mark.parametrize("hint,answer,stem", [
    ("此时三极管处于截止、失真状态", "截止失真", "判断此时的失真类型。"),
    ("输出电流为500 mA", "0.5 A", "求输出电流。"),
    ("中频干扰 fn=fi 与镜像干扰 fn=f0+fi=fs+2fi 的频率关系", "中频干扰 fn=fi；镜像干扰 fn=f0+fi=fs+2fi。", "请推导中频干扰和镜像干扰的频率关系。"),
])
def test_final_hint_gate_covers_all_authored_paths(service: ReviewService, field: str, hint: str, answer: str, stem: str) -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    payload["concepts"][0].update(name="相关原理", summary_one_line="", definition="", formulas=[])
    payload["concepts"][0][field] = hint
    payload["items"][0].update(kind="short_answer", stem=stem, options=[], answer={"text": answer}, explanation=answer)
    service.put_spec(ReviewSpec.model_validate(payload))
    result = asyncio.run(ReviewHintTool().execute(spec_id=SPEC_ID, item_id="item-resonance-choice", level=1 if field == "name" else 2))
    assert json.loads(result.content)["fallback"] is True


@pytest.mark.parametrize("command", ["mdfivesum", "filesize", "filemoddate"])
def test_tikz_rejects_file_metadata_primitives(command: str) -> None:
    with pytest.raises(TikzRenderError, match="forbidden"):
        _prepare_source("\\" + command + r"{C:/Windows/win.ini}")


@pytest.mark.parametrize("command", ["mdfivesum", "filesize", "filemoddate"])
def test_tikz_metadata_is_rejected_before_toolchain_access(command: str, monkeypatch) -> None:
    from deeptutor.review import tikz

    def forbidden_toolchain():
        pytest.fail("unsafe metadata reached the renderer")

    monkeypatch.setattr(tikz, "_resolve_tools", forbidden_toolchain)
    with pytest.raises(TikzRenderError, match="forbidden"):
        render_tikz_document("\\" + command + r"{C:/Windows/win.ini}")


def test_review_rejects_untrusted_browser_origin(client) -> None:
    response = client.get("/api/review/specs", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403
    response = client.get("/api/review/specs", headers={"Origin": "http://localhost:3782"})
    assert response.status_code == 200


def test_review_rejects_untrusted_put_before_storage(client, service: ReviewService) -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    response = client.put(f"/api/review/specs/{SPEC_ID}", json=payload, headers={"Origin": "https://evil.example"})
    assert response.status_code == 403
    assert service.store.spec_count() == 0


def test_utf16_nonfinite_body_is_rejected_at_transport_edge(client) -> None:
    response = client.put("/api/review/specs/utf16", content='{"value": NaN}'.encode("utf-16"), headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_number"
