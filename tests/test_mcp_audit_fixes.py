from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from starlette.testclient import TestClient

from deeptutor.review import jobs
from deeptutor.review.agent_tools import ReviewAttemptTool, ReviewGenerateTool, ReviewHintTool
from deeptutor.review.fixture import SPEC_ID, build_fixture_spec
from deeptutor.review.mcp_server import build_server, create_http_app
from deeptutor.review.models import ReviewSpec
from deeptutor.review.service import ReviewService

ITEM_ID = "item-resonance-choice"
HEADERS = {"Accept": "application/json, text/event-stream"}
INITIALIZE = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "audit", "version": "1"}},
}


@pytest.mark.parametrize("host,origin,expected", [
    ("127.0.0.1:8767", None, 200),
    ("localhost:8767", "http://localhost:8767", 200),
    ("[::1]:8767", "http://[::1]:8767", 200),
    ("rebind.attacker.invalid", "http://rebind.attacker.invalid", 421),
    ("127.0.0.1:8767", "https://attacker.invalid", 403),
    ("127.0.0.1:8767.attacker.invalid", None, 421),
    ("127.0.0.1:9999", None, 421),
])
def test_http_transport_checks_authority_and_origin(host: str, origin: str | None, expected: int) -> None:
    headers = {**HEADERS, "Host": host}
    if origin is not None:
        headers["Origin"] = origin
    with TestClient(create_http_app(build_server())) as client:
        response = client.post("/mcp", json=INITIALIZE, headers=headers)
    assert response.status_code == expected, response.text


def test_explicit_http_deployment_authorities_work() -> None:
    app = create_http_app(
        build_server(), host="0.0.0.0", port=8877,
        allowed_hosts=["review.example.com"], allowed_origins=["https://study.example.com"],
        auth_token="test-token",
    )
    with TestClient(app) as client:
        response = client.post("/mcp", json=INITIALIZE, headers={
            **HEADERS, "Host": "review.example.com", "Origin": "https://study.example.com",
            "Authorization": "Bearer test-token",
        })
    assert response.status_code == 200


@pytest.mark.parametrize("host,origin", [("localhost", "http://localhost"), ("localhost:80", "http://localhost:80")])
def test_http_default_port_accepts_standard_authorities(host: str, origin: str) -> None:
    with TestClient(create_http_app(build_server(), port=80)) as client:
        response = client.post("/mcp", json=INITIALIZE, headers={**HEADERS, "Host": host, "Origin": origin})
    assert response.status_code == 200


@pytest.mark.parametrize("field", ["name", "summary_one_line", "definition"])
@pytest.mark.parametrize("hint,kind,answer", [
    ("计算得到η=0.765与P_o=9.18", "short_answer", "依据给定波形参数计算得到η=0.765，从功率守恒得P_o=9.18并满足额定管耗限制。"),
    (r"输出效率为76.5\%和输出功率9.18\mathrm{W}", "short_answer", r"依据公式得到输出效率为76.5\%，交流输出功率为9.18\mathrm{W}且小于直流输入功率。"),
    ("计算得到效率η=.765", "short_answer", "依据公式得到效率η=0.765，满足额定限制。"),
    ("计算得到输出功率9.18e0 W", "short_answer", "输出功率为9.18 W，同时满足管耗限制。"),
    ("该题应选A项", "single_choice", ""),
    ("The correct choice is A", "single_choice", ""),
    ("正确答案是Ａ", "single_choice", ""),
    ("该题答案为选项A", "single_choice", ""),
    (r"计算η=\frac{153}{200}", "short_answer", "根据已知量计算得到η=0.765，满足限制。"),
    ("计算η=0.765", "short_answer", r"根据已知量得到η=\frac{153}{200}，满足限制。"),
])
def test_hint_does_not_reveal_results_or_choice_labels(
    service: ReviewService, field: str, hint: str, kind: str, answer: str,
) -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    concept = payload["concepts"][0]
    concept.update(name="计算与判别方法", summary_one_line="", definition="", formulas=[])
    concept[field] = hint
    item = payload["items"][0]
    item["stem"] = "根据题目条件计算所求量，并说明判断理由。"
    if kind == "short_answer":
        item.update(kind=kind, options=[], answer={"text": answer}, explanation=answer)
    else:
        item["answer"] = {"choice_index": 0}
        item["explanation"] = item["options"][0]["text"]
    service.put_spec(ReviewSpec.model_validate(payload))
    level = 2 if field in {"definition", "summary_one_line"} else 1
    result = asyncio.run(ReviewHintTool().execute(spec_id=SPEC_ID, item_id=ITEM_ID, level=level))
    body = json.loads(result.content)
    assert body["fallback"] is True, body
    assert hint not in body["hint"]


@pytest.mark.parametrize("name", [
    "按 η=0.5ξg1(θc) 分析效率关系",
    "根据题干 ξ=0.9 和 g1=1.7 分析功率与效率",
])
def test_hint_keeps_coefficients_and_given_values(service: ReviewService, name: str) -> None:
    answer = "由 η=0.5ξg1(θc) 得 η=0.765，输出功率 P_o=9.18 W。"
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    payload["concepts"][0].update(name=name, summary_one_line="", definition="", formulas=[])
    payload["items"][0].update(
        kind="short_answer", stem="已知 ξ=0.9，g1=1.7，试求效率。", options=[],
        answer={"text": answer}, explanation=answer,
    )
    service.put_spec(ReviewSpec.model_validate(payload))
    result = asyncio.run(ReviewHintTool().execute(spec_id=SPEC_ID, item_id=ITEM_ID, level=1))
    assert json.loads(result.content)["fallback"] is False


@pytest.mark.parametrize("name", [
    "按η=0.5ξg1分析效率关系",
    r"按$\eta=0.5\cdot\xi g_1$分析效率关系",
    r"按$\eta=\frac{1}{2}\cdot\xi g_1$分析效率关系",
    r"按$\eta=0.5\,\xi g_1$分析效率关系",
])
def test_formula_coefficient_matching_a_result_is_still_a_direction(service: ReviewService, name: str) -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    payload["concepts"][0].update(name=name, summary_one_line="", definition="", formulas=[])
    answer = "由η=0.5ξg1得η=0.5，满足限制。"
    payload["items"][0].update(kind="short_answer", stem="已知ξ=0.8，g1=1.25，求效率。", options=[], answer={"text": answer}, explanation=answer)
    service.put_spec(ReviewSpec.model_validate(payload))
    result = asyncio.run(ReviewHintTool().execute(spec_id=SPEC_ID, item_id=ITEM_ID, level=1))
    assert json.loads(result.content)["fallback"] is False


def test_completed_session_is_a_recoverable_conflict(service: ReviewService) -> None:
    service.put_spec(build_fixture_spec())
    queue = service.today(SPEC_ID, include_draft=True)
    service.complete(SPEC_ID, queue["session_id"])
    result = asyncio.run(ReviewAttemptTool().execute(
        spec_id=SPEC_ID, item_id=ITEM_ID, session_id=queue["session_id"],
        submission_id="after-complete", rating="good",
    ))
    assert result.success is True
    body = json.loads(result.content)
    assert body["error"] == "session_conflict"
    assert "review_today" in body["instruction"]
    assert not service.store.events_for_spec(SPEC_ID)
    following = service.today(SPEC_ID)
    result = asyncio.run(ReviewAttemptTool().execute(
        spec_id=SPEC_ID, item_id=ITEM_ID, session_id=following["session_id"],
        submission_id="new-session", rating="good",
    ))
    assert result.success is True and json.loads(result.content)["created"]


@pytest.mark.parametrize("updates", [
    {"spec_id": None}, {"spec_id": ""}, {"qa_passed": False}, {"qa_passed": "true"},
    {"qa_count": 0}, {"qa_count": True}, {"counts": None},
    {"counts": {"items": 5}}, {"warnings": "corrupt"}, {"error": "failed to save"},
    {"spec_id": "-bad"}, {"spec_id": "fixture."},
])
def test_inconsistent_done_job_never_claims_success(
    service: ReviewService, updates: dict[str, Any],
) -> None:
    job = jobs.create_job(kb="hf", chapter="第5章", num_questions=5, question_types=["choice"], difficulty="")
    job.update(status="done", spec_id="fixture-test", qa_passed=True, qa_count=6, skipped=0,
               counts={"items": 6, "concepts": 2}, warnings=[], error=None)
    job.update(updates)
    jobs.save_job(job)
    result = asyncio.run(ReviewGenerateTool().execute(kb="hf", job_id=job["job_id"]))
    body = json.loads(result.content)
    assert body.get("error") == "invalid_job_record", body
    assert "review_specs" in body["instruction"]
    assert "saved and passed" not in body["instruction"]


def test_done_job_accepts_a_valid_dotted_spec_id(service: ReviewService) -> None:
    job = jobs.create_job(kb="hf", chapter="第5章", num_questions=5, question_types=["choice"], difficulty="")
    job.update(status="done", spec_id="fixture.v1", qa_passed=True, qa_count=6, skipped=0,
               counts={"items": 6}, warnings=[], error=None)
    jobs.save_job(job)
    result = asyncio.run(ReviewGenerateTool().execute(kb="hf", job_id=job["job_id"]))
    body = json.loads(result.content)
    assert not body.get("error")
    assert body["spec_id"] == "fixture.v1"
