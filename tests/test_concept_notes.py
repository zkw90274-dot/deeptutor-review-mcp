"""Concept-note enrichment: parsing, gating and the best-effort contract."""

from __future__ import annotations

import json

import pytest

from deeptutor.review.concept_notes import enrich_concepts
from deeptutor.review.generate import _clean_text
from deeptutor.review.models import Answer, Concept, Item, Option


def _concept(name: str = "谐振频率", concept_id: str = "c-resonance") -> Concept:
    return Concept(id=concept_id, name=name, source_ids=["src"])


def _numeric_item(concept_id: str = "c-resonance") -> Item:
    return Item(
        id="i1",
        concept_ids=[concept_id],
        kind="numeric",
        stem="某串联谐振回路 L = 100 μH，C = 100 pF，求谐振频率 f0（MHz）。",
        answer=Answer(value=1.592, tolerance=0.01),
        explanation="f0 = 1/(2π√(LC)) ≈ 1.592 MHz。",
        source_ids=["src"],
    )


def _choice_item(concept_id: str = "c-resonance") -> Item:
    return Item(
        id="i2",
        concept_ids=[concept_id],
        kind="single_choice",
        stem="串联谐振时回路阻抗为什么最小？",
        options=[
            Option(label="A", text="感抗与容抗相互抵消"),
            Option(label="B", text="电阻被短路"),
        ],
        answer=Answer(choice_index=0),
        explanation="谐振时 XL = XC。",
        source_ids=["src"],
    )


def _run(
    concepts: list[Concept],
    reply: str | None = None,
    *,
    error: Exception | None = None,
    items: list[Item] | None = None,
    extras: dict | None = None,
) -> tuple[int, list[str], list[str]]:
    warnings: list[str] = []
    prompts: list[str] = []
    concept_items = items if items is not None else [_numeric_item()]

    def complete(prompt: str, system_prompt: str) -> str:
        prompts.append(prompt)
        if error is not None:
            raise error
        assert reply is not None
        return reply

    enriched = enrich_concepts(
        concepts,
        {"c-resonance": concept_items},
        extras=extras,
        chapter="第2章 谐振",
        complete=complete,
        clean=_clean_text,
        warn=warnings.append,
    )
    return enriched, warnings, prompts


def test_notes_fill_summary_and_definition() -> None:
    concept = _concept()
    reply = json.dumps(
        [
            {
                "name": "谐振频率",
                "summary": "回路的固有频率由电感与电容共同决定。",
                "definition": "串联回路的固有角频率为 $\\omega_0 = 1/\\sqrt{LC}$，"
                "对应频率随 L 或 C 增大而降低。",
            }
        ],
        ensure_ascii=False,
    )

    enriched, warnings, prompts = _run([concept], reply)

    assert enriched == 1
    assert warnings == []
    assert concept.summary_one_line == "回路的固有频率由电感与电容共同决定。"
    assert "\\omega_0" in concept.definition
    # One batched call carries the concept and its question.
    assert len(prompts) == 1
    assert "谐振频率" in prompts[0]
    assert "求谐振频率" in prompts[0]


def test_fenced_reply_and_unknown_name() -> None:
    concept = _concept()
    reply = (
        "```json\n"
        + json.dumps(
            [
                {"name": "不存在的概念", "summary": "x", "definition": "y"},
                {"name": "谐振频率", "summary": "由电感与电容共同决定。", "definition": ""},
            ],
            ensure_ascii=False,
        )
        + "\n```"
    )

    enriched, warnings, _ = _run([concept], reply)

    assert enriched == 1
    assert warnings == []
    assert concept.summary_one_line == "由电感与电容共同决定。"
    assert concept.definition == ""


def test_malformed_reply_only_warns() -> None:
    concept = _concept()

    enriched, warnings, _ = _run([concept], "这不是 JSON")

    assert enriched == 0
    assert concept.summary_one_line == ""
    assert concept.definition == ""
    assert any("概念摘要生成失败" in message for message in warnings)


def test_completion_error_only_warns() -> None:
    concept = _concept()

    enriched, warnings, _ = _run([concept], error=TimeoutError("slow"))

    assert enriched == 0
    assert concept.definition == ""
    assert any("TimeoutError" in message for message in warnings)


def test_note_that_restates_the_option_is_dropped() -> None:
    concept = _concept("串联谐振")
    reply = json.dumps(
        [
            {
                "name": "串联谐振",
                "summary": "谐振时回路呈电阻性。",
                # Contains the correct option verbatim: the hint would leak.
                "definition": "谐振发生时感抗与容抗相互抵消，所以阻抗最小。",
            }
        ],
        ensure_ascii=False,
    )

    enriched, warnings, _ = _run(
        [concept], reply, items=[_choice_item()]
    )

    assert enriched == 1
    assert concept.summary_one_line == "谐振时回路呈电阻性。"
    assert concept.definition == ""
    assert any("定义可能泄露答案" in message for message in warnings)


def test_raw_comparison_inside_math_is_escaped() -> None:
    concept = _concept("频率关系")
    reply = json.dumps(
        [
            {
                "name": "频率关系",
                "summary": "$f_s<f_p$ 时呈现不同特性。",
                "definition": "$f_s<f_p$ 时容性占优，$f_s>f_p$ 时感性占优。",
            }
        ],
        ensure_ascii=False,
    )

    enriched, _, _ = _run([concept], reply)

    assert enriched == 1
    assert "\\lt{}" in concept.summary_one_line
    assert "<" not in concept.summary_one_line.replace("\\lt{}", "")
    assert "\\gt{}" in concept.definition


def test_identical_summary_and_definition_keep_one_rung() -> None:
    concept = _concept()
    same = "谐振频率由电感与电容决定。"
    reply = json.dumps(
        [{"name": "谐振频率", "summary": same, "definition": same}],
        ensure_ascii=False,
    )

    enriched, _, _ = _run([concept], reply)

    assert enriched == 1
    assert concept.summary_one_line == ""
    assert concept.definition == same


def test_reply_in_another_schema_warns_instead_of_silently_skipping() -> None:
    concept = _concept()
    reply = json.dumps(
        [{"knowledge_point": "谐振频率", "answer": "1.592 MHz"}],
        ensure_ascii=False,
    )

    enriched, warnings, _ = _run([concept], reply)

    assert enriched == 0
    assert any("格式不符" in message for message in warnings)


def test_empty_reply_warns() -> None:
    concept = _concept()

    enriched, warnings, _ = _run([concept], "[]")

    assert enriched == 0
    assert any("概念摘要为空" in message for message in warnings)


def test_notes_collect_symbols_pitfalls_and_self_check() -> None:
    concept = _concept()
    extras: dict = {}
    reply = json.dumps(
        [
            {
                "name": "谐振频率",
                "summary": "",
                "definition": "",
                "symbols": "f0 — 谐振频率 / Hz；L — 电感 / H",
                "pitfalls": [
                    "单位混用（μH 与 H）",
                    "漏掉 1/(2π)",
                    "把 f0 与 ω0 混淆",
                    "第四条应被截断",
                ],
                "self_check": "把 L、C 换算到 SI 后重新计算，并核对数量级。",
                "assumptions": [
                    "忽略电感与电容的寄生参数",
                    "信号为小信号",
                    "第四条应被截断",
                ],
            }
        ],
        ensure_ascii=False,
    )

    enriched, warnings, _ = _run([concept], reply, extras=extras)

    assert enriched == 1
    note = extras[concept.id]
    assert "f0" in note["symbols"] and "Hz" in note["symbols"]
    assert note["pitfalls"] == [
        "单位混用（μH 与 H）",
        "漏掉 1/(2π)",
        "把 f0 与 ω0 混淆",
    ]
    assert "数量级" in note["self_check"]
    assert note["assumptions"] == [
        "忽略电感与电容的寄生参数",
        "信号为小信号",
        "第四条应被截断",
    ]
    assert warnings == []


def test_large_packs_are_chunked_before_the_expanded_schema() -> None:
    concepts = [
        _concept(name=f"概念{index}", concept_id=f"c{index}") for index in range(5)
    ]
    items = {concept.id: [_numeric_item(concept.id)] for concept in concepts}
    calls: list[str] = []

    def complete(prompt: str, system_prompt: str) -> str:
        calls.append(prompt)
        return "[]"

    enrich_concepts(
        concepts,
        items,
        chapter="第1章",
        complete=complete,
        clean=_clean_text,
        warn=lambda _message: None,
    )

    # _CHUNK is 2: five concepts must be three replies, not one oversized
    # reply whose truncated tail would be invalid JSON.
    assert len(calls) == 3


def test_notes_completion_retries_an_empty_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deeptutor.review.jobs as jobs
    from deeptutor.services.llm import client as llm_client

    calls: list[str] = []

    class FakeClient:
        def complete_sync(self, prompt: str, system_prompt: str) -> str:
            calls.append(prompt)
            return "" if len(calls) == 1 else "[]"

    monkeypatch.setattr(llm_client, "get_llm_client", lambda: FakeClient())
    monkeypatch.setattr(jobs.time, "sleep", lambda _seconds: None)

    assert jobs._complete_concept_notes("prompt", "system") == "[]"
    assert len(calls) == 2


def test_notes_collect_a_comparison_table() -> None:
    from deeptutor.review.concept_notes import compare_markdown

    concept = _concept()
    extras: dict = {}
    reply = json.dumps(
        [
            {
                "name": "谐振频率",
                "summary": "阻抗达到最小。",
                "definition": "串联回路中感抗与容抗相互抵消的状态。",
                "compare": {
                    "title": "串联与并联对比",
                    "columns": ["项目", "串联谐振", "并联谐振"],
                    "rows": [
                        ["阻抗", "最小", "最大"],
                        ["电流", "最大", "最小"],
                    ],
                },
            }
        ],
        ensure_ascii=False,
    )

    _run([concept], reply, extras=extras)

    table = extras[concept.id]["compare"]
    assert table["title"] == "串联与并联对比"
    assert table["columns"] == ["项目", "串联谐振", "并联谐振"]
    assert table["rows"][0] == ["阻抗", "最小", "最大"]
    markdown = compare_markdown(table)
    assert "| 项目 | 串联谐振 | 并联谐振 |" in markdown
    # A pipe inside a cell would split the row; it becomes a full-width bar.
    assert "x｜y" in compare_markdown(
        {"title": "", "columns": ["A", "B"], "rows": [["x|y", "z"]]}
    )


def test_malformed_comparison_tables_are_dropped() -> None:
    concept = _concept()
    for compare in (
        {"columns": ["只有一列"], "rows": [["x"]]},
        {"columns": ["A", "B"], "rows": [["x"]]},
        {"columns": ["A", "B"], "rows": [["x", "y"], ["z"]]},
        {"columns": ["A", "B", "C", "D", "E"], "rows": [["1", "2", "3", "4", "5"]]},
        {"columns": ["A", "B"], "rows": []},
    ):
        extras: dict = {}
        reply = json.dumps(
            [{"name": "谐振频率", "compare": compare}], ensure_ascii=False
        )
        _run([concept], reply, extras=extras)
        assert "compare" not in extras.get(concept.id, {}), compare


def test_symbols_that_reveal_the_answer_are_dropped() -> None:
    concept = _concept()
    extras: dict = {}
    reply = json.dumps(
        [
            {
                "name": "谐振频率",
                "summary": "",
                "definition": "",
                "symbols": "感抗与容抗相互抵消",
                "pitfalls": ["注意单位"],
                "self_check": "核对量纲。",
            }
        ],
        ensure_ascii=False,
    )

    enriched, warnings, _ = _run(
        [concept], reply, items=[_choice_item()], extras=extras
    )

    assert enriched == 1
    assert "symbols" not in extras[concept.id]
    assert extras[concept.id]["pitfalls"] == ["注意单位"]
    assert any("符号与单位可能泄露答案" in message for message in warnings)
