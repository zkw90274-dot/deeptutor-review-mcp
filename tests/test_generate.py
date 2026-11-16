"""Fixture-driven tests for the KB -> ReviewSpec generator.

No real LLM is involved: ``AgentCoordinator.generate_from_topic`` is replaced
with an async fake that returns the exact payload shape the pipeline emits
(``results: [{qa_pair, metadata}]``). The filesystem seam
(``_knowledge_bases_root``) is replaced for chapter inference.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from deeptutor.review import generate
from deeptutor.review.generate import Chapter, generate_spec_from_kb, infer_chapters
from deeptutor.review.models import ReviewSpec
from deeptutor.review.qa import run_gates
from deeptutor.review.service import ReviewService
from deeptutor.review.storage import ReviewStore

CHAPTER = "第1章_绪论"
KB = "hf"


def _make_kb(tmp_path: Path, filenames: list[str], kb: str = KB) -> Path:
    root = tmp_path / "knowledge_bases"
    raw = root / kb / "raw"
    raw.mkdir(parents=True)
    for filename in filenames:
        (raw / filename).write_text("# 内容", encoding="utf-8")
    return root


def _patch_chapters(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    monkeypatch.setattr(
        generate,
        "infer_chapters",
        lambda kb_name: [Chapter(name=name, doc_count=1) for name in names],
    )


def _patch_generation(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    captured: dict[str, Any] | None = None,
) -> None:
    from deeptutor.agents.question.coordinator import AgentCoordinator

    async def fake(self: Any, **kwargs: Any) -> dict[str, Any]:
        if captured is not None:
            captured["kwargs"] = kwargs
            captured["kb_name"] = self.kb_name
            captured["language"] = self.language
        return payload

    monkeypatch.setattr(AgentCoordinator, "generate_from_topic", fake)


def _qa(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "question_id": "q",
        "question": "题目",
        "question_type": "choice",
        "options": {"A": "甲", "B": "乙", "C": "丙"},
        "correct_answer": "A",
        "explanation": "",
        "difficulty": "medium",
        "concentration": "知识点",
    }
    base.update(overrides)
    return base


def _payload(*qa_pairs: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": True,
        "completed": len(qa_pairs),
        "failed": 0,
        "mode": "topic",
        "results": [
            {"qa_pair": qa_pair, "metadata": None} for qa_pair in qa_pairs
        ],
    }


# -- infer_chapters ----------------------------------------------------------


def test_infer_chapters_groups_sorts_and_collects_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_kb(
        tmp_path,
        [
            "000_第2章_选频网络__第1讲.md",
            "001_第2章_选频网络__第2讲.pdf",  # every file type counts
            "002_第2章_选频网络.md",  # extension stripped before matching
            "003_第1章_绪论__第1讲.md",
            "004_三级项目__实验一_谐振放大器__00_总览.md",
            "005_杂项.md",
            "006_第一章_绪论__第2讲.md",  # Chinese numeral merges into 第1章
            "007_notes.txt",
            "008_第10章_直流电源.md",
            "009_实验二_混频器__指导书.pdf",
            "010_学习资料__第1章_绪论.md",  # duplicate chapter source, dropped
            "011_学习资料__第6章_振荡器.md",  # only source of chapter 6, kept
        ],
    )
    monkeypatch.setattr(generate, "_knowledge_bases_root", lambda: root)

    assert infer_chapters(KB) == [
        Chapter(name="第1章_绪论", doc_count=2),
        Chapter(name="第2章_选频网络", doc_count=3),
        Chapter(name="第6章_振荡器", doc_count=1),
        Chapter(name="第10章_直流电源", doc_count=1),
        Chapter(name="实验一_谐振放大器", doc_count=1),
        Chapter(name="实验二_混频器", doc_count=1),
        Chapter(name="其他", doc_count=2),
    ]


def test_infer_chapters_groups_a_lecture_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 微机 is filed by lecture (第N讲), not by chapter. The generic
    # 课程重点/复核报告 titles must not turn into chapter names.
    root = _make_kb(
        tmp_path,
        [
            "000_微机原理及应用_第10讲_课程重点.md",
            "001_微机原理及应用_第10讲_复核报告.md",
            "002_微机原理及应用_第1讲_课程重点.md",
            "003_微机原理及应用_课程重点_核对报告.md",
            "004_微机原理课程ppt__md__微机原理01.md",
        ],
    )
    monkeypatch.setattr(generate, "_knowledge_bases_root", lambda: root)

    assert infer_chapters(KB) == [
        Chapter(name="第1讲", doc_count=1),
        Chapter(name="第10讲", doc_count=2),
        Chapter(name="其他", doc_count=2),
    ]


def test_infer_chapters_groups_a_courseware_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 信号 courseware is filed as chN; the 速成 lectures use 第N讲. Section
    # numbers alone (ch1-1.1) are not a topic, so the group keeps the code.
    root = _make_kb(
        tmp_path,
        [
            "000_课件__ch1-1.1.md",
            "001_课件__CH3-2_周期信号的频谱分析——傅里叶级数.md",
            "002_速成课讲义__第一讲_信号与系统概述.md",
            "003_试题题库__期末试卷.md",
        ],
    )
    monkeypatch.setattr(generate, "_knowledge_bases_root", lambda: root)

    assert infer_chapters(KB) == [
        Chapter(name="第1讲_信号与系统概述", doc_count=1),
        Chapter(name="ch1", doc_count=1),
        Chapter(name="ch3_周期信号的频谱分析——傅里叶级数", doc_count=1),
        Chapter(name="其他", doc_count=1),
    ]


def test_infer_chapters_cleans_chapter_title_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # DSP keeps chapter grouping, but its titles carry media codes and
    # numeral restatements; only the real topic belongs in the name.
    root = _make_kb(
        tmp_path,
        [
            "000_数字信号处理_第1章_第10讲_课程重点.md",
            "001_数字信号处理_第1章_MC1-1-1离散时间信号的由来.md",
            "002_md转换__第四章__第四章.md",
        ],
    )
    monkeypatch.setattr(generate, "_knowledge_bases_root", lambda: root)

    assert infer_chapters(KB) == [
        Chapter(name="第1章_离散时间信号的由来", doc_count=2),
        Chapter(name="第4章", doc_count=1),
    ]


def test_infer_chapters_rejects_unknown_kb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generate, "_knowledge_bases_root", lambda: tmp_path)

    with pytest.raises(FileNotFoundError):
        infer_chapters("missing")


def test_infer_chapters_rejects_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generate, "_knowledge_bases_root", lambda: tmp_path)

    with pytest.raises(ValueError):
        infer_chapters("../etc")


# -- generate_spec_from_kb ---------------------------------------------------


def _conversion_payload() -> dict[str, Any]:
    return _payload(
        # 1. choice: dict order differs from label order; empty explanation.
        _qa(
            question="串联谐振时回路阻抗为什么最小？",
            options={"C": "容抗占主导", "A": "感抗与容抗相互抵消", "B": "电阻被短路"},
            correct_answer="A",
            explanation="",
            difficulty="easy",
            concentration="串联谐振",
        ),
        # 2. choice whose letter is not in the options -> short_answer.
        _qa(
            question="混频器的功能是什么？",
            correct_answer="E",
            explanation="",
            concentration="混频器",
        ),
        # 3. written -> short_answer.
        _qa(
            question="写出品质因数的定义式。",
            question_type="written",
            options=None,
            correct_answer="Q = ω0 L / R",
            explanation="",
            concentration="品质因数",
        ),
        # 4. choice with no concentration: concept falls back to the chapter.
        _qa(
            question="并联谐振回路在谐振时呈什么性质？",
            options={"A": "纯电阻性", "B": "纯电感性"},
            correct_answer="A",
            explanation="A. 纯电阻性",
            concentration=None,
        ),
        # 5. unusable: no reference answer.
        _qa(
            question="缺少答案的题",
            correct_answer="",
        ),
        # 6. unusable: pipeline failure placeholder.
        _qa(
            question="[Generation failed] 单调谐回路计算",
            correct_answer="N/A",
            explanation="N/A",
        ),
    )


def test_generate_converts_choice_written_and_degrades_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_chapters(monkeypatch, CHAPTER)
    captured: dict[str, Any] = {}
    _patch_generation(monkeypatch, _conversion_payload(), captured)

    result = generate_spec_from_kb(
        kb_name=KB,
        chapter=CHAPTER,
        num_questions=5,
        question_types=["single_choice", "short_answer"],
        difficulty="hard",
    )

    spec = result.spec
    assert isinstance(spec, ReviewSpec)
    assert result.kb_name == KB
    assert result.chapter == CHAPTER
    assert result.qa_count == 3
    # no answer + generation placeholder + unparseable choice
    assert result.skipped == 3
    assert len(spec.items) == 3

    # The pipeline receives the coordinator's own question-type domain.
    assert captured["kb_name"] == KB
    assert captured["language"] == "zh"
    assert captured["kwargs"]["user_topic"] == CHAPTER
    assert captured["kwargs"]["num_questions"] == 5
    assert captured["kwargs"]["question_types"] == ["choice", "written"]
    assert captured["kwargs"]["difficulty"] == "hard"

    # 1. Options are sorted by label and the letter maps to that order.
    first = spec.items[0]
    assert first.kind == "single_choice"
    assert [(option.label, option.text) for option in first.options] == [
        ("A", "感抗与容抗相互抵消"),
        ("B", "电阻被短路"),
        ("C", "容抗占主导"),
    ]
    assert first.answer.choice_index == 0
    assert "感抗与容抗相互抵消" in first.explanation

    # 2. A choice whose letter matches no option is skipped, never shown as
    #    a short-answer item keyed by the raw letter.
    second = spec.items[1]
    assert second.kind == "short_answer"
    assert second.answer.text == "Q = ω0 L / R"
    assert second.explanation == "正确答案：Q = ω0 L / R"
    assert second.self_check == generate.DEFAULT_SELF_CHECK
    assert second.pitfalls == []
    assert second.assumptions == []

    # 3. Existing explanation is preserved when it already names the option.
    third = spec.items[2]
    assert third.kind == "single_choice"
    assert third.explanation == "A. 纯电阻性"
    assert third.self_check == generate.DEFAULT_SELF_CHECK

    # Concepts: concentration when present, chapter name otherwise. The
    # skipped choice question contributed no concept.
    assert [concept.name for concept in spec.concepts] == [
        "串联谐振",
        "品质因数",
        CHAPTER,
    ]
    keys = [concept.key for concept in spec.concepts]
    assert all(keys)
    assert len(set(keys)) == len(keys)
    assert all(concept.source_ids == [f"kb:{KB}·{CHAPTER}"] for concept in spec.concepts)
    assert all(item.source_ids == [f"kb:{KB}·{CHAPTER}"] for item in spec.items)
    assert spec.sources[f"kb:{KB}·{CHAPTER}"].kind == "doc"

    # Section mirrors the item order and starts with the prose summary.
    blocks = spec.sections[0].blocks
    assert blocks[0].kind == "prose"
    assert [block.item_id for block in blocks[1:]] == [item.id for item in spec.items]
    assert spec.sections[0].concept_ids == [concept.id for concept in spec.concepts]

    # Item ids are unique and deterministic.
    assert len({item.id for item in spec.items}) == len(spec.items)
    assert spec.id == generate_spec_from_kb(
        kb_name=KB,
        chapter=CHAPTER,
        num_questions=5,
        question_types=["choice", "written"],
        difficulty="hard",
    ).spec.id


def test_generate_returns_no_spec_when_every_question_is_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_chapters(monkeypatch, CHAPTER)
    _patch_generation(monkeypatch, _payload(_qa(question=""), _qa(correct_answer="")))

    result = generate_spec_from_kb(kb_name=KB, chapter=CHAPTER)

    assert result.spec is None
    assert result.qa_count == 0
    assert result.skipped == 2
    assert any("no usable questions" in warning for warning in result.warnings)


def test_generate_rejects_a_chapter_not_in_the_kb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_chapters(monkeypatch, "第2章_选频网络")

    with pytest.raises(ValueError, match="not found"):
        generate_spec_from_kb(kb_name=KB, chapter=CHAPTER)


def test_generate_propagates_llm_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    from deeptutor.agents.question.coordinator import AgentCoordinator

    _patch_chapters(monkeypatch, CHAPTER)

    async def boom(self: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("provider down")

    monkeypatch.setattr(AgentCoordinator, "generate_from_topic", boom)

    with pytest.raises(RuntimeError, match="provider down"):
        generate_spec_from_kb(kb_name=KB, chapter=CHAPTER)


def test_generated_spec_passes_qa_gates_and_put_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_chapters(monkeypatch, CHAPTER)
    _patch_generation(monkeypatch, _conversion_payload())

    result = generate_spec_from_kb(kb_name=KB, chapter=CHAPTER, num_questions=5)
    assert result.spec is not None

    report = run_gates(result.spec)
    assert report.passed, [gate.model_dump() for gate in report.gates if not gate.passed]

    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(tmp_path / "review"))
    monkeypatch.setenv("DEEPTUTOR_HOME", str(tmp_path / "core-home"))
    service = ReviewService(ReviewStore(tmp_path / "review"))
    stored = service.put_spec(result.spec)

    assert stored["spec_id"] == result.spec.id
    assert stored["quality"]["passed"] is True


def test_raw_math_symbols_are_escaped_before_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLM LaTeX often writes raw ``<``/``>``/``|``; G3 rejects exactly that.

    The converter must sanitize inside math segments (and only there) before
    the spec reaches the gate, otherwise real generations never pass.
    """

    from deeptutor.review.qa import run_gates

    _patch_chapters(monkeypatch, CHAPTER)
    payload = _payload(
        _qa(
            question="已知 $S<1$ 且 $|A_v|>10$ 时放大器是否稳定？",
            options={"A": "稳定", "B": "不稳定"},
            correct_answer="A",
            explanation=r"当 $S\lt 1$、$\vert{}A_v\vert{}\gt 10$ 时稳定。A. 稳定",
            concentration="稳定性",
        ),
    )
    _patch_generation(monkeypatch, payload, {})

    result = generate_spec_from_kb(kb_name=KB, chapter=CHAPTER, num_questions=1)
    spec = result.spec
    assert spec is not None
    item = spec.items[0]
    assert r"\lt" in item.stem and r"\gt" in item.stem
    assert r"\vert{}" in item.stem
    report = run_gates(spec)
    gate3 = next(g for g in report.gates if g.gate == "G3-math-escape")
    assert gate3.passed is True, gate3.details
    # Prose outside math keeps its characters untouched.
    assert "放大器是否稳定" in item.stem


def test_markdown_keeps_tables_but_flattens_bold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bold is still flattened; pipe tables now survive for the table renderer."""

    _patch_chapters(monkeypatch, CHAPTER)
    payload = _payload(
        _qa(
            question="**重要**：级联噪声系数怎么算？",
            options={"A": "Friis 公式", "B": "香农公式"},
            correct_answer="A",
            explanation="使用 **Friis 公式**：\n| 级 | 增益 |\n|:--:|--:|\n| 1 | 10 |",
            concentration="噪声",
        ),
    )
    _patch_generation(monkeypatch, payload, {})
    result = generate_spec_from_kb(kb_name=KB, chapter=CHAPTER, num_questions=1)

    item = result.spec.items[0]
    texts = [item.stem, item.explanation, *(option.text for option in item.options)]
    for text in texts:
        assert "**" not in text, text
    assert "| 级 | 增益 |" in item.explanation
    assert "|:--:|--:|" in item.explanation
    assert "| 1 | 10 |" in item.explanation


def test_concept_case_variants_share_the_first_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recheck N-05: 'RC Circuit' and 'rc circuit' are one concept.

    Recomputing the id for the second spelling produced a dangling
    concept_id and failed the entire chapter generation.
    """
    _patch_chapters(monkeypatch, CHAPTER)
    _patch_generation(
        monkeypatch,
        _payload(
            _qa(concentration="RC Circuit", correct_answer="A"),
            _qa(concentration="rc circuit", correct_answer="B"),
        ),
    )

    result = generate_spec_from_kb(
        kb_name=KB, chapter=CHAPTER, num_questions=2
    )

    assert result.spec is not None
    assert len(result.spec.concepts) == 1
    concept_id = result.spec.concepts[0].id
    assert [item.concept_ids for item in result.spec.items] == [
        [concept_id],
        [concept_id],
    ]


def test_generate_uses_llm_complete_for_concept_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The note hook fills hints; without it concepts stay name-only."""

    _patch_chapters(monkeypatch, CHAPTER)
    _patch_generation(
        monkeypatch,
        _payload(
            _qa(
                question="串联谐振时回路阻抗为什么最小？",
                concentration="串联谐振",
            )
        ),
    )
    calls: list[tuple[str, str]] = []

    def complete(prompt: str, system_prompt: str) -> str:
        calls.append((prompt, system_prompt))
        return (
            '[{"name": "串联谐振", "summary": "阻抗达到最小。", '
            '"definition": "感抗与容抗相等时，回路呈纯电阻性。", '
            '"symbols": "R — 电阻 / Ω；C — 电容 / F", '
            '"pitfalls": ["单位混用", "忘记谐振条件"], '
            '"self_check": "把 L、C 换算到 SI 后重新计算。", '
            '"assumptions": ["忽略电感与电容的寄生参数"], '
            '"compare": {"title": "串联与并联", "columns": ["项目", "串联", "并联"], '
            '"rows": [["阻抗", "最小", "最大"]]}}]'
        )

    enriched = generate_spec_from_kb(
        kb_name=KB, chapter=CHAPTER, llm_complete=complete
    )
    plain = generate_spec_from_kb(kb_name=KB, chapter=CHAPTER)

    assert enriched.spec is not None and plain.spec is not None
    assert len(calls) == 1
    assert "串联谐振" in calls[0][0]
    assert enriched.spec.concepts[0].summary_one_line == "阻抗达到最小。"
    assert "纯电阻性" in enriched.spec.concepts[0].definition
    assert plain.spec.concepts[0].summary_one_line == ""
    assert plain.spec.concepts[0].definition == ""

    blocks = enriched.spec.sections[0].blocks
    callouts = [block for block in blocks if block.kind == "callout"]
    assert any(
        "符号与单位" in block.text and "电阻" in block.text for block in callouts
    )
    item = enriched.spec.items[0]
    assert item.self_check == "把 L、C 换算到 SI 后重新计算。"
    assert item.pitfalls == ["单位混用", "忘记谐振条件"]
    assert item.assumptions == ["忽略电感与电容的寄生参数"]
    assert item.study_table.splitlines()[:3] == [
        "串联与并联",
        "",
        "| 项目 | 串联 | 并联 |",
    ]
    assert "| 阻抗 | 最小 | 最大 |" in item.study_table
    assert "【自检】" not in item.explanation

    # Without the optional notes call the pack still gets the generic
    # self-check field, but no invented symbols or pitfalls.
    plain_item = plain.spec.items[0]
    assert plain_item.self_check == generate.DEFAULT_SELF_CHECK
    assert plain_item.pitfalls == []
    assert plain_item.assumptions == []
    assert all(block.kind != "callout" for block in plain.spec.sections[0].blocks)


def test_clean_text_keeps_dotted_subscripts_whole() -> None:
    from deeptutor.review.generate import _clean_text

    # 高频 uses K_{r0.1}/B_{0.1}; a dot was outside the old subscript class,
    # producing "$K_{r0}$.1}" and a stray brace in the page.
    cleaned = _clean_text("矩形系数 K_{r0.1} 与 B_{0.1} 的关系")
    assert "$K_{r0.1}$" in cleaned
    assert "$B_{0.1}$" in cleaned
    assert "$K_{r0}$" not in cleaned
    assert "$B_{0}$" not in cleaned


def test_clean_text_keeps_tables_and_protects_fenced_code() -> None:
    from deeptutor.review.generate import _clean_text

    text = (
        "| 指令 | 功能 |\n"
        "|:---|:---:|\n"
        "| MOV A, #30H | 送立即数 |\n\n"
        "```asm\nMOV A, #30H ; V_CC and a|b stay literal\n```\n\n"
        "外部 V_CC 要包成数学。"
    )
    cleaned = _clean_text(text)
    # Table structure survives for the export renderer.
    assert "| 指令 | 功能 |" in cleaned
    assert "|:---|:---:|" in cleaned
    # Code keeps its identifiers and comparison/pipe characters verbatim.
    assert "MOV A, #30H ; V_CC and a|b stay literal" in cleaned
    # Prose is still normalised.
    assert "外部 $V_{CC}$ 要包成数学" in cleaned


def test_symbols_callout_dedupes_and_caps_entries() -> None:
    from deeptutor.review.generate import _symbols_callout
    from deeptutor.review.models import Concept

    concepts = [
        Concept(id="c1", name="概念一", source_ids=["s"]),
        Concept(id="c2", name="概念二", source_ids=["s"]),
    ]
    extras = {
        "c1": {"symbols": "f_s：射频，Hz；f_0：本振，Hz"},
        "c2": {"symbols": "f_s：射频，Hz；f_I：中频，Hz"},
    }
    text = _symbols_callout(concepts, extras)
    assert text.count("f_s") == 1
    assert "概念一：" not in text and "概念二：" not in text
    # One definition per line, not a semicolon paragraph.
    assert text.startswith("符号与单位")
    assert text.splitlines()[1:] == ["f_s：射频，Hz", "f_0：本振，Hz", "f_I：中频，Hz"]

    many = {"c1": {"symbols": "；".join(f"s{i}：含义{i}" for i in range(20))}}
    capped = _symbols_callout([concepts[0]], many)
    assert len(capped.splitlines()) <= 13  # heading + 12 entries


def test_bare_formula_is_wrapped_whole_instead_of_per_identifier() -> None:
    from deeptutor.review.generate import _clean_text

    # A bare formula used to become "\frac{\Delta $f_{m}${F}": the optional
    # closing brace in the identifier regex ate the brace that closed
    # \frac's first argument, and the identifier was wrapped inside it.
    cleaned = _clean_text(r"\frac{\Delta f_m}{F}")
    assert cleaned == r"$\frac{\Delta f_m}{F}$"
    assert "$f_{m}$" not in cleaned
    # Prose keeps per-identifier wrapping.
    assert _clean_text("普通文本 V_CC") == "普通文本 $V_{CC}$"
    # A Windows path starts with a backslash command too, but is not math.
    assert _clean_text(r"C:\Users\demo\notes") == r"C:\Users\demo\notes"
    assert _clean_text(r"data\knowledge_bases") == r"data\knowledge_bases"


def test_boolean_written_answers_are_localised() -> None:
    from deeptutor.review.generate import _parse_qa

    # 判断题 arrive as written questions with an English boolean. The learner
    # must see 正确/错误, and the appended suffix has to use the same word.
    for raw, expected in (("false", "错误"), ("TRUE", "正确")):
        parsed, reason = _parse_qa(
            {
                "question": "判断下列说法是否正确：调制就是把低频信号搬移到高频。",
                "question_type": "written",
                "correct_answer": raw,
                "explanation": "命题方向正确，但把 FM 与 PM 的解调方式混为一谈，因此整体为假。",
                "difficulty": "medium",
                "concentration": "调制与解调",
            },
            chapter=CHAPTER,
            requested_difficulty="",
        )
        assert reason == ""
        assert parsed is not None
        assert parsed.answer.text == expected
        assert expected in parsed.explanation


# -- index staleness ---------------------------------------------------------


def _make_indexed_kb(
    tmp_path: Path,
    last_indexed_at: str | None,
    *,
    newest_mtime: datetime | None = None,
    kb: str = KB,
) -> Path:
    root = _make_kb(tmp_path, ["000_第1章_绪论__第1讲.md"], kb=kb)
    if last_indexed_at is not None:
        (root / kb / "metadata.json").write_text(
            json.dumps({"last_indexed_at": last_indexed_at}, ensure_ascii=False),
            encoding="utf-8",
        )
    if newest_mtime is not None:
        stamp = newest_mtime.timestamp()
        for entry in (root / kb / "raw").iterdir():
            os.utime(entry, (stamp, stamp))
    return root


def test_index_staleness_warning_reports_a_changed_raw_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The core never re-flags a KB when a raw file is edited, so the corrected
    # text does not reach retrieval until a rebuild. Generation must say so.
    root = _make_indexed_kb(
        tmp_path,
        "2026-11-01 13:35:07",
        newest_mtime=datetime(2026, 11, 9, 18, 22),
    )
    monkeypatch.setattr(generate, "_knowledge_bases_root", lambda: root)

    warning = generate.index_staleness_warning(KB)

    assert warning is not None
    assert "索引可能已过期" in warning
    assert "2026-11-09" in warning and "2026-11-01" in warning


def test_index_staleness_warning_is_quiet_for_a_fresh_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_indexed_kb(
        tmp_path,
        "2026-11-09 20:00:00",
        newest_mtime=datetime(2026, 11, 9, 18, 22),
    )
    monkeypatch.setattr(generate, "_knowledge_bases_root", lambda: root)

    assert generate.index_staleness_warning(KB) is None


def test_index_staleness_warning_ignores_kbs_without_index_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_indexed_kb(tmp_path, None)
    monkeypatch.setattr(generate, "_knowledge_bases_root", lambda: root)

    assert generate.index_staleness_warning(KB) is None
    # An absent KB is not evidence of staleness either.
    assert generate.index_staleness_warning("missing") is None


def test_generate_surfaces_the_stale_index_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_chapters(monkeypatch, CHAPTER)
    _patch_generation(monkeypatch, _payload(_qa()))
    root = _make_indexed_kb(
        tmp_path,
        "2026-11-01 13:35:07",
        newest_mtime=datetime(2026, 11, 9, 18, 22),
    )
    monkeypatch.setattr(generate, "_knowledge_bases_root", lambda: root)

    result = generate_spec_from_kb(kb_name=KB, chapter=CHAPTER, num_questions=1)

    assert result.spec is not None
    assert any("索引可能已过期" in warning for warning in result.warnings)
