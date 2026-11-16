"""The fidelity audit must canonicalise P4 lecture/courseware markers.

A 微机 or 信号 pack used to fall through to ``实验N`` because the audit's
canonicaliser only knew chapters and experiments; grounding then silently
skipped every source document.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "audit_content_fidelity.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("audit_content_fidelity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lecture_and_courseware_markers_are_canonicalised() -> None:
    module = _module()
    assert module._canonical_chapter_name("003_微机原理及应用_第1讲_课程重点") == (
        "第1讲",
        True,
    )
    assert module._canonical_chapter_name(
        "课件__CH3-2_周期信号的频谱分析——傅里叶级数"
    ) == ("ch3_周期信号的频谱分析——傅里叶级数", True)
    assert module._canonical_chapter_name(
        "000_数字信号处理_第1章_第10讲_课程重点"
    ) == ("第1章", False)


def test_markdown_check_allows_tables_and_fenced_code() -> None:
    module = _module()
    # P5 renders pipe tables and fenced code as real blocks.
    assert module._markdown_issues("| a | b |\n|---|---|\n| 1 | 2 |") == []
    assert module._markdown_issues("```asm\nMOV A, **x**\n```") == []


def test_markdown_check_flags_only_unbalanced_emphasis() -> None:
    # The export resolves balanced *em* / **strong** into real nodes, so only an
    # odd asterisk count still shows a stray star. __ has no renderer at all.
    module = _module()
    assert module._markdown_issues("这是 **重点**") == []
    assert module._markdown_issues("*倍频 $\\times N_1=100$：* 属于线性搬移") == []
    assert "unbalanced '*' marker" in module._markdown_issues("这是 *重点")
    assert "bold marker '__'" in module._markdown_issues("这是 __重点__")
    # Multiplication inside math is not a delimiter.
    assert module._markdown_issues("$a*b*c$ 与 $2*\\pi$") == []


def test_callout_list_literal_is_flagged() -> None:
    # 6 of 18 packs shipped this shape once: quotes and a stray bracket in the
    # pre-answer card, invisible to every item-level check.
    module = _module()
    assert module._callout_issues("$v_p$：相速，m/s', '$x$：含义']")
    assert module._callout_issues("$a$：含义']")
    assert module._callout_issues("$v_p$：相速，m/s\n$a,b$：内、外半径，m") == []


def test_split_subscript_is_flagged() -> None:
    # One stored concept name still had "$f_{0}$.7,n" after the regex fix: the
    # page rendered f_0 and then literal ".7,n". The audit now catches it.
    module = _module()
    assert any("split subscript" in issue for issue in module._math_issues("2Δ$f_{0}$.7,n"))
    assert module._math_issues("2Δ$f_{0.7,n}$") == []


def _audit(module, item: dict):
    return module.audit_item(item, 0, {}, module._KbCache(), 0.35)


def _explanation_failures(report: dict) -> list[dict]:
    return [f for f in report["failures"] if f["category"] == "explanation"]


def test_written_answer_accepts_the_paraphrase_the_gate_accepts() -> None:
    # G4 passes this: "固定中频" covers 2 of its 3 bigrams (> 0.6). The audit
    # used to demand a verbatim substring and flagged natural prose as
    # blocking, which stopped a real 6-item pack from passing.
    module = _module()
    report = _audit(
        module,
        {
            "id": "q1",
            "kind": "short_answer",
            "stem": "超外差接收机把不同载频变换为____",
            "answer": {"text": "固定中频"},
            "explanation": "混频器利用差频分量得到固定的中频信号，后续电路工作在固定频率上。",
            "concept_ids": ["c1"],
        },
    )
    assert _explanation_failures(report) == []


def test_written_answer_still_flags_a_missing_answer() -> None:
    module = _module()
    report = _audit(
        module,
        {
            "id": "q2",
            "kind": "short_answer",
            "stem": "超外差接收机把不同载频变换为____",
            "answer": {"text": "固定中频"},
            "explanation": "混频器把信号搬到另一个频率上，便于后续放大。",
            "concept_ids": ["c1"],
        },
    )
    assert _explanation_failures(report)


def test_numeric_answer_accepts_the_unit_spelling_the_gate_accepts() -> None:
    # G4's _mentions_number reads "1.592 MHz" as 1.592e6; a literal substring
    # check cannot, so the audit would have failed a correct explanation.
    module = _module()
    report = _audit(
        module,
        {
            "id": "q3",
            "kind": "numeric",
            "stem": "中频频率是多少",
            "answer": {"value": 1592000},
            "explanation": "取中频为 1.592 MHz。",
            "concept_ids": ["c1"],
        },
    )
    assert _explanation_failures(report) == []
