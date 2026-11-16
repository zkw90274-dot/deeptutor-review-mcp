"""Round-4 audit regressions for the deterministic QA gates (G4 and G9).

Every case here was reproduced against the round-3 code. The companion
positive controls keep the fixes from becoming blunt instruments: escaped math
comparisons, ordinary equations, double negatives, valid parameter sets and
high-precision numbers must all keep their previous verdicts.
"""

from __future__ import annotations

import pytest

from deeptutor.review.models import (
    Answer,
    Concept,
    Item,
    QualityReport,
    ReviewSpec,
    SourceRef,
)
from deeptutor.review.qa import (
    _mentions_number,
    answer_coverage,
    parameter_infeasibility,
    run_gates,
)


def _spec(item: Item) -> ReviewSpec:
    item = item.model_copy(update={"source_ids": ["src-probe"]})
    return ReviewSpec(
        id="round4-gates",
        title="Round 4 gate probe",
        concepts=[
            Concept(id=concept_id, name=f"concept {concept_id}", coverage="inferred")
            for concept_id in item.concept_ids
        ],
        sections=[],
        items=[item],
        sources={"src-probe": SourceRef(id="src-probe", kind="doc", title="probe")},
        # ``inferred`` concepts must be announced in the client warnings; the
        # gate under test does not read them.
        quality=QualityReport(
            warnings=[f"concept {concept_id} inferred" for concept_id in item.concept_ids]
        ),
        content_hash="",
    )


def _g4_passes(item: Item) -> bool:
    report = run_gates(_spec(item))
    gate = next(gate for gate in report.gates if gate.gate == "G4-answer-key")
    return gate.passed


def _short_answer_item(explanation: str, answer: str) -> Item:
    return Item(
        id="item-g4",
        concept_ids=["c-af"],
        kind="short_answer",
        stem="判断放大器的工作状态。",
        answer=Answer(text=answer),
        explanation=explanation,
    )


def _numeric_item(explanation: str, value: float) -> Item:
    return Item(
        id="item-g4-numeric",
        concept_ids=["c-af"],
        kind="numeric",
        stem="求该量。",
        answer=Answer(value=value),
        explanation=explanation,
    )


def _rf_item(stem: str) -> Item:
    return Item(
        id="item-g9",
        concept_ids=["c-rf"],
        kind="short_answer",
        stem=stem,
        answer=Answer(text="按题设条件计算。"),
        explanation="按题设条件计算。",
    )


# -- A: comparison operators survive ``_text_bigrams`` --------------------


@pytest.mark.parametrize(
    "explanation,answer",
    [
        ("AF<1", "AF>1"),
        ("AF>1", "AF<1"),
        ("AF≤1", "AF>1"),
        ("AF≥1", "AF<1"),
        ("V_CC<12 V", "V_CC>12 V"),
        ("V_CC≤12 V", "V_CC≥12 V"),
    ],
)
def test_opposite_comparisons_are_not_covered(explanation: str, answer: str) -> None:
    assert answer_coverage(explanation, answer) == 0.0
    assert _g4_passes(_short_answer_item(explanation, answer)) is False


@pytest.mark.parametrize(
    "explanation,answer",
    [
        ("该条件满足 AF<1。", "AF<1"),
        ("该条件满足 AF≤1。", "AF≤1"),
        ("电源 V_CC>12 V 时饱和。", "V_CC>12 V"),
        (r"该条件满足 $AF\lt{}1$。", r"$AF\lt{}1$"),
        (r"该条件满足 $AF\gt{}1$。", r"$AF\gt{}1$"),
    ],
)
def test_matching_comparisons_are_still_covered(explanation: str, answer: str) -> None:
    assert answer_coverage(explanation, answer) >= 0.6
    assert _g4_passes(_short_answer_item(explanation, answer)) is True


def test_verbatim_answer_is_not_vetoed_by_extra_comparisons() -> None:
    """A literal statement of the answer wins over unrelated comparisons.

    The operator-signature check used to demand that the *whole* explanation
    carry exactly the answer's comparison tokens. Real generated explanations
    discuss other quantities with ``<``/``>`` around a verbatim answer, which
    G4 then rejected; the explanation below reproduces that shape.
    """
    answer = "串联谐振时感抗与容抗相互抵消，回路呈纯电阻性。"
    explanation = (
        "当品质因数 Q>10 时选择性较好；串联谐振时感抗与容抗相互抵消，"
        "回路呈纯电阻性。若 Q<1 则幅频曲线变得平缓。"
    )
    assert answer_coverage(explanation, answer) == 1.0
    assert _g4_passes(_short_answer_item(explanation, answer)) is True


def test_verbatim_answer_with_its_own_comparison_survives_extra_ones() -> None:
    answer = "AF>1"
    explanation = "当 Q>10 时 AF>1 成立，但 Q<1 时反馈不足。"
    assert answer_coverage(explanation, answer) == 1.0
    assert _g4_passes(_short_answer_item(explanation, answer)) is True


def test_escaped_comparison_mismatch_stays_below_the_threshold() -> None:
    # Round 3 already rejected this (0.5); the signature check must not silently
    # turn it into a pass.
    assert answer_coverage(r"$AF\gt{}1$", r"$AF\lt{}1$") <= 0.6


def test_plain_equations_are_not_a_comparison_signature() -> None:
    explanation = "先求 I_c1m=0.5 A，再由 P_o=6.25 W 得效率 η=81.8%。"
    answer = "I_c1m=0.5 A；P_o=6.25 W"
    assert answer_coverage(explanation, answer) >= 0.6
    assert _g4_passes(_short_answer_item(explanation, answer)) is True


# -- B: negation detection ------------------------------------------------


@pytest.mark.parametrize(
    "explanation",
    [
        "不会因为负载变化而增大",
        "不随温度变化而增大",
        "没有增大",
        "未能增大",
        "不会大幅增大",
        "不会因此增大",
        "输出不会导致增大。",
        "该假设并不成立，不能说明增大。",
    ],
)
def test_negated_answer_is_not_covered(explanation: str) -> None:
    assert answer_coverage(explanation, "增大") < 0.6
    assert _g4_passes(_short_answer_item(explanation, "增大")) is False


def test_long_negated_answer_is_not_exempt_by_length() -> None:
    answer = "增大" + "的" * 24  # 26 characters, past the old 24-char cutoff
    assert len(answer) == 26
    assert answer_coverage("输出不会" + answer, answer) == 0.0
    assert _g4_passes(_short_answer_item("输出不会" + answer, answer)) is False


@pytest.mark.parametrize(
    "explanation",
    [
        "电压不变，电流增大。",  # different clause
        "输出不得不增大。",
        "输出无不增大。",
        "输出不可不增大。",
        "输出未尝不增大。",
        "输出不能不增大。",
        "电流不断增大。",
        "参数不同，结果增大。",
    ],
)
def test_double_negatives_and_other_clauses_keep_positive_meaning(
    explanation: str,
) -> None:
    assert answer_coverage(explanation, "增大") >= 0.6


def test_double_negation_keeps_positive_answer_coverage() -> None:
    assert answer_coverage("该条件不是不可以满足。", "可以") >= 0.6


# -- C: the G4 threshold is inclusive -------------------------------------


def test_exactly_sixty_percent_coverage_is_rejected() -> None:
    explanation = "先调谐但不要测量"
    answer = "先调谐再测量"
    assert answer_coverage(explanation, answer) == pytest.approx(0.6)
    assert _g4_passes(_short_answer_item(explanation, answer)) is False


# -- D: numeric equivalence ----------------------------------------------


@pytest.mark.parametrize(
    "text,value",
    [
        ("增益为1,200。", 1200),
        ("结果为2×10^3。", 2000),
        (r"结果为$2\times10^{3}$。", 2000),
        ("占空比为50%。", 0.5),
        ("输出为500 mV。", 0.5),
        ("电源为2 kV。", 2000),
        ("电容为1 μF。", 1e-6),
        ("时钟为3 MHz。", 3e6),
        ("输出为5e-1 A。", 0.5),
        ("输出为0.50 A。", 0.5),
    ],
)
def test_equivalent_numeric_notations_pass_g4(text: str, value: float) -> None:
    assert _g4_passes(_numeric_item(text, value)) is True


@pytest.mark.parametrize(
    "text,value",
    [
        ("结果为12。", 2),
        ("增益为1,201。", 1200),
        ("输出为5 A。", 0.5),
        ("输出为500 mA。", 0.125),
    ],
)
def test_non_equivalent_numbers_still_fail_g4(text: str, value: float) -> None:
    assert _g4_passes(_numeric_item(text, value)) is False


def test_high_precision_and_large_integers_stay_exact() -> None:
    # ``Answer.value`` is a float, so the exactness proof has to use the string
    # path of ``_mentions_number``: no candidate may be routed through float.
    assert _mentions_number("结果为0.1234567890123456789。", "0.1234567890123456789") is True
    assert _mentions_number("结果为0.12345678901234568。", "0.1234567890123456789") is False
    assert _mentions_number("结果为9007199254740993。", "9007199254740993") is True
    assert _mentions_number("结果为9007199254740992。", "9007199254740993") is False
    assert answer_coverage("结果为0.12345678901234568。", "0.1234567890123456789") == 0.0
    assert answer_coverage("结果为9007199254740992。", "9007199254740993") == 0.0
    item = _short_answer_item("结果为9007199254740992。", "9007199254740993")
    assert _g4_passes(item) is False


# -- E: G9 parameter feasibility -----------------------------------------


@pytest.mark.parametrize(
    "stem,detail",
    [
        ("谐振功放 V_CC=12 V，且 2 I_c1m R_p=30 V。", "V_cm=15"),
        ("谐振功放 V_CC=20 V，且 2 I_c1m R_p=30 V。", None),
        (r"谐振功放 V_CC=12 V，且 $2I_{c1m}R_p=30$ V。", "V_cm=15"),
        (r"谐振功放 V_CC=20 V，且 $2I_{c1m}R_p=30$ V。", None),
        ("谐振功放 V_CC=12 V，且 2×I_c1m×R_p=30 V。", "V_cm=15"),
        ("谐振功放 V_CC=20 V，且 2×I_c1m×R_p=30 V。", None),
        ("谐振功放 V_CC=12 V，V_cm=1.2×V_CC。", "V_cm=14.4"),
        ("谐振功放 V_CC=14 V，V_cm=1.2×V_CC。", "V_cm=16.8"),
        ("谐振功放 V_CC=12 V，xi=1.2。", "xi=1.2"),
        ("归一化频率ξ=1.2，分析幅频响应。", None),
    ],
)
def test_implicit_products_and_scaled_supply(stem: str, detail: str | None) -> None:
    reason = parameter_infeasibility(_rf_item(stem))
    assert bool(reason) is (detail is not None)
    if detail is not None:
        assert detail in reason


@pytest.mark.parametrize(
    "stem",
    [
        "谐振功放 V_CC=12 V，且 2×I_c1m×R_p=30 V，求归一化负载。",
        "谐振功放 V_CC=12 V，且 2 I_c1m R_p=30 V，求归一化负载。",
        "谐振功放 V_CC=12 V，V_cm=1.2×V_CC，并求归一化效率。",
    ],
)
def test_normalization_request_does_not_hide_a_violation(stem: str) -> None:
    assert parameter_infeasibility(_rf_item(stem))


def test_non_rf_xi_is_not_rejected() -> None:
    item = Item(
        id="item-g9-filter",
        concept_ids=["c-filter"],
        kind="short_answer",
        stem="二阶低通滤波器阻尼比 ξ=1.2，分析阶跃响应。",
        answer=Answer(text="呈振荡收敛。"),
        explanation="阻尼比小于 1 时才振荡。",
    )
    assert parameter_infeasibility(item) is None


def test_g9_still_rejects_and_keeps_gate_detail() -> None:
    report = run_gates(_spec(_rf_item("谐振功放 V_CC=12 V，V_cm=1.2×V_CC。")))
    gate = next(gate for gate in report.gates if gate.gate == "G9-parameter-feasibility")
    assert gate.passed is False
    assert "V_cm=14.4" in gate.details[0]
