"""The correct option must not live on A for most of a pack.

The generator returns the right answer first almost every time; one real DSP pack
had 10 of 13 answers on "A", which a learner can beat without reading. Rotation
fixes the position, but explanations discuss the distractors by letter, so the
letters have to travel with the options — and formulas that use ``A`` as a
residue (``A_1``, ``A_k``) must be left alone.
"""

from __future__ import annotations

from deeptutor.review.balance import (
    answer_distribution,
    balance_items,
    pack_offset,
    rotate_item,
)
from deeptutor.review.models import Answer, Item, Option


def _item(qid: str, correct: int, explanation: str = "") -> Item:
    return Item(
        id=qid,
        concept_ids=["c-1"],
        kind="single_choice",
        stem="题干",
        options=[Option(label=letter, text=f"选项{letter}") for letter in "ABCD"],
        answer=Answer(choice_index=correct),
        explanation=explanation,
    )


def test_rotation_moves_the_answer_and_relabels_options() -> None:
    item = _item("q1", 0, "逐项看：B 误用线性关系，C 多乘了系数，D 忽略了相位。")
    rotated, changed = rotate_item(item, 2)

    assert changed
    assert rotated.answer.choice_index == 2
    # The option that used to sit at A is now at C and carries label C.
    assert rotated.options[2].text == "选项A"
    assert [o.label for o in rotated.options] == list("ABCD")
    # Every original option survives exactly once.
    assert sorted(o.text for o in rotated.options) == sorted(o.text for o in item.options)


def test_explanation_letters_follow_their_options() -> None:
    item = _item("q1", 0, "逐项看：B 误用线性关系，C 多乘了系数；A 才是定义式。")
    rotated, _ = rotate_item(item, 2)
    # A->C, B->D, C->A, D->B for a shift of 2.
    assert "D 误用线性关系" in rotated.explanation
    assert "A 多乘了系数" in rotated.explanation
    assert "C 才是定义式" in rotated.explanation


def test_formula_residues_are_not_rewritten() -> None:
    # A_1 / A_k are residues, not option names. Rewriting them would corrupt
    # the maths the learner reads.
    explanation = r"留数 $A_1=2$、$A_2=1$，故 $H(z)=\dfrac{A_k}{1-e^{s_kT}z^{-1}}$。"
    item = _item("q1", 0, explanation)
    rotated, changed = rotate_item(item, 1)
    assert changed
    assert rotated.explanation == explanation


def test_pack_distribution_is_evenly_spread() -> None:
    items = [_item(f"q{i}", 0) for i in range(8)]
    balanced, moved = balance_items(items, offset=0)
    # Targets cycle A,B,C,D; the two items already sitting on their target are
    # left alone rather than rotated for its own sake.
    assert moved == 6
    assert answer_distribution(balanced) == {"A": 2, "B": 2, "C": 2, "D": 2}


def test_balancing_is_idempotent() -> None:
    items = [_item(f"q{i}", 0) for i in range(5)]
    once, _ = balance_items(items, offset=pack_offset("spec-x"))
    twice, moved = balance_items(once, offset=pack_offset("spec-x"))
    assert moved == 0
    assert [i.answer.choice_index for i in twice] == [i.answer.choice_index for i in once]


def test_other_kinds_pass_through_untouched() -> None:
    numeric = Item(
        id="q2",
        concept_ids=["c-1"],
        kind="numeric",
        stem="求值",
        answer=Answer(value=1.0, tolerance=0.01),
    )
    three = Item(
        id="q3",
        concept_ids=["c-1"],
        kind="single_choice",
        stem="三选一",
        options=[Option(label=x, text=x) for x in "ABC"],
        answer=Answer(choice_index=0),
    )
    out, moved = balance_items([numeric, three], offset=1)
    assert moved == 0
    assert out[0] is numeric and out[1] is three
