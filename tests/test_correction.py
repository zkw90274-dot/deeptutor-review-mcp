from __future__ import annotations

from deeptutor.review.correction import (
    CORRECTION_LABEL_CHARS,
    _label,
    build_correction,
)
from deeptutor.review.fixture import build_fixture_spec


def test_short_label_is_untouched() -> None:
    assert _label("  $A_v$ 与增益  ") == "$A_v$ 与增益"


def test_long_label_does_not_cut_inside_math() -> None:
    # The distractor used to be clipped mid-formula, leaving "$g_{p}…" on
    # screen as raw text instead of a rendered symbol.
    text = "接入系数 " + "$C_{oe}$+" * 12 + "$g_{p}$ 结尾"
    out = _label(text)
    assert out.endswith("…")
    assert out.count("$") % 2 == 0
    assert len(out) <= CORRECTION_LABEL_CHARS


def test_long_label_does_not_cut_inside_display_math() -> None:
    # ``$$…$$`` is two dollars on each side, so an even ``$`` count still hid
    # a dangling display opener that reached the page as raw TeX.
    text = "1. 展开式为 " + "$$" + "u_{FM}(t)=U_{cm}" + "x" * 100
    out = _label(text)
    assert out == "1. 展开式为…"
    assert len(out) <= CORRECTION_LABEL_CHARS


def test_fenced_answer_becomes_a_one_line_label() -> None:
    text = "```matlab\n%% 参数与系数\nfs = 1000;  f0 = 50;\nb = [1, -2*cos(w0), 1];\n```"
    out = _label(text)
    assert "```" not in out
    assert "\n" not in out
    assert out.startswith("代码：")
    assert len(out) <= CORRECTION_LABEL_CHARS


def test_fenced_answer_keeps_prose_outside_the_code() -> None:
    text = "结论：零点落在单位圆上。\n```matlab\nb = [1 -2 1];\n```"
    assert _label(text) == "结论：零点落在单位圆上。"


def test_multiline_answer_collapses_to_one_line() -> None:
    text = "（1）三个基本参量。\n\n（2）对 $Q$ 的要求。"
    out = _label(text)
    assert "\n" not in out
    assert out == "（1）三个基本参量。 （2）对 $Q$ 的要求。"


def test_correction_uses_the_label_for_a_fenced_answer() -> None:
    spec = build_fixture_spec()
    item = next(i for i in spec.items if i.kind == "flashcard")
    item.answer.text = "```matlab\n%% 参数与系数\nb = [1, -2*cos(w0), 1];\n```"

    correction = build_correction(spec, item)

    correct = correction["options"][correction["correct_index"]]["text"]
    assert correct.startswith("代码：")
    assert "```" not in correct
    assert all("\n" not in option["text"] for option in correction["options"])
