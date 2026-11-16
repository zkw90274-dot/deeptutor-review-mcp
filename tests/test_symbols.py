"""The pre-answer symbol callout must read as a list, not a paragraph.

The notes model is asked for ``符号：含义，单位`` but sometimes returns a
summary sentence plus the same symbols restated per concept. The normaliser
is the deterministic boundary between that free-form answer and the page.
"""

from __future__ import annotations

from deeptutor.review.symbols import (
    MAX_SYMBOL_CHARS,
    MAX_SYMBOL_LINES,
    SYMBOL_HEADING,
    normalize_symbol_callout,
)

# Shape taken from the real 高频 · 第1章_绪论 pack: a summary sentence, a
# prose preamble before each group, and f_s/f_0 repeated across concepts.
PARAGRAPH = (
    "符号与单位：超外差接收机把任意电台信号统一变换为固定中频"
    "（调幅 465 kHz、调频 10.7 MHz）后再放大解调：$f_s$：接收到的射频频率，Hz；"
    "$f_0$：本机振荡频率，Hz；$f_I$：中频频率，Hz，$f_I=\\vert{}f_0-f_s\\vert{}$；"
    "发射机载波支路中倍频器的作用及其必须置于调制之前的理由：$n$：倍频次数，无量纲；"
    "超外差接收机的完整信号链与固定中频带来的优势：$f_s$：射频频率，Hz；"
    "$f_0$：本振频率，Hz"
)


def test_paragraph_becomes_one_definition_per_line() -> None:
    lines = normalize_symbol_callout(PARAGRAPH).splitlines()
    assert lines[0] == SYMBOL_HEADING
    assert lines[1] == "$f_s$：接收到的射频频率，Hz"
    assert "$n$：倍频次数，无量纲" in lines
    # The prose preamble and the repeated definitions are gone.
    assert "超外差接收机把任意电台信号" not in "\n".join(lines)
    assert "发射机载波支路中倍频器的作用" not in "\n".join(lines)
    assert sum("$f_s$" in line for line in lines) == 1


# Shape taken from the real 电磁场 · 第3章 pack: the model answered with a
# Python list literal. The opening ``['`` was stripped upstream, so the
# callout kept the quote-comma-quote separators and the trailing bracket, and
# every definition collapsed onto one unreadable line.
LIST_LITERAL = (
    "符号与单位\n"
    "$v_p$：相速，m/s', '$\\varepsilon_r$：相对介电常数，无量纲', "
    "'$\\mu$、$\\varepsilon$：介质磁导率、介电常数，H/m、F/m']"
)


def test_list_literal_answer_becomes_one_definition_per_line() -> None:
    lines = normalize_symbol_callout(LIST_LITERAL).splitlines()
    assert lines[0] == SYMBOL_HEADING
    assert lines[1] == "$v_p$：相速，m/s"
    assert "$\\varepsilon_r$：相对介电常数，无量纲" in lines
    assert any("介质磁导率" in line for line in lines)
    body = "\n".join(lines)
    assert "'" not in body and "]" not in body


def test_list_literal_spliced_mid_note_is_cleaned() -> None:
    # The real IIR pack arrived as one long line with the closing ``']`` in the
    # middle: ``... rad/s'] $s_k$：模拟极点 ``. It is not at the end of the
    # string, so an anchored pattern left the bracket on the learner's card.
    text = (
        "符号与单位\n"
        "$\\Omega_c$：3 dB截止角频率，rad/s'] $s_k$：模拟极点，单位 $\\mathrm{s}^{-1}$"
    )
    out = normalize_symbol_callout(text)
    assert "']" not in out
    assert "]" not in out
    assert "3 dB截止角频率" in out
    assert "模拟极点" in out


def test_list_literal_does_not_split_a_legitimate_comma() -> None:
    # ``$a,b$：内、外导体半径`` must survive the unpacking untouched.
    text = "符号与单位\n$a,b$：内、外导体半径，m\n$b/a$：内外半径比，无量纲"
    assert normalize_symbol_callout(text).splitlines()[1] == "$a,b$：内、外导体半径，m"


def test_doubled_backslash_commands_are_collapsed() -> None:
    # The notes answer can arrive double-escaped. MathJax reads ``\\Omega`` as
    # a line break plus the word "Omega", so the learner sees italics instead
    # of Ω — the mobile screenshot showed exactly that.
    text = "符号与单位\n$Z_0$：特性阻抗，$\\\\Omega$"
    out = normalize_symbol_callout(text)
    assert "$\\Omega$" in out
    assert "\\\\" not in out


def test_normalizer_is_idempotent() -> None:
    once = normalize_symbol_callout(PARAGRAPH)
    assert normalize_symbol_callout(once) == once


def test_plain_symbols_and_formula_fallbacks_stay_one_per_line() -> None:
    text = normalize_symbol_callout("f_s — 射频，Hz；Q — 品质因数；n：倍频次数")
    assert text.splitlines()[1:] == ["f_s — 射频，Hz", "Q — 品质因数", "n：倍频次数"]


def test_trailing_relation_is_kept() -> None:
    lines = normalize_symbol_callout("$f_I$：中频频率，Hz，$f_I=|f_0-f_s|$").splitlines()
    assert lines[1].endswith("$f_I=|f_0-f_s|$")


def test_prose_without_definitions_is_left_alone() -> None:
    prose = "符号与单位：先分清射频与中频，再讨论混频。"
    assert normalize_symbol_callout(prose) == prose


def test_cap_keeps_twelve_definitions() -> None:
    text = normalize_symbol_callout(
        "符号与单位：" + "；".join(f"s{i}：含义{i}" for i in range(30))
    )
    assert len(text.splitlines()) == MAX_SYMBOL_LINES + 1


def test_char_budget_drops_whole_entries_only() -> None:
    long_entries = "；".join(f"s{i}：" + "含义" * 60 for i in range(12))
    text = normalize_symbol_callout("符号与单位：" + long_entries)
    assert len(text) <= MAX_SYMBOL_CHARS
    assert len(text.splitlines()) > 1
    assert all(line.endswith("含义") for line in text.splitlines()[1:])
