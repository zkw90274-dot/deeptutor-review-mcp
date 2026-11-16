"""Static design contracts for the generated review page.

The visual brief is the Apple-style card system: 28px cards, 20px gaps, a
low-contrast grey page with near-white cards, continuous-curvature corners and
nested radii that shrink with the gap. These tests keep future edits from
silently drifting back to the old 12–14px document look, and pin the review
flow chrome that makes the page feel like a session rather than a printout.
"""

from __future__ import annotations

import re
from importlib.resources import files

TEMPLATE = (
    files("deeptutor.review")
    .joinpath("templates/review.html")
    .read_text(encoding="utf-8")
)
OVERVIEW = (
    files("deeptutor.review")
    .joinpath("templates/overview.html")
    .read_text(encoding="utf-8")
)


def _luminance(color: str) -> float:
    value = color.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    channels = []
    for offset in (0, 2, 4):
        raw = int(value[offset : offset + 2], 16) / 255
        channels.append(
            raw / 12.92 if raw <= 0.03928 else ((raw + 0.055) / 1.055) ** 2.4
        )
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _ratio(a: str, b: str) -> float:
    first, second = _luminance(a), _luminance(b)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def test_card_system_follows_the_visual_spec() -> None:
    assert "--radius:22px" in TEMPLATE
    assert "--gap:20px" in TEMPLATE
    assert "--radius-inner:14px" in TEMPLATE
    assert "corner-shape:squircle" in TEMPLATE
    # The sticky dock must not cover a card scrolled into view.
    assert "scroll-margin-top:148px" in TEMPLATE


def test_topbar_is_hidden_when_printing() -> None:
    print_block = TEMPLATE.partition("@media print")[2]
    assert ".topbar{display:none}" in print_block


def test_page_and_cards_are_low_contrast_neighbours() -> None:
    root = re.search(r":root\{([^}]*)\}", TEMPLATE).group(1)
    variables = {
        name: value
        for name, value in re.findall(r"--([\w-]+):(#[0-9a-fA-F]{6})", root)
    }
    # "A little bit of grey difference": the card floats without a hard edge.
    assert _ratio(variables["bg"], variables["card"]) < 1.5
    assert _ratio(variables["card"], variables["card-soft"]) < 1.5


def test_review_flow_chrome_exists() -> None:
    for element_id in (
        "session-dock",
        "segments",
        "complete",
        "start-review",
        "theme-toggle",
        "dock-progress",
        "dock-meta",
    ):
        assert f'id="{element_id}"' in TEMPLATE, element_id
    assert "function refreshSession" in TEMPLATE
    assert "just-completed" in TEMPLATE
    assert 'aria-valuenow' in TEMPLATE
    assert 'data-state' in TEMPLATE
    # "全部很稳" used to fire as soon as every item had any rating, so a pack
    # with one 还是不会 said 全部很稳 in the dock while the completion panel
    # said 5 很稳 / 1 下次再见.
    assert 'done===total&&good===total' in TEMPLATE
    assert '"study-aids"' in TEMPLATE
    assert '"aids-title"' in TEMPLATE
    assert "white-space:pre-line" in TEMPLATE


def test_answer_suffix_strip_covers_choice_items_too() -> None:
    # Choice items carry the same "（正确答案：B. …）" tail, and the answer line
    # above already shows it. The strip matches the whole suffix, so gating it
    # on the item kind only left the duplicate on screen.
    assert 'var answerSuffix = "（正确答案："' in TEMPLATE
    assert 'item.kind === "short_answer"' not in TEMPLATE


def test_each_item_has_a_mouse_answer_toggle() -> None:
    # Keyboard Enter and 显示全部答案 were the only reveal paths, so a mouse
    # user could not check one item. The per-item toggle flips its own label
    # and is hidden outside 自测 (where answers are already visible).
    for marker in (
        "reveal-toggle",
        "看答案",
        "收起答案",
        'aria-expanded',
        'body:not([data-mode="selftest"]) .item-actions',
    ):
        assert marker in TEMPLATE, marker


def test_quick_mode_hides_options_and_ratings() -> None:
    # phase-11 §2.2: 速览 keeps the stem, the answer and the explanation. The
    # options and the self-rating buttons belong to the self-test pass; without
    # this rule 速览 differed from 精读 only by the self-explain box.
    assert 'body[data-mode="quick"] .options' in TEMPLATE
    assert 'body[data-mode="quick"] .ratings' in TEMPLATE


def test_explanation_is_broken_into_paragraphs_and_lists() -> None:
    # A 500+ character 解析 used to be a single grey text node; learners saw a
    # wall of text. Blank lines must become paragraphs and "-"/"1." runs lists.
    for marker in (
        "function appendParagraphs",
        '"rt-p"',
        '"rt-list"',
        "block-label",
        ".rich-text p + p",
        ".rich-text .rt-list",
    ):
        assert marker in TEMPLATE, marker


def test_item_heading_typesets_math_and_truncates_safely() -> None:
    # The heading used to be a plain text node, so a concept name containing
    # $...$ stayed raw LaTeX forever (the typeset pass only sees .math nodes).
    # Its 59-char slice could also land inside a formula and strand a "$".
    # The cap must stay at the backend label bound (200): a 61-char concept
    # name was cut into a misleading ellipsis that looked like a lost stem.
    assert "heading.appendChild(mathText(" in TEMPLATE
    assert "document.createTextNode(conceptName" not in TEMPLATE
    assert "truncateMath(conceptName, 200)" in TEMPLATE
    assert "function truncateMath(text, max)" in TEMPLATE


def test_item_badges_are_chinese_for_every_enum_value() -> None:
    # The badge used to print the raw enum ("medium · single_choice") in an
    # otherwise Chinese page. The maps must cover every ItemKind and every
    # difficulty, otherwise a new pack silently falls back to English.
    for label in ("简单", "中等", "困难", "单选", "数值", "简答", "闪卡"):
        assert label in TEMPLATE, label
    assert 'DIFFICULTY_LABELS[item.difficulty]' in TEMPLATE
    assert 'KIND_LABELS[item.kind]' in TEMPLATE
    assert 'item.difficulty + " · " + item.kind' not in TEMPLATE


def test_theme_toggle_and_dark_palette_exist() -> None:
    assert 'html[data-theme="dark"]' in TEMPLATE
    assert "prefers-color-scheme: dark" in TEMPLATE
    assert "review-studio:theme" in TEMPLATE
    assert 'id="theme-toggle"' in TEMPLATE


def test_print_forces_light_and_hides_review_chrome() -> None:
    _, _, print_block = TEMPLATE.partition("@media print")
    assert 'html[data-theme="dark"]' in print_block
    assert "color-scheme:light" in print_block
    assert ".session-dock,.theme-toggle,.complete,.start-review,.kbd-help,.hero-actions,.mode-switch" in print_block
    assert "outline:none" in print_block
    assert ".figure{overflow:visible}" in print_block


def test_selected_rating_keeps_its_semantic_colour() -> None:
    # The pressed state is an inset ring, not a blue repaint that erases the
    # good/hard/again colour the learner just chose.
    assert 'button.rating[aria-pressed="true"]' not in TEMPLATE
    assert "box-shadow:inset 0 0 0 2px currentColor" in TEMPLATE


def test_dark_palette_keeps_text_readable() -> None:
    dark = {
        "ink": "#f2f4f7",
        "muted": "#9aa3af",
        "accent": "#8db8e8",
        "ok": "#7fd3a5",
        "warn": "#e0b166",
        "bad": "#f0a0a0",
        "line": "#7d8794",
    }
    surfaces = {
        "page": "#0a0b0d",
        "card": "#141619",
        "panel": "#1b1e22",
        "selected": "#1d2938",
        "warn_soft": "#2e2719",
        "bad_soft": "#301d1d",
    }
    for name in ("ink", "muted", "accent", "ok", "warn", "bad"):
        for surface_name, surface in surfaces.items():
            assert _ratio(dark[name], surface) >= 4.5, f"{name} on {surface_name}"
    for surface_name, surface in surfaces.items():
        assert _ratio(dark["line"], surface) >= 3.0, f"line on {surface_name}"


def test_surfaces_get_depth_from_highlight_and_texture_tokens() -> None:
    # The reference language is "surface slightly lighter than canvas plus a
    # hairline of light on top", not a heavier drop shadow. Without these the
    # page flattens back to plain white rectangles on grey.
    for token in ("--card-highlight:", "--dot:", "--glow:"):
        assert token in TEMPLATE, token
    assert "--card-highlight:inset 0 1px 0 rgba(255,255,255,.9)" in TEMPLATE
    assert 'html[data-theme="dark"]' in TEMPLATE and "--dot:rgba(255,255,255,.1)" in TEMPLATE
    # Hero composition: spotlight plus a masked dot field behind the copy.
    assert "header.hero::before" in TEMPLATE
    assert "header.hero::after" in TEMPLATE
    assert "mask-image:radial-gradient" in TEMPLATE
    # The dock is translucent glass, with an opaque fallback when unsupported.
    assert "backdrop-filter:blur(18px) saturate(1.25)" in TEMPLATE
    assert "@supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px)))" in TEMPLATE


def test_hero_uses_an_editorial_heading_and_pill_metadata() -> None:
    assert "font-size:clamp(2rem,4.2vw,3rem)" in TEMPLATE
    assert "letter-spacing:-.045em" in TEMPLATE
    # Metadata reads as discrete chips rather than a run-on sentence.
    assert ".meta span{display:inline-flex" in TEMPLATE
    assert ".eyebrow::before" in TEMPLATE
    # Phone layout: metadata becomes a full-width label/value list, and the
    # difficulty badge drops under the heading instead of squeezing it.
    assert ".meta{display:grid;grid-template-columns:1fr;gap:6px" in TEMPLATE
    assert ".meta span b{text-align:right}" in TEMPLATE
    assert ".item-head{display:block" in TEMPLATE
    assert ".badge{display:inline-flex;margin-top:10px;white-space:nowrap}" in TEMPLATE


def test_answer_panel_renders_a_comparison_table() -> None:
    for marker in (
        "var hasTable = !!aids.table",
        '"对照表"',
        "appendBlocks(tableBox, aids.table)",
        ".study-aids .study-table",
    ):
        assert marker in TEMPLATE, marker


def test_tables_and_code_blocks_are_first_class_rich_content() -> None:
    for marker in (
        "function appendBlocks",
        "function appendTable",
        "function appendCode",
        "table-wrap",
        "code-block",
        "TABLE_SEP",
        "appendBlocks(explain",
        "appendInlineText",
        "inline-code",
        "table-hint",
        "（正确答案：",
    ):
        assert marker in TEMPLATE, marker
    # A table/code block is not valid inside a <p>; the explanation is a div.
    assert 'el("div", "explain")' in TEMPLATE
    # Short-answer values can contain a code block or table too.
    assert "appendBlocks(value, item.answer_display)" in TEMPLATE
    # Print keeps code readable instead of clipping it.
    assert ".code-block{white-space:pre-wrap" in TEMPLATE


def test_hero_shows_long_term_progress_and_symbols_read_as_lines() -> None:
    # Pack progress (knowledge points), not the session counter in the dock.
    assert "data.progress" in TEMPLATE
    assert "个知识点" in TEMPLATE
    assert "个较稳" in TEMPLATE
    # The notes model may return a paragraph; export marks the block and the
    # page preserves its newlines so each definition gets its own line.
    assert ".symbols-callout{white-space:pre-line}" in TEMPLATE
    assert ".symbols-callout::first-line{font-weight:600}" in TEMPLATE
    assert 'block.format === "symbols"' in TEMPLATE


def test_overview_uses_the_same_card_system() -> None:
    assert "--radius:28px" in OVERVIEW
    assert "--gap:20px" in OVERVIEW
    assert "border-radius:var(--radius)" in OVERVIEW
    assert "Review Studio · 复习包" in OVERVIEW
    # The catalog follows the system preference and keeps the export CTA as the
    # card's only filled control.
    assert "prefers-color-scheme: dark" in OVERVIEW
    assert "corner-shape:squircle" in OVERVIEW
    assert "grid-template-columns:repeat(2,minmax(0,1fr))" in OVERVIEW
    assert "header.hero::after" in OVERVIEW
    assert ".card .cta{margin:auto 0 0" in OVERVIEW
