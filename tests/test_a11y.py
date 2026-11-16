"""Static accessibility guards for the offline export template.

The exported page is a single self-contained HTML file that learners open
without a server, so there is no CSS pipeline and no browser in the test path.
These checks read the template source and assert the properties that are cheap
to verify statically: colour contrast computed with the WCAG formula, landmark
rules, ARIA wiring, print overrides, and the offline/self-contained contract.

Anything requiring layout, focus traversal, or assistive technology is listed
as a known limitation in ``docs/maintainer-notes.md``.
"""

from __future__ import annotations

import re
from importlib.resources import files

import pytest

TEMPLATE_PATH = (
    files("deeptutor.review").joinpath("templates/review.html")
)
TEMPLATE = TEMPLATE_PATH.read_text(encoding="utf-8")


# --- WCAG 2.1 relative luminance / contrast ratio -------------------------


def _luminance(color: str) -> float:
    """Relative luminance of ``#rgb``/``#rrggbb`` (WCAG 2.1 §relative luminance)."""
    value = color.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        raise ValueError(f"not a hex colour: {color!r}")
    channels = []
    for offset in (0, 2, 4):
        raw = int(value[offset : offset + 2], 16) / 255
        channels.append(
            raw / 12.92 if raw <= 0.03928 else ((raw + 0.055) / 1.055) ** 2.4
        )
    return (
        0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    )


def contrast_ratio(foreground: str, background: str) -> float:
    """Contrast ratio between two opaque colours, in the range 1..21."""
    first, second = _luminance(foreground), _luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def _css_vars(block: str | None = None) -> dict[str, str]:
    """Parse ``--name:#rrggbb`` declarations from the first ``:root`` block."""
    source = TEMPLATE if block is None else block
    match = re.search(r":root\{([^}]*)\}", source)
    assert match, "template has no :root custom-property block"
    return {
        name: value
        for name, value in re.findall(r"--([\w-]+):(#[0-9a-fA-F]{3,8})", match.group(1))
    }


# Surfaces the template actually paints text on. Kept explicit (rather than
# parsed) because a new background colour should be a deliberate decision that
# adds a row here and re-runs the contrast assertion.
TEXT_SURFACES = {
    "card": "#ffffff",        # header.hero / item / formula / button
    "page": "#f0f1f3",        # body background
    "panel": "#f5f6f7",       # .answer, .summary, .sources, .formula
    "selected": "#eaf1f8",    # .rating[aria-pressed="true"]
    "callout_warn": "#fbf3e3",
    "callout_tip": "#eaf5ef",
    "callout_bad": "#f9eded",
}


def test_contrast_function_matches_wcag_reference() -> None:
    # Sanity: black on white is 21:1, a colour against itself is 1:1.
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0, abs=1e-6)
    assert contrast_ratio("#2457a7", "#2457a7") == pytest.approx(1.0, abs=1e-6)
    # Shorthand expands: #fff == #ffffff.
    assert contrast_ratio("#fff", "#000") == pytest.approx(21.0, abs=1e-6)


def test_body_text_and_muted_text_meet_aa_on_every_surface() -> None:
    variables = _css_vars()
    for name in ("ink", "muted"):
        colour = variables[name]
        for surface_name, surface in TEXT_SURFACES.items():
            ratio = contrast_ratio(colour, surface)
            assert ratio >= 4.5, (
                f"--{name} {colour} on {surface_name} {surface} is {ratio:.2f}:1,"
                " below the 4.5:1 required for normal-size text"
            )


def test_status_accents_meet_aa_on_every_surface() -> None:
    variables = _css_vars()
    for name in ("accent", "ok", "warn", "bad"):
        colour = variables[name]
        for surface_name, surface in TEXT_SURFACES.items():
            ratio = contrast_ratio(colour, surface)
            assert ratio >= 4.5, (
                f"--{name} {colour} on {surface_name} {surface} is {ratio:.2f}:1"
            )


def test_control_borders_meet_non_text_contrast() -> None:
    """WCAG 1.4.11: component boundaries need 3:1 against adjacent colours."""
    variables = _css_vars()
    line = variables["line"]
    for surface_name, surface in TEXT_SURFACES.items():
        ratio = contrast_ratio(line, surface)
        assert ratio >= 3.0, (
            f"--line {line} on {surface_name} {surface} is {ratio:.2f}:1,"
            " below the 3:1 required for UI component boundaries"
        )


def test_focus_ring_meets_non_text_contrast() -> None:
    variables = _css_vars()
    for surface_name, surface in TEXT_SURFACES.items():
        ratio = contrast_ratio(variables["accent"], surface)
        assert ratio >= 3.0, f"focus ring on {surface_name} is {ratio:.2f}:1"


def test_reduced_motion_disables_transitions_and_animations() -> None:
    assert "prefers-reduced-motion: reduce" in TEMPLATE
    match = re.search(
        r"@media \(prefers-reduced-motion: reduce\)\{(.*?)\n\}", TEMPLATE, re.S
    )
    assert match, "reduced-motion block must be a standalone @media rule"
    block = match.group(1)
    assert "transition-duration" in block
    assert "animation-duration" in block
    assert "animation-iteration-count" in block
    # The wildcard selector keeps future transitions covered without edits.
    assert "*::before" in block and "*::after" in block


def test_screen_focus_ring_is_not_disabled_by_print_styles() -> None:
    screen, _, after = TEMPLATE.partition("@media print")
    assert after, "template has no @media print block"
    assert ":focus-visible" in screen
    assert "outline:3px solid var(--accent)" in screen
    print_block = after.split("</style>", 1)[0]
    # Every ``outline:none`` in the file lives inside the print media query;
    # outside it the focus ring must stay intact.
    assert "outline:none" in print_block
    assert TEMPLATE.count("outline:none") == print_block.count("outline:none")


def test_figure_svg_gets_role_img_and_accessible_name() -> None:
    assert 'setAttribute("role", "img")' in TEMPLATE
    assert 'setAttribute("aria-label", figureTitle)' in TEMPLATE
    # Name falls back when the provider supplies no title.
    assert '"图"' in TEMPLATE
    # Decorative marking from upstream must not silence the new label.
    assert 'removeAttribute("aria-hidden")' in TEMPLATE


def test_placeholder_is_not_exposed_as_an_image() -> None:
    assert TEMPLATE.count('setAttribute("role", "note")') >= 2, (
        "figure and widget placeholders must both be notes, not graphics"
    )


def test_repeated_regions_are_live() -> None:
    # Every live region is attached in JS, so both spellings count: literal
    # markup attributes and ``setAttribute("aria-live", ...)`` calls.
    live_regions = TEMPLATE.count('aria-live="polite"') + len(
        re.findall(r'setAttribute\("aria-live",\s*"polite"\)', TEMPLATE)
    )
    assert live_regions >= 2
    # Slider readout, rating status, answer panel, correction feedback.
    assert 'output.setAttribute("aria-live", "polite")' in TEMPLATE
    assert 'answer.setAttribute("aria-live", "polite")' in TEMPLATE


def test_slider_announces_value_with_unit() -> None:
    assert 'input.setAttribute("aria-valuetext", text)' in TEMPLATE
    # The announced string is built from the value plus the widget unit.
    assert re.search(r'var unit = String\(\(widget && widget\.unit\) \|\| ""\)', TEMPLATE)
    assert re.search(r"toFixed\(precision\) \+ unit", TEMPLATE)


def test_interactive_targets_meet_minimum_touch_size() -> None:
    # 44 CSS px is the WCAG 2.5.5 / Apple HIG target size.
    assert TEMPLATE.count("min-height:44px") >= 2
    button_rule = re.search(r"\nbutton\{([^}]*)\}", TEMPLATE)
    assert button_rule and "min-height:44px" in button_rule.group(1), (
        "every <button> needs a 44px target; the base rule is the cheapest place"
    )


def test_rating_digits_only_apply_in_self_test_mode() -> None:
    block = TEMPLATE.split('if (key === "1" || key === "2" || key === "3")', 1)
    assert len(block) == 2
    following = block[1][:400]
    assert 'getAttribute("data-mode") !== "selftest"' in following


def test_escape_does_not_swallow_browser_level_actions() -> None:
    # Esc inside fullscreen (or any UA surface) must keep its default action.
    escape_block = TEMPLATE.split('if (key === "Escape" || key === "Esc")', 1)[1][:200]
    assert "document.fullscreenElement" in escape_block


def test_template_has_no_script_protocol_or_remote_assets() -> None:
    assert "javascript:" not in TEMPLATE.lower()
    # The only ``http`` occurrences allowed are the CSP meta tag and XML
    # namespace strings; neither is a network fetch.
    urls = [
        match.group(0)
        for match in re.finditer(r"https?://[^\s\"'<>]+", TEMPLATE)
        if not match.group(0).startswith(
            ("http://www.w3.org/", "https://www.w3.org/")
        )
    ]
    assert urls == []
    assert "@import" not in TEMPLATE
    assert "<link" not in TEMPLATE.lower()


def test_export_tokens_remain_a_single_occurrence() -> None:
    for token in ("{{TITLE}}", "{{PAYLOAD}}", "{{MATHJAX}}"):
        assert TEMPLATE.count(token) == 1, f"{token} drifted"


def test_document_declares_language_and_no_autofocus_trap() -> None:
    assert '<html lang="zh-CN">' in TEMPLATE
    assert "autofocus" not in TEMPLATE.lower()
    # The item is focusable so Enter/Esc work without a mouse.
    assert 'article.setAttribute("tabindex", "0")' in TEMPLATE


def test_runtime_svg_sanitizer_drops_external_references() -> None:
    """svgNode is the last stop before the DOM; it must strip external hrefs."""
    template = TEMPLATE
    match = re.search(r"function svgNode\(text\) \{(.*?)\n  \}\n", template, re.S)
    assert match is not None, "svgNode not found"
    body = match.group(1)
    assert '"xlink:href"' in body
    assert 'charAt(0) === "#"' in body
    assert "data:image" in body and "base64" in body
