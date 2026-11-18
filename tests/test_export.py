from __future__ import annotations

import re
from datetime import datetime, timedelta
from importlib.resources import files
from zoneinfo import ZoneInfo

import pytest

from deeptutor.learning.models import RepetitionState

from deeptutor.review.export import (
    _needs_mathjax,
    _payload,
    _safe_json,
    build_html,
    normalize_math_placeholders,
    normalize_prose_math,
)
from deeptutor.review.fixture import build_fixture_spec
from deeptutor.review.models import ReviewSpec
from deeptutor.review.qa import run_gates
from deeptutor.review.service import QualityGateFailed, ReviewService
from deeptutor.review.queue import concept_state_map

NOW = 1_800_000_000.0


def _grade(service: ReviewService):
    spec = build_fixture_spec()
    service.put_spec(spec)
    return spec


def test_export_is_byte_identical_for_the_same_now(service: ReviewService) -> None:
    spec = _grade(service)
    first = service.export(spec.id, now=NOW)
    second = service.export(spec.id, now=NOW)
    assert first == second
    assert first.encode("utf-8") == second.encode("utf-8")


def test_export_snapshot_is_explicit_and_stable(service: ReviewService) -> None:
    spec = _grade(service)
    without = service.export(spec.id, include_snapshot=False, now=NOW)
    assert '"snapshot":null' in without
    first = service.export(spec.id, include_snapshot=True, now=NOW)
    second = service.export(spec.id, include_snapshot=True, now=NOW)
    assert first == second
    assert '"generated_at"' in first
    assert '"concepts"' in first


def test_export_renders_a_legacy_symbol_paragraph_as_a_list() -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["sections"][0]["blocks"].insert(
        1,
        {
            "kind": "callout",
            "tone": "info",
            "text": "符号与单位：先认符号：$f_s$：射频，Hz；$f_s$：重复；$n$：次数",
        },
    )
    payload["content_hash"] = ""
    spec = ReviewSpec.model_validate(payload)
    states = {
        concept.id: RepetitionState(next_review_at=NOW) for concept in spec.concepts
    }
    report = run_gates(spec)
    assert report.passed, report
    data = _payload(
        spec, states, now=NOW, include_snapshot=False, quality=report
    )
    callout = next(
        block for block in data["sections"][0]["blocks"] if block.get("format") == "symbols"
    )
    # Stored text keeps the paragraph, the exported page gets the list.
    assert callout["text"] == "符号与单位\n$f_s$：射频，Hz\n$n$：次数"


def test_g3_ignores_dollar_signs_inside_fenced_code() -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    # A fenced block is rendered as plain text, so ``$x|y$`` is code, not a
    # formula that must escape its pipe. Appending it must not fail G3.
    payload["items"][0]["explanation"] += "\n\n```asm\nMOV A, $x|y$\n```"
    payload["content_hash"] = ""
    spec = ReviewSpec.model_validate(payload)
    report = run_gates(spec)
    assert report.passed, report


def test_math_placeholder_runs_become_visible_blanks() -> None:
    assert (
        normalize_math_placeholders(r"$$\Omega=____$$")
        == r"$$\Omega=\underline{\hspace{2em}}$$"
    )


def test_single_subscripts_and_fenced_code_are_untouched() -> None:
    assert normalize_math_placeholders(r"$x_1 + x_2$") == r"$x_1 + x_2$"
    fenced = "```\n$x=____$\n```"
    assert normalize_math_placeholders(fenced) == fenced


def test_math_placeholder_normalisation_is_idempotent() -> None:
    once = normalize_math_placeholders(r"$a=____$")
    assert normalize_math_placeholders(once) == once


def test_bare_tex_prose_fields_get_delimiters() -> None:
    assert normalize_prose_math(r"H(j\omega)") == r"$H(j\omega)$"
    assert (
        normalize_prose_math(r"f_c = \frac{1}{2\pi RC}", pure_math_only=True)
        == r"$f_c = \frac{1}{2\pi RC}$"
    )


def test_bare_tex_rule_skips_paths_and_chinese_prose() -> None:
    assert normalize_prose_math(r"C:\Users") == r"C:\Users"
    prose = r"截止频率是 f_c = \frac{1}{2\pi RC}"
    assert normalize_prose_math(prose, pure_math_only=True) == prose
    assert normalize_prose_math(r"$H(j\omega)$") == r"$H(j\omega)$"


def test_export_payload_normalises_render_copy_only() -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    flashcard = next(item for item in payload["items"] if item["id"] == "item-quality-flashcard")
    flashcard["stem"] = r"求 $$Q=____$$ 的值"
    flashcard["answer"]["text"] = r"f_c = \frac{1}{2\pi RC}"
    payload["concepts"][0]["formulas"][0]["variables"] = {r"H(j\omega)": "频率响应"}
    spec = ReviewSpec.model_validate(payload)
    report = run_gates(spec)
    assert report.passed, report.model_dump()

    states = {
        concept.id: RepetitionState(next_review_at=NOW) for concept in spec.concepts
    }
    data = _payload(spec, states, now=NOW, include_snapshot=False, quality=report)

    item_payload = next(
        block["item"]
        for block in data["sections"][0]["blocks"]
        if block.get("kind") == "item" and block["item"]["id"] == flashcard["id"]
    )
    assert "____" not in item_payload["stem"]
    assert r"\underline{\hspace{2em}}" in item_payload["stem"]
    assert item_payload["answer_display"] == r"$f_c = \frac{1}{2\pi RC}$"
    assert "$H(j\\omega)$" in data["concepts"][spec.concepts[0].id]["formulas"][0]["variables"]
    # The correction choices reuse the answer text; they are rendered by the
    # same rich-text path, so they need the same wrapping.
    assert any(
        option["text"] == r"$f_c = \frac{1}{2\pi RC}$"
        for option in item_payload["correction"]["options"]
    )
    # G11 is generator guidance with internal paths, not a learner caveat.
    assert not any(
        warning.startswith("G11-bare-tex:") for warning in data["quality"]["warnings"]
    )

    # The stored spec is untouched: only the rendered copy is normalised.
    stored_item = next(item for item in spec.items if item.id == flashcard["id"])
    assert "____" in stored_item.stem
    assert stored_item.answer.text == r"f_c = \frac{1}{2\pi RC}"


def test_export_progress_counts_concepts_not_items() -> None:
    spec = build_fixture_spec()
    first, second = spec.concepts
    states = {
        first.id: RepetitionState(next_review_at=NOW, review_count=3),
        second.id: RepetitionState(next_review_at=NOW),
    }
    report = run_gates(spec)
    data = _payload(
        spec, states, now=NOW, include_snapshot=False, quality=report
    )
    assert data["progress"] == {"total": 2, "reviewed": 1, "strong": 1}


def test_export_contains_items_and_tomorrow(service: ReviewService) -> None:
    spec = _grade(service)
    html = service.export(spec.id, now=NOW)
    for item in spec.items:
        assert item.stem in html
    tomorrow = (
        datetime.fromtimestamp(NOW, tz=ZoneInfo("Asia/Shanghai")).date()
        + timedelta(days=1)
    ).isoformat()
    assert tomorrow in html
    assert "还是不会" in html and "有点模糊" in html and "会了" in html
    assert "第2章 2.1 串联谐振" in html


def test_export_has_no_external_resource_references(service: ReviewService) -> None:
    spec = _grade(service)
    html = service.export(spec.id, now=NOW)
    # MathJax contains W3C namespace strings, a CSP-blocked SRE fallback and
    # an attribution link in its (disabled) help menu. A blanket http(s) scan
    # is wrong; no resource may be *loaded* from the network.
    assert 'src="http' not in html
    assert "src='http" not in html
    assert "url(http" not in html
    assert "@import" not in html
    assert "<link" not in html
    assert len(html.encode("utf-8")) <= 8_000_000


def test_export_inlines_mathjax(service: ReviewService) -> None:
    spec = _grade(service)
    html = service.export(spec.id, now=NOW)
    assert "window.MathJax" in html
    assert "typesetPromise" in html
    assert "mjx-container" in html
    # The config block only exists inside the vendored bundle, so this proves
    # the bundle itself was injected rather than just referenced.
    assert "loader:{load:[]}" in html
    # \oiint / \oiiint live in the esint extension, which an offline page
    # cannot autoload; the config has to define them or the formula degrades
    # to raw text. Same for any future page that uses the closed-surface sign.
    assert "macros:{oiint:" in html
    assert "oiiint:" in html
    assert len(html.encode("utf-8")) > 2_000_000


def test_mathjax_config_keeps_its_section_separators() -> None:
    # A missing comma after the macros block turned the JS object into
    # ``...}}svg:{...}``: MathJax never initialised and every formula on every
    # page stayed raw. Braces balance alone is not enough — check the join.
    from deeptutor.review.export import MATHJAX_CONFIG

    assert MATHJAX_CONFIG.count("{") == MATHJAX_CONFIG.count("}")
    assert "macros:{oiint:" in MATHJAX_CONFIG
    assert "}},svg:" in MATHJAX_CONFIG


def _plain_spec() -> ReviewSpec:
    """The fixture with every formula removed: text only, no math markers."""
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    for concept in payload["concepts"]:
        concept["formulas"] = []
    # The fixture's flashcard answer is bare TeX; the exporter now wraps such
    # answers in $...$ (a math marker), so a genuinely math-free page needs
    # plain prose here as well.
    for item in payload["items"]:
        if item["answer"]["text"]:
            item["answer"]["text"] = "品质因数 Q 的定义式"
    return ReviewSpec.model_validate(payload)


def test_export_skips_the_mathjax_bundle_without_math(
    service: ReviewService,
) -> None:
    spec = _plain_spec()
    service.put_spec(spec)
    html = service.export(spec.id, now=NOW)
    assert "loader:{load:[]}" not in html
    assert "公式由内嵌 MathJax 渲染" not in html
    assert "自评仅保存在本机浏览器" in html
    assert len(html.encode("utf-8")) < 200_000
    # The runtime stays: a missing renderer only skips typesetting, never the
    # print-ready flag or the rest of the page script.
    assert 'document.body.classList.add("mathjax-ready");' in html


def test_needs_mathjax_sees_formulas_and_widget_formulas() -> None:
    assert _needs_mathjax({"concepts": {}, "widgets": {}}) is False
    assert _needs_mathjax(
        {"concepts": {"c": {"formulas": [{"latex": "x^2"}]}}, "widgets": {}}
    ) is True
    assert _needs_mathjax(
        {"concepts": {}, "widgets": {"w": {"formula_latex": "x^2"}}}
    ) is True
    assert _needs_mathjax(
        {
            "concepts": {},
            "widgets": {},
            "sections": [{"blocks": [{"text": "求 $x$ 的值"}]}],
        }
    ) is True


def test_export_css_and_app_js_budgets() -> None:
    template = (
        files("deeptutor.review")
        .joinpath("templates/review.html")
        .read_text(encoding="utf-8")
    )
    style = re.search(r"<style>(.*?)</style>", template, re.S)
    assert style is not None
    assert len(style.group(1).encode("utf-8")) <= 60_000
    scripts = re.findall(r"<script>(.*?)</script>", template, re.S)
    assert len(scripts) == 1
    # P2 study aids (~1 KB), P5 table/code blocks (~4 KB), the explanation
    # paragraph/list splitter (~1.5 KB) and the emphasis pass (~1.5 KB) are the
    # largest additions; the per-item answer toggle adds a few hundred bytes.
    # Still far below a typical bundle — keep the offline page's JS under 56 KB.
    assert len(scripts[0].encode("utf-8")) <= 56_000


def test_payload_escape_prevents_script_breakout() -> None:
    escaped = _safe_json({"text": "</script><script>alert(1)</script>"})
    assert "</script>" not in escaped
    assert "\\u003c/script\\u003e" in escaped


def test_title_cannot_inject_a_template_token(service: ReviewService) -> None:
    spec = build_fixture_spec()
    payload = spec.model_dump(mode="json")
    payload["title"] = "{{PAYLOAD}} 恶意标题"
    payload["content_hash"] = ""
    altered = ReviewSpec.model_validate(payload)
    service.put_spec(altered)
    html = service.export(altered.id, now=NOW)
    # The title is escaped inside the <title> element; the same braces in the
    # JSON payload are data and must not trigger a second template pass.
    assert "<title>&#123;&#123;PAYLOAD&#125;&#125; 恶意标题</title>" in html
    assert '"title":"{{PAYLOAD}} 恶意标题"' in html


def test_orphan_items_are_not_dropped(service: ReviewService) -> None:
    spec = build_fixture_spec()
    payload = spec.model_dump(mode="json")
    payload["sections"][0]["blocks"] = [
        block
        for block in payload["sections"][0]["blocks"]
        if block.get("item_id") != "item-resonance-numeric"
    ]
    payload["content_hash"] = ""
    altered = ReviewSpec.model_validate(payload)
    service.put_spec(altered)
    html = service.export(altered.id, now=NOW)
    assert "某串联谐振回路" in html
    assert "其他题目" in html


def test_executable_content_is_blocked_before_export(service: ReviewService) -> None:
    spec = build_fixture_spec()
    payload = spec.model_dump(mode="json")
    payload["sections"][0]["blocks"][0]["text"] = "</script><script>alert(1)</script>"
    payload["content_hash"] = ""
    altered = ReviewSpec.model_validate(payload)
    report = run_gates(altered)
    assert report.passed is False
    with pytest.raises(QualityGateFailed):
        service.put_spec(altered)


def test_export_requires_passing_quality(service: ReviewService) -> None:
    spec = _grade(service)
    states = concept_state_map(spec, service.store, now=NOW)
    failed = run_gates(spec)
    failed.passed = False
    with pytest.raises(ValueError):
        build_html(spec, states, now=NOW, quality=failed)


def test_export_includes_deterministic_correction(service: ReviewService) -> None:
    from deeptutor.review.correction import build_correction

    spec = _grade(service)
    for item in spec.items:
        correction = build_correction(spec, item)
        assert 2 <= len(correction["options"]) <= 3
        assert 0 <= correction["correct_index"] < len(correction["options"])
        assert correction["retry_after_items"] == 5
    html = service.export(spec.id, now=NOW)
    assert "选一个关键结论" in html
    assert '"correction"' in html
    assert html.count('"retry_after_items"') >= len(spec.items)


def test_export_contains_reading_mode_controls(service: ReviewService) -> None:
    spec = _grade(service)
    html = service.export(spec.id, now=NOW)
    assert 'data-mode="selftest"' in html
    assert 'data-mode="quick"' in html
    assert 'data-mode="deep"' in html
    assert 'self-explain' in html
    assert "用自己的话写一句" in html
    assert "review-studio:" in html


def test_export_renders_tikz_through_provider(service: ReviewService, monkeypatch) -> None:
    from deeptutor.review import tikz
    from deeptutor.review.models import FigureSpec

    monkeypatch.setattr(
        tikz,
        "render_tikz_document",
        lambda source, title="", check_layout=True: tikz.TikzRender(
            svg='<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>'
        ),
    )
    spec = _grade(service)
    spec.figures.append(
        FigureSpec(
            id="fig-tikz-export",
            kind="tikz",
            title="TikZ 导出",
            tikz=r"\draw (0,0) -- (1,1);",
        )
    )
    payload = spec.model_dump(mode="json")
    payload["content_hash"] = ""
    spec = ReviewSpec.model_validate(payload)
    service.put_spec(spec)
    html = service.export(spec.id, now=NOW)
    assert "fig-tikz-export" in html
    assert "TikZ" in html
    assert "\\draw (0,0)" not in html


def test_export_payload_lists_quality_warnings_and_source_gaps(
    service: ReviewService,
) -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    concept = next(
        entry for entry in payload["concepts"] if entry["id"] == "c-quality-factor"
    )
    concept["coverage"] = "supplement"
    concept["source_ids"] = []
    payload["quality"]["warnings"] = [
        "c-quality-factor: supplement without a source"
    ]
    spec = ReviewSpec.model_validate(payload)
    service.put_spec(spec)

    page = service.export(spec.id, now=NOW)

    assert (
        '"source_gaps":[{"coverage":"supplement","id":"c-quality-factor"' in page
    )
    assert "c-quality-factor: supplement without a source" in page
    assert "质量与来源提示" in page
    assert "以下概念没有来源锚点" in page


def test_export_stops_tikz_after_the_render_budget(
    service: ReviewService, monkeypatch
) -> None:
    from deeptutor.review import export as export_module
    from deeptutor.review import tikz
    from deeptutor.review.models import FigureSpec

    calls: list[int] = []
    monkeypatch.setattr(
        tikz, "render_tikz_document", lambda *args, **kwargs: calls.append(1)
    )
    monkeypatch.setattr(export_module, "MAX_TIKZ_RENDER_SECONDS", 0.0)
    spec = _grade(service)
    for index in range(2):
        spec.figures.append(
            FigureSpec(
                id=f"fig-budget-{index}",
                kind="tikz",
                title=f"预算图{index}",
                tikz=r"\draw (0,0) -- (1,1);",
            )
        )
    payload = spec.model_dump(mode="json")
    payload["content_hash"] = ""
    spec = ReviewSpec.model_validate(payload)
    service.put_spec(spec)

    page = service.export(spec.id, now=NOW)

    assert calls == []
    # The warning is rendered both on the figure card and in the summary.
    assert page.count("tikz render budget exhausted") >= 2


def _linked_spec(service: ReviewService) -> ReviewSpec:
    """Fixture spec whose concepts form a two-node prerequisite chain."""
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    for concept in payload["concepts"]:
        if concept["id"] == "c-quality-factor":
            concept["prerequisites"] = ["c-series-resonance"]
    spec = ReviewSpec.model_validate(payload)
    service.put_spec(spec)
    return spec


def test_export_embeds_concept_graph_when_prerequisites_exist(
    service: ReviewService,
) -> None:
    spec = _linked_spec(service)
    html = service.export(spec.id, now=NOW)
    assert '"concept_graph"' in html
    assert '"svg"' in html
    # Payloads are JSON-escaped, so markup arrives as ``<marker``; that is
    # the whole point of ``_safe_json`` and a learner title cannot break out.
    assert "\\u003cmarker" in html
    assert "概念依赖" in html


def test_concept_graph_renders_real_dag_content(service: ReviewService) -> None:
    spec = _linked_spec(service)
    html = service.export(spec.id, now=NOW)
    # Both concept names must reach the page as labels, not ids, and the graph
    # must come through without a warning.
    assert "\\u003ctext" in html
    assert '"warning":""' in html.replace(" ", "")


def test_export_skips_concept_graph_without_edges(service: ReviewService) -> None:
    # The fixture has no prerequisites: a column of disconnected boxes would
    # restate the concept list, so the block must be omitted entirely.
    spec = _grade(service)
    html = service.export(spec.id, now=NOW)
    assert '"concept_graph":null' in html.replace(" ", "")


def test_concept_graph_failure_is_a_warning_not_a_crash(
    service: ReviewService, monkeypatch
) -> None:
    from deeptutor.review import concept_graph

    monkeypatch.setattr(
        concept_graph,
        "render_concept_graph",
        lambda concepts, title="": (_ for _ in ()).throw(RuntimeError("boom")),
    )
    spec = _linked_spec(service)
    html = service.export(spec.id, now=NOW)
    assert "concept graph render failed: RuntimeError" in html
    assert '"concept_graph":null' not in html.replace(" ", "")


def _review_template() -> str:
    return (
        files("deeptutor.review")
        .joinpath("templates/review.html")
        .read_text(encoding="utf-8")
    )


def _media_block(css: str, header: str) -> str:
    """Return the body of the first ``@media`` block introduced by ``header``."""
    start = css.index(header)
    open_brace = css.index("{", start)
    depth = 0
    for index in range(open_brace, len(css)):
        if css[index] == "{":
            depth += 1
        elif css[index] == "}":
            depth -= 1
            if depth == 0:
                return css[open_brace + 1 : index]
    raise AssertionError(f"unbalanced media block: {header}")


def test_narrow_screen_figures_scroll_instead_of_shrinking(
    service: ReviewService,
) -> None:
    # At 375px a 640-unit viewBox scales to ~0.47x, which makes 10px labels
    # unreadable. Narrow screens must scroll the figure instead of shrinking it.
    template = _review_template()
    narrow = _media_block(template, "@media (max-width:640px)")
    # The rule now also makes the scroll affordance visible; match the
    # behaviour, not the exact declaration list.
    assert ".figure{overflow-x:auto;" in narrow
    assert "scrollbar-width:thin" in narrow
    assert ".figure svg{min-width:520px}" in narrow

    spec = _grade(service)
    html = service.export(spec.id, now=NOW)
    assert ".figure{overflow-x:auto;" in html
    assert ".figure svg{min-width:520px}" in html


def test_narrow_screens_scroll_long_inline_formulas(service: ReviewService) -> None:
    # A single long inline formula (e.g. a sampled-signal expression) can be
    # 600px wide at a 375px viewport; without a scroll container the whole
    # page pans sideways.
    template = _review_template()
    narrow = _media_block(template, "@media (max-width:640px)")
    assert "mjx-container{max-width:100%;overflow-x:auto;overflow-y:hidden}" in narrow
    print_block = _media_block(template, "@media print")
    assert "mjx-container{max-width:100%;overflow:visible}" in print_block

    spec = _grade(service)
    html = service.export(spec.id, now=NOW)
    assert "mjx-container{max-width:100%;overflow-x:auto;overflow-y:hidden}" in html
    assert "mjx-container{max-width:100%;overflow:visible}" in html


def test_print_media_resets_narrow_figure_rules(service: ReviewService) -> None:
    # Printing must not inherit the narrow-screen scroll box: paper width is
    # authoritative and a horizontal crop would silently lose chart content.
    template = _review_template()
    print_block = _media_block(template, "@media print")
    assert ".figure{overflow:visible}" in print_block
    assert ".figure svg{min-width:0}" in print_block
    # A narrow viewport can also match ``print``; the reset only wins because
    # it comes later in source order at equal specificity.
    assert template.index("@media print") > template.index("@media (max-width:640px)")

    spec = _grade(service)
    html = service.export(spec.id, now=NOW)
    assert ".figure{overflow:visible}" in html
    assert ".figure svg{min-width:0}" in html


