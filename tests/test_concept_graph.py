from __future__ import annotations

import re

from deeptutor.review.concept_graph import (
    MAX_LABEL_CHARS,
    MAX_NODES,
    ConceptGraphRender,
    render_concept_graph,
)
from deeptutor.review.models import Concept


def concept(
    concept_id: str,
    *,
    name: str = "",
    prerequisites: list[str] | None = None,
) -> Concept:
    return Concept(
        id=concept_id,
        name=name or concept_id,
        prerequisites=list(prerequisites or []),
    )


def _rects(svg: str) -> list[tuple[float, float]]:
    return [
        (float(x), float(y))
        for x, y in re.findall(r'<rect x="([\d.]+)" y="([\d.]+)"', svg)
    ]


def test_linear_chain_layers_go_left_to_right() -> None:
    graph = [
        concept("a"),
        concept("b", prerequisites=["a"]),
        concept("c", prerequisites=["b"]),
    ]
    rendered = render_concept_graph(graph, title="线性链")
    assert rendered.warning == ""
    assert rendered.svg.count("marker-end=") == 2
    assert rendered.svg.count("<rect ") == 3
    xs = [x for x, _ in _rects(rendered.svg)]
    assert xs[0] < xs[1] < xs[2]


def test_fork_draws_two_edges() -> None:
    rendered = render_concept_graph(
        [
            concept("root"),
            concept("left", prerequisites=["root"]),
            concept("right", prerequisites=["root"]),
        ]
    )
    assert rendered.warning == ""
    assert rendered.svg.count("marker-end=") == 2


def test_join_draws_two_edges() -> None:
    rendered = render_concept_graph(
        [
            concept("left"),
            concept("right"),
            concept("sink", prerequisites=["left", "right"]),
        ]
    )
    assert rendered.warning == ""
    assert rendered.svg.count("marker-end=") == 2
    xs = [x for x, _ in _rects(rendered.svg)]
    # left/right share the first column, sink sits in the second.
    assert xs[0] == xs[1]
    assert xs[2] > xs[1]


def test_isolated_node_is_a_dashed_root_without_edges() -> None:
    rendered = render_concept_graph([concept("lonely", name="孤立节点")])
    assert rendered.warning == ""
    assert "marker-end=" not in rendered.svg
    assert 'stroke-dasharray="5 3"' in rendered.svg
    assert "孤立节点" in rendered.svg


def test_only_roots_use_dashed_strokes() -> None:
    rendered = render_concept_graph(
        [concept("root"), concept("child", prerequisites=["root"])]
    )
    assert rendered.svg.count('stroke-dasharray="5 3"') == 1


def test_cycle_returns_placeholder_and_deduplicated_warning() -> None:
    graph = [
        concept("a", prerequisites=["c"]),
        concept("b", prerequisites=["a"]),
        concept("c", prerequisites=["b"]),
    ]
    first = render_concept_graph(graph)
    second = render_concept_graph(graph)
    assert "cycle detected" in first.warning
    assert "a -> b -> c -> a" in first.warning or "a -> c -> b -> a" in first.warning
    assert "图暂不可用" in first.svg
    assert first.svg == second.svg
    assert "marker-end=" not in first.svg


def test_unknown_prerequisite_is_reported_not_dropped() -> None:
    rendered = render_concept_graph([concept("a"), concept("b", prerequisites=["missing"])])
    assert "unknown prerequisite 'missing'" in rendered.warning
    assert "concept 'b'" in rendered.warning
    assert "图暂不可用" in rendered.svg


def test_self_prerequisite_is_reported_separately() -> None:
    rendered = render_concept_graph([concept("a", prerequisites=["a"])])
    assert "self prerequisite on concept 'a'" in rendered.warning
    assert "cycle" not in rendered.warning
    assert "图暂不可用" in rendered.svg


def test_oversize_graph_degrades_with_explicit_warning() -> None:
    graph = [concept(f"c{index:02d}") for index in range(MAX_NODES + 1)]
    rendered = render_concept_graph(graph)
    assert f"{MAX_NODES + 1} > {MAX_NODES}" in rendered.warning
    assert "too many concepts" in rendered.warning
    assert "图暂不可用" in rendered.svg


def test_empty_concepts_returns_placeholder() -> None:
    rendered = render_concept_graph([])
    assert rendered.warning == "concept graph needs at least one concept"
    assert "图暂不可用" in rendered.svg


def test_labels_escape_markup_and_keep_chinese() -> None:
    rendered = render_concept_graph(
        [concept("a", name="混合 <a> & 中文")], title="图 <测试> & 说明"
    )
    assert "混合 &lt;a&gt; &amp; 中文" in rendered.svg
    assert "a < b" not in rendered.svg
    assert "<测试>" not in rendered.svg
    assert "图 &lt;测试&gt; &amp; 说明" in rendered.svg


def test_long_labels_are_truncated_with_ellipsis() -> None:
    name = "这是一个非常非常长的概念名称用于测试截断行为"
    rendered = render_concept_graph([concept("a", name=name)])
    assert "…" in rendered.svg
    assert name not in rendered.svg
    assert len(name) > MAX_LABEL_CHARS


def test_same_input_renders_byte_identical_svg() -> None:
    graph = [
        concept("b", name="乙", prerequisites=["a"]),
        concept("a", name="甲"),
        concept("c", name="丙", prerequisites=["a", "b"]),
    ]
    first = render_concept_graph(graph, title="确定性")
    second = render_concept_graph(list(reversed(graph)), title="确定性")
    assert first.svg == second.svg


def test_duplicate_prerequisites_collapse_to_one_edge() -> None:
    rendered = render_concept_graph(
        [concept("a"), concept("b", prerequisites=["a", "a"])]
    )
    assert rendered.svg.count("marker-end=") == 1


def test_output_has_no_script_or_event_handlers() -> None:
    graph = [
        concept("a", name="<script>alert(1)</script>"),
        concept("b", prerequisites=["a"]),
    ]
    rendered = render_concept_graph(graph)
    assert "<script" not in rendered.svg.lower()
    assert re.search(r"\son[a-z]+\s*=", rendered.svg) is None


def test_arrow_marker_is_defined_and_referenced() -> None:
    rendered = render_concept_graph([concept("a"), concept("b", prerequisites=["a"])])
    assert "<defs>" in rendered.svg
    assert '<marker id="cg-arrow"' in rendered.svg
    assert 'marker-end="url(#cg-arrow)"' in rendered.svg


def test_svg_has_accessible_role_and_label() -> None:
    rendered = render_concept_graph([concept("a")], title="依赖图")
    assert 'role="img"' in rendered.svg
    assert 'aria-label="依赖图"' in rendered.svg
    assert "<title>依赖图</title>" in rendered.svg


def test_as_dict_matches_dataclass_fields() -> None:
    rendered = render_concept_graph([concept("a")])
    payload = rendered.as_dict()
    assert set(payload) == {"svg", "warning"}
    assert isinstance(rendered, ConceptGraphRender)
    assert payload["svg"] == rendered.svg
    assert payload["warning"] == rendered.warning


def test_canvas_height_is_capped() -> None:
    # 30 roots each with one child: 60 nodes, 30 per layer, two layers.
    graph = [concept(f"r{index:02d}") for index in range(30)]
    graph += [
        concept(f"c{index:02d}", prerequisites=[f"r{index:02d}"])
        for index in range(30)
    ]
    rendered = render_concept_graph(graph)
    assert rendered.warning == ""
    match = re.search(r'viewBox="0 0 (\d+) (\d+)"', rendered.svg)
    assert match is not None
    assert int(match.group(1)) == 720
    assert int(match.group(2)) == 900


def test_single_layer_density_degrades_instead_of_tiny_text() -> None:
    # A single flat layer cannot shrink indefinitely inside the height cap.
    rendered = render_concept_graph(
        [concept(f"c{index:02d}") for index in range(MAX_NODES)]
    )
    assert "too dense to render legibly" in rendered.warning
    assert "图暂不可用" in rendered.svg


def test_full_width_labels_respect_the_node_width() -> None:
    """A CJK label must not exceed the node it sits in.

    The previous character-count budget treated a full-width glyph as 0.95em,
    so a fully CJK label overflowed a 160px node by a few pixels.
    """
    name = "品质因数与通频带选择性的完整关系"  # 15 full-width glyphs
    rendered = render_concept_graph(
        [concept("a", name=name), concept("b", name="后继", prerequisites=["a"])]
    )
    assert rendered.warning == ""
    assert "…" in rendered.svg, "an over-budget CJK label must be truncated"

    rect = re.search(r'<rect x="[\d.-]+" y="[\d.-]+" width="([\d.]+)"', rendered.svg)
    assert rect is not None
    width = float(rect.group(1))
    label = re.search(r'<text[^>]*>([^<]+)</text>', rendered.svg)
    assert label is not None
    # 11px font, so the visible text must fit in ``width`` minus 10px padding.
    estimated = len(label.group(1)) * 11
    assert estimated <= width - 8


def test_label_units_count_full_width_glyphs_as_one_em() -> None:
    from deeptutor.review.text_metrics import label_units

    assert abs(label_units("ab") - 1.2) < 1e-9
    assert abs(label_units("品质") - 2.0) < 1e-9
