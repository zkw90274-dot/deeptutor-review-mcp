"""Preparing a living-book drawing for another page.

A pack inlines the same SVG the book shows, but a verbatim copy breaks in four
measured ways: the ``<?xml`` declaration is illegal as raw HTML, glyph ids
collide when two drawings share a page, machine-precision coordinates inflate
the file, and a drawing carrying a script or an external link would turn an
offline page into something else. ``deeptutor.review.figures`` answers each,
and these tests pin the answers without needing a deployment's real books.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deeptutor.review import figures, materials

SVG = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
    'viewBox="0 0 100 60">'
    "<defs><g id=\"glyph-0-0\"><path d=\"M 3.125 -0.328125 L 4.000000 0\"/></g>"
    "<clipPath id=\"clip-1\"><rect x=\"0.000000\" y=\"0.000000\" width=\"10.000000\" "
    "height=\"10.000000\"/></clipPath></defs>"
    "<use xlink:href=\"#glyph-0-0\" x=\"5.747439\" y=\"12.421036\"/>"
    "<path d=\"M 1.5 2.5 L 9.25 4.125\" stroke=\"url(#clip-1)\"/>"
    "</svg>"
)


def test_prepare_svg_drops_the_declaration_and_keeps_one_root() -> None:
    out = figures.prepare_svg(SVG, "f0-0-", "图 1-1 波形")
    assert out is not None
    assert "<?xml" not in out
    assert out.startswith("<svg")
    assert out.count("<svg") == 1


def test_prepare_svg_namespaces_every_id_and_reference() -> None:
    out = figures.prepare_svg(SVG, "f3-1-", "图 1-1 波形")
    assert out is not None
    assert 'id="glyph-0-0"' not in out
    assert 'id="f3-1-glyph-0-0"' in out
    assert 'xlink:href="#f3-1-glyph-0-0"' in out
    assert 'url(#f3-1-clip-1)' in out


def test_prepare_svg_rounds_geometry_but_not_printed_text() -> None:
    svg = SVG.replace("</svg>", '<text x="12.345678">403.649845</text></svg>')
    out = figures.prepare_svg(svg, "f0-0-", "图 1-1")
    assert out is not None
    # Five-or-more decimals in a geometry attribute are rounded…
    assert 'x="12.35"' in out
    # …while a number printed as page text keeps every digit.
    assert "403.649845" in out


def test_prepare_svg_adds_a_label_without_touching_an_existing_one() -> None:
    out = figures.prepare_svg(SVG, "f0-0-", "图 1-1 调幅波，$f_s+f_i$")
    assert out is not None
    assert 'role="img"' in out
    assert 'aria-label="图 1-1 调幅波，f_s+f_i"' in out
    labelled = figures.prepare_svg(
        SVG.replace("<svg ", '<svg role="img" aria-label="作者的标签" '), "f0-0-", "别的"
    )
    assert labelled is not None
    assert labelled.count('role="img"') == 1
    assert "作者的标签" in labelled


@pytest.mark.parametrize(
    "payload",
    [
        SVG.replace("</svg>", "<script>alert(1)</script></svg>"),
        SVG.replace("</svg>", '<foreignObject width="1" height="1"/></svg>'),
        SVG.replace('x="5.747439"', 'x="1" onload="alert(1)"'),
        SVG.replace("</svg>", '<image href="https://example.com/x.png"/></svg>'),
        "<!DOCTYPE svg [<!ENTITY x SYSTEM 'file:///etc/passwd'>]>",
        "not an svg at all",
        "<svg><rect width='1'</svg>",
        "",
    ],
)
def test_prepare_svg_refuses_what_must_not_ship(payload: str) -> None:
    assert figures.prepare_svg(payload, "f0-0-", "图") is None


def test_prepare_svg_refuses_an_oversized_drawing() -> None:
    huge = '<svg xmlns="http://www.w3.org/2000/svg">' + "<!--" * 400_000 + "</svg>"
    assert figures.prepare_svg(huge, "f0-0-", "图") is None


def test_natural_key_orders_two_digit_sections() -> None:
    numbers = ["2.10", "2.9", "2.1", "10.1", "9.2"]
    assert sorted(numbers, key=figures.natural_key) == ["2.1", "2.9", "2.10", "9.2", "10.1"]


def _write_book(path: Path, figures_map: dict[str, str], sections: dict[str, list[str]]) -> None:
    """A built book: the FIGURES map plus one <section> per anchor."""
    marks = "".join(
        f'<section class="section" id="{anchor}">'
        + "".join(f'<figure data-fig="{name}"></figure>' for name in names)
        + "</section>"
        for anchor, names in sections.items()
    )
    path.write_text(
        "<html><script>window.FIGURES = "
        + json.dumps(figures_map, ensure_ascii=False)
        + ";</script>"
        + marks
        + "</html>",
        encoding="utf-8",
    )


def test_book_figures_reads_the_map_the_page_was_built_with(tmp_path: Path) -> None:
    book = tmp_path / "book.html"
    _write_book(book, {"a": SVG, "b": SVG}, {"s2-1": ["a", "b"]})
    assert sorted(figures.book_figures(book)) == ["a", "b"]
    assert figures.section_names(book, "s2-1") == ["a", "b"]
    assert figures.section_names(book, "nope") == []
    assert figures.book_figures(tmp_path / "absent.html") == {}


def test_book_figures_cache_notices_a_rebuild(tmp_path: Path) -> None:
    book = tmp_path / "book.html"
    _write_book(book, {"a": SVG}, {})
    assert list(figures.book_figures(book)) == ["a"]
    _write_book(book, {"a": SVG, "b": SVG}, {})
    assert sorted(figures.book_figures(book)) == ["a", "b"]


@pytest.fixture()
def deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "review-home"
    home.mkdir()
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(home))
    book = tmp_path / "books" / "高频_活书.html"
    book.parent.mkdir()
    _write_book(book, {"series_resonance": SVG}, {"s1-1": ["series_resonance"]})
    (home / "books.json").write_text(
        json.dumps({"高频": {"html": str(book)}}, ensure_ascii=False), encoding="utf-8"
    )
    (home / "book-figures.json").write_text(
        json.dumps(
            {
                "version": 1,
                "books": {
                    "高频": {
                        "1.1": {
                            "title": "串联谐振回路",
                            "anchor": "s1-1",
                            "figures": [{"name": "series_resonance", "label": "图 1-1", "caption": "图 1-1 串联回路"}],
                        }
                    },
                    "微机": {
                        "1.1": {
                            "title": "计算机基础",
                            "anchor": "s9-1",
                            "figures": [{"name": "intro", "label": "图 1-1", "caption": "图 1-1 别的书"}],
                        }
                    },
                },
                "chapters": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (home / "book-sections.json").write_text(
        json.dumps(
            {
                "chapters": {
                    "第1章_绪论": {
                        "笔记": [{"no": "1.1", "title": "串联谐振回路", "anchor": "s1-1"}]
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return home


def test_pack_for_title_inlines_the_chapter_drawing(deployment: Path) -> None:
    payload = figures.pack_for_title("高频 · 第1章_绪论 复习包")
    assert payload is not None
    assert payload["book"] == "高频_活书"
    assert payload["missing"] == []
    section = payload["sections"][0]
    assert section["no"] == "1.1"
    assert section["anchor"] == "s1-1"
    assert section["figures"][0]["svg"].startswith("<svg")
    assert 'xlink:href="#f0-0-glyph-0-0"' in section["figures"][0]["svg"]


def test_pack_for_title_matches_the_subject_not_only_the_number(deployment: Path) -> None:
    # Both books print a section 1.1, with different titles and drawings. A
    # chapter mapped in 微机 must not borrow 高频's figure.
    payload = figures.pack_for_title("微机 · 第1章_绪论 复习包")
    assert payload is None, "no plan entry for 微机's chapter, so no claim is safe"


def test_pack_for_title_is_none_without_a_book(deployment: Path, monkeypatch) -> None:
    (deployment / "books.json").write_text("{}", encoding="utf-8")
    assert figures.pack_for_title("高频 · 第1章_绪论 复习包") is None


def test_pack_for_title_is_none_for_an_unplanned_chapter(deployment: Path) -> None:
    # Experiment packs are not plan chapters: no section maps, so nothing is
    # claimed even though the subject's book exists.
    assert figures.pack_for_title("高频 · 实验五_振幅调制与解调实验 复习包") is None


def test_pack_for_title_keeps_undrawable_rows_for_the_tree(deployment: Path) -> None:
    book = Path(json.loads((deployment / "books.json").read_text(encoding="utf-8"))["高频"]["html"])
    _write_book(book, {}, {"s1-1": []})
    payload = figures.pack_for_title("高频 · 第1章_绪论 复习包")
    assert payload is not None
    figure = payload["sections"][0]["figures"][0]
    assert "svg" not in figure
    assert payload["missing"] == ["图 1-1"]
