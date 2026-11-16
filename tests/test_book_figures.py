"""The diagram index: which living-book section owns which figure.

A book draws its figures as inline SVG, so a plan, a card and an MCP client all
see text only. scripts/build-book-figures.py reads the built books once and
writes <review home>/book-figures.json; deeptutor.review.materials loads it and
review_book_figures reports it. These tests pin the parse, the lookup and the
tool contract without depending on a deployment's real books.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from deeptutor.review import materials
from deeptutor.review.agent_tools import ReviewBookFiguresTool

from _layout import repo_root

SCRIPT = repo_root() / "scripts" / "build-book-figures.py"

BOOK_HTML = """<html><body>
<section class="section" id="s2-1" data-ch="2" data-no="2.1" data-title="系统总线" data-tags="x">
  <h2>系统总线</h2>
  <figure data-fig="fig_pinout"><figcaption>图 2-1　8088 的 40 个引脚</figcaption></figure>
  <figure data-fig="fig_minmode"><figcaption>图 2-2　最小模式</figcaption></figure>
</section>
<section class="section" id="s2-2" data-ch="2" data-no="2.2" data-title="主机板结构" data-tags="x">
  <p>no figure here</p>
</section>
<section class="section" id="kj-1" data-ch="kj" data-appendix="1" data-no="课件 01" data-title="课件">
  <figure data-fig="x"><figcaption>图 9-9　附录里的图</figcaption></figure>
</section>
</body></html>
"""


def _script_module():
    spec = importlib.util.spec_from_file_location("build_book_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_read_book_keeps_numbered_figures_and_skips_appendix(tmp_path: Path) -> None:
    book = tmp_path / "book.html"
    book.write_text(BOOK_HTML, encoding="utf-8")
    module = _script_module()
    found = module.read_book(book)
    assert set(found) == {"2.1"}
    assert found["2.1"]["anchor"] == "s2-1"
    assert found["2.1"]["title"] == "系统总线"
    assert [figure["label"] for figure in found["2.1"]["figures"]] == ["图 2-1", "图 2-2"]
    assert found["2.1"]["figures"][0]["caption"] == "图 2-1 8088 的 40 个引脚"


def test_build_joins_books_onto_planned_chapters(tmp_path: Path) -> None:
    book = tmp_path / "book.html"
    book.write_text(BOOK_HTML, encoding="utf-8")
    (tmp_path / "books.json").write_text(
        json.dumps({"微机": {"html": str(book)}}), encoding="utf-8"
    )
    (tmp_path / "book-sections.json").write_text(
        json.dumps(
            {
                "chapters": {
                    "第2讲": {
                        "第2讲_课程重点": [
                            {"no": "2.1", "title": "系统总线", "anchor": "s2-1"},
                            {"no": "2.2", "title": "主机板结构", "anchor": "s2-2"},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    payload = _script_module().build(tmp_path)
    assert set(payload["books"]) == {"微机"}
    assert payload["chapters"] == {"第2讲": {"2.1": [{"label": "图 2-1", "caption": "图 2-1 8088 的 40 个引脚"}, {"label": "图 2-2", "caption": "图 2-2 最小模式"}]}}


INDEX = {
    "version": 1,
    "generated_at": "2026-11-16T05:00:00+00:00",
    "books": {
        "微机": {
            "2.1": {"title": "系统总线", "anchor": "s2-1", "figures": [{"label": "图 2-1", "caption": "图 2-1 引脚"}]},
            "2.10": {"title": "本讲自查清单", "anchor": "s2-10", "figures": [{"label": "", "caption": "自查图"}]},
        },
        "数字信号处理": {
            "1.4": {"title": "序列的基本运算", "anchor": "s01-4", "figures": [{"label": "", "caption": "序列运算"}]},
        },
    },
    "chapters": {"第2讲": {"2.1": [{"label": "图 2-1", "caption": "图 2-1 引脚"}]}},
}


@pytest.fixture()
def review_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(tmp_path))
    (tmp_path / "book-figures.json").write_text(
        json.dumps(INDEX, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


def _run(**kwargs):
    return asyncio.run(ReviewBookFiguresTool().execute(**kwargs))


def _payload(result) -> dict:
    assert result.success is True, result.content
    return json.loads(result.content)


def test_book_figures_reads_one_subject(review_home: Path) -> None:
    found = materials.book_figures("微机")
    assert sorted(found) == ["2.1", "2.10"]
    assert materials.book_figures("nope") == {}


def test_chapter_figures_uses_the_planned_chapter(review_home: Path) -> None:
    assert list(materials.chapter_figures("第2讲")) == ["2.1"]
    assert materials.chapter_figures("第9讲") == {}


def test_tool_reports_every_book_sorted_by_section(review_home: Path) -> None:
    payload = _payload(_run())
    assert payload["total_figures"] == 3
    assert sorted(payload["books"]) == sorted(["微机", "数字信号处理"])
    assert [row["no"] for row in payload["books"]["微机"]["sections"]] == ["2.1", "2.10"]
    assert payload["instruction"]


def test_tool_filters_one_subject(review_home: Path) -> None:
    payload = _payload(_run(subject="数字信号处理"))
    assert list(payload["books"]) == ["数字信号处理"]


def test_tool_reports_a_chapter_without_a_map(review_home: Path) -> None:
    payload = _payload(_run(chapter="第9讲"))
    assert payload["error"]
    assert payload["instruction"]


def test_tool_without_an_index_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(tmp_path))
    payload = _payload(_run())
    assert payload["error"] == "no figure index"
    assert "scripts/build-book-figures.py" in payload["instruction"]
