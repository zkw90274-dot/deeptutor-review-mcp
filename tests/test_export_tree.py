"""The subject-classified export tree shared by the MCP tool and the script."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pytest

from deeptutor.review.agent_tools import ReviewExportTreeTool
from deeptutor.review.export_tree import (
    UNNAMED,
    _book_href,
    build_index,
    export_tree,
    safe_component,
    split_pack_title,
)
from deeptutor.review.fixture import build_fixture_spec
from deeptutor.review.models import QualityReport
from deeptutor.review.service import QualityGateFailed


@dataclass
class _Spec:
    id: str
    title: str
    items: list = field(default_factory=list)


class _Store:
    def __init__(self, specs: list[_Spec]) -> None:
        self.specs = {spec.id: spec for spec in specs}

    def list_specs(self) -> list[dict]:
        return [{"spec_id": spec_id} for spec_id in self.specs]

    def get_spec(self, spec_id: str) -> _Spec:
        return self.specs[spec_id]


class _Service:
    def __init__(self, specs: list[_Spec], failing: set[str] | None = None) -> None:
        self.store = _Store(specs)
        self.failing = failing or set()

    def export(self, spec_id: str) -> str:
        if spec_id in self.failing:
            raise QualityGateFailed(QualityReport(passed=False))
        spec = self.store.specs[spec_id]
        return f"<html><body>{spec.title}</body></html>"


def _service() -> _Service:
    return _Service(
        [
            _Spec("kb-1", "高频 · 第1章_绪论 复习包", [object(), object()]),
            _Spec("kb-2", "高频 · 第2章_选频网络 复习包", [object()]),
            _Spec("kb-3", "微机 · 第1讲 复习包", [object(), object(), object()]),
        ]
    )


def test_split_pack_title() -> None:
    assert split_pack_title("高频 · 第1章_绪论 复习包") == ("高频", "第1章_绪论")
    assert split_pack_title("独立包 复习包") == (UNNAMED, "独立包")
    assert split_pack_title("") == (UNNAMED, UNNAMED)
    # The separator is the title's own convention, not a slash in the text.
    assert split_pack_title("电磁场 · 第2章_作业 复习包") == ("电磁场", "第2章_作业")


def test_safe_component_never_escapes_the_folder() -> None:
    assert safe_component("a/b:c*?") == "a-b-c-"
    assert safe_component(" trailing. ") == "trailing"
    assert safe_component("") == UNNAMED


def test_export_tree_classifies_by_subject(tmp_path: Path) -> None:
    out = tmp_path / "packs"
    summary = export_tree(_service(), out)

    assert (out / "高频" / "第1章_绪论.html").is_file()
    assert (out / "高频" / "第2章_选频网络.html").is_file()
    assert (out / "微机" / "第1讲.html").is_file()
    index = (out / "index.html").read_text(encoding="utf-8")
    assert 'href="高频/第1章_绪论.html"' in index
    assert 'href="微机/第1讲.html"' in index
    # One section per subject, in first-seen order.
    assert index.index("高频") < index.index("微机")

    assert summary["packs"] == 3
    assert summary["questions"] == 6
    assert [group["name"] for group in summary["subjects"]] == ["高频", "微机"]
    assert summary["index_path"] == str(out / "index.html")


def test_export_tree_removes_pages_of_renamed_chapters(tmp_path: Path) -> None:
    out = tmp_path / "packs"
    original = _service()
    original.store.specs["kb-1"].title = "高频 · 旧章名 复习包"
    export_tree(original, out)
    stale = out / "高频" / "旧章名.html"
    notes = stale.with_name("notes.html")
    notes.write_text("personal", encoding="utf-8")

    summary = export_tree(_service(), out)

    assert not stale.exists()
    assert notes.read_text(encoding="utf-8") == "personal"
    assert summary["stale_removed"] == ["高频/旧章名.html"]


@pytest.mark.parametrize("relative", ["index.html", "高频/第2章_选频网络.html"])
def test_export_preflight_preserves_unmanaged_collision(tmp_path: Path, relative: str) -> None:
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("personal", encoding="utf-8")
    with pytest.raises(ValueError, match=relative):
        export_tree(_service(), tmp_path)
    assert list(tmp_path.rglob("*.html")) == [target]
    assert target.read_text(encoding="utf-8") == "personal"


def test_export_preserves_edited_pages_and_removes_retired_subjects(tmp_path: Path) -> None:
    old_service = _service()
    export_tree(old_service, tmp_path)
    edited = tmp_path / "高频" / "第1章_绪论.html"
    with edited.open("a", encoding="utf-8") as handle:
        handle.write("personal edit")
    del old_service.store.specs["kb-1"]
    del old_service.store.specs["kb-3"]
    summary = export_tree(old_service, tmp_path)
    assert edited.read_text(encoding="utf-8").endswith("personal edit")
    assert summary["stale_removed"] == ["微机/第1讲.html"]


def test_export_tree_subject_filter_leaves_other_folders_alone(tmp_path: Path) -> None:
    out = tmp_path / "packs"
    other = out / "微机" / "留着手写.html"
    other.parent.mkdir(parents=True)
    other.write_text("<html>keep</html>", encoding="utf-8")

    summary = export_tree(_service(), out, subjects=["高频"])

    assert summary["packs"] == 2
    assert (out / "高频" / "第1章_绪论.html").is_file()
    assert not (out / "微机" / "第1讲.html").exists()
    assert other.is_file()


def test_export_tree_skips_and_reports_quality_failures(tmp_path: Path) -> None:
    out = tmp_path / "packs"
    summary = export_tree(_Service(_service().store.specs.values(), {"kb-2"}), out)

    assert summary["packs"] == 2
    assert summary["skipped"] == [
        {"title": "高频 · 第2章_选频网络 复习包", "reason": "quality gates did not pass"}
    ]
    assert not (out / "高频" / "第2章_选频网络.html").exists()


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "packs"
    summary = export_tree(_service(), out, dry_run=True)

    assert summary["packs"] == 3
    assert summary["index_path"] is None
    assert not out.exists()
    assert summary["warnings"] == []


def test_unknown_subject_is_reported(tmp_path: Path) -> None:
    out = tmp_path / "packs"
    summary = export_tree(_service(), out, subjects=["高频", "不存在的科目"])

    assert summary["packs"] == 2
    assert summary["warnings"] == [
        "unknown subject: 不存在的科目 (no pack title starts with it)"
    ]


def test_tool_writes_the_tree(service, tmp_path: Path) -> None:
    service.put_spec(build_fixture_spec())
    result = asyncio.run(
        ReviewExportTreeTool().execute(
            out_dir=str(service.store.home / "exports" / "tree")
        )
    )
    payload = json.loads(result.content)

    assert result.success is True
    assert payload["packs"] == 1
    assert Path(payload["out_dir"]).is_dir()
    assert Path(payload["index_path"]).is_file()
    assert payload["instruction"]
    assert "index.html" in payload["instruction"]


def test_tool_rejects_an_out_dir_outside_the_review_home(service, tmp_path: Path) -> None:
    service.put_spec(build_fixture_spec())
    outside = tmp_path / "outside"
    result = asyncio.run(ReviewExportTreeTool().execute(out_dir=str(outside)))
    payload = json.loads(result.content)

    assert result.success is True
    assert "outside the allowed roots" in payload["error"]
    assert "--allow-out-dir" in payload["instruction"]
    assert not outside.exists()


def _entry(chapter: str, *, subject: str = "高频", items: int = 1) -> dict:
    return {
        "subject": subject,
        "folder": subject,
        "file": f"{chapter}.html",
        "chapter": chapter,
        "items": items,
        "bytes": 1024,
    }


def test_build_index_is_byte_identical_without_now(tmp_path: Path) -> None:
    entries = [_entry("第1章"), _entry("第2章")]
    build_index(tmp_path, entries)
    first = (tmp_path / "index.html").read_bytes()
    build_index(tmp_path, entries)
    assert (tmp_path / "index.html").read_bytes() == first
    assert "导出于" not in first.decode("utf-8")


def test_build_index_orders_chapters_naturally(tmp_path: Path) -> None:
    build_index(
        tmp_path,
        [_entry("第10章_数字滤波器"), _entry("第2章_采样"), _entry("第1章_绪论")],
    )
    text = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert text.index("第1章_绪论") < text.index("第2章_采样") < text.index("第10章_数字滤波器")


def test_build_index_accepts_an_explicit_timestamp(tmp_path: Path) -> None:
    stamp = datetime(2026, 1, 1, 1, 1)
    build_index(tmp_path, [_entry("第1章")], now=stamp)
    assert "导出于 2026-01-01 01:01" in (tmp_path / "index.html").read_text(
        encoding="utf-8"
    )


class _FigureService(_Service):
    """A service whose export() accepts the optional living-book index."""

    def __init__(self, specs, books=None, chapters=None) -> None:
        super().__init__(specs)
        self.books = books or {}
        self.chapters = chapters or {}
        self.calls: list[tuple[str, object]] = []

    def export(self, spec_id: str, *, book_figures=None) -> str:  # noqa: D401
        self.calls.append((spec_id, book_figures))
        spec = self.store.specs[spec_id]
        block = f"<section class='book-figures'>{book_figures['href']}</section>" if book_figures else ""
        return f"<html><body>{spec.title}{block}</body></html>"


def _write_book_index(home: Path, book: Path, subject: str, chapter: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "book-figures.json").write_text(
        json.dumps(
            {
                "books": {
                    subject: {
                        "2.1": {
                            "title": "系统总线",
                            "anchor": "s2-1",
                            "figures": [{"label": "图 2-1", "caption": "图 2-1 引脚"}],
                        }
                    }
                },
                "chapters": {chapter: {"2.1": [{"label": "图 2-1", "caption": "图 2-1 引脚"}]}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (home / "books.json").write_text(
        json.dumps({subject: {"html": str(book)}}, ensure_ascii=False), encoding="utf-8"
    )


def test_export_tree_links_the_chapter_figures_into_the_book(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "review-home"
    book = tmp_path / "books" / "高频_活书.html"
    book.parent.mkdir(parents=True)
    book.write_text("<html></html>", encoding="utf-8")
    _write_book_index(home, book, "高频", "第1章_绪论")
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(home))

    service = _FigureService([_Spec("kb-1", "高频 · 第1章_绪论 复习包", [object()])])
    export_tree(service, tmp_path / "packs")

    sent = service.calls[0][1]
    assert sent is not None
    assert sent["book"] == "高频_活书"
    # The link is relative to the pack page, not to the export root.
    assert sent["href"] == "../../books/高频_活书.html"
    assert sent["sections"][0]["anchor"] == "s2-1"
    page = (tmp_path / "packs" / "高频" / "第1章_绪论.html").read_text(encoding="utf-8")
    assert "book-figures" in page
    assert "../../books/高频_活书.html" in page


def test_book_href_falls_back_to_a_uri_across_windows_drives(
    tmp_path: Path, monkeypatch
) -> None:
    book = tmp_path / "books" / "高频_活书.html"

    def refuse(*_args, **_kwargs):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr("deeptutor.review.export_tree.os.path.relpath", refuse)
    assert _book_href(book, tmp_path / "packs" / "高频" / "第1章.html") == book.as_uri()


def test_export_tree_omits_the_block_when_the_pack_has_no_book_figures(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "review-home"
    book = tmp_path / "books" / "高频_活书.html"
    book.parent.mkdir(parents=True)
    book.write_text("<html></html>", encoding="utf-8")
    _write_book_index(home, book, "高频", "第1章_绪论")
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(home))

    # 实验课 packs are not plan chapters, so the index must not claim them.
    service = _FigureService([_Spec("kb-9", "高频 · 实验四_振荡器实验 复习包", [object()])])
    export_tree(service, tmp_path / "packs")

    assert service.calls[0][1] is None
    page = (tmp_path / "packs" / "高频" / "实验四_振荡器实验.html").read_text(encoding="utf-8")
    assert "book-figures" not in page


def test_export_tree_omits_the_block_when_the_book_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "review-home"
    _write_book_index(home, tmp_path / "books" / "不在本机.html", "高频", "第1章_绪论")
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(home))

    service = _FigureService([_Spec("kb-1", "高频 · 第1章_绪论 复习包", [object()])])
    export_tree(service, tmp_path / "packs")

    # A link into a file that is not there is worse than no link at all.
    assert service.calls[0][1] is None
