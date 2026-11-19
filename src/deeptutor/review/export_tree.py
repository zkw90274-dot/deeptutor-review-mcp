"""Export every review pack into one subject-classified folder tree.

Layout, rewritten on every run::

    <out>/index.html
    <out>/<subject>/<chapter>.html

The subject and chapter come from the pack title
('"<subject> · <chapter> 复习包"'), so the folders are derived from the
learner's own course names and no list of subjects is hard-coded. The tree is
rebuilt from the live store each time: renamed chapters leave no stale file,
retired packs disappear, and there are never dated snapshot folders. Packs that
fail the quality gates are skipped and reported instead of written.

This module is the implementation behind the MCP tool
'review_export_tree' and the repository script
'scripts/export-review-packs.py'; both pass in their own destination and
subject filter.
"""

from __future__ import annotations

import html
import hashlib
import inspect
import os
import re
import stat
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from deeptutor.review import materials
from deeptutor.review.service import QualityGateFailed

#: Separator between subject and chapter in a pack title.
SUBJECT_SEPARATOR = " · "

#: Trailing noun every generated pack title carries.
PACK_SUFFIX = " 复习包"

#: Folder used when the caller does not name one, relative to the review home.
DEFAULT_FOLDER = "exports"

#: Folder/file name fallback when a title is empty or only unsafe characters.
UNNAMED = "未命名"

_UNSAFE_NAME = re.compile(r'[\\/:*?"<>|]+')

#: Split a chapter name into digit and non-digit runs for natural ordering.
_NATURAL_TOKEN = re.compile(r"(\d+)")
_EXPORT_MARK = b"\n<!-- deeptutor-review-export-v1:"
_EXPORT_END = b" -->\n"


class ExportConflict(ValueError):
    pass


def _is_link(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def _is_managed(path: Path) -> bool:
    if _is_link(path) or not path.is_file():
        return False
    content, marker, digest = path.read_bytes().rpartition(_EXPORT_MARK)
    return bool(marker) and digest == hashlib.sha256(content).hexdigest().encode() + _EXPORT_END


def _check_target(out: Path, path: Path) -> None:
    folder = path.parent
    if (
        _is_link(folder)
        or (folder.exists() and not folder.is_dir())
        or _is_link(path)
        or (path.exists() and not _is_managed(path))
    ):
        raise ExportConflict(f"export destination already contains {path.relative_to(out).as_posix()}")


def _write_page(path: Path, data: bytes) -> None:
    marked = data + _EXPORT_MARK + hashlib.sha256(data).hexdigest().encode() + _EXPORT_END
    if not path.exists():
        with path.open("xb") as handle:
            handle.write(marked)
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        try:
            handle.write(marked)
            handle.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def _natural_key(value: str) -> list[tuple[int, object]]:
    """Sort ``第1章`` before ``第2章`` before ``第10章``.

    A plain string sort puts ``第10章`` between ``第1章`` and ``第2章``. Each
    token is tagged so a digit run and a text run never compare directly
    (``TypeError: '<' not supported between instances of 'int' and 'str'``).
    """
    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in _NATURAL_TOKEN.split(value)
    ]


def split_pack_title(title: str) -> tuple[str, str]:
    """Return (subject, chapter) for a pack title.

    A title without the separator is treated as a chapter on its own, and the
    subject falls back to UNNAMED so the tree still has one level.
    """
    raw = (title or "").strip()
    subject, sep, rest = raw.partition(SUBJECT_SEPARATOR)
    if not sep:
        subject, rest = "", raw
    chapter = rest.strip()
    if chapter.endswith(PACK_SUFFIX):
        chapter = chapter[: -len(PACK_SUFFIX)].strip()
    return subject.strip() or UNNAMED, chapter or UNNAMED


def safe_component(text: str) -> str:
    """A single path component safe on Windows, macOS and POSIX."""
    name = _UNSAFE_NAME.sub("-", (text or "").strip())
    name = name.strip(" .")
    return name or UNNAMED


def _megabytes(size: int) -> str:
    return f"{size / 1048576:.2f} MB"


def _gate_reason(exc: QualityGateFailed) -> str:
    report = getattr(exc, "report", None)
    failed = [
        str(getattr(gate, "gate", "") or getattr(gate, "name", "") or "gate")
        for gate in getattr(report, "gates", ())
        if not getattr(gate, "passed", True)
    ]
    if failed:
        return "quality gates: " + ", ".join(sorted(set(failed)))
    return "quality gates did not pass"


def _pack_book_figures(
    subject: str,
    chapter: str,
    destination: Path,
    index: dict[str, Any],
    books: dict[str, Any],
    sections: dict[str, Any],
) -> dict[str, Any] | None:
    """This chapter's diagrams with a relative link back into its book.

    The drawings themselves are prepared by ``figures.pack_figures``; the tree
    only adds what a multi-file export has and a single page does not: the
    book sits beside the pack folders, so a relative path lets an undrawable
    figure still point at the section that shows it. A pack whose chapter is
    not in the plan (experiments) or whose book is not on this machine gets
    None, and the page renders without the block.
    """
    from deeptutor.review import figures as figure_source

    payload = figure_source.pack_figures(
        subject, chapter, index=index, books=books, sections=sections
    )
    if payload is None:
        return None
    resolved = materials.resolve_book(str((books.get(subject) or "")))
    if resolved is None:
        return None
    payload["href"] = _book_href(resolved, destination)
    return payload


def _book_href(book: Path, destination: Path) -> str:
    """Relative path from a pack page to its book.

    A book and an export folder can sit on different Windows drives, where no
    relative path exists; fall back to a file URI so the link still opens
    instead of aborting the whole tree.
    """
    try:
        return os.path.relpath(book, destination.parent).replace(os.sep, "/")
    except ValueError:
        return book.as_uri()


def _accepts_book_figures(service: Any) -> bool:
    """Whether this service's export() takes the optional figure index.

    A third-party or test double may implement the older two-argument export;
    the block is additive, so degrade to the old call instead of failing the
    whole tree.
    """
    try:
        parameters = inspect.signature(service.export).parameters
    except (TypeError, ValueError):
        return False
    return "book_figures" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _group(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for entry in entries:
        group = groups.setdefault(
            entry["subject"],
            {"name": entry["subject"], "folder": entry["folder"], "packs": 0, "questions": 0, "bytes": 0},
        )
        group["packs"] += 1
        group["questions"] += entry["items"]
        group["bytes"] += entry["bytes"]
    return list(groups.values())


def build_index(out_dir: Path, entries: Sequence[dict[str, Any]], *, now: datetime | None = None) -> Path:
    """Write index.html linking every exported pack, grouped by subject.

    ``now`` is opt-in: without it the page carries no export time, so two runs
    over unchanged packs are byte-identical. Callers that want a visible
    timestamp pass it explicitly.
    """
    path = out_dir / "index.html"
    _check_target(out_dir, path)
    _write_page(path, _index_content(entries, now=now))
    return path


def _index_content(entries: Sequence[dict[str, Any]], *, now: datetime | None) -> bytes:
    stamp = now.strftime("%Y-%m-%d %H:%M") if now is not None else None
    total_items = sum(e["items"] for e in entries)
    total_bytes = sum(e["bytes"] for e in entries)
    groups = _group(entries)

    sections: list[str] = []
    for group in groups:
        rows = sorted(
            (e for e in entries if e["subject"] == group["name"]),
            key=lambda e: _natural_key(e["chapter"]),
        )
        body = "\n".join(
            f'          <tr><td><a href="{group["folder"]}/{html.escape(row["file"])}">'
            f'{html.escape(row["chapter"])}</a></td>'
            f'<td class="num">{row["items"]}</td><td class="num">{_megabytes(row["bytes"])}</td></tr>'
            for row in rows
        )
        sections.append(
            f"""      <section class="subject">
        <h2>{html.escape(group["name"])}<span>{group["packs"]} 包 · {group["questions"]} 题</span></h2>
        <table>
          <thead><tr><th>章节</th><th class="num">题目</th><th class="num">体积</th></tr></thead>
          <tbody>
{body}
          </tbody>
        </table>
      </section>"""
        )

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>复习包 · 全部分类索引</title>
<style>
:root{{color-scheme:light;--bg:#eceef1;--card:#fbfbfc;--ink:#1f2430;--muted:#5a6472;--line:#e4e7ec;--brand:#0f766e;--radius:22px}}
@media (prefers-color-scheme:dark){{
  :root{{color-scheme:dark;--bg:#0b0d0f;--card:#16191c;--ink:#f2f4f7;--muted:#9aa3af;--line:#262a30;--brand:#5ec8bd}}
}}
*{{box-sizing:border-box}}
body{{margin:0;padding:36px 20px 64px;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC","Microsoft YaHei",sans-serif}}
main{{max-width:960px;margin:0 auto}}
h1{{font-size:24px;margin:0 0 6px;letter-spacing:-.02em}}
p.meta{{color:var(--muted);margin:0 0 24px;font-size:13px}}
section.subject{{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
padding:18px 20px 6px;margin:0 0 18px;box-shadow:0 1px 2px rgba(16,24,40,.04)}}
section.subject h2{{display:flex;align-items:baseline;gap:10px;font-size:16px;margin:0 0 12px}}
section.subject h2 span{{font-size:12px;font-weight:500;color:var(--muted)}}
table{{width:100%;border-collapse:collapse}}
th,td{{padding:9px 6px;border-bottom:1px solid var(--line);text-align:left}}
th{{font-size:12px;color:var(--muted);font-weight:600}}
tr:last-child td{{border-bottom:0}}
a{{color:var(--brand);text-decoration:none}}
a:hover{{text-decoration:underline}}
td a{{display:inline-block;padding:10px 0}}
.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
footer{{color:var(--muted);font-size:12px;margin-top:8px}}
@media print{{
  :root{{color-scheme:light;--bg:#fff;--card:#fff;--ink:#000;--muted:#333;--line:#bbb;--brand:#000}}
  body{{background:#fff;padding:0}}section.subject{{box-shadow:none}}a{{color:inherit}}
}}
</style></head>
<body><main>
<h1>复习包 · 全部分类索引</h1>
<p class="meta">{len(entries)} 个包 · {total_items} 道题 · 合计 {_megabytes(total_bytes)}{f" · 导出于 {stamp}" if stamp else ""}</p>
{chr(10).join(sections)}
<footer>按科目分目录存放；每次导出整树重写，不再生成按日期的快照目录。</footer>
</main></body></html>
"""
    return page.encode("utf-8")


def export_tree(
    service: Any,
    out_dir: str | Path,
    *,
    subjects: Sequence[str] | None = None,
    index: bool = True,
    dry_run: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Write every pack under out_dir, one folder per subject.

    service is a deeptutor.review.service.ReviewService; it is a parameter
    rather than resolved here so the MCP tool shares its store resolution and
    tests can pass a fake. subjects filters by the title prefix; omitting it
    exports every pack. Files written by an earlier run in the folders touched
    this time are removed, so a renamed chapter leaves no stale page behind.
    """
    out = Path(out_dir).expanduser().resolve()
    allow = {str(name).strip() for name in (subjects or []) if str(name).strip()}

    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    warnings: list[str] = []
    pages: dict[Path, bytes] = {}
    retained: set[str] = set()
    names: set[str] = set()
    selected_folders: dict[str, str] = {}
    takes_figures = _accepts_book_figures(service)
    figure_index = materials.load_book_figures() if takes_figures else {}
    book_files = materials.load_books() if takes_figures else {}
    section_plan = materials.load_book_sections() if takes_figures else {}

    for meta in service.store.list_specs():
        spec = service.store.get_spec(meta["spec_id"])
        subject, chapter = split_pack_title(getattr(spec, "title", ""))
        if allow and subject not in allow:
            continue
        folder = safe_component(subject)
        filename = safe_component(chapter) + ".html"
        destination = out / folder / filename
        relative = destination.relative_to(out).as_posix()
        previous_subject = selected_folders.setdefault(folder.casefold(), subject)
        if previous_subject != subject or relative.casefold() in names:
            raise ExportConflict(f"more than one pack maps to {relative}")
        names.add(relative.casefold())
        try:
            figures = (
                _pack_book_figures(
                    subject, chapter, destination, figure_index, book_files, section_plan
                )
                if takes_figures
                else None
            )
            page = service.export(spec.id, book_figures=figures) if takes_figures else service.export(spec.id)
        except QualityGateFailed as exc:
            retained.add(relative.casefold())
            skipped.append({
                "title": (getattr(spec, "title", "") or spec.id).strip(),
                "reason": _gate_reason(exc),
            })
            continue
        data = page.encode("utf-8")
        pages[destination] = data
        retained.add(relative.casefold())
        entries.append({
            "subject": subject,
            "folder": folder,
            "chapter": chapter,
            "file": filename,
            "items": len(getattr(spec, "items", ()) or ()),
            "bytes": len(data),
        })

    if allow:
        matched = {entry["subject"] for entry in entries}
        for name in sorted(allow - matched):
            warnings.append(f"unknown subject: {name} (no pack title starts with it)")

    index_path: Path | None = None
    wanted_index = out / "index.html"
    if index and (entries or (not allow and _is_managed(wanted_index))):
        pages[wanted_index] = _index_content(entries, now=now)

    for path in pages:
        _check_target(out, path)

    stale: list[str] = []
    if not dry_run:
        for path, data in pages.items():
            _check_target(out, path)
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_page(path, data)
        if wanted_index in pages:
            index_path = wanted_index
        for folder in sorted(out.iterdir()) if out.is_dir() else ():
            if _is_link(folder) or not folder.is_dir():
                continue
            if allow and folder.name.casefold() not in selected_folders:
                continue
            for path in sorted(folder.glob("*.html")):
                relative = path.relative_to(out).as_posix()
                if relative.casefold() not in retained and _is_managed(path):
                    path.unlink()
                    stale.append(relative)

    return {
        "out_dir": str(out),
        "index_path": str(index_path) if index_path else None,
        "packs": len(entries),
        "questions": sum(e["items"] for e in entries),
        "bytes": sum(e["bytes"] for e in entries),
        "subjects": _group(entries),
        "skipped": skipped,
        "warnings": warnings,
        "stale_removed": stale,
        "dry_run": dry_run,
        "files": [f'{e["folder"]}/{e["file"]}' for e in entries],
    }
