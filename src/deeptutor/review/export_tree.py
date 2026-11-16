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
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

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
    """Write index.html linking every exported pack, grouped by subject."""
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M")
    total_items = sum(e["items"] for e in entries)
    total_bytes = sum(e["bytes"] for e in entries)
    groups = _group(entries)

    sections: list[str] = []
    for group in groups:
        rows = sorted(
            (e for e in entries if e["subject"] == group["name"]),
            key=lambda e: e["chapter"],
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
:root{{--bg:#eceef1;--card:#fbfbfc;--ink:#1f2430;--muted:#667085;--line:#e4e7ec;--brand:#0f766e;--radius:22px}}
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
.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
footer{{color:var(--muted);font-size:12px;margin-top:8px}}
@media print{{body{{background:#fff;padding:0}}section.subject{{box-shadow:none}}a{{color:inherit}}}}
</style></head>
<body><main>
<h1>复习包 · 全部分类索引</h1>
<p class="meta">{len(entries)} 个包 · {total_items} 道题 · 合计 {_megabytes(total_bytes)} · 导出于 {stamp}</p>
{chr(10).join(sections)}
<footer>按科目分目录存放；每次导出整树重写，不再生成按日期的快照目录。</footer>
</main></body></html>
"""
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


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
    written: dict[str, set[str]] = {}

    for meta in service.store.list_specs():
        spec = service.store.get_spec(meta["spec_id"])
        subject, chapter = split_pack_title(getattr(spec, "title", ""))
        if allow and subject not in allow:
            continue
        try:
            page = service.export(spec.id)
        except QualityGateFailed as exc:
            skipped.append({
                "title": (getattr(spec, "title", "") or spec.id).strip(),
                "reason": _gate_reason(exc),
            })
            continue
        folder = safe_component(subject)
        filename = safe_component(chapter) + ".html"
        data = page.encode("utf-8")
        if not dry_run:
            (out / folder).mkdir(parents=True, exist_ok=True)
            (out / folder / filename).write_bytes(data)
            written.setdefault(folder, set()).add(filename)
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
    stale: list[str] = []
    if not dry_run:
        for folder, names in written.items():
            for path in sorted((out / folder).glob("*.html")):
                if path.name not in names:
                    path.unlink()
                    stale.append(path.relative_to(out).as_posix())
            if not any((out / folder).iterdir()):
                (out / folder).rmdir()
        if index and entries:
            out.mkdir(parents=True, exist_ok=True)
            index_path = build_index(out, entries, now=now)

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
