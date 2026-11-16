#!/usr/bin/env python
"""Install the Review Studio entry into the core personalization MCP catalog.

The user-level MCP store has no plugin hook: its catalog is vendored inside the
core ``deeptutor`` wheel and its entries cannot supply a URL at install time.
This deployment patch adds the smallest missing capability — a ``url``
credential target — and merges the Review Studio entry beside the curated
services. Re-run it after a core upgrade; it is idempotent and keeps backups.

    .venv/Scripts/python.exe src/deeptutor/review/integrations/mcp_catalog/apply.py
    .venv/Scripts/python.exe .../apply.py --check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

ENTRY_FILE = Path(__file__).with_name("deeptutor-review.json")
ENTRY_ID = "deeptutor-review"

TARGET_LINE = 'CredentialTarget = Literal["env", "header", "url_param", "arg"]'
TARGET_LINE_PATCHED = 'CredentialTarget = Literal["env", "header", "url", "url_param", "arg"]'

REMOTE_TARGETS = '_REMOTE_TARGETS = frozenset({"header", "url_param"})'
REMOTE_TARGETS_PATCHED = '_REMOTE_TARGETS = frozenset({"header", "url", "url_param"})'

BRANCH_OLD = '''        elif kind == "arg":
            cfg.args = _with_arg(list(cfg.args), name, injected)
        else:
            cfg.url = _with_url_param(cfg.url, name, injected)'''

BRANCH_NEW = '''        elif kind == "arg":
            cfg.args = _with_arg(list(cfg.args), name, injected)
        elif kind == "url":
            if not injected.lower().startswith("https://"):
                raise ValueError(f"Field {spec.key!r} must be an https:// URL")
            cfg.url = injected
        else:
            cfg.url = _with_url_param(cfg.url, name, injected)'''


def core_paths() -> tuple[Path, Path]:
    import deeptutor

    core = Path(deeptutor.__file__).resolve().parent
    return (
        core / "services/mcp/catalog/models.py",
        core / "services/mcp/catalog/vendor/curated.json",
    )


def _backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    shutil.copy2(path, backup)
    return backup


def patch_models(path: Path, *, write: bool) -> list[str]:
    text = path.read_text(encoding="utf-8")
    original = text
    changes: list[str] = []
    if TARGET_LINE_PATCHED not in text:
        if TARGET_LINE not in text:
            raise SystemExit(f"cannot find CredentialTarget line in {path}")
        text = text.replace(TARGET_LINE, TARGET_LINE_PATCHED, 1)
        changes.append("CredentialTarget += url")
    if REMOTE_TARGETS_PATCHED not in text:
        if REMOTE_TARGETS not in text:
            raise SystemExit(f"cannot find _REMOTE_TARGETS line in {path}")
        text = text.replace(REMOTE_TARGETS, REMOTE_TARGETS_PATCHED, 1)
        changes.append("_REMOTE_TARGETS += url")
    if 'elif kind == "url":' not in text:
        if BRANCH_OLD not in text:
            raise SystemExit(f"cannot find build_server_config branch in {path}")
        text = text.replace(BRANCH_OLD, BRANCH_NEW, 1)
        changes.append("build_server_config += url target")
    if text != original and write:
        backup = _backup(path)
        path.write_text(text, encoding="utf-8")
        changes.append(f"backup {backup.name}")
    return changes


def merge_catalog(path: Path, entry: dict, *, write: bool) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise SystemExit(f"unexpected catalog shape in {path}")
    existing = next((row for row in rows if row.get("id") == ENTRY_ID), None)
    if existing == entry:
        return []
    rows = [row for row in rows if row.get("id") != ENTRY_ID]
    rows.append(entry)
    if isinstance(data, dict):
        data["entries"] = rows
    else:
        data = rows
    if write:
        backup = _backup(path)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return [f"merged {ENTRY_ID}", f"backup {backup.name}"]
    return [f"would merge {ENTRY_ID}"]


def validate() -> None:
    code = (
        "from deeptutor.services.mcp.catalog import build_server_config, get_entry\n"
        f"entry = get_entry({ENTRY_ID!r})\n"
        "assert entry is not None, 'catalog entry missing'\n"
        "built = build_server_config(entry, {'url': 'https://review.example.com/mcp', 'token': 't'})\n"
        "assert built.config.url == 'https://review.example.com/mcp', built.config.url\n"
        "assert built.config.headers['Authorization'].startswith('${secret:'), built.config.headers\n"
        "assert built.secret_values['token'] == 'Bearer t', built.secret_values\n"
        "print('catalog patch OK')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without writing")
    parser.add_argument("--dry-run", action="store_true", help="show changes without writing")
    args = parser.parse_args()
    write = not (args.check or args.dry_run)

    models_path, catalog_path = core_paths()
    entry = json.loads(ENTRY_FILE.read_text(encoding="utf-8"))
    print("models :", models_path)
    print("catalog:", catalog_path)

    changes = patch_models(models_path, write=write)
    changes += merge_catalog(catalog_path, entry, write=write)
    for change in changes:
        print(" -", change)
    if not changes:
        print(" - already applied")
    if args.check or args.dry_run:
        if args.check:
            validate()
        return 0
    validate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
