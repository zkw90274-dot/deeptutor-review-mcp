"""Process-level capability switches shared by the MCP server and agent tools.

The MCP server and the in-app agent use the same tool classes. A deployment may
want to disable the tools that write or spend money without forking those
classes. The switches live in environment variables and are read at call time,
so stdio, Streamable HTTP and an in-process agent all resolve them the same way;
the MCP CLI translates its flags into these variables before building the
server.

Defaults are permissive to preserve the historical in-app behavior. The MCP
HTTP transport additionally turns ``generate`` off unless explicitly allowed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

READ_ONLY = "DEEPTUTOR_REVIEW_READ_ONLY"
NO_ATTEMPTS = "DEEPTUTOR_REVIEW_NO_ATTEMPTS"
NO_PLAN = "DEEPTUTOR_REVIEW_NO_PLAN"
NO_GENERATE = "DEEPTUTOR_REVIEW_NO_GENERATE"
NO_EXPORT_WRITE = "DEEPTUTOR_REVIEW_NO_EXPORT_WRITE"
ALLOW_OUT_DIRS = "DEEPTUTOR_REVIEW_ALLOW_OUT_DIRS"

_TRUTHY = {"1", "true", "yes", "on"}


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


@dataclass(frozen=True)
class Capabilities:
    read_only: bool
    attempts: bool
    plan: bool
    generate: bool
    export_write: bool
    allowed_out_dirs: tuple[str, ...]


def current() -> Capabilities:
    read_only = _flag(READ_ONLY)
    return Capabilities(
        read_only=read_only,
        attempts=not read_only and not _flag(NO_ATTEMPTS),
        plan=not read_only and not _flag(NO_PLAN),
        generate=not read_only and not _flag(NO_GENERATE),
        export_write=not read_only and not _flag(NO_EXPORT_WRITE),
        allowed_out_dirs=tuple(
            part.strip()
            for part in os.environ.get(ALLOW_OUT_DIRS, "").split(os.pathsep)
            if part.strip()
        ),
    )


_DISABLED = {
    "review_attempt": ("attempts", "recording attempts"),
    "review_plan": ("plan", "managing study plans"),
    "review_generate": ("generate", "generating new packs"),
    "review_export_tree": ("export_write", "writing export trees"),
}


def disabled(tool: str) -> str | None:
    """The instruction to return when ``tool`` is disabled, or None.

    Read-only tools are not in the mapping, so they stay available even under
    ``--read-only``. ``review_export_link`` is deliberately read-only: it hands
    back a URL and does not write, so disabling export *writes* must not hide it.
    """
    entry = _DISABLED.get(tool)
    if entry is None:
        return None
    field, action = entry
    if getattr(current(), field):
        return None
    return (
        f"{tool} is disabled in this deployment ({action} is off). "
        "Use the read-only tools instead: review_specs, review_today, "
        "review_hint, review_weakness, review_export_link, review_book_figures. "
        "The operator can enable it with the server's launch flags."
    )
