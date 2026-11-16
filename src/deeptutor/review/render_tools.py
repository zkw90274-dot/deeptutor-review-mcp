"""Deterministic external toolchain probing for the figure providers (RS-801).

The TikZ provider needs ``xelatex`` and ``pdftocairo``; the layout gate needs
``pdftotext`` and the font-size gate needs poppler's ``pdftohtml``. A missing or
broken binary must degrade the pipeline
(``tikz -> plot -> visible placeholder``) instead of raising at request time,
so the toolchain is probed once, cached as an immutable object, and the reason
for every unavailable tool stays visible to the caller.

Only ``PATH`` lookups and explicit ``DEEPTUTOR_*`` overrides are considered.
No shell profile is read and no command is executed through a shell.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

XELATEX = "xelatex"
PDFTOCAIRO = "pdftocairo"
PDFTOTEXT = "pdftotext"
PDFTOHTML = "pdftohtml"

#: Upper bound for a ``--version`` style call; a hung binary must not block a
#: build request.
VERSION_TIMEOUT_SECONDS = 5.0

_VERSION_TEXT_LIMIT = 200

#: Xpdf's command-line tools (for example the ``pdftotext`` shipped with Git
#: for Windows, version 4.00) print their version banner and then exit with
#: 99 instead of 0. Only that tool gets the exception; accepting 99 globally
#: would hide a genuinely broken xelatex/pdftocairo.
_XPDF_VERSION_EXIT_CODES = frozenset({0, 99})
_DEFAULT_VERSION_EXIT_CODES = frozenset({0})


@dataclass(frozen=True)
class _ToolSpec:
    """Static description of one probed binary."""

    name: str
    env_var: str
    version_args: tuple[str, ...]
    accepted_exit_codes: frozenset[int] = _DEFAULT_VERSION_EXIT_CODES
    #: Substring the version output must contain for the build to use the
    #: tool. ``pdftotext -bbox`` exists only in poppler; the Xpdf build found
    #: on many Windows PATHs reports its version fine and then rejects -bbox.
    #: ``pdftohtml -xml`` is only shipped by poppler at all.
    requires_in_usage: str = ""


_TOOL_SPECS: tuple[_ToolSpec, ...] = (
    _ToolSpec(XELATEX, "DEEPTUTOR_XELATEX", ("--version",)),
    _ToolSpec(PDFTOCAIRO, "DEEPTUTOR_PDFTOCAIRO", ("-v",)),
    _ToolSpec(
        PDFTOTEXT,
        "DEEPTUTOR_PDFTOTEXT",
        ("-v",),
        _XPDF_VERSION_EXIT_CODES,
        requires_in_usage="-bbox",
    ),
    # poppler's ``pdftohtml -v`` exits 0, so the Xpdf 99 exception must not be
    # extended to it: a 99 here means the binary is broken, not Xpdf.
    _ToolSpec(PDFTOHTML, "DEEPTUTOR_PDFTOHTML", ("-v",), requires_in_usage="-xml"),
)


@dataclass(frozen=True)
class ToolCapability:
    """Probe result for a single binary.

    ``available`` is ``False`` both when the binary is missing and when it
    exists but cannot report a version; callers should treat a broken binary
    as unavailable rather than risk a half-working render.
    """

    name: str
    executable: str | None = None
    available: bool = False
    version: str = ""
    reason: str = ""
    #: Flags this build actually supports (e.g. ``-bbox``). Empty for tools
    #: whose capability is implied by their version.
    features: tuple[str, ...] = ()

    def supports(self, feature: str) -> bool:
        return feature in self.features

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "executable": self.executable,
            "available": self.available,
            "version": self.version,
            "reason": self.reason,
            "features": list(self.features),
        }


@dataclass(frozen=True)
class Toolchain:
    """Immutable snapshot of every probed binary."""

    capabilities: tuple[ToolCapability, ...]

    def capability(self, name: str) -> ToolCapability | None:
        for capability in self.capabilities:
            if capability.name == name:
                return capability
        return None

    def available(self, name: str) -> bool:
        capability = self.capability(name)
        return capability is not None and capability.available

    def missing(self) -> tuple[str, ...]:
        return tuple(
            capability.name
            for capability in self.capabilities
            if not capability.available
        )

    def warnings(self) -> tuple[str, ...]:
        """Degradation messages that must stay visible in the exported page."""
        return tuple(
            f"{capability.name} unavailable: {capability.reason}"
            for capability in self.capabilities
            if not capability.available
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tools": {
                capability.name: capability.as_dict()
                for capability in self.capabilities
            },
            "missing": list(self.missing()),
            "warnings": list(self.warnings()),
            "tikz_available": self.available(XELATEX)
            and self.available(PDFTOCAIRO),
            "layout_check_available": self.available(PDFTOTEXT),
            "font_check_available": self.available(PDFTOHTML),
        }


def _valid_override(raw: str) -> str | None:
    """Return the override only when it is an existing absolute executable."""
    candidate = Path(raw)
    if not candidate.is_absolute() or not candidate.is_file():
        return None
    if not os.access(candidate, os.X_OK):
        return None
    return str(candidate)


def _first_line(raw: str) -> str:
    for line in raw.splitlines():
        text = " ".join(line.split())
        if text:
            return text[:_VERSION_TEXT_LIMIT]
    return ""


def _decode(raw: bytes | str) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw


def _probe_version(spec: _ToolSpec, executable: str) -> ToolCapability:
    try:
        completed = subprocess.run(
            [executable, *spec.version_args],
            capture_output=True,
            timeout=VERSION_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ToolCapability(
            name=spec.name,
            executable=executable,
            reason=f"version command timed out after {VERSION_TIMEOUT_SECONDS:g}s",
        )
    except OSError as exc:
        return ToolCapability(
            name=spec.name,
            executable=executable,
            reason=f"version command could not run: {exc}",
        )

    if completed.returncode not in spec.accepted_exit_codes:
        return ToolCapability(
            name=spec.name,
            executable=executable,
            reason=f"version command exited with status {completed.returncode}",
        )

    # ``pdftotext -v`` writes to stderr while ``xelatex --version`` uses stdout.
    version = _first_line(_decode(completed.stdout)) or _first_line(
        _decode(completed.stderr)
    )
    if not version:
        return ToolCapability(
            name=spec.name,
            executable=executable,
            reason="version command produced no output",
        )

    if spec.requires_in_usage and not _supports_flag(
        spec, executable, spec.requires_in_usage
    ):
        return ToolCapability(
            name=spec.name,
            executable=executable,
            version=version,
            reason=(
                f"this build does not support {spec.requires_in_usage}; "
                "install poppler's pdftotext or set the DEEPTUTOR_* override"
            ),
        )

    return ToolCapability(
        name=spec.name,
        executable=executable,
        available=True,
        version=version,
        features=(
            (spec.requires_in_usage,) if spec.requires_in_usage else ()
        ),
    )


def _supports_flag(spec: _ToolSpec, executable: str, flag: str) -> bool:
    """Ask the binary whether ``flag`` exists by reading its usage text.

    Running the tool with only the flag makes both poppler and Xpdf print
    their option list and exit non-zero; the option string itself is the
    evidence. A binary whose usage omits the flag would fail every call, so
    treating it as unavailable is more honest than a later "pdftotext failed".
    """
    try:
        completed = subprocess.run(
            [executable, flag],
            capture_output=True,
            timeout=VERSION_TIMEOUT_SECONDS,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    usage = _decode(completed.stdout) + "\n" + _decode(completed.stderr)
    return flag in usage


def _probe(spec: _ToolSpec) -> ToolCapability:
    override = os.environ.get(spec.env_var, "").strip()
    if override:
        executable = _valid_override(override)
        if executable is None:
            # Do not silently fall back to PATH: a bad override is a
            # deployment error and must be reported as such.
            return ToolCapability(
                name=spec.name,
                reason=(
                    f"{spec.env_var} must be an existing absolute executable path"
                ),
            )
    else:
        executable = shutil.which(spec.name)
        if executable is None:
            return ToolCapability(name=spec.name, reason="not found on PATH")

    return _probe_version(spec, executable)


@lru_cache(maxsize=1)
def _cached_toolchain() -> Toolchain:
    return Toolchain(tuple(_probe(spec) for spec in _TOOL_SPECS))


def probe_toolchain(*, refresh: bool = False) -> Toolchain:
    """Probe the external binaries once and reuse the immutable result.

    Args:
        refresh: Drop the process cache first; use after installing a missing
            binary instead of restarting the process.
    """
    if refresh:
        _cached_toolchain.cache_clear()
    return _cached_toolchain()


def clear_toolchain_cache() -> None:
    """Drop the cached probe result (tests or explicit reconfiguration)."""
    _cached_toolchain.cache_clear()


def capability_summary(toolchain: Toolchain | None = None) -> dict[str, Any]:
    """Return a JSON-serializable summary suitable for API responses.

    Args:
        toolchain: Probe to summarize; defaults to the cached probe.
    """
    probe = probe_toolchain() if toolchain is None else toolchain
    return probe.as_dict()


__all__ = (
    "PDFTOCAIRO",
    "PDFTOHTML",
    "PDFTOTEXT",
    "XELATEX",
    "VERSION_TIMEOUT_SECONDS",
    "ToolCapability",
    "Toolchain",
    "capability_summary",
    "clear_toolchain_cache",
    "probe_toolchain",
)
