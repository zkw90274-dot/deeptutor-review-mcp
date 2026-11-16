"""TikZ -> SVG provider (RS-803).

A TikZ body is compiled by ``xelatex`` (no shell escape) into a single-page
PDF, converted by ``pdftocairo -svg``, then sanitized and normalized before it
may reach the exporter. Three properties are non-negotiable:

* **No silent disappearance.** Every failure raises :class:`TikzRenderError`
  with a ``reason``; the caller degrades to the plot provider or a visible
  placeholder instead of dropping the figure.
* **Trusted tool, untrusted LaTeX.** The source is scanned with a command
  allowlist (drawing, math and text names) on top of a denylist of primitives
  that can read files, write files or spawn processes. ``-no-shell-escape`` is
  the second line of defence, not the first. This is not an OS sandbox; see
  ``docs/maintainer-notes.md`` §4.
* **Deterministic bytes.** Comments, XML declaration, DOCTYPE and the
  metadata block are removed, newlines are normalized to ``\\n``, and the
  serialized ``<svg>`` root is exactly what the next run produces.

The provider owns no cache. RS-805 keys the result on
``sha256(provider|source|tokens|style|toolchain|locale)`` and the toolchain
version comes from :mod:`deeptutor.review.render_tools`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.dom import Node, minidom

from deeptutor.review.render_tools import (
    PDFTOCAIRO,
    PDFTOHTML,
    XELATEX,
    probe_toolchain,
)

#: Hard ceiling for the produced SVG. A 2 MB figure is already far beyond what
#: an A4 review sheet needs; anything larger is a runaway render or an attack.
MAX_SVG_BYTES = 2 * 1024 * 1024

#: Hard ceiling for the LaTeX body. Keeps compile time bounded on a request
#: path and keeps the dangerous-command scan linear.
MAX_SOURCE_BYTES = 50_000

#: Cache namespace. Bump it whenever the sanitizer or the generated document
#: format changes: an old entry would otherwise replay SVG produced by a
#: different implementation under the same key.
CACHE_NAMESPACE = "tikz-svg-v1"

#: How much compiler output is echoed in ``reason``. Enough to name the failing
#: line, bounded so a verbose TeX log cannot bloat a response.
_ERROR_TAIL_CHARS = 400

#: Upper bound for entity declarations in the SVG prolog. Cairo emits a dozen
#: namespace entities; anything beyond this is an expansion attack.
_MAX_ENTITY_DECLARATIONS = 32

_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"
_XML_NS = "http://www.w3.org/XML/1998/namespace"

#: Elements removed with their whole subtree. Non-visual metadata is dropped
#: because it carries tool versions and creation dates (non-determinism).
_DROP_ELEMENTS = frozenset(
    {
        "script",
        "foreignobject",
        "iframe",
        "object",
        "embed",
        "audio",
        "video",
        "canvas",
        "handler",
        "listener",
        "metadata",
        "title",
        "desc",
        "animate",
        "animatemotion",
        "animatetransform",
        "set",
    }
)

#: Attributes that can load or execute content regardless of their value.
_DROP_ATTRIBUTES = frozenset(
    {"src", "srcset", "action", "formaction", "poster", "background", "srcdoc"}
)

_HREF_ATTRIBUTES = frozenset({"href", "xlink:href"})

#: ``href`` survives only when it points inside the document, or when it is an
#: inline raster image. Everything else (javascript:, data:text/html, http)
#: is removed: the export must not carry network references.
_SAFE_HREF = re.compile(
    r"^(?:#[^\s]*|data:image/(?:png|jpeg|jpg|gif|webp);base64,[A-Za-z0-9+/=\s]+)$",
    re.IGNORECASE,
)

_DANGEROUS_VALUE = re.compile(
    r"(?:javascript:|vbscript:|data:text/html)", re.IGNORECASE
)
_STYLE_EXPRESSION = re.compile(r"expression\(|behaviou?r\s*:", re.IGNORECASE)
_STYLE_URL = re.compile(r"url\(\s*(?!['\"]?#)", re.IGNORECASE)

#: TeX primitives that can touch the filesystem, the shell or the tokenizer.
#: ``includegraphics`` is included deliberately: it reads arbitrary local files
#: and can embed a local PDF into the exported page, which is an exfiltration
#: path rather than a drawing feature. Order matters only for the reported
#: name (leftmost-first alternation).
_FORBIDDEN_COMMANDS: tuple[str, ...] = (
    "write18",
    "openout",
    "openin",
    "closeout",
    "closein",
    "immediate",
    "write",
    "read",
    "@@input",
    "@input",
    "input",
    "include",
    "makeatletter",
    "makeatother",
    "catchfile",
    "catchfiledef",
    "catchfileedef",
    "catchfilexdef",
    "readfile",
    "verbatiminput",
    "includegraphics",
    "includepdf",
    "inputiffileexists",
    "lstinputlisting",
    "inputminted",
    "import",
    "subimport",
    "pdfximage",
    "pdfobj",
    "pdfprimitive",
    "pdfliteral",
    "pdfextension",
    "pdfcatalog",
    "xetexpdffile",
    "xetexpdfpage",
    "xetexpicfile",
    "iffileexists",
    "mdfivesum",
    "filesize",
    "filemoddate",
    "catcode",
    "csname",
    "directlua",
    "latelua",
    "special",
    "endlinechar",
    "everyeof",
    # The recheck proved a second file-reading family: expl3's namespace IO
    # and PGF's own image loader. ``\ExplSyntaxOn`` makes ``\file_input:n``
    # callable; blocking the switch and the entry points covers both spellings
    # (direct and ``\use:c``/``\cs:w`` indirection).
    "explsyntaxon",
    "explsyntaxoff",
    "file_input",
    "file_get",
    "ior_open",
    "iow_open",
    "tl_set_from_file",
    "use:c",
    "cs:w",
    "cs_end:",
    "pgfimage",
    "pgfdeclareimage",
    "pgfuseimage",
)

_FORBIDDEN_PATTERN = re.compile(
    r"\\(" + "|".join(_FORBIDDEN_COMMANDS) + r")(?![A-Za-z])",
    re.IGNORECASE,
)

#: Phase 36.1 command allowlist. The denylist above only closes escape hatches
#: that have been discovered; this list inverts the default so an unknown
#: primitive is rejected instead of trusted. It covers what the generator and
#: hand-written packs actually use: TikZ drawing, pgfmath, circuitikz unit
#: macros, tikz-3dplot, and the math/text commands available in the preamble
#: packages. A missing name costs a placeholder plus a warning; a missing
#: *blocked* name costs the deployment.
_ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {
        # preamble lines (validated separately by name)
        "usepackage",
        "usetikzlibrary",
        # environments and structure
        "begin",
        "end",
        "tikz",
        # drawing
        "draw",
        "fill",
        "filldraw",
        "shade",
        "shadedraw",
        "path",
        "node",
        "coordinate",
        "foreach",
        "breakforeach",
        "clip",
        "useasboundingbox",
        "matrix",
        "pic",
        "tikzset",
        "tikzstyle",
        "pattern",
        "pgfdeclarelayer",
        "pgfsetlayers",
        "pgfextra",
        "pgfkeys",
        "pgfkeysvalueof",
        "pgfmathsetmacro",
        "pgfmathsetlengthmacro",
        "pgfmathtruncatemacro",
        "pgfmathsetcount",
        "pgfmathsetseed",
        "pgfmathanglebetweenpoints",
        "pgfmathparse",
        "pgfmathresult",
        "pgfgetlastxy",
        "pgfpoint",
        "pgfpointanchor",
        "pgfpointorigin",
        "pgfpathmoveto",
        "pgfpathlineto",
        "pgfpathclose",
        "pgfusepath",
        "pgfsetlinewidth",
        "pgfsetstrokecolor",
        "pgfsetfillcolor",
        # tikz-3dplot
        "tdplotsetmaincoords",
        "tdplotsetrotatedcoords",
        "tdplotsetcoord",
        "tdplotdrawarc",
        "tdplotgetpolarcoords",
        "tdplotmainphi",
        "tdplotmaintheta",
        # user macros
        "def",
        "gdef",
        "newcommand",
        "renewcommand",
        "providecommand",
        "declaremathoperator",
        # colours
        "color",
        "textcolor",
        "colorbox",
        "fcolorbox",
        "definecolor",
        "colorlet",
        # text and spacing
        "text",
        "textbf",
        "textit",
        "textrm",
        "textsf",
        "texttt",
        "textnormal",
        "emph",
        "underline",
        "tiny",
        "scriptsize",
        "footnotesize",
        "small",
        "normalsize",
        "large",
        "Large",
        "LARGE",
        "huge",
        "Huge",
        "quad",
        "qquad",
        "enspace",
        "thinspace",
        "hspace",
        "vspace",
        "hfill",
        "vfill",
        "smallskip",
        "medskip",
        "bigskip",
        "newline",
        "linebreak",
        "par",
        "noindent",
        "phantom",
        "hphantom",
        "vphantom",
        "mbox",
        "fbox",
        "framebox",
        "makebox",
        "parbox",
        "raisebox",
        "resizebox",
        "scalebox",
        "rotatebox",
        "rule",
        "smash",
        "strut",
        "mathstrut",
        "hskip",
        "vskip",
        "kern",
        "mkern",
        "mskip",
        "enskip",
        "ensuremath",
        "textsuperscript",
        "textsubscript",
        "c",
        "v",
        "u",
        "H",
        "k",
        "r",
        "d",
        "b",
        "t",
        "o",
        "O",
        "ss",
        "ae",
        "AE",
        "oe",
        "OE",
        "aa",
        "AA",
        "l",
        "L",
        "i",
        "j",
        "S",
        "P",
        "dag",
        "ddag",
        "copyright",
        "pounds",
        "LaTeX",
        "TeX",
        # siunitx / circuitikz unit macros
        "SI",
        "si",
        "num",
        "qty",
        "unit",
        "ang",
        "numlist",
        "SIlist",
        "SIrange",
        "ohm",
        "micro",
        "milli",
        "kilo",
        "mega",
        "nano",
        "pico",
        "femto",
        "volt",
        "ampere",
        "watt",
        "farad",
        "henry",
        "siemens",
        "coulomb",
        "second",
        "meter",
        "gram",
        "litre",
        # math: greek
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "varepsilon",
        "zeta",
        "eta",
        "theta",
        "vartheta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "varpi",
        "rho",
        "varrho",
        "sigma",
        "varsigma",
        "tau",
        "upsilon",
        "phi",
        "varphi",
        "chi",
        "psi",
        "omega",
        "Gamma",
        "Delta",
        "Theta",
        "Lambda",
        "Xi",
        "Pi",
        "Sigma",
        "Upsilon",
        "Phi",
        "Psi",
        "Omega",
        # math: operators and relations
        "pm",
        "mp",
        "times",
        "div",
        "cdot",
        "ast",
        "star",
        "circ",
        "bullet",
        "oplus",
        "ominus",
        "otimes",
        "oslash",
        "odot",
        "cap",
        "cup",
        "uplus",
        "sqcap",
        "sqcup",
        "vee",
        "wedge",
        "setminus",
        "wr",
        "diamond",
        "bigtriangleup",
        "bigtriangledown",
        "triangleleft",
        "triangleright",
        "lhd",
        "rhd",
        "unlhd",
        "unrhd",
        "leq",
        "geq",
        "neq",
        "le",
        "ge",
        "ne",
        "equiv",
        "sim",
        "simeq",
        "approx",
        "cong",
        "prec",
        "succ",
        "preceq",
        "succeq",
        "ll",
        "gg",
        "subset",
        "supset",
        "subseteq",
        "supseteq",
        "in",
        "ni",
        "notin",
        "parallel",
        "perp",
        "mid",
        "models",
        "vdash",
        "dashv",
        "asymp",
        "propto",
        "leftarrow",
        "rightarrow",
        "leftrightarrow",
        "Leftarrow",
        "Rightarrow",
        "Leftrightarrow",
        "longleftarrow",
        "longrightarrow",
        "longleftrightarrow",
        "Longleftarrow",
        "Longrightarrow",
        "Longleftrightarrow",
        "uparrow",
        "downarrow",
        "updownarrow",
        "Uparrow",
        "Downarrow",
        "Updownarrow",
        "mapsto",
        "hookleftarrow",
        "hookrightarrow",
        "nearrow",
        "searrow",
        "swarrow",
        "nwarrow",
        "to",
        "gets",
        "infty",
        "partial",
        "nabla",
        "forall",
        "exists",
        "nexists",
        "neg",
        "emptyset",
        "varnothing",
        "aleph",
        "hbar",
        "ell",
        "Re",
        "Im",
        "wp",
        "top",
        "bot",
        "angle",
        "measuredangle",
        "triangle",
        "square",
        "blacksquare",
        "blacklozenge",
        "clubsuit",
        "diamondsuit",
        "heartsuit",
        "spadesuit",
        "flat",
        "natural",
        "sharp",
        "prime",
        "sum",
        "prod",
        "coprod",
        "int",
        "iint",
        "iiint",
        "oint",
        "bigcup",
        "bigcap",
        "bigsqcup",
        "bigvee",
        "bigwedge",
        "bigoplus",
        "bigotimes",
        "bigodot",
        "biguplus",
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "coth",
        "log",
        "ln",
        "lg",
        "exp",
        "lim",
        "liminf",
        "limsup",
        "max",
        "min",
        "sup",
        "inf",
        "det",
        "dim",
        "ker",
        "hom",
        "arg",
        "deg",
        "gcd",
        "Pr",
        "mod",
        "bmod",
        "pmod",
        # math: layout and fonts
        "frac",
        "dfrac",
        "tfrac",
        "sqrt",
        "root",
        "binom",
        "dbinom",
        "tbinom",
        "left",
        "right",
        "middle",
        "big",
        "Big",
        "bigg",
        "Bigg",
        "bigl",
        "bigr",
        "Bigl",
        "Bigr",
        "biggl",
        "biggr",
        "Biggl",
        "Biggr",
        "displaystyle",
        "textstyle",
        "scriptstyle",
        "scriptscriptstyle",
        "limits",
        "nolimits",
        "mathrm",
        "mathbf",
        "mathit",
        "mathsf",
        "mathtt",
        "mathcal",
        "mathbb",
        "mathfrak",
        "boldsymbol",
        "bm",
        "operatorname",
        "overset",
        "underset",
        "stackrel",
        "underbrace",
        "overbrace",
        "widehat",
        "widetilde",
        "hat",
        "check",
        "breve",
        "acute",
        "grave",
        "tilde",
        "bar",
        "vec",
        "dot",
        "ddot",
        "dddot",
        "mathring",
        "overrightarrow",
        "overleftarrow",
        "xrightarrow",
        "xleftarrow",
        "nonumber",
        "notag",
        "ldots",
        "cdots",
        "vdots",
        "ddots",
        "dots",
        "colon",
        "mathop",
        "mathrel",
        "mathbin",
        "mathord",
        "mathopen",
        "mathclose",
        "mathpunct",
        "mathinner",
    }
)

#: Macro-definition commands: the name they define is legal to use later in
#: the same figure. Without this a ``\pgfmathsetmacro{\angle}{30}`` followed
#: by ``\angle`` would read as an unknown command.
_DEFINITION_PATTERN = re.compile(
    r"\\(def|gdef|newcommand|renewcommand|providecommand|declaremathoperator"
    r"|pgfmathsetmacro|pgfmathsetlengthmacro|pgfmathtruncatemacro|pgfmathsetcount)"
    r"\*?\s*\{?\s*\\([A-Za-z@]+)"
)

#: ``\foreach`` declares its loop variables implicitly; the head up to ``in``
#: may also carry an ``[evaluate=\x as \y using ...]`` clause whose names are
#: equally legal afterwards. Without this a multi-letter loop variable such as
#: ``\point`` would read as an unknown command.
_FOREACH_HEAD_PATTERN = re.compile(r"\\foreach\b(.*?)\bin\b", re.DOTALL)

#: ``\name`` or ``\<char>``, exactly as TeX tokenizes it. ``@`` is a letter in
#: the package namespace, so it must be part of the name.
_COMMAND_PATTERN = re.compile(r"\\([A-Za-z@]+|\S)")

#: Deployment escape hatch for a trusted corpus that needs a command the list
#: has not grown to yet. Default off: untrusted content stays allowlisted.
_ALLOW_UNKNOWN_ENV = "DEEPTUTOR_TIKZ_ALLOW_UNKNOWN"

#: Environments whose body is processed as file data or code. ``filecontents``
#: writes a file even though ``\begin``/``\end`` themselves are allowed;
#: a drawing never needs it.
_BLOCKED_ENVIRONMENTS = frozenset({"filecontents", "filecontents*"})
_ENVIRONMENT_PATTERN = re.compile(r"\\(?:begin|end)\s*\{\s*([^}]*?)\s*\}")

#: Case-folded sets for lookup; TeX control-sequence names are case-sensitive,
#: but the list mixes ``\SI`` with ``\si`` and a false reject is worse than a
#: case-insensitive match here (the names themselves have no side effects).
_ALLOWED_COMMANDS_LOWER = frozenset(name.lower() for name in _ALLOWED_COMMANDS)
_FORBIDDEN_PATTERN_NAMES = frozenset(
    name.lower() for name in _FORBIDDEN_COMMANDS
)

#: Preamble-only declarations hoisted out of the body. Circuits need
#: ``\usetikzlibrary{circuits.ee.IEC}``; without hoisting the provider cannot
#: render the exact class of figures it exists for.
_PREAMBLE_LINE = re.compile(
    r"^\s*\\(?:usepackage|usetikzlibrary)\s*(?:\[[^\]]*\])?\s*\{[^}]*\}\s*$"
)
_PREAMBLE_NAMES = re.compile(
    r"^\s*\\(usepackage|usetikzlibrary)\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}\s*$"
)

#: The preamble is an **allowlist**, not a denylist. Any package can pull in
#: file-reading macros (catchfile, csvsimple, datatool, readarray, ...), and
#: ``pgfplots`` is deliberately absent too: ``\addplot table {file}`` reads a
#: local file. A drawing does not need arbitrary packages.
_ALLOWED_PACKAGES = frozenset(
    {
        "amsmath",
        "amssymb",
        "amsthm",
        "array",
        "bm",
        "booktabs",
        "calc",
        "circuitikz",
        "longtable",
        "mathtools",
        "siunitx",
        "tikz",
        "tikz-3dplot",
        "upgreek",
        "xcolor",
    }
)

#: TikZ libraries are code shipped with the TeX distribution, not user
#: content; allow by root name (``circuits.ee.IEC`` → ``circuits``).
_ALLOWED_LIBRARY_ROOTS = frozenset(
    {
        "angles",
        "arrows",
        "automata",
        "backgrounds",
        "babel",
        "calc",
        "calendar",
        "chains",
        "circuits",
        "decorations",
        "er",
        "fadings",
        "fit",
        "intersections",
        "lindenmayersystems",
        "matrix",
        "mindmap",
        "patterns",
        "petri",
        "plotmarks",
        "positioning",
        "quotes",
        "scopes",
        "shadows",
        "shapes",
        "snakes",
        "through",
        "topaths",
        "trees",
        "3d",
        "perspective",
    }
)

_DOCTYPE = re.compile(r"<!DOCTYPE[^>\[]*(?:\[[^\]]*\])?\s*>", re.IGNORECASE | re.DOTALL)
_ENTITY = re.compile(r"<!ENTITY\b[^>]*>", re.IGNORECASE | re.DOTALL)
_EXTERNAL_ENTITY = re.compile(
    r"<!ENTITY\b[^>]*?\b(?:SYSTEM|PUBLIC)\b", re.IGNORECASE | re.DOTALL
)


class TikzRenderError(RuntimeError):
    """A TikZ figure could not be produced.

    Attributes:
        reason: Human-readable cause, safe to surface in an export warning.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class TikzRender:
    """A rendered figure plus the optional layout verdict.

    ``render_tikz`` keeps returning plain SVG for callers that only need the
    image; the exporter uses this object so a layout failure can be surfaced
    as a warning without losing the SVG.
    """

    svg: str
    layout_error: str = ""
    layout_warnings: tuple[str, ...] = ()


def render_tikz(
    source: str,
    *,
    title: str = "",
    timeout_seconds: float = 30.0,
) -> str:
    """Render a TikZ body into sanitized SVG text.

    This is the SVG-only wrapper over :func:`render_tikz_document`.

    Args:
        source: TikZ drawing commands (the body of a ``tikzpicture``). A full
            ``tikzpicture`` environment is also accepted. ``\\usepackage`` and
            ``\\usetikzlibrary`` lines are hoisted into the preamble.
        title: Optional accessible title; inserted as the first child of the
            ``<svg>`` root. Empty means no title element is emitted.
        timeout_seconds: Timeout for *each* external command.

    Returns:
        Normalized SVG text starting with ``<svg`` and ending with ``\\n``.

    Raises:
        TikzRenderError: Missing toolchain, forbidden LaTeX, compile or
            conversion failure, or output that fails sanitization.
        ValueError: ``timeout_seconds`` is not a positive finite number; that
            is a caller bug, not a render failure.
    """
    if not isinstance(timeout_seconds, (int, float)) or not (
        timeout_seconds > 0 and float(timeout_seconds) != float("inf")
    ):
        raise ValueError("timeout_seconds must be a positive finite number")
    timeout = float(timeout_seconds)
    # No layout gate here: this wrapper returns a bare string and has nowhere
    # to report a warning. Export goes through ``render_tikz_document``.
    return render_tikz_document(
        source, title=title, timeout_seconds=timeout, check_layout=False
    ).svg


def _toolchain_fingerprint() -> str:
    """Stable identity of the binaries that produce the SVG.

    The version string alone is not enough: a different ``pdftocairo`` build
    renders different glyph outlines at the same version number, and an
    overridden executable path is a different toolchain by definition. Both
    are hashed into the cache key so an upgrade never replays an old figure.
    """
    toolchain = probe_toolchain()
    parts: list[str] = []
    for name in (XELATEX, PDFTOCAIRO):
        capability = toolchain.capability(name)
        if capability is None or not capability.available:
            # ``_resolve_tools`` raises before this is ever used; the empty
            # marker only keeps the function total.
            parts.append(f"{name}=unavailable")
            continue
        parts.append(
            f"{name}={capability.executable or name}|{capability.version}"
        )
    return "\n".join(parts)


def _cache_for_rendering():
    """Return a cache handle, or ``None`` when caching is not configured.

    Caching is opt-in through ``DEEPTUTOR_REVIEW_CACHE``. Silence by default is
    deliberate: a render-path cache changes how often an external compiler
    runs, and a test (or a debugging session) that expects the compiler to run
    must not be surprised by a warm entry. Deployments that want RS-805 set the
    variable once.
    """
    if not os.environ.get("DEEPTUTOR_REVIEW_CACHE", "").strip():
        return None
    try:
        from deeptutor.review.cache import DiskCache
    except ImportError:
        return None
    return DiskCache()


def _layout_fingerprint(check_layout: bool) -> str:
    """Identity of the gates, so a tool change invalidates old verdicts.

    The gate results are cached alongside the SVG. If ``pdftotext`` or
    ``pdftohtml`` appears, disappears or is upgraded, the stored verdict no
    longer describes what this machine would report, so it must not be
    replayed.
    """
    if not check_layout:
        return "layout=disabled"
    from deeptutor.review.render_tools import PDFTOTEXT

    toolchain = probe_toolchain()

    def _identity(name: str) -> str:
        capability = toolchain.capability(name)
        if capability is None or not capability.available:
            return "unavailable"
        return f"{capability.executable or name}|{capability.version}"

    return f"layout={_identity(PDFTOTEXT)};font={_identity(PDFTOHTML)}"


def _cached_render(
    cache: Any,
    key: str,
    *,
    title: str = "",
) -> TikzRender | None:
    """Decode a cached render; any malformed entry is a miss, never an error."""
    raw = cache.get(key, suffix="json")
    if raw is None:
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
        svg = payload["svg"]
        layout_error = payload.get("layout_error", "")
        warnings = payload.get("layout_warnings", [])
    except (KeyError, TypeError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(svg, str) or not svg.startswith("<svg"):
        return None
    if not isinstance(layout_error, str):
        return None
    if not isinstance(warnings, list) or not all(
        isinstance(item, str) for item in warnings
    ):
        return None
    # Defense in depth: the entry was sanitized before it was stored, but the
    # cache is a file on disk and may have been written by an older build or
    # tampered with. Sanitizing again costs milliseconds next to a render.
    # ``sanitize_svg`` drops existing metadata, so pass the title through to
    # keep exactly one accessible title on the re-sanitized document.
    try:
        svg = sanitize_svg(svg, title=title)
    except TikzRenderError:
        return None
    return TikzRender(
        svg=svg, layout_error=layout_error, layout_warnings=tuple(warnings)
    )


def _store_render(cache: Any, key: str, rendered: TikzRender) -> None:
    """Persist one render. A cache write must never fail the figure."""
    try:
        payload = json.dumps(
            {
                "svg": rendered.svg,
                "layout_error": rendered.layout_error,
                "layout_warnings": list(rendered.layout_warnings),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return
    try:
        cache.put(key, suffix="json", data=payload)
    except (OSError, ValueError):
        # ``ValueError`` would be a caller bug, but losing an optimisation is
        # still better than losing the figure the caller already paid for.
        return


def render_tikz_document(
    source: str,
    *,
    title: str = "",
    timeout_seconds: float = 30.0,
    check_layout: bool = True,
) -> TikzRender:
    """Render TikZ and optionally gate the PDF layout.

    The layout gate only ever downgrades to a warning: a figure that renders
    but has overlapping labels is still better than a blank placeholder, and
    the warning is visible in ``quality.warnings``. Toolchain absence is
    reported in ``layout_warnings`` rather than raised, because the figure
    itself is fine.
    """
    if not isinstance(timeout_seconds, (int, float)) or not (
        timeout_seconds > 0 and float(timeout_seconds) != float("inf")
    ):
        raise ValueError("timeout_seconds must be a positive finite number")
    timeout = float(timeout_seconds)

    body = _prepare_source(source)

    xelatex, pdftocairo = _resolve_tools()

    cache = _cache_for_rendering()
    cache_key_value = ""
    if cache is not None:
        from deeptutor.review.cache import cache_key

        cache_key_value = cache_key(
            CACHE_NAMESPACE,
            _toolchain_fingerprint(),
            _layout_fingerprint(check_layout),
            str(round(timeout, 3)),
            title,
            body,
        )
        cached = _cached_render(cache, cache_key_value, title=title)
        if cached is not None:
            return cached

    with tempfile.TemporaryDirectory(prefix="deeptutor-tikz-") as raw_dir:
        workdir = Path(raw_dir)
        tex_path = workdir / "figure.tex"
        pdf_path = workdir / "figure.pdf"
        svg_path = workdir / "figure.svg"
        tex_path.write_text(_build_document(body), encoding="utf-8", newline="\n")

        _run(
            [
                xelatex,
                "-no-shell-escape",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "figure.tex",
            ],
            cwd=workdir,
            timeout=timeout,
        )
        # "Exit code 0" is not proof: the product is.
        if not _non_empty(pdf_path):
            raise TikzRenderError(
                "xelatex reported success but produced no PDF "
                "(figure.pdf missing or empty)"
            )

        layout_error = ""
        layout_warnings: list[str] = []
        if check_layout:
            layout_error, layout_warnings = _gate_layout(pdf_path)

        _run(
            [pdftocairo, "-svg", "figure.pdf", "figure.svg"],
            cwd=workdir,
            timeout=timeout,
        )
        if not _non_empty(svg_path):
            raise TikzRenderError(
                "pdftocairo reported success but produced no SVG "
                "(figure.svg missing or empty)"
            )
        raw_svg = svg_path.read_bytes()

    svg = sanitize_svg(raw_svg, title=title)
    rendered_document = TikzRender(
        svg=svg,
        layout_error=layout_error,
        layout_warnings=tuple(layout_warnings),
    )
    if cache is not None:
        _store_render(cache, cache_key_value, rendered_document)
    return rendered_document


def _gate_layout(pdf_path: Path) -> tuple[str, list[str]]:
    """Run the layout and font-size gates without letting either fail a render.

    A machine without a ``-bbox``-capable ``pdftotext`` or without poppler's
    ``pdftohtml`` simply has no such gate: that is a deployment choice, not a
    defect in this figure, so it is silent. Only a gate that actually ran and
    rejected the figure is reported.

    Two severities, matching the existing layout semantics:

    * a measured defect (overlapping words, a font below 6 pt) becomes the
      single ``layout_error`` — the figure is still delivered, but the export
      carries a blocking verdict;
    * a gate that could not run (timeout, non-zero exit, unparseable output)
      becomes a warning, because losing the figure over a broken probe would
      be worse than delivering it with a caveat.
    """
    layout_error, warnings = _run_layout_gate(pdf_path)
    font_error, font_warnings = _run_font_gate(pdf_path)
    warnings.extend(font_warnings)
    # A measured layout defect outranks a font-size defect as the blocking
    # verdict — the author should fix the overlap first — but a second measured
    # defect must not disappear entirely, or fixing the first one silently
    # reveals a problem the author already paid a render to discover.
    if layout_error and font_error:
        warnings.append(font_error)
    return layout_error or font_error, warnings


def _run_layout_gate(pdf_path: Path) -> tuple[str, list[str]]:
    try:
        from deeptutor.review.layout_gate import check_layout
    except ImportError:
        return "", []
    try:
        report = check_layout(pdf_path, min_font_size=None)
    except Exception as exc:  # noqa: BLE001 - a gate must not lose the figure
        return "", [f"layout check failed: {type(exc).__name__}"]

    if report.ok:
        return "", []
    # Toolchain absence surfaces as its own error code; treat it as "not
    # configured" rather than as a warning about the figure.
    codes = {issue.code for issue in report.issues}
    if codes and codes <= {"pdftotext_unavailable", "pdf_missing"}:
        return "", []
    errors = [issue for issue in report.issues if issue.severity == "error"]
    if not errors:
        return "", [issue.detail for issue in report.issues]
    first = errors[0]
    summary = f"layout check failed: {first.code}"
    if first.detail:
        summary += f" ({first.detail})"
    return summary, [issue.detail for issue in report.issues if issue.severity == "warning"]


def _run_font_gate(pdf_path: Path) -> tuple[str, list[str]]:
    """Second half of :func:`_gate_layout`; see there for the severity rules."""
    # Read the capability from this module's toolchain snapshot: when the probe
    # has no ``pdftohtml`` entry at all (older deployments, test doubles) the
    # gate is simply not configured. That is silent, and it also guarantees no
    # subprocess is started for a tool that was never advertised.
    capability = probe_toolchain().capability(PDFTOHTML)
    if capability is None or not capability.available:
        return "", []
    try:
        from deeptutor.review.font_gate import check_font_sizes
    except ImportError:
        return "", []
    try:
        report = check_font_sizes(pdf_path)
    except Exception as exc:  # noqa: BLE001 - a gate must not lose the figure
        return "", [f"font check failed: {type(exc).__name__}"]

    # Deployment choice: without the tool there is no gate, and the figure is
    # not at fault. A missing PDF was already reported by the layout gate.
    if all(
        issue.code in {"pdftohtml_unavailable", "pdf_missing"}
        for issue in report.issues
    ):
        return "", []

    too_small = [
        issue
        for issue in report.issues
        if issue.code == "font_size_below_minimum"
    ]
    if too_small:
        first = too_small[0]
        size = f"{first.size:g}pt" if first.size is not None else "below minimum"
        summary = (
            f"font size below {report.min_font_size:g}pt: "
            f"{size} (page {first.page})"
        )
        if len(too_small) > 1:
            summary += f" and {len(too_small) - 1} more"
        # Secondary warnings (a truncated issue list, say) stay visible even
        # when the blocking verdict is already taken.
        return summary, [
            issue.detail
            for issue in report.issues
            if issue.severity == "warning"
        ]

    failures = [
        f"font check failed: {issue.code}"
        for issue in report.issues
        if issue.severity == "error"
    ]
    warnings = [issue.detail for issue in report.issues if issue.severity == "warning"]
    return "", [*failures, *warnings]


def sanitize_svg(raw: bytes | str, *, title: str = "") -> str:
    """Return a deterministic, script-free SVG document.

    Removes the DOCTYPE, XML declaration, comments, processing instructions,
    ``<script>``/``<foreignObject>`` and friends, event handler attributes,
    external or ``javascript:`` links, and inline style that can execute.
    Comments and metadata are dropped so byte-identical input yields
    byte-identical output.

    Args:
        raw: SVG bytes or text as produced by the converter.
        title: Optional accessible title inserted as the root's first child.

    Raises:
        TikzRenderError: Output is not UTF-8, exceeds :data:`MAX_SVG_BYTES`,
            is not an ``<svg>`` document, or cannot be parsed.
    """
    if isinstance(raw, bytes):
        declared_size = len(raw)
    else:
        declared_size = len(raw.encode("utf-8"))
    if declared_size > MAX_SVG_BYTES:
        raise TikzRenderError(f"rendered SVG exceeds {MAX_SVG_BYTES} bytes")
    text = _decode_svg(raw)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    _guard_entities(text)

    try:
        document = minidom.parseString(_strip_prolog(text))
    except Exception as exc:  # expat raises a family of ParseError subclasses
        raise TikzRenderError(f"rendered SVG could not be parsed: {exc}") from exc

    root = document.documentElement
    if root is None or (root.localName or root.nodeName).lower() != "svg":
        raise TikzRenderError("rendered output is not an <svg> document")
    if root.namespaceURI not in (None, _SVG_NS):
        raise TikzRenderError(
            f"rendered output has unexpected root namespace {root.namespaceURI!r}"
        )

    # The root carries ``onload`` just as readily as any child; sanitize it too.
    _sanitize_attributes(root)
    _sanitize_children(root)

    if title:
        title_node = document.createElement("title")
        title_node.appendChild(document.createTextNode(_normalize_text(title)))
        root.insertBefore(title_node, root.firstChild)

    cleaned = root.toxml().replace("\r\n", "\n").replace("\r", "\n")
    if not cleaned.endswith("\n"):
        cleaned += "\n"
    if len(cleaned.encode("utf-8")) > MAX_SVG_BYTES:
        raise TikzRenderError(f"rendered SVG exceeds {MAX_SVG_BYTES} bytes")
    return cleaned


def _prepare_source(source: str) -> str:
    if not isinstance(source, str):
        raise TikzRenderError("TikZ source must be text")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise TikzRenderError(
            f"TikZ source exceeds {MAX_SOURCE_BYTES} bytes"
        )
    stripped = _strip_comments(source).strip()
    if not stripped:
        raise TikzRenderError("TikZ source is empty")

    match = _FORBIDDEN_PATTERN.search(stripped)
    if match is not None:
        raise TikzRenderError(
            f"forbidden LaTeX command \\{match.group(1)} in TikZ source"
        )
    # TeX expands ``^^xx`` (and ``^^X``) before tokenizing, so ``\^^69nput``
    # reaches xelatex as ``\input`` without ever containing the forbidden
    # substring. No legitimate drawing needs the escape: reject it outright
    # instead of trying to normalize every spelling.
    if "^^" in stripped:
        raise TikzRenderError("forbidden TeX ^^ character escape in TikZ source")
    _validate_preamble(stripped)
    _validate_commands(stripped)
    return stripped


def _validate_commands(source: str) -> None:
    """Reject control sequences outside the drawing allowlist.

    Single-character control sequences (``\\``, ``\\%``, ``\\,``, accents) are
    always allowed: no TeX primitive names a file or a process with one
    character. Multi-letter names must be listed above or declared elsewhere in
    the same source by a definition command or a ``\\foreach`` loop variable;
    the definition body is scanned by the same rule.
    """
    for match in _ENVIRONMENT_PATTERN.finditer(source):
        if match.group(1).lower() in _BLOCKED_ENVIRONMENTS:
            raise TikzRenderError(
                f"forbidden LaTeX environment: {match.group(1)}"
            )
    if os.environ.get(_ALLOW_UNKNOWN_ENV, "").strip() == "1":
        return
    defined = {
        match.group(2).lower()
        for match in _DEFINITION_PATTERN.finditer(source)
        if match.group(2).lower() not in _FORBIDDEN_PATTERN_NAMES
    }
    for head in _FOREACH_HEAD_PATTERN.finditer(source):
        for match in _COMMAND_PATTERN.finditer(head.group(1)):
            name = match.group(1).lower()
            if len(name) > 1 and name not in _FORBIDDEN_PATTERN_NAMES:
                defined.add(name)
    unknown: list[str] = []
    for match in _COMMAND_PATTERN.finditer(source):
        name = match.group(1)
        if len(name) == 1:
            continue
        bare = name.rstrip("*").lower()
        if bare in ("usepackage", "usetikzlibrary"):
            # ``_validate_preamble`` only inspects whole lines; an inline
            # invocation would reach the document body, where LaTeX can still
            # try to load code. Only the hoisted form is legal.
            start = source.rfind("\n", 0, match.start()) + 1
            end = source.find("\n", match.start())
            line = source[start : end if end != -1 else len(source)]
            if _PREAMBLE_NAMES.match(line) is None:
                unknown.append(name)
            continue
        if bare in _ALLOWED_COMMANDS_LOWER or bare in defined:
            continue
        if name not in unknown:
            unknown.append(name)
        if len(unknown) >= 5:
            break
    if unknown:
        shown = ", ".join(f"\\{name}" for name in unknown)
        raise TikzRenderError(
            f"unsupported LaTeX command(s) in TikZ source: {shown}"
        )


def _validate_preamble(source: str) -> None:
    """Reject packages/libraries outside the allowlist.

    ``_build_document`` hoists these lines into the real preamble, so this is
    the only place a figure can load code with file-reading side effects.
    """
    for line in source.split("\n"):
        match = _PREAMBLE_NAMES.match(line)
        if match is None:
            continue
        command, names = match.groups()
        for raw_name in names.split(","):
            name = raw_name.strip().lower()
            if not name:
                continue
            if command == "usepackage":
                allowed = name in _ALLOWED_PACKAGES
            else:
                allowed = name.split(".", 1)[0] in _ALLOWED_LIBRARY_ROOTS
            if not allowed:
                if command == "usepackage":
                    supported = ", ".join(sorted(_ALLOWED_PACKAGES))
                else:
                    supported = ", ".join(sorted(_ALLOWED_LIBRARY_ROOTS))
                raise TikzRenderError(
                    f"unsupported LaTeX {command} name: {name}; "
                    f"supported: {supported}"
                )


def _strip_comments(source: str) -> str:
    """Remove TeX ``%`` comments while keeping escaped ``\\%``."""
    kept_lines: list[str] = []
    for line in source.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        end = len(line)
        for index, char in enumerate(line):
            if char != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                end = index
                break
        kept_lines.append(line[:end])
    return "\n".join(kept_lines)


def _split_preamble(body: str) -> tuple[list[str], str]:
    preamble: list[str] = []
    remaining: list[str] = []
    for line in body.split("\n"):
        if _PREAMBLE_LINE.match(line):
            preamble.append(line.strip())
        else:
            remaining.append(line)
    return preamble, "\n".join(remaining).strip()


def _build_document(body: str) -> str:
    preamble, drawing = _split_preamble(body)
    if re.search(r"\\begin\s*\{tikzpicture\}", drawing):
        picture = drawing
    else:
        picture = "\\begin{tikzpicture}\n" + drawing + "\n\\end{tikzpicture}"

    lines = [
        "\\documentclass[border=2pt]{standalone}",
        "\\usepackage{tikz}",
        "\\usepackage{amsmath}",
        "\\usepackage{amssymb}",
        *preamble,
        "\\begin{document}",
        picture,
        "\\end{document}",
        "",
    ]
    return "\n".join(lines)


def _resolve_tools() -> tuple[str, str]:
    """Return executable paths, or raise when the toolchain is incomplete."""
    toolchain = probe_toolchain()
    problems: list[str] = []
    executables: dict[str, str] = {}
    for name in (XELATEX, PDFTOCAIRO):
        capability = toolchain.capability(name)
        if capability is None or not capability.available:
            detail = capability.reason if capability is not None else "not probed"
            problems.append(f"{name} ({detail})")
        else:
            executables[name] = capability.executable or name
    if problems:
        raise TikzRenderError("TikZ toolchain incomplete: " + "; ".join(problems))
    return executables[XELATEX], executables[PDFTOCAIRO]


def _run(args: list[str], *, cwd: Path, timeout: float) -> None:
    """Run one external tool; turn every failure mode into a render error."""
    tool = Path(args[0]).stem
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd),
            shell=False,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TikzRenderError(f"{tool} timed out after {timeout:g}s") from exc
    except OSError as exc:
        raise TikzRenderError(f"{tool} could not run: {exc}") from exc
    except ValueError as exc:  # e.g. embedded NUL byte in an override path
        raise TikzRenderError(f"{tool} could not run: {exc}") from exc

    if completed.returncode != 0:
        tail = _output_tail(completed)
        suffix = f": {tail}" if tail else ""
        raise TikzRenderError(
            f"{tool} failed with status {completed.returncode}{suffix}"
        )


def _output_tail(completed: subprocess.CompletedProcess[bytes]) -> str:
    chunks = [completed.stdout or b"", completed.stderr or b""]
    raw = b"\n".join(chunk for chunk in chunks if chunk)
    text = raw.decode("utf-8", errors="replace")
    return " ".join(text.split())[-_ERROR_TAIL_CHARS:]


def _non_empty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _decode_svg(raw: bytes | str) -> str:
    if isinstance(raw, str):
        return raw
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TikzRenderError("rendered SVG is not valid UTF-8") from exc


def _guard_entities(text: str) -> None:
    """Reject external entities and absurd entity amplification before parsing."""
    declarations = _ENTITY.findall(text)
    if len(declarations) > _MAX_ENTITY_DECLARATIONS:
        raise TikzRenderError("rendered SVG declares too many XML entities")
    if _EXTERNAL_ENTITY.search(text):
        raise TikzRenderError("rendered SVG declares an external XML entity")


def _strip_prolog(text: str) -> str:
    """Drop the XML declaration and DOCTYPE.

    Cairo writes an internal DTD subset full of namespace entities; the
    declarations themselves stay (expat needs them to resolve ``&ns_extend;``)
    but the prolog is removed from the serialized output because we serialize
    ``documentElement`` only. A DOCTYPE without a subset is removed here so an
    external DTD can never be fetched.
    """
    text = re.sub(r"^\s*<\?xml[^>]*\?>", "", text, count=1, flags=re.IGNORECASE)
    doctype = _DOCTYPE.search(text)
    # Keep a DOCTYPE that carries an internal subset: expat needs it to resolve
    # the namespace entities cairo writes. A bare DOCTYPE only names an
    # external DTD, which must never be fetched.
    if doctype is not None and "[" not in doctype.group(0):
        text = text[: doctype.start()] + text[doctype.end() :]
    return text


def _sanitize_children(parent: Node) -> None:
    for child in list(parent.childNodes):
        if child.nodeType in (
            Node.COMMENT_NODE,
            Node.PROCESSING_INSTRUCTION_NODE,
            Node.DOCUMENT_TYPE_NODE,
        ):
            parent.removeChild(child)
            continue
        if child.nodeType != Node.ELEMENT_NODE:
            continue
        if _is_dropped_element(child):
            parent.removeChild(child)
            continue
        _sanitize_attributes(child)
        _sanitize_children(child)


def _normalize_text(value: str) -> str:
    """Collapse control characters that XML 1.0 cannot represent."""
    return "".join(
        char if char >= " " or char in "\t\n" else " " for char in str(value)
    )


def _is_dropped_element(node: Node) -> bool:
    local = (node.localName or node.nodeName or "").split(":")[-1].lower()
    if local in _DROP_ELEMENTS:
        return True
    namespace = node.namespaceURI
    return namespace is not None and namespace != _SVG_NS


def _sanitize_attributes(node: Node) -> None:
    for item in list(node.attributes.items()):
        attr = node.getAttributeNode(item[0])
        if attr is None:
            continue
        name = attr.name
        bare = name.split(":")[-1].lower()
        value = attr.value

        if bare.startswith("on") and len(bare) > 2:
            node.removeAttribute(name)
            continue
        if bare in _DROP_ATTRIBUTES:
            node.removeAttribute(name)
            continue
        if name.startswith("xmlns"):
            continue
        if bare in _HREF_ATTRIBUTES:
            if not _SAFE_HREF.match(value.strip()):
                node.removeAttribute(name)
            continue
        if bare == "style":
            if _style_is_unsafe(value):
                node.removeAttribute(name)
            continue
        if _DANGEROUS_VALUE.search(value):
            node.removeAttribute(name)


def _style_is_unsafe(value: str) -> bool:
    return bool(
        _DANGEROUS_VALUE.search(value)
        or _STYLE_EXPRESSION.search(value)
        or _STYLE_URL.search(value)
    )


__all__ = (
    "MAX_SOURCE_BYTES",
    "MAX_SVG_BYTES",
    "TikzRenderError",
    "render_tikz",
    "sanitize_svg",
)
