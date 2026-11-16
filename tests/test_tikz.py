"""Tests for the TikZ provider (RS-803).

Every test replaces ``probe_toolchain`` and ``subprocess.run``; no real
``xelatex`` or ``pdftocairo`` is executed. The fake runner writes the same
artifacts the real tools would, so the provider's "check the product, not the
exit code" path is exercised as well.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from deeptutor.review import tikz
from deeptutor.review.render_tools import (
    PDFTOCAIRO,
    PDFTOHTML,
    XELATEX,
    ToolCapability,
    Toolchain,
)

_CLEAN_SVG = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<svg xmlns="http://www.w3.org/2000/svg" width="100pt" height="50pt">'
    '<path d="M 0 0 L 10 10" fill="none" stroke="#000000"/></svg>\n'
)

_DIRTY_SVG = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" \
"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd" [
<!ENTITY ns_extend "http://ns.adobe.com/Extensibility/1.0/">
]>
<!-- Created with cairo 1.17.4 -->
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="100pt" height="50pt">
<metadata><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:x="&ns_extend;"></rdf:RDF></metadata>
<script type="text/javascript">alert(1)</script>
<script src="evil.js" />
<g ONLOAD="alert(2)" onclick='steal()'>
<foreignObject width="10" height="10"><div>hi</div></foreignObject>
<a xlink:href="javascript:alert(3)"><path d="M0 0"/></a>
<a href="data:text/html;base64,AAAA"><path d="M1 1"/></a>
<a xlink:href="https://example.invalid/x.svg"><path d="M2 2"/></a>
<use xlink:href="#glyph0-1"/>
<path d="M0 0 L1 1" fill="none" stroke="#000"/>
</g>
</svg>
"""


def _available_toolchain() -> Toolchain:
    return Toolchain(
        capabilities=(
            ToolCapability(
                name=XELATEX,
                executable=str(Path("C:/tex/xelatex.exe")),
                available=True,
                version="XeTeX 3.14",
            ),
            ToolCapability(
                name=PDFTOCAIRO,
                executable=str(Path("C:/poppler/pdftocairo.exe")),
                available=True,
                version="pdftocairo version 22.02.0",
            ),
        )
    )


def _missing_toolchain(*missing: str) -> Toolchain:
    capabilities = []
    for name in (XELATEX, PDFTOCAIRO):
        gone = name in missing
        capabilities.append(
            ToolCapability(
                name=name,
                executable=None,
                available=not gone,
                reason="not found on PATH" if gone else "",
            )
        )
    return Toolchain(capabilities=tuple(capabilities))


def _completed(
    args: list[str],
    returncode: int = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        args=args, returncode=returncode, stdout=stdout, stderr=stderr
    )


class FakeRunner:
    """Stand-in for ``subprocess.run`` that writes the expected artifacts."""

    def __init__(
        self,
        *,
        svg: bytes | str = _CLEAN_SVG,
        xelatex_status: int = 0,
        xelatex_output: bytes = b"",
        pdftocairo_status: int = 0,
        pdftocairo_output: bytes = b"",
        timeout_tool: str | None = None,
        write_pdf: bool = True,
        write_svg: bool = True,
    ) -> None:
        self.svg = svg
        self.xelatex_status = xelatex_status
        self.xelatex_output = xelatex_output
        self.pdftocairo_status = pdftocairo_status
        self.pdftocairo_output = pdftocairo_output
        self.timeout_tool = timeout_tool
        self.write_pdf = write_pdf
        self.write_svg = write_svg
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.tex_sources: list[str] = []

    def __call__(self, args: list[str], **kwargs: Any) -> Any:
        self.calls.append((list(args), dict(kwargs)))
        tool = Path(args[0]).stem.lower()
        if self.timeout_tool == tool:
            raise subprocess.TimeoutExpired(args, kwargs.get("timeout"))
        workdir = Path(kwargs["cwd"])
        if tool == "xelatex":
            tex_path = workdir / "figure.tex"
            if tex_path.is_file():
                self.tex_sources.append(tex_path.read_text(encoding="utf-8"))
            if self.write_pdf:
                (workdir / "figure.pdf").write_bytes(b"%PDF-1.5\n")
            return _completed(
                args, self.xelatex_status, stdout=self.xelatex_output
            )
        if self.write_svg:
            target = workdir / args[-1]
            if isinstance(self.svg, bytes):
                target.write_bytes(self.svg)
            else:
                target.write_text(self.svg, encoding="utf-8", newline="")
        return _completed(
            args, self.pdftocairo_status, stderr=self.pdftocairo_output
        )


@pytest.fixture()
def available_toolchain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tikz, "probe_toolchain", _available_toolchain)


def _install_runner(
    monkeypatch: pytest.MonkeyPatch, runner: FakeRunner
) -> FakeRunner:
    monkeypatch.setattr(subprocess, "run", runner)
    return runner


def test_render_returns_sanitized_svg(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None
) -> None:
    runner = _install_runner(monkeypatch, FakeRunner(svg=_DIRTY_SVG))

    svg = tikz.render_tikz(
        r"\draw (0,0) -- (1,1);", title="图 A & B <test>", timeout_seconds=5.0
    )

    assert svg.startswith("<svg")
    assert svg.endswith("\n")
    assert "<script" not in svg.lower()
    assert "foreignobject" not in svg.lower()
    assert "onload" not in svg.lower()
    assert "onclick" not in svg.lower()
    assert "javascript:" not in svg.lower()
    assert "data:text/html" not in svg.lower()
    assert "https://example.invalid" not in svg
    assert "ns_extend" not in svg
    assert "rdf:RDF" not in svg
    assert "<!--" not in svg
    assert "<!DOCTYPE" not in svg
    assert "<?xml" not in svg
    assert "\r" not in svg
    assert 'xlink:href="#glyph0-1"' in svg
    assert "<path" in svg
    assert "<title>图 A &amp; B &lt;test&gt;</title>" in svg

    assert len(runner.calls) == 2
    xelatex_args, xelatex_kwargs = runner.calls[0]
    assert Path(xelatex_args[0]).stem == "xelatex"
    assert "-no-shell-escape" in xelatex_args
    assert "-interaction=nonstopmode" in xelatex_args
    assert "-halt-on-error" in xelatex_args
    assert xelatex_args[-1] == "figure.tex"
    assert xelatex_kwargs["shell"] is False
    assert xelatex_kwargs["timeout"] == 5.0
    assert Path(xelatex_kwargs["cwd"]).name.startswith("deeptutor-tikz-")
    assert runner.tex_sources[0].startswith("\\documentclass")
    assert "\\begin{tikzpicture}" in runner.tex_sources[0]
    assert "\\draw (0,0) -- (1,1);" in runner.tex_sources[0]

    pdftocairo_args, pdftocairo_kwargs = runner.calls[1]
    assert Path(pdftocairo_args[0]).stem == "pdftocairo"
    assert pdftocairo_args[1:] == ["-svg", "figure.pdf", "figure.svg"]
    assert pdftocairo_kwargs["shell"] is False


def test_render_is_deterministic(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None
) -> None:
    _install_runner(monkeypatch, FakeRunner(svg=_DIRTY_SVG))

    first = tikz.render_tikz(r"\draw (0,0) -- (1,1);", title="T")
    second = tikz.render_tikz(r"\draw (0,0) -- (1,1);", title="T")

    assert first == second


def test_render_without_title_has_no_title_element(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None
) -> None:
    _install_runner(monkeypatch, FakeRunner())

    svg = tikz.render_tikz(r"\draw (0,0) -- (1,1);")

    assert "<title>" not in svg
    assert svg.startswith("<svg")
    assert "<path" in svg


def test_missing_toolchain_reports_every_missing_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tikz, "probe_toolchain", lambda: _missing_toolchain(XELATEX, PDFTOCAIRO)
    )
    runner = _install_runner(monkeypatch, FakeRunner())

    with pytest.raises(tikz.TikzRenderError) as excinfo:
        tikz.render_tikz(r"\draw (0,0) -- (1,1);")

    reason = excinfo.value.reason
    assert "xelatex" in reason
    assert "pdftocairo" in reason
    assert "not found on PATH" in reason
    assert runner.calls == []


def test_missing_single_tool_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tikz, "probe_toolchain", lambda: _missing_toolchain(PDFTOCAIRO)
    )
    runner = _install_runner(monkeypatch, FakeRunner())

    with pytest.raises(tikz.TikzRenderError) as excinfo:
        tikz.render_tikz(r"\draw (0,0) -- (1,1);")

    assert "pdftocairo" in excinfo.value.reason
    assert "xelatex" not in excinfo.value.reason
    assert runner.calls == []


def test_xelatex_nonzero_exit_raises_with_output_tail(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None
) -> None:
    _install_runner(
        monkeypatch,
        FakeRunner(
            xelatex_status=1,
            xelatex_output=b"! Undefined control sequence.\nl.5 \\foo\n",
        ),
    )

    with pytest.raises(tikz.TikzRenderError) as excinfo:
        # The allowlist is not what this test covers: let the fake compiler
        # report the failure the way a real undefined command would.
        monkeypatch.setenv(tikz._ALLOW_UNKNOWN_ENV, "1")
        tikz.render_tikz(r"\foo")

    reason = excinfo.value.reason
    assert "xelatex" in reason
    assert "status 1" in reason
    assert "Undefined control sequence" in reason


def test_pdftocairo_nonzero_exit_raises(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None
) -> None:
    _install_runner(
        monkeypatch,
        FakeRunner(pdftocairo_status=2, pdftocairo_output=b"Syntax Error: bad pdf"),
    )

    with pytest.raises(tikz.TikzRenderError) as excinfo:
        tikz.render_tikz(r"\draw (0,0) -- (1,1);")

    reason = excinfo.value.reason
    assert "pdftocairo" in reason
    assert "status 2" in reason
    assert "Syntax Error" in reason


def test_timeout_raises_with_tool_name(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None
) -> None:
    _install_runner(
        monkeypatch, FakeRunner(timeout_tool="xelatex")
    )

    with pytest.raises(tikz.TikzRenderError) as excinfo:
        tikz.render_tikz(r"\draw (0,0) -- (1,1);", timeout_seconds=3)

    reason = excinfo.value.reason
    assert "xelatex" in reason
    assert "timed out" in reason
    assert "3" in reason


def test_success_exit_without_pdf_is_rejected(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None
) -> None:
    _install_runner(monkeypatch, FakeRunner(write_pdf=False))

    with pytest.raises(tikz.TikzRenderError) as excinfo:
        tikz.render_tikz(r"\draw (0,0) -- (1,1);")

    assert "no pdf" in excinfo.value.reason.lower()


def test_success_exit_without_svg_is_rejected(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None
) -> None:
    _install_runner(monkeypatch, FakeRunner(write_svg=False))

    with pytest.raises(tikz.TikzRenderError) as excinfo:
        tikz.render_tikz(r"\draw (0,0) -- (1,1);")

    assert "no svg" in excinfo.value.reason.lower()


@pytest.mark.parametrize(
    "source, expected",
    [
        (r"\write18{cmd}", "write18"),
        (r"\immediate\write18{cmd}", "immediate"),
        (r"\input{/etc/passwd}", "input"),
        (r"\include{chapter}", "include"),
        (r"\openout1=evil.aux", "openout"),
        (r"\read1 to \x", "read"),
        (r"\catcode`\%=12", "catcode"),
        (r"\csname relax\endcsname", "csname"),
        (r"\INPUT{/etc/passwd}", "INPUT"),
        (r"\Input {x}", "Input"),
        (r"\includegraphics{/etc/passwd}", "includegraphics"),
        (r"\XeTeXpdffile \"/etc/passwd\"", "xetexpdffile"),
        (r"\pdfliteral{1 0 0 rg}", "pdfliteral"),
        (r"\InputIfFileExists{/etc/passwd}{}{}", "inputiffileexists"),
        (r"\draw (0,0) -- (1,1); % harmless" + "\n" + r"\write18{x}", "write18"),
    ],
)
def test_dangerous_latex_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    available_toolchain: None,
    source: str,
    expected: str,
) -> None:
    runner = _install_runner(monkeypatch, FakeRunner())

    with pytest.raises(tikz.TikzRenderError) as excinfo:
        tikz.render_tikz(source)

    assert "forbidden LaTeX command" in excinfo.value.reason
    assert expected.lower() in excinfo.value.reason.lower()
    assert runner.calls == []


@pytest.mark.parametrize(
    "source",
    [
        r"% \input{/etc/passwd}" + "\n" + r"\draw (0,0) -- (1,1);",
        r"\draw (0,0) -- (1,1); % \write18{rm -rf /}",
        r"\node at (0,0) {a \\% b};",
        r"\draw (0,0) node {50\%}; % escaped percent, comment survives",
    ],
)
def test_harmless_lookalikes_are_accepted(
    monkeypatch: pytest.MonkeyPatch,
    available_toolchain: None,
    source: str,
) -> None:
    runner = _install_runner(monkeypatch, FakeRunner())

    svg = tikz.render_tikz(source)

    assert svg.startswith("<svg")
    assert len(runner.calls) == 2


def test_oversized_svg_is_rejected(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None
) -> None:
    oversized = b"<svg xmlns=\"http://www.w3.org/2000/svg\">" + (
        b" " * (tikz.MAX_SVG_BYTES + 1)
    ) + b"</svg>"
    _install_runner(monkeypatch, FakeRunner(svg=oversized))

    with pytest.raises(tikz.TikzRenderError) as excinfo:
        tikz.render_tikz(r"\draw (0,0) -- (1,1);")

    assert "svg" in excinfo.value.reason.lower()
    assert "exceed" in excinfo.value.reason.lower()


def test_empty_and_oversized_source_are_rejected(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None
) -> None:
    runner = _install_runner(monkeypatch, FakeRunner())

    with pytest.raises(tikz.TikzRenderError):
        tikz.render_tikz("   \n\t ")

    with pytest.raises(tikz.TikzRenderError):
        tikz.render_tikz("x" * (tikz.MAX_SOURCE_BYTES + 1))

    assert runner.calls == []


def test_non_svg_root_is_rejected(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None
) -> None:
    _install_runner(
        monkeypatch,
        FakeRunner(svg='<?xml version="1.0"?>\n<html><body/></html>\n'),
    )

    with pytest.raises(tikz.TikzRenderError) as excinfo:
        tikz.render_tikz(r"\draw (0,0) -- (1,1);")

    assert "svg" in excinfo.value.reason.lower()


def test_non_positive_timeout_is_a_programming_error(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None
) -> None:
    _install_runner(monkeypatch, FakeRunner())

    with pytest.raises(ValueError):
        tikz.render_tikz(r"\draw (0,0) -- (1,1);", timeout_seconds=0)


def _svg(body: str, *, root: str = '<svg xmlns="http://www.w3.org/2000/svg" '
    'xmlns:xlink="http://www.w3.org/1999/xlink" width="10" height="10">') -> str:
    return root + body + "</svg>"


def test_sanitize_drops_external_doctype_and_processing_instructions() -> None:
    raw = (
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
        '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
        '<?xml-stylesheet href="evil.css"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>\n'
    )

    svg = tikz.sanitize_svg(raw)

    assert "<!DOCTYPE" not in svg
    assert "<?xml" not in svg
    assert "evil.css" not in svg
    assert "<path" in svg


def test_sanitize_removes_executable_style_and_dangerous_values() -> None:
    raw = _svg(
        '<path d="M0 0" style="width: expression(alert(1))"/>'
        '<path d="M0 0" style="background:url(javascript:alert(1))"/>'
        '<path d="M0 0" style="fill: url(#grad)"/>'
        '<path d="M0 0" fill="url(javascript:alert(1))"/>'
        '<path d="M0 0" src="evil.js"/>'
    )

    svg = tikz.sanitize_svg(raw)

    assert "expression(" not in svg
    assert "javascript:" not in svg
    assert "evil.js" not in svg
    assert 'style="fill: url(#grad)"' in svg


def test_sanitize_keeps_only_local_or_raster_hrefs() -> None:
    raw = _svg(
        '<use xlink:href="#glyph0-1"/>'
        '<image xlink:href="data:image/png;base64,iVBORw0KGgo="/>'
        '<image xlink:href="data:text/html;base64,PHNjcmlwdD4="/>'
        '<a xlink:href="https://evil.invalid/"><path d="M0 0"/></a>'
        '<a xlink:href="//evil.invalid/x"><path d="M1 1"/></a>'
    )

    svg = tikz.sanitize_svg(raw)

    assert 'xlink:href="#glyph0-1"' in svg
    assert "data:image/png;base64" in svg
    assert "data:text/html" not in svg
    assert "evil.invalid" not in svg


def test_sanitize_drops_non_svg_namespaces_and_nested_scripts() -> None:
    raw = _svg(
        '<g><switch><script>alert(1)</script></switch>'
        '<foo:bar xmlns:foo="http://evil.invalid/ns"><path d="M0 0"/></foo:bar>'
        '<path d="M1 1"/></g>'
    )

    svg = tikz.sanitize_svg(raw)

    assert "<script" not in svg.lower()
    assert "evil.invalid/ns" not in svg
    assert '<path d="M1 1"' in svg


def test_sanitize_rejects_external_entities_and_amplification() -> None:
    external = (
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///C:/Windows/win.ini">]>\n'
        '<svg xmlns="http://www.w3.org/2000/svg"><text>&xxe;</text></svg>\n'
    )
    with pytest.raises(tikz.TikzRenderError) as excinfo:
        tikz.sanitize_svg(external)
    assert "entity" in excinfo.value.reason.lower()

    many = "".join(
        f'<!ENTITY e{index} "x">' for index in range(tikz._MAX_ENTITY_DECLARATIONS + 1)
    )
    bomb = (
        '<?xml version="1.0"?>\n'
        f"<!DOCTYPE svg [{many}]>\n"
        '<svg xmlns="http://www.w3.org/2000/svg"><text>ok</text></svg>\n'
    )
    with pytest.raises(tikz.TikzRenderError):
        tikz.sanitize_svg(bomb)


def test_sanitize_rejects_non_utf8_bytes() -> None:
    raw = b'<svg xmlns="http://www.w3.org/2000/svg"><text>\xff</text></svg>'

    with pytest.raises(tikz.TikzRenderError) as excinfo:
        tikz.sanitize_svg(raw)

    assert "utf-8" in excinfo.value.reason.lower()


def test_sanitize_escapes_title_and_strips_control_characters() -> None:
    raw = _svg('<path d="M0 0"/>')

    svg = tikz.sanitize_svg(raw, title="a & <b>\x00c")

    assert "<title>a &amp; &lt;b&gt; c</title>" in svg
    assert svg.count("<title>") == 1
    assert svg.index("<title>") < svg.index("<path")


def test_sanitize_replaces_an_existing_title_instead_of_duplicating() -> None:
    raw = _svg("<title>old</title><desc>old desc</desc><path d=\"M0 0\"/>")

    svg = tikz.sanitize_svg(raw, title="new")

    assert "old" not in svg
    assert svg.count("<title>") == 1


def test_sanitize_strips_root_level_event_handlers() -> None:
    raw = (
        '<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)" '
        'onmouseover="steal()" width="10pt" height="10pt">'
        '<path d="M0 0"/></svg>'
    )

    svg = tikz.sanitize_svg(raw)

    assert "onload" not in svg.lower()
    assert "onmouseover" not in svg.lower()
    assert 'width="10pt"' in svg


def test_sanitize_rejects_a_foreign_root_namespace() -> None:
    raw = '<svg xmlns="http://evil.invalid/ns"><path d="M0 0"/></svg>'

    with pytest.raises(tikz.TikzRenderError) as excinfo:
        tikz.sanitize_svg(raw)

    assert "namespace" in excinfo.value.reason.lower()


def test_layout_gate_failure_is_a_warning_not_a_lost_figure(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None
) -> None:
    runner = _install_runner(monkeypatch, FakeRunner())
    from deeptutor.review import layout_gate

    class _Report:
        ok = False
        issues = (
            layout_gate.LayoutIssue(
                code="word_overlap", detail="two labels overlap", page=1
            ),
            layout_gate.LayoutIssue(
                code="font_size_unsupported",
                detail="font size not measured",
                page=0,
                severity="warning",
            ),
        )

    monkeypatch.setattr(layout_gate, "check_layout", lambda *a, **k: _Report())

    rendered = tikz.render_tikz_document(r"\draw (0,0) -- (1,1);")

    assert rendered.svg.startswith("<svg")
    assert "word_overlap" in rendered.layout_error
    assert rendered.layout_warnings == ("font size not measured",)
    assert len(runner.calls) == 2


def test_layout_gate_crash_never_loses_the_figure(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None
) -> None:
    _install_runner(monkeypatch, FakeRunner())
    from deeptutor.review import layout_gate

    def boom(*_args, **_kwargs):
        raise RuntimeError("gate exploded")

    monkeypatch.setattr(layout_gate, "check_layout", boom)

    rendered = tikz.render_tikz_document(r"\draw (0,0) -- (1,1);")

    assert rendered.svg.startswith("<svg")
    assert "layout check failed" in rendered.layout_warnings[0]


def test_layout_check_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None
) -> None:
    _install_runner(monkeypatch, FakeRunner())
    from deeptutor.review import layout_gate

    monkeypatch.setattr(
        layout_gate,
        "check_layout",
        lambda *a, **k: pytest.fail("layout gate must not run"),
    )

    rendered = tikz.render_tikz_document(
        r"\draw (0,0) -- (1,1);", check_layout=False
    )

    assert rendered.svg.startswith("<svg")
    assert rendered.layout_error == ""


# --- RS-805: opt-in render cache -------------------------------------------

class _OkLayoutReport:
    ok = True
    issues: tuple[Any, ...] = ()


def _tool_calls(runner: FakeRunner) -> list[list[str]]:
    """Only xelatex/pdftocairo calls; the layout gate probes pdftotext too."""
    return [
        args
        for args, _kwargs in runner.calls
        if Path(args[0]).stem.lower() in {"xelatex", "pdftocairo"}
    ]


@pytest.fixture()
def cache_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    from deeptutor.review import layout_gate

    monkeypatch.setenv("DEEPTUTOR_REVIEW_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(layout_gate, "check_layout", lambda *a, **k: _OkLayoutReport())
    return tmp_path / "cache"


def test_cache_is_opt_in(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None, tmp_path: Path
) -> None:
    monkeypatch.delenv("DEEPTUTOR_REVIEW_CACHE", raising=False)
    runner = _install_runner(monkeypatch, FakeRunner())

    tikz.render_tikz_document(r"\draw (0,0) -- (1,1);", check_layout=False)
    tikz.render_tikz_document(r"\draw (0,0) -- (1,1);", check_layout=False)

    # Without the env var every call compiles; nothing is written to disk.
    assert len(_tool_calls(runner)) == 4
    assert not (tmp_path / "cache").exists()


def test_second_render_hits_the_cache(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None, cache_dir: Path
) -> None:
    runner = _install_runner(monkeypatch, FakeRunner())

    first = tikz.render_tikz_document(r"\draw (0,0) -- (1,1);", title="缓存")
    second = tikz.render_tikz_document(r"\draw (0,0) -- (1,1);", title="缓存")

    assert len(_tool_calls(runner)) == 2
    assert second.svg == first.svg
    assert second.layout_error == first.layout_error
    assert second.layout_warnings == first.layout_warnings
    assert list(cache_dir.glob("*.json"))


def test_cache_key_includes_title_and_source(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None, cache_dir: Path
) -> None:
    runner = _install_runner(monkeypatch, FakeRunner())

    tikz.render_tikz_document(r"\draw (0,0) -- (1,1);", title="A")
    tikz.render_tikz_document(r"\draw (0,0) -- (1,1);", title="B")
    tikz.render_tikz_document(r"\draw (0,0) -- (2,2);", title="A")

    assert len(_tool_calls(runner)) == 6


def test_corrupt_cache_entry_is_a_miss(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None, cache_dir: Path
) -> None:
    runner = _install_runner(monkeypatch, FakeRunner())
    tikz.render_tikz_document(r"\draw (0,0) -- (1,1);")
    for entry in cache_dir.glob("*.json"):
        entry.write_bytes(b"{not json")
    runner.calls.clear()

    rendered = tikz.render_tikz_document(r"\draw (0,0) -- (1,1);")

    assert rendered.svg.startswith("<svg")
    assert len(_tool_calls(runner)) == 2


def test_cache_write_failure_keeps_the_figure(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None, cache_dir: Path
) -> None:
    _install_runner(monkeypatch, FakeRunner())

    def broken_put(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("deeptutor.review.cache.DiskCache.put", broken_put)

    rendered = tikz.render_tikz_document(r"\draw (0,0) -- (1,1);")

    assert rendered.svg.startswith("<svg")
    assert rendered.layout_error == ""


def test_cache_key_separates_gate_on_from_gate_off(
    monkeypatch: pytest.MonkeyPatch, available_toolchain: None, cache_dir: Path
) -> None:
    runner = _install_runner(monkeypatch, FakeRunner())

    tikz.render_tikz_document(r"\draw (0,0) -- (1,1);", check_layout=False)
    tikz.render_tikz_document(r"\draw (0,0) -- (1,1);", check_layout=True)

    # A cached "no gate" render must not answer a request that asked for it.
    assert len(_tool_calls(runner)) == 4


# --- RS-505: font-size gate wiring -----------------------------------------


def _toolchain_with(*, pdftohtml: bool) -> Toolchain:
    """Toolchain for the font-gate tests: pdf engines always, html optional."""
    capabilities = [
        ToolCapability(
            name=XELATEX,
            executable=str(Path("C:/tex/xelatex.exe")),
            available=True,
            version="XeTeX 3.14",
        ),
        ToolCapability(
            name=PDFTOCAIRO,
            executable=str(Path("C:/poppler/pdftocairo.exe")),
            available=True,
            version="pdftocairo version 22.02.0",
        ),
    ]
    capabilities.append(
        ToolCapability(
            name=PDFTOHTML,
            executable=str(Path("C:/poppler/pdftohtml.exe")) if pdftohtml else None,
            available=pdftohtml,
            version="pdftohtml version 22.02.0" if pdftohtml else "",
            reason="" if pdftohtml else "not found on PATH",
        )
    )
    return Toolchain(capabilities=tuple(capabilities))


class _FontReport:
    def __init__(self, issues: tuple[Any, ...] = (), min_font_size: float = 6.0):
        self.ok = not any(issue.severity == "error" for issue in issues)
        self.issues = issues
        self.pages = 1
        self.fonts_seen = 2
        self.smallest_size = 4
        self.min_font_size = min_font_size


def _font_issue(
    code: str,
    detail: str,
    *,
    page: int = 0,
    size: float | None = None,
    severity: str = "error",
) -> Any:
    from deeptutor.review.font_gate import FontSizeIssue

    return FontSizeIssue(
        code=code, detail=detail, page=page, size=size, severity=severity
    )


def test_font_gate_reports_a_too_small_font_as_layout_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        tikz, "probe_toolchain", lambda: _toolchain_with(pdftohtml=True)
    )
    from deeptutor.review import font_gate

    monkeypatch.setattr(
        font_gate,
        "check_font_sizes",
        lambda *a, **k: _FontReport(
            (_font_issue("font_size_below_minimum", "4pt", page=1, size=4),)
        ),
    )

    error, warnings = tikz._gate_layout(tmp_path / "figure.pdf")

    assert error == "font size below 6pt: 4pt (page 1)"
    assert warnings == []


def test_font_gate_mentions_the_extra_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        tikz, "probe_toolchain", lambda: _toolchain_with(pdftohtml=True)
    )
    from deeptutor.review import font_gate

    monkeypatch.setattr(
        font_gate,
        "check_font_sizes",
        lambda *a, **k: _FontReport(
            (
                _font_issue("font_size_below_minimum", "4pt", page=1, size=4),
                _font_issue("font_size_below_minimum", "5pt", page=2, size=5),
            )
        ),
    )

    error, _warnings = tikz._gate_layout(tmp_path / "figure.pdf")

    assert error == "font size below 6pt: 4pt (page 1) and 1 more"


def test_font_gate_is_silent_without_pdftohtml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No pdftohtml entry at all is the common case: silently skip, no call."""
    monkeypatch.setattr(tikz, "probe_toolchain", _available_toolchain)
    from deeptutor.review import font_gate

    monkeypatch.setattr(
        font_gate,
        "check_font_sizes",
        lambda *a, **k: pytest.fail("font gate must not run without pdftohtml"),
    )

    error, warnings = tikz._gate_layout(tmp_path / "figure.pdf")

    assert error == ""
    assert warnings == []


def test_font_gate_marks_unavailable_toolchain_as_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        tikz, "probe_toolchain", lambda: _toolchain_with(pdftohtml=False)
    )
    from deeptutor.review import font_gate

    monkeypatch.setattr(
        font_gate,
        "check_font_sizes",
        lambda *a, **k: pytest.fail("font gate must not run when unavailable"),
    )

    error, warnings = tikz._gate_layout(tmp_path / "figure.pdf")

    assert error == ""
    assert warnings == []


def test_font_gate_tool_failure_is_only_a_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        tikz, "probe_toolchain", lambda: _toolchain_with(pdftohtml=True)
    )
    from deeptutor.review import font_gate

    monkeypatch.setattr(
        font_gate,
        "check_font_sizes",
        lambda *a, **k: _FontReport(
            (_font_issue("pdftohtml_timeout", "pdftohtml exceeded 10s"),)
        ),
    )

    error, warnings = tikz._gate_layout(tmp_path / "figure.pdf")

    assert error == ""
    assert warnings == ["font check failed: pdftohtml_timeout"]


def test_font_gate_crash_never_loses_the_figure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        tikz, "probe_toolchain", lambda: _toolchain_with(pdftohtml=True)
    )
    from deeptutor.review import font_gate

    def boom(*_args, **_kwargs):
        raise RuntimeError("font gate exploded")

    monkeypatch.setattr(font_gate, "check_font_sizes", boom)

    error, warnings = tikz._gate_layout(tmp_path / "figure.pdf")

    assert error == ""
    assert "font check failed" in warnings[0]


def test_layout_error_outranks_font_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        tikz, "probe_toolchain", lambda: _toolchain_with(pdftohtml=True)
    )
    from deeptutor.review import font_gate, layout_gate

    class _LayoutReport:
        ok = False
        issues = (
            layout_gate.LayoutIssue(
                code="word_overlap", detail="two labels overlap", page=1
            ),
        )

    monkeypatch.setattr(layout_gate, "check_layout", lambda *a, **k: _LayoutReport())
    monkeypatch.setattr(
        font_gate,
        "check_font_sizes",
        lambda *a, **k: _FontReport(
            (_font_issue("font_size_below_minimum", "4pt", page=1, size=4),)
        ),
    )

    error, _warnings = tikz._gate_layout(tmp_path / "figure.pdf")

    assert "word_overlap" in error
    assert "font size" not in error


def test_render_with_pdftohtml_present_runs_the_font_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A measurable font defect must reach the exported ``layout_error``."""
    monkeypatch.setattr(
        tikz, "probe_toolchain", lambda: _toolchain_with(pdftohtml=True)
    )
    _install_runner(monkeypatch, FakeRunner())
    from deeptutor.review import font_gate, layout_gate

    monkeypatch.setattr(layout_gate, "check_layout", lambda *a, **k: _OkLayoutReport())
    monkeypatch.setattr(
        font_gate,
        "check_font_sizes",
        lambda *a, **k: _FontReport(
            (_font_issue("font_size_below_minimum", "4pt", page=2, size=4),)
        ),
    )

    rendered = tikz.render_tikz_document(r"\draw (0,0) -- (1,1);")

    assert rendered.svg.startswith("<svg")
    assert rendered.layout_error == "font size below 6pt: 4pt (page 2)"


# -- Phase 36.1: command allowlist ----------------------------------------

_REALISTIC_SOURCES = (
    r"""
\draw[->,thick] (0,0) -- (3,0) node[right] {$U$};
\draw (0,0) to[R=$R_1$, V=$10\,\mathrm{V}$] (3,0);
\node at (1.5,1) {$\frac{U}{R}=I$};
""",
    r"""
\pgfmathsetmacro{\r}{1.5}
\foreach \a in {0,90,180,270} {
  \draw (\a:\r) circle (0.2);
}
\coordinate (O) at (0,0);
\fill[gray!20] (O) circle (0.5);
""",
    r"""
\newcommand{\lab}[1]{\textsf{#1}}
\begin{scope}[rotate=30]
\draw (0,0) rectangle (2,1) node[midway] {\lab{面积}};
\end{scope}
""",
    r"""
\tdplotsetmaincoords{60}{110}
\begin{tikzpicture}[tdplot_main_coords]
\draw (0,0,0) -- (1,0,0);
\end{tikzpicture}
""",
)


def test_command_allowlist_accepts_realistic_drawings() -> None:
    for source in _REALISTIC_SOURCES:
        assert tikz._prepare_source(source) == source.strip()


def test_command_allowlist_rejects_unknown_commands_on_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even with the denylist disabled, an unknown primitive must not pass:
    # this is what closes "a file reader nobody has discovered yet".
    import re

    monkeypatch.setattr(tikz, "_FORBIDDEN_PATTERN", re.compile(r"$^"))
    for name in ("input", "XeTeXpicfile", "IfFileExists", "openin"):
        with pytest.raises(tikz.TikzRenderError) as excinfo:
            tikz._prepare_source(rf"\draw (0,0) -- (1,1); \{name}{{x}}")
        assert f"\\{name}" in excinfo.value.reason


def test_defined_macros_are_usable_but_unknown_names_are_not() -> None:
    source = r"\newcommand{\foo}{x}\draw (0,0) node {\foo};"
    assert tikz._prepare_source(source) == source

    with pytest.raises(tikz.TikzRenderError) as excinfo:
        tikz._prepare_source(r"\draw (0,0) node {\zzz};")
    assert "\\zzz" in excinfo.value.reason


def test_foreach_loop_variables_are_usable_but_undeclared_names_are_not() -> None:
    source = (
        "\\foreach \\point/\\radius in {(0,0)/1,(2,0)/0.5} {\n"
        "\\fill (\\point) circle (\\radius);\n"
        "}\n"
    )
    assert tikz._prepare_source(source) == source.strip()

    with pytest.raises(tikz.TikzRenderError) as excinfo:
        tikz._prepare_source(r"\foreach \point in {1} \zzz{2}")
    assert "\\zzz" in excinfo.value.reason


def test_allow_unknown_escape_hatch_skips_the_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = r"\draw (0,0) -- (1,1); \customcommand{x}"
    monkeypatch.setenv(tikz._ALLOW_UNKNOWN_ENV, "1")
    assert tikz._prepare_source(source) == source
    # The hatch widens the allowlist, not the denylist: a known file reader
    # must still be rejected for a trusted corpus.
    with pytest.raises(tikz.TikzRenderError):
        tikz._prepare_source(r"\draw (0,0) -- (1,1); \input{/etc/passwd}")


def test_preamble_declarations_are_only_allowed_on_their_own_line() -> None:
    # Whole-line declarations are hoisted into the preamble and validated
    # there; an inline one would stay in the body and load code mid-document.
    hoisted = "\\usepackage{xcolor}\n\\draw (0,0) -- (1,1);"
    assert tikz._prepare_source(hoisted) == hoisted
    for name in ("usepackage", "usetikzlibrary"):
        with pytest.raises(tikz.TikzRenderError) as excinfo:
            tikz._prepare_source(rf"\draw (0,0) -- (1,1); \{name}{{xcolor}}")
        assert f"\\{name}" in excinfo.value.reason


def test_filecontents_environment_is_rejected_even_with_the_escape_hatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("filecontents", "filecontents*"):
        source = f"\\begin{{{name}}}{{x}}\\end{{{name}}}"
        with pytest.raises(tikz.TikzRenderError) as excinfo:
            tikz._prepare_source(source)
        assert "filecontents" in excinfo.value.reason
    monkeypatch.setenv(tikz._ALLOW_UNKNOWN_ENV, "1")
    with pytest.raises(tikz.TikzRenderError):
        tikz._prepare_source("\\begin{filecontents}{x}\\end{filecontents}")
