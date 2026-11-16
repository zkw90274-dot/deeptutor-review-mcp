"""MCP server for the ten Review Studio agent tools.

Phase 29 exposed the review tools through the core tool protocol
(:mod:`deeptutor.review.agent_tools`), which only a chat mounted inside this
process can see. This module puts the *same* tools behind MCP, so an external
agent can mount them too, without re-implementing a single branch of their
behaviour.

Two transports, one server:

* ``python -m deeptutor.review.mcp_server`` speaks **stdio** (the default).
  That is the zero-port mode an admin-level MCP entry uses.
* ``python -m deeptutor.review.mcp_server --transport streamable-http`` serves
  the Streamable HTTP endpoint at ``/mcp`` on ``127.0.0.1:8767``. User-level
  MCP entries only accept remote URLs and reject loopback (see
  ``deeptutor.services.mcp.network``), so this is for a deployment that
  terminates the public connection itself.

The protocol layer is the SDK's low-level :class:`mcp.server.lowlevel.Server`
rather than ``FastMCP``. ``FastMCP`` derives an input schema from a Python
signature, which would be a second, drift-prone copy of the parameter list:
the ten tools already publish theirs through ``get_definition()``. Registering
those definitions directly keeps name, description and JSON Schema generated
from the one source the tests compare against.

Tool semantics are unchanged. ``ToolResult.content`` is returned verbatim as
the text block, and ``success=False`` maps to the MCP ``isError`` flag. A soft
failure (unknown spec or item) stays a *successful* call carrying ``error`` and
``instruction`` — the model recovers by calling ``review_specs`` /
``review_today``.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import logging
import os
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server

from deeptutor.core.tool_protocol import BaseTool, ToolDefinition
from deeptutor.review.agent_tools import (
    ReviewAttemptTool,
    ReviewBookFiguresTool,
    ReviewExportLinkTool,
    ReviewExportTreeTool,
    ReviewGenerateTool,
    ReviewHintTool,
    ReviewPlanTool,
    ReviewSpecsTool,
    ReviewTodayTool,
    ReviewWeaknessTool,
)

logger = logging.getLogger(__name__)

SERVER_NAME = "deeptutor-review"

DEFAULT_TRANSPORT = "stdio"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8767
DEFAULT_PATH = "/mcp"

#: Optional bearer token for the HTTP transport. Unset means the historical
#: no-auth behavior; a shared deployment should set this (or terminate auth in
#: its own reverse proxy). Read from the environment rather than a CLI flag so
#: the secret does not appear in the process list.
AUTH_TOKEN_ENV = "DEEPTUTOR_REVIEW_MCP_TOKEN"

TRANSPORTS = ("stdio", "streamable-http")

#: One entry per tool, in the order the agent tool surface advertises.
#: The classes are stateless (each ``execute`` resolves its own service from
#: ``DEEPTUTOR_REVIEW_HOME``), so a single instance per name is enough.
TOOL_CLASSES: dict[str, type[BaseTool]] = {
    "review_specs": ReviewSpecsTool,
    "review_today": ReviewTodayTool,
    "review_attempt": ReviewAttemptTool,
    "review_weakness": ReviewWeaknessTool,
    "review_export_link": ReviewExportLinkTool,
    "review_export_tree": ReviewExportTreeTool,
    "review_book_figures": ReviewBookFiguresTool,
    "review_generate": ReviewGenerateTool,
    "review_hint": ReviewHintTool,
    "review_plan": ReviewPlanTool,
}


def _package_version() -> str:
    """The installed distribution version, or ``unknown`` in a bare checkout."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("deeptutor-review")
    except PackageNotFoundError:  # pragma: no cover - wheel-less checkout
        return "unknown"


def _input_schema(definition: ToolDefinition) -> dict[str, Any]:
    """The tool's JSON Schema, from the same builder the chat uses.

    ``to_openai_schema`` already encodes ``ToolParameter`` (type, required,
    enum, bounds, nullable, array items). Re-encoding it here would drift.
    """
    return definition.to_openai_schema()["function"]["parameters"]


#: Deployment context an MCP client shows to its model at initialize time.
#: Deliberately generic: a published wheel must not carry one learner's
#: directory names or pack counts. A site that wants local context points
#: DEEPTUTOR_REVIEW_MCP_CONTEXT at a text file, or drops mcp-context.md
#: into the review home; that note is appended at build time and stays out of
#: git when it lives under the ignored data directory.
INSTRUCTIONS = """Review Studio - spaced review for DeepTutor

The ten tools read and write one local review store. Its root is resolved at
call time: DEEPTUTOR_REVIEW_HOME when set, otherwise <workspace>/review. The
web UI, the offline export and these tools all share that store.

Typical session: review_specs -> review_today -> review_hint (only when the
learner is stuck) -> review_attempt -> next item. review_generate creates a new
pack from one knowledge-base chapter; review_plan manages multi-week plans;
review_export_link produces a single-file offline HTML page.

Non-negotiable: never grade for the learner (transcribe their own again /
hard / good only); never reveal an answer or an option before review_attempt;
never improvise a hint - hints come only from review_hint. A freshly generated
pack has no due cards until its first attempt.

review_generate chapter must match a chapter name the knowledge base derives
from its file layout (第N章 / 第N讲 / chN / 实验N_xxx); anything else returns
"chapter ... not found".

Two exports, two jobs. review_export_link hands over one pack as a single
self-contained HTML page (a link the host downloads). review_export_tree writes
the whole collection to disk and returns the absolute path: one folder per
subject, one HTML page per pack, plus index.html. That tree is the shared
layout, not a per-call choice - <out_dir>/<subject>/<chapter>.html and
<out_dir>/index.html, where subject and chapter are split from the pack title
("<subject> · <chapter> 复习包") and made filesystem-safe. Every call rebuilds
the tree from the store, so a renamed chapter leaves no stale page and no dated
snapshot folders ever accumulate; packs failing the quality gates are skipped
and reported, never silently written. Send the learner to index.html, and do
not paste pack contents into the chat. These pages are the spaced-repetition
packs for a chapter, not the living book's in-section quizzes or flashcards.

Plans name their reading tasks after the material, not after the clock: one
task per source document or section, labelled "<chapter> · <document/section>".
Never split a chapter into equal-time slices such as "part i/n". A document
that spans several days stays one task and continues with a numbered suffix
(一)(二)(三), not (上)/(下). A day note lists only that day's documents, groups
them under the chapter once, marks the last one when the chapter is finished,
and names the practice pack for every chapter touched that day. Say what to do
today, never just how big the pack is: on the closing day "复习包：<chapter> ·
今天做 N 题（一次做完）", while the chapter is still being read "复习包：<chapter>
· 今天不做，学完本章后一次做完 N 题", "暂无复习包" when no pack exists. A pack
holds a handful of items and fits one sitting (the queue's hard cap is 12), so
only an oversized pack is split, as "今天做 12 题（共 N 题，余下按到期继续）"; a
bare "共 N 题" must never stand alone, because it reads as "do all of them
today" while the chapter is still in progress. The chapter's 完成 milestone
carries the same pointer in its detail. The prefix is "复习包：" and never "练习：": the pack
is the spaced-repetition set generated from that chapter, while the interactive
book's in-section quizzes and flashcards are self-check only and are not
scheduled - do not present them as the same thing. A deployment may also map each note onto a book section (book-map.json for chapter names, book-sections.json for section numbers, both keyed by the same note labels). When it does, the chapter heading is the book's chapter name and the rows are section numbers with titles, one row per section however many notes fold into it; a note that maps to nothing keeps its own label and that chapter falls back to note names. Never add a separate pointer line naming the book chapter: the heading already says it.

One main study document per subject, and it is the most complete version. When a
subject has accumulated several generations of the same material, keep the
fuller one, merge any block that exists only in the older copy into it, move the
rest out of the knowledge base (archive it, do not leave it next to the keeper),
and reindex; a base that still holds two generations will produce packs and
plans from both. Review packs are then regenerated from that one source.

That document is an interactive living book: a single self-contained HTML file
that works offline, with math typeset inline, no external script and no <img>
(figures become inline SVG or a labelled placeholder). Its parts are a sticky
top bar of chapter chips, a left table of contents with search, a per-section
"mark as mastered" control plus a progress ring, and per-chapter auto-graded
questions, flashcards and adjustable widgets - sliders that recompute a formula
instead of a static screenshot. It supports a light and a dark theme and ships
print CSS that drops the chrome for A4.

Performance is a property of the book, not a later polish. A chapter-dense book
runs to hundreds of formulas and ~10^5 DOM nodes; typesetting all of it at load
blocks the first paint for seconds, and any interaction then re-lays out the
whole document. Build it lazily:

- MathJax config takes startup: { typeset: false }. Typeset a section only when
  it approaches the viewport (an IntersectionObserver with roughly 900px of
  rootMargin, plus the first two or three sections immediately), and keep a
  record of the sections already done. Never typeset document.body at boot.
- Typeset a block that just became visible (a revealed answer, a flipped card)
  on demand, and skip blocks with no markup at all - that skip is what removes
  the 150-200ms stall a click otherwise causes.
- Off-screen sections get content-visibility: auto with a contain-intrinsic-size
  near their real height, and @media print restores content-visibility: visible
  and contain-intrinsic-size: none so no section is clipped on paper.
- Heights are then estimates, so in-page anchor jumps must be instant
  (scrollIntoView behavior "instant") with one re-align about 160ms later; HTML
  smooth scrolling under-shoots long jumps by more than a thousand pixels. Cache
  the DOM queries that scan the whole document (every .quiz, every option)
  instead of re-running them per click.
- A print control and the p shortcut typeset every section first and only then
  call window.print(); printing an untypeset chapter emits raw $...$ text.

Acceptance: first paint under a second, and choosing an option never produces a
long task over 100ms.

Use the Review Studio visual language, so every book and pack looks like one
product: light canvas #eceef1, card #fbfbfc, inner #f5f6f7, hairline #e5e7eb,
accent #0f766e; dark #0b0d0f / #16191c / #1b1e22 / #262a30 / #5ec8bd; corner
radii 22/14/9px; a 60px sticky header with a translucent background and blur;
pill buttons and chips; a hero title with tight negative letter-spacing.

Book diagrams are inline SVG drawn inside the section that needs one, so they are
invisible to a plan. A local index (book-figures.json, written by
scripts/build-book-figures.py from the built books) records which section owns
which figure, and review_book_figures reports it. When you send the learner to a
section that carries a diagram, name the figure the way the book labels it
(图 2-1) and say in one clause what it shows; a section absent from the index has
none, so never promise one, and never paste SVG or base64 into the chat.

Never hand-edit the built HTML. Each book keeps its build script plus content
fragments (or markdown) and data files for questions, flashcards and widget
bindings, so the page is reproducible; rebuild after changing content, and keep
plan task details pointing at that book's section anchors (ids of the form
s<chapter>-<n>) rather than at raw markdown.
"""

#: Optional per-deployment context appended to INSTRUCTIONS. A public wheel
#: stays generic; an operator can add local material (directory layout, course
#: list, house rules) without editing the package.
CONTEXT_ENV = "DEEPTUTOR_REVIEW_MCP_CONTEXT"
CONTEXT_FILENAME = "mcp-context.md"


def _extra_context() -> str:
    """Read the optional deployment note, if one exists.

    Order: the explicit env path, then <review_home>/mcp-context.md. The
    review home is only resolvable with a core installation, so a failure to
    resolve it means "no note" - never a startup error.
    """
    candidates: list[Path] = []
    explicit = os.environ.get(CONTEXT_ENV)
    if explicit:
        candidates.append(Path(explicit).expanduser())
    else:
        try:
            from deeptutor.review.storage import review_home

            candidates.append(review_home() / CONTEXT_FILENAME)
        except Exception:  # pragma: no cover - core missing / no request ctx
            logger.debug("review MCP: no review home for %s", CONTEXT_FILENAME)
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            return text
    return ""


def build_instructions() -> str:
    """Base instructions plus the optional deployment note, if any."""
    base = INSTRUCTIONS.strip()
    extra = _extra_context()
    return f"{base}\n\n---\n\n{extra}" if extra else base


def build_server() -> Server[Any, Any]:
    """Build the MCP server with the ten review tools registered.

    A fresh mapping of tool instances is created per call so tests can build
    isolated servers; the tools themselves hold no state.
    """
    tools = {name: tool_type() for name, tool_type in TOOL_CLASSES.items()}
    server: Server[Any, Any] = Server(
        SERVER_NAME, version=_package_version(), instructions=build_instructions()
    )

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=definition.name,
                description=definition.description,
                inputSchema=_input_schema(definition),
            )
            for definition in (tool.get_definition() for tool in tools.values())
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        tool = tools.get(name)
        if tool is None:
            # The SDK turns a raised exception into ``isError=True`` plus this
            # message; the MCP convention for a name the server does not serve.
            raise ValueError(f"Unknown Review Studio tool: {name}")

        result = await tool.execute(**arguments)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result.content)],
            isError=not result.success,
        )

    return server


def create_http_app(
    server: Server[Any, Any], *, path: str = DEFAULT_PATH,
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
    allowed_hosts: list[str] | None = None,
    allowed_origins: list[str] | None = None,
    auth_token: str | None = None,
) -> Any:
    """Wrap ``server`` in a Starlette app serving Streamable HTTP at ``path``.

    The session manager owns the per-client sessions and must be started in the
    app lifespan; this mirrors the wiring the SDK's own FastMCP transport uses.

    ``auth_token`` enables a bearer-token check when non-empty. It defaults to
    ``DEEPTUTOR_REVIEW_MCP_TOKEN``; leaving both unset preserves the previous
    no-auth behavior for local stdio/HTTP use.
    """
    from contextlib import asynccontextmanager

    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.types import Receive, Scope, Send

    token = (
        auth_token if auth_token is not None else os.environ.get(AUTH_TOKEN_ENV, "")
    ).strip()

    def _authorized(scope: Scope) -> bool:
        headers = dict(scope.get("headers") or [])
        raw = headers.get(b"authorization", b"").decode("latin-1")
        scheme, _, presented = raw.partition(" ")
        return scheme.lower() == "bearer" and bool(token) and hmac.compare_digest(
            presented.strip(), token
        )

    names = {"127.0.0.1", "localhost", "[::1]"}
    if host not in {"0.0.0.0", "::", "[::]"}:
        names.add(f"[{host}]" if ":" in host and not host.startswith("[") else host)
    authorities = sorted(f"{name}:{port}" for name in names)
    if port == 80:
        authorities.extend(sorted(names))
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[*authorities, *(allowed_hosts or [])],
        allowed_origins=[*(f"http://{authority}" for authority in authorities), *(allowed_origins or [])],
    )
    manager = StreamableHTTPSessionManager(app=server, security_settings=security)

    class _Endpoint:
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if token and not _authorized(scope):
                response = JSONResponse(
                    {"error": "unauthorized"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return
            await manager.handle_request(scope, receive, send)

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        async with manager.run():
            yield

    return Starlette(routes=[Route(path, endpoint=_Endpoint())], lifespan=lifespan)


async def _serve_stdio(server: Server[Any, Any]) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m deeptutor.review.mcp_server",
        description="Serve the Review Studio agent tools over MCP.",
    )
    parser.add_argument(
        "--transport",
        choices=TRANSPORTS,
        default=DEFAULT_TRANSPORT,
        help="stdio (default) or streamable-http.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind address for streamable-http (default {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Bind port for streamable-http (default {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
        help=f"HTTP path for streamable-http (default {DEFAULT_PATH}).",
    )
    parser.add_argument("--allowed-host", action="append", default=[], help="Additional trusted Host authority; repeat for multiple hosts.")
    parser.add_argument("--allowed-origin", action="append", default=[], help="Additional trusted browser Origin; repeat for multiple origins.")
    args = parser.parse_args(argv)
    if not args.path.startswith("/"):
        parser.error("--path must start with '/'")
    return args


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: stdio by default, Streamable HTTP when asked."""
    args = _parse_args(argv)
    server = build_server()

    if args.transport == "stdio":
        asyncio.run(_serve_stdio(server))
        return 0

    import uvicorn

    logger.info(
        "serving %s over streamable-http at http://%s:%s%s",
        SERVER_NAME,
        args.host,
        args.port,
        args.path,
    )
    uvicorn.run(
        create_http_app(
            server, path=args.path, host=args.host, port=args.port,
            allowed_hosts=args.allowed_host, allowed_origins=args.allowed_origin,
        ),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
