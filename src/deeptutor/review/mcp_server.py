"""MCP server for the Review Studio agent tools.

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
the tools already publish theirs through ``get_definition()``. Registering
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
import sys
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server

from deeptutor.core.tool_protocol import BaseTool, ToolDefinition
from deeptutor.review import capabilities
from deeptutor.review.agent_tools import (
    ReviewAttemptTool,
    ReviewBookFiguresTool,
    ReviewDueTool,
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
    "review_due": ReviewDueTool,
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

#: MCP ``ToolAnnotations`` per tool, so a client can auto-approve reads and
#: warn before a write instead of asking the same question for both. They are
#: *hints*, not enforcement: the authoritative switch stays
#: :mod:`deeptutor.review.capabilities` (``--read-only`` and friends).
#:
#: Entries that deliberately contradict the tool's name:
#:
#: * ``review_today`` reads like a query but in normal mode reuses/creates a
#:   session and records views, so it is not read-only.
#: * ``review_attempt`` is only idempotent when the caller passes
#:   ``submission_id``; without it every call mints a new event. A client that
#:   auto-retries on timeout would double-record the learner's history, so the
#:   hint stays ``False`` (the tool description still documents the
#:   ``submission_id`` retry contract).
#: * ``review_export_tree`` writes files and deletes the stale ``.html`` of the
#:   folders it touches, so it is destructive. Its output is byte-stable for
#:   the same store state (the page timestamp lives only in the opt-in
#:   snapshot), so ``idempotentHint`` stays true.
#: * ``review_generate`` is destructive: ``replace=True`` overwrites the pack
#:   with the same deterministic id and passes ``allow_history_rewrite=True``,
#:   so old events end up pointing at replaced content.
#:
#: ``openWorldHint`` means "may talk to something outside this deployment":
#: ``review_generate`` calls the configured model, and ``review_plan`` mirrors
#: onto the calendar / task board / Co-Writer.
TOOL_ANNOTATIONS: dict[str, dict[str, bool]] = {
    "review_specs": {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    "review_due": {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    "review_today": {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
    "review_attempt": {
        "readOnlyHint": False,
        "destructiveHint": False,
        # Conditional idempotency is not idempotency: without submission_id
        # every call appends a new event and moves the schedule.
        "idempotentHint": False,
        "openWorldHint": False,
    },
    "review_weakness": {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    "review_export_link": {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    "review_export_tree": {
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    "review_book_figures": {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    "review_generate": {
        "readOnlyHint": False,
        # replace=True overwrites the pack with the same deterministic id and
        # rewrites history protection, so this is an overwrite, not append-only.
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    },
    "review_hint": {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    "review_plan": {
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    },
}


def _annotations(name: str) -> types.ToolAnnotations:
    """The annotation block for one tool; a missing entry is a build error.

    Raising here (rather than silently omitting annotations) makes "added a
    tool, forgot its hints" fail at server construction instead of shipping a
    tool every client treats as unknown-risk.
    """
    try:
        hints = TOOL_ANNOTATIONS[name]
    except KeyError:  # pragma: no cover - tests assert the key sets match
        raise RuntimeError(
            f"no MCP annotations registered for tool {name!r}; add it to TOOL_ANNOTATIONS"
        ) from None
    return types.ToolAnnotations(**hints)


def _package_version() -> str:
    """The installed distribution version, or ``unknown`` in a bare checkout."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("deeptutor-review")
    except PackageNotFoundError:  # pragma: no cover - wheel-less checkout
        return "unknown"


def _check() -> int:
    """Report whether this install can serve the review module.

    This is deliberately a diagnostic, not a repair: the common failure is a
    core/wheel version mismatch, and the safe response is to stop before the
    server starts rather than let an agent call tools that will 404.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        core = version("deeptutor")
    except PackageNotFoundError:  # pragma: no cover - import would already fail
        core = "<not installed>"
    wheel = _package_version()

    gate = False
    try:
        from deeptutor.modules.catalog import module_enabled

        gate = bool(module_enabled("review"))
        gate_detail = "ok" if gate else "version mismatch"
    except Exception as exc:  # manifest missing or catalog unavailable
        gate_detail = f"unavailable ({exc})"

    try:
        from deeptutor.review.storage import review_home

        home = str(review_home())
    except Exception as exc:
        home = f"<unresolved: {exc}>"

    print(f"core deeptutor    : {core}")
    print(f"deeptutor-review  : {wheel}")
    print(f"module gate       : {gate_detail}")
    print(f"review home       : {home}")
    print(f"mcp tools         : {len(TOOL_CLASSES)}")
    caps = capabilities.current()
    print(
        "capabilities      : "
        f"read_only={caps.read_only} attempts={caps.attempts} plan={caps.plan} "
        f"generate={caps.generate} export_write={caps.export_write}"
    )

    if core == wheel and gate:
        print("ready: versions match and the review module is enabled")
        return 0

    print(
        "not ready: deeptutor and deeptutor-review must be the same version, "
        "and the module must be enabled. Do not downgrade the core; install "
        "the wheel built for this core version instead."
    )
    return 1


def _input_schema(definition: ToolDefinition) -> dict[str, Any]:
    """The tool's JSON Schema, from the same builder the chat uses.

    ``to_openai_schema`` already encodes ``ToolParameter`` (type, required,
    enum, bounds, nullable, array items). Re-encoding it here would drift.
    """
    return {
        **definition.to_openai_schema()["function"]["parameters"],
        "additionalProperties": False,
    }


#: Deployment context an MCP client shows to its model at initialize time.
#: Deliberately generic: a published wheel must not carry one learner's
#: directory names or pack counts. A site that wants local context points
#: DEEPTUTOR_REVIEW_MCP_CONTEXT at a text file, or drops mcp-context.md
#: into the review home; that note is appended at build time and stays out of
#: git when it lives under the ignored data directory.
INSTRUCTIONS = """Review Studio - spaced review for DeepTutor

All tools share one local store: DEEPTUTOR_REVIEW_HOME when set, otherwise
<workspace>/review; the web UI and offline export read the same store.

Flow: review_specs (what exists) -> review_due (what has work today, across
packs) -> review_today (start one pack) -> review_hint (only when stuck) ->
review_attempt -> next item. review_generate builds a pack from one
knowledge-base chapter; review_plan schedules multi-week study;
review_export_link returns one self-contained HTML page; review_export_tree
writes the whole collection.

Prefer review_due over review_today for several packs: review_today opens or
reuses a session and records what was shown; review_due records nothing.

Non-negotiable: never grade for the learner (transcribe their own again /
hard / good only); never reveal an answer or option before review_attempt;
hints come only from review_hint; a freshly generated pack has no due cards
until its first attempt. Do not paste pack contents into the chat.

review_generate's chapter must match a name derived from the knowledge-base
file layout (第N章 / 第N讲 / chN / 实验N_xxx), else "chapter ... not found".

The living-book, plan-naming, visual and figure rules are in the review skill
(docs/skill/SKILL.md in the deeptutor-review-mcp repository). Never hand-edit
a built book's HTML; rebuild it from its source.

Operators may append deployment notes through mcp-context.md."""

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
    """Build the MCP server with all review tools registered.

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
                annotations=_annotations(definition.name),
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


def _http_token(configured: str | None = None) -> str | None:
    value = os.environ.get(AUTH_TOKEN_ENV) if configured is None else configured
    if value is None:
        return None
    token = value.strip()
    if not token:
        raise ValueError(f"{AUTH_TOKEN_ENV} is set but blank. Set a token or unset it for loopback HTTP.")
    return token


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

    token = _http_token(auth_token)
    if token is None and not _is_loopback(host):
        raise ValueError(f"refusing to serve on {host} without {AUTH_TOKEN_ENV}")

    def _authorized(scope: Scope) -> bool:
        headers = dict(scope.get("headers") or [])
        raw = headers.get(b"authorization", b"").decode("latin-1")
        scheme, _, presented = raw.partition(" ")
        return scheme.lower() == "bearer" and token is not None and hmac.compare_digest(
            presented.strip().encode("latin-1"), token.encode("utf-8")
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


def _is_loopback(host: str) -> bool:
    """True when binding ``host`` is reachable only from this machine."""
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _configure_capabilities(args: argparse.Namespace) -> None:
    """Translate CLI flags into the env vars the tool classes read."""
    if args.read_only:
        os.environ[capabilities.READ_ONLY] = "1"
    if args.no_attempts:
        os.environ[capabilities.NO_ATTEMPTS] = "1"
    if args.no_plan:
        os.environ[capabilities.NO_PLAN] = "1"
    if args.no_generate or (
        args.transport == "streamable-http" and not args.allow_generate
    ):
        os.environ[capabilities.NO_GENERATE] = "1"
    if args.no_export_write:
        os.environ[capabilities.NO_EXPORT_WRITE] = "1"
    if args.allow_out_dir:
        os.environ[capabilities.ALLOW_OUT_DIRS] = os.pathsep.join(args.allow_out_dir)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m deeptutor.review.mcp_server",
        description="Serve the Review Studio agent tools over MCP.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print version, module gate, review home and tool count; exit non-zero when not ready.",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Disable attempts, plans, generation and export writes; today does not create a session or record views.",
    )
    parser.add_argument("--no-attempts", action="store_true", help="Disable review_attempt.")
    parser.add_argument("--no-plan", action="store_true", help="Disable review_plan.")
    parser.add_argument("--no-generate", action="store_true", help="Disable review_generate.")
    parser.add_argument(
        "--no-export-write",
        action="store_true",
        help="Disable review_export_tree; review_export_link stays available.",
    )
    parser.add_argument(
        "--allow-generate",
        action="store_true",
        help="Allow review_generate on the HTTP transport (off by default there).",
    )
    parser.add_argument(
        "--allow-out-dir",
        action="append",
        default=[],
        help="Additional directory review_export_tree may write to; repeat for multiple roots.",
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
    _configure_capabilities(args)
    if args.transport == "streamable-http":
        try:
            token = _http_token()
            if token is None and not _is_loopback(args.host):
                raise ValueError(f"refusing to serve on {args.host} without {AUTH_TOKEN_ENV}")
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args.check:
        return _check()

    caps = capabilities.current()
    logger.info(
        "capabilities: read_only=%s attempts=%s plan=%s generate=%s export_write=%s",
        caps.read_only,
        caps.attempts,
        caps.plan,
        caps.generate,
        caps.export_write,
    )
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
