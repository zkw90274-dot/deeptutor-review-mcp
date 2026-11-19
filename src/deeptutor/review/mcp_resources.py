from __future__ import annotations

import asyncio
import json
import re

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.shared.exceptions import McpError
from pydantic import AnyUrl

from deeptutor.review import materials
from deeptutor.review.export import ExportTooLarge
from deeptutor.review.service import QualityGateFailed, ReviewService
from deeptutor.review.storage import ReviewStore, ReviewStoreError, SpecNotFound

BOOK_FIGURES_URI = "review://book-figures"
EXPORT_URI_TEMPLATE = "review://spec/{spec_id}/export"
_EXPORT_URI = re.compile(r"review://spec/([A-Za-z0-9][A-Za-z0-9._-]{0,127})/export")
_RESOURCE_NOT_FOUND = -32002
_RESOURCE_UNAVAILABLE = -32000


def _resource_error(code: int, message: str) -> McpError:
    return McpError(types.ErrorData(code=code, message=message))


def _export_page(spec_id: str) -> str:
    return ReviewService(ReviewStore(read_only=True)).export(spec_id)


def register_resources(server: Server) -> None:
    @server.list_resources()
    async def list_resources() -> list[types.Resource]:
        return [types.Resource(
            uri=AnyUrl(BOOK_FIGURES_URI), name="book-figures", mimeType="application/json",
            description="Read the local living-book figure index.",
        )]

    @server.list_resource_templates()
    async def list_resource_templates() -> list[types.ResourceTemplate]:
        return [types.ResourceTemplate(
            uriTemplate=EXPORT_URI_TEMPLATE, name="review-pack-export", mimeType="text/html",
            description="Read a self-contained offline review pack. Locate spec_id with review_specs.",
        )]

    @server.read_resource()
    async def read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
        value = str(uri)
        if value == BOOK_FIGURES_URI:
            index = await asyncio.to_thread(materials.load_book_figures)
            return [ReadResourceContents(json.dumps(index, ensure_ascii=False), "application/json")]

        match = _EXPORT_URI.fullmatch(value)
        if match is None or match[1].endswith("."):
            raise _resource_error(_RESOURCE_NOT_FOUND, "Unknown Review Studio resource")
        try:
            page = await asyncio.to_thread(_export_page, match[1])
        except SpecNotFound:
            raise _resource_error(_RESOURCE_NOT_FOUND, "Review pack not found") from None
        except ExportTooLarge:
            raise _resource_error(_RESOURCE_UNAVAILABLE, "Review export exceeds the resource size limit") from None
        except QualityGateFailed:
            raise _resource_error(_RESOURCE_UNAVAILABLE, "Review export failed quality gates") from None
        except (ReviewStoreError, ValueError):
            raise _resource_error(_RESOURCE_UNAVAILABLE, "Review export is unavailable") from None
        return [ReadResourceContents(page, "text/html")]
