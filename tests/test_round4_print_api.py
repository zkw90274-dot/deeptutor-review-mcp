"""Round-4 trust fixes: print must not leak self-test answers, and storage
outages must not be reported as either a raw 500 or data corruption."""

from __future__ import annotations

import re
import sqlite3
from importlib.resources import files
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from deeptutor.review import api as review_api
from deeptutor.review.models import ReviewSpec
from deeptutor.review.storage import (
    ReviewStore,
    ReviewStoreError,
    ReviewStoreIOError,
    SpecIntegrityError,
)

TEMPLATE = (
    files("deeptutor.review").joinpath("templates/review.html").read_text(encoding="utf-8")
)


def _print_block() -> str:
    start = TEMPLATE.index("@media print")
    return TEMPLATE[start : TEMPLATE.index("</style>", start)]


def _validation_error() -> ValidationError:
    try:
        ReviewSpec.model_validate({"id": ""})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


# -- A. template ------------------------------------------------------------


def test_print_hides_selftest_answers_only() -> None:
    block = _print_block()
    assert 'body[data-mode="selftest"] .answer{display:none !important}' in block
    # Quick/deep keep their answers on paper.
    assert ".answer{display:block !important" in block


def test_print_hides_self_explain_and_math_progress() -> None:
    block = _print_block()
    assert ".self-explain,.self-explain-label,#math-progress{display:none !important}" in block


def test_print_waits_for_mathjax_with_bounded_fallback() -> None:
    start = TEMPLATE.index('document.getElementById("print")')
    handler = TEMPLATE[start : TEMPLATE.index("var footer", start)]
    assert "mathjax-ready" in handler
    assert "MutationObserver" in handler
    assert re.search(r"setTimeout\(printNow, 3000\)", handler)
    assert "window.print()" in handler
    # The toolbar/ratings hiding that already shipped must survive the edit.
    assert ".toolbar,.ratings,.correction,button:not(.option){display:none !important}" in _print_block()


def test_mathjax_failure_clears_progress_and_marks_ready() -> None:
    start = TEMPLATE.index("chain.then(function () {")
    block = TEMPLATE[start : TEMPLATE.index("};", start)]
    # Both the success and the failure handler must drop the progress line and
    # mark the document ready, so a failed batch cannot leave a stale banner
    # or make print wait for the 3 s fallback.
    assert block.count("if (progress) progress.remove();") == 2
    assert block.count('document.body.classList.add("mathjax-ready");') == 2
    assert "showWarning();" in block


def test_mathjax_batches_drop_earlier_items_from_the_document_registry() -> None:
    start = TEMPLATE.index("for (var offset = 0;")
    end = TEMPLATE.index(
        "      chain.then(function () {\n        if (progress) progress.remove();", start
    )
    block = TEMPLATE[start:end]
    # Each call renders only its own slice...
    assert "nodes.slice(offset, offset + 32)" in block
    assert "typesetPromise(batch)" in block
    # ...and clears the document registry afterwards: every typesetPromise
    # re-walks all registered items, so without the clear the total work grows
    # quadratically with the number of formulas.
    assert "window.MathJax.typesetClear()" in block


# -- B1. /specs -------------------------------------------------------------


def test_specs_maps_store_failure_to_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenService:
        def list_specs(self):
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(review_api, "_service", lambda: _BrokenService())
    response = client.get("/api/review/specs")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "storage_unavailable"
    assert "OperationalError" not in response.text


# -- B2. /ui ----------------------------------------------------------------


def _ui_with_card_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> str:
    class _Service:
        def list_specs(self):
            return [{"spec_id": "pack-broken"}]

    def _boom(service, meta):
        raise exc

    monkeypatch.setattr(review_api, "_service", lambda: _Service())
    monkeypatch.setattr(review_api, "_overview_card", _boom)
    page = client.get("/api/review/ui")
    assert page.status_code == 200
    return page.text


@pytest.mark.parametrize(
    "exc",
    [
        sqlite3.OperationalError("database is locked"),
        ReviewStoreIOError("review store cannot be migrated read-write"),
        ReviewStoreError("migration failed"),
    ],
)
def test_ui_reports_storage_outage_not_corruption(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    body = _ui_with_card_error(client, monkeypatch, exc)
    assert "pack-broken" in body
    assert "存储暂时不可用" in body
    assert "已损坏" not in body


@pytest.mark.parametrize("exc", [_validation_error(), SpecIntegrityError("row/spec mismatch")])
def test_ui_still_reports_corruption_for_bad_data(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    body = _ui_with_card_error(client, monkeypatch, exc)
    assert "该复习包已损坏" in body
    assert "存储暂时不可用" not in body


# -- B3. storage._migrate ---------------------------------------------------


def test_migrate_raises_io_error_and_maps_to_503(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = ReviewStore(tmp_path / "review")

    def _read_only(self):
        raise sqlite3.OperationalError("attempt to write a readonly database")

    monkeypatch.setattr(ReviewStore, "_migrate_locked", _read_only)
    with pytest.raises(ReviewStoreIOError):
        store._migrate()
    assert issubclass(ReviewStoreIOError, ReviewStoreError)
    assert review_api._handle(ReviewStoreIOError("x")).status_code == 503
    assert review_api._handle(ReviewStoreError("x")).status_code == 409
