from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

# ``deeptutor`` is a regular package in site-packages, so putting the checkout
# on sys.path does not merge the namespace portion: the installed wheel always
# wins and a source edit would only take effect after a wheel rebuild. Prepend
# the checkout to the package's own ``__path__`` instead, so the suite exercises
# the working tree. ``deeptutor.modules`` has no ``__init__.py`` in the checkout,
# so the manifest catalog and ``module_enabled`` still resolve to the installed
# distribution, which is what the runtime uses.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import deeptutor  # noqa: E402

_CHECKOUT_PACKAGE = SRC / "deeptutor"
if str(_CHECKOUT_PACKAGE) not in deeptutor.__path__:
    deeptutor.__path__.insert(0, str(_CHECKOUT_PACKAGE))

from deeptutor.review.fixture import build_fixture_spec  # noqa: E402
from deeptutor.review.models import (  # noqa: E402
    Answer,
    Concept,
    Formula,
    Item,
    Option,
    ReviewPlan,
    ReviewSpec,
    SourceRef,
)
from deeptutor.review.service import ReviewService  # noqa: E402
from deeptutor.review.storage import ReviewStore  # noqa: E402

NOW = 1_800_000_000.0
EVENING = datetime(2026, 11, 6, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()


@pytest.fixture(autouse=True)
def _isolate_review_home(monkeypatch: pytest.MonkeyPatch, tmp_path_factory) -> None:
    """Never read the operator's ``book-map.json`` or other local review data."""
    home = tmp_path_factory.mktemp("review-home")
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(home))


@pytest.fixture(autouse=True)
def _isolate_core_state(monkeypatch: pytest.MonkeyPatch):
    """Unit tests never read the machine's real mastery store by accident."""
    from deeptutor.review import core_state

    monkeypatch.setattr(core_state, "load_core_progress", lambda path_id: None)


@pytest.fixture()
def fixture_spec() -> ReviewSpec:
    return build_fixture_spec()


@pytest.fixture()
def linked_spec() -> ReviewSpec:
    """Fixture spec with an explicit mastery path and two objectives."""
    payload = build_fixture_spec().model_dump(mode="json")
    payload["content_hash"] = ""
    payload["mastery_path_id"] = "fixture-mastery"
    knowledge_points = {
        "c-series-resonance": "kp-series-resonance",
        "c-quality-factor": "kp-quality-factor",
    }
    for concept in payload["concepts"]:
        concept["knowledge_point_id"] = knowledge_points.get(concept["id"], "")
    return ReviewSpec.model_validate(payload)


@pytest.fixture()
def store(tmp_path: Path) -> ReviewStore:
    return ReviewStore(tmp_path / "review")


@pytest.fixture()
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReviewService:
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(tmp_path / "review"))
    # Linked concepts may read the core mastery store; keep that isolated too.
    monkeypatch.setenv("DEEPTUTOR_HOME", str(tmp_path / "core-home"))
    return ReviewService(ReviewStore(tmp_path / "review"))


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEEPTUTOR_REVIEW_HOME", str(tmp_path / "review"))
    # Isolate the core runtime too; otherwise TestClient writes default
    # settings/logs under the current working directory.
    monkeypatch.setenv("DEEPTUTOR_HOME", str(tmp_path / "core-home"))
    monkeypatch.setenv("AUTH_ENABLED", "false")
    # The first-run bootstrap that writes main.yaml/agents.yaml runs once per
    # process, at `deeptutor.api.main` import. When an earlier test already
    # imported it (or cached the PathService against another home), a fresh
    # DEEPTUTOR_HOME never gets its settings and the app dies on a missing
    # main.yaml. Reset the singleton and re-run the (idempotent) bootstrap, or
    # this fixture only passes depending on which files ran before it.
    from deeptutor.services.path_service import PathService

    PathService.reset_instance()
    from deeptutor.services.setup.init import init_user_directories

    init_user_directories()
    from fastapi.testclient import TestClient
    from deeptutor.api.main import app

    return TestClient(app)


def make_multi_item_spec(items: int) -> ReviewSpec:
    """One concept with many items, for queue-rotation tests."""
    source = SourceRef(
        id="src-multi",
        kind="manual",
        title="多题概念来源",
        snippet="测试用来源，不含外部链接。",
    )
    concept = Concept(
        id="c-multi",
        name="多题概念",
        definition="一个概念对应多道题。",
        formulas=[Formula(latex="x = 1", source_ids=["src-multi"])],
        source_ids=["src-multi"],
        knowledge_type="concept",
        coverage="kb",
    )
    entries = [
        Item(
            id=f"item-{index:02d}",
            concept_ids=[concept.id],
            kind="single_choice",
            stem=f"多题概念第 {index} 题：哪个正确？",
            options=[Option(label="A", text="正确"), Option(label="B", text="错误")],
            answer=Answer(choice_index=0),
            explanation="A. 正确。",
            source_ids=["src-multi"],
        )
        for index in range(items)
    ]
    return ReviewSpec(
        id=f"multi-{items}",
        kb_ids=["test"],
        title="多题概念",
        plan=ReviewPlan(daily_minutes=10),
        concepts=[concept],
        items=entries,
        sources={"src-multi": source},
    )


def make_wide_spec(due: int, new: int) -> ReviewSpec:
    """A synthetic spec for budget/backlog tests; every item is answerable."""
    sources = {
        "src-test": SourceRef(
            id="src-test",
            kind="manual",
            title="测试来源",
            snippet="测试用来源，不含外部链接。",
        )
    }
    concepts: list[Concept] = []
    items: list[Item] = []
    for index in range(due + new):
        due_part = index < due
        concept_id = f"c-{'due' if due_part else 'new'}-{index}"
        concepts.append(
            Concept(
                id=concept_id,
                name=f"{'到期' if due_part else '新'}概念{index}",
                definition="测试概念",
                formulas=[Formula(latex=f"x_{index} = {index}", source_ids=["src-test"])],
                source_ids=["src-test"],
                knowledge_type="concept",
                coverage="kb",
            )
        )
        items.append(
            Item(
                id=f"item-{index}",
                concept_ids=[concept_id],
                kind="single_choice",
                stem=f"测试题 {index}：哪一个选项正确？",
                options=[
                    Option(label="A", text="正确"),
                    Option(label="B", text="错误"),
                ],
                answer=Answer(choice_index=0),
                explanation="A. 正确，所以选 A。",
                source_ids=["src-test"],
            )
        )
    return ReviewSpec(
        id=f"wide-{due}-{new}",
        kb_ids=["test"],
        title="宽队列测试",
        plan=ReviewPlan(daily_minutes=10),
        concepts=concepts,
        items=items,
        sources=sources,
    )
