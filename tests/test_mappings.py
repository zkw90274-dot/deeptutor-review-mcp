from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from deeptutor.review.models import (
    Answer,
    Concept,
    Item,
    Option,
    ReviewPlan,
    ReviewSpec,
    SourceRef,
)
from deeptutor.review.service import ReviewService
from deeptutor.review.storage import SpecNotFound


def _concept(concept_id: str, title: str, *, knowledge_point_id: str = "") -> Concept:
    return Concept(
        id=concept_id,
        name=title,
        definition="映射测试概念。",
        source_ids=["src-map"],
        knowledge_type="concept",
        coverage="kb",
        knowledge_point_id=knowledge_point_id,
    )


def _item(item_id: str, concept_ids: list[str]) -> Item:
    return Item(
        id=item_id,
        concept_ids=concept_ids,
        kind="single_choice",
        stem=f"映射测试题 {item_id}：哪一个正确？",
        options=[Option(label="A", text="正确"), Option(label="B", text="错误")],
        answer=Answer(choice_index=0),
        explanation="A. 正确。",
        source_ids=["src-map"],
    )


def _mapping_spec() -> ReviewSpec:
    """Three concepts (two linked, one not) and items in a meaningful order.

    ``item-shared`` names two concepts on purpose: it must show up under both.
    """
    return ReviewSpec(
        id="mapping-spec",
        mastery_path_id="path-mapping",
        kb_ids=["test"],
        title="映射表测试",
        plan=ReviewPlan(daily_minutes=10),
        concepts=[
            _concept("c-alpha", "阿尔法概念", knowledge_point_id="kp-alpha"),
            _concept("c-beta", "贝塔概念", knowledge_point_id="kp-beta"),
            # Whitespace-only must be normalised to "" and count as unlinked.
            _concept("c-gamma", "伽马概念", knowledge_point_id="   "),
        ],
        items=[
            _item("item-beta", ["c-beta"]),
            _item("item-alpha-1", ["c-alpha"]),
            _item("item-shared", ["c-alpha", "c-gamma"]),
            _item("item-alpha-2", ["c-alpha"]),
            _item("item-gamma", ["c-gamma"]),
        ],
        sources={
            "src-map": SourceRef(
                id="src-map",
                kind="manual",
                title="映射测试来源",
                snippet="测试用来源，不含外部链接。",
            )
        },
    )


def test_mappings_reports_linkage_and_item_order(service: ReviewService) -> None:
    spec = _mapping_spec()
    service.put_spec(spec)

    body = service.mappings(spec.id)

    assert body["mastery_path_id"] == "path-mapping"
    assert body["total_concepts"] == 3
    assert body["linked_concepts"] == 2
    assert [concept["concept_id"] for concept in body["concepts"]] == [
        "c-alpha",
        "c-beta",
        "c-gamma",
    ]
    alpha, beta, gamma = body["concepts"]
    assert alpha["title"] == "阿尔法概念"
    assert alpha["knowledge_point_id"] == "kp-alpha"
    assert beta["knowledge_point_id"] == "kp-beta"
    # Whitespace-only input is stripped by the model and is not a link.
    assert gamma["knowledge_point_id"] == ""
    # Items follow ``spec.items`` order, not concept order; a multi-concept
    # item appears under every concept it names.
    assert alpha["item_ids"] == ["item-alpha-1", "item-shared", "item-alpha-2"]
    assert beta["item_ids"] == ["item-beta"]
    assert gamma["item_ids"] == ["item-shared", "item-gamma"]
    # The service layer returns the inner payload; the router owns ``spec_id``.
    assert "spec_id" not in body


def test_mappings_on_empty_concept_spec(service: ReviewService) -> None:
    spec = ReviewSpec(
        id="mapping-empty",
        kb_ids=["test"],
        title="空映射表",
        plan=ReviewPlan(daily_minutes=10),
        concepts=[],
        items=[],
        sources={},
    )
    service.put_spec(spec)

    body = service.mappings(spec.id)

    assert body == {
        "mastery_path_id": "",
        "concepts": [],
        "total_concepts": 0,
        "linked_concepts": 0,
    }


def test_mappings_unknown_spec_raises(service: ReviewService) -> None:
    with pytest.raises(SpecNotFound):
        service.mappings("does-not-exist")


def test_mappings_is_read_only(service: ReviewService) -> None:
    spec = _mapping_spec()
    service.put_spec(spec)
    before = service.store.get_spec(spec.id)

    service.mappings(spec.id)
    service.mappings(spec.id)

    assert service.store.get_spec(spec.id) == before
    assert service.store.list_specs() != []


def _put(client: TestClient, spec: ReviewSpec) -> None:
    response = client.put(
        f"/api/review/specs/{spec.id}", json=spec.model_dump(mode="json")
    )
    assert response.status_code in (200, 201), response.text


def test_mappings_endpoint_shape(client: TestClient) -> None:
    spec = _mapping_spec()
    _put(client, spec)

    response = client.get(f"/api/review/{spec.id}/mappings")

    assert response.status_code == 200
    body = response.json()
    assert body["spec_id"] == spec.id
    assert body["mastery_path_id"] == "path-mapping"
    assert body["total_concepts"] == 3
    assert body["linked_concepts"] == 2
    assert body["concepts"][0]["item_ids"] == [
        "item-alpha-1",
        "item-shared",
        "item-alpha-2",
    ]


def test_mappings_endpoint_missing_spec_is_404(client: TestClient) -> None:
    response = client.get("/api/review/no-such-spec/mappings")

    assert response.status_code == 404
    detail = response.json()["detail"]
    # Same error shape as every other review endpoint.
    assert detail["code"] == "not_found"
    assert "message" in detail


def test_mappings_route_is_mounted(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert "/api/review/{spec_id}/mappings" in paths
