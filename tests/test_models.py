from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from deeptutor.review.fixture import build_fixture_spec
from deeptutor.review.models import (
    Concept,
    QualityReport,
    ReviewSpec,
    canonical_json,
    concept_key,
    content_hash,
)
from deeptutor.review.qa import apply_quality, run_gates


def test_fixture_passes_all_gates(fixture_spec: ReviewSpec) -> None:
    report = run_gates(fixture_spec)
    assert report.passed, report.model_dump()
    assert {gate.gate for gate in report.gates} == {
        "G1-schema",
        "G2-sources",
        "G3-math-escape",
        "G4-answer-key",
        "G6-injection",
        "G8-offline",
        "G9-parameter-feasibility",
        "G10-math-balance",
    }


def test_extra_field_is_rejected(fixture_spec: ReviewSpec) -> None:
    payload = fixture_spec.model_dump(mode="json")
    payload["concepts"][0]["unexpected"] = True
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_item_referencing_unknown_concept_is_rejected(fixture_spec: ReviewSpec) -> None:
    payload = fixture_spec.model_dump(mode="json")
    payload["items"][0]["concept_ids"] = ["does-not-exist"]
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_item_repeating_a_concept_is_rejected(fixture_spec: ReviewSpec) -> None:
    payload = fixture_spec.model_dump(mode="json")
    payload["items"][0]["concept_ids"] = [
        payload["items"][0]["concept_ids"][0],
        payload["items"][0]["concept_ids"][0],
    ]
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_item_without_source_is_rejected(fixture_spec: ReviewSpec) -> None:
    payload = fixture_spec.model_dump(mode="json")
    payload["items"][0]["source_ids"] = []
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_spec_id_rejects_windows_aliases(fixture_spec: ReviewSpec) -> None:
    payload = fixture_spec.model_dump(mode="json")
    payload["content_hash"] = ""
    for bad in ("trailing.", "trailing "):
        payload["id"] = bad
        with pytest.raises(ValidationError):
            ReviewSpec.model_validate(payload)


def test_item_id_rejects_control_characters(fixture_spec: ReviewSpec) -> None:
    payload = fixture_spec.model_dump(mode="json")
    payload["content_hash"] = ""
    payload["items"][0]["id"] = "item\nforged"
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_source_key_must_match_source_id(fixture_spec: ReviewSpec) -> None:
    payload = fixture_spec.model_dump(mode="json")
    source = payload["sources"].pop("src-hf-2-1")
    payload["sources"]["renamed"] = source
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_duplicate_option_labels_are_rejected(fixture_spec: ReviewSpec) -> None:
    payload = fixture_spec.model_dump(mode="json")
    payload["items"][0]["options"][1]["label"] = payload["items"][0]["options"][0]["label"]
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_duplicate_ids_are_rejected(fixture_spec: ReviewSpec) -> None:
    payload = fixture_spec.model_dump(mode="json")
    payload["items"][1]["id"] = payload["items"][0]["id"]
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_content_hash_is_stable_and_quality_independent(fixture_spec: ReviewSpec) -> None:
    first = fixture_spec.model_dump(mode="json")
    second = ReviewSpec.model_validate(first).model_dump(mode="json")
    assert first["content_hash"] == second["content_hash"]
    assert content_hash(fixture_spec) == fixture_spec.content_hash

    graded = apply_quality(fixture_spec)
    assert graded.quality.passed
    assert content_hash(graded) == content_hash(fixture_spec)


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 1, "a": [2, 3]}) == canonical_json({"a": [2, 3], "b": 1})


def test_formula_case_is_identity_significant() -> None:
    # L and l are different circuit variables; lowercasing a formula would
    # make two distinct concepts share one review state.
    assert concept_key("谐振", "L", ["kb"]) != concept_key("谐振", "l", ["kb"])
    assert concept_key("Resonance", "L", ["kb"]) == concept_key("resonance", "L", ["kb"])


def test_concept_key_is_filled_from_primary_formula() -> None:
    spec = build_fixture_spec()
    assert all(concept.key for concept in spec.concepts)
    assert len({concept.key for concept in spec.concepts}) == len(spec.concepts)


def test_legacy_payload_without_keys_or_schema_version_keeps_identity() -> None:
    # A pack written before concept keys existed must load with the same
    # derived keys and content hash instead of failing the hash check.
    spec = build_fixture_spec()
    payload = spec.model_dump(mode="json")
    for concept in payload["concepts"]:
        concept.pop("key")
    payload.pop("schema_version")

    loaded = ReviewSpec.model_validate(payload)

    assert loaded.schema_version == spec.schema_version
    assert [concept.key for concept in loaded.concepts] == [
        concept.key for concept in spec.concepts
    ]
    assert loaded.content_hash == spec.content_hash


def test_future_spec_schema_version_is_rejected() -> None:
    payload = build_fixture_spec().model_dump(mode="json")
    payload["schema_version"] = "2.0"
    payload["content_hash"] = ""
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_content_hash_mismatch_is_rejected(fixture_spec: ReviewSpec) -> None:
    payload = fixture_spec.model_dump(mode="json")
    payload["content_hash"] = "0" * 64
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_reserved_route_segment_is_rejected_as_spec_id() -> None:
    with pytest.raises(ValidationError):
        ReviewSpec(id="specs", title="reserved")


def test_oversized_string_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ReviewSpec(id="too-big", title="x" * 200_001)


def test_coverage_warning_must_name_the_exact_concept_id() -> None:
    with pytest.raises(ValidationError):
        ReviewSpec(
            id="warning-boundary",
            title="warning boundary",
            concepts=[Concept(id="c1", name="concept one", coverage="supplement")],
            quality=QualityReport(warnings=["c11 needs a review"]),
        )
    accepted = ReviewSpec(
        id="warning-boundary",
        title="warning boundary",
        concepts=[Concept(id="c1", name="concept one", coverage="supplement")],
        quality=QualityReport(warnings=["c1 needs a review"]),
    )
    assert accepted.concepts[0].key


def test_unlinked_spec_hash_is_stable_across_linkage_fields(
    fixture_spec: ReviewSpec,
) -> None:
    """Phase 0.5 stored specs must keep their old content_hash after upgrade."""
    payload = fixture_spec.model_dump(mode="json")
    payload.pop("mastery_path_id", None)
    for concept in payload["concepts"]:
        concept.pop("knowledge_point_id", None)
    payload["content_hash"] = ""
    old_shape = ReviewSpec.model_validate(payload)
    assert content_hash(old_shape) == content_hash(fixture_spec)


def test_mastery_linkage_is_explicit_and_complete(linked_spec: ReviewSpec) -> None:
    assert linked_spec.mastery_path_id == "fixture-mastery"
    assert [concept.knowledge_point_id for concept in linked_spec.concepts] == [
        "kp-series-resonance",
        "kp-quality-factor",
    ]
    # Linkage is content, not derived QA metadata.
    assert content_hash(linked_spec) != content_hash(build_fixture_spec())


def test_knowledge_point_without_mastery_path_is_rejected(fixture_spec: ReviewSpec) -> None:
    payload = fixture_spec.model_dump(mode="json")
    payload["content_hash"] = ""
    payload["concepts"][0]["knowledge_point_id"] = "kp-orphan"
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_mastery_path_without_any_objective_is_rejected(fixture_spec: ReviewSpec) -> None:
    payload = fixture_spec.model_dump(mode="json")
    payload["content_hash"] = ""
    payload["mastery_path_id"] = "path-without-objective"
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_duplicate_knowledge_point_is_rejected(linked_spec: ReviewSpec) -> None:
    payload = linked_spec.model_dump(mode="json")
    payload["content_hash"] = ""
    payload["concepts"][1]["knowledge_point_id"] = payload["concepts"][0][
        "knowledge_point_id"
    ]
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_mastery_path_id_rejects_path_segments(fixture_spec: ReviewSpec) -> None:
    payload = fixture_spec.model_dump(mode="json")
    payload["content_hash"] = ""
    payload["mastery_path_id"] = "../../etc/passwd"
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)
    payload["mastery_path_id"] = "a..b"
    with pytest.raises(ValidationError):
        ReviewSpec.model_validate(payload)


def test_committed_spec_schema_matches_model() -> None:
    path = Path(__file__).resolve().parents[1] / "spec_schema.json"
    committed = json.loads(path.read_text(encoding="utf-8"))
    live = json.loads(json.dumps(ReviewSpec.model_json_schema(), sort_keys=True))
    assert committed == live
