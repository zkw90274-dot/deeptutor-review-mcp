from __future__ import annotations

from typing import Any


def _output_schema(
    required: dict[str, Any], optional: dict[str, Any] | None = None,
) -> dict[str, Any]:
    instruction = {"instruction": {"type": "string"}}
    return {
        "type": "object",
        "properties": {**required, **(optional or {}), **instruction, "error": {"type": "string"}},
        "required": ["instruction"],
        "anyOf": [{"required": list(required)}, {"required": ["error"]}],
    }


_STRING = {"type": "string"}
_COUNT = {"type": "integer", "minimum": 0}
_QUALITY_STATE = {"enum": ["verified", "legacy", "draft", "rejected"]}
_HASH = {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"}
_ARTIFACT_FIELDS = {
    "spec_id": _STRING,
    "title": _STRING,
    "chapter_fingerprint": _HASH,
    "content_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "book_fingerprint": _HASH,
    "status": {"enum": ["match", "drift", "unknown"]},
    "reason": _STRING,
}

OUTPUT_SCHEMAS = {
    "review_quality": _output_schema({
        "spec_id": _STRING,
        "state": _QUALITY_STATE,
        "content_hash": _STRING,
        "quality_state": _QUALITY_STATE,
        "quality_warning": {"type": ["string", "null"]},
        "anchored_items": _COUNT,
        "items": _COUNT,
    }, {
        "gates": {"type": "object"},
        "support": {"type": "object", "additionalProperties": _STRING},
        "coverage": {"type": "number"},
        "duplicate_stems": {"type": "array"},
        "answer_leaks": {"type": "array", "items": _STRING},
        "generated_at": {"type": "number"},
    }),
    "review_metrics": _output_schema({
        "spec_id": _STRING,
        "generated_at": _STRING,
        "window_days": {"type": "integer", "minimum": 1},
        "retention": {"type": "array", "items": {"type": "object"}},
        "completion": {"type": "object"},
        "objective_accuracy": {"type": "object"},
        "learner_calibration": {"type": "object"},
        "sampled_events": _COUNT,
        "sampled_sessions": _COUNT,
        "truncated": {"type": "boolean"},
    }),
    "review_artifact_status": _output_schema({
        "items": {"type": "array", "items": {
            "type": "object", "properties": _ARTIFACT_FIELDS, "required": list(_ARTIFACT_FIELDS),
        }},
        "count": _COUNT,
        "omitted": _COUNT,
    }),
}
