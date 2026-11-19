"""The importable Review Studio skill package must stay importable.

The core skill service parses SKILL.md frontmatter with ``yaml.safe_load`` and
routes agents by ``name``/``description``/``tags``. This test uses the core's
own public reader (``deeptutor.services.skill.hub.read_skill_metadata``) rather
than a second parser, so a format change in the service is caught here.
"""

from __future__ import annotations

from pathlib import Path
from importlib.resources import files

from deeptutor.services.skill.hub import read_skill_metadata

from _layout import skill_dir

SKILL_DIR = skill_dir()
SKILL_MD = SKILL_DIR / "SKILL.md"

TOOL_NAMES = (
    "review_specs",
    "review_due",
    "review_today",
    "review_attempt",
    "review_weakness",
    "review_export_link",
    "review_export_tree",
    "review_book_figures",
    "review_generate",
    "review_hint",
    "review_plan",
    "review_reveal",
    "review_quality",
    "review_correct",
    "review_metrics",
    "review_session",
    "review_link",
    "review_session_plan",
    "review_artifact_status",
)


def _meta() -> dict:
    meta = read_skill_metadata(SKILL_DIR)
    assert meta, f"SKILL.md frontmatter did not parse: {SKILL_MD}"
    return meta


def _body() -> str:
    text = SKILL_MD.read_text(encoding="utf-8")
    assert text.startswith("---"), "SKILL.md must start with YAML frontmatter"
    _, _, body = text.partition("\n---")
    assert body.strip(), "SKILL.md has no body after the frontmatter"
    return body


def test_skill_md_exists() -> None:
    assert SKILL_MD.is_file(), f"missing skill package: {SKILL_MD}"


def test_skill_doc_sync() -> None:
    assert files("deeptutor.review").joinpath("skill", "SKILL.md").read_bytes() == SKILL_MD.read_bytes()


def test_frontmatter_required_keys() -> None:
    meta = _meta()
    for key in ("name", "description", "tags"):
        assert key in meta, f"frontmatter is missing `{key}`"
    assert meta["name"] == "review-studio"
    assert isinstance(meta["description"], str)
    assert len(meta["description"].strip()) >= 40, "description is too short to route on"
    assert isinstance(meta["tags"], list)
    assert set(meta["tags"]) == {"education", "review", "spaced-repetition"}


def test_body_names_every_tool() -> None:
    body = _body()
    for tool in TOOL_NAMES:
        assert tool in body, f"body does not mention `{tool}`"


def test_body_states_the_red_lines() -> None:
    lowered = _body().lower()
    assert "self-assess" in lowered, "body must tell the agent to wait for self-assessment"
    assert "never grade" in lowered, "body must forbid grading on the learner's behalf"
    assert "spec_id" in lowered, "body must require ids copied from tool results"
