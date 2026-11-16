"""ReviewSpec v1: the single source of truth for generated review material.

Every consumer (queue, export, future in-app renderer) reads this model. The
model is strict on purpose: an unknown field is a typo or a schema drift, not
something to silently drop before the content reaches a learner.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from datetime import date
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"

# Phase 0.5 is a local, hand-written spec. These bounds keep a malformed or
# hostile request from turning validation/QA/export into an OOM amplifier.
MAX_SPEC_BYTES = 1_500_000
MAX_STRING_CHARS = 200_000
MAX_CONTAINER_ITEMS = 2_000
# TikZ source is compiled by an external process; keep it far below the generic
# string budget so one figure cannot monopolise a build.
MAX_TIKZ_CHARS = 20_000
# ``GET /api/review/specs/{id}`` and ``GET /api/review/{id}/today`` are
# ambiguous when id == "specs"; reserve the literal route segment.
RESERVED_SPEC_IDS = {"specs"}

# ``quality`` is a derived QA report, not content. Excluding it keeps the
# content hash stable when gates are re-run; otherwise every QA fix would look
# like a new spec to storage and the offline exporter.
_HASH_EXCLUDE = {"content_hash", "quality"}

Difficulty = Literal["easy", "medium", "hard"]
KnowledgeTypeName = Literal["memory", "concept", "procedure", "design"]
ItemKind = Literal["single_choice", "numeric", "short_answer", "flashcard"]
SourceKind = Literal["doc", "chunk", "book_page", "manual"]
Coverage = Literal["kb", "supplement", "inferred"]
# Mastery path ids become LearningStore path components in core. Keep them in
# the same conservative alphabet as ReviewSpec ids instead of forwarding an
# arbitrary string to the learning store.
_MASTERY_PATH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def normalize_text(value: str) -> str:
    """NFKC + collapsed whitespace + lowercase, for names and grading."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().lower()


def normalize_formula(value: str) -> str:
    """NFKC + collapsed whitespace, preserving case.

    ``L`` and ``l`` are different circuit variables; lowercasing a formula
    would make two distinct concepts share an identity.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def concept_key(name: str, primary_formula_latex: str, kb_ids: list[str]) -> str:
    """Stable concept identity across regenerations (RS-202).

    Deliberately independent of ids, order, and dictionary order. Two books
    that teach the same named concept with the same primary formula in the
    same KB set share a review state; a renamed concept does not silently
    inherit another concept's schedule.
    """
    raw = "|".join(
        (
            normalize_text(name),
            normalize_formula(primary_formula_latex),
            "|".join(sorted(kb_ids)),
        )
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class Formula(StrictModel):
    latex: str = Field(min_length=1)
    conditions: str = ""
    variables: dict[str, str] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)


class Concept(StrictModel):
    id: str = Field(min_length=1)
    key: str = ""
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    definition: str = ""
    formulas: list[Formula] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)  # concept ids
    difficulty: Difficulty = "medium"
    knowledge_type: KnowledgeTypeName = "concept"
    source_ids: list[str] = Field(default_factory=list)
    summary_one_line: str = ""
    coverage: Coverage = "kb"
    # Optional objective in the linked mastery path. Empty means this concept
    # is review-only; it must not be guessed from the name or book id.
    knowledge_point_id: str = ""

    @field_validator("knowledge_point_id", mode="before")
    @classmethod
    def _clean_knowledge_point_id(cls, value: object) -> str:
        raw = str(value or "").strip()
        if len(raw) > 200:
            raise ValueError("knowledge_point_id exceeds 200 characters")
        if any(character in raw for character in ("\r", "\n", "\x00")):
            raise ValueError("knowledge_point_id contains a control character")
        return raw


class Option(StrictModel):
    label: str = Field(min_length=1)
    text: str = Field(min_length=1)


class Answer(StrictModel):
    choice_index: int | None = None
    value: float | None = None
    tolerance: float = 0.0
    text: str = ""


class Item(StrictModel):
    id: str = Field(min_length=1)
    concept_ids: list[str] = Field(min_length=1)
    kind: ItemKind

    @field_validator("id", mode="before")
    @classmethod
    def _clean_item_id(cls, value: object) -> str:
        # Item ids reach core logs as question_id; a newline would let a spec
        # forge log lines even though it never becomes a path component.
        raw = str(value or "")
        if any(character in raw for character in ("\r", "\n", "\x00")):
            raise ValueError("item id contains a control character")
        return raw
    stem: str = Field(min_length=1)
    options: list[Option] = Field(default_factory=list)
    answer: Answer
    explanation: str = ""
    #: Study aids shown only inside the answer panel, after the attempt. They
    #: deliberately have no place in ``review_today``'s pre-answer payload.
    self_check: str = Field(default="", max_length=600)
    pitfalls: list[str] = Field(default_factory=list, max_length=3)
    assumptions: list[str] = Field(default_factory=list, max_length=3)
    #: Markdown comparison table shown only in the answer panel. It reuses the
    #: same table renderer as the explanation; ``review_today`` never exposes it.
    study_table: str = Field(default="", max_length=4000)
    difficulty: Difficulty = "medium"
    source_ids: list[str] = Field(default_factory=list)
    variant_of: str | None = None

    @field_validator("stem")
    @classmethod
    def _clean_stem(cls, value: str) -> str:
        # ``min_length=1`` accepts a single space, and ``str.strip`` does not
        # remove zero-width/format characters (U+200B and friends): either
        # would reach the learner as an empty card and pass every later gate.
        visible = "".join(
            character
            for character in value
            if unicodedata.category(character) not in {"Cf", "Cc"}
        )
        if not visible.strip():
            raise ValueError("stem must not be blank")
        return value.strip()

    @model_validator(mode="after")
    def _answer_matches_kind(self) -> "Item":
        if self.kind == "single_choice":
            if len(self.options) < 2:
                raise ValueError("single_choice requires at least two options")
            labels = [option.label.strip().upper() for option in self.options]
            if len(set(labels)) != len(labels):
                raise ValueError("single_choice option labels must be unique")
            # Duplicate option text makes two answers "correct" and turns the
            # item into a coin flip; labels alone are not enough.
            texts = [" ".join(option.text.split()).casefold() for option in self.options]
            if len(set(texts)) != len(texts):
                raise ValueError("single_choice option texts must be unique")
            if self.answer.choice_index is None:
                raise ValueError("single_choice requires answer.choice_index")
            if not 0 <= self.answer.choice_index < len(self.options):
                raise ValueError("answer.choice_index is out of range")
        elif self.kind == "numeric":
            if self.answer.value is None:
                raise ValueError("numeric requires answer.value")
            if not math.isfinite(self.answer.value):
                raise ValueError("numeric answer.value must be finite")
            if not math.isfinite(self.answer.tolerance):
                raise ValueError("answer.tolerance must be finite")
            if self.answer.tolerance < 0:
                raise ValueError("answer.tolerance must be non-negative")
        else:
            if not self.answer.text.strip():
                raise ValueError(f"{self.kind} requires answer.text")
        return self


class ProseBlock(StrictModel):
    kind: Literal["prose"] = "prose"
    text: str = Field(min_length=1)


class CalloutBlock(StrictModel):
    kind: Literal["callout"] = "callout"
    tone: Literal["info", "warning", "tip"] = "info"
    text: str = Field(min_length=1)


class FigureBlock(StrictModel):
    kind: Literal["figure"] = "figure"
    figure_id: str = Field(min_length=1)


class WidgetBlock(StrictModel):
    kind: Literal["widget"] = "widget"
    widget_id: str = Field(min_length=1)


class ItemBlock(StrictModel):
    kind: Literal["item"] = "item"
    item_id: str = Field(min_length=1)


class SummaryBlock(StrictModel):
    kind: Literal["summary"] = "summary"
    text: str = Field(min_length=1)


Block = Annotated[
    Union[
        ProseBlock,
        CalloutBlock,
        FigureBlock,
        WidgetBlock,
        ItemBlock,
        SummaryBlock,
    ],
    Field(discriminator="kind"),
]


class Section(StrictModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    concept_ids: list[str] = Field(default_factory=list)
    order: int = 0
    blocks: list[Block] = Field(default_factory=list)


class SourceRef(StrictModel):
    id: str = Field(min_length=1)
    kind: SourceKind
    title: str = Field(min_length=1)
    kb: str = ""
    ref: str = ""
    snippet: str = ""
    quote_hash: str = ""


class FigureMark(StrictModel):
    """A labelled point on a panel (an extremum, zero crossing, or annotation)."""

    x: float
    value: float
    text: str = ""


class FigurePanel(StrictModel):
    """One panel of a shared-x-axis engineering figure.

    ``shear`` panels draw piecewise-constant segments (Q diagrams); ``moment``
    and ``line`` panels draw a polyline/area through points. Positive and
    negative values are coloured differently and always labelled with their
    numeric value, so colour is not the only information channel.
    """

    kind: Literal["shear", "moment", "line"] = "line"
    title: str = ""
    unit: str = ""
    segments: list[tuple[float, float, float]] = Field(default_factory=list)
    points: list[tuple[float, float]] = Field(default_factory=list)
    marks: list[FigureMark] = Field(default_factory=list)


class FigureSpec(StrictModel):
    id: str = Field(min_length=1)
    # ``plot``, ``bar`` and ``panels`` are rendered by the deterministic
    # provider; ``tikz`` goes through the external LaTeX toolchain; any other
    # kind falls back to a visible placeholder with a warning.
    kind: str = "placeholder"
    title: str = ""
    caption: str = ""
    svg: str = ""
    # LaTeX/TikZ source for ``kind="tikz"``. Rendered at export time, never
    # executed in the browser.
    tikz: str = Field(default="", max_length=MAX_TIKZ_CHARS)
    # Plot data is supplied as points, never as LLM-written SVG/coordinates.
    points: list[tuple[float, float]] = Field(default_factory=list)
    bar_labels: list[str] = Field(default_factory=list)
    bar_values: list[float] = Field(default_factory=list)
    # Multi-panel engineering data for ``kind="panels"``. Empty on every older
    # spec; ``_content_dump`` removes the empty default so existing hashes stay
    # stable.
    panels: list[FigurePanel] = Field(default_factory=list, max_length=4)
    x_label: str = ""
    y_label: str = ""
    unit: str = ""
    source_ids: list[str] = Field(default_factory=list)


class WidgetSpec(StrictModel):
    id: str = Field(min_length=1)
    # ``slider_plot`` reuses the same expression/sampling fields as
    # ``slider_formula``; only the browser rendering differs (curve + marker
    # instead of a bare readout), so no extra field is needed.
    kind: Literal["slider_formula", "slider_plot"] = "slider_formula"
    title: str = ""
    formula_latex: str = ""
    # Restricted arithmetic expression; the renderer parses it with a closed
    # AST and precomputes values. It is never eval'd in the browser.
    expression: str = ""
    parameter: str = "x"
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.1
    unit: str = ""
    precision: int = Field(default=3, ge=0, le=6)
    samples: int = Field(default=41, ge=5, le=101)
    source_ids: list[str] = Field(default_factory=list)


class ReviewPlan(StrictModel):
    daily_minutes: int = Field(default=15, ge=1, le=120)
    exam_date: date | None = None
    timezone: str = "Asia/Shanghai"
    target_retention: float = Field(default=0.9, ge=0.7, le=0.99)
    # ``timezone`` is validated on the write path (ReviewService.put_spec),
    # never here: a row written before that validation existed must stay
    # readable so one bad pack cannot make the whole module unusable.
    # ``timeutil.zone`` falls back to UTC with a warning for such rows.


class GateResult(StrictModel):
    gate: str
    passed: bool
    message: str = ""
    details: list[str] = Field(default_factory=list)


class QualityReport(StrictModel):
    passed: bool = True
    warnings: list[str] = Field(default_factory=list)
    gates: list[GateResult] = Field(default_factory=list)


class AssetRef(StrictModel):
    id: str = Field(min_length=1)
    kind: str = ""
    path: str = ""
    sha256: str = ""


def _content_dump(value: Any) -> Any:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", exclude=_HASH_EXCLUDE)
        # These optional fields are additive to v1. Omitting their empty
        # defaults keeps every Phase 0.5 spec's content hash stable across the
        # upgrade; a configured linkage still changes the hash because the
        # non-empty value remains in the payload.
        if not payload.get("mastery_path_id"):
            payload.pop("mastery_path_id", None)
        for concept in payload.get("concepts", []):
            if isinstance(concept, dict) and not concept.get("knowledge_point_id"):
                concept.pop("knowledge_point_id", None)
        # Same additive-field rule for the TikZ source: specs written before
        # the field existed must keep their content hash.
        for figure in payload.get("figures", []):
            if isinstance(figure, dict) and not figure.get("tikz"):
                figure.pop("tikz", None)
            # Same additive-field rule for panels: an empty default must not
            # change the identity of every pre-panel spec.
            if isinstance(figure, dict) and not figure.get("panels"):
                figure.pop("panels", None)
        # Study aids are additive too: old items keep their content hash until
        # a generator (or an author) actually supplies one.
        for item in payload.get("items", []):
            if not isinstance(item, dict):
                continue
            for field_name in ("self_check", "pitfalls", "assumptions", "study_table"):
                if not item.get(field_name):
                    item.pop(field_name, None)
        return payload
    return value


def canonical_json(value: Any) -> str:
    """Canonical JSON: sorted keys, compact separators, no NaN.

    This is the persistence and transport representation, so it keeps every
    field (including the derived QA report and the content hash). The byte-for
    -byte contract (C5) is enforced here rather than in each caller.
    """
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    """Hash only content, not the derived QA report or the hash field itself."""
    return hashlib.sha256(canonical_json(_content_dump(value)).encode("utf-8")).hexdigest()


def _check_payload_limits(value: Any, path: str = "$") -> None:
    if isinstance(value, str):
        if len(value) > MAX_STRING_CHARS:
            raise ValueError(f"{path} exceeds {MAX_STRING_CHARS} characters")
        return
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_CONTAINER_ITEMS:
            raise ValueError(f"{path} exceeds {MAX_CONTAINER_ITEMS} entries")
        for index, child in enumerate(value):
            _check_payload_limits(child, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        if len(value) > MAX_CONTAINER_ITEMS:
            raise ValueError(f"{path} exceeds {MAX_CONTAINER_ITEMS} keys")
        for key, child in value.items():
            _check_payload_limits(child, f"{path}.{key}")


class ReviewSpec(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    id: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )
    # Optional target mastery path. Explicit on purpose: ``book_id`` is the
    # content source, not necessarily the learning goal.
    mastery_path_id: str = ""
    book_id: str | None = None
    kb_ids: list[str] = Field(default_factory=list)
    language: Literal["zh", "en"] = "zh"
    title: str = Field(min_length=1)
    plan: ReviewPlan = Field(default_factory=ReviewPlan)
    concepts: list[Concept] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    figures: list[FigureSpec] = Field(default_factory=list)
    widgets: list[WidgetSpec] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    sources: dict[str, SourceRef] = Field(default_factory=dict)
    quality: QualityReport = Field(default_factory=QualityReport)
    assets: list[AssetRef] = Field(default_factory=list)
    content_hash: str = ""

    @field_validator("id", mode="before")
    @classmethod
    def _clean_spec_id(cls, value: object) -> str:
        raw = str(value or "")
        # Windows strips trailing dots/spaces from directory names, so "abc"
        # and "abc." would share one snapshot directory.
        if raw.endswith(".") or raw.endswith(" "):
            raise ValueError("spec id must not end with '.' or a space")
        return raw

    @field_validator("mastery_path_id", mode="before")
    @classmethod
    def _clean_mastery_path_id(cls, value: object) -> str:
        raw = str(value or "").strip()
        if raw and (not _MASTERY_PATH_ID.match(raw) or ".." in raw or raw.endswith(".")):
            raise ValueError(
                "mastery_path_id must start with an alphanumeric character, "
                "contain only letters, digits, '.', '_' or '-', and must not "
                "contain '..' or end with '.'"
            )
        return raw

    @model_validator(mode="after")
    def _fill_and_check(self) -> "ReviewSpec":
        if self.id in RESERVED_SPEC_IDS:
            raise ValueError(f"spec id {self.id!r} is reserved by the router")
        _check_payload_limits(self.model_dump(mode="json"))
        if len(canonical_json(self).encode("utf-8")) > MAX_SPEC_BYTES:
            raise ValueError(f"ReviewSpec exceeds {MAX_SPEC_BYTES} bytes")
        # Stable concept keys need the spec-level kb_ids, so they are filled
        # here rather than in Concept itself. Build a new list instead of
        # mutating a caller-owned Concept instance.
        self.concepts = [
            concept
            if concept.key
            else concept.model_copy(
                update={
                    "key": concept_key(
                        concept.name,
                        concept.formulas[0].latex if concept.formulas else "",
                        self.kb_ids,
                    )
                }
            )
            for concept in self.concepts
        ]

        concept_ids = [concept.id for concept in self.concepts]
        if len(set(concept_ids)) != len(concept_ids):
            raise ValueError("concept ids must be unique")
        keys = [concept.key for concept in self.concepts]
        if len(set(keys)) != len(keys):
            raise ValueError("concept keys collide; rename or add a primary formula")

        # Mastery linkage is all-or-nothing at the spec level: a knowledge
        # point without a path, or a path with no objective, is a configuration
        # mistake that would otherwise fail silently at projection time.
        mapped = [concept for concept in self.concepts if concept.knowledge_point_id]
        if mapped and not self.mastery_path_id:
            raise ValueError(
                "concepts carry knowledge_point_id but mastery_path_id is empty"
            )
        if self.mastery_path_id and not mapped:
            raise ValueError(
                "mastery_path_id is set but no concept carries knowledge_point_id"
            )
        knowledge_point_ids = [concept.knowledge_point_id for concept in mapped]
        if len(set(knowledge_point_ids)) != len(knowledge_point_ids):
            raise ValueError("knowledge_point_id must be unique within a spec")

        entity_ids: list[str] = []
        for collection in (
            self.concepts,
            self.sections,
            self.figures,
            self.widgets,
            self.items,
            list(self.sources.values()),
            self.assets,
        ):
            entity_ids.extend(entity.id for entity in collection)
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError("all entity ids must be globally unique")

        for key, source in self.sources.items():
            if key != source.id:
                raise ValueError(
                    f"source key {key!r} does not match SourceRef.id {source.id!r}"
                )

        concept_id_set = set(concept_ids)
        source_id_set = set(self.sources)
        figure_ids = {figure.id for figure in self.figures}
        widget_ids = {widget.id for widget in self.widgets}
        item_ids = {item.id for item in self.items}

        for concept in self.concepts:
            unknown = set(concept.prerequisites) - concept_id_set
            if unknown:
                raise ValueError(f"concept {concept.id} has unknown prerequisites: {sorted(unknown)}")
            unknown_sources = set(concept.source_ids) - source_id_set
            if unknown_sources:
                raise ValueError(
                    f"concept {concept.id} has unknown sources: {sorted(unknown_sources)}"
                )
            if concept.coverage != "kb" and not any(
                re.search(
                    rf"(?<![A-Za-z0-9_-]){re.escape(concept.id)}(?![A-Za-z0-9_-])",
                    warning,
                )
                for warning in self.quality.warnings
            ):
                raise ValueError(
                    f"concept {concept.id} is {concept.coverage} and must be listed "
                    "in quality.warnings"
                )

        for item in self.items:
            if len(set(item.concept_ids)) != len(item.concept_ids):
                raise ValueError(f"item {item.id} repeats a concept id")
            unknown = set(item.concept_ids) - concept_id_set
            if unknown:
                raise ValueError(f"item {item.id} has unknown concepts: {sorted(unknown)}")
            if not item.source_ids:
                raise ValueError(f"item {item.id} must have at least one source")
            unknown_sources = set(item.source_ids) - source_id_set
            if unknown_sources:
                raise ValueError(f"item {item.id} has unknown sources: {sorted(unknown_sources)}")
            if item.variant_of is not None and item.variant_of not in item_ids:
                raise ValueError(f"item {item.id} has unknown variant_of")

        for section in self.sections:
            unknown = set(section.concept_ids) - concept_id_set
            if unknown:
                raise ValueError(f"section {section.id} has unknown concepts: {sorted(unknown)}")
            for block in section.blocks:
                if isinstance(block, FigureBlock) and block.figure_id not in figure_ids:
                    raise ValueError(
                        f"section {section.id} references unknown figure {block.figure_id}"
                    )
                if isinstance(block, WidgetBlock) and block.widget_id not in widget_ids:
                    raise ValueError(
                        f"section {section.id} references unknown widget {block.widget_id}"
                    )
                if isinstance(block, ItemBlock) and block.item_id not in item_ids:
                    raise ValueError(
                        f"section {section.id} references unknown item {block.item_id}"
                    )

        computed = content_hash(self)
        if self.content_hash and self.content_hash != computed:
            raise ValueError(
                f"content_hash mismatch: declared {self.content_hash}, computed {computed}"
            )
        self.content_hash = computed
        return self

    def concept_by_id(self, concept_id: str) -> Concept:
        for concept in self.concepts:
            if concept.id == concept_id:
                return concept
        raise KeyError(concept_id)

    def concept_by_key(self, key: str) -> Concept:
        for concept in self.concepts:
            if concept.key == key:
                return concept
        raise KeyError(key)

    def item_by_id(self, item_id: str) -> Item:
        for item in self.items:
            if item.id == item_id:
                return item
        raise KeyError(item_id)

    def items_for_concept(self, concept_id: str) -> list[Item]:
        """Items whose *primary* concept is ``concept_id``.

        Phase 0.5 schedules one concept per item. Additional ``concept_ids``
        are display metadata for now; counting the same item under two
        concepts would put it in two queues while an attempt updates only one
        state. W1/RS-203 owns the explicit multi-concept policy.
        """
        return [item for item in self.items if item.concept_ids[0] == concept_id]
