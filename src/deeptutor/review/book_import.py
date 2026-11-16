"""Deterministic Book -> ReviewSpec importer (Phase 1 W3 slice).

This is deliberately not an LLM generation pipeline. It reads an existing
Book's pages and turns its text/quiz/flashcard blocks into a ReviewSpec, so a
compiled book can enter Review without duplicating Book generation. Optional
``review_item_id`` / ``knowledge_point_id`` fields on quiz questions are
preserved when present; without them the importer derives stable ids.

The remaining Book-side hook (calling Review's attempt endpoint from the Book
reader) is documented in ``docs/maintainer-notes.md``
§2; the core Book source is
not part of this repository.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from deeptutor.review.models import (
    Answer,
    Concept,
    Item,
    ItemBlock,
    Option,
    ProseBlock,
    ReviewPlan,
    ReviewSpec,
    Section,
    SourceRef,
)
from deeptutor.review.qa import answer_coverage

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_id(
    raw: object, *, prefix: str, fallback: str, unique: bool = False
) -> str:
    raw_text = str(raw or "").strip()
    text = _SAFE.sub("-", raw_text).strip("-._")
    # ``unique`` only matters when ASCII slugging lost information. Clean
    # ASCII ids keep their historical form (they are stable public ids);
    # CJK/punctuation-only ids all collapse to the same fallback and would
    # otherwise share a source or concept id.
    degraded = (
        not text
        or not text[0].isalnum()
        or _SAFE.search(raw_text) is not None
    )
    if not text or not text[0].isalnum():
        text = fallback
    if unique and degraded:
        digest = hashlib.sha1(raw_text.encode("utf-8")).hexdigest()[:8]
        return f"{prefix}{text[:90].rstrip('-._')}-{digest}"
    return f"{prefix}{text[:100]}"


def _source_kind(kind: str) -> str:
    return {
        "kb": "doc",
        "web": "doc",
        "notebook": "manual",
        "chat": "manual",
    }.get(str(kind or "").lower(), "manual")


def _block_type(block: Any) -> str:
    value = getattr(block, "type", "")
    return str(getattr(value, "value", value))


def _review_text(text: object) -> str:
    """Normalize one free-text field the way the generator does.

    G3 scans every string in the stored spec, so Book prose with raw ``<``/``>``
    inside math (``$a<b$``) would reject the whole import at ``put_spec``. The
    helpers live in ``generate``, which imports this module first, so they are
    imported lazily here rather than at module load (that would be circular).
    """
    from deeptutor.review.generate import (
        _escape_math,
        _normalize_identifiers,
        _plain_markdown,
    )

    return _normalize_identifiers(
        _escape_math(_plain_markdown(str(text or "").strip()))
    )


def _text_body(block: Any) -> str:
    payload = getattr(block, "payload", {}) or {}
    for key in ("body", "text", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _page_sources(page: Any) -> tuple[dict[str, SourceRef], list[str]]:
    """Build stable sources for a page; never return an empty source list."""
    sources: dict[str, SourceRef] = {}
    ids: list[str] = []
    for block_index, block in enumerate(getattr(page, "blocks", []) or []):
        for anchor_index, anchor in enumerate(
            getattr(block, "source_anchors", []) or []
        ):
            source_id = _safe_id(
                f"{page.id}-{block_index}-{anchor_index}",
                prefix="src-",
                fallback=f"src-page-{block_index}",
                unique=True,
            )
            if source_id in sources:
                continue
            sources[source_id] = SourceRef(
                id=source_id,
                kind=_source_kind(getattr(anchor, "kind", "")),
                title=_review_text(
                    getattr(anchor, "ref", "")
                    or getattr(page, "title", "")
                    or "Book 来源"
                )[:200],
                kb=str(getattr(anchor, "kb_name", "") or ""),
                ref=_review_text(getattr(anchor, "ref", "")),
                snippet=_review_text(
                    getattr(anchor, "snippet", "")
                    or getattr(page, "title", "")
                    or "Book 来源"
                )[:300],
            )
            ids.append(source_id)
    if not ids:
        source_id = _safe_id(
            page.id, prefix="src-", fallback="src-book-page", unique=True
        )
        body = ""
        for block in getattr(page, "blocks", []) or []:
            body = _text_body(block)
            if body:
                break
        sources[source_id] = SourceRef(
            id=source_id,
            kind="manual",
            title=_review_text(getattr(page, "title", "") or "Book 页面")[:200],
            snippet=_review_text(body)[:300] or "Book 页面（无来源锚点）",
        )
        ids.append(source_id)
    return sources, ids


def _explanation_with_answer(
    explanation: str,
    answer: str,
    *,
    needle: str | None = None,
    label: str = "正确答案",
) -> str:
    """Make the explanation name the answer, appending it when it does not.

    Book payloads legitimately omit hints and explanations: the upstream
    ``FlashCardsGenerator`` writes ``hint: ""`` when the model omits one, and a
    ``written`` question may carry no explanation at all. QA gate G4 rejects an
    item whose explanation is empty (and, for choices, one that does not name
    the correct option), so the importer appends the Book's own answer rather
    than inventing an explanation the source never had.

    ``needle`` is the text G4 looks for; it defaults to the answer itself and
    differs for choices, where G4 checks the option text while the appended
    string also carries the label (``B. 感抗与容抗相互抵消``).
    """
    text = explanation.strip()
    answer = answer.strip()
    if not answer:
        return text
    if (answer if needle is None else needle.strip()) in text:
        return text
    if needle is None and answer_coverage(text, answer) >= 0.6:
        # A generated short answer may paraphrase instead of quoting; appending
        # the whole answer would duplicate the explanation shown to the learner.
        return text
    return f"{text}（{label}：{answer}）" if text else f"{label}：{answer}"


def _correct_choice(
    options: dict[str, Any], correct_answer: str
) -> tuple[int, Option] | None:
    answer = str(correct_answer or "").strip()
    if not answer:
        return None
    items = [(str(label), str(text)) for label, text in options.items()]
    for index, (label, text) in enumerate(items):
        if answer in {label, text, f"{label}. {text}", f"{label}、{text}"}:
            return index, Option(label=label, text=text)
    # Some generators store only the correct text in a longer explanation-like
    # string; exact matching is required here rather than guessing.
    return None


def _question_item(
    question: dict[str, Any],
    *,
    page: Any,
    block_id: str,
    concept_id: str,
    source_ids: list[str],
    index: int,
) -> Item | None:
    stem = _review_text(question.get("question") or question.get("stem"))
    if not stem:
        return None
    raw_id = question.get("review_item_id")
    item_id = _safe_id(
        raw_id,
        prefix="",
        fallback=f"item-{_safe_id(block_id, prefix='', fallback='block')}-{index}",
    ) if raw_id else f"item-{_safe_id(block_id, prefix='', fallback='block')}-{index}"
    options = question.get("options")
    correct = str(question.get("correct_answer") or "").strip()
    explanation = _review_text(question.get("explanation"))
    difficulty = str(question.get("difficulty") or "medium").lower()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"

    if isinstance(options, dict) and options:
        # ``_correct_choice`` matches the Book's own strings; escaping happens
        # afterwards so the reference answer still resolves to one option.
        choice = _correct_choice(options, correct)
        if choice is None:
            return None
        choice_index, option = choice
        option_label = _review_text(option.label)
        option_text = _review_text(option.text)
        answer = Answer(choice_index=choice_index)
        explanation = _explanation_with_answer(
            explanation,
            f"{option_label}. {option_text}",
            needle=option_text,
        )
        return Item(
            id=item_id,
            concept_ids=[concept_id],
            kind="single_choice",
            stem=stem,
            options=[
                Option(label=_review_text(label), text=_review_text(text))
                for label, text in options.items()
            ],
            answer=answer,
            explanation=explanation,
            difficulty=difficulty,
            source_ids=source_ids,
        )

    # No options: numeric when the reference answer parses, otherwise a card.
    try:
        value = float(correct)
    except (TypeError, ValueError):
        value = None
    if value is not None:
        answer = Answer(value=value, tolerance=0.0)
        explanation = _explanation_with_answer(explanation, correct)
        return Item(
            id=item_id,
            concept_ids=[concept_id],
            kind="numeric",
            stem=stem,
            answer=answer,
            explanation=explanation,
            difficulty=difficulty,
            source_ids=source_ids,
        )
    reference = _review_text(correct) or "（Book 未提供参考答案）"
    return Item(
        id=item_id,
        concept_ids=[concept_id],
        kind="flashcard",
        stem=stem,
        answer=Answer(text=reference),
        # A written question without a reference answer becomes a card whose
        # answer is explicitly "not provided"; G4 still needs non-empty
        # explanation text, and the placeholder is the answer itself.
        explanation=_explanation_with_answer(
            explanation, reference, label="参考答案"
        ),
        difficulty=difficulty,
        source_ids=source_ids,
    )


def _flashcard_item(
    card: dict[str, Any],
    *,
    page: Any,
    block_id: str,
    concept_id: str,
    source_ids: list[str],
    index: int,
) -> Item | None:
    front = _review_text(card.get("front"))
    back = _review_text(card.get("back"))
    if not front or not back:
        return None
    raw_id = card.get("review_item_id")
    return Item(
        id=(
            _safe_id(
                raw_id,
                prefix="",
                fallback=f"item-{_safe_id(block_id, prefix='', fallback='block')}-card-{index}",
            )
            if raw_id
            else f"item-{_safe_id(block_id, prefix='', fallback='block')}-card-{index}"
        ),
        concept_ids=[concept_id],
        kind="flashcard",
        stem=front,
        answer=Answer(text=back),
        # ``hint`` is optional upstream (FlashCardsGenerator writes "" when the
        # model omitted it). Falling back to the card's own back text keeps G4
        # satisfied without inventing an explanation.
        explanation=_explanation_with_answer(
            _review_text(card.get("hint")), back, label="参考答案"
        ),
        difficulty="medium",
        source_ids=source_ids,
    )


def build_spec_from_pages(
    book: Any,
    pages: Iterable[Any],
    *,
    spec_id: str | None = None,
    mastery_path_id: str = "",
    warnings: list[str] | None = None,
) -> ReviewSpec:
    """Build a ReviewSpec from real ``deeptutor.book`` Book/Page models.

    ``warnings`` is an optional out-list for content skipped during the
    import (for example duplicate explicit ``review_item_id`` values). The
    caller can surface it; the ReviewSpec itself has no place for it before
    QA runs.
    """
    pages = [page for page in pages if getattr(page, "blocks", None)]
    if not pages:
        raise ValueError("book has no pages with blocks")
    if warnings is None:
        warnings = []

    concepts: list[Concept] = []
    sections: list[Section] = []
    items: list[Item] = []
    seen_item_ids: set[str] = set()
    sources: dict[str, SourceRef] = {}
    knowledge_points: set[str] = set()

    for page_index, page in enumerate(sorted(pages, key=lambda row: getattr(row, "order", 0))):
        page_sources, source_ids = _page_sources(page)
        sources.update(page_sources)
        concept_id = _safe_id(
            page.id,
            prefix="concept-",
            fallback=f"concept-page-{page_index}",
            unique=True,
        )
        page_title = _review_text(
            getattr(page, "title", "") or f"Book 页面 {page_index + 1}"
        )
        definition = ""
        for block in getattr(page, "blocks", []) or []:
            definition = _text_body(block)
            if definition:
                break
        definition = _review_text(definition)

        page_items: list[Item] = []
        blocks: list[Any] = []
        for block_index, block in enumerate(getattr(page, "blocks", []) or []):
            block_type = _block_type(block)
            payload = getattr(block, "payload", {}) or {}
            if block_type == "text":
                body = _text_body(block)
                if body:
                    blocks.append(ProseBlock(text=_review_text(body)))
            elif block_type == "quiz":
                questions = payload.get("questions") or []
                if not isinstance(questions, list):
                    continue
                for question_index, question in enumerate(questions):
                    if not isinstance(question, dict):
                        continue
                    item = _question_item(
                        question,
                        page=page,
                        block_id=str(getattr(block, "id", "") or "block"),
                        concept_id=concept_id,
                        source_ids=source_ids,
                        index=question_index,
                    )
                    if item is not None:
                        if item.id in seen_item_ids:
                            warnings.append(
                                f"duplicate item id skipped: {item.id}"
                            )
                            continue
                        seen_item_ids.add(item.id)
                        page_items.append(item)
            elif block_type == "flash_cards":
                cards = payload.get("cards") or []
                if not isinstance(cards, list):
                    continue
                for card_index, card in enumerate(cards):
                    if not isinstance(card, dict):
                        continue
                    item = _flashcard_item(
                        card,
                        page=page,
                        block_id=str(getattr(block, "id", "") or "block"),
                        concept_id=concept_id,
                        source_ids=source_ids,
                        index=card_index,
                    )
                    if item is not None:
                        if item.id in seen_item_ids:
                            warnings.append(
                                f"duplicate item id skipped: {item.id}"
                            )
                            continue
                        seen_item_ids.add(item.id)
                        page_items.append(item)

        if not page_items:
            continue

        # The first question that carries a knowledge_point_id wins. Mixed
        # mappings are not representable by one Review concept yet; the
        # importer keeps the page review-only rather than guessing.
        page_kps = {
            str(question.get("knowledge_point_id") or "").strip()
            for block in getattr(page, "blocks", []) or []
            if _block_type(block) == "quiz"
            for question in ((getattr(block, "payload", {}) or {}).get("questions") or [])
            if isinstance(question, dict) and question.get("knowledge_point_id")
        }
        page_kp = next(iter(page_kps)) if len(page_kps) == 1 else ""
        if page_kp:
            knowledge_points.add(page_kp)

        concepts.append(
            Concept(
                id=concept_id,
                name=page_title[:200],
                definition=definition[:2000] or page_title,
                source_ids=source_ids,
                knowledge_type="concept",
                coverage="kb",
                knowledge_point_id=page_kp if mastery_path_id else "",
            )
        )
        for item in page_items:
            blocks.append(ItemBlock(item_id=item.id))
            items.append(item)
        sections.append(
            Section(
                id=_safe_id(
                    page.id,
                    prefix="sec-",
                    fallback=f"sec-page-{page_index}",
                    unique=True,
                ),
                title=page_title[:200],
                concept_ids=[concept_id],
                order=page_index,
                blocks=blocks,
            )
        )

    if not items:
        raise ValueError("book has no quiz/flashcard items")

    if mastery_path_id:
        if not knowledge_points:
            raise ValueError(
                "mastery_path_id was requested but no Book question carries "
                "knowledge_point_id"
            )
        if len(knowledge_points) > 1:
            raise ValueError(
                "Book page maps to multiple knowledge_point_id values; "
                "split the book or link it manually"
            )
    effective_mastery = mastery_path_id if knowledge_points else ""

    language = str(getattr(book, "language", "zh") or "zh")
    if language not in {"zh", "en"}:
        language = "zh"
    return ReviewSpec(
        id=spec_id
        or _safe_id(getattr(book, "id", ""), prefix="book-", fallback="book-import"),
        book_id=str(getattr(book, "id", "") or "") or None,
        mastery_path_id=effective_mastery,
        kb_ids=[str(kb) for kb in (getattr(book, "knowledge_bases", []) or [])],
        language=language,
        title=_review_text(getattr(book, "title", "") or "Book 复习")[:200],
        plan=ReviewPlan(),
        concepts=concepts,
        sections=sections,
        items=items,
        sources=sources,
    )


def load_book_pages(book_id: str) -> tuple[Any, list[Any]]:
    """Read a Book and its pages through the public Book storage API."""
    from deeptutor.book.storage import BookStorage

    storage = BookStorage()
    book = storage.load_book(book_id)
    if book is None:
        raise ValueError(f"book not found: {book_id}")
    pages = storage.list_pages(book_id)
    return book, pages


__all__ = ["build_spec_from_pages", "load_book_pages"]
