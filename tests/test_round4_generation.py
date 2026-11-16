"""Round-4 regressions: comparison-operator escaping on both import paths.

R4 found that ``concentration`` (which becomes the concept name) and the whole
Book import path bypassed the ``_plain_markdown`` -> ``_escape_math`` ->
``_normalize_identifiers`` chain. G3 scans every string in the spec, so a raw
``$f_s<f_p$`` in either path rejected the entire pack at ``put_spec``.

No real LLM and no real DB: generation is a fake coordinator payload and the
store lives under ``tmp_path``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from deeptutor.book.models import Block, BlockType, Book, Page
from deeptutor.review import book_import, generate
from deeptutor.review.book_import import build_spec_from_pages
from deeptutor.review.generate import Chapter, generate_spec_from_kb
from deeptutor.review.qa import run_gates
from deeptutor.review.service import ReviewService

CHAPTER = "第1章_绪论"
KB = "hf"


def _patch_chapters(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    monkeypatch.setattr(
        generate,
        "infer_chapters",
        lambda kb_name: [Chapter(name=name, doc_count=1) for name in names],
    )


def _patch_generation(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    from deeptutor.agents.question.coordinator import AgentCoordinator

    async def fake(self: Any, **kwargs: Any) -> dict[str, Any]:
        return payload

    monkeypatch.setattr(AgentCoordinator, "generate_from_topic", fake)


def _payload(*qa_pairs: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": True,
        "completed": len(qa_pairs),
        "failed": 0,
        "mode": "topic",
        "results": [{"qa_pair": pair, "metadata": None} for pair in qa_pairs],
    }


def _qa(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "question_id": "q",
        "question": "题目",
        "question_type": "choice",
        "options": {"A": "甲", "B": "乙"},
        "correct_answer": "A",
        "explanation": "",
        "difficulty": "medium",
        "concentration": "知识点",
    }
    base.update(overrides)
    return base


def _assert_gates_pass(spec: Any) -> None:
    report = run_gates(spec)
    assert report.passed, [gate.model_dump() for gate in report.gates if not gate.passed]


def test_concentration_with_raw_comparison_survives_to_storage(
    service: ReviewService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``concentration`` of ``$f_s<f_p$`` must not reject the whole pack."""
    _patch_chapters(monkeypatch, CHAPTER)
    _patch_generation(
        monkeypatch,
        _payload(
            _qa(
                question="比较信号频率与谐振频率。",
                correct_answer="A",
                concentration="$f_s<f_p$",
            ),
        ),
    )

    result = generate_spec_from_kb(kb_name=KB, chapter=CHAPTER, num_questions=1)
    spec = result.spec
    assert spec is not None

    name = spec.concepts[0].name
    assert r"\lt{}" in name
    assert "<" not in name
    _assert_gates_pass(spec)

    service.put_spec(spec)
    stored = service.store.get_spec(spec.id)
    assert r"\lt{}" in stored.concepts[0].name
    assert "<" not in stored.concepts[0].name


def test_math_in_the_chapter_name_is_escaped_in_titles_and_sources(
    service: ReviewService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Section title, spec title, source ref and the concept fallback all come
    from the raw chapter name and all reach G3."""
    chapter = r"第1章_$f_s<f_p$"
    _patch_chapters(monkeypatch, chapter)
    _patch_generation(
        monkeypatch,
        _payload(_qa(question="题目", correct_answer="A", concentration=None)),
    )

    spec = generate_spec_from_kb(kb_name=KB, chapter=chapter, num_questions=1).spec
    assert spec is not None

    expected = r"$f_s\lt{}f_p$"
    assert spec.concepts[0].name == f"第1章_{expected}"
    assert spec.sections[0].title == f"第1章_{expected}"
    assert spec.title == f"{KB} · 第1章_{expected} 复习包"
    source = next(iter(spec.sources.values()))
    assert source.ref == f"第1章_{expected}"
    assert r"\lt{}" in source.id
    _assert_gates_pass(spec)

    service.put_spec(spec)
    stored = service.store.get_spec(spec.id)
    assert "<" not in json.dumps(stored.model_dump(mode="json"), ensure_ascii=False)


def _book_with_math() -> tuple[Book, Page]:
    book = Book(
        id="bk-math",
        title="比较器 $a<b$ 讲义",
        language="zh",
        knowledge_bases=["kb-math"],
    )
    text = Block(
        id="blk-text",
        type=BlockType.TEXT,
        payload={"body": "当 $a<b$ 时比较器输出高电平。"},
    )
    quiz = Block(
        id="blk-quiz",
        type=BlockType.QUIZ,
        payload={
            "questions": [
                {
                    "question_id": "q1",
                    "review_item_id": "item-math-1",
                    "question": "当 $a<b$ 时输出是什么？",
                    "question_type": "choice",
                    "options": {"A": "输出 $V_{high}$", "B": "输出 $a<b$ 时的低电平"},
                    "correct_answer": "A",
                    "explanation": "因为 $a<b$，所以选 A。",
                    "difficulty": "medium",
                }
            ]
        },
    )
    cards = Block(
        id="blk-cards",
        type=BlockType.FLASH_CARDS,
        payload={
            "cards": [
                {"front": "何时 $a<b$ 成立？", "back": "$b>a$", "hint": "比较大小"}
            ]
        },
    )
    page = Page(
        id="pg-math",
        book_id="bk-math",
        title="第一章_比较器 $a<b$",
        order=0,
        blocks=[text, quiz, cards],
    )
    return book, page


def test_book_import_escapes_math_in_every_free_text_field(
    service: ReviewService,
) -> None:
    book, page = _book_with_math()
    spec = build_spec_from_pages(book, [page], spec_id="book-math")

    item = next(item for item in spec.items if item.id == "item-math-1")
    assert r"$a\lt{}b$" in item.stem
    assert any(r"$a\lt{}b$" in option.text for option in item.options)
    assert r"$a\lt{}b$" in item.explanation
    card = next(item for item in spec.items if item.kind == "flashcard")
    assert r"\gt{}" in card.answer.text
    assert r"\lt{}" in spec.concepts[0].name
    assert r"\lt{}" in spec.title
    _assert_gates_pass(spec)

    service.put_spec(spec)
    stored_json = json.dumps(
        service.store.get_spec(spec.id).model_dump(mode="json"), ensure_ascii=False
    )
    assert r"\lt{}" in stored_json
    assert r"\gt{}" in stored_json
    # The Book source snippets/prose are scanned too; none may keep raw "<".
    assert r"$a<b$" not in stored_json


def test_book_import_endpoint_accepts_math(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HTTP path calls ``put_spec`` (and therefore G3) on the same spec."""
    book, page = _book_with_math()
    monkeypatch.setattr(book_import, "load_book_pages", lambda book_id: (book, [page]))

    response = client.post(
        "/api/review/imports/book",
        json={"book_id": "bk-math", "spec_id": "book-math-api"},
    )
    assert response.status_code == 201, response.text

    from deeptutor.review.storage import ReviewStore

    stored = ReviewStore().get_spec("book-math-api")
    stored_json = json.dumps(stored.model_dump(mode="json"), ensure_ascii=False)
    assert r"\lt{}" in stored_json
    assert r"$a<b$" not in stored_json


def test_normalization_is_idempotent() -> None:
    samples = [
        "$f_s<f_p$",
        "当 $a<b$ 且 $|x|>1$ 时",
        "**粗体** $a<b$",
        "V_CC 与 I_c1m^2",
        r"已经是 $\lt{}$、$\vert{}\gt{}$ 的写法。",
    ]
    for sample in samples:
        once = generate._clean_text(sample)
        assert generate._clean_text(once) == once, sample
        once_book = book_import._review_text(sample)
        assert book_import._review_text(once_book) == once_book, sample
        assert once == once_book, sample


def test_plain_text_is_unchanged() -> None:
    plain = "串联谐振时感抗与容抗相互抵消，回路阻抗最小。"
    assert generate._clean_text(plain) == plain
    assert book_import._review_text(plain) == plain
