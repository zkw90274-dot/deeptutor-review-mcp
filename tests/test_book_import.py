from __future__ import annotations

from fastapi.testclient import TestClient

from deeptutor.book.models import Block, BlockType, Book, Page
from deeptutor.review.book_import import build_spec_from_pages
from deeptutor.review.fixture import build_fixture_spec
from deeptutor.review.qa import run_gates


def _book_page() -> tuple[Book, Page]:
    book = Book(
        id="bk-test",
        title="测试书",
        language="zh",
        knowledge_bases=["kb-test"],
    )
    text = Block(
        id="blk-text",
        type=BlockType.TEXT,
        payload={"body": "串联谐振时感抗与容抗相互抵消。"},
    )
    quiz = Block(
        id="blk-quiz",
        type=BlockType.QUIZ,
        payload={
            "questions": [
                {
                    "question_id": "q1",
                    "review_item_id": "item-book-1",
                    "knowledge_point_id": "kp-book",
                    "question": "串联谐振时回路阻抗为什么最小？",
                    "question_type": "choice",
                    "options": {"A": "电阻被短路", "B": "感抗与容抗相互抵消"},
                    "correct_answer": "B",
                    "explanation": "B. 感抗与容抗相互抵消。",
                    "difficulty": "easy",
                }
            ]
        },
    )
    cards = Block(
        id="blk-cards",
        type=BlockType.FLASH_CARDS,
        payload={"cards": [{"front": "品质因数 Q 的定义？", "back": "Q = ω0 L / R", "hint": "串联回路"}]},
    )
    page = Page(
        id="pg-1",
        book_id="bk-test",
        title="第一章 谐振",
        order=0,
        blocks=[text, quiz, cards],
    )
    return book, page


def test_build_spec_from_book_preserves_linkage_and_items() -> None:
    book, page = _book_page()
    spec = build_spec_from_pages(
        book, [page], spec_id="book-test", mastery_path_id="path-test"
    )

    assert spec.id == "book-test"
    assert spec.book_id == "bk-test"
    assert spec.mastery_path_id == "path-test"
    assert [item.id for item in spec.items] == ["item-book-1", "item-blk-cards-card-0"]
    assert spec.concepts[0].knowledge_point_id == "kp-book"
    assert spec.items[0].kind == "single_choice"
    assert spec.items[1].kind == "flashcard"
    assert run_gates(spec).passed


def test_build_spec_requires_kp_when_mastery_path_requested() -> None:
    book, page = _book_page()
    page.blocks[1].payload["questions"][0].pop("knowledge_point_id")
    try:
        build_spec_from_pages(
            book, [page], spec_id="book-test", mastery_path_id="path-test"
        )
    except ValueError as exc:
        assert "knowledge_point_id" in str(exc)
    else:
        raise AssertionError("expected missing knowledge_point_id to fail")


def test_flashcard_without_hint_still_passes_answer_gate() -> None:
    """Upstream FlashCardsGenerator writes ``hint: ""``; G4 must still pass.

    The importer may only reuse Book content, so the card's own back text is
    the explanation fallback — not an invented sentence.
    """
    book, page = _book_page()
    page.blocks[2].payload["cards"][0]["hint"] = ""
    spec = build_spec_from_pages(book, [page], spec_id="book-test")

    card = next(item for item in spec.items if item.kind == "flashcard")
    assert card.answer.text in card.explanation
    report = run_gates(spec)
    assert report.passed, [gate.details for gate in report.gates if not gate.passed]


def test_written_question_without_explanation_passes_answer_gate() -> None:
    """A Book written answer often carries no explanation at all."""
    book = Book(id="bk-written", title="书面题", language="zh")
    page = Page(
        id="pg-written",
        book_id="bk-written",
        title="第一章",
        order=0,
        blocks=[
            Block(
                id="blk-text",
                type=BlockType.TEXT,
                payload={"body": "串联谐振时感抗与容抗相互抵消。"},
            ),
            Block(
                id="blk-quiz",
                type=BlockType.QUIZ,
                payload={
                    "questions": [
                        {
                            "question_id": "w1",
                            "question": "解释串联谐振。",
                            "question_type": "written",
                            "correct_answer": "感抗与容抗抵消",
                            "explanation": "",
                        }
                    ]
                },
            ),
        ],
    )
    spec = build_spec_from_pages(book, [page], spec_id="book-written")

    item = next(item for item in spec.items if item.kind == "flashcard")
    assert "感抗与容抗抵消" in item.explanation
    report = run_gates(spec)
    assert report.passed, [gate.details for gate in report.gates if not gate.passed]


def test_choice_explanation_that_omits_the_option_gets_the_book_answer() -> None:
    book, page = _book_page()
    page.blocks[1].payload["questions"][0]["explanation"] = "见教材对应章节。"
    spec = build_spec_from_pages(book, [page], spec_id="book-test")

    item = next(item for item in spec.items if item.id == "item-book-1")
    assert "感抗与容抗相互抵消" in item.explanation
    assert run_gates(spec).passed


def test_multi_select_question_is_skipped_not_guessed() -> None:
    """``correct_answer="AB"`` matches no single option, so the item is dropped."""
    book, page = _book_page()
    page.blocks[1].payload["questions"].append(
        {
            "question_id": "q-multi",
            "question": "以下哪些属于谐振特征？",
            "question_type": "multiple_choice",
            "options": {"A": "阻抗最小", "B": "电流最大"},
            "correct_answer": "AB",
            "explanation": "A、B 正确。",
        }
    )
    spec = build_spec_from_pages(book, [page], spec_id="book-test")

    assert [item.id for item in spec.items] == [
        "item-book-1",
        "item-blk-cards-card-0",
    ]
    assert run_gates(spec).passed


def test_import_is_deterministic_across_calls() -> None:
    book, page = _book_page()
    first = build_spec_from_pages(book, [page], spec_id="book-test")
    second = build_spec_from_pages(book, [page], spec_id="book-test")

    assert [item.id for item in first.items] == [item.id for item in second.items]
    assert first.content_hash == second.content_hash


def test_import_book_endpoint_stores_spec(
    client: TestClient, monkeypatch
) -> None:
    from deeptutor.review import book_import

    book, page = _book_page()
    monkeypatch.setattr(book_import, "load_book_pages", lambda book_id: (book, [page]))

    response = client.post(
        "/api/review/imports/book",
        json={"book_id": "bk-test", "spec_id": "book-api"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["spec_id"] == "book-api"
    assert body["items"] == 2
    assert client.get("/api/review/book-api/today").status_code == 200


def test_book_attempt_maps_correctness(client: TestClient) -> None:
    spec = build_fixture_spec()
    payload = spec.model_dump(mode="json")
    client.put(f"/api/review/specs/{spec.id}", json=payload)
    today = client.get(f"/api/review/{spec.id}/today").json()

    correct = client.post(
        f"/api/review/{spec.id}/book-attempt",
        json={
            "item_id": "item-resonance-choice",
            "is_correct": True,
            "session_id": today["session_id"],
            "submission_id": "book-correct-1",
            "answer": 1,
        },
    )
    assert correct.status_code == 200
    assert correct.json()["rating"] == "good"
    assert correct.json()["result"] == "correct"

    wrong = client.post(
        f"/api/review/{spec.id}/book-attempt",
        json={
            "item_id": "item-resonance-choice",
            "is_correct": False,
            "session_id": today["session_id"],
            "submission_id": "book-wrong-1",
            "answer": 0,
        },
    )
    assert wrong.status_code == 200
    assert wrong.json()["rating"] == "again"
    assert wrong.json()["result"] == "incorrect"


def test_duplicate_explicit_item_id_is_skipped_with_a_warning() -> None:
    """N-11: a duplicate ``review_item_id`` must not fail the whole import."""
    book, page = _book_page()
    quiz = page.blocks[1]
    first = quiz.payload["questions"][0]
    quiz.payload["questions"].append(dict(first, question_id="q1-duplicate"))
    warnings: list[str] = []

    spec = build_spec_from_pages(
        book, [page], spec_id="book-test", warnings=warnings
    )

    assert [item.id for item in spec.items].count("item-book-1") == 1
    assert any("duplicate item id" in warning for warning in warnings)
