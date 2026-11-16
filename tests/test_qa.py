from __future__ import annotations

from typing import Callable

from deeptutor.review.fixture import build_fixture_spec
from deeptutor.review.models import ReviewSpec
from deeptutor.review.qa import run_gates


def _altered(mutate: Callable[[dict], None]) -> ReviewSpec:
    payload = build_fixture_spec().model_dump(mode="json")
    mutate(payload)
    payload["content_hash"] = ""
    return ReviewSpec.model_validate(payload)


def test_raw_less_than_in_math_fails() -> None:
    spec = _altered(lambda p: p["sections"][0]["blocks"][0].__setitem__("text", "$x < y$"))
    report = run_gates(spec)
    gate = next(g for g in report.gates if g.gate == "G3-math-escape")
    assert gate.passed is False
    assert any("<" in detail for detail in gate.details)


def test_raw_pipe_in_math_fails() -> None:
    spec = _altered(lambda p: p["sections"][0]["blocks"][0].__setitem__("text", "$a|b$"))
    report = run_gates(spec)
    gate = next(g for g in report.gates if g.gate == "G3-math-escape")
    assert gate.passed is False
    assert any("\\vert" in detail for detail in gate.details)


def test_unbalanced_dollar_fails_math_balance_gate() -> None:
    # MathJax does not report an odd "$" as an error: it typesets the pairs and
    # leaves the stray delimiter as literal text. The gate has to catch it.
    spec = _altered(lambda p: p["sections"][0]["blocks"][0].__setitem__("text", "增益 $A_v 与带宽"))
    report = run_gates(spec)
    gate = next(g for g in report.gates if g.gate == "G10-math-balance")
    assert gate.passed is False
    assert any("$" in detail for detail in gate.details)


def test_balanced_dollar_passes_math_balance_gate() -> None:
    spec = _altered(lambda p: p["sections"][0]["blocks"][0].__setitem__("text", "增益 $A_v$ 与带宽"))
    report = run_gates(spec)
    gate = next(g for g in report.gates if g.gate == "G10-math-balance")
    assert gate.passed is True


def test_executable_content_fails_injection_gate() -> None:
    spec = _altered(
        lambda p: p["sections"][0]["blocks"][0].__setitem__(
            "text", '<img src=x onerror="alert(1)">'
        )
    )
    report = run_gates(spec)
    gate = next(g for g in report.gates if g.gate == "G6-injection")
    assert gate.passed is False


def test_external_url_fails_offline_gate() -> None:
    spec = _altered(
        lambda p: p["sources"]["src-hf-2-1"].__setitem__(
            "snippet", "see https://example.com/chapter"
        )
    )
    report = run_gates(spec)
    gate = next(g for g in report.gates if g.gate == "G8-offline")
    assert gate.passed is False


def test_client_quality_warning_cannot_bypass_offline_gate() -> None:
    def mutate(p: dict) -> None:
        p["quality"]["warnings"] = ["see https://example.com/private"]

    spec = _altered(mutate)
    report = run_gates(spec)
    gate = next(g for g in report.gates if g.gate == "G8-offline")
    assert gate.passed is False


def test_answer_explanation_conflict_fails() -> None:
    spec = _altered(
        lambda p: p["items"][0].__setitem__("explanation", "与选项无关的解释")
    )
    report = run_gates(spec)
    gate = next(g for g in report.gates if g.gate == "G4-answer-key")
    assert gate.passed is False


def test_kb_concept_without_source_fails() -> None:
    def mutate(p: dict) -> None:
        p["concepts"][0]["source_ids"] = []
        p["concepts"][0]["formulas"] = []
        p["concepts"][0]["coverage"] = "kb"

    spec = _altered(mutate)
    report = run_gates(spec)
    gate = next(g for g in report.gates if g.gate == "G2-sources")
    assert gate.passed is False


def test_multi_concept_item_is_warned_not_silently_double_scheduled() -> None:
    def mutate(p: dict) -> None:
        p["items"][0]["concept_ids"] = [
            p["concepts"][0]["id"],
            p["concepts"][1]["id"],
        ]

    spec = _altered(mutate)
    report = run_gates(spec)
    assert report.passed is True
    assert any("multiple concepts" in warning for warning in report.warnings)
    assert spec.items[0].id not in [
        item.id for item in spec.items_for_concept(spec.concepts[1].id)
    ]


def test_warnings_are_preserved_and_failures_are_reported() -> None:
    def mutate(p: dict) -> None:
        p["sections"][0]["blocks"][0]["text"] = "<script>alert(1)</script>"
        p["quality"]["warnings"] = ["人工复核：生成器标记的低置信内容"]

    spec = _altered(mutate)
    report = run_gates(spec)
    assert report.passed is False
    assert "人工复核：生成器标记的低置信内容" in report.warnings
    assert any(warning.startswith("G6-injection") for warning in report.warnings)
