from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from deeptutor.review.correction import answer_text
from deeptutor.review.models import QualityReport, ReviewSpec
from deeptutor.review import provenance_text as text_match

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".tex", ".csv", ".json", ".html", ".htm", ".rst"}
MAX_DOC_CHARS = 300_000
MAX_CHAPTER_CHARS = 1_200_000
WINDOW = 300
_HEAD = re.compile(r"(?m)^#\s+([^\r\n]+)")
_GENERIC = {"正确", "错误", "不正确", "不成立", "以上都对", "以上都不对", "true", "false", "yes", "no", "all of the above", "none of the above"}
_POLARITY = re.compile(r"不|无|未|非|禁止|\b(?:not|never|without|no)\b", re.IGNORECASE)
_WRONG_CUE = re.compile(r"[❌✗✘]|(?:例)?错误(?:说法|示例|公式|推导)|反例|\b(?:incorrect|counterexample)\b", re.IGNORECASE)
_CONTRASTS = (("超前", "滞后"), ("增大", "减小"), ("升高", "降低"), ("上升", "下降"),
              ("高于", "低于"), ("大于", "小于"), ("感性", "容性"), ("同相", "反相"))
WARNINGS = {
    "legacy": "这是旧包，未做来源核对。",
    "draft": "来源支持或题目检查不足，默认为草稿。",
    "rejected": "题包未通过质量检查。",
}


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def chapter_paths(kb: str, chapter: str, *, audit: bool = False, kb_root: Path | None = None) -> list[Path]:
    from deeptutor.review.generate import _chapter_marker, _knowledge_bases_root, _validated_kb_name

    root = (kb_root if kb_root is not None else _knowledge_bases_root()).resolve()
    raw = root / _validated_kb_name(kb) / "raw"
    if not raw.is_dir() or not raw.resolve().is_relative_to(root):
        return []
    target = _chapter_marker(chapter)
    primary, secondary = [], []
    for path in sorted(raw.iterdir()):
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        if not path.resolve().is_relative_to(raw.resolve()):
            continue
        marker = _chapter_marker(path.stem)
        if audit and target is not None and marker is None and path.suffix.lower() in {".md", ".markdown"}:
            try:
                with path.open(encoding="utf-8-sig") as source:
                    heading = _HEAD.search(source.read(4096))
            except (OSError, UnicodeError):
                continue
            if heading and _chapter_key(heading[1]) == _chapter_key(chapter):
                secondary.append(path)
        if target is None:
            if chapter == "其他" and marker is None:
                primary.append(path)
        elif marker and marker[:2] == target[:2] and (
            normalize(marker[3]) == normalize(target[3])
            or (audit and _chapter_key(marker[3]) == _chapter_key(target[3]))
        ):
            (primary if marker[4] or marker[0] != "chapter" else secondary).append(path)
    return primary or secondary


def _chapter_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", normalize(text).casefold())


def chapter_documents(kb: str, chapter: str, *, kb_root: Path | None = None) -> list[tuple[str, str]]:
    documents = []
    remaining = MAX_CHAPTER_CHARS
    for path in chapter_paths(kb, chapter, audit=True, kb_root=kb_root):
        if not remaining:
            break
        try:
            with path.open(encoding="utf-8", newline="") as source:
                text = source.read(min(MAX_DOC_CHARS, remaining) + 1)
        except (OSError, UnicodeError):
            continue
        if len(text) > min(MAX_DOC_CHARS, remaining):
            boundary = max(text.rfind("\n", 0, len(text) - 1), text.rfind("。", 0, len(text) - 1))
            text = text[:boundary + 1]
        remaining -= len(text)
        if text.strip():
            documents.append((path.name, text))
    return documents


def source_chapter(spec: ReviewSpec, source_id: str) -> tuple[str, str] | None:
    source = spec.sources.get(source_id)
    if source is None or not (source.kb or source.id.startswith("kb:")):
        return None
    prefix, separator, suffix = source.title.partition("·")
    kb = source.kb or (prefix.strip() if separator else next(iter(spec.kb_ids), ""))
    chapter = source.ref or (suffix.strip() if separator else source.title)
    return (kb, chapter) if kb and chapter else None


def _anchor(
    doc: str, text: str, start: int, end: int, level: str, reason: str, terms: list[str],
) -> dict[str, Any]:
    end = min(end, start + WINDOW)
    quote = text[start:end]
    return {
        "doc": doc, "char_start": start, "char_end": end, "quote": quote,
        "quote_hash": hashlib.sha256(quote.encode()).hexdigest(), "support_level": level,
        "support_reason": reason, "matched_terms": [term[:96] for term in terms[:8]],
    }


def _none(reason: str) -> dict[str, Any]:
    return {"support_level": "none", "support_reason": reason, "matched_terms": []}


def _substantive(statement: str, prose: str, terms: set[str]) -> bool:
    key = _chapter_key(statement)
    return key not in terms and statement.casefold() not in _GENERIC and (
        len(re.findall(r"[\u3400-\u9fff]", prose)) >= 8 or len(text_match.words(prose)) >= 3
    )


def _contradictory(claim: str, source: str) -> bool:
    return bool(_POLARITY.search(claim)) != bool(_POLARITY.search(source)) or any(
        (left in claim and right in source and left not in source)
        or (right in claim and left in source and right not in source)
        for left, right in _CONTRASTS
    )


def match_anchor(
    documents: list[tuple[str, str]], candidates: list[str], terms: list[str], tokens: list[str]
) -> dict[str, Any]:
    if not documents:
        return _none("chapter_unavailable")
    term_labels = {key: term for term in terms if len(key := _chapter_key(term)) >= 4}
    term_keys = set(term_labels)
    direct_formulas = text_match.formulas(candidates[0]) if candidates else set()
    if candidates and text_match.prose(candidates[0]).strip():
        direct_formulas = set()
    cues = set().union(*(text_match.formulas(token) for token in tokens))
    claims = []
    exact_claims = set()
    for candidate in dict.fromkeys(candidates):
        for _, _, sentence in text_match.passages(candidate):
            if len(sentence) > WINDOW:
                continue
            statement = text_match.statement(sentence)
            prose = text_match.prose(sentence)
            values = text_match.quantities(sentence)
            equations = text_match.formulas(sentence)
            pairs = text_match.grams(prose, 2)
            triples = text_match.grams(prose, 3)
            if _substantive(statement, prose, term_keys):
                exact_claims.add(statement)
            elif (not values and not equations) or len(triples) < 3:
                continue
            claims.append((prose, values, equations, pairs, triples))
    best_score = 0.0
    best = _none("insufficient_overlap")
    term_anchor = None
    for doc, text in documents:
        for start, end, excerpt in text_match.passages(text):
            if end - start > WINDOW:
                continue
            statement = text_match.statement(excerpt)
            wrong_cue = bool(_WRONG_CUE.search(excerpt))
            if statement in exact_claims and not wrong_cue:
                return _anchor(doc, text, start, end, "exact", "normalized_statement", [statement])
            prose = text_match.prose(excerpt)
            equations = text_match.formulas(excerpt)
            if term_anchor is None and equations and equations == direct_formulas and not (wrong_cue or _POLARITY.search(prose)):
                term_anchor = _anchor(doc, text, start, end, "term", "formula_answer", sorted(equations))
            quantities = text_match.quantities(excerpt)
            pairs, triples = text_match.grams(prose, 2), text_match.grams(prose, 3)
            for claim, values, formulae, claim_pairs, claim_triples in claims:
                shared_pairs, shared_triples = pairs & claim_pairs, triples & claim_triples
                pair_coverage = len(shared_pairs) / max(1, len(claim_pairs))
                triple_coverage = len(shared_triples) / max(1, len(claim_triples))
                score = (pair_coverage + triple_coverage) / 2
                if len(shared_triples) < 2 or score < 0.35:
                    continue
                contradictory = wrong_cue or _contradictory(claim, prose)
                quantity_conflict = bool(values and quantities and values.keys() != quantities.keys())
                formula_conflict = bool(formulae and equations and formulae != equations)
                context = pair_coverage >= 0.45 and triple_coverage >= 0.35
                if term_anchor is None and context and not (contradictory or quantity_conflict or formula_conflict):
                    if values and values.keys() == quantities.keys():
                        term_anchor = _anchor(doc, text, start, end, "term", "quantity_and_context", sorted(values))
                    elif formulae and formulae == equations:
                        term_anchor = _anchor(doc, text, start, end, "term", "formula_and_context", sorted(formulae))
                if score > best_score:
                    best_score = score
                    reason = "conflicting_cues" if contradictory or quantity_conflict or formula_conflict else "ngram_overlap"
                    best = _anchor(doc, text, start, end, "weak", reason, sorted(shared_triples))
            if best["support_level"] == "none" and not wrong_cue:
                if cues & equations:
                    best = _anchor(doc, text, start, end, "weak", "formula_cue_only", sorted(cues & equations))
                elif term_labels:
                    excerpt_key = _chapter_key(prose)
                    matched = [term_labels[key] for key in sorted(term_keys) if key in excerpt_key]
                    if matched:
                        best = _anchor(doc, text, start, end, "weak", "concept_cue_only", matched)
    return term_anchor or best


def assess(
    spec: ReviewSpec, gates: QualityReport, *, kb_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    by_concept = {concept.id: concept for concept in spec.concepts}
    cache: dict[tuple[str, str], list[tuple[str, str]]] = {}
    anchors, stems, duplicates, leaks = {}, {}, [], []
    for item in spec.items:
        documents = []
        chapters = {}
        answer = answer_text(item)
        for source_id in dict.fromkeys(item.source_ids):
            key = source_chapter(spec, source_id)
            if key is None:
                continue
            kb, chapter = key
            if key not in cache:
                cache[key] = chapter_documents(kb, chapter, kb_root=kb_root) if kb_root else chapter_documents(kb, chapter)
            for filename, text in cache[key]:
                doc = f"{kb}/{filename}"
                documents.append((doc, text))
                chapters[doc] = chapter
        concepts = [by_concept[concept_id] for concept_id in item.concept_ids]
        terms = [concept.name for concept in concepts]
        candidates = [answer, item.explanation]
        tokens = [item.stem, *(f"${formula.latex.strip('$')}$" for concept in concepts for formula in concept.formulas)]
        anchors[item.id] = match_anchor(documents, candidates, terms, tokens)
        if "doc" in anchors[item.id]:
            anchors[item.id]["chapter"] = chapters[anchors[item.id]["doc"]]
        stem = normalize(item.stem).casefold()
        if stem in stems:
            duplicates.append([stems[stem], item.id])
        else:
            stems[stem] = item.id
        normalized_answer = normalize(answer).casefold()
        if len(normalized_answer) >= 4 and normalized_answer in stem:
            leaks.append(item.id)
    support = {item_id: anchor["support_level"] for item_id, anchor in anchors.items()}
    supported = sum(level in {"exact", "term"} for level in support.values())
    state = "draft"
    if not gates.passed:
        state = "rejected"
    elif supported == len(spec.items) and not duplicates and not leaks:
        state = "verified"
    return {
        "state": state, "content_hash": spec.content_hash, "gates": gates.model_dump(mode="json"),
        "support": support, "anchored_items": supported, "items": len(spec.items),
        "support_details": {
            item_id: {field: anchor[field] for field in ("support_reason", "matched_terms")}
            for item_id, anchor in anchors.items()
        },
        "coverage": supported / len(spec.items) if spec.items else 0,
        "duplicate_stems": duplicates, "answer_leaks": leaks, "generated_at": time.time(),
    }, anchors
