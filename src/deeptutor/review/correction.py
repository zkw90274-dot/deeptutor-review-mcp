"""Deterministic three-step correction payload shared by API and export.

ReviewSpec has no separate ``key_points`` field. Instead of adding one (which
would change the content hash of every existing spec), the low-cost restatement
is derived from content that is already present:

1. the item's correct answer;
2. other concept names;
3. a final "none of the above" fallback.

Other items' answers are deliberately **not** reused as distractors: doing so
put another question's full answer (sometimes several paragraphs) in front of
a learner who had just missed this one, spoiling the rest of the pack. Concept
names are neutral labels and still make a plausible restatement choice.

The order is stable per item (sha256 of the item id), so an export built twice
from the same spec is byte-identical.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from deeptutor.review.models import Item, ReviewSpec

#: A distractor is a label, not a second worked solution. Generated concept
#: names can be long sentences; clip them so an option cannot smuggle a whole
#: conclusion into the correction payload.
CORRECTION_LABEL_CHARS = 80

FENCE_LINE = re.compile(r"^\s*(```+|~~~+)")


def _fenced_summary(text: str) -> str:
    """Turn a fenced answer into a one-line label.

    A short-answer item can answer with a whole MATLAB listing. The correction
    option is a button-sized label, so leaving the fence markers in showed
    learners raw ``` lines and code `*` operators. Keep prose outside fences
    when there is any; otherwise name the first code line.
    """
    prose: list[str] = []
    code: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if FENCE_LINE.match(line):
            in_fence = not in_fence
            continue
        (code if in_fence else prose).append(line)
    outside = " ".join(part.strip() for part in prose if part.strip())
    if outside:
        return outside
    first = next((line.strip() for line in code if line.strip()), "")
    first = first.lstrip("%#/ \t")
    return f"代码：{first}" if first else "代码示例"


def _open_math_start(text: str) -> int | None:
    """Return the index of an inline/display math opener that never closes."""
    index = 0
    while index < len(text):
        if text.startswith("$$", index):
            close = text.find("$$", index + 2)
            if close < 0:
                return index
            index = close + 2
        elif text[index] == "$":
            close = text.find("$", index + 1)
            if close < 0:
                return index
            index = close + 1
        else:
            index += 1
    return None


def _label(text: str) -> str:
    text = text.strip()
    if any(FENCE_LINE.match(line) for line in text.splitlines()):
        text = _fenced_summary(text)
    # Answers are prose blocks with blank lines. The option is a button-sized
    # label, so fold every whitespace run into one space before measuring.
    text = " ".join(text.split())
    if len(text) <= CORRECTION_LABEL_CHARS:
        return text
    cut = text[: CORRECTION_LABEL_CHARS - 1]
    opener = _open_math_start(cut)
    if opener is not None:
        # A half formula would reach the page as raw TeX (and an even number of
        # ``$`` can still be a dangling ``$$``), so stop before it.
        cut = cut[:opener]
    cut = cut.rstrip(" \t、，,;；")
    return (cut + "…") if cut else "（公式见解析）"


def answer_text(item: Item) -> str:
    if item.kind == "single_choice" and item.answer.choice_index is not None:
        option = item.options[item.answer.choice_index]
        return f"{option.label}. {option.text}"
    if item.kind == "numeric" and item.answer.value is not None:
        return f"{item.answer.value:g}"
    return item.answer.text


def build_correction(spec: ReviewSpec, item: Item) -> dict[str, Any]:
    correct = _label(answer_text(item).strip()) or "（无标准答案）"
    other_answers = [
        answer_text(other).strip()
        for other in spec.items
        if other.id != item.id and answer_text(other).strip()
    ]
    candidates: list[str] = []
    for concept in spec.concepts:
        if concept.id in item.concept_ids:
            continue
        text = _label(concept.name)
        if not text or text == correct or text in candidates:
            continue
        # A concept name that spells out another item's answer would reintroduce
        # the spoiler this payload is meant to avoid.
        if any(answer and answer in text for answer in other_answers):
            continue
        candidates.append(text)
    choices = [correct, *candidates[:2]]
    filler = "以上都不对"
    while len(choices) < 2:
        # The correct answer itself may be "以上都不是", so a fixed filler can
        # collide with it and produce two identical, ungradable choices.
        if filler not in choices:
            choices.append(filler)
        else:
            filler = filler + "（请再想一步）"
    offset = hashlib.sha256(item.id.encode("utf-8")).digest()[0] % len(choices)
    rotated = choices[offset:] + choices[:offset]
    correct_index = rotated.index(correct)
    options = [
        {"label": chr(ord("A") + index), "text": text}
        for index, text in enumerate(rotated)
    ]
    return {
        "prompt": "选一个关键结论，确认你能复述：",
        "options": options,
        "correct_index": correct_index,
        "correct_label": options[correct_index]["label"],
        "retry_after_items": 5,
    }
