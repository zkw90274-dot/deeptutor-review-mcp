"""Keep the correct option from always sitting in the same position.

The generation pipeline returns the correct option first almost every time, so a
pack can end up with 10 of 13 answers on "A". A learner who notices can score
without reading the questions, and the pack trains the wrong habit. Rotating the
options fixes the position, but every explanation discusses the distractors by
letter ("B 误用线性关系 …"), so the letters must be remapped through the same
permutation or the explanation starts pointing at the wrong option.

The letter pattern is deliberately narrow. It only matches A–D that sit on a
real boundary — after a colon/semicolon/space and before a space or full stop —
because the same letters appear inside formulas as residues (``A_1``, ``A_k``)
and rewriting those would corrupt the mathematics.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

from deeptutor.review.models import Item

#: Option labels, in order. Only four-way items are rotated: the explanation
#: sweep below understands A–D, so a five-option item is left as it is.
LETTERS = "ABCD"

#: A letter that is being *named* as an option, not used as a symbol.
_LETTER_REF = re.compile(
    r"(?:(?<=[：；。、，\s])|^)([ABCD])(?=[\s.。：、，])",
    re.MULTILINE,
)


def pack_offset(seed: str) -> int:
    """Stable starting position for one pack, so re-runs land identically."""

    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return digest[0] % len(LETTERS)


def balance_items(items: Iterable[Item], *, offset: int = 0) -> tuple[list[Item], int]:
    """Rotate four-option items so the correct answer cycles through A–D.

    Returns ``(items, moved)``. Items that are not four-way single choice, or
    whose answer index is unusable, pass through untouched.
    """

    out: list[Item] = []
    slot = 0
    moved = 0
    for item in items:
        if (
            item.kind != "single_choice"
            or len(item.options) != len(LETTERS)
            or item.answer.choice_index is None
        ):
            out.append(item)
            continue
        target = (offset + slot) % len(LETTERS)
        slot += 1
        rotated, changed = rotate_item(item, target)
        out.append(rotated)
        moved += int(changed)
    return out, moved


def rotate_item(item: Item, target_index: int) -> tuple[Item, bool]:
    """Move the correct option to ``target_index`` and relabel the rest."""

    options = list(item.options)
    count = len(options)
    old_index = item.answer.choice_index
    if count != len(LETTERS) or old_index is None or not 0 <= old_index < count:
        return item, False
    shift = (old_index - target_index) % count
    if shift == 0:
        return item, False

    new_options = [
        options[(position + shift) % count].model_copy(
            update={"label": LETTERS[position]}
        )
        for position in range(count)
    ]
    # Old position p ends up at new position (p - shift) mod count.
    mapping = {
        LETTERS[position]: LETTERS[(position - shift) % count]
        for position in range(count)
    }
    new_answer = item.answer.model_copy(update={"choice_index": target_index})
    new_explanation = _remap_letters(item.explanation or "", mapping)
    return (
        item.model_copy(
            update={
                "options": new_options,
                "answer": new_answer,
                "explanation": new_explanation,
            }
        ),
        True,
    )


def _remap_letters(text: str, mapping: dict[str, str]) -> str:
    if not text:
        return text
    return _LETTER_REF.sub(lambda match: mapping.get(match.group(1), match.group(1)), text)


def answer_distribution(items: Iterable[Item]) -> dict[str, int]:
    """Count where the correct answers sit, for tests and audits."""

    counts = {letter: 0 for letter in LETTERS}
    for item in items:
        if item.kind != "single_choice" or item.answer.choice_index is None:
            continue
        index = item.answer.choice_index
        if 0 <= index < len(LETTERS):
            counts[LETTERS[index]] += 1
    return counts
