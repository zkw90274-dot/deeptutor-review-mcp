"""Hand-written Phase 0.5 fixture: 2 concepts, 3 items.

No LLM is involved. The fixture exercises the three item kinds that the
vertical slice must render and grade: single choice, numeric, and flashcard.
It is also the acceptance script's seed. Content is deliberately standard
textbook material (series RLC resonance) so the
export can be checked by a human, not just by a schema validator.
"""

from __future__ import annotations

from deeptutor.review.models import (
    Answer,
    CalloutBlock,
    Concept,
    Formula,
    Item,
    ItemBlock,
    Option,
    ProseBlock,
    ReviewPlan,
    ReviewSpec,
    Section,
    SourceRef,
    SummaryBlock,
)

SPEC_ID = "fixture-hf-resonance"


def build_fixture_spec() -> ReviewSpec:
    sources = {
        "src-hf-2-1": SourceRef(
            id="src-hf-2-1",
            kind="book_page",
            title="第2章 2.1 串联谐振",
            kb="hf",
            ref="chapter2/section-2.1",
            snippet="谐振时 XL = XC，电抗相互抵消，回路阻抗 Z = R 最小。",
        ),
        "src-hf-2-2": SourceRef(
            id="src-hf-2-2",
            kind="book_page",
            title="第2章 2.2 品质因数",
            kb="hf",
            ref="chapter2/section-2.2",
            snippet="串联谐振回路的品质因数 Q = ω0 L / R = 1/(ω0 C R)。",
        ),
    }

    resonance = Concept(
        id="c-series-resonance",
        name="串联谐振",
        aliases=["电压谐振"],
        definition="串联回路中感抗与容抗相互抵消、回路呈纯电阻性的状态。",
        formulas=[
            Formula(
                latex=r"f_0 = \frac{1}{2\pi\sqrt{LC}}",
                conditions="理想串联 RLC 回路",
                variables={"f_0": "谐振频率 / Hz", "L": "电感 / H", "C": "电容 / F"},
                source_ids=["src-hf-2-1"],
            )
        ],
        difficulty="medium",
        knowledge_type="concept",
        source_ids=["src-hf-2-1"],
        summary_one_line="谐振时 XL = XC，Z = R 最小，电流最大。",
        coverage="kb",
    )

    quality = Concept(
        id="c-quality-factor",
        name="品质因数",
        aliases=["Q 值"],
        definition="谐振回路储能与每周期耗能之比，衡量回路的选择性。",
        formulas=[
            Formula(
                latex=r"Q = \frac{1}{R}\sqrt{\frac{L}{C}}",
                conditions="串联谐振回路",
                variables={"Q": "品质因数", "R": "回路电阻 / Ω", "L": "电感 / H", "C": "电容 / F"},
                source_ids=["src-hf-2-2"],
            )
        ],
        difficulty="medium",
        knowledge_type="concept",
        source_ids=["src-hf-2-2"],
        summary_one_line="Q 越高，谐振曲线越尖锐，选择性越好。",
        coverage="kb",
    )

    items = [
        Item(
            id="item-resonance-choice",
            concept_ids=[resonance.id],
            kind="single_choice",
            stem="串联谐振发生时，回路阻抗为什么最小？",
            options=[
                Option(label="A", text="因为电感电压为零"),
                Option(label="B", text="因为感抗与容抗相互抵消"),
                Option(label="C", text="因为电阻被短路"),
                Option(label="D", text="因为电容电流为零"),
            ],
            answer=Answer(choice_index=1),
            explanation="谐振时 XL = XC，因为感抗与容抗相互抵消，Z = R 最小。",
            difficulty="easy",
            source_ids=["src-hf-2-1"],
        ),
        Item(
            id="item-resonance-numeric",
            concept_ids=[resonance.id],
            kind="numeric",
            stem="某串联谐振回路 L = 100 μH，C = 100 pF，求谐振频率 f0（MHz，保留三位小数）。",
            answer=Answer(value=1.592, tolerance=0.01),
            explanation="f0 = 1/(2π√(LC)) = 1/(2π×1e-7) ≈ 1.592 MHz。",
            difficulty="medium",
            source_ids=["src-hf-2-1"],
        ),
        Item(
            id="item-quality-flashcard",
            concept_ids=[quality.id],
            kind="flashcard",
            stem="写出串联谐振回路品质因数 Q 的定义式。",
            answer=Answer(text=r"Q = \frac{1}{R}\sqrt{\frac{L}{C}}"),
            explanation="Q = ω0 L / R = 1/(ω0 C R)，也可写成 Q = (1/R)√(L/C)。",
            difficulty="medium",
            source_ids=["src-hf-2-2"],
        ),
    ]

    sections = [
        Section(
            id="section-resonance",
            title="串联谐振与品质因数",
            concept_ids=[resonance.id, quality.id],
            order=1,
            blocks=[
                ProseBlock(
                    text=(
                        "串联 RLC 回路在某一频率下感抗与容抗大小相等、方向相反，"
                        "回路电流达到最大，这一状态称为串联谐振。"
                    )
                ),
                CalloutBlock(
                    tone="tip",
                    text="做题时先判断回路是否满足 XL = XC，再谈阻抗与电流。",
                ),
                ItemBlock(item_id="item-resonance-choice"),
                ItemBlock(item_id="item-resonance-numeric"),
                ItemBlock(item_id="item-quality-flashcard"),
                SummaryBlock(
                    text="谐振条件 XL = XC；谐振频率 f0 = 1/(2π√(LC))；Q 衡量选择性。"
                ),
            ],
        )
    ]

    return ReviewSpec(
        id=SPEC_ID,
        kb_ids=["hf"],
        language="zh",
        title="示例课程 · 串联谐振复习",
        plan=ReviewPlan(daily_minutes=10, timezone="Asia/Shanghai", target_retention=0.9),
        concepts=[resonance, quality],
        sections=sections,
        items=items,
        sources=sources,
    )
