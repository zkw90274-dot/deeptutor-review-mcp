"""RS-1301 golden spec set: deterministic export baselines.

Run with the project virtualenv from the repo root:

    .venv/Scripts/python.exe scripts/build_golden_specs.py --write
    .venv/Scripts/python.exe scripts/build_golden_specs.py

``--write`` rebuilds the three specs in Python (no randomness, no LLM), checks
that ``content_hash`` survives a validation round-trip, and writes
``tests/golden/*.json``. The default mode reads those JSON files, puts each
spec through ``ReviewService.put_spec``, and exports
``dist/golden/<spec-id>.html`` with a fixed ``now`` so two runs are
byte-identical.

Contract notes:

* the set covers inline/block LaTeX + all three item kinds, a plot with huge
  axis values (V2), a bar with 20 long Chinese labels (V1), a concept graph
  with prerequisite edges, both slider kinds, and narrow-screen stress;
* TikZ is deliberately absent: it needs a local xelatex and is therefore not
  reproducible across machines. TikZ has its own gates and tests.

Exit code is 0 only when every step succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
GOLDEN_DIR = ROOT / "tests" / "golden"
DIST_DIR = ROOT / "dist" / "golden"

# ``deeptutor`` is a regular package in site-packages, so putting the checkout
# on ``sys.path`` does not merge the namespace portion: the installed wheel
# would win and the script would exercise stale code. Prepend the checkout to
# the package's own ``__path__`` instead, the same seam ``tests/conftest.py``
# uses.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import deeptutor  # noqa: E402

_CHECKOUT_PACKAGE = SRC / "deeptutor"
if str(_CHECKOUT_PACKAGE) not in deeptutor.__path__:
    deeptutor.__path__.insert(0, str(_CHECKOUT_PACKAGE))

from deeptutor.review.models import (  # noqa: E402
    Answer,
    CalloutBlock,
    Concept,
    FigureBlock,
    FigureSpec,
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
    WidgetBlock,
    WidgetSpec,
)
from deeptutor.review.qa import run_gates  # noqa: E402
from deeptutor.review.service import ReviewService  # noqa: E402
from deeptutor.review.storage import ReviewStore  # noqa: E402

#: Fixed export instant. Matches ``tests/test_export.py`` so the goldens and
#: the rest of the suite reason about the same "today".
GOLDEN_NOW = 1_800_000_000.0

#: Written to ``tests/golden/<id>.json`` in this order.
GOLDEN_IDS = ("golden-formula", "golden-figures", "golden-widgets")


def _plan() -> ReviewPlan:
    return ReviewPlan(daily_minutes=10, timezone="Asia/Shanghai", target_retention=0.9)


def _sources(entries: list[SourceRef]) -> dict[str, SourceRef]:
    return {source.id: source for source in entries}


# ---------------------------------------------------------------------------
# golden-formula: Chinese stems + inline/block LaTeX + three item kinds.
# ---------------------------------------------------------------------------


def build_formula_spec() -> ReviewSpec:
    sources = _sources(
        [
            SourceRef(
                id="src-gf-ch4-1",
                kind="book_page",
                title="第4章 4.1 混频器",
                kb="hf",
                ref="chapter4/section-4.1",
                snippet="混频器把射频信号变换为中频，本振与射频频率之差即中频。",
            ),
            SourceRef(
                id="src-gf-ch4-2",
                kind="book_page",
                title="第4章 4.2 变频增益",
                kb="hf",
                ref="chapter4/section-4.2",
                snippet="变频增益用中频输出与射频输入电压比的分贝值表示。",
            ),
        ]
    )

    mixer = Concept(
        id="c-gf-mixer",
        name="混频器",
        aliases=["变频器"],
        definition="利用本振信号与射频信号相乘，把射频频率变换为中频的电路。",
        formulas=[
            Formula(
                latex=r"f_{IF} = f_{LO} - f_{RF}",
                conditions="下变频，且本振频率高于射频频率",
                variables={
                    "f_{IF}": "中频 / Hz",
                    "f_{LO}": "本振频率 / Hz",
                    "f_{RF}": "射频频率 / Hz",
                },
                source_ids=["src-gf-ch4-1"],
            )
        ],
        difficulty="medium",
        knowledge_type="concept",
        source_ids=["src-gf-ch4-1"],
        summary_one_line="中频等于本振频率与射频频率之差。",
        coverage="kb",
    )
    gain = Concept(
        id="c-gf-gain",
        name="变频增益",
        aliases=["转换增益"],
        definition="混频器中频输出电压与射频输入电压之比的分贝值。",
        formulas=[
            Formula(
                latex=r"G_c = 20\log_{10}\frac{U_{IF}}{U_{RF}}",
                conditions="输入输出端口均匹配",
                variables={
                    "G_c": "变频增益 / dB",
                    "U_{IF}": "中频输出电压 / V",
                    "U_{RF}": "射频输入电压 / V",
                },
                source_ids=["src-gf-ch4-2"],
            )
        ],
        difficulty="medium",
        knowledge_type="concept",
        source_ids=["src-gf-ch4-2"],
        summary_one_line="电压比每增大十倍，变频增益增加 20 dB。",
        coverage="kb",
    )

    items = [
        Item(
            id="item-gf-choice",
            concept_ids=[mixer.id],
            kind="single_choice",
            stem=(
                "某下变频混频器的本振频率为 $f_{LO} = 10\\,\\mathrm{MHz}$，"
                "射频输入为 $f_{RF} = 7\\,\\mathrm{MHz}$，中频 $f_{IF}$ 是多少？"
            ),
            options=[
                Option(label="A", text="1 MHz"),
                Option(label="B", text="3 MHz"),
                Option(label="C", text="7 MHz"),
                Option(label="D", text="17 MHz"),
            ],
            answer=Answer(choice_index=1),
            explanation=(
                "下变频时中频等于两频率之差："
                "$f_{IF} = f_{LO} - f_{RF} = 10 - 7 = 3\\,\\mathrm{MHz}$，"
                "所以选项 B（3 MHz）正确。"
            ),
            difficulty="easy",
            source_ids=["src-gf-ch4-1"],
        ),
        Item(
            id="item-gf-numeric",
            concept_ids=[gain.id],
            kind="numeric",
            stem=(
                "某混频器射频输入电压 $U_{RF} = 20\\,\\mathrm{mV}$，"
                "中频输出电压 $U_{IF} = 200\\,\\mathrm{mV}$，"
                "求变频增益 $G_c$（dB）。"
            ),
            answer=Answer(value=20.0, tolerance=0.1),
            explanation=(
                "变频增益按电压比的分贝值定义：\n\n"
                "$$G_c = 20\\log_{10}\\frac{U_{IF}}{U_{RF}}$$\n\n"
                "代入 $U_{IF} = 200\\,\\mathrm{mV}$、$U_{RF} = 20\\,\\mathrm{mV}$，"
                "得 $G_c = 20\\log_{10}10 = 20\\,\\mathrm{dB}$。\n\n"
                "这里的关键是：电压比是 10 倍，对应 20 dB；若用功率比计算，"
                "同样的 10 倍功率才对应 10 dB。分母必须是射频输入电压，"
                "而不是本振电压，否则会得到一个与器件无关的数值。\n\n"
                "实际测量时输入输出端口都要匹配到 50 Ω，否则分压会直接叠加到"
                "测量结果里，使读数偏离器件本身的增益。"
            ),
            difficulty="medium",
            source_ids=["src-gf-ch4-2"],
        ),
        Item(
            id="item-gf-flashcard",
            concept_ids=[mixer.id],
            kind="flashcard",
            stem="写出下变频混频器中频频率与输入频率的关系式。",
            answer=Answer(text=r"f_{IF} = f_{LO} - f_{RF}"),
            explanation=(
                "本振与射频相乘后，差频分量即中频："
                "$f_{IF} = f_{LO} - f_{RF}$"
                "（下变频时本振频率高于射频频率）。"
            ),
            difficulty="easy",
            source_ids=["src-gf-ch4-1"],
        ),
    ]

    sections = [
        Section(
            id="sec-gf-mixer",
            title="混频与变频增益",
            concept_ids=[mixer.id, gain.id],
            order=1,
            blocks=[
                ProseBlock(
                    text=(
                        "混频器是超外差接收机的核心部件：本振与射频输入相乘后，"
                        "差频分量被选出作为中频，即 $f_{IF} = f_{LO} - f_{RF}$。\n\n"
                        "中频频率固定，后续各级滤波器才能按固定中心频率设计：\n\n"
                        "$$f_{IF} = f_{LO} - f_{RF}$$\n\n"
                        "因此射频频率改变时，只需同步调整本振频率。"
                    )
                ),
                CalloutBlock(
                    tone="tip",
                    text=(
                        "判断上变频还是下变频时，先比较本振与射频的大小；"
                        "本振更高时为下变频，$f_{IF} = f_{LO} - f_{RF}$。"
                    ),
                ),
                ItemBlock(item_id="item-gf-choice"),
                ItemBlock(item_id="item-gf-numeric"),
                ItemBlock(item_id="item-gf-flashcard"),
                SummaryBlock(
                    text=(
                        "混频器的中频 $f_{IF} = f_{LO} - f_{RF}$；"
                        "变频增益 $G_c = 20\\log_{10}(U_{IF}/U_{RF})$。"
                    )
                ),
            ],
        )
    ]

    return ReviewSpec(
        id="golden-formula",
        kb_ids=["hf"],
        language="zh",
        title="示例课程 · 公式与题型基线",
        plan=_plan(),
        concepts=[mixer, gain],
        sections=sections,
        items=items,
        sources=sources,
    )


# ---------------------------------------------------------------------------
# golden-figures: huge plot ticks (V2), 20 long bar labels (V1), concept DAG.
# ---------------------------------------------------------------------------

BAR_LABELS_20 = [
    "串联谐振回路品质因数实测值随温度变化",
    "并联谐振回路品质因数实测值随频率变化",
    "高频变压器漏感折算到初级后的等效值",
    "中频放大器通频带宽度测量值与理论值",
    "混频器变频损耗随本振功率变化曲线",
    "本振泄漏到射频端口的功率测量结果",
    "镜像频率抑制比在不同中频下的测试",
    "输入匹配网络驻波比实测最大值记录",
    "输出匹配网络驻波比实测最大值记录",
    "晶体滤波器矩形系数测量值与标称值",
    "陶瓷滤波器插入损耗随频率变化曲线",
    "声表面波滤波器群时延波动测量结果",
    "低噪声放大器噪声系数随频率的变化",
    "功率放大器三阶交调截点测量结果",
    "振荡器频率稳定度长期漂移实测数据",
    "鉴相器线性范围实测数据与拟合曲线",
    "环路滤波器相位裕度测量值与设计值",
    "压控振荡器调谐灵敏度曲线实测记录",
    "混频器本振端口隔离度随频率变化",
    "天线端口回波损耗随频率变化测量值",
]

BAR_VALUES_20 = [
    42.5,
    38.2,
    3.14,
    12.8,
    6.35,
    23.4,
    55.1,
    1.82,
    1.95,
    2.4,
    4.6,
    18.3,
    2.1,
    33.0,
    0.85,
    41.2,
    62.0,
    9.4,
    27.6,
    15.9,
]


def build_figures_spec() -> ReviewSpec:
    sources = _sources(
        [
            SourceRef(
                id="src-gf2-ch3-1",
                kind="book_page",
                title="第3章 3.1 欧姆定律与功率",
                kb="ct",
                ref="chapter3/section-3.1",
                snippet="欧姆定律写为 U = IR；电阻功率 P = UI。",
            ),
            SourceRef(
                id="src-gf2-ch3-2",
                kind="book_page",
                title="第3章 3.2 分压与基尔霍夫定律",
                kb="ct",
                ref="chapter3/section-3.2",
                snippet="串联电阻分压；任一节点各支路电流代数和为零。",
            ),
        ]
    )

    resistor = Concept(
        id="c-gf2-resistor",
        name="电阻元件",
        definition="电压与电流同相的耗能元件，其伏安特性为过原点的直线。",
        difficulty="easy",
        knowledge_type="concept",
        source_ids=["src-gf2-ch3-1"],
        summary_one_line="电阻的电压与电流成正比。",
        coverage="kb",
    )
    ohm = Concept(
        id="c-gf2-ohm",
        name="欧姆定律",
        prerequisites=[resistor.id],
        definition="电阻两端电压等于电流与电阻的乘积。",
        formulas=[
            Formula(
                latex=r"U = IR",
                conditions="线性电阻",
                variables={"U": "电压 / V", "I": "电流 / A", "R": "电阻 / Ω"},
                source_ids=["src-gf2-ch3-1"],
            )
        ],
        difficulty="easy",
        knowledge_type="concept",
        source_ids=["src-gf2-ch3-1"],
        summary_one_line="电压等于电流乘电阻。",
        coverage="kb",
    )
    power = Concept(
        id="c-gf2-power",
        name="电阻功率",
        prerequisites=[ohm.id],
        definition="电阻把电能转化为热能，功率等于电压与电流的乘积。",
        formulas=[
            Formula(
                latex=r"P = UI = I^{2}R = \frac{U^{2}}{R}",
                conditions="线性电阻，采用关联参考方向",
                variables={"P": "功率 / W", "U": "电压 / V", "I": "电流 / A"},
                source_ids=["src-gf2-ch3-1"],
            )
        ],
        difficulty="medium",
        knowledge_type="concept",
        source_ids=["src-gf2-ch3-1"],
        summary_one_line="功率可以写成电压与电流乘积的三种等价形式。",
        coverage="kb",
    )
    divider = Concept(
        id="c-gf2-divider",
        name="串联分压",
        prerequisites=[ohm.id],
        definition="串联电阻上的电压按电阻值比例分配。",
        formulas=[
            Formula(
                latex=r"U_1 = \frac{R_1}{R_1 + R_2}U",
                conditions="两个电阻串联，总电压为 U",
                variables={"U_1": "R_1 上的电压 / V", "R_1": "第一个电阻 / Ω"},
                source_ids=["src-gf2-ch3-2"],
            )
        ],
        difficulty="medium",
        knowledge_type="procedure",
        source_ids=["src-gf2-ch3-2"],
        summary_one_line="串联电路中电阻越大分到的电压越多。",
        coverage="kb",
    )
    kcl = Concept(
        id="c-gf2-kcl",
        name="基尔霍夫电流定律",
        prerequisites=[ohm.id],
        definition="任一节点上各支路电流的代数和为零。",
        difficulty="medium",
        knowledge_type="concept",
        source_ids=["src-gf2-ch3-2"],
        summary_one_line="流入节点的电流等于流出节点的电流。",
        coverage="kb",
    )

    figures = [
        FigureSpec(
            id="fig-gf2-large-ticks",
            kind="plot",
            title="大数值坐标轴（V2 压力用例）",
            caption="纵轴数值达到 7.5e11，刻度标签宽于左侧留白。",
            points=[
                (0.0, 7.5e11),
                (1.0, 7.42e11),
                (2.0, 7.58e11),
                (3.0, 7.5e11),
                (4.0, 7.65e11),
            ],
            x_label="采样点",
            y_label="电压 / V",
            unit="V",
            source_ids=["src-gf2-ch3-1"],
        ),
        FigureSpec(
            id="fig-gf2-long-labels",
            kind="bar",
            title="20 个长中文标签（V1 压力用例）",
            caption="每个标签超过 12 个汉字，按槽宽截断后才不会相互重叠。",
            bar_labels=list(BAR_LABELS_20),
            bar_values=list(BAR_VALUES_20),
            x_label="测试项目",
            y_label="测得值",
            source_ids=["src-gf2-ch3-2"],
        ),
    ]

    items = [
        Item(
            id="item-gf2-choice",
            concept_ids=[ohm.id],
            kind="single_choice",
            stem="下列哪个表达式是欧姆定律？",
            options=[
                Option(label="A", text="U = IR"),
                Option(label="B", text="U = I/R"),
                Option(label="C", text="I = UR"),
                Option(label="D", text="R = UI"),
            ],
            answer=Answer(choice_index=0),
            explanation="欧姆定律写为 $U = IR$，即电压等于电流与电阻的乘积。",
            difficulty="easy",
            source_ids=["src-gf2-ch3-1"],
        ),
        Item(
            id="item-gf2-numeric",
            concept_ids=[power.id],
            kind="numeric",
            stem=(
                "某电阻两端电压 $U = 10\\,\\mathrm{V}$，"
                "电流 $I = 0.5\\,\\mathrm{A}$，求功率 $P$（W）。"
            ),
            answer=Answer(value=5.0, tolerance=0.01),
            explanation="功率等于电压与电流的乘积，所以功率为 5 W。",
            difficulty="easy",
            source_ids=["src-gf2-ch3-1"],
        ),
    ]

    sections = [
        Section(
            id="sec-gf2-theorems",
            title="电路定理与图表基线",
            concept_ids=[resistor.id, ohm.id, power.id, divider.id, kcl.id],
            order=1,
            blocks=[
                ProseBlock(
                    text=(
                        "线性电阻满足欧姆定律 $U = IR$；功率可以写成 $P = UI$。"
                        "多个概念之间的前置关系由图中的有向边表示：先掌握电阻元件，"
                        "再学习欧姆定律，最后才能讨论功率与分压。"
                    )
                ),
                FigureBlock(figure_id="fig-gf2-large-ticks"),
                FigureBlock(figure_id="fig-gf2-long-labels"),
                ItemBlock(item_id="item-gf2-choice"),
                ItemBlock(item_id="item-gf2-numeric"),
                SummaryBlock(
                    text="欧姆定律 $U = IR$；功率 $P = UI$；串联分压按电阻比例分配。"
                ),
            ],
        )
    ]

    return ReviewSpec(
        id="golden-figures",
        kb_ids=["ct"],
        language="zh",
        title="电路定理 · 图表基线",
        plan=_plan(),
        concepts=[resistor, ohm, power, divider, kcl],
        sections=sections,
        figures=figures,
        items=items,
        sources=sources,
    )


# ---------------------------------------------------------------------------
# golden-widgets: slider_formula + slider_plot + long-label bar + narrow stress.
# ---------------------------------------------------------------------------

BAR_LABELS_12 = [
    "空载条件下电压传输系数实测平均值",
    "额定负载条件下电压传输系数实测值",
    "轻载条件下电压传输系数实测平均值",
    "重载条件下电压传输系数实测平均值",
    "容性负载条件下传输系数测量结果",
    "感性负载条件下传输系数测量结果",
    "纯电阻负载条件下传输系数测量值",
    "温度升高后传输系数变化量测量值",
    "频率偏移百分之一时传输系数变化",
    "电源电压下降时传输系数变化测量",
    "输入端加隔离电阻后传输系数测量",
    "输出端加隔直电容后传输系数测量",
]

BAR_VALUES_12 = [
    0.98,
    0.95,
    0.97,
    0.92,
    0.94,
    0.93,
    0.96,
    0.89,
    0.9,
    0.87,
    0.85,
    0.82,
]


def build_widgets_spec() -> ReviewSpec:
    sources = _sources(
        [
            SourceRef(
                id="src-gw-ch5-1",
                kind="book_page",
                title="第5章 5.1 RC 低通滤波器",
                kb="hf",
                ref="chapter5/section-5.1",
                snippet="截止频率 f_c = 1/(2πRC)，幅频特性随频率升高而下降。",
            ),
            SourceRef(
                id="src-gw-ch5-2",
                kind="book_page",
                title="第5章 5.2 频率响应测量",
                kb="hf",
                ref="chapter5/section-5.2",
                snippet="测量幅频特性时按倍频程取点，并记录相位变化。",
            ),
        ]
    )

    lowpass = Concept(
        id="c-gw-rc",
        name="RC 低通滤波器",
        definition="由电阻与电容构成的一阶低通网络，高频分量被电容旁路。",
        formulas=[
            Formula(
                latex=r"H(j\omega) = \frac{1}{1 + j\omega RC}",
                conditions="一阶无源 RC 网络，输出取自电容两端",
                variables={
                    "H(j\\omega)": "电压传输函数",
                    "R": "电阻 / Ω",
                    "C": "电容 / F",
                },
                source_ids=["src-gw-ch5-1"],
            )
        ],
        difficulty="medium",
        knowledge_type="concept",
        source_ids=["src-gw-ch5-1"],
        summary_one_line="低频通过、高频衰减。",
        coverage="kb",
    )
    cutoff = Concept(
        id="c-gw-cutoff",
        name="截止频率",
        prerequisites=[lowpass.id],
        definition="幅值下降到低频值的 0.707 倍时对应的频率。",
        formulas=[
            Formula(
                latex=r"f_c = \frac{1}{2\pi RC}",
                conditions="一阶 RC 低通",
                variables={"f_c": "截止频率 / Hz", "R": "电阻 / Ω", "C": "电容 / F"},
                source_ids=["src-gw-ch5-1"],
            )
        ],
        difficulty="medium",
        knowledge_type="concept",
        source_ids=["src-gw-ch5-1"],
        summary_one_line="截止频率由时间常数 RC 决定。",
        coverage="kb",
    )
    slope = Concept(
        id="c-gw-slope",
        name="阻带衰减斜率",
        prerequisites=[cutoff.id],
        definition="一阶低通在截止频率以上按每十倍频 20 dB 下降。",
        difficulty="hard",
        knowledge_type="concept",
        source_ids=["src-gw-ch5-2"],
        summary_one_line="一阶网络的阻带斜率是每十倍频 20 dB。",
        coverage="kb",
    )

    widgets = [
        WidgetSpec(
            id="w-gw-gain",
            kind="slider_formula",
            title="RC 低通幅频特性",
            formula_latex=r"\left\vert H \right\vert = \frac{1}{\sqrt{1 + (2\pi f RC)^{2}}}",
            expression="1/sqrt(1+(2*pi*f*0.001)**2)",
            parameter="f",
            minimum=0.0,
            maximum=10000.0,
            step=100.0,
            unit="倍",
            precision=3,
            samples=41,
            source_ids=["src-gw-ch5-1"],
        ),
        WidgetSpec(
            id="w-gw-ripple",
            kind="slider_plot",
            title="补偿网络幅频特性（抛物线近似）",
            formula_latex=r"A(f) = 1.5 - 0.5\left(\frac{f}{10000}\right)^{2}",
            expression="1.5 - 0.5*(f/10000)**2",
            parameter="f",
            minimum=0.0,
            maximum=10000.0,
            step=100.0,
            unit="倍",
            precision=3,
            samples=41,
            source_ids=["src-gw-ch5-2"],
        ),
    ]

    figures = [
        FigureSpec(
            id="fig-gw-bar",
            kind="bar",
            title="不同负载条件下电压传输系数实测值（窄屏压力用例）",
            caption="12 个长中文标签；375px 视口下需要横向滚动或按槽宽截断。",
            bar_labels=list(BAR_LABELS_12),
            bar_values=list(BAR_VALUES_12),
            x_label="负载与测试条件",
            y_label="传输系数",
            source_ids=["src-gw-ch5-2"],
        ),
        FigureSpec(
            id="fig-gw-plot",
            kind="plot",
            title="一阶低通归一化幅频特性（长轴标签压力用例）",
            caption="横轴标签较长，窄屏下与刻度标签争夺左右留白。",
            points=[
                (0.0, 1.0),
                (200.0, 0.98),
                (400.0, 0.9),
                (600.0, 0.78),
                (800.0, 0.62),
                (1000.0, 0.5),
            ],
            x_label="频率 / Hz",
            y_label="归一化幅值",
            unit="倍",
            source_ids=["src-gw-ch5-1"],
        ),
    ]

    items = [
        Item(
            id="item-gw-numeric",
            concept_ids=[cutoff.id],
            kind="numeric",
            stem=(
                "某 RC 低通滤波器 $R = 1\\,\\mathrm{k}\\Omega$、"
                "$C = 1\\,\\mu\\mathrm{F}$，求截止频率 $f_c$（Hz）。"
            ),
            answer=Answer(value=159.155, tolerance=0.5),
            explanation=(
                "$f_c = 1/(2\\pi RC) = 1/(2\\pi \\times 1000 \\times 1e-6) "
                "\\approx 159.155\\,\\mathrm{Hz}$。"
            ),
            difficulty="medium",
            source_ids=["src-gw-ch5-1"],
        ),
        Item(
            id="item-gw-flashcard",
            concept_ids=[lowpass.id],
            kind="flashcard",
            stem="写出 RC 低通滤波器截止频率与元件参数的关系式。",
            answer=Answer(text=r"f_c = \frac{1}{2\pi RC}"),
            explanation=(
                "截止频率由时间常数决定：$f_c = \\frac{1}{2\\pi RC}$，"
                "增大 $R$ 或 $C$ 都会降低截止频率。"
            ),
            difficulty="easy",
            source_ids=["src-gw-ch5-1"],
        ),
    ]

    sections = [
        Section(
            id="sec-gw-filters",
            title="RC 低通滤波器的频率响应",
            concept_ids=[lowpass.id, cutoff.id, slope.id],
            order=1,
            blocks=[
                ProseBlock(
                    text=(
                        "一阶 RC 低通的传输函数为 "
                        "$H(j\\omega) = \\frac{1}{1 + j\\omega RC}$。"
                        "当频率等于截止频率 $f_c = \\frac{1}{2\\pi RC}$ 时，"
                        "幅值下降到低频值的 $1/\\sqrt{2}$。"
                    )
                ),
                WidgetBlock(widget_id="w-gw-gain"),
                WidgetBlock(widget_id="w-gw-ripple"),
                FigureBlock(figure_id="fig-gw-bar"),
                FigureBlock(figure_id="fig-gw-plot"),
                CalloutBlock(
                    tone="warning",
                    text=(
                        "拖动滑块时注意参数单位：$f$ 取 Hz，$R$ 取 Ω，$C$ 取 F；"
                        "混用单位会得到量级错误的曲线。"
                    ),
                ),
                ItemBlock(item_id="item-gw-numeric"),
                ItemBlock(item_id="item-gw-flashcard"),
                SummaryBlock(
                    text=(
                        "截止频率 $f_c = \\frac{1}{2\\pi RC}$；"
                        "一阶低通在截止频率以上按每十倍频 20 dB 衰减。"
                    )
                ),
            ],
        )
    ]

    return ReviewSpec(
        id="golden-widgets",
        kb_ids=["hf"],
        language="zh",
        title="频率响应 · 交互基线",
        plan=_plan(),
        concepts=[lowpass, cutoff, slope],
        sections=sections,
        figures=figures,
        widgets=widgets,
        items=items,
        sources=sources,
    )


def golden_specs() -> list[ReviewSpec]:
    return [build_formula_spec(), build_figures_spec(), build_widgets_spec()]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def _checked_round_trip(spec: ReviewSpec) -> ReviewSpec:
    """Validate a dump of *spec* and prove the content hash is stable."""
    rebuilt = ReviewSpec.model_validate(spec.model_dump(mode="json"))
    if rebuilt.content_hash != spec.content_hash:
        raise RuntimeError(
            f"{spec.id}: content_hash drifted across a round-trip "
            f"({spec.content_hash} -> {rebuilt.content_hash})"
        )
    report = run_gates(rebuilt)
    if not report.passed:
        details = [detail for gate in report.gates for detail in gate.details]
        raise RuntimeError(f"{spec.id}: QA gates failed: {details}")
    return rebuilt


def write_mode() -> int:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    failures = 0
    for spec in golden_specs():
        try:
            rebuilt = _checked_round_trip(spec)
        except RuntimeError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            failures += 1
            continue
        path = GOLDEN_DIR / f"{rebuilt.id}.json"
        payload = json.dumps(
            rebuilt.model_dump(mode="json"), ensure_ascii=False, indent=2
        )
        path.write_text(payload + "\n", encoding="utf-8")
        written.append(rebuilt.id)
        print(
            f"[OK] wrote {path} "
            f"(content_hash={rebuilt.content_hash[:12]}, "
            f"items={len(rebuilt.items)}, figures={len(rebuilt.figures)}, "
            f"widgets={len(rebuilt.widgets)})"
        )
    if failures:
        return 1
    if sorted(written) != sorted(GOLDEN_IDS):
        print("[FAIL] golden id list drifted from the builders", file=sys.stderr)
        return 1
    return 0


def check_mode() -> int:
    failures = 0
    for spec in golden_specs():
        rebuilt = _checked_round_trip(spec)
        path = GOLDEN_DIR / f"{rebuilt.id}.json"
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"[FAIL] golden baseline unavailable: {path}: {exc}", file=sys.stderr)
            failures += 1
            continue
        if actual != rebuilt.model_dump(mode="json"):
            print(f"[FAIL] golden baseline differs: {path}", file=sys.stderr)
            failures += 1
        else:
            print(f"[OK] checked {path} without modifying it")
    return int(bool(failures))


def export_mode() -> int:
    paths = [GOLDEN_DIR / f"{spec_id}.json" for spec_id in GOLDEN_IDS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        print(
            "[FAIL] missing golden spec files: "
            + ", ".join(missing)
            + " (run with --write first)",
            file=sys.stderr,
        )
        return 1

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    # Isolated store: an acceptance run must never write into the learner's
    # real workspace, and the export only needs the spec for the duration of
    # this process.
    with tempfile.TemporaryDirectory(prefix="golden-review-") as home:
        os.environ["DEEPTUTOR_REVIEW_HOME"] = home
        os.environ["DEEPTUTOR_HOME"] = str(Path(home) / "core")
        service = ReviewService(ReviewStore(home))
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                spec = ReviewSpec.model_validate(payload)
                service.put_spec(spec)
                document = service.export(spec.id, now=GOLDEN_NOW)
            except Exception as exc:  # noqa: BLE001 - report and keep going
                print(f"[FAIL] {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                failures += 1
                continue
            if not document.strip():
                print(f"[FAIL] {path.name}: export produced an empty document", file=sys.stderr)
                failures += 1
                continue
            out = DIST_DIR / f"{spec.id}.html"
            out.write_text(document, encoding="utf-8")
            size = len(document.encode("utf-8"))
            print(f"[OK] exported {out} ({size} bytes, from {path.name})")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate tests/golden/*.json instead of exporting",
    )
    parser.add_argument("--check", action="store_true", help="compare builders against existing golden inputs without writing")
    args = parser.parse_args(argv)
    if args.write:
        return write_mode()
    if args.check:
        return check_mode()
    return export_mode()


if __name__ == "__main__":
    raise SystemExit(main())
