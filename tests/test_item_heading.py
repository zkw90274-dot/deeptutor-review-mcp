"""Behavioural contract for the item-heading truncation.

Generated packs use the concept name as a one-line summary, and real names run
to 120 characters. The heading used to clip at 60, so 高频第 5 章 q006 (61
characters) lost its last two characters to a "…" and looked like a stem that
had been cut off. The cap now matches the backend label bound (200). These
cases run the template's own ``truncateMath`` in Node instead of asserting on
strings, because the guard also has to keep ``$...$`` intact.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from importlib.resources import files

import pytest

TEMPLATE = (
    files("deeptutor.review").joinpath("templates/review.html").read_text(encoding="utf-8")
)

NODE = shutil.which("node")

_START = TEMPLATE.index("  function el(")
_END = TEMPLATE.index("  function isTableRow(")
SLICE = TEMPLATE[_START:_END]

HARNESS = """
var storageKey = "test";
var window = { localStorage: { getItem: function () { return null; },
                               setItem: function () {} } };
__SLICE__
var cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
console.log(JSON.stringify(cases.map(function (c) {
  return truncateMath(c.text, c.max);
})));
""".replace("__SLICE__", SLICE)

#: The real concept name behind kb-3a47352689-5-f83749-q006.
Q006_CONCEPT = (
    "实验三中改变负载电阻 RL 观察三种状态波形、"
    "由实测峰峰值与集电极电源电压估算输出功率，"
    "并联系负载特性说明集电极调幅的成因"
)


def _truncate(cases: list[dict]) -> list[str]:
    if not NODE:
        pytest.skip("node is not installed")
    proc = subprocess.run(
        [NODE, "-e", HARNESS],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_real_61_char_concept_name_is_shown_in_full() -> None:
    assert len(Q006_CONCEPT) == 61
    assert _truncate([{"text": Q006_CONCEPT, "max": 200}]) == [Q006_CONCEPT]


def test_names_up_to_the_backend_label_bound_are_not_cut() -> None:
    name = "高频概念" * 30
    assert len(name) == 120
    assert _truncate([{"text": name, "max": 200}]) == [name]


def test_pathological_name_clips_without_breaking_a_formula() -> None:
    name = "x" * 150 + "$A_v=G_m R_L$" + "y" * 100
    (out,) = _truncate([{"text": name, "max": 200}])
    assert out.endswith("…")
    assert len(out) <= 200
    assert out.count("$") % 2 == 0


def test_pathological_plain_name_clips_to_the_bound() -> None:
    (out,) = _truncate([{"text": "概念" * 150, "max": 200}])
    assert out.endswith("…")
    assert len(out) <= 200
