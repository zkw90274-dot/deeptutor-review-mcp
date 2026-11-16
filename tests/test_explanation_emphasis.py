"""Behavioural contract for emphasis in the exported explanation.

Pack q005 of 高频第 8 章 ships lead-ins like ``*倍频 $\\times N_1=100$：*``. The
`mathText` pass used to own the whole inline path, so the span was cut into
``*倍频 `` / ``$...$`` / ``：*`` and neither half could pair: learners saw the
asterisks. Emphasis therefore has to be resolved *before* math is split, and the
scan must skip ``$...$`` so a multiplication star never opens a span.

These run the template's own functions in Node against a minimal DOM shim
instead of asserting on strings -- a marker can be present and still be wrong.
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

# The slice runs from `el` to the function after `appendRich`; everything the
# inline path needs sits in that range and nothing in it executes on load
# beyond two constant assignments.
_START = TEMPLATE.index("  function el(")
_END = TEMPLATE.index("  function isTableRow(")
SLICE = TEMPLATE[_START:_END]

HARNESS = """
var storageKey = "test";
var memory = {};
var window = { localStorage: { getItem: function () { return null; },
                               setItem: function () {} } };
function makeEl(tag) {
  var node = {
    tagName: String(tag).toUpperCase(),
    className: "",
    children: [],
    _text: "",
    nodeType: 1,
    appendChild: function (child) { this.children.push(child); return child; },
    setAttribute: function () {},
    querySelector: function () { return null; },
    querySelectorAll: function () { return []; }
  };
  Object.defineProperty(node, "textContent", {
    get: function () {
      if (this.children.length) {
        return this._text + this.children.map(function (c) { return c.textContent; }).join("");
      }
      return this._text;
    },
    set: function (value) { this._text = String(value); this.children = []; }
  });
  return node;
}
var document = {
  createElement: makeEl,
  createTextNode: function (text) {
    return { nodeType: 3, textContent: String(text), children: [] };
  }
};
__SLICE__
function serialize(node) {
  return {
    tag: node.nodeType === 3 ? "#text" : node.tagName.toLowerCase(),
    cls: node.className || "",
    text: node.nodeType === 3 ? node.textContent : null,
    children: (node.children || []).map(serialize)
  };
}
var cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
console.log(JSON.stringify(cases.map(function (text) {
  var root = makeEl("div");
  appendRich(root, text);
  return serialize(root);
})));
""".replace("__SLICE__", SLICE)


def _render(cases: list[str]) -> list[dict]:
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


def _walk(nodes: list[dict]):
    for node in nodes:
        yield node
        yield from _walk(node.get("children") or [])


def _flat_text(nodes: list[dict]) -> str:
    return "".join(node["text"] or "" for node in _walk(nodes))


def test_emphasis_spanning_a_formula_becomes_em_without_asterisks() -> None:
    # The exact shape found in 高频第 8 章 q005.
    (tree,) = _render(["*倍频 $\\times N_1=100$：* 倍频属于对瞬时相位乘以 $N$。"])
    nodes = list(_walk([tree]))
    em = [n for n in nodes if n["cls"] == "rt-em"]
    assert len(em) == 1, tree
    assert "*" not in _flat_text([tree])
    inner = _flat_text(em[0]["children"])
    assert "倍频" in inner and "：" in inner
    # the formula inside the label is still typeset, not left as LaTeX
    assert any(n["cls"] == "math" for n in _walk(em[0]["children"]))


def test_emphasis_in_a_list_item_and_strong_both_render() -> None:
    tree_list, tree_strong = _render(["*必须倍频：* 间接调频用积分", "**重点**"])
    em = [n for n in _walk([tree_list]) if n["cls"] == "rt-em"]
    assert len(em) == 1
    assert _flat_text(em[0]["children"]) == "必须倍频："
    assert "*" not in _flat_text([tree_list])
    strong = [n for n in _walk([tree_strong]) if n["cls"] == "rt-strong"]
    assert len(strong) == 1
    assert _flat_text(strong[0]["children"]) == "重点"


def test_multiplication_stars_in_prose_and_math_stay_literal() -> None:
    # No word boundary at either end -> not a span. Inside $...$ -> never scanned.
    prose, math = _render(["2*pi*f0 与 20*log10(x)", "$a*b*c$"])
    assert not [n for n in _walk([prose]) if n["cls"] in ("rt-em", "rt-strong")]
    assert "2*pi*f0" in _flat_text([prose])
    assert not [n for n in _walk([math]) if n["cls"] in ("rt-em", "rt-strong")]
    assert any(n["cls"] == "math" for n in _walk([math]))


def test_unpaired_asterisk_is_left_alone() -> None:
    # Nothing pairs up, so the star must survive as text rather than swallow the
    # rest of the sentence into an <em>.
    (tree,) = _render(["这是 *重点"])
    assert not [n for n in _walk([tree]) if n["cls"] in ("rt-em", "rt-strong")]
    assert _flat_text([tree]) == "这是 *重点"
