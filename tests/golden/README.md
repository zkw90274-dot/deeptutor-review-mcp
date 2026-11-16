# Golden spec set (RS-1301)

Deterministic `ReviewSpec` inputs used as the export baseline for the review
studio. Phase-28 §3.1 freezes the coverage matrix below; `tests/test_golden_specs.py`
enforces it so a future edit cannot silently shrink the baseline.

The files are generated, never hand-edited:

```bash
# regenerate tests/golden/*.json (deterministic, no LLM)
.venv/Scripts/python.exe scripts/build_golden_specs.py --write

# validate + export to dist/golden/<spec-id>.html (dist is gitignored)
.venv/Scripts/python.exe scripts/build_golden_specs.py
```

Run from the repo root. `--write` refuses to emit a spec whose
`content_hash` changes across a validation round-trip or whose QA gates fail.

## Coverage matrix

| Case | `golden-formula` | `golden-figures` | `golden-widgets` |
|---|---|---|---|
| Chinese stems/explanations | 3 items | 2 items | 2 items |
| Inline LaTeX (`$...$`) | stems, callout, summary | prose, summary | prose, callout, summary |
| Block LaTeX (`$$...$$`) | prose + long explanation | — | — |
| Item kinds | `single_choice` / `numeric` / `flashcard` | `single_choice` / `numeric` | `numeric` / `flashcard` |
| Long explanation (> 200 chars) | numeric item | — | — |
| Sources | 2 sources, every item anchored | 2 sources, every item anchored | 2 sources, every item anchored |
| Formula (`Concept.formulas`) | 2 | 3 | 2 |
| Plot | — | `fig-gf2-large-ticks`: y ≈ 7.5e11 (V2) | `fig-gw-plot`: long axis labels |
| Bar | — | `fig-gf2-long-labels`: 20 long Chinese labels (V1) | `fig-gw-bar`: 12 long labels |
| Concept graph | none (no prerequisite edges) | 4 prerequisite edges over 5 concepts | 2 edges over 3 concepts |
| `slider_formula` | — | — | `w-gw-gain` |
| `slider_plot` | — | — | `w-gw-ripple` |
| Narrow-screen stress | — | 20 long bar labels | long bar labels + long captions + long axis labels |
| TikZ | excluded | excluded | excluded |

## Why TikZ is absent

TikZ figures compile through the local `xelatex` toolchain, so their output is
not reproducible on a machine without it. TikZ has dedicated gates and tests
(`tests/test_tikz.py`, `tests/test_layout_gate.py`); the golden set stays
toolchain-free so every checkout can regenerate and export it.

## Slider expressions

`WidgetSpec.expression` is parsed by a closed AST, not evaluated from source.
Whitelisted functions may be used as call targets — `sqrt(x)` works, a bare
`sqrt` does not — and the goldens exercise the function form directly:

* `1/sqrt(1+(2*pi*f*0.001)**2)`
* `1.5 - 0.5*(f/10000)**2`

`tests/test_golden_specs.py::test_golden_assets_render_instead_of_falling_back`
fails if an expression degrades to a placeholder.

## Determinism

`scripts/build_golden_specs.py` pins export to `now = 1_800_000_000.0` and
`tests/test_golden_specs.py` asserts that two exports of the same spec at the
same instant are byte-identical. The JSON pins `content_hash`; any content
edit without regeneration fails validation with a hash mismatch.
