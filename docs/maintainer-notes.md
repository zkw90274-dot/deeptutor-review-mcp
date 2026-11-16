# Maintainer notes

Architecture invariants, release steps and known limits for `deeptutor-review-mcp`.
User-facing docs live in [`USAGE.md`](USAGE.md), [`usage-guide.md`](usage-guide.md) and
[`skill/SKILL.md`](skill/SKILL.md).

## What this package is

- A DeepTutor module wheel (`deeptutor-review`) plus an MCP server over the same service layer.
- Spaced repetition is delegated to the core library: `deeptutor.learning.scheduler` is the
  only scheduler. Review never implements a second SRS.
- Grades are self-assessments (`again` / `hard` / `good`). The server never judges an answer;
  it records the learner's own rating and projects it into core `AssessmentRecord`s.
- LLM generation (`review_generate`) is optional and needs a configured model plus a knowledge
  base. Everything else works offline.

## Layout

| Path | Contents |
|---|---|
| `src/deeptutor/review/` | service layer, API, MCP server, templates, export |
| `src/deeptutor/review/agent_tools.py` | the ten MCP tools |
| `src/deeptutor/review/mcp_server.py` | stdio / Streamable HTTP entry point |
| `src/deeptutor/modules/manifests/review.json` | module manifest read by the core loader |
| `tests/` | pytest suite (`python -m pytest -q`) |
| `examples/demo/` | fixture pack seeder and end-to-end MCP smoke test |

## Version pin

The wheel version must equal `deeptutor.__version__` exactly, otherwise the core module loader
skips it (`module_enabled("review")` is `False`) and `/api/review/*` is not mounted.

## Install and release

```bash
python -m pip install .            # NOT -e: the wheel merges the module manifest
deeptutor-review-catalog           # registers the MCP store entry once
python -m pip wheel . --no-deps -w dist
```

- CI installs non-editable, runs `deeptutor-review-catalog`, then `pytest` and the demo smoke test.
- `scripts/build-book-sections.py` and `scripts/build-book-figures.py` regenerate the optional
  `book-sections.json` / `book-figures.json` indexes from a local library; both files are optional
  and the server falls back to plain document names when they are absent.

## Known limits

- Single tenant: one instance serves one learner; no multi-user isolation.
- Plans schedule chapters, not individual knowledge points.
- Printing a large pack (hundreds of formulas) in a mobile browser may stall; use a desktop.
- tikz figures need `xelatex` + `pdftocairo`; without them the export shows a placeholder and a warning.

## Verification checklist

```bash
python -m pytest -q                       # full suite
python examples/demo/smoke.py             # starts the real server over stdio
```
