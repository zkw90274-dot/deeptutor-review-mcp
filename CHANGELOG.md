# Changelog

## v1.7.3-r3

- Paginate `review_specs` and `review_due`, with a default of 50 packs and a limit of 100 per page. Counts and due totals cover all matching packs.
- Page pack metadata in SQLite and replay review evidence in batches of up to 256 events per concept. Due totals still scan the matching packs.
- Reject unknown MCP tool arguments. Resolve the default `review_plan today` date in the plan's timezone.
- Use build tag `3` in `deeptutor_review-1.7.3-3-py3-none-any.whl`. The installed package version stays `1.7.3` for the core module gate.

## v1.7.3-r2

- Add `review_due`, a read-only summary across review packs, and tool annotations.
- Make SQLite migrations atomic across processes and reject newer schemas before writes.
- Enforce calendar-date format in study plans and return actionable validation errors.
- Preserve unmanaged and edited HTML during classified export.
- Require an initialized database for `--read-only`. Add an explicit REST session-start route and a read-only preview option.
- Reject blank HTTP bearer tokens at startup.
- Drop optional study-aid text when clipping would leave math delimiters unbalanced.
- Keep the wheel version at `1.7.3` for DeepTutor's exact-version module gate. The `v1.7.3-r2` Git tag identifies this build.

## 1.7.3 — 2026-11-17

- Track DeepTutor core 1.7.3; the review wheel version must equal the core
  version or the module does not mount.
- Publish hardening: `--check`, a pre-install guard
  (`scripts/install-review.py`), and an idempotent publish pipeline.
- Capability flags: `--read-only`, `--no-attempts`, `--no-plan`,
  `--no-generate`, `--no-export-write`, `--allow-generate`,
  `--allow-out-dir`. `review_today` no longer creates a session or records
  views under `--read-only`.
- HTTP transport: refuses to bind a non-loopback address without a bearer
  token; `review_generate` is off by default there.
- `review_export_tree` keeps writes inside `<review home>/exports` unless an
  extra root is explicitly allowed.
- The in-app manifest gains `review_book_figures`; `review_export_tree`
  remains MCP-only by design.
- The MCP store sideload patch (`deeptutor-review-catalog`) is removed from
  the published package and kept under `experimental/` for future work.
- Privacy: no absolute paths, auth files, or learner pack ids in the public
  tree; example course names are illustrative only.
