# Changelog

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
