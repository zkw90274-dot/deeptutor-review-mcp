# Security Policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private
vulnerability reporting on this repository (Security -> Report a vulnerability),
or contact the maintainer through the same GitHub account.

Include the affected version (`python -m deeptutor.review.mcp_server --check`
prints it), the transport (stdio or Streamable HTTP), and a minimal
reproduction. Do not include learner data, tokens or database contents.

## Scope

The review store is local and single-tenant. The HTTP transport binds
`127.0.0.1` by default; if you expose it beyond loopback, set
`DEEPTUTOR_REVIEW_MCP_TOKEN` — the server refuses to start on a non-loopback
address without it. `review_export_tree` refuses paths outside
`<review home>/exports` unless `--allow-out-dir` is passed.
