# deeptutor-review-mcp

[![CI](https://github.com/zkw90274-dot/deeptutor-review-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/zkw90274-dot/deeptutor-review-mcp/actions/workflows/ci.yml)

A standalone **MCP server for spaced-repetition review**: it turns course material into *review packs* and gives any MCP-capable agent the tools to quiz you, record your own three-level self-assessment (again / hard / good), track weak concepts, and export a printable page. It never grades for you.

Eleven tools, no account required. **Ten of them need no DeepTutor web server**; `review_export_link` returns a URL served by the DeepTutor web app, so that one tool does need it (see §4). Works over **stdio** (default) or **Streamable HTTP**.

```
review_specs  review_due  review_today  review_attempt  review_hint  review_weakness
review_export_link  review_export_tree  review_book_figures  review_generate  review_plan
```

---

## 1. Install

Python **3.11+**.

```bash
git clone https://github.com/zkw90274-dot/deeptutor-review-mcp.git
cd deeptutor-review-mcp
git checkout v1.7.3-r3
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install .
```

This installs the MCP entry point `deeptutor-review-mcp` plus its dependencies from PyPI
(`deeptutor`, `deeptutor-mastery`, `fastapi`, `pydantic`).

> Use a regular install, not `-e`: the wheel also drops the module manifest into the
> shared `deeptutor` namespace, and an editable install does not merge that data file.
> Editing the source in place is still fine — re-run `pip install .` after changes.

Or install straight from git:

```bash
pip install "git+https://github.com/zkw90274-dot/deeptutor-review-mcp.git@v1.7.3-r3"
```

For an existing DeepTutor 1.7.3 environment, download
`deeptutor_review-1.7.3-3-py3-none-any.whl` from the [r3 release](https://github.com/zkw90274-dot/deeptutor-review-mcp/releases/tag/v1.7.3-r3)
and install it with the target environment's Python:

```bash
python -m pip install --force-reinstall --no-deps deeptutor_review-1.7.3-3-py3-none-any.whl
deeptutor-review-mcp --check
```

The wheel build tag `3` identifies r3. The installed version remains `1.7.3`
to match the DeepTutor module gate.

### DeepTutor desktop app

The MCP store entry (`deeptutor-review-catalog`) is **not** part of the published
package: it lives in the repository's experimental directory and is not built
into the wheel. Plain MCP clients connect directly — see section 3.

## 2. Prove it works (no model, no knowledge base)

```bash
python examples/demo/seed.py     # writes demo-review-home/ with a 3-question pack
python examples/demo/smoke.py    # seeds a temp home, starts the MCP server, calls 3 tools
```

`smoke.py` starts the real server over stdio, checks that every advertised tool is present,
lists the pack, summarises due work and pulls today's queue. Exit code 0 means the whole surface is wired.

## 3. Connect your agent

**stdio** (Claude Desktop, DSH, Cursor, Claude Code, …):

```json
{
  "mcpServers": {
    "deeptutor-review": {
      "command": "deeptutor-review-mcp",
      "args": ["--transport", "stdio"],
      "env": {
        "DEEPTUTOR_REVIEW_HOME": "/absolute/path/to/deeptutor-review-mcp/demo-review-home"
      }
    }
  }
}
```

**Streamable HTTP** (shared box, reverse proxy in front):

```bash
export DEEPTUTOR_REVIEW_MCP_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
deeptutor-review-mcp --transport streamable-http --host 127.0.0.1 --port 8767 --path /mcp
```

Endpoint: `http://127.0.0.1:8767/mcp`, header `Authorization: Bearer <token>`.
Loopback HTTP permits an unset token. An explicitly blank token fails startup.
Non-loopback HTTP requires a token. Supply the forwarded Host and browser Origin
as exact values with `--allowed-host` and `--allowed-origin` when using a proxy.

`--read-only` opens an existing, current-schema review database in SQLite read-only
mode. Initialize or migrate the database separately before using this flag.
The capability flags apply to MCP and in-app agent tools. DeepTutor REST routes
use the app's authentication and permissions.

Then say to the agent:

> List my review packs, then quiz me on the ones I have today.

## 4. What the eleven tools do

| Tool | Purpose | Needs a model? | Needs a knowledge base? | Needs the DeepTutor web app? |
|---|---|---|---|---|
| `review_specs` | List stored packs (id, title, item/concept counts, gate result) | no | no | no |
| `review_due` | Read-only summary of which packs have work today, across the whole library | no | no | no |
| `review_today` | Today's queue for one pack, answers withheld | no | no | no |
| `review_attempt` | Record the learner's own rating, return mastery projection + correction | no | no | no |
| `review_hint` | Hint ladder (L1 pointer → L2 narrowing) | no | no | no |
| `review_weakness` | Rank weak concepts from history | no | no | no |
| `review_export_link` | URL of one-pack offline HTML | no | no | **yes** |
| `review_export_tree` | Whole library as `<subject>/<chapter>.html` + index | no | no | no |
| `review_book_figures` | Which figures a live-book section has | no | no | no |
| `review_generate` | Generate a new pack from one KB chapter (background job) | **yes** | **yes** | no |
| `review_plan` | Multi-week study plan, mirrored to calendar/task board | no (drafting a chapter note does) | **yes** | no |

`review_specs` and `review_due` return up to 50 packs by default. Pass `limit`
from 1 to 100 and pass the returned `next_offset` as `offset` to get another
page. `count` and `review_due.totals` cover all matching packs. `review_due.omitted`
counts unreadable or invalid packs across the whole store, including when `subject`
is set. If packs or review progress change between requests, restart from offset 0
to refresh the listing.

`review_plan` takes calendar dates in `YYYY-MM-DD` form. `review_export_tree`
only replaces or removes pages it generated and can still verify by checksum.
Use a fresh output folder for your first export from this release if the old
folder already contains exports. A same-name conflict preserves the existing file.

REST clients can start a review session with `POST /api/review/{spec_id}/today`.
For a preview, use `GET /api/review/{spec_id}/session` or `GET /api/review/{spec_id}/today?read_only=true`.
The legacy bare `GET /today` continues to open a session and record views.

So: if you already have packs, ten of the eleven tools run fully offline.
`review_export_link` is the exception: it returns `/api/review/{spec_id}/export`, a route the
DeepTutor web app serves behind login. Without that app running, use `review_export_tree`
(writes self-contained HTML to disk) or open the export page through DeepTutor itself.
`review_generate` and the `create` action of `review_plan` read a DeepTutor knowledge base
(`<workspace>/knowledge_bases/<subject>/`), and `review_generate` also needs a working model
configured for DeepTutor (`OPENAI_API_KEY` / `OPENAI_BASE_URL` or the DeepTutor settings store).

## 5. Where state lives

Everything is per-workspace, in one directory:

| Env var | Meaning |
|---|---|
| `DEEPTUTOR_REVIEW_HOME` | The review workspace (default: `<DeepTutor workspace>/review`). Holds `review.sqlite3`, exported HTML, drafts. |
| `DEEPTUTOR_REVIEW_CACHE` | Optional cache dir for rendered output. |
| `DEEPTUTOR_REVIEW_MCP_TOKEN` | Bearer token for the HTTP transport. |
| `DEEPTUTOR_REVIEW_MCP_CONTEXT` | Optional markdown file appended to the server's `instructions`. |
| `DEEPTUTOR_REVIEW_PUBLIC_BASE_URL` | Absolute prefix for `review_export_link` URLs (e.g. behind a reverse proxy). |

Back up the home directory and you back up the learner's whole review history.

The SQLite store migrates itself forward on open (current schema: 5, which added
`review_events.learner_quote`). A newer database is never silently downgraded:
an older build refuses to open it with `review database schema N is newer than
this package`. Back up `review.sqlite3` before moving a workspace between
versions.

## 5.1 Known issue: DeepSeek thinking models on core 1.7.3

With DeepTutor core 1.7.3, `review_generate` and Book quiz-block generation can
fail with `The reasoning_content in the thinking mode must be passed back to
the API` when the active model is a thinking model such as `deepseek-flash`.
The failure is in the core question/quiz pipeline, not in the MCP tools; it is
tracked upstream in
[HKUDS/DeepTutor#2766](https://github.com/HKUDS/DeepTutor/issues/2766).
Until a fixed core release ships, use a non-thinking model (for example
`deepseek-chat`) for generation. The other ten tools are unaffected.

## 6. Documentation

| Need | Read |
|---|---|
| Install, configuration, backup, troubleshooting | [docs/USAGE.md](https://github.com/zkw90274-dot/deeptutor-review-mcp/blob/main/docs/USAGE.md) |
| Pack lifecycle, quality gates, plans, exports | [docs/usage-guide.md](https://github.com/zkw90274-dot/deeptutor-review-mcp/blob/main/docs/usage-guide.md) |
| Agent skill and review protocol | [docs/skill/SKILL.md](https://github.com/zkw90274-dot/deeptutor-review-mcp/blob/main/docs/skill/SKILL.md) |
| Design and validation notes | [docs/maintainer-notes.md](docs/maintainer-notes.md) |
| Security policy | [SECURITY.md](https://github.com/zkw90274-dot/deeptutor-review-mcp/blob/main/SECURITY.md) |
| Changelog | [CHANGELOG.md](https://github.com/zkw90274-dot/deeptutor-review-mcp/blob/main/CHANGELOG.md) |

> The public-only docs above live in the standalone repository; `docs/maintainer-notes.md` ships with this package.

## 7. Development

```bash
pip install -e ".[test]"
python -m pytest -q                 # unit + integration suite
python examples/demo/smoke.py        # end-to-end MCP handshake
```

CI runs Linux and Windows on Python 3.11, 3.12 and 3.13 (see `.github/workflows/ci.yml`).

## 8. 中文速览

这是把「复习工坊」单独拆出来的 **MCP 服务器**：除 `review_export_link` 外都不需要 DeepTutor 网页端，也不依赖任何账号：

```bash
pip install -e .
python examples/demo/seed.py          # 造一个 3 题的演示复习包
deeptutor-review-mcp                  # stdio 启动，配置进你的 MCP 客户端即可
```

十一个工具里十个离线可用（列包、跨包到期汇总、今日队列、作答自评、分级提示、薄弱点、导出整库、活书插图）。两处例外：「从知识库生成新复习包」需要知识库 + 模型配置；「排多周学习计划」需要知识库章节；`review_export_link` 返回的是 DeepTutor 网页端路由，需要网页服务与登录（离线导出请改用 `review_export_tree`）。所有数据都在 `DEEPTUTOR_REVIEW_HOME` 指向的目录里，备份它等于备份全部复习记录。

## License

MIT — see `LICENSE`.
