# deeptutor-review-mcp

[![CI](https://github.com/zkw90274-dot/deeptutor-review-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/zkw90274-dot/deeptutor-review-mcp/actions/workflows/ci.yml)

A standalone **MCP server for spaced-repetition review**: it turns course material into *review packs* and gives any MCP-capable agent the tools to quiz you, record your own three-level self-assessment (again / hard / good), track weak concepts, and export a printable page. It never grades for you.

Ten tools, no DeepTutor web server required, no account required. Works over **stdio** (default) or **Streamable HTTP**.

```
review_specs  review_today  review_attempt  review_hint  review_weakness
review_export_link  review_export_tree  review_book_figures  review_generate  review_plan
```

---

## 1. Install

Python **3.11+**.

```bash
git clone https://github.com/zkw90274-dot/deeptutor-review-mcp.git
cd deeptutor-review-mcp
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
pip install "git+https://github.com/zkw90274-dot/deeptutor-review-mcp.git"
```

### DeepTutor desktop app (optional)

The package also ships an entry definition for the DeepTutor desktop app's MCP
store. If you use that app and want the review server to show up there:

```bash
deeptutor-review-catalog     # patches the app's catalog once, safe to re-run
```

Everything below works without this step — plain MCP clients connect directly.

## 2. Prove it works (no model, no knowledge base)

```bash
python examples/demo/seed.py     # writes demo-review-home/ with a 3-question pack
python examples/demo/smoke.py    # seeds a temp home, starts the MCP server, calls 3 tools
```

`smoke.py` starts the real server over stdio, checks that all ten tools are advertised,
lists the pack and pulls today's queue. Exit code 0 means the whole surface is wired.

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
No token set means no auth — fine for loopback, not for anything reachable.

Then say to the agent:

> List my review packs, then quiz me on the ones I have today.

## 4. What the ten tools do

| Tool | Purpose | Needs a model? | Needs a knowledge base? |
|---|---|---|---|
| `review_specs` | List stored packs (id, title, item/concept counts, gate result) | no | no |
| `review_today` | Today's queue for one pack, answers withheld | no | no |
| `review_attempt` | Record the learner's own rating, return mastery projection + correction | no | no |
| `review_hint` | Hint ladder (L1 pointer → L2 narrowing) | no | no |
| `review_weakness` | Rank weak concepts from history | no | no |
| `review_export_link` | One-pack offline HTML (single file) | no | no |
| `review_export_tree` | Whole library as `<subject>/<chapter>.html` + index | no | no |
| `review_book_figures` | Which figures a live-book section has | no | no |
| `review_generate` | Generate a new pack from one KB chapter (background job) | **yes** | **yes** |
| `review_plan` | Multi-week study plan, mirrored to calendar/task board | no (drafting a chapter note does) | **yes** |

So: if you already have packs, nine of ten tools run fully offline. `review_generate`
and the `create` action of `review_plan` read a DeepTutor knowledge base
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

## 6. Slightly deeper

- `docs/usage-guide.md` — the full operator guide: pack lifecycle, quality gates, plan templates, exports, troubleshooting.
- `docs/USAGE.md` — tool-by-tool parameters and return shapes.
- `docs/skill/SKILL.md` — the agent-facing skill: how an agent should talk to a learner while reviewing.
- `docs/maintainer-notes.md` — validation log and design decisions from this repo's own runs.

## 7. Development

```bash
pip install -e ".[test]"
python -m pytest -q                 # unit + integration suite
python examples/demo/smoke.py        # end-to-end MCP handshake
```

CI runs both on Python 3.11 and 3.12 (see `.github/workflows/ci.yml`).

## 8. 中文速览

这是把「复习工坊」单独拆出来的 **MCP 服务器**，不需要 DeepTutor 网页端，也不依赖任何账号：

```bash
pip install -e .
python examples/demo/seed.py          # 造一个 3 题的演示复习包
deeptutor-review-mcp                  # stdio 启动，配置进你的 MCP 客户端即可
```

十个工具里九个离线可用（列包、今日队列、作答自评、分级提示、薄弱点、导出单包/整库、活书插图）。只有「从知识库生成新复习包」需要知识库 + 模型配置；「排多周学习计划」需要知识库章节。所有数据都在 `DEEPTUTOR_REVIEW_HOME` 指向的目录里，备份它等于备份全部复习记录。

## License

MIT — see `LICENSE`.
