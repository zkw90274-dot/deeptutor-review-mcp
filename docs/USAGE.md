# 安装与使用（Review Studio）

面向第一次安装的人。目标是三件事：装得上、起得来、接得进（Web + MCP）。
每天怎么用见[操作手册](usage-guide.md)；工程细节见[维护备注](https://github.com/zkw90274-dot/deeptutor-review-mcp/blob/main/docs/maintainer-notes.md)。

## 0. 前置条件

| 项 | 要求 |
|---|---|
| Python | 3.11+（项目在 3.12 上开发与验证） |
| 操作系统 | Windows / Linux / macOS 均可；示例同时给 PowerShell/Bash 两种写法 |
| 核心 | `deeptutor` 已安装且可启动（本模块依赖它，不单独运行） |
| 端口 | 后端默认 8001，前端默认 3782 |

## 1. 安装

三条路任选一条；共同规则：**review 的版本必须等于 `deeptutor.__version__`**，
否则核心不会挂载模块（`module_enabled("review")` 返回 False，页面 404）。

### 1.1 从源码安装（开发/自用推荐）

```bash
pip install "deeptutor==1.7.3"
pip install .        # 在本仓库根目录；不要用 -e
```

### 1.2 构建 wheel 再安装（便于分发）

```bash
# 在本仓库根目录
python -m pip wheel . --no-deps --no-build-isolation -w dist
python -m pip install --force-reinstall --no-deps dist/deeptutor_review-1.7.3-py3-none-any.whl
```

### 1.3 从已发布的包安装

若 `deeptutor-review` 已发布到你可访问的索引（当前仓库不自带 PyPI 发布，公开使用请走 1.1 或 1.2）：

```bash
pip install "deeptutor-review==1.7.3"
```

### 安装后检查

```bash
python -c "import deeptutor; print(deeptutor.__version__)"
python -c "from importlib.metadata import version; print(version('deeptutor-review'))"
```

两个版本号必须一致。装好后重启 `deeptutor`，核心会在启动时挂载 review 模块；
不需要额外的目录同步命令（商店补丁 `deeptutor-review-catalog` 不在公开包内，
仅存在于仓库的实验目录）。

## 2. 配置

全部通过环境变量，没有必须修改的配置文件。

| 变量 | 默认 | 作用 |
|---|---|---|
| `DEEPTUTOR_HOME` | 仓库根/当前数据目录 | 核心数据根；复习库默认落在其 `workspace/review` 下 |
| `DEEPTUTOR_REVIEW_HOME` | `<workspace>/review` | 复习库目录（测试与多实例隔离用） |
| `AUTH_ENABLED` | 由核心决定 | 本机单人使用可设 `false` |
| `DEEPTUTOR_REVIEW_MCP_TOKEN` | 空 | 仅 HTTP 传输：非空即启用 Bearer 校验 |
| `DEEPTUTOR_REVIEW_MCP_CONTEXT` | 空 | 追加到 MCP 说明的本地文本文件路径（可选） |

数据只有一处：`<review home>/review.sqlite3`。删除它等于清空全部复习记录。

## 3. 启动

```bash
deeptutor start            # 前台
deeptutor start --detach   # 后台
deeptutor stop             # 停止
```

打开 <http://127.0.0.1:8001/api/review/ui> 看到复习包列表即安装成功。
用 `curl` 探本机端口时加 `--noproxy '*'`（或 `--noproxy 127.0.0.1`），
否则系统代理可能劫持回环地址。

## 4. 接入 MCP

### 4.1 stdio（本机客户端）

```json
{
  "mcpServers": {
    "deeptutor-review": {
      "command": "python",
      "args": ["-m", "deeptutor.review.mcp_server"],
      "env": { "DEEPTUTOR_HOME": "/path/to/deeptutor" }
    }
  }
}
```

Windows 下 `command` 建议写虚拟环境里的解释器绝对路径
（例如 `C:\path\to\repo\.venv\Scripts\python.exe`），避免 PATH 里是另一个 Python。

### 4.2 Streamable HTTP（远程/常驻）

```bash
python -m deeptutor.review.mcp_server --transport streamable-http
# 默认 127.0.0.1:8767/mcp；--host/--port/--path 可改
```

需要鉴权时设 `DEEPTUTOR_REVIEW_MCP_TOKEN=<随机串>`，客户端用
`Authorization: Bearer <同值>`。安全边界由部署方负责（TLS、反代、来源限制）。

### 4.3 服务端下发什么

`initialize` 返回的 `instructions` 是**通用**的：复习库位置、会话流程、
两条红线（不评分、不提前给答案）、以及 `review_generate` 的章节名规则。
想补充本机目录或课程约定，写 `<review home>/mcp-context.md`，或让
`DEEPTUTOR_REVIEW_MCP_CONTEXT` 指向一个文件；内容会追加在通用说明之后。

### 4.4 十个工具

| 工具 | 用途 |
|---|---|
| `review_specs` | 列出复习包（id、题数、概念数、质量状态） |
| `review_today` | 今日队列；答案与解析刻意不下发 |
| `review_attempt` | 记录三档自评（`again`/`hard`/`good`），幂等键 `submission_id` |
| `review_hint` | 两级提示（L1 点名考点，L2 复述定义/公式），泄露检查失败则回落章节指引 |
| `review_weakness` | 只读薄弱概念排序，不写库 |
| `review_export_link` | 单文件离线 HTML 链接 |
| `review_export_tree` | 按科目分类导出整批包：`<out>/<科目>/<章>.html` + `index.html`，每次整树重建 |
| `review_book_figures` | 活书插图索引：哪一节有哪几张图（按科目 / 计划章过滤），只读 |
| `review_generate` | 从知识库某一章生成新包（同章重复需 `replace=true`） |
| `review_plan` | 多周计划：create/today/complete/replan/confirm_note/status/sync_calendar |

工具行为与参数的完整说明见 [SKILL.md](https://github.com/zkw90274-dot/deeptutor-review-mcp/blob/main/docs/skill/SKILL.md)。

## 5. 第一场复习

1. 在核心知识库里准备好材料（PDF/Markdown 均可，走核心的入库流程）。
2. 用 `review_generate`（或 REST `POST /api/review/imports/book`）生成包。
   `chapter` 必须是知识库按文件布局推导出的章名：`第N章`、`第N讲`、`chN`、
   `实验N_xxx`；对不上会返回 `chapter ... not found`。
3. `review_specs` 找到新包 id → `review_today` 取队列（新包首次作答前没有到期卡）。
4. 逐题作答后 `review_attempt` 记录自评；不会时先走 `review_hint` L1，再 L2。
5. 想要离线页：`review_export_link`，得到单个 HTML，断网可开，可打印成 PDF；要给一整批包做分类目录，用 `review_export_tree`（默认写到 `<review home>/exports`）。

## 6. 数据与备份

```bash
# 备份（服务运行中也可做，SQLite 单文件）
python -c "import sqlite3,sys; s=sqlite3.connect(r'<review home>/review.sqlite3'); d=sqlite3.connect(r'review-backup.sqlite3'); s.backup(d); d.close(); s.close()"
```

- 迁移机器：复制整个 `<review home>`（库 + 导出的 HTML 都在里面或在其旁边）。
- 恢复：停服务 → 用备份覆盖 `review.sqlite3` → 启动。
- 完整性自查：`PRAGMA integrity_check;` 应返回 `ok`。

## 7. 升级与卸载

```bash
pip install --force-reinstall "deeptutor-review==<新版本>"
pip uninstall -y deeptutor-review
```

升级前先备份 `review.sqlite3`；库结构有版本号（`review_schema_migrations`），
降级不保证兼容。卸载只删代码，不动 `<review home>` 下的数据。

## 8. 排查

| 现象 | 原因与处理 |
|---|---|
| `/api/review/ui` 404 | 版本不一致（review ≠ 核心），或模块未启用；核对两个版本号后重启 |
| 页面能开但导出 404 | 导出文件在旧路径；重新导出，新页面内嵌 MathJax |
| MCP 连上但看不到 `review_plan` | 核心与 review 版本不一致，或模块未启用；先跑 `--check` 并重启（商店补丁已移到实验目录，不随包发布） |
| MCP 工具调用报 unknown spec | id 必须原样取自 `review_specs` |
| 计划提示「排不下」 | 延长 `end_date` 或提高 `daily_minutes`（上限 240/天） |
| 日历看不到学习任务 | 创建计划时核心服务未运行，`sync` 失败；重启后 `sync_calendar` |
| 生成失败且提示门禁 | 答案键冲突/公式分隔符不配对/解析空洞；换更小 `num_questions` 重试 |
| 生成失败：`The reasoning_content in the thinking mode must be passed back` | DeepTutor 1.7.3 的 thinking 模型（如 `deepseek-flash`）在 quiz pipeline 续轮丢 reasoning；改用 `deepseek-chat` 生成，或等上游 [HKUDS/DeepTutor#2766](https://github.com/HKUDS/DeepTutor/issues/2766) 修复 |
| Windows 端口被占用 | 换 `--port`，或先 `deeptutor stop` 再启动 |
