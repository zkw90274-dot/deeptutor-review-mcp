# Review Studio（复习工坊）· 使用说明

> 把教材章节变成**能离线打开、能记录掌握情况、能排学习计划**的复习页。
> 这份是给人看的操作手册；工程细节见[维护备注](https://github.com/zkw90274-dot/deeptutor-review-mcp/blob/main/docs/maintainer-notes.md)。

## 1. 它是什么

| 能做 | 说明 |
|---|---|
| 生成复习包 | 选一个知识库章节，后台生成 5–20 道题（选择 / 简答），过质量门禁才入库 |
| 每天复习 | 按到期日排出今日队列，三档自评（会了 / 费劲 / 不会），自动安排下次复习 |
| 离线导出 | 单文件 HTML，断网可开，内嵌 MathJax，可打印成 PDF |
| 多周学习计划 | 说清范围与结束日期，自动排学习 + 复习，落到日历、看板和笔记本 |
| 薄弱点与指标 | 只读的薄弱度排序、会话完成率、预测校准 |

**不做什么**：不给你打分（三档自评全部来自你自己的判断）；不替你做计划外的生成；不做多用户隔离（单租户，一台机器一个人用）。

## 2. 启动与入口

```bash
# 在仓库根目录（Windows）
.venv\Scripts\deeptutor.exe start --detach     # 启动
.venv\Scripts\deeptutor.exe stop               # 停止

# macOS / Linux
deeptutor start --detach
deptutor stop
```

启动后：

| 入口 | 地址 | 用途 |
|---|---|---|
| 复习工坊总览 | <http://127.0.0.1:8001/api/review/ui> | 复习包一览，点「导出离线复习页」 |
| 工作台 | <http://127.0.0.1:3782> | `/chat` 对话、`/calendar` 日历、`/kanban` 看板、`/co-writer` 笔记本 |
| 后端 API | `http://127.0.0.1:8001/api/review/...` | 给脚本或 MCP 用 |

> 用 `curl` 探本机端口时加 `--noproxy '*'`，否则系统代理会劫持 `127.0.0.1`。

## 3. 五分钟走通

1. 打开总览页，挑一个包，点「导出离线复习页」→ 得到一个 `.html`，双击就能看。
2. 页面上点「开始复习」，读完题自己判断：**会了 / 费劲 / 不会**。
3. 答错的题会出现「关键点复述」，选一个最接近的，再想一步。
4. 底部进度条会记录本次会话；关掉页面，明天再打开就是新一天的队列。

## 4. 生成复习包（`review_generate`）

三种调用模式，都是一个工具：

| 你想要的 | 调用 | 返回 |
|---|---|---|
| 看这个库有哪些章 | `kb="模拟电路"` | 章节列表（含每章资料量） |
| 生成某一章 | `kb="模拟电路", chapter="第3章_小信号放大器"` | `job_id`，后台开始跑 |
| 查进度 | `kb="模拟电路", job_id="..."` | `status` / 进度 / 失败原因 |

要点：

- 章节名必须**原样**用列表返回的字符串，自己改写会匹配不到。
- 生成需要几分钟，期间可以做别的；**只有轮询到 done 才算成功**，失败的包不入库。
- `num_questions` 会被夹到 5–20；`question_types` 可选 `choice` / `written`。
- 同一章重复生成会命中同一个包 id；要覆盖必须显式 `replace=true`。
- 生成内容受门禁约束（数学分隔符必须配对、答案必须覆盖、解析不能空洞等）；不过门禁的内容不会进库。
- 核心 `1.7.3` 使用 `deepseek-flash` 等 thinking 模型出题时，可能报 `reasoning_content in the thinking mode must be passed back`。可用 `deepseek-chat` 生成。[上游问题](https://github.com/HKUDS/DeepTutor/issues/2766)

## 5. 每天复习

### 5.1 三种阅读模式

页面顶部三个按钮：

| 模式 | 显示什么 | 记不记「会了」 |
|---|---|---|
| **自测** | 题干 + 选项 + 自评按钮；答案折叠 | 你自己点评分才记 |
| **速览** | 题干 + 答案 + 解析，隐藏选项与评分 | 不记，只写本地 `viewed` |
| **精读** | 在速览基础上多一个「自述」输入框 | 不记，只写本地 `self_explain` |

速览与精读都**不会**污染复习历史——它们只是给你看的。

### 5.2 三档自评

- **会了**：不费力就想起来了 → 下次间隔拉长
- **费劲**：想了一会儿 / 有点犹豫 → 间隔小幅推进
- **不会**：没想起来 → 很快再来，并记录一次 lapse

评分必须来自你本人。AI 只负责转述你的原话，不会替你判断。

### 5.3 纠错复述

答错或点「不会」后出现一组关键点选项，选一个最接近的，页面会告诉你正确结论。这是为了让你**重新组织一次答案**，而不是只看解析。

### 5.4 进度与薄弱点

- 顶部胶囊：`已复习 X / N 个知识点`、`每日预算`、`目标记忆率`、明天预计到期数
- 底部 dock：本次会话进度；全部「会了」显示「全部很稳」，否则「很稳 N · 下次再见 M」
- 想看**哪些概念最弱**：`review_weakness`，按遗忘风险和错误率排序

## 6. 多周学习计划（`review_plan`）

### 6.1 创建

跟 AI 说清楚三件事：**学什么、学到哪、什么时候学完**。

> 「我要学模拟电路，第 1 到第 4 章，11 月 30 日前学完，每天一小时。」

对应参数：

| 参数 | 含义 | 默认 |
|---|---|---|
| `subject` | 知识库名，如 `模拟电路` | 必填 |
| `chapters` | 章节名列表（原样） | 必填 |
| `start_date` | 开始日 `YYYY-MM-DD` | 今天 |
| `end_date` | 结束日 | 必填 |
| `daily_minutes` | 每天总预算（含复习） | 60（范围 15–240） |

排期规则：按 `raw/` 资料字数估算（500 字/分钟），新学 : 复习 ≈ 6 : 4，每 7 天插一个半量缓冲日。**排不下会直接拒绝**并给出建议结束日，不会把内容硬压进几天。

创建后返回 `coverage`——每章有没有现成的复习包、包里有几个知识点。没有包的章节会被点名，AI 不应该向你承诺那章的复习清单。

### 6.2 会发生什么

| 位置 | 结果 |
|---|---|
| 日历 `/calendar` | 每天一个**整天**事件：`学习：模拟电路 · 24 分钟（2026-11-12）` |
| 看板 `/kanban` | 每天一张卡，备注列出当天任务 + 当天到期复习题数 |
| 笔记本 `/co-writer` | 整章学完后生成笔记草稿，确认后写入「模拟电路 · 学习笔记」 |

### 6.3 每天怎么用

| 说法 | 动作 |
|---|---|
| 「今天学什么」 | `today`：当天任务 + 逾期任务 + 到期复习题数 + 各章知识点 |
| 「学完了」 | `complete`：标记完成，可带一句 `self_note`（会进笔记） |
| 「笔记呢」 | `confirm_note`：把草稿写进笔记本 |

### 6.4 调整

| 情况 | 做法 |
|---|---|
| 进度落后 / 想改节奏 | `replan`，给新的 `end_date` 或 `daily_minutes`；**已完成的任务和笔记会保留** |
| 一周作息固定（周一 90、周二 120、周六 240…） | `replan` 传 `weekly_budget`：周一到周日 7 个分钟数、逗号分隔、`0` 表示那天休息（如 `90,120,90,120,90,120,360`），结束日自动算到内容排完为止；某天特殊就直接给 `day_capacities`（`日期=分钟`） |
| 某天没空 | 在日历上把那天的事件**整天拖**到别的日子，然后 `sync_calendar` 读回计划 |
| 想看整体进度 | `status`：所有计划、每天完成情况 |

**边界**：只识别整天拖动；单任务拖动、跨天拆分、拖到过去都不回写；目标日已有任务时**拒绝合并**，避免两天工作量叠一起。

## 7. 十一个工具速查

| 工具 | 什么时候用 | 关键参数 |
|---|---|---|
| `review_specs` | 看有哪些复习包 | 可选 `limit`、`offset` |
| `review_due` | 跨包看今日待练题量，保留会话状态 | 可选 `subject`、`limit`、`offset` |
| `review_today` | 今天该复习什么 | `spec_id` |
| `review_attempt` | 记录自评 | `spec_id, item_id, rating, session_id` |
| `review_hint` | 卡住了要提示 | `spec_id, item_id, level(1 概念方向 / 2 定义公式)` |
| `review_weakness` | 哪些概念最弱 | `spec_id` |
| `review_export_link` | 要导出链接 | `spec_id` |
| `review_export_tree` | 要整批分类导出 | `out_dir`（省略用 `<review home>/exports`）、`subjects`（省略＝全部科目） |
| `review_book_figures` | 查活书插图在哪一节 | `subject`（省略＝全部）、`chapter`（计划章名，省略） |
| `review_generate` | 生成新包 | `kb`（列章节）→ `kb+chapter`（开始）→ `kb+job_id`（轮询） |
| `review_plan` | 学习计划 | `action` = create / today / complete / replan / confirm_note / status / sync_calendar |

提示阶梯：先给 1 级（方向），不够再给 2 级（定义/公式）。用了提示会在评分里体现，用于校准。

## 8. 导出与打印

- 导出：总览页点按钮，或 `GET /api/review/{spec_id}/export`，存成 `.html`。
- 整批导出：`review_export_tree`（本仓库也用它：`python scripts/export-review-packs.py`）写 `<out>/<科目>/<章>.html` + `<out>/index.html`；科目与章名从包标题拆出，每次整树重建，改名不留旧页。
- 单文件、约 2.2 MB（内嵌 MathJax），**断网可用**，不发任何外部请求。
- 打印：浏览器打印即可。**自测模式打印是空白卷**（隐藏答案），速览/精读打印含答案。
- 打印已做：强制浅色、隐藏交互控件、长公式不再内部滚动、图表与表格整卡不跨页。

## 9. 界面与主题

- 跟随系统深浅色；右上角按钮可手动切换。
- 窄屏（≤640px）：超过一屏宽的公式**在自身容器内横向滚动**，不会把整页撑歪。
- 长概念名完整显示（上限 200 字），不会再被截出省略号。
- 每页只有一个强调色（青绿 `#0f766e`，深色 `#5ec8bd`）；卡片 22px 圆角，内层 14px；吸顶顶栏 + 章节胶囊。

## 10. 数据、备份与恢复

| 数据 | 位置 |
|---|---|
| 复习历史（事件 / 会话 / 计划 / 笔记草稿） | `<deeptutor home>/workspace/review/review.sqlite3` |
| 日历 | `<deeptutor home>/.runtime/calendar/calendar.sqlite` |
| 看板 | `<deeptutor home>/.runtime/task-board/tasks.sqlite` |
| 笔记本 | `<deeptutor home>/workspace/co-writer/documents/` |

备份：停止服务后直接复制这几个文件；恢复同理。SQLite 单文件，`PRAGMA integrity_check` 可自检。

**不要手工编辑** `review.sqlite3` 的事件表——复习间隔由事件重放得出，改一行会让到期日失去依据。

## 11. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `curl` 连不上 8001，但浏览器能开 | 系统代理劫持了 `127.0.0.1`，加 `--noproxy '*'` |
| 生成任务一直 running | 后台任务在跑，继续轮询；超过预期就是上游超时，重新生成 |
| 生成直接失败 | 内容没过门禁（数学分隔符不配对、答案覆盖不足等），换章节资料或重试 |
| 导出页公式显示成原文 | 打开的是旧文件；重新导出。新页面内嵌了 MathJax 与扩展宏 |
| 日历上看不到学习任务 | 计划创建时 `sync` 里 `calendar.ok=false`，检查核心服务是否在跑 |
| 拖了日历但计划没变 | 只识别**整天拖动**；拖完要调一次 `sync_calendar` |
| 笔记找不到 | 草稿在复习库里，`confirm_note` 之后才进 `/co-writer` |
| 提示「排不下」 | 延长 `end_date` 或提高 `daily_minutes` |

## 12. 边界与不要做的事

- **单租户**：一个实例只给一个人用，不要共享给多个人。
- **AI 不判分**：所有评分必须是你自己的判断。
- **不自动生成**：缺复习包的章节不会被偷偷生成，会明确告诉你「这章还没有包」。
- **不做逐知识点排期**：计划到章，不到单个知识点。
- **移动端打印**：整包（2.2 MB / 300+ 公式）在手机浏览器上打印预览可能停在 preparing，**整包打印请用桌面端**。

## 13. 导出目录约定

复习包导出到哪个目录由你自己定（`review_export_tree` 的 `out_dir` 参数）。整树导出按下面的结构落盘：

```
<out_dir>/
  index.html            # 科目 → 章 → 包 的索引页
  <科目>/
    <章>.html           # 单文件，公式/样式内嵌，双击即看
```

- 科目与章名取自复习包标题（`<科目> · <章> 复习包`），可用 `subjects` 只导部分科目。
- 每次整树导出会重建 `index.html` 并清掉本次没写的旧页（同科目下），所以目录只放导出物。
- 导出页是离线单文件，手机上也能开；整包体积大（2 MB 量级、公式数百条）时，打印预览请用桌面端。

## 附录 A：环境变量

| 变量 | 作用 |
|---|---|
| `DEEPTUTOR_HOME` | 核心数据根目录（默认仓库根） |
| `DEEPTUTOR_REVIEW_HOME` | 复习库目录（默认走核心 workspace） |
| `AUTH_ENABLED` | 本地单人用设 `false` |
| `DEEPTUTOR_REVIEW_CACHE` | 设置后启用 TikZ 渲染缓存（内容寻址） |
| `DEEPTUTOR_REVIEW_MCP_TOKEN` | HTTP 传输的可选 Bearer 令牌 |

## 附录 B：验收脚本

```bash
# 在本仓库根目录
.venv\Scripts\python.exe scripts/accept_all.py          # 6 步全链路
.venv\Scripts\python.exe scripts/audit_content_fidelity.py \
    --home <review home>                                # 内容一致性审计
.venv\Scripts\python.exe -m pytest -q                   # 全量
```

## 附录 C：MCP 接入

```bash
# stdio（Claude Code / 本机客户端）
python -m deeptutor.review.mcp_server

# Streamable HTTP（默认 127.0.0.1:8767/mcp）
python -m deeptutor.review.mcp_server --transport streamable-http
```

MCP 客户端可以连接 stdio 或 Streamable HTTP。11 个工具包含 `review_plan` 和跨包只读汇总 `review_due`。
商店补丁（`deeptutor-review-catalog`）已移到实验目录，不随公开包发布。

## 附录 D：相关文档

- [维护备注](https://github.com/zkw90274-dot/deeptutor-review-mcp/blob/main/docs/maintainer-notes.md)：架构不变量、发布步骤、已知限制
- [安装与使用](USAGE.md)：安装、配置、MCP 接入、备份与排查
- [Agent skill](https://github.com/zkw90274-dot/deeptutor-review-mcp/blob/main/docs/skill/SKILL.md)：给 agent 的调用协议
- [CHANGELOG](https://github.com/zkw90274-dot/deeptutor-review-mcp/blob/main/CHANGELOG.md)
- [SECURITY](https://github.com/zkw90274-dot/deeptutor-review-mcp/blob/main/SECURITY.md)
- 上游 issue：[HKUDS/DeepTutor#2766](https://github.com/HKUDS/DeepTutor/issues/2766)（1.7.3 thinking 模型生成问题）
