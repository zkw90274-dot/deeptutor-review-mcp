---
name: review-studio
description: Run spaced review sessions for the learner and tell them what to
  review today, what they are weakest at, and when each card comes back. Also
  generates new review packs from a knowledge-base chapter or lecture group. Use when the
  learner asks for spaced review, revision, drilling, memory scheduling, their
  weak points, an offline review page, or review material for a chapter they
  have not reviewed yet.
tags: [education, review, spaced-repetition]
---

# Review Studio

Review Studio schedules the learner's review with a spaced-repetition
scheduler. These eleven tools read and write the same store as the Review Studio
overview page and the offline export, so a rating recorded here is the
learner's real review history. `review_generate` is the one tool that creates a
pack instead of consuming one.

## 资料目录（部署约定，可选）

包本身不假设任何目录名：复习库位置由 `DEEPTUTOR_REVIEW_HOME`（默认
`<workspace>/review`）决定；学习者看的成品放在哪里，由部署方自己安排。
常见做法是按编号分科目（计划 / 各科资料 / 离线导出 / 源工程 / 旧版），
但工具只依赖 `review_specs` 返回的 `spec_id`，与目录结构无关。

想把这些约定告诉接入的 agent：写 `<review home>/mcp-context.md`，服务端会把它
追加到 `initialize` 下发的通用说明后面（也可用 `DEEPTUTOR_REVIEW_MCP_CONTEXT`
指向任意文本文件）。不必改包。

生成注意：`chapter` 必须用知识库按文件布局推导出的章名（`第N章` / `第N讲` /
`chN` / `实验N_xxx`）；用空格代替下划线会得到 chapter not found。

## Tools

| Tool | Use it for | Parameters |
|---|---|---|
| `review_specs` | List the review packs available to this learner, with each pack's id, item and concept counts, and generation quality status. | None. |
| `review_due` | Summarise what is queued today across every pack, read-only: per pack the queued/due/new counts, estimated minutes, recovery flag and last review time. Use it to answer "what should I review today?" or to pick a pack; it opens no session and records nothing, unlike `review_today`. It returns no question text. | optional `subject` (substring of the pack title). |
| `review_today` | Return today's queue for one pack: the budget, planned minutes, and each due item's kind, stem and options, with answers and explanations deliberately withheld. When the pack has an exam date, the payload carries an `exam` block (`days_left`, `mode`: sprint within 14 days, final_week within 3) you can mention to the learner. | `spec_id` (verbatim from `review_specs`). |
| `review_attempt` | Record the learner's own self-assessment of one item and let the scheduler set its next review date. A `good` reported after any hint is stored as `hard`. | `spec_id`, `item_id` (both verbatim from `review_today`), `rating` (`again`, `hard`, or `good`), optional `session_id` (pass back the value from `review_today`), optional `submission_id` (idempotency key for retrying the same rating), optional `hints_used` (0-2, the number of hints you actually showed), optional `confidence` (0-100, only when the learner volunteers it, never invented). |
| `review_hint` | Give the learner a graded hint for one item without revealing the answer: level 1 names the tested concept, level 2 restates its definition (and formula when the pack has one); when the pack has no definition, level 2 points at the definition to recall and the source material instead of repeating level 1. The text is derived deterministically from the pack and leak-checked against the answer and every option; if that check fails it falls back to source-chapter guidance. | `spec_id`, `item_id` (both verbatim from `review_today`), optional `level` (`1` or `2`, default `1`; offer 1 first). |
| `review_weakness` | Show the pack's weakest concepts with their weakness score and evidence count, read-only. | `spec_id` (verbatim from `review_specs`). |
| `review_export_link` | Give the learner the URL of the printable review page for one pack. The URL is served by the DeepTutor web app, so it needs that app running and the learner signed in; in a standalone MCP deployment offer `review_export_tree` instead. The payload carries `url` (relative) and, when the deployment configures a public base URL, `absolute_url`; give `absolute_url` when present. | `spec_id` (verbatim from `review_specs`). |
| `review_export_tree` | Export packs as `<out_dir>/<subject>/<chapter>.html` and `<out_dir>/index.html`. A rebuild can replace or remove pages tracked by a valid export checksum. If an existing personal or legacy page conflicts, preserve it and choose a fresh export destination. Packs failing the QA gate are skipped and reported. | `out_dir` (optional, defaults to `<review home>/exports`), `subjects` (optional array of subject names, omit for every subject in the store). |
| `review_book_figures` | List the diagrams the living books print, section by section: each entry carries the section number, title, book anchor and figure caption, optionally filtered to one subject or one planned chapter. Figures are inline SVG inside the built book, so this is how a reading pointer can name the diagram (图 2-1) instead of sending the learner into text only. Read-only. | `subject` (optional), `chapter` (optional, a planned chapter name). |
| `review_plan` | Plan and run a multi-week study course for one subject. `create` takes the knowledge base, the chapter list, an end date and a daily budget, spreads the reading across days, and refuses a schedule that does not fit; each planned day is mirrored onto the workbench calendar and task board (best-effort). `today` returns today's study tasks, the due-review count across the subject's packs, and overdue sessions; `complete` marks one session done and drafts the chapter note once the chapter is finished; `confirm_note` writes that draft into the Co-Writer notebook; `replan` rebuilds the remaining schedule from today; `sync_calendar` pulls whole-day calendar drags back into the plan; `status` lists plans and progress. | `action` (`create`/`today`/`complete`/`replan`/`confirm_note`/`status`/`sync_calendar`), `subject` (KB name), `chapters` (verbatim from `review_generate`'s chapter list), `start_date`, `end_date`, `daily_minutes` (15-240, default 60), `weekly_budget` (for `replan`: seven comma-separated minutes, Monday..Sunday, `0` = day off, e.g. `90,120,90,120,90,120,360`; the end date is derived from the remaining work), `plan_id`, `date`, `seq`, `self_note`, `note_id` — only the ones the action needs. |
| `review_generate` | Generate a new pack from one knowledge-base chapter or lecture group. Three modes: call with `kb` only to list the chapter/lecture groups first, with `kb` + `chapter` to start a background generation job, or with `kb` + `job_id` to poll that job. Once a job is started, every call returns in seconds — the generation itself runs in the background and takes minutes. | `kb` (knowledge base name), `chapter` (verbatim from the chapter list; mutually exclusive with `job_id`), `job_id` (from the start call, to poll its status), optional `num_questions` (default 12, clamped to 5-20), optional `question_types` (`choice` and/or `written`, default both), optional `difficulty`. |

## Generating a new pack

When the learner asks for review material that does not exist yet:

1. Call `review_generate` with only `kb`. It returns the knowledge base's
   chapter/lecture/courseware groups with their document counts: `第N章`,
   `实验N`, `第N讲`, and `chN` courseware groups all appear. Quote the name
   verbatim when starting a job.
2. Ask the learner which group to use. If the list is empty, say the
   knowledge base has no chapter/lecture-structured documents and offer
   `review_specs` for the packs that already exist.
3. Tell them generation takes minutes, then call `review_generate` with `kb`
   and the chapter name verbatim. It returns a `job_id` right away and the
   generation runs in the background. Adjust `num_questions`, `question_types`,
   or `difficulty` only if they ask. Regenerating a chapter that already has a
   pack is rejected instead of silently replacing it: tell the learner, and
   pass `replace=true` only if they explicitly choose to replace that pack.
4. Poll `review_generate` with `kb` and the `job_id` every ~20 seconds, telling
   the learner it is still running between polls. A large batch can take five
   to ten minutes, so keep polling until the job reports `done` or `failed`
   (or the learner asks you to stop — then hand them the `job_id` so they can
   resume). Never start a second generation for the same chapter while one is
   in flight.
5. On `done`, the result carries the new `spec_id`, the question count, and
   whether the QA gates passed. On `failed`, report the error; a gates failure
   means nothing was stored, so do not claim the pack is ready. Retry at most
   once with fewer questions or `choice` only.
6. If a poll comes back `stale: true`, generation may have been interrupted by
   a service restart. Say so, check `review_specs` in case the pack was stored
   anyway, then offer to start the generation again.
7. On success, call `review_today` with the new `spec_id` to start the session.

## Standard session flow

1. Call `review_specs` first. Every other tool needs a `spec_id`, and only the
   ids returned here exist.
2. If the learner asks "what should I review today?" without naming a pack, call
   `review_due` instead of `review_today` for each pack: it is read-only and
   returns the queued counts across the library.
3. If more than one pack fits, ask the learner which one; do not pick for them.
4. Call `review_today` with that `spec_id`.
5. Read **one** item's stem to the learner exactly as given, including its
   options when present. Do not guess or reveal the answer, and do not
   improvise a hint: the tool does not give you the answer, and hints go
   through `review_hint` (below).
6. **Stop and wait.** The learner answers, then says in their own words how it
   went (easily, with effort, or not at all). If they say they are stuck, go to
   "Helping a stuck learner" below instead of revealing the answer.
7. Call `review_attempt` with that item's `item_id`, the rating they gave
   transcribed as `good`, `hard`, or `again`, and the `session_id` from
   `review_today` passed back verbatim. Then tell them the correct answer and
   explanation from the result and when the card returns.
8. Continue with the next item from the same `review_today` list. When the
   learner asks what to focus on, call `review_weakness`; when they want to
   print or open a page, call `review_export_link` (needs the DeepTutor web app);
   when they want the whole library as classified offline folders, call `review_export_tree`.

## Helping a stuck learner

- Offer a hint only when the learner asks for one or clearly says they are
  stuck — not on your own initiative.
- Call `review_hint` with level 1 first. Only if they are still stuck, call it
  again with level 2. Never show both levels at once, and never more than two
  hints; extending the ladder turns the exercise into a reading task.
- If they are still stuck after two hints, take the answer path through
  `review_attempt` (record `again` and read out the answer and explanation).
  Do not keep generating hints.
- Count every hint you actually showed and pass that number as `hints_used` to
  `review_attempt`. A hinted answer the learner reports as `good` is stored as
  `hard` — tell them that plainly rather than implying they recalled it unaided.
- After a correct answer you may ask one elaborative "why" question (精加工);
  after an `again` you may ask the learner to restate the key step in their own
  words (teach-back). Both are optional conversation, never graded and never
  passed to `review_attempt`.

## 计划命名规程（review_plan）

计划要说清「读什么」，而不是把时长切成段：

- 每日学习任务按原文命名——`<章> · <篇/课件/小节>`，一篇一条；禁止把一章按时长均分成「第 i/n 段」。
- 同一篇跨天时用中文序号（一）（二）（三）接续，不用（上）/（下）；同一份材料的不同版本用（v1）/（v2）区分。
- 每日备注只列当天篇目；同章多篇时章名只出现一次并标篇数，篇目缩进排列；「本章读完」标在当天该章最后一篇。
- 每天为当天涉及的每一章给出配套练习，且写清**今天做多少**而不是包有多大：该章当天收尾写「**复习包**：<章> · 今天做 N 题（一次做完）」；该章仍在进行写「**复习包**：<章> · 今天不做，学完本章后一次做完 N 题」；没有题包写「暂无复习包」。一个包装得下一次坐完（队列硬上限 12 题），只有超限的包才拆成「今天做 12 题（共 N 题，余下按到期继续）」；裸写「共 N 题」会被读成今天要做完。
- 前缀必须是「复习包：」而不是「练习：」：它指的是按遗忘曲线排期的复习包，活书里按小节放的自测题与闪卡只是随手自检、不参与排期，两者不能混为一谈。

## 主资料：交互活书

一门课**只有一份主资料**，而且是最完整的那一版。同一份材料在知识库里积累了几代时：保留最全的一代 → 把旧版独有的段落并进来 → 旧版移出知识库（归档，不放在旁边留着）→ 重索引。库里留两代，题包和计划就会从两份材料各生成一遍。

主资料是**单文件离线交互活书**，不是 PDF 或只有目录的全文页：

- 顶栏章节胶囊 + 左侧目录与搜索 + 每节「标记为已掌握」与进度环；
- 章末自动判分题、闪卡、可调组件（滑块实时重算公式，而不是静态截图）；
- 公式内联渲染，`<script src>` 与 `<img>` 都为 0（图用内联 SVG 或带页码的占位）；
- 浅色/深色两套主题，打印样式去掉顶栏与侧栏、满 A4。

视觉与复习包共用一套令牌：浅色 `#eceef1` 画布 / `#fbfbfc` 卡片 / `#f5f6f7` 内层 / `#e5e7eb` 分隔 / `#0f766e` 强调；深色 `#0b0d0f` / `#16191c` / `#1b1e22` / `#262a30` / `#5ec8bd`；圆角 22/14/9px；60px 吸顶玻璃顶栏；按钮与 chip 全 pill。

**绝不手改产物 HTML**：每本书留 build 脚本 + 内容片段（或 md）+ 题目/闪卡/组件的绑定数据，改完重建。计划任务的「资料」指向活书的节锚点（id 形如 `s<章>-<节>`），不指原始 md。

### 活书性能（构建与首屏）

- MathJax 配置 `startup: { typeset: false }`；用 IntersectionObserver（`rootMargin` 约 900px）只排接近视口的节，首屏先排前 2–3 节，并记录已排过的节；**绝不在启动时 typeset `document.body`**。
- 刚显示的块（揭示答案、翻卡）按需排版；没有标记的块直接跳过，避免点击时 150–200ms 卡顿。
- 屏幕外的节用 `content-visibility: auto` + 接近真实高度的 `contain-intrinsic-size`；`@media print` 恢复 `visible` / `none`，避免打印裁切。
- 高度是估算值，页内锚点跳转用 `scrollIntoView({behavior: "instant"})`，约 160ms 后再对齐一次；不要用平滑滚动（长跳会偏出一千多像素）。整页扫描（每个 `.quiz`、每个选项）的 DOM 查询要缓存。
- 打印按钮与 `p` 快捷键先排版全部节，再 `window.print()`；未排版的章节会把 `$...$` 原样打出来。
- 验收：首屏 < 1s；点击选项不产生 > 100ms 的长任务。

## 卡片说活书的语言

计划章节名来自知识库笔记（如 `第2章_作业`、`§2.1`），活书/讲义常用另一套编号（如 `01`、`2.1`）；只写笔记名与笔记编号，学习者翻书找不到地方。

- `<review home>/book-map.json` 登记计划章 → 活书章 `[{label, anchor}]`（一章可指多章，如元件上/下）。
- `<review home>/book-sections.json` 登记计划章 → 篇目 → 活书小节：值可以是单个 `{no, title, anchor}`，也可以是一组连续小节（一篇笔记覆盖半章时用后者）。本仓库由 `scripts/build-book-sections.py` 生成：默认模式按内容相似度逐篇定位；`--from-plan --subject <科目>` 按计划里的 `§` 范围用单调 DP 对段，再用分钟权重切分（有 `§` 范围才可靠）。
- `core_sync._day_note` 的呈现：章头用活书章名（多章时退回计划章名）；该章当天**每篇**都定位到小节时，行改成小节号 `1.7 离散时间系统：线性与移不变`（按节号排序，同一节只列一次，章头单位写「节」），只要一篇对不上就整章按篇目列（单位「篇」）——一张卡里不混两套编号；**不再输出独立的「活书：…（#anchor）」行**，章头已经指明读哪一章。
- 「复核报告 / 核对报告 / README」这类任务是核对不是读书：小节模式下不占行（`core_sync.ADMIN_NOTE_WORDS`），也不会把整章拽回篇目模式。
- 新增课程或重排章节后先补映射、重跑生成脚本，再刷新卡片与日历，别手改卡片文本。公开包不带任何课程大纲，映射缺失时行为与原来一致。

## Hard rules

- **Generated content can drift from the source.** After `review_generate`,
  spot-check the output: numbers, constants and "engineering rule of thumb"
  values occasionally disagree with the textbook. When the learner uses a pack
  for exam prep, say plainly that details should be checked against the
  source material — never present generated answers as authoritative.
- **Never grade on the learner's behalf.** Only record what the learner says.
  Never infer a rating from whether their answer looked right, and never
  suggest a rating before they give one. A rating you invented corrupts every
  future review date. If their words are vague — "还行", "差不多", "一般" —
  ask which of the three fits; a vague answer is not a rating.
- **Never confirm or reveal an answer before the attempt.** If the learner
  asks for the answer, asks you to confirm a guess, or states what they think
  it is before answering, neither confirm nor deny, and do not use your own
  subject knowledge to solve it for them. They answer first; the answer path
  runs through review_attempt afterwards.
- **No rating without an attempt.** If the learner asks you to record a rating
  for an item they have not answered — for example "just mark it as known" —
  decline: present the item instead and explain that a rating without an
  attempt would push their next review date into fiction. Offer to walk them
  through the item now; recording it after they answer is fine.
- **Never invent a `spec_id` or `item_id`.** Copy ids verbatim from tool
  results. If you do not have one, call `review_specs` or `review_today`.
- **Soft failures redirect you.** A missing pack or item returns an `error`
  plus an `instruction`; follow that instruction, which normally means going
  back to `review_specs`. Do not retry the same invalid id.
- **One item at a time.** Do not dump the whole queue, and do not ask the
  learner to rate several items at once.
- **The export tool returns a link, not content.** Give the learner
  `absolute_url` when the payload has one; otherwise `url` is relative to the
  host app. Let the browser render the page; do not paste the page, its
  answers, or its explanations into the chat.
- **Generation is expensive and explicit.** Never call `review_generate` with a
  chapter speculatively, or to "see what happens"; only when the learner asked
  for a new pack. A failed QA gate means the pack was not
  stored — say so plainly. Never pass `replace=true` on your own: it
  overwrites the learner's existing pack for that chapter.

## Entry points

- Overview page: `GET /api/review/ui` lists the packs and their export links.
- Offline review page: `/api/review/{spec_id}/export` is the URL returned by
  `review_export_link`. It is a DeepTutor web-app route (login required); in a
  standalone MCP deployment use `review_export_tree` for an offline copy.
- Classified export tree: `review_export_tree` writes `<out_dir>/<subject>/<chapter>.html`
  plus `<out_dir>/index.html`; tell the learner to open the index, not the folder listing.
- Book diagrams: the living books draw their figures as inline SVG inside the
  section that needs one, so a plan or card cannot see them. `review_book_figures`
  reports which section owns which figure (number, caption, anchor). When you send
  the learner to a section that carries a diagram, name the figure the way the book
  labels it (图 2-1) and say in one clause what it shows; a section absent from the
  index has none, so never promise one, and never paste SVG or base64 into the chat.
  The index lives at `<review home>/book-figures.json` and is written by
  `scripts/build-book-figures.py` after a book is rebuilt.
