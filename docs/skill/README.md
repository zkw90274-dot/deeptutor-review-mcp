# Review Studio 技能包

`review-studio` 提供学习 MCP 的工具调用、复习计划和导出规则。采用与 DeepTutor 内置技能一致的 YAML frontmatter 和 Markdown 格式。[Review Studio Skill](SKILL.md)

`interactive-study-book-svg` 提供课件逐页核对、矢量重绘、构建和浏览器验收流程，适合维护活书或讲义的 agent。使用支持 Agent Skills 的客户端时，复制 `skills/interactive-study-book-svg/` 整个目录到客户端的技能目录。也可以在 DeepTutor 技能页单独导入该目录中的 `SKILL.md`。[活书 SVG Skill](https://github.com/zkw90274-dot/deeptutor-review-mcp/blob/main/skills/interactive-study-book-svg/SKILL.md)

## 如何导入

该技能包由核心 skills 服务管理，接口与字段以核心实现为准
（`deeptutor/api/routers/skills.py`）。

- **本地 SKILL.md**：在前端技能页导入上面的 `SKILL.md`（前端解析 frontmatter 后落盘），或调用
  `POST /api/skills/create`，JSON 为
  `{"name": "review-studio", "description": "...", "content": "<正文 Markdown>", "tags": ["education", "review", "spaced-repetition"]}`。
  `name` 必须匹配 `^[a-z0-9][a-z0-9-]{0,63}$`，每个 tag 必须匹配
  `^[a-z0-9][a-z0-9\- _]{0,31}$`。
- **Hub 包**：`POST /api/skills/install` 只接受
  `{"ref": "<hub>:<slug>[@version]"}`（可选 `name`、`force`、`allow_unverified`），
  用于从 EduHub 等 hub 安装已发布的包，**不能**用它直接导入本地 `SKILL.md`。

`review-studio` 的工具调用需要主项目启用 `review` 模块。`interactive-study-book-svg` 可用于独立活书项目，所需的构建和浏览器工具由项目决定。
