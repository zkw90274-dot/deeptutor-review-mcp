# Review Studio · 个性化 MCP 商店条目

这个目录把 Review Studio 注册进 DeepTutor 个性化页的 **MCP 服务商店**。它只适用于
**每个使用者各自部署自己的 Review MCP** 的场景：每人的 URL、token 与复习数据都是
自己的。

## 为什么需要这个补丁

个性化商店的用户级条目是**远程服务模板**：

- 只接受 `streamableHttp` / `sse`，stdio 永久只允许管理员部署级；
- 用户级严格禁止 loopback 与私网地址；
- 商店目录写死在核心 `deeptutor` 包的 `curated.json`，安装 payload 不允许覆盖 URL。

核心原本没有“安装时填写 URL”的字段类型。这个补丁给核心 catalog 增加一个最小的
`url` credential target，然后合并 `deeptutor-review` 条目。安装时用户填写自己的
公网 HTTPS 端点与 Bearer token。

## 前提

1. Review MCP 以 Streamable HTTP 暴露在**公网 HTTPS** 上（反代/隧道均可），并设置
   `DEEPTUTOR_REVIEW_MCP_TOKEN`；
2. 服务端设置 `DEEPTUTOR_REVIEW_PUBLIC_BASE_URL`，否则 `review_export_link` 只返回相对
   路径；
3. 每个使用者自己的服务实例使用自己的 `DEEPTUTOR_REVIEW_HOME`。**不要把同一个实例
   同时给多个学习者使用**：当前服务是单租户的。

## 应用

```bash
.venv/Scripts/python.exe -m deeptutor.review.integrations.mcp_catalog.apply
# 或安装 wheel 后：
deeptutor-review-catalog
```

`--check` 只验证，`--dry-run` 只显示改动。脚本会：

1. 给核心 `catalog/models.py` 增加 `url` target（幂等、先备份）；
2. 把 `deeptutor-review.json` 合并进核心 `curated.json`（按 id 替换）；
3. 在新进程里验证 `get_entry("deeptutor-review")` 与 `build_server_config(...)`。

**核心升级会覆盖这些文件，升级后需要重新运行一次。**

## 安装后的使用

个性化 → MCP 服务 → 商店，搜索 “Review Studio”，安装时填写：

- **Review MCP 地址**：`https://<你的域名>/mcp`
- **访问令牌**：服务端 `DEEPTUTOR_REVIEW_MCP_TOKEN` 的值

安装后 MCP 客户端会以 `Authorization: Bearer <token>` 连接；地址必须是公网 HTTPS，
loopback/私网会被核心的 SSRF 防护拒绝。同机自用请走管理员部署级 stdio，而不是这个
用户级商店条目。
