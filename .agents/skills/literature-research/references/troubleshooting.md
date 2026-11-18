# 故障手册

> 先查环境（`sub_root\deps_management\deps_manager\update-deps.cmd status`）再查网络再查 API。

| 症状 | 原因/诊断 | 处置 |
|---|---|---|
| Google Scholar 检索 403/CAPTCHA | 反爬封禁（已知风险，ADR 已记） | 退避后改走 arXiv/OpenAlex/Crossref 组合；不要硬重试 |
| OpenAlex 403/限速 | 匿名配额尽 | 确认 `OPENALEX_API_KEY` 已配；每日 100k credits 免费 |
| MinerU 提取失败/超时 | 网络、配额、PDF 扫描件过重 | 重试一次；仍失败降级 pymupdf4llm 并在 frontmatter `source` 标注 |
| `MINERU_TOKEN 缺失` | `.env` 未迁移/被清 | 检查 `sub_root/.env` 的 `MINERU_TOKEN`（密钥不入库、不外传） |
| OA 下载 404 / 非 %PDF | 预印本撤稿、URL 失效 | 走 `download_with_fallback` 回退链；全链失败回 screen 步换文献 |
| zotero-mcp 连接失败 | 桌面端未开 / 本地 API 未启用 | 启动 Zotero，设置→高级→勾选"允许其他应用访问"；核对 `ZOTERO_LOCAL_API_BASE` |
| citecheck FAIL（missing） | 文内引用缺文献条目 | 补 refs.bib/笔记条目，或删引用；**不许**改 grade 蒙混 |
| citecheck WARN（orphan） | 文献条目未被引用 | 清理条目或补引用后放行 |
| frontmatter `ValueError` | 缺必填/年份非数字/status 越界 | 按 note-rubric 字段表修复后重跑 `normalize_note` |
| INDEX 出现 `INVALID` 行 | 该笔记归一化失败 | 修该笔记 frontmatter，重建 INDEX |
