---
name: literature-research
description: 文献调研能力域①。工作流：多源检索→去重→OA 下载→MinerU 云提取→结构化笔记 rubric→citecheck 引用核验门→综述脚手架。当需要调研主题、检索/获取/解析论文、写结构化文献笔记、核验引用一致性、维护 Zotero 文献库或起草综述/调研报告时使用。
---

# 文献调研 (Literature Research)

能力域①的技能面：**技能文档 + 社区工具 + 薄胶水**。检索/下载/解析/入库全走社区工具，自研只有薄胶水层
（`sub_root/src/pysci/skills/literature_research/`，约 400 行）；方法论（笔记 rubric、citecheck 门、
取证规范、综述脚手架）由本技能与 `references/` 承载。编排形态借鉴 gpt-researcher / STORM
（任务拆分→证据聚合→逐节写作），**不整机引入**。

## 能力边界

**做**：多源检索聚合与去重、引文网络两跳滚雪球、OA 全文获取（arXiv 直下 + OA 回退链）、PDF→MD
（公式→LaTeX、表格→HTML）、arXiv e-print LaTeX 源清洗、结构化笔记与评估 rubric、引用核验门
（citecheck）、Zotero 入库与语义检索问答、综述/调研报告脚手架、非学术网页取证。

**不做**（用户裁决剔除，勿重建）：SciMago 自建分区索引（agent 查分区表即可）、
WoS 自研客户端（需 WoS/Scopus 数据时走 paper-search-mcp 显式连接器并自备 key）、缓存/doctor
自检（目录约定替代）、付费墙浏览器抓取（降级 agent web 工具按需处理）。

**可选件**：paper-qa（科研 RAG 深度问答，带引文+撤稿检查）——不进主链，仅当 zotero-mcp 语义检索
不敷时按需启用（见 [tooling](references/tooling.md)）。

## 工具面（谁干什么）

| 角色 | 入口 | 说明 |
|---|---|---|
| 检索/下载/引文 | **paper-search-mcp**（MCP） | 22 源聚合+去重+OA 回退+引文关系；CLI `paper-search` 兜底 |
| PDF→MD | **MinerU 云 API** → 胶水 `mineru_cloud.extract_markdown` | `.env` 的 `MINERU_TOKEN`；本机不跑本地提取 |
| PDF 兜底 | pymupdf4llm | MinerU 不可达/配额尽时轻量抽取（公式丢失；AGPL，已接受） |
| arXiv LaTeX 清洗 | 胶水 `tex_clean.clean_arxiv_tex` | e-print 源→去注释/preamble/thebibliography→轻量 MD |
| 文献库+RAG | **Zotero + zotero-mcp**（MCP） | 入库、BibTeX、ChromaDB 语义检索、Scite 警报 |
| 笔记归一化 | 胶水 `frontmatter.normalize_note` / `build_index` | frontmatter 契约 + INDEX.md |
| 引用核验 | 胶水 `citecheck.reconcile_note` + MCP DOI 工具 | 文内引用×文献列表对账，见"引用核验门" |
| 综述/报告 | 本技能 + agent 写作 | 借鉴 STORM：多视角→大纲→逐节 |

## 工作流（9 步 + 综述）

1. **search**（多源检索）：paper-search-mcp 按主题多路检索（关键词/年份/引文过滤；OpenAlex 需免费
   key 提额，`OPENALEX_API_KEY` 已在 `.env`）。
2. **screen**（筛选）：title/abstract 快筛，入选快照落 `shortlists/<date>-<topic>.md`。
3. **add**（去重入库）：按 DOI → arXiv id → 标题归一去重；元数据落 Zotero（zotero-mcp）。
4. **citegraph**（引文滚雪球）：paper-search-mcp 的 OpenAlex references/citers，两跳内回补 screen。
5. **read**（获取+提取）：OA 直下（arXiv / `download_with_fallback`）→ `mineru_cloud.extract_markdown`
   → 提取稿入 `papers/<key>.md`；MinerU 失败降级 pymupdf4llm 并记录降级。有 arXiv e-print LaTeX
   源时用 `tex_clean.clean_arxiv_tex` 替代 PDF 提取。
6. **note**（结构化笔记）：按 [note-rubric](references/note-rubric.md) 写 `papers/<key>.note.md`，
   frontmatter 过 `frontmatter.normalize_note`。
7. **index**：`frontmatter.build_index` 生成 `INDEX.md`（勿手改）。
8. **review**（综述/调研报告）：见"综述脚手架"。
9. **citecheck**（引用核验门）：见下节；任何对外产物写入前必过。

## 引用核验门（citecheck，硬门）

对外产物（综述、论文段落、笔记定稿）在**写入前**必须通过两级核验：

1. **机械对账**：`citecheck.reconcile_note(body, bib_text)` —— 文内引用键（`\cite{}` / `[@key]` / `@key`）
   × 文献列表（`refs.bib` / 笔记条目）对账，分级 `PASS ✓ / WARN △ / FAIL ✗ / NOT_FOUND ?`。
   **FAIL**（文内引用缺文献条目）→ 拦截写入；**WARN**（孤儿条目）→ 提示清理后放行。
2. **元数据核验**：对每条 DOI/arXiv id 调 paper-search-mcp 的 DOI 核验工具（如
   `get_crossref_paper_by_doi`）与 OpenAlex 交叉比对 title/author/year。**硬冲突**（标题/年份/一作
   不符）→ 拦截并要求人工裁决；软差异（页码、venue 简写）标注后放行。

分级结果（✓/△/✗/?）随产物落盘留审计；引用路径必须可解析（抽样链接校验）。

## 取证规范

非学术网页证据统一双标签：**[直取]**（官方/一手页面直引）与 **[二手互证]**（两个独立二手源交叉）。
403/地域封锁决策树、来源分级与引用格式见 [evidence-norms](references/evidence-norms.md)。
学术主张一律优先学术来源，网页证据不得替代论文引用。

## 综述脚手架

借鉴 STORM：多视角提问（背景/方法/对比/开放问题）→ 大纲 → 逐节写作（每句事实主张挂引用键）→
citecheck → 结论/开放问题。骨架与逐节纪律见 [review-scaffold](references/review-scaffold.md)。

## 数据布局

`data/skills/literature_research/`（由 `pysci.paths.LITERATURE_ROOT` 解析，勿手写相对路径）：

- `papers/`：`<key>.md`（提取稿）、`<key>.note.md`（笔记）、原始 PDF；
- `shortlists/`：筛选快照；`reviews/`：综述产物；`templates/`：笔记/综述模板；
- `INDEX.md`：笔记索引（`frontmatter.build_index` 生成）。

key 约定：`<第一作者姓><年份><标题首词>`，无 DOI 的来源加 `-arxiv` 等后缀。

## references（细则，按需加载）

- [tooling.md](references/tooling.md) — MCP/胶水调用细则：paper-search-mcp、zotero-mcp、MinerU 云、
  pymupdf4llm 兜底、paper-qa 可选启用。
- [note-rubric.md](references/note-rubric.md) — 笔记 rubric、frontmatter 契约（字段表+status 机）、
  note merge 合同、INDEX.md。
- [review-scaffold.md](references/review-scaffold.md) — 综述/调研报告脚手架与逐节纪律。
- [evidence-norms.md](references/evidence-norms.md) — 非学术网页取证规范。
- [bilingual.md](references/bilingual.md) — 中英对照生产线（2026-11-17 用户裁决吸收，取代旧「不做」条目）：
  五命令工作流（prepare/status/submit/check-work/render + reorder-suggest/quality）、三契约、质量闸门、与 MinerU 提取衔接。
- [troubleshooting.md](references/troubleshooting.md) — 故障手册（封禁/配额/404/API 未开）。

## 选型与豁免（ADR 摘要）

paper-search-mcp 差活跃度口径（2026-11 无提交）以"唯一 22 源聚合通道"**豁免引入并 pin**；paper-qa
**豁免 pin 为可选件**（不进主链）；PDF→MD 走 MinerU **云 API**（本机硬件不足，用户裁决）；
pymupdf4llm 的 AGPL 已接受（个人不分发）。完整调研、逐月提交证据与豁免记录：
`sub_root/others/community_research/2026-11-17/domain-01-literature-selection.md`。
