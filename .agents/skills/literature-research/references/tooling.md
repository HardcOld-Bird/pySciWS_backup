# 工具面细则：MCP 与薄胶水调用

> SKILL.md「工具面」的展开。配置类密钥一律在 `sub_root/.env`（不入库）。

## paper-search-mcp（检索/下载/引文，MCP 主路）

- 由并行实施者以 DSH profile bundle 接入并 pin 版本；CLI `paper-search` 作命令行兜底。
- 工具面（以其实装的 MCP 工具名为准）：多源检索（22 源聚合+去重）、全文下载
  （`download_with_fallback`：arXiv 直下 → OA 回退链 OpenAIRE/CORE/PMC/Unpaywall）、
  DOI 核验（如 `get_crossref_paper_by_doi`）、OpenAlex references/citers 引文关系。
- 风险注记（ADR）：Google Scholar 源易封——连续 403/CAPTCHA 时退避并改走 arXiv/OpenAlex/
  Crossref 组合，不要硬重试。OpenAlex 用 `OPENALEX_API_KEY` 提额（免费 100k credits/日）。
- WoS/Scopus 显式连接器需要 key（`WOS_API_*`）；不配则引文数以 OpenAlex 为准。

## MinerU 云 API（PDF→MD 主路）

- 胶水入口：`pysci.skills.literature_research.mineru_cloud.extract_markdown(pdf_path) -> str`。
- 认证：`MINERU_TOKEN`（环境变量或 `sub_root/.env`）；API 根默认 `https://mineru.net`，
  可用 `MINERU_API_BASE` 覆盖。协议：提交批次→预签名 URL 上传→轮询→下载结果 zip 取 `full.md`。
- 本机**不跑** MinerU 本地提取（硬件不足，用户裁决）。网络/配额失败见 troubleshooting.md。
- 提取产物公式为 LaTeX、表格为 HTML——笔记/正文中引用公式时保留 LaTeX 原样。

## pymupdf4llm（兜底）

MinerU 不可达或配额尽时降级：`pymupdf4llm.to_markdown(path)`。公式丢失、表格弱，必须在笔记
frontmatter `source` 字段标注 `pymupdf4llm`（降级可追溯）。AGPL-3.0，个人科研使用已获用户接受。

## zotero-mcp（文献库 + RAG，MCP）

- 需 Zotero 桌面端 + 本地 API（`ZOTERO_LOCAL_API_BASE`，默认 `http://127.0.0.1:23119/api`，
  Zotero 设置里开启"允许其他应用访问"）或 Web API（`ZOTERO_USER_ID`/`ZOTERO_API_KEY`）。
- 能力：条目增删改查、BibTeX 导出（无需 Better BibTeX）、**ChromaDB 语义检索**（承接旧 RAG）、
  Scite 撤稿警报、linked-file/WebDAV 附件。
- 入库后把 `zotero_key`/URI 回写到笔记 frontmatter；`refs.bib` 由 zotero-mcp 导出、勿手改。

## paper-qa（可选件，不进主链）

仅当 zotero-mcp 语义检索不敷（需要带引文定位的深度问答）时按需启用：`pqa index <目录>` 建索引、
`pqa ask "<问题>"` 问答（引文+撤稿检查）。CalVer 无兼容承诺，必须 pin 版本；用毕可移除，主链不受影响。

## 薄胶水入口一览（`pysci.skills.literature_research`）

| 模块 | 入口 | 职责 |
|---|---|---|
| `tex_clean` | `clean_arxiv_tex(src)` | e-print LaTeX 源清洗（去注释/preamble/thebibliography→轻量 MD） |
| `citecheck` | `reconcile_note(body, bib_text)` | 文内引用×文献列表对账（PASS/WARN/FAIL/NOT_FOUND） |
| `frontmatter` | `normalize_note(text)` / `build_index(notes_dir)` | 笔记 frontmatter 归一化 / INDEX.md 生成 |
| `mineru_cloud` | `extract_markdown(pdf_path)` | MinerU 云 PDF→MD 协议薄封装 |

路径一律用 `pysci.paths.LITERATURE_ROOT` 等锚点，禁手写 `parents[N]`。
