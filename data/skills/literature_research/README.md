# `data/skills/literature_research/` — AI 文献调研数据区

本目录是 pySciWS 的**结构化文献知识库（数据区）**，由 AI Agent（Qoder）主导维护、用户审校。
它是 `literature-research` skill 的**数据后端**；工具链**代码**已随项目重构收入 `pysci` 包。三者分工如下：

| 位置 | 角色 | 内容 |
|---|---|---|
| `data/skills/literature_research/`（本目录） | **数据** | `papers/` 笔记、`shortlists/` 检索快照、`reviews/` 综述、`templates/` 模板、`INDEX.md` 索引、`cache/` 缓存 |
| `src/pysci/skills/literature_research/` | **代码** | `tools/` 工具链（`research` CLI + 8 个 client 模块），随 `pysci` 包 editable 安装 |
| `.qoder/skills/literature-research/` | **Skill（说明书）** | `SKILL.md` + `references/`，教 AI 何时、如何调用 `research` CLI |

> **命名对应**：代码包 `literature_research`（下划线，合法 Python 包名）↔ skill `literature-research`（连字符，skill 命名规范）。一一对应，见名知意。
> **代码/数据分离**：代码入 `src/pysci/skills/`（纳入包与版本控制），数据留 `data/skills/literature_research/`（git-ignore 的 `cache/`、PDF 不混入包树）。数据区路径由 `pysci.paths.LITERATURE_ROOT` 统一锚定，不依赖脆弱的相对层级数学。

本目录与项目根目录下的 `参考资料/`（用户手动整理）**互不干涉**：

| 目录 | 管理者 | 用途 | 是否入 Git |
|---|---|---|---|
| `参考资料/` | 用户 | 已有的手动笔记、书籍式资料 | 否（已在根 .gitignore） |
| `data/skills/literature_research/` | **AI Agent** + 用户审校 | 单篇论文笔记、检索快照、综述草稿 | **是**（除 `cache/`、PDF） |

---

## 1. 目录结构

```
data/skills/literature_research/     # 本目录（数据区）
├── README.md                # 本文件
├── INDEX.md                 # 全库索引（由 `research index` 自动维护，勿手改）
├── .gitignore               # 忽略 cache/、PDF、.env
├── papers/                  # 每篇论文一个 markdown（结构化笔记）
├── reviews/                 # 综述性长文（多篇论文的组合分析）
├── shortlists/              # 检索结果快照（一次检索 = 一个文件）
├── templates/               # 笔记模板
│   ├── paper_note.md
│   ├── review_note.md
│   └── shortlist.md
└── cache/                   # 临时缓存（.gitignore，不入 Git）
    ├── api_responses/       # Tier B：原始 API 响应（TTL 过期自动清理）
    ├── pdfs/                # Tier A：下载的 PDF
    ├── extracted/           # Tier A：抽取的全文 markdown（{slug}_fulltext.md）
    ├── html_fulltext/       # Tier A：浏览器抓取的 HTML 全文（+ .by_url/ URL→bundle 命中清单）
    └── .autoclean_state.json  # 上次自动清理时间戳

src/pysci/skills/literature_research/   # 代码区（随 pysci 包 editable 安装）
├── __init__.py              # 使本目录成为 Python 包（供 -m 调用）
└── tools/                   # 本地 Python 工具链
    ├── research.py          # ★ 统一 CLI 门面：doctor/search/read/get/add/library/index/cache
    ├── config.py            # 统一配置加载（读项目根 .env，暴露 settings；路径经 pysci.paths 锚定）
    ├── openalex_client.py   # OpenAlex 检索（主源）+ 元数据规范化
    ├── arxiv_client.py      # arXiv 检索 + PDF 下载
    ├── wos_client.py        # Web of Science：官方 JIF/JCR/ESI 增强
    ├── semantic_scholar_client.py  # Semantic Scholar：TLDR 增强（可选，常不可达）
    ├── zotero_bridge.py     # Zotero 桥接（本地 API 优先，回退 Web API）
    ├── browser_fetch.py     # Playwright 抓取付费墙全文 / PDF / 补充材料（URL 命中即复用）
    ├── pdf_extract.py       # PDF → Markdown（MinerU 云端主力 / pymupdf4llm 兜底）
    └── cache_manager.py     # 两层缓存治理：stats/clean/prune + LRU（bump_mtime）
```

**唯一入口是 `research` CLI**（`tools/research.py`）——它编排上面所有 client，日常无需直接调用各 `*_client.py`。

---

## 2. 命名规范

### 论文笔记（`papers/`）

格式：`{year}_{firstauthor_lastname}_{slug}.md`

- `year`：论文发表年（若为 arXiv 预印本，用首次挂出的年份）
- `firstauthor_lastname`：第一作者姓氏（英文小写，无声调符号）
- `slug`：3–5 个英文单词，用 `-` 连接，概括论文主题

示例：
- `2026_zhang_ep-acoustic-active-gain.md`
- `2025_jones_nonhermitian-bic-topological.md`
- `2018_zhu_simultaneous-observation-of-topological.md`

### 综述（`reviews/`）

格式：`{YYYY-MM}_{topic-slug}_survey.md`，示例：`2026-09_active-EP-acoustic_survey.md`

### 检索快照（`shortlists/`）

格式：`{YYYY-MM-DD}_{query-slug}.md`，示例：`2026-09-21_nonhermitian-acoustic-2024-2026.md`

---

## 3. PDF 与全文的存储策略

**PDF 不入 Git**（由 Zotero 云同步管理），Markdown 笔记入 Git。

- Zotero 数据目录：`D:\XiGPrograms\zotero\data`（用户本地）
- 每篇论文笔记的 frontmatter 通过 `zotero_key` / `zotero_uri` 关联 Zotero 条目
- 需要离线全文时，`research read` 会临时下载 PDF 到 `cache/pdfs/`，抽取为 markdown 存到
  `cache/extracted/<slug>_fulltext.md`（唯一规范副本，路径写入笔记的 `extracted_md_path`）；两者都不入 Git

**缓存分两层**（`cache_manager.py` 治理，`research cache` 为门面）：

| 层 | 目录 | 重获代价 | 生命周期 | 淘汰 |
|---|---|---|---|---|
| **Tier A 持久层** | `pdfs/`、`extracted/`、`html_fulltext/` | 高（MinerU 配额 / 浏览器爬取 / 下载） | 默认**永久保留**，命中即复用 | **仅手动** `research cache prune`（按 LRU） |
| **Tier B 易失层** | `api_responses/` | 低（秒级可重取的 JSON） | TTL 过期即死文件 | **自动**钩子定期清 + 手动 `research cache clean` |

- **命中即复用**：同一篇文章再次 `research read` 时直接读 `cache/extracted/<slug>_fulltext.md`，**跳过抓取与抽取**（打印 `[read] 命中缓存全文`）；`--force` 强制重取。浏览器抓取层也按 URL 复用已缓存的 HTML/PDF bundle。
- **磁盘治理**：`research cache stats`（分层概览 + 软上限告警）、`cache clean`（清 Tier B，`--all`/`--older-than N`）、`cache prune --max-mb N`（按 LRU 淘汰 Tier A，`--keep-referenced` 保护被笔记引用者）；三者均支持 `--dry-run` 先列后删。
- **自动钩子**：`search`/`read`/`get`/`add` 入口按 `CACHE_AUTOCLEAN_INTERVAL_DAYS` 间隔自动清理过期 Tier B，全程非致命、无常驻进程（靠 `cache/.autoclean_state.json` 记时）。

**为什么这样设计**：Git 擅长版本化文本、不擅长版本化二进制大文件。笔记的每次修订都是有价值的历史，
而 PDF 只是笔记的"上游数据源"，Zotero 已经管理得很好。

---

## 4. 与 Zotero 的关系

- Zotero 是 PDF 与元数据的**权威源**（source of truth）
- 本目录的 markdown 是**在 Zotero 之上的增值层**：AI 生成的摘要、评价、跨文献链接、研究关联笔记
- 双向：
  - Zotero → 本目录：`research library` 查询用户库；`research add` 拉取元数据建笔记
  - 本目录 → Zotero：`research add` 通过 API 在 Zotero 建条目，并把返回的 `zotero_key` 回写笔记

---

## 5. AI 工作流（围绕 `research` CLI）

> 完整触发条件、flag、输出解读与**评价 rubric** 见 skill：
> `.qoder/skills/literature-research/`（`SKILL.md` + `references/search.md`、`read.md`、`maintenance.md`）。此处只给主干。

调用方式（PowerShell；多词 query 用**单引号**，命令分隔用 `;` 而非 `&&`）：

```
.venv\Scripts\python.exe -m pysci.skills.literature_research.tools.research <子命令> …
```

标准流程：

1. **体检** `research doctor` — 确认路径、各数据源、PDF 后端、Playwright、Zotero 是否就绪
2. **检索** `research search '<query>' [--year 2020-2026] [--sort citations] [--save --purpose '<goal>']`
   — 默认融合 OpenAlex + arXiv 并去重；`--save` 生成 `shortlists/` 快照
3. **看单篇元数据** `research get <doi｜arxiv-id｜openalex-id>` — 打印完整 frontmatter（含官方 JIF/JCR）
4. **入库** `research add <id> [--tags …]` — 在 Zotero 建条目 + 写 `papers/` 笔记骨架（不抓全文）
5. **精读** `research read <doi｜url｜arxiv-id>` — 抓全文（MinerU 云端抽取）+ 建笔记骨架；随后 AI
   读 `cache/extracted/*_fulltext.md`，在 `papers/*.md` 填 TLDR / Key Claims / Novelty / Rigor /
   Journal-tier / Relevance 各章节
6. **更新索引** `research index` — 扫描 `papers/` 重建 `INDEX.md`（**勿手动编辑 INDEX.md**）
7. **（可选）综述** — 多篇读完后可生成 `reviews/{YYYY-MM}_{topic}_survey.md`

**增强策略**：`get`/`read`/`add` 单篇默认做 WoS/S2 增强；`search` 的逐条增强需显式 `--enrich`
（默认关，避免 N 次慢调用）。Semantic Scholar 为**可选末位源**——无 key 静默跳过、失败视作预期、不告警。

**缓存维护**：昂贵产物（PDF / 抽取全文 / 抓取 HTML）默认永久保留、命中即复用；用
`research cache stats｜clean｜prune` 治理磁盘（分层模型见 §3）。

---

## 6. 用户可以做什么

- **审校**：随时修改 `papers/*.md` 中 AI 写的评价段落，AI 会尊重你的修改（`research read` 默认不覆盖
  已存在笔记，需 `--overwrite` 才重建骨架）
- **提问**：直接问 "帮我看看 `papers/2018_zhu_….md` 里的方法部分"
- **触发流程**：例如 "帮我搜索过去两年关于非厄米 EP 的声学有源实现"，AI 会启动 `research` 全流程
- **不要做**：手动重命名 `papers/` 下的文件（会破坏索引链接）；手动编辑 `INDEX.md`（由 `research index`
  生成）；把 PDF 直接放到本目录（应放 Zotero）

---

## 7. 依赖

工具链依赖以下 Python 包，均在项目 uv 虚拟环境（`.venv`）中、已声明于根 `pyproject.toml`：

**核心**：
- `requests` — HTTP 客户端（OpenAlex / arXiv / WoS / Zotero / MinerU 云端）
- `python-dotenv` — 加载项目根 `.env`
- `pyzotero` — Zotero API 官方 Python 封装
- `markdown` — markdown → HTML（写入 Zotero 笔记时需要）

**PDF 抽取与全文抓取**：
- **MinerU 云端 Open API** — 公式精读**主力**（VLM + OCR，公式→LaTeX、表格→HTML）；仅需 `requests`
  + `MINERU_TOKEN`，无本地重型依赖
- `pymupdf4llm` — 本地**兜底**后端：纯 CPU、快，但**公式会丢失**
- `playwright>=1.63.0` — 付费墙论文的 HTML 全文 / 正文 PDF / 补充材料抓取

> 旧的 `marker-pdf` / `magic-pdf`（本地 MinerU）实测本机不可用，且云端 MinerU 可完全替代，已从依赖中**移除**。

**安装 / 更新**：
```
uv sync
playwright install chromium   # 仅当系统无 Chrome/Edge 时，browser_fetch 需要捆绑的 chromium
```

**运行**：
```
.venv\Scripts\python.exe -m pysci.skills.literature_research.tools.research doctor
# 或：uv run python -m pysci.skills.literature_research.tools.research doctor
```

---

## 8. 凭据管理

所有 API key、token、Zotero userID 等敏感信息都放在**项目根目录**的 `.env`
（已被根 `.gitignore` 与本目录 `.gitignore` 忽略）。常用键：

`OPENALEX_EMAIL`、`WOS_API_KEY`、`SEMANTIC_SCHOLAR_API_KEY`（可选）、`ZOTERO_USER_ID`、
`ZOTERO_API_KEY`、`MINERU_TOKEN`、`PDF_EXTRACT_BACKEND`、`CACHE_DIR`、`HTTP_TIMEOUT_SECONDS`、
`HTTP_MAX_RETRIES`。缓存治理（均可选）：`CACHE_B_MAX_AGE_DAYS`（默认 7）、
`CACHE_AUTOCLEAN_INTERVAL_DAYS`（默认 7）、`CACHE_SOFT_LIMIT_MB`（默认 2048）。

除 OpenAlex / arXiv 外全部可选；缺失时对应功能静默降级。用 `research doctor` 一览当前凭据与可达性。

---

_Last updated: 2026-09-21 · Maintained by: AI Agent (Qoder) · 配对 skill：`.qoder/skills/literature-research/`_
