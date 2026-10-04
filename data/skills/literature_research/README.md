# `data/skills/literature_research/` — AI 文献调研数据区

本目录是 pySciWS 的**结构化文献知识库（数据区）**，由 AI Agent（Qoder）主导维护、用户审校。
它是 `literature-research` skill 的**数据后端**；工具链**代码**已随项目重构收入 `pysci` 包。三者分工如下：

| 位置 | 角色 | 内容 |
|---|---|---|
| `data/skills/literature_research/`（本目录） | **数据** | `papers/` 笔记、`shortlists/` 检索快照、`reviews/` 综述、`templates/` 模板、`INDEX.md` 索引、`cache/` 缓存 |
| `src/pysci/skills/literature_research/` | **代码** | `tools/` 工具链（`research` CLI + 13 个后端模块），随 `pysci` 包 editable 安装 |
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
    ├── extracted/           # Tier A：抽取的全文 markdown（三种命名，见下）
    ├── html_fulltext/       # Tier A：浏览器抓取的网页正文 bundle（trafilatura 抽成 markdown，**不是原始 HTML**；
    │                        #   每个 bundle 一个 <slug>/ 子目录，另有 .by_url/ URL→bundle 命中清单）
    ├── rag/                 # 本地 PaperQA2 embedding 索引（index.pkl + index_meta.json；research rag，可重建）
    └── .autoclean_state.json  # 上次自动清理时间戳

data/skills/literature_research/data/   # 数据资产（**入 Git**，不在 cache/* 规则内）
├── scimago_index.json       # SCImago SJR 紧凑索引（按 ISSN → 分区/SJR/h-index）；research journal build-scimago 生成
└── SOURCE.md                # 上述索引的出处（下载 URL / 版本年 / 下载日期 / 归属声明 / 年度刷新步骤）

src/pysci/skills/literature_research/   # 代码区（随 pysci 包 editable 安装）
├── __init__.py              # 使本目录成为 Python 包（供 -m 调用）
└── tools/                   # 本地 Python 工具链
    ├── research.py          # ★ 统一 CLI 门面，14 个子命令：doctor/search/read/get/add/citecheck/citegraph/library/journal/review/rag/index/cache/ingest
    ├── config.py            # 统一配置加载（读项目根 .env，暴露 settings；路径经 pysci.paths 锚定）
    ├── notes.py             # 叶子模块：paper-note frontmatter 的读写/规范化/合并 + slugify/声调转写/journal_tier 推导（只依赖 stdlib + PyYAML + config）
    ├── journal_metrics.py   # 期刊质量指标的免费替代层：SCImago SJR 本地索引（按 ISSN 精确匹配分区）；research journal
    ├── openalex_client.py   # OpenAlex 检索（主源）+ 元数据规范化
    ├── arxiv_client.py      # arXiv 检索 + PDF 下载
    ├── wos_client.py        # Web of Science（Starter API）：Times Cited + wos_id + JCR 链接增强
    ├── citation_verify.py   # 引用完整性门：OpenAlex + Crossref + arXiv 三源交叉核验（citecheck / add 写入前 gate）
    ├── zotero_cli.py        # Zotero 桥接：委托社区 zotero-mcp 的 `zotero-cli --json`（未安装即优雅降级）
    ├── browser_fetch.py     # Playwright 抓取付费墙全文 / PDF / 补充材料（URL 命中即复用）
    ├── pdf_extract.py       # PDF → Markdown（arXiv 论文优先走 LaTeX 源码→公式忠实；否则 MinerU 云端主力 / pymupdf4llm 兜底）
    ├── local_ingest.py      # 批量归档本地 PDF 文件夹（复制 → 抽取 → manifest 台账）；research ingest
    ├── rag.py               # 本地语义检索（PaperQA2 + 硅基流动 bge-m3）：research rag index/search/ask（embedding 免费检索为主，ask 可选免费→付费→降级）
    └── cache_manager.py     # 两层缓存治理：stats/clean/prune + LRU（bump_mtime）
```

**唯一入口是 `research` CLI**（`tools/research.py`）——它编排上面所有 client，日常无需直接调用各 `*_client.py`。
（例外：每个后端模块另带一个 `python -m …` **调试后门**，仅用于把单层隔离出来排障；它们启动时会向
stderr 打一行 banner 并指向等价的 `research` 子命令，见 SKILL.md。）

`cache/extracted/` 的**三种**命名（由不同入口产生，共存不冲突）：

| 命名 | 谁写的 | 说明 |
|---|---|---|
| `{theme}/{slug}.md` | `research ingest` | 批量归档本地 PDF，按主题分子目录；路径记在 `ingest/manifest.json` |
| `{stem}_fulltext.md` | `research read` | 单篇精读的**规范副本**，路径写进笔记的 `extracted_md_path` |
| `{stem}_{sha1[:12]}.{backend}.md` | `pdf_extract` 自身 | 按 PDF 内容哈希 + 后端名的缓存（仅 `write_cache=True` 时）。`read`/`ingest` 都传 `False` 以免留双副本，故这一种主要来自 `python -m …pdf_extract` 调试后门 |

三种都是 `rag index` 的语料（它扫 `cache/extracted/**/*.md`）。

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
  `cache/extracted/{stem}_fulltext.md`（单篇精读的规范副本，路径写入笔记的 `extracted_md_path`）；两者都不入 Git
- **拿不到 PDF 时的网页兜底**：`read` 若三档抓取全失败（arXiv / OA 直链 / 浏览器），会退而用
  `browser_fetch` 已抽好的网页正文 bundle，把路径写进笔记的 **`extracted_html_path`**，而
  `extracted_md_path` **刻意留空**——`rag index` 只扫 `cache/extracted/**/*.md`，因此这份质量较低的
  兜底文本不会污染语义检索语料。它仍受 `cache prune --keep-referenced` 保护（该字段已列入
  `cache_manager.REFERENCED_FIELDS`），淘汰时以整个 bundle 目录为单元

**缓存分两层**（`cache_manager.py` 治理，`research cache` 为门面）：

| 层 | 目录 | 重获代价 | 生命周期 | 淘汰 |
|---|---|---|---|---|
| **Tier A 持久层** | `pdfs/`、`extracted/`、`html_fulltext/` | 高（MinerU 配额 / 浏览器爬取 / 下载） | 默认**永久保留**，命中即复用 | **仅手动** `research cache prune`（按 LRU） |
| **Tier B 易失层** | `api_responses/` | 低（秒级可重取的 JSON） | TTL 过期即死文件 | **自动**钩子定期清 + 手动 `research cache clean` |

- **命中即复用**：同一篇文章再次 `research read` 时直接读 `cache/extracted/{stem}_fulltext.md`，**跳过抓取与抽取**（打印 `[read] 命中缓存全文`）；`--refresh` 强制重取（旧名 `--force` 仍可用，已降级为隐藏弃用别名并打一行迁移提示）。浏览器抓取层也按 URL 复用已缓存的 HTML/PDF bundle。
- **磁盘治理**：`research cache stats`（分层概览 + 软上限告警）、`cache clean`（清 Tier B，`--all`/`--older-than N`）、`cache prune --max-mb N`（按 LRU 淘汰 Tier A，`--keep-referenced` 保护被笔记引用者）；三者均支持 `--dry-run` 先列后删。
- **自动钩子**：`search`/`read`/`get`/`add`/`citecheck`/`citegraph` 六个会产生网络响应的入口按 `CACHE_AUTOCLEAN_INTERVAL_DAYS` 间隔自动清理过期 Tier B，全程非致命、无常驻进程（靠 `cache/.autoclean_state.json` 记时）。`ingest`/`rag` 不挂钩子（前者已自带断点续传与配额节流，后者不写 `api_responses`）。

**为什么这样设计**：Git 擅长版本化文本、不擅长版本化二进制大文件。笔记的每次修订都是有价值的历史，
而 PDF 只是笔记的"上游数据源"，Zotero 已经管理得很好。

---

## 4. 与 Zotero 的关系

- Zotero 是 PDF 与元数据的**权威源**（source of truth）
- 本目录的 markdown 是**在 Zotero 之上的增值层**：AI 生成的摘要、评价、跨文献链接、研究关联笔记
- 双向：
  - Zotero → 本目录：`research library` 查询用户库；`research add` 拉取元数据建笔记
  - 本目录 → Zotero：`research add` 把 frontmatter 规范化为 BibTeX，委托社区 `zotero-cli`（zotero-mcp）建条目，并把返回的 `zotero_key` 回写笔记

---

## 5. AI 工作流（围绕 `research` CLI）

> 完整触发条件、flag、输出解读与**评价 rubric** 见 skill：
> `.qoder/skills/literature-research/`（`SKILL.md` + `references/search.md`、`read.md`、`maintenance.md`）。此处只给主干。

调用方式（PowerShell；多词 query 用**单引号**，命令分隔用 `;` 而非 `&&`）：

```
.venv\Scripts\python.exe -m pysci.skills.literature_research.tools.research <子命令> …
```

标准流程（与 `SKILL.md` 的 *Standard workflow* 九步一致，`doctor` 为前置体检）：

0. **体检** `research doctor` — 确认路径、各数据源、PDF 后端、Playwright、Zotero、期刊指标层是否就绪
1. **检索** `research search '<query>' [--year 2020-2026] [--sort citations] [--save --purpose '<goal>']`
   — 默认融合 OpenAlex + arXiv 并去重；`--save` 生成 `shortlists/` 快照；`--json` 则输出机器可读的
   结果数组（供程序化筛选，不必去解析人类可读的对齐文本）
2. **筛选** — 读快照，挑出要留的
3. **入库** `research add <id> [--tags …]` — **写入前默认三源引用核验**（OpenAlex+Crossref+arXiv，FAIL
   阻止入库；`--allow-fail` 越过 / `--no-verify` 跳过）→ 在 Zotero 建条目 + 写 `papers/` 笔记骨架（不抓全文）
4. **（可选）滚雪球** `research citegraph <seed-id> --save` — 取种子论文的参考文献（backward）与引用
   它的论文（forward），`--save` 写成快照后回到第 2 步——一篇好种子能扩展候选集，比再猜一条检索式可靠
5. **精读** `research read <doi｜url｜arxiv-id>` — 抓全文 + 建/合并笔记骨架；有 arXiv 预印本时优先走
   **LaTeX 源码**路径（公式是作者亲手写的 `$$…$$` 原文，而非 PDF 解析碎片），否则走 MinerU
   云端抽取；随后 AI 读 `cache/extracted/*_fulltext.md`，在 `papers/*.md` 填 TLDR / Key Claims /
   Novelty / Rigor / Journal-tier / Relevance 各章节
6. **更新索引** `research index` — 扫描 `papers/` 重建 `INDEX.md`（**勿手动编辑 INDEX.md**）；
   `index --check` 只校验（不一致退出码 1，可作 CI 门），`index --fix [--dry-run]` 则把既有笔记的
   frontmatter 规范化（补齐模板字段、统一字段序与 block style），**正文逐字节保留、既有值一律不动**
7. **（可选）综述骨架** `research review new '<topic>' --from-shortlist shortlists/<f>.md`
   → 生成 `reviews/{YYYY-MM}_{topic}_survey.md`，填好 frontmatter 并从快照聚合 `sources_used` /
   `query_strings` 后**就此停住**——§1-§5 的综述正文是 AI/人的判断，`review` 刻意不生成任何一行
8. **自己写综述**，然后 `research citecheck --review` 审它引用的每条文献；`research review status`
   查 `[[wiki-link]]` 断链与计数漂移（有断链退出码 1），`research review sync` 重算计数并只重写
   frontmatter（正文不动）

单篇元数据随时可看：`research get <doi｜arxiv-id｜openalex-id>` 打印完整 frontmatter（含被引、
JIF 估算、期刊分级与 WoS 收录号）；`research journal lookup <issn>` 则并排看一本刊在各免费指标层的结果。

**增强策略**：`get`/`read`/`add` 单篇默认做 WoS 增强；`search` 的逐条增强需显式 `--enrich`
（默认关，避免 N 次慢调用）。**主源是 OpenAlex**（检索 + 元数据 + JIF 估算），arXiv 为预印本源；
WoS Starter API 只是**可选增强**——补权威 Times Cited、WoS 收录号（`wos_id`）与 JCR 链接，但它
**不含**官方 JIF / JCR 分区 / ESI（那些需另行申请 **WoS Journals API**，升级路径与“WoS API
Expanded 也不含 JIF”这个易混淆点记在 `tools/wos_client.py` 的模块 docstring 里）。未配置或调用
失败时静默跳过，不阻塞主流程。

**期刊质量指标**（`research journal lookup <issn>` 可并排看）：官方 JIF 到位前由两个**免费**层填充——
`journal_tier`（OpenAlex `listed_in` 里 JUFO / Norway / KI-JL 的**专家评议**分级，零额外请求；对低引用
密度领域比 JIF 更贴近共识）与 `scimago_quartile`（SCImago SJR 本地索引 `data/scimago_index.json`，已随仓库
版本化，当前为 **SJR 2025 / 53404 条 ISSN**；换年版需手工下载官方 CSV 后跑
`research journal build-scimago --csv <path>` 重建，版本表与刷新步骤见 `data/SOURCE.md`；索引缺失或
该刊无 ISSN 时静默留空）。`jcr_quartile` 在 Journals API 接入前恒为空，三者语义不同、并存不冲突。

**引用完整性门**：`add` 写入前自动用 **OpenAlex + Crossref + arXiv 三源**交叉核验每条引用（标题/DOI/年份/期刊/首作者），实质冲突（FAIL）阻止入库（`--allow-fail` 越过 / `--no-verify` 跳过）；也可独立跑 `research citecheck <doi｜标题>`、`citecheck --all`（审计 `papers/` 全部笔记）、`citecheck --bib <refs.bib｜refs.md>`（核验一份参考文献）或 `citecheck --review`（前者的简写，扫全部综述）；有 FAIL 退出码 1，可作 CI 门。仅硬冲突（标题/DOI 不符、年份差≥2）判 FAIL；作者姓 / 年份差 1 / 期刊名差异只 WARN；某源不可达只降级、绝不误判。

> 这道门能证的是「引用存在且三源元数据一致」，**证不了**「综述对这篇论文说的话真是这篇论文说的」——
> 后者无法机械校验，靠约定兜：综述里每条事实性论断带一个指向 `cache/extracted/*.md` 的锚点。
> 指针是白捡的：`read` 会打印该路径，`rag search` 的每个 chunk 也自带 `source_path`。详见
> `.qoder/skills/literature-research/references/read.md`。

**缓存维护**：昂贵产物（PDF / 抽取全文 / 抓取 HTML）默认永久保留、命中即复用；用
`research cache stats｜clean｜prune` 治理磁盘（分层模型见 §3）。

**本地语义检索（RAG）**：`read`/`ingest` 抽取到 `cache/extracted/` 的全文可用 `research rag index` 建本地 embedding 索引（硅基流动 bge-m3，免费），随后 `research rag search '<query>'` 做纯 embedding 语义检索（返回 top-k 相关段落 + 出处引文 + 文件路径，无 LLM、零成本、不阻塞，是跨本地文献综合的主力）；`research rag ask '<query>'` 为可选的一句话综述（免费 Qwen2.5-7B → 不可用丝滑回退付费 Qwen2.5-32B → 全失败静默降级为 search，永不阻塞）。需 `SILICONFLOW_API_KEY`；`research rag status` 查看后端就绪与索引状态。

---

## 6. 用户可以做什么

- **审校**：随时修改 `papers/*.md` 中 AI 写的评价段落，AI 会尊重你的修改。重跑 `research read <id>`
  时走的是 **merge 而非覆写**：只填 frontmatter 里空着的键（并把补齐了哪些字段记进笔记的
  `## Changelog`），**正文逐字节不动**，你填过的 `my_rating` / `status` / `related_to_my_work` 永不被顶掉；
  无空键可补时文件根本不被触碰（mtime 不变）。只有显式给 `--overwrite` 才会用模板重建正文
  ——那会丢弃你写的全部评价，慎用。代价是：**merge 改不了错值**（它只填空、不覆写），
  一个字段语义修正前写下的旧值需手工改
- **提问**：直接问 "帮我看看 `papers/2018_zhu_….md` 里的方法部分"
- **触发流程**：例如 "帮我搜索过去两年关于非厄米 EP 的声学有源实现"，AI 会启动 `research` 全流程
- **不要做**：手动重命名 `papers/` 下的文件（会破坏索引链接）；手动编辑 `INDEX.md`（由 `research index`
  生成）；把 PDF 直接放到本目录（应放 Zotero）

---

## 7. 依赖

工具链依赖以下 Python 包，均在项目 uv 虚拟环境（`.venv`）中、已声明于根 `pyproject.toml`：

**核心**：
- `requests` — HTTP 客户端（OpenAlex / arXiv / WoS / Crossref 引用核验）
- `python-dotenv` — 加载项目根 `.env`
- `pyyaml` — paper-note frontmatter 的解析与序列化，由叶子模块 `tools/notes.py` 统一承担。
  它本就是项目运行时依赖（此前只 `scientific_plotting` 在用），**自本次重构起 literature_research
  也正式使用它**：原先 `research.py` / `cache_manager.py` / `citation_verify.py` 各自手搓的行式
  解析器已全部删除——那三份实现会跳过所有以 `-` 开头的行，使 block-style 的 `authors` / `topics`
  列表整体丢失，且返回值连引号一起带出。**不要再手搓 YAML。**

> **期刊指标层与参考文献解析无新增 pip 依赖**：SCImago 索引的构建/查询只用标准库 `csv` + `json`
> （`tools/journal_metrics.py`）；`citecheck --bib` 的 BibTeX / markdown 参考文献解析用正则而非
> `bibtexparser`（只需支持本项目 `compose refs.bib` 从 Zotero 产出的规整格式）。

> **引用完整性门无新增 pip 依赖**：`citation_verify.py`（`research citecheck` 与 `add` 写入前 gate）
> 走 **Crossref 免费 REST**（无需 key，`OPENALEX_EMAIL` 兼作 `mailto` polite-pool 标识）+ 复用既有
> `openalex_client` / `arxiv_client`，仅用标准库 `difflib` / `unicodedata` 做归一化比对。

**Zotero 文献库层**（不再是本项目 pip 依赖）：
- 委托社区 **`zotero-mcp`** 的 `zotero-cli`，经 `uv tool install "zotero-mcp-server[pdf,scite]"`
  独立安装于隔离工具环境（见 `scripts/zotero_mcp/`）；本项目 `research add/library` 与
  `document_writing` 的 refs 导出仅 subprocess 调用它。原 `pyzotero` / `markdown`（旧
  `zotero_bridge.py` 手写 API 回退与 md→html 所需）已随该模块删除而移除。

**PDF 抽取与全文抓取**：
- **`mineru-open-sdk`** — 公式精读**主力**：封装 MinerU 云端 Open API（VLM + OCR，公式→LaTeX、
  表格→HTML）的鉴权/上传/轮询，仅依赖 `httpx` + `MINERU_TOKEN`，无本地重型依赖
- `pymupdf4llm` — 本地**兜底**后端：纯 CPU、快，但**公式会丢失**
- `playwright>=1.63.0` — 付费墙论文的 HTML 全文 / 正文 PDF / 补充材料抓取
- `trafilatura` — 从 Playwright 抓取的 HTML 提取结构化 markdown 正文（替代手写 innerText）

> 旧的 `marker-pdf` / `magic-pdf`（本地 MinerU）实测本机不可用，且云端 MinerU 可完全替代，已从依赖中**移除**。

**本地语义检索（RAG）**：
- **`paper-qa`**（PaperQA2，**核心依赖**，无 torch）— 本地文献库语义索引/检索引擎；embedding 走硅基流动
- **硅基流动 SiliconFlow**（OpenAI 兼容，经 `litellm`）— `bge-m3` embedding（免费）建索引 + 检索；`ask` 综述可选用 Qwen 免费档 → 付费回退。需 `SILICONFLOW_API_KEY`

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

`OPENALEX_EMAIL`、`OPENALEX_API_KEY`、`WOS_API_KEY`、`ZOTERO_USER_ID`、
`ZOTERO_API_KEY`、`MINERU_TOKEN`、`PDF_EXTRACT_BACKEND`、`SILICONFLOW_API_KEY`、`CACHE_DIR`、
`HTTP_TIMEOUT_SECONDS`、`HTTP_MAX_RETRIES`。缓存治理（均可选）：`CACHE_B_MAX_AGE_DAYS`（默认 7）、
`CACHE_AUTOCLEAN_INTERVAL_DAYS`（默认 7）、`CACHE_SOFT_LIMIT_MB`（默认 2048）。本地 RAG（均可选，
有内置默认值）：`SILICONFLOW_BASE_URL`、`PQA_EMBEDDING`（默认 `openai/BAAI/bge-m3`）、`PQA_LLM`（免费档
`openai/Qwen/Qwen2.5-7B-Instruct`）、`PQA_LLM_FALLBACK`（付费回退 `openai/Qwen/Qwen2.5-32B-Instruct`）、
`PQA_HOME`（默认 `cache/rag`）。

除 arXiv 外全部可选；缺失时对应功能静默降级（OpenAlex 无 key 时每天仅 $0.10 用量预算，免费 key 提到 $1/天）。**Crossref 引用核验无需任何 key**（`OPENALEX_EMAIL` 仍作 `mailto` 附加，但自 2026-02-13 起配额由 key 决定，不再是 polite-pool 模型）。用 `research doctor` 一览当前凭据与可达性（含 Crossref 就绪行）。

---

_Last updated: 2026-10-03 · Maintained by: AI Agent (Qoder) · 配对 skill：`.qoder/skills/literature-research/`_
