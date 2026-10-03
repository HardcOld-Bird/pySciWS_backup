# `data/` —— 外部数据资产（非缓存、非笔记）

本目录存放**需要随仓库版本化的第三方数据资产**：它们既不是可随手重取的缓存
（`cache/`，已 git-ignore），也不是用户/AI 产出的笔记（`papers/`、`reviews/`、
`shortlists/`），而是「下载一次、用很久、丢了要重新走一遍人工流程」的中间产物。

| 文件 | 内容 | 是否入 Git | 大小 |
|---|---|---|---|
| `scimago_index.json` | SCImago SJR 分区索引（按 ISSN 精确匹配） | **是** | 约 1.4 MB |
| `SOURCE.md` | 本文件：数据来源、版本、归属声明与刷新步骤 | **是** | — |
| `*.csv` | 用户手工下载的官方原始导出 | **否**（`.gitignore` 已排除） | 约 15 MB |

---

## SCImago Journal & Country Rank（SJR）

### 归属声明（二次分发必须保留）

> **SCImago Journal & Country Rank**, data based on **Scopus (Elsevier B.V.)**.
> 来源：<https://www.scimagojr.com/journalrank.php>

### 当前版本

| 项 | 值 |
|---|---|
| 下载 URL | <https://www.scimagojr.com/journalrank.php>（页面底部 “Download data” → *Scimago Journal & Country Rank 2024*） |
| SJR 版本年 | **待填**（构建时从 CSV 表头的 `Total Docs. (YYYY)` 推断，或用 `--year` 指定） |
| 下载日期 | **待填** |
| 源 CSV 文件名 | **待填**（原始 CSV 不入库，只在此记录文件名以便追溯） |
| 索引条目数 | **待填**（`research journal status` 会打印） |
| 索引构建时间 | 见 `scimago_index.json` 的 `_meta.built_at` |

> 索引尚未构建时上表的「待填」保持原样即可——所有查询路径都会静默返回空值，
> `scimago_quartile` 字段留空，不影响任何其它功能。

### 为什么只跟踪索引 JSON、不跟踪原始 CSV

官方导出约 15 MB，其中绝大部分列（`Total Docs.` / `Total Refs.` / `%Female` /
`Overton` / `SDG` / `Country` …）对本项目的**分区判断**毫无用处。
`journal_metrics.build_scimago_index` 只保留 `Issn` / `SJR` / `SJR Best Quartile` /
`H index` 四列，压成约 1.4 MB 的紧凑 JSON——这个体积 git 可以长期承受，而 15 MB
的二进制式宽表不行。原始 CSV 已被 `.gitignore` 的 `*.csv` 规则排除，**不要**强制添加。

### 为什么不自动下载

`scimagojr.com` 对程序化请求返回 **403**，且下载入口需要表单/JS 交互。
因此构建流程刻意设计为「用户手工下载 → 把路径传给 CLI」，不做任何抓取尝试。

### 年度刷新步骤

SCImago 每年（约 5-6 月）随新版 Scopus 数据发布一次 SJR。刷新流程：

1. 浏览器打开 <https://www.scimagojr.com/journalrank.php>，滚到页面底部的
   “Download data” 区，下载 **Scimago Journal & Country Rank \<年份\>** 的 CSV
   （默认是分号分隔的欧洲风格；本模块会自动判定分隔符，逗号版同样可用）。
2. 把 CSV 放到本目录（`data/skills/literature_research/data/`）或任意临时位置——
   放在本目录也不会被提交（`*.csv` 已忽略）。
3. 构建索引（会**原地覆盖** `scimago_index.json`）：

   ```powershell
   [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
   uv run pysci-research journal build-scimago --csv 'data\skills\literature_research\data\scimagojr 2025.csv'
   ```

   版本年通常能从 CSV 表头自动推断；推断不出时用 `--year 2025` 显式指定。
4. 核对：

   ```powershell
   uv run pysci-research journal status
   uv run pysci-research journal lookup 0031-9007   # PRL，应为 Q1
   ```

5. 更新本文件「当前版本」表里的 SJR 版本年 / 下载日期 / 源 CSV 文件名 / 条目数。
6. 让既有笔记吃到新分区：`scimago_quartile` 是**只在字段为空时才补**的（`merge_frontmatter`
   的「只填空、绝不覆盖」语义）。因此旧笔记不会自动刷新分区——这是刻意的：
   分区会随年份变动，自动覆盖会让用户的历年判断失去参照。确需批量刷新时，
   把目标笔记的 `scimago_quartile` 清空后重跑 `research read <id>`。
7. 原始 CSV 可以删掉（索引已自足），也可以留着——反正不入 Git。

---

## 与 `journal_tier` 的分工

`scimago_quartile` 不是本目录之外唯一的免费期刊质量指标。三者语义不同、并存不冲突：

| 字段 | 来源 | 需要本目录的数据吗 | 语义 |
|---|---|---|---|
| `scimago_quartile` | SCImago SJR（基于 Scopus 引用数据） | **是** | 按学科领域内的引用表现分 Q1-Q4 |
| `journal_tier` | OpenAlex `sources.listed_in`（JUFO / Norway / KI-JL） | 否（零额外请求） | **专家小组按学科评议**的档次：top / leading / basic |
| `jcr_quartile` | WoS Journals API | 否（**尚未接入**） | 官方 JCR 分区，当前恒为空 |

对声学这类**低引用密度**领域，`journal_tier` 往往比引用类指标更贴近领域共识：
`J. Acoust. Soc. Am.` 的 OpenAlex `2yr_mean_citedness` 只有 0.82，但 JUFO 把它判为
3 级（最高档），与 Nature / PRL 同级。判读时请把两者放在一起看。
