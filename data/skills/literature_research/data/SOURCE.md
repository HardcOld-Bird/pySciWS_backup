# `data/` —— 外部数据资产（非缓存、非笔记）

本目录存放**需要随仓库版本化的第三方数据资产**：它们既不是可随手重取的缓存
（`cache/`，已 git-ignore），也不是用户/AI 产出的笔记（`papers/`、`reviews/`、
`shortlists/`），而是「下载一次、用很久、丢了要重新走一遍人工流程」的中间产物。

| 文件 | 内容 | 是否入 Git | 大小 |
|---|---|---|---|
| `scimago_index.json` | SCImago SJR 分区索引（按 ISSN 精确匹配） | **是** | 约 1.4 MB |
| `SOURCE.md` | 本文件：数据来源、版本、归属声明与刷新步骤 | **是** | — |
| `*.csv` | 用户手工下载的官方原始导出 | **否**（`.gitignore` 已排除） | 约 11 MB |

---

## SCImago Journal & Country Rank（SJR）

### 归属声明（二次分发必须保留）

> **SCImago Journal & Country Rank**, data based on **Scopus (Elsevier B.V.)**.
> 来源：<https://www.scimagojr.com/journalrank.php>

### 当前版本

| 项 | 值 |
|---|---|
| 下载 URL | <https://www.scimagojr.com/journalrank.php>（页面底部的数据下载区，选 *Scimago Journal & Country Rank \<年份\>* 的 **CSV**） |
| SJR 版本年 | **2025**（构建时从 CSV 表头的 `Total Docs. (2025)` 自动推断，非人工填写） |
| 下载日期 | **2026-10-04** |
| 源 CSV 文件名 | **`scimagojr 2025.csv`**（10.73 MB；原始 CSV 不入库，只在此记录文件名以便追溯） |
| 索引条目数 | **53404** 条 ISSN（`research journal status` 会打印） |
| 索引构建时间 | 见 `scimago_index.json` 的 `_meta.built_at` |

> 上表随每次刷新更新（见下方第 5 步）。索引若被删掉、尚未重建，所有查询路径都会
> 静默返回空值、`scimago_quartile` 字段留空，不影响任何其它功能。

> 下载步骤刻意**不写死版本年**（上表记的是「已构建的这一版」，随刷新更新）：
> `scimagojr.com` 对程序化请求返回 403（含 AI 的网页读取工具），维护者无从核实
> 页面当前提供的是哪一版，写下一个年份只会变成一句无法检验的断言。SCImago 每年
> 约 5-6 月随新版 Scopus 发布上一年的数据，下载时以页面实际标明的最新版为准；
> 版本年由构建过程从 CSV 表头自行推断，不靠人记。

### 为什么只跟踪索引 JSON、不跟踪原始 CSV

官方导出约 11 MB（2025 版实测 10.73 MB），其中绝大部分列（`Total Docs.` /
`Total Refs.` / `%Female` / `Overton` / `SDG` / `Country` …）对本项目的**分区判断**
毫无用处。
`journal_metrics.build_scimago_index` 只保留 `Issn` / `SJR` / `SJR Best Quartile` /
`H index` 四列，压成约 1.4 MB 的紧凑 JSON（2025 版实测 1,427,727 B，含末尾一个换行，
以免被 pre-commit 的 `end-of-file-fixer` 改写）——这个体积 git 可以长期承受，而 11 MB
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
   分区会随年份变动，自动覆盖会让用户的历年判断失去参照。

   确需刷新时，**不要**清空后重跑 `research read <id>`。`_merge_note` 的目标路径由
   **本次新取到的** frontmatter 现算（`notes.note_filename`，输入含 `short_title`），而
   `_make_short_title` 会剔除 of/and/in 等虚词后取前 6 个实词；一旦算出的名字与磁盘上的
   旧名不同，`read` 就**新建一个重复笔记**而不是合并——实测当前三篇存量笔记里有两篇
   会这样（`2018_zhu` 因 `short_title` 新旧不一致，`2023_fang` 因文件名本就待改）。
   安全做法是按白名单只改 `scimago_quartile` 一个键：值取自
   `research journal lookup <ISSN>`（ISSN 出自该笔记对应的 OpenAlex source 记录），
   再往该笔记的 `## Changelog` 追加一行；改前改后用正文 SHA256 自证逐字节未变
   （配方见 `.qoder/skills/literature-research/references/maintenance.md` §5e）。
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
