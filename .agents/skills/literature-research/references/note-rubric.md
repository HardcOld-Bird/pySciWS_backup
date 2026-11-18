# 结构化笔记 rubric 与 frontmatter 契约

> 笔记文件：`data/skills/literature_research/papers/<key>.note.md`。
> frontmatter 机械契约由胶水 `frontmatter` 模块执行（`normalize_note` 写入前归一化）。

## frontmatter 字段表

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `title` | ✓ | str | 论文标题（去末尾句点） |
| `authors` | ✓ | str | 第一作者 et al. 或全列，自由文本 |
| `year` | ✓ | int | 数字，非法即拦截 |
| `venue` | | str | 期刊/会议；预印本写 `arXiv` |
| `doi` | | str | 归一化 DOI（不带 `https://doi.org/` 前缀） |
| `arxiv_id` | | str | 如 `2401.01234` |
| `status` | | enum | 状态机，见下；缺省 `unread` |
| `tags` | | list | 行内列表 `[a, b]`；归一化为小写、去重、字母序 |
| `zotero_key` | | str | Zotero 条目 key（入库后回写） |
| `source` | | str | 提取来源：`mineru` / `pymupdf4llm` / `tex`（降级可追溯） |

status 状态机：`unread → skimmed → read → cited`（只进不退；`cited` 表示已被对外产物引用并通过
citecheck）。非法值写入前拦截（`ValueError`）。

## 笔记正文 rubric（五段固定骨架）

1. **TLDR**：≤3 句，说清问题/方法/结论。
2. **Key Claims**：逐条列核心主张，**每条挂页码或章节证据**（`(p.6, Fig.3)` 式）；无证据的主张
   标注 `[推断]`。
3. **Novelty**（1–5 + 一句话理由）：相对既有工作的增量。
4. **Rigor**（1–5 + 一句话理由）：方法/实验/统计的可靠性；列出作者自认的局限。
5. **Relevance**（1–5 + 一句话理由）：与当前研究线的关联；指出可复用的代码/数据/方法。

评分一律给出理由句——只有分数没有理由的笔记视为未完成。

## note merge 合同（增量合并）

- 已有字段**不丢**：合并只增不删。
- 新证据**追加不覆盖**：同一 claim 的新证据追加行，不改旧行。
- 冲突**显式标注**：`conflict: <描述>（<来源>）`，交人工裁决，不静默择一。
- 合并后必须重跑 `frontmatter.normalize_note` 并重建 INDEX.md。

## INDEX.md

由 `frontmatter.build_index(papers_dir)` 生成（key | title | year | status | tags，按年份倒序），
**勿手改**；坏笔记（归一化失败）在表中标注 `INVALID` 并继续汇总，事后修复。
