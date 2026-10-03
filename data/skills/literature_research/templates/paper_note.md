---
# ============================================================
# 元数据（由 AI 从 OpenAlex / arXiv / WoS / Zotero 自动填入，用户审校）
# ============================================================
title: ""
short_title: ""
authors: []
first_author_last_name: ""
corresponding_author: ""
year: null
publication_date: ""
journal: ""                        # 纯刊名（如 "Phys. Rev. Lett."）；arXiv 尚未发表时为 "arXiv preprint"
journal_ref: ""                    # 源数据的原始引文串（如 "Phys. Rev. Lett. 121, 124501 (2018)"）；拆解是有损的，留着原文才能审计
publisher: ""
volume: ""
issue: ""
pages: ""
doi: ""
arxiv_id: ""
openalex_id: ""
wos_id: ""
zotero_key: ""
zotero_uri: ""
local_pdf_path: ""
extracted_md_path: ""              # AI 从 PDF 抽取的全文 Markdown（cache/extracted/）；进 rag 语料，cache prune 会保护它
extracted_html_path: ""            # 兜底路径：无 PDF 时从网页抓的正文（cache/html_fulltext/，trafilatura 抽取）。公式常丢失，且**刻意不进** rag 语料；prune 同样保护它
oa_url: ""
oa_status: ""            # gold | green | bronze | hybrid | closed

# ============================================================
# 质量与影响力指标
# ============================================================
cited_by_count: null
cited_by_count_normalized: null   # OpenAlex 的规范化引用数（相对于同领域同年份）
jif: null                          # OpenAlex summary_stats.2yr_mean_citedness 估算值，**非官方 JIF**；中段（3-9）较准，顶刊与低引用密度刊可低估 2-3 倍
jif_5yr: null                      # 无免费程序化来源，恒为 null
jcr_quartile: ""                   # Q1|Q2|Q3|Q4；官方 JCR 分区需 WoS Journals API（未接入）→ 当前恒为空
scimago_quartile: ""               # Q1|Q2|Q3|Q4；来自本地 SCImago SJR 索引（按 ISSN 精确匹配），索引未建时为空
citescore: null                    # Scopus CiteScore；无免费程序化来源 → 当前恒为 null
esi_highly_cited: null             # 是否 ESI 高被引（前 1%）；null = 未知——需 WoS Journals API，当前无法获取
esi_hot_paper: null                # 是否 ESI 热点论文（前 0.1%）；null = 未知——同上
journal_h_index: null              # OpenAlex summary_stats.h_index（期刊层面的 h 指数）
journal_tier: ""                   # top|leading|basic；由 listed_in 的**专家评议**名单派生（JUFO/Norway/KI-JL 取最高档）；"" = 三套名单都没收录（无从判断，不等于低档）
journal_tier_basis: []             # 得出上面档次的依据条目，如 [jufo-3, norway-2]；留着才能审计（光看一个 top 不知道是哪国的评议）
listed_in: []                      # OpenAlex sources.listed_in 原始值；除上述三套外还含 cwts-core/erih-plus/medline/doaj 等**二元**收录标记（不含档次，不参与派生）

# ============================================================
# AI 分类标签
# ============================================================
topics: []
# 示例: [non-hermitian, exceptional-point, acoustic-metasurface, active-gain, topological]
# 注：topics 是留给 AI 填的**研究主题**标签，不要把它当成源数据库的分类码
#     （arXiv categories 已存在 keywords_auto 里，两键重复会让 AI 无处填自己的判断）
methods: []
# 示例: [transfer-matrix, coupled-mode-theory, FEM-simulation, coupled-oscillator-model]
systems: []
# 示例: [tube-array, membrane-resonator, piezoelectric, loudspeaker-array]
related_to_my_work: null           # high | medium | low | none
related_to_my_work_reason: ""

# ============================================================
# 状态
# ============================================================
status: unread                     # unread | reading | read | archived | rejected
my_rating: null                    # 1–5 星；null 表示未评
added_date: ""
last_reviewed: ""
review_count: 0

# ============================================================
# AI 生成的关键词（用于语义检索）
# ============================================================
keywords_auto: []
---

# {{short_title}} ({{first_author_last_name}} {{year}})

> **一句话定位**：（AI 读完 abstract 后填写：这篇论文在做什么、在哪个层次上做得好/不好）

---

## TLDR

_（≤150 字的段落总结，读完 abstract + intro 最后一段 + conclusion 生成）_

---

## Key Claims

1.
2.
3.

---

## Method Summary

**模型 / 理论框架**：

**关键假设**：

**近似 / 简化**：

**数值 / 实验手段**：

**可复用性**：（我能否把方法直接搬到我自己的系统上？）

---

## Main Results

**核心结果**：

**关键图表**：（列出图号 + 一句话说明；必要时用 `![](path)` 嵌入截取）

**关键公式**：
```latex
% 若论文核心结论可用 1–3 个公式概括，写在这里
```

---

## Novelty Assessment

- **What's new**：
- **Compared to prior work**：（列出 1–3 篇最接近的前期工作，说明差异）
- **Field-context**：（相对于同期同领域工作，本文是引领、跟进、还是补漏？）
- **Novelty score**：（1–5，AI 打分并给理由）

---

## Rigor Assessment

- **Assumptions validity**：（假设是否在实验/数值条件下成立？）
- **Potential weaknesses**：（AI 找到的漏洞或可质疑处）
- **Reproducibility**：（是否给出足够细节复现？补充材料是否完备？代码/数据是否公开？）
- **Rigor score**：（1–5）

---

## Journal-tier Justification

- **Journal**：{{journal}}
- **Metrics**：JIF {{jif}}（OpenAlex 估算，非官方 JCR）｜SCImago {{scimago_quartile}}｜JCR {{jcr_quartile}}（待 WoS Journals API）
- **Expert tier**：{{journal_tier}}（依据 {{journal_tier_basis}}）
  > 对声学这类**低引用密度**领域，专家评议档次比 JIF 更贴近领域共识：
  > JASA 的 2yr_mean_citedness 只有 0.82，但 JUFO 把它判为 3 级（最高档）。
- **Fit assessment**：
  - [ ] **Over-claimed**：论文的分量不足以支撑该刊
  - [ ] **Matched**：恰当
  - [ ] **Under-placed**：论文的分量高于该刊
- **Reasoning**：（说明为什么这样判断）

---

## Relevance to My Research

- **Direct borrow**：（我能直接借鉴的点：方法、模型、公式、实验方案）
- **Indirect inspiration**：（启发性的点：思路、类比、可迁移的物理图像）
- **Comparison needed**：（我的未来工作是否需要与之对比或明确区分？）
- **Citation intent**：（若我引用，会放在什么章节？background / method / result / discussion）

---

## Related Papers

_内部链接使用 `[[filename-without-ext]]` 或 `[显示名](papers/xxx.md)`_

-
-
-

---

## Quoted Excerpts

> "..." (Sec. X, p. Y)

> "..." (Fig. Z caption)

---

## Follow-up Questions

- [ ]
- [ ]

---

## My Notes (Free-form)

_（用户或 AI 随时补充；此段落不受模板约束）_

---

## Changelog

- YYYY-MM-DD: Created (AI auto-fill from OpenAlex/arXiv)
- YYYY-MM-DD: TLDR & Key Claims drafted
- YYYY-MM-DD: Full read completed, evaluation filled
- YYYY-MM-DD: User review
