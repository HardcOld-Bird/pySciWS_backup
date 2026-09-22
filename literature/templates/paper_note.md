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
journal: ""
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
extracted_md_path: ""              # AI 抽取的全文 Markdown 路径（cache/extracted/），cache prune 会保护它
oa_url: ""
oa_status: ""            # gold | green | bronze | hybrid | closed

# ============================================================
# 质量与影响力指标
# ============================================================
cited_by_count: null
cited_by_count_normalized: null   # OpenAlex 的规范化引用数（相对于同领域同年份）
jif: null                          # Journal Impact Factor（WoS 官方或 OpenAlex 估算）
jif_5yr: null
jcr_quartile: ""                   # Q1 | Q2 | Q3 | Q4
scimago_quartile: ""               # Q1 | Q2 | Q3 | Q4
citescore: null                    # Scopus 指标（若有）
esi_highly_cited: false            # 是否 ESI 高被引（前 1%）
esi_hot_paper: false               # 是否 ESI 热点论文（前 0.1%）
journal_h_index: null

# ============================================================
# AI 分类标签
# ============================================================
topics: []
# 示例: [non-hermitian, exceptional-point, acoustic-metasurface, active-gain, topological]
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

- **Journal**：{{journal}}（JIF {{jif}}, {{jcr_quartile}}）
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
