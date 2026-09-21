---
# ============================================================
# 综述元数据
# ============================================================
topic: ""
topic_slug: ""
created_date: ""
last_updated: ""
author: "AI Agent + User"
status: draft                 # draft | in-progress | completed | archived

# ============================================================
# 检索范围
# ============================================================
time_window: ""               # 例如 "2024-01 至 2026-09"
sources_used: []              # [openalex, arxiv, wos, zotero]
query_strings: []             # 实际使用的检索式
inclusion_criteria: ""
exclusion_criteria: ""

# ============================================================
# 涉及的论文（内部链接）
# ============================================================
papers_reviewed: []           # ["[[2026_zhang_xxx]]", "[[2025_jones_yyy]]"]
papers_total_count: 0
papers_read_count: 0
---

# {{topic}} — 综述笔记

> **综述目标**：（一句话说明这份综述要回答什么问题）

---

## 1. 领域概览（Field Overview）

_（AI 起草：这个子领域当前的研究图景，主要问题，主流思路）_

---

## 2. 关键脉络（Key Threads）

按研究路线分组，每组列出代表论文与核心贡献。

### 2.1 路线 A：{{...}}

- 代表论文：[[...]]
- 核心思想：
- 优势 / 局限：

### 2.2 路线 B：{{...}}

- 代表论文：[[...]]
- 核心思想：
- 优势 / 局限：

---

## 3. 时间线（Timeline）

| 年份 | 里程碑工作 | 主要突破 |
|---|---|---|
|      |            |          |
|      |            |          |

---

## 4. 争议与未解问题（Controversies & Open Questions）

- **争议 1**：
  - 正方（论文）：
  - 反方（论文）：
  - 我的判断：

- **未解问题 1**：
  - 为什么难：
  - 现有尝试：
  - 可能的路径：

---

## 5. 与我的研究的接口（Interface with My Work）

### 5.1 可以直接借鉴

-

### 5.2 需要与之区分（差异化定位）

-

### 5.3 潜在的合作/竞争者（研究组画像）

- 组名：
- PI：
- 机构：
- 代表性工作：
- 与我方向的重合度：（high/medium/low）

---

## 6. 投稿建议（Journal Landscape）

针对这个子领域，我的成果适合投哪些期刊？

| 期刊 | JIF | 分区 | 领域匹配度 | 接收难度 | 备注 |
|---|---|---|---|---|---|
|      |     |      |            |          |      |

---

## 7. 检索式记录（Reproducibility）

_（记录本次综述使用的检索式、时间窗、过滤条件，便于未来更新）_

```
OpenAlex:
  search = "..."
  filter = concepts.id:..., publication_year:2024-2026, cited_by_count:>10

arXiv:
  search_query = "cat:cond-mat.mes-hall AND all:..."
  date_range = "..."
```

---

## 8. Changelog

- YYYY-MM-DD: 初稿由 AI 生成
- YYYY-MM-DD: 用户审校 & 补充
