---
# ============================================================
# 检索快照元数据
# ============================================================
query_date: ""              # YYYY-MM-DD
query_slug: ""              # 用于文件命名
queried_by: "AI Agent"
purpose: ""                 # 本次检索的目标（用户原话或 AI 概括）

# ============================================================
# 检索参数
# ============================================================
sources: []                 # ["openalex", "arxiv", "wos", "scopus"]
query_string: ""            # 实际提交的检索式
filters:
  year_range: ""
  min_citations: null
  venues: []
  concepts: []
  oa_only: false
sort_by: ""                 # "relevance" | "cited_by_count:desc" | "publication_date:desc"
results_returned: 0
results_after_screening: 0

# ============================================================
# 状态
# ============================================================
screening_status: pending   # pending | screened | converted-to-notes | archived
---

# Shortlist: {{query_slug}} ({{query_date}})

> **检索目的**：{{purpose}}

---

## 检索式（Reproducibility）

```
Source: OpenAlex
  URL: https://api.openalex.org/works?search=...&filter=...&sort=...
  Params:
    search = "..."
    filter = "..."
    sort   = "..."
    per-page = 50
```

---

## 初筛标准（Screening Criteria）

- [ ] 主题相关性（AI 判定：直接相关 / 间接相关 / 无关）
- [ ] 期刊质量（JIF > 阈值 或 分区 ≤ Q2）
- [ ] 引用数（发表 ≥ 1 年时，被引 > N）
- [ ] 时效性（发表年 ∈ [YYYY, YYYY]）
- [ ] OA 可获取性（gold/green/bronze）

---

## 候选论文列表

### 强烈推荐（Highly Recommended）

#### 1. {{paper_title}}
- **Authors**: {{...}}
- **Journal / Year**: {{journal}} ({{year}}), JIF {{jif}}, {{quartile}}
- **DOI**: {{doi}}
- **Cited by**: {{cited_by_count}} (normalized: {{cited_by_count_normalized}})
- **OA**: {{oa_status}}, URL: {{oa_url}}
- **arXiv**: {{arxiv_id}}
- **Why recommended**: （AI 一句话说明）
- **Relevance to my work**: high / medium / low
- **Action**: [ ] 入库 Zotero  [ ] 生成 paper note  [ ] 精读

#### 2. ...

---

### 值得跟进（Worth Following）

#### 1. ...

---

### 已排除（Excluded）

| # | Title | Journal | Year | 排除理由 |
|---|---|---|---|---|
|   |       |       |      |          |

---

## AI 元观察（Meta-observations）

_（读完候选列表后，AI 对整个检索结果的宏观判断）_

- **领域热度**：这个方向近 2 年是升温 / 平稳 / 降温？
- **主导研究组**：出现频率最高的团队？
- **方法论趋势**：主流方法有什么变化？
- **空白点**：有哪些明显该做但没人做的工作？

---

## 下一步行动

- [ ] 将 "强烈推荐" 全部入库 Zotero
- [ ] 为入库论文批量生成 `papers/*.md` 骨架
- [ ] 下载并抽取全文
- [ ] 精读并填充评价
- [ ] 更新 `INDEX.md`
- [ ] （可选）基于本快照生成 `reviews/{{YYYY-MM}}_{{topic}}_survey.md`

---

## Changelog

- YYYY-MM-DD HH:MM: 检索由 AI 执行，得到 N 条结果
- YYYY-MM-DD HH:MM: 用户审校 & 排除 M 条
- YYYY-MM-DD HH:MM: K 条入库
