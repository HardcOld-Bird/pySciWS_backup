# reviewer 长期记忆

> 本文件由 reviewer 自维护（启动自动加载）。只记稳定事实与约定。

## 身份

- pySci 编排体系 reviewer（期刊审稿人式主观质量审查员），charter 见 `.qoder/rules/charter.md`。
- 当前 rubric：`rubrics/generic.md`（最小占位版 v0.1，判定从宽；正式版待用户审定）。

## 审查经验（初始为空——工作中积累：领域基准、常见缺陷模式、对照文献线索）

- 2026-10-09 rev-20261009-231247（fig0_orch_smoke，PASS）：图件审查路径 = Read `out/*_preview.png`
  与主 png 做视觉判读 + 独立复跑 `pysci-figures audit <figdir> --panels` 核验生产者自检声明
  （只读、低成本）；数据自洽可按 CSV 头声明的模型手算抽查（ω± = ω̄ ± √(δ²+κ²)）。
  小间隙双向箭头（Δω 标注）箭头头部易大于间隙本身而呈实心块状——美观度扣分点而非硬缺陷。
- 审查任务书仅给产物路径、不嵌原生产任务目标陈述；G2 判定须依赖生产者 notes.md 自述，
  独立性略欠（已 infra_suggestion 转呈）。
