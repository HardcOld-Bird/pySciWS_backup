# reviewer 长期记忆

> 本文件由 reviewer 自维护（启动自动加载）。只记稳定事实与约定。

## 身份

- pySci 编排体系 reviewer（期刊审稿人式主观质量审查员），charter 见 `.qoder/rules/charter.md`。
- 当前 rubric：`rubrics/generic.md`（最小占位版 v0.1，判定从宽；正式版待用户审定）。
- 图件专用 rubric：`rubrics/scientific-plotting.md`（科研绘图通用审查量表 **v0.2，已生效**）——
  数据驱动图件的六维度 D1–D6 + 四步审查流程 + 硬项 H1–H5，与 generic 叠加使用；视觉图件审查
  以本表为准。该表末节说明「非 scaffold 图目录 → `pysci-figures audit` 返回 N/A」的适用边界。
- 演示文稿专用 rubric：`rubrics/presentation.md`（演示文稿通用审查量表 **v0.1，已生效**）——
  pptx/幻灯片类产物的六维度 D1–D6 + 四步审查流程 + 硬项 H1–H5（逐页视觉核验 / 文字可读性 /
  图文一致 / 零遮挡与模板一致 / 非专业听众信息密度），与 generic 叠加；**先跑机械视觉闸**
  `compose slides verify`（派发侧 `check="pptx-visual"`）逐页渲染，再 Read 渲染 PNG 逐页判读。

## 审查经验（初始为空——工作中积累：领域基准、常见缺陷模式、对照文献线索）

- 2026-10-09 rev-20261009-231247（fig0_orch_smoke，PASS）：图件审查路径 = Read `out/*_preview.png`
  与主 png 做视觉判读 + 独立复跑 `pysci-figures audit <figdir> --panels` 核验生产者自检声明
  （只读、低成本）；数据自洽可按 CSV 头声明的模型手算抽查（ω± = ω̄ ± √(δ²+κ²)）。
  小间隙双向箭头（Δω 标注）箭头头部易大于间隙本身而呈实心块状——美观度扣分点而非硬缺陷。
- 审查任务书仅给产物路径、不嵌原生产任务目标陈述；G2 判定须依赖生产者 notes.md 自述，
  独立性略欠（已 infra_suggestion 转呈）。
