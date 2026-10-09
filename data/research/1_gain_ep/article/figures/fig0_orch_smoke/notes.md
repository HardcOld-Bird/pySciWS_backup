# fig0_orch_smoke — 迭代日志

- **所属研究**：gain_ep
- **风格预设**：aps / 宽度 single
- **产物**：`out/fig0_orch_smoke.eps`（投稿）、`.pdf`、`.svg`（人工微调）、`_preview.png`（Agent 视觉校验）

## 迭代记录

<!-- 每次修改在此追加：日期 / 需求（换数据|换画法|换风格）/ 结果 -->

- **2026-10-09 / 首版（task-20261009-201037-fig0-orch-smoke）**：合成避免交叉色散单面板图。
  数据 `data/dispersion.csv`（模型 ω± = ω̄ ± √(δ²+κ²)，裸分支 ω₁ = ω₀ + c(k−k₀)、
  ω₂ = ω₀ + s₂(k−k₀)，参数 c=1, s₂=−0.5, k₀=1, ω₀=1, κ=0.06 → Δω=0.12, g₀=0.18；
  生成脚本见 figure pod 的 `bench/gen_dispersion_fig0.py`）。
  绘图代码：实线 Re ω±（okabe-ito 1/2 号）、灰虚线裸分支、k₀ 处 Δω 双向箭头与注记、图例右下。
  当时管线代码暂放本目录内，因 `runner.discover_pipeline()` src 优先且 figure pod 对 `src/`
  无写权限，`build` 只能出脚手架占位图 → 该次交付为 blocked（建议已采纳，见审批回复）。
- **2026-10-09 / 换代码位（task-20261009-202508-fig0-orch-smoke-2）**：白名单扩至
  `src/pysci/research/gain_ep/` 后，绘图逻辑移植进
  `src/pysci/research/gain_ep/article/figures/fig0_orch_smoke.py`（替换脚手架 2×2 正弦占位实现），
  本目录内的重复管线已删除（不留两份实现）。数据仍由本目录 `data/dispersion.csv` 提供，
  管线经 `research_dir` 注入或 `pysci.paths.research_fig_dir` 解析定位。
  新增本目录 `STYLE.yaml`（aps / single 85 mm），使不带 `--width` 的 `build`/`audit` 也按单栏判定。
- **状态：已走正规管线产出交付件**。`pysci-figures build <figdir>` → out/ 四格式 + `_preview.png`；
  已 Read 预览做视觉自检（曲线、间隙标注、图例位置、Arial 字体均无缺陷）；
  `pysci-figures audit <figdir> --panels` → width 85.0/85.0 mm、min_fontsize 7.0 pt、
  色盲不可辨对 0、**PASS (0 error, 0 warn)**。
