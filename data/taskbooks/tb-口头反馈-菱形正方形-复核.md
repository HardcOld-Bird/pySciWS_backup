# 任务：口头反馈修正复核——正方形菱形/1.2m/r=0.60（VERDICT）

## 对象

`data/research/1_gain_ep/article/figures/beam_scatter/beam_scatter_combined_r7.png`
（`_assemble_r7.py`）；素材 `field_r7_*.png`＋`_field_r7.py`；数据
`simulation/field_8period_1p2m_ep3/`＋`11 场图域_1.2m方域.mph`。

## 复核范围（通用标准＋硬项，一次收口）

1. **正方形硬项**（用户明文「菱形永远正方形」）：内框四边等长 **0.31m**、外框四边
   等长 **0.3813m**——独立复算顶点间距离（勿沿用交付自报）；近边 0.25m、远边
   0.56m；±45° 两区；
2. **包容**：内框在 r=0.60 显示半圆内留边（远边 4.0cm、远角 1.9cm 读数复核）；
3. **vmax 严格 y>0**：`pysci-figures vmax-check` 逐组比值∈[1.0,1.1]（独立复跑）；
   schema 声明值与实测一致（F9 口径）；
4. **EP 子问＋域/寻点版本**：domain-check PASS 无软告警（1.2m-v1）；寻点偏移
   （Δ=0 @2e-4）申报如实；
5. 禁回退抽查：col-cb 紧邻｜雷达 sha 不变｜行序｜两框制+虚线｜仅首列标签｜
   平直直径边｜脚注 air-region 口径无 symlog｜全英文｜ink-gap ≥10px（同源）。

## 交付

审查报告＋**VERDICT: PASS / FAIL**（FAIL 附复现读数）。

## 白名单（写权限）

data/research/1_gain_ep
