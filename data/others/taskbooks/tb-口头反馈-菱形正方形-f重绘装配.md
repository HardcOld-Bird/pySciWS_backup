# 任务：场图×6＋大图重绘装配——正方形菱形/1.2m 数据/r=0.60（用户口头反馈修正）

## 背景

用户口头反馈（2026-10-20）：**菱形永远正方形**（边长 31cm、近边 25cm、远边 56cm）、
显示半圆 r≈0.60m、域 1.2m。sim 已交付 `simulation/field_8period_1p2m_ep3/`
（六组＋schema）；几何登记已更正（`scan_geometry`：正方形 0.31@0.25，矩形参数被拒）。

## 数据与几何（同源/只读）

- 数据：`field_8period_1p2m_ep3/`（六组 L/R＋F6 schema；逐文件严格 y>0 display max
  已走 `display_declaration`）；旧 1p8m/1p5m 数据与 r4/r5/r6 图**不得**作数据源；
- 显示半圆 **r=0.60m**，center=(0.28284,0)（schema 根 `display_semicircle`）；
- 内框：**正方形 0.31m**（四边等长）、近边 0.25m、远边 0.56m、±45° 两区、
  边平行/垂直径向——**从 `pysci.research.gain_ep.scan_geometry` 登记导入顶点**
  （同源纪律，禁自造）；远角 0.581m、弧边余量 1.9cm（schema 读数）；
- 外框：内框放大 1.23×＝**0.3813m 正方形**（同为正方形——菱形永远正方形），
  置于半圆外不遮挡处、两角虚线连接；完整入子图 bbox（必要时缩放显示，报读数）。

## f2 六幅场图（文件名含 r7）

1. vmax 严格 y>0 气域自适应：每组 vmax＝所服务两面板实测 max|Re p_t| 紧致取整
   （余量 ≤5%），`pysci-figures vmax-check`（y>0 口径、r=0.60）逐组比值∈[1.0,1.1]
   附交付；三组独立、线性、5 刻度、无 ± 浮动标注；
2. 两框制（内外框**均正方形**、虚线两角、内框包容留边）｜Exp.†/Sim. 框外且仅
   第一列｜直径边平直遮盖｜小标题 `L/R-incidence Re p_t`｜全英文｜零遮挡/不相交；
3. 读数表：各组 vmax/比值、内框四边等长断言（0.31）、内外框顶点、缩放读数。

## f3 大图装配（`beam_scatter_combined_r7.png`）

- R5 col-cb 紧邻版式沿用；雷达素材＝r4 `radar_r4_group*.png` **sha 断言零改动**；
- 色标量程＝f2 新 vmax、脚注 air-region (y>0) 口径更新；
- `pysci-figures ink-gap` 同源测距 ≥10px（禁自写测距）；行序/角标/上远下近/脚注
  禁回退同前例；r4/r5/r6 图保留对照。

## 白名单（写权限）

data/research/1_gain_ep

## 验收预期

内外框四边等长 0.31/0.3813（正方形断言）；vmax-check 逐组 PASS；ink-gap ≥10px；
雷达 sha 不变；禁回退逐项无回退。
