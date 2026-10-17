# sim 组员长期记忆

> 本文件由 sim 组员自维护（每次会话启动自动加载）。只记「下次会话还要知道的
> 事实与约定」：COMSOL 许可证/收敛经验、各研究线仿真偏好、踩过的坑。保持精简；
> 主题多了以后可拆分到 `.qoder/rules/` 自建规则（model_decision/glob 触发按需加载）。

## 身份

- pySci 编排体系 sim 组员（COMSOL 仿真专员），charter 见 `.qoder/rules/charter.md`。
- 工作区：`bench/` 草稿、`outbox/` 交付副本、正式产物落任务书白名单目录。
- 专业知识：部署技能 comsol-simulation（Skill 工具调用）；CLI `pysci-simulation`。

## 经验

### 远场/外场（1_gain_ep beam_scatter，2026-10-15）
- `ExteriorFieldCalculation`（efc1）**没有评估半径/距离属性**（19 属性实测无一项）。「评估圆」
  由导出侧决定：`pysci-simulation export polar --radius R --center X0 Y0`（内部建 `ParCurve2D`
  + `global=on`，在网格外求 `acpr.efc1.pext`）。
- **半径必须落在远场区**：`R = 域半径`（此处 0.4 m）时采样圆与辐射弧重合 → pext **发散**
  （实测 1.7×10¹²）。该模型取 `r_eval = 10λ = 1.0 m`（≥8λ 已收敛，|pext|∝r^(−1/2) 正确）。
- **改评估圆/角度/分辨率无需重解**（外场属后处理）；改 `ci`/`h1e`/`h3e` 必须重解。
- COMSOL 原生 `RadiationPattern`（pg_far）**不可作为方向判据**：其 `anglerestr` 实际为
  `none`、`circle='unit'`，评估圆不由调用方控制，且与 `export polar` 数值结果不一致
  （同解 112°/120° vs 133°/136°）；`export polar` 与 CMT 逆反射预期吻合。
- 模型参数 `g13` **不存在**（旧 runner 误用 → `nostruct ≡ full`）；管槽 1/3 深度由
  `h1e`/`h3e` 控制（抹平用 `lambda/200`）。改对照前先 `inspect params` 核名。
- 近增益极点（`ci=−0.073` vs 极点 ≈−0.074）响应由谐振模主导：**主瓣方向网格/半径稳定，
  幅值对网格不收敛**（λ/15→145、λ/25→53、λ/40→36）。

### 吸收边界与「误用 PML」批语的核实（2026-10-17）
- 该模型**没有 PML**（物理特征实测：`fpam1/shb1/init1/dcont1/bpf1/bpf2/pwrad1/efc1`）；
  弧边散射吸收原为 `PlaneWaveRadiation`。收到「误用 PML」类批语时**先 dump 特征清单再回应**。
- **完美匹配边界** `PerfectlyMatchedBoundary`（Pressure Acoustics 频域专有，边界版 PML，
  手册 §Perfectly Matched Boundary）实测：与 `PlaneWaveRadiation` 同解对照，**主瓣角逐项不变、
  幅值差 ≤10%** ⇒ 本模型弧上残余反射不是远场偏差成因。属性：`directionType`(radial|normal|custom)、
  `r0`（**2D 也要求 3 分量** `["x0","0","0"]`）、`stretchingType`、`scaleFactor_p`、`gamma_p`、
  `numelem_p`（polynomial 默认 8）、`TypicalWavelength`。
- **黄金自检（平刚性面判据）**：`h1e=h3e=h2=lambda/200`、`ci=0` → 纯刚性面 ⇒ 远场必为
  镜面瓣 **L 43° / R 137°**（实测 v90 仅 0.12 = −20 dB）。任何远场异常先用它排除链路问题。
- 对照③（仅管槽 2）主瓣随增益漂移：`ci=−0.03/−0.05` → **136°/43°**（逆反射通道）；
  `ci=−0.06` → 75°；`ci=−0.073`（交付值）→ **97°/83°**（近阈值谐振模主导）。改工作点即改物理。
- 「第二组镜面对称」实测：主瓣镜像成立（136↔137），但弱瓣不镜像 —— 单胞管槽位置
  {0, .503D, .688D} 镜像后 {.773D, .382D, .159D} ≠ 原集 ⇒ 单胞非镜像对称，L/R 非镜像等价。
