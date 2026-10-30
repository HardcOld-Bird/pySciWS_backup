# sim 组员长期记忆

> 自维护；只记稳定事实与约定。charter `.qoder/rules/charter.md`；技能 comsol-simulation；
> CLI `pysci-simulation`；`bench/` 草稿、`outbox/` 交付副本；正式产物落白名单。

## 仿真设置硬清单（开工前逐条核，依据进设置表）
- **背景/入射压力场只选空气域，不得覆盖散射体内部子域**（用户 2026-10-17）。
- 外场/吸收设置查手册留痕；接「误用 PML」批语先 dump 特征清单再回应。
- 两表义务：设置依据表 + 画前理论预期预登记（对称性/主瓣方向/量级），画后对照，
  不符则报疑，不得事后圆说。
- 交付前核收敛/网格质量；渲染 PNG 后 Read 图像判读。

## 通用 COMSOL 经验
- 评估圆由导出侧定：`pysci-simulation export polar --radius R --center X0 0`
  （ParCurve2D+`global=on`，网格外取 `acpr.efc1.pext`）；**R 必须落远场区**，取域半径会与
  弧重合致 pext 发散（实测 1.7e12）。`efc1` 须显式绑定弧边界；原生 `RadiationPattern`
  不可作方向判据。
- 网格 size tag 是 `size`（非 size1），改 hmax 先 `set("custom","on")`。载入已求解 mph 可
  直接后处理（改评估圆/角度/分辨率不重解；改材料/几何参数必须重解）；Java 导出 Image 前须
  `result(pg).set("data", ds)`，否则空白 PNG（≈10994 B）。
- PMB 属性：`directionType`(radial|normal|custom)、`r0`（**2D 也要 3 分量**）、
  `stretchingType`、`scaleFactor_p`、`gamma_p`、`numelem_p`、`TypicalWavelength`。

## Label/多组件纪律
- 序列标签（selection/material/physics/geometry/mesh）模型级唯一；**长标签会让材料内部
  变量求解后不可见**。一律 `geomN/meshN/matK/acprN/sairN`，描述写 label；`comp.geom("geomN")` 显式取。
- 求解后每组件一个数据集（`dsetN`，属性 `geom`），表达式在该数据集内**不加组件前缀**；
  一个 study 自动含全部物理 + 每个 PMB 派生的 `<phy>_pmb1_xdim` 组件。
- 结果侧：极坐标组类型 `PolarGroup`（非 PolarPlotGroup）；`Filter` 是值域过滤
  （expr+lowerexpr/upperexpr，指示函数可做几何裁剪）；1D LineGraph 对 ParCurve2D 绑角度须
  `xdata=expr`+`xdataexpr`+`xdataunit=deg`；PolarGroup 角度随采样点坐标旋转 180°，
  远场方向图一律用 1D 直角坐标。
- **acpr1 命名空间缺陷**：首物理场的 `acpr1.*` 在部分材料域未定义（连 omega 也无），
  换材料/选择/重跑均无效；未加前缀的 `p` 可用。绕行：解析入射补偿（`p+if(条件,解析入射,0)`
  形式）或直接牺牲带前缀场量。

## 仿真三原则（harnessing，用户 2026-10-27）
- 勤整理 / 最大化复用（一 mph、同组件优先）/ 人类终检友好（默认参数=最终绘图参数、mph 内可一次「计算」）；
  细则见技能 comsol-simulation `references/principles.md`。
