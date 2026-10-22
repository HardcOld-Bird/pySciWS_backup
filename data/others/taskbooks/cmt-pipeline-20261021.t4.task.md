# 任务书：COMSOL 参数化 mph 模板集（cmt-pipeline-20261021 / G2.G2b.t4，sim）

## 背景（自包含）

- **权威理论文档（必读附录 C「仿真设置」）**：
  `data/research/1_gain_ep/theory/参考-透反射管槽结构的S矩阵理论计算/CMT计算与仿真参考.md`
- **参考模型**：`data/research/1_gain_ep/simulation/refs/1 周期化-傅里叶变换法-上下对调.mph`
  ——Floquet 周期＋PML＋背景场，附录 C 无限周期口径（用户点名以此为参考）。
- **已验证标定（勿重推）**：`scripts/research/gain_ep/simulation/CMT-COMSOL单点验证.py`
  文档头记载了探针→S 元映射、无损锚点 4 位有效数字吻合、ci 符号约定（与 CMT 同号）、
  「勿切周期条件 active」等实测教训。
- **程序化建模先例**：`scripts/research/gain_ep/simulation/build_beam_scatter_mph.py`
  （jpype+mph 从零建模＋全局参数＋`run solve --set-param` 切换）。

## 目标

1. **参数化反射型管槽超表面验证 mph 模板**：3 管槽单模板起步，全局参数化
   D（周期）、θᵢ（入射角）、逐槽 w/h/x/c_r/c_i、频率。
2. **变 J 多模板路由**：J=2..6 拓扑不同需多模板（用户已认可多 mph＋智能路由），
   模板选择判据约定清楚（下一环节 t5 的智能路由直接消费）。
3. **阶次提取覆盖全部传播阶次集合**（用户裁决 2026-10-21）：探针＋空间 FFT 的提取
   不得硬编码 0/±1，须按设计点 (f₀, θᵢ, D) 的传播阶次集合提取。
4. 沿用既有探针标定与 ci 符号约定；槽内等效复声速 = 材料复声速 c₀(c_r+ic_i)。

## 交付

- mph 模板（放 `data/research/1_gain_ep/simulation/`，命名有序，与既有 0~11 编号风格协调）；
- 构建/批扫脚本（`scripts/research/gain_ep/simulation/`）；
- 每模板交付说明（仿照既有 `*_交付说明.md` 风格，含参数表与路由判据）。

## 验收

- 3 管槽模板参数化完备：D/θᵢ/逐槽参数可经 `run solve --set-param` 或等效方式设置；
- 无损锚点单点验证与 CMT 吻合（沿用 4 位有效数字口径，留对照数据）；
- 至少一个 **3 端口（或更多）设计点**的阶次提取用例验证提取面覆盖完整传播阶次集合；
- 批扫在**一次 JVM 会话**内完成（COMSOL license 纪律）。

## 白名单（仅可写这些目录）

- `data/research/1_gain_ep/simulation`
- `scripts/research/gain_ep/simulation`

## 备注

- COMSOL 手册纪律（外场计算/完美匹配边界等历史坑见参考模型与单点验证文档头）；
  本环节只做 mph 模板与提取层，`validate_s_matrix` 封装是下一环节 t5 的事。
- 拿不准先 blocked 问组长。
