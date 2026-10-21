# theory 组员长期记忆

> 本文件由 theory 组员自维护（每次会话启动自动加载）。只记「下次会话还要知道的
> 事实与约定」：符号推导/数值对拍经验、各研究线模型与符号约定、踩过的坑。保持精简；
> 主题多了以后可拆分到 `.qoder/rules/` 自建规则（model_decision/glob 触发按需加载）。

## 身份

- pySci 编排体系 theory 组员（理论计算专员），charter 见 `.qoder/rules/charter.md`。
- 工作区：`bench/` 草稿、`outbox/` 交付副本、正式产物落任务书白名单目录。
- 专业知识：部署技能 theoretical-computation（Skill 工具调用）；CLI `pysci-theory`。

## 经验

### gain_ep CMT 反射 S 矩阵（2026-10-21 通用化）

- 模块 `pysci.research.gain_ep.theory.cmt_reflection_s_matrix`：D/θᵢ/f0 已解耦为
  自由输入；`solve_s_matrix` 返回多端口 S＋端口语义（Port.order/kx/angle），
  `compute_s_matrix` 仍返回 ndarray（±45°、D=√2λ/2 与旧 2×2 逐元一致）；
  `sweep_s_matrix` 逐频返回列表（端口数随 f 变，不能堆叠成一个数组）。
- 精确互易锚点：自通道 r₀(θ)≡r₀(−θ)（任意非对称几何 ~1e-16），端口基下
  S₊θ[p,p]=S₋θ[n(p),n(p)]；2×2 即 S11≡S22，S12/S21 不受约束。
  非对角元的转置对称**不成立**（本端口基非标准多端口波导 S 矩阵）。
- 记号召回：文献 B.2 的 d_s 是槽深（本模块 `h`）；本模块 `d` 是硬壁间距/首槽偏移
  （B.1 记号）。槽位可用显式 `x`（优先）或由 `d` 递推，二者都给须一致。
- 全缺省才用 EP 默认（仅 J=3）；部分缺省抛 `MetagratingParameterError`。
- 数值坑：`c_complex` 全实时须 cast 为 complex，否则 β=sqrt(负实数) 出 NaN。

### gain_ep CMT 快核（G2，2026-10-21）

- 唯一物理路径：P3 块对角 → Schur 消元，主系统 (N+JK) 降到 N（N=2n_orders+1）。
  **U→−1 的共振模（|1+U|≤1e-10）不能消元**：须保留为 N+Ns 补充块并保留 P3 的
  微小对角元；硬置零会给出错误结果（实测相对误差 1e11）。
- 介质无关装配（P1/P2/I3/V1/V2，I3=conj(I2)、P1=P2[:,n_orders]）按
  (f0, 入射角, 几何) LRU 缓存；c/h 不入缓存键。批量接口：
  `sweep_medium`→(B,P,P)、`solve_s_matrix_batch`（种群/多参）、`sweep_s_matrix`（多频）。
- 实测（J=3, n_orders=10, k_modes=31）：冷启动 0.83 ms、缓存命中 0.13 ms
  （约 7.7k 次/s）；4 端口热 0.26 ms；介质扫描 ~0.13 ms/次。
  基准脚本：`scripts/research/gain_ep/theory/bench_cmt_kernel.py`。
- 缓存键含 x/w（构造后改几何自动重装配）；`params` 的 c/h 变化无需清缓存。
- 谐波闭式：槽内 α=kπ/w 使 s·w=kπ，积分核无需三角函数（`_overlap_harmonic`）；
  ±θ 成对端口可由 +θ 装配量反序+共轭导出（`_mirror_assembly`），EP 冷启动再减半。

### gain_ep 逆向设计（G2a.t3，2026-10-21）

- 模块 `pysci.research.gain_ep.theory.inverse_design`：`DesignSpace`（dim=5J−1 归一化
  基因型 h|w|x|cr|ci，解码修复槽宽/位置约束）、目标项 `MatrixMatch` / `ZeroElements` /
  `JordanBlock` + `CombinedObjective`、`jordan_structure` 谱诊断（代数 vs 几何重数）、
  自研 DE（rand1/best1）+ `ProcessPoolExecutor` + 检查点续算（定点可复现）、结果导出。
- Jordan 判据：2×2 的 `|tr²−4det|` 与 `|λ1−λ2|²` 恒等；本征值合并 ≠ 约当块，须用
  `jordan_structure` 的 defective 复核（代数重数 > 几何重数）。
- 回收：±1% 扰动初值，EP/DP 120 代内 max|ΔS|<1e-2；14 维 demo 250 代
  max|ΔS|≈2.6e-3，导出在 `data/research/1_gain_ep/theory/inverse_design/g2a_t3_ep/`。
- 吞吐：单核评估 ~3000 次/s（best-of-3，验收 ≥2000）；多进程小种群每代 IPC 调度
  开销明显，大批量并行才接近线性（受机器负载影响大，测吞吐前先看单核基准）。

### gain_ep CMT×COMSOL 验证（G2b.t5，2026-10-21）

- `theory.cmt_comsol_validation.validate_s_matrix(params)`：输入/输出与
  `compute_s_matrix` 同形（|M|×|M|、端口 kx 降序）；按有效槽数 J（前导 w/h>1µm）
  路由 J2–J6 模板；单 JVM 会话 + 全传播阶次 Fourier 提取；J∉2..6 / c0≠343 /
  非前导 sliver 明确抛 `MetagratingParameterError`。
- **提取约定（关键）**：扫描场 x 相位为 e^{−i kx x}；按**参考端口栅格**用
  e^{+i kx^m x} 积分直接得到端口阶次（勿用入射栅格——t4 的阶次翻转/3 端口 L 侧
  弱反射即源于此）。L/R 侧 = θ_q 正/负，模板 theta=|θ_q|，scan1/scan2。
- 幅度口径：无损锚点主系数 4 位、小量 ~2–4%；J4 max|Δ|≈9.1e-3；含损 ci=0.05
  ≈1.2e-2；3 端口 3×3 ≈1.0e-2。COMSOL 相位参考与 CMT 差固定相位，只比幅度。
- 测试无 COMSOL 自动 skip（模块 fixture）；对照数据
  `data/research/1_gain_ep/simulation/cmt_validation.json`；复跑脚本
  `scripts/research/gain_ep/theory/validate_s_matrix_cases.py`。
