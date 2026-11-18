# legacy-02：旧项目「CMT 理论计算 + 仿真验证管线」（gain_ep/theory）解剖（只读分析）

> 2026-11-17 · 分析对象 `D:\XXXIIIGGG\projects\pySci\pySciWS\src\pysci\research\gain_ep\theory` 及其接口（只读，未复制任何代码）
> 目的：为 pySciWS2 是否吸收该管线提供组织/实现/耦合事实底座。本文不含社区选型结论（另轮）。
> **特别约定**：按蓝图"研究线不入仓库"原则，全文用 ⭐ 标"研究线专用"、☆ 标"可泛化能力"，供用户裁决吸收边界。
> 注：旧 ④域文档 `orchestration/skills/theoretical-computation/references/cmt-pipeline-api.md` 以
> `pysci.research.example.theory` 指代本管线三模块，实际包路径为 `pysci.research.gain_ep.theory`
> （文档漂移/泛化命名，吸收时以实际代码为准）。

---

## 1. 资产定位总览

CMT 管线不是技能域，而是**研究线 gain_ep（管槽超表面增益 EP 研究）的 theory 子包**：
耦合模理论（Coupled Mode Theory）前向求解散射矩阵 → EP（exceptional point）拓扑/流形研究 →
差分进化逆向设计 → COMSOL 独立验证对账。全部资产（行数为精确计数）：

### 1.1 核心包 `src/pysci/research/gain_ep/theory/`（14 文件，8,975 行）

| 文件 | 行数 | 角色 | 属性 |
|---|---|---|---|
| `cmt_reflection_s_matrix.py` | 1,721 | **前向核**：反射型管槽超表面 CMT 模态匹配求解 → \|M\|×\|M| 散射矩阵；快核/批量/扫描 API | ☆ 数值核可泛化，物理模型 ⭐ |
| `inverse_design.py` | 977 | **逆设计**：目标 S 规格（MatrixMatch/ZeroElements/JordanBlock）+ 自研 DE + 断点续算 + 多进程 | ☆ 方法形态，几何约束 ⭐ |
| `sim.py` | 1,343 | COMSOL 仿真自动化：`GainEPSimulator`（标定/ground truth/反馈环模拟，gainEP_10in16out 专用） | ⭐ |
| `ep_manifold.py` | 734 | 9D EP 流形：带边行列式铅笔判别式、Jacobian、Newton 回投影 | ☆ 判据方法论，9D 坐标 ⭐ |
| `theory.py` | 730 | COMSOL 仿真函数（cr/ci 参数扫描、探针提取、pyvista 3D，gainEP_basic/10in16out 专用） | ⭐ |
| `ep_continuation.py` | 765 | EP 流形伪弧长延拓 + 黎曼度量随机游走（预测-校正、步长自适应） | ☆ |
| `ep_morphology.py` | 655 | 流形形态学：采样云、坐标极值、kmapper Mapper、ripser 持续同调 | ☆（已是社区件封装） |
| `cmt_comsol_validation.py` | 484 | **仿真验证对账**：mph 模板同入同出求 S + `compare_s_matrix` 幅度对账 | ☆ 对账方法论，模板管理经验 |
| `ep_lambda_map.py` | 385 | λ: M→S²（黎曼球）映射研究：环绕数/环路分类/纤维探查/双叶切片/τ 残差 | ☆ |
| `transfer_function_analysis.py` | 386 | 32 探针复声压传递关系分析（内嵌实测数据，一次性研究脚本） | ⭐ |
| `ep_pearls.py` | 318 | EP 珍珠：τ 对合配对、Puiseux 系数、响应指标（log max\|S\| 替代 Petermann K）、Newton 匹配求根 | ☆ |
| `ep_sigma.py` | 285 | σ 候选测试器：7 类参数变换实现 λ→−λ 的机械测试（含手征判据 {S,Γ}=0） | ☆ |
| `field_period.py` | 186 | 波场-结构周期 LCM 判据（周期化仿真的可行性门槛，用户裁决 2026-10-28） | ☆ LCM 通用，±45° 双通道约束 ⭐ |
| `__init__.py` | 6 | — | — |

### 1.2 测试 `tests/research/gain_ep/`（theory 相关 ≈2,848 行）

| 文件 | 行数 | 覆盖 |
|---|---|---|
| `theory/test_cmt_reflection_s_matrix.py` | 568 | 前向核回归：EP/DP 名义值、互易性（机器精度）、端口语义、快核=参考实现、批量 |
| `test_ep_manifold.py`（gain_ep 级） | 481 | 铅笔=谱形式恒等、延拓/游走守流形、τ 对合、珍珠、σ/手征、形态学封装 |
| `theory/test_inverse_design.py` | 367 | 编解码 roundtrip、间隙修复、目标函数、并行=串行、checkpoint 续算=直算、吞吐 ≥2000/s |
| `theory/test_cmt_comsol_validation.py` | 227 | COMSOL 真验证（无 license/模板**明示 skip**）+ 路由/六槽复位（免 COMSOL） |
| `theory/test_10in16out_function.py` + `test_gain_ep_functions.py` | 424 | **非 pytest**：模块级连 COMSOL 的手动脚本，conftest 默认拒收集 |
| `test_lambda_map.py`（gain_ep 级） | 149 | 环绕数/分类/球面分箱/纤维秩/切片分支点 |
| `theory/test_lcm_logic.py` | 143 | LCM 判据、±45° 模型族约束（validate 入口在 COMSOL 之前即拒绝界外） |
| `theory/test_gap_constraints.py` | 141 | 环回间隙 2mm 约束 |
| `theory/test_duality_mirror.py` | 104 | duality_mirror 对合/幺正不动点/谱映射 |
| `theory/test_batch_cmt_orders.py` | 101 | 阶次提取对照（importorskip mph） |
| `theory/test_cmt_template_snapshot.py` | 79 | 模板只读快照互斥（纯文件操作，防读半成品模板） |
| `test_batch_entry_blas_pin.py`（gain_ep 级） | 49 | BLAS 线程钉死先于重依赖导入 |

### 1.3 编排脚本 `scripts/research/gain_ep/`（研究产线，不属包）

| 目录 | 规模 | 角色 |
|---|---|---|
| `theory/`（18 脚本） | 4,155 行 | 研究编排：ep_lambda_scan、ep_manifold_{continuation,bridge,slices}、ep_{pearls,sigma,boundary_flip,shape_scan}_*、inverse_design_demo、增益半平面谱奇点探测、validate_s_matrix_cases、bench_cmt_kernel 等（argparse --phase 分段跑，产物落 data） |
| `simulation/`（34 脚本） | 9,116 行 | COMSOL 产线：ep_v2*/ep16_* 模板开发与修复、CMT-COMSOL 单点验证、batch_cmt_orders、ep_gap2mm_validate、ep_lcm_check、ep_manifold_fem、export_ci_scan 等（大量消费 ②域 comsol_simulation.tools） |

### 1.4 数据/文档/配置（`data/research/1_gain_ep/`）

| 路径 | 规模 | 角色 |
|---|---|---|
| `theory/`（8 个研究会话） | 280 文件 / 10.3 MB | 产物：results(npz/json)、plots(png)、notes.md、report.md、figures/（fig* 目录带 STYLE.yaml+ANNOTATIONS.yaml ③域图契约+图注事实核对表）；会话含 cmt_reflection_s_matrix、ep_manifold、inverse_design{ep_nonzero_eig,gap2mm}、ci_flip_duality、lcm_logic、order_convergence、gain_ep_ss_probe |
| `theory/参考-透反射管槽结构的S矩阵理论计算/` | 权威物理文档 | **CMT计算与仿真参考.md**（+PDF）：附录 A 衍射模式分布图、附录 B CMT 推导（B.1 平面波模、B.2 矩形管槽模态匹配 B.2.5–B.2.11）；配套 MATLAB 参考代码 reflect_cmt.m/transmit_cmt.m/N1jifen.m/M22jifen.m/fenbujifen.m、黎曼面绘图 liman.m |
| `simulation/管槽结构CMT验证/` | 模板资产 | **反射管槽_J参数化模板.mph**（13 MB，J=2..6 五合一，本地资产）+ 12 反射管槽模板集_交付说明.md + cmt_batch*.json 对账结果 + 5 个模板改造工作记录目录 |
| `simulation/old_mphs|refs/`、`experiment/storage/`、`汇报PPT/` | ⭐ | 旧 mph 模型/实验标定 pkl/演示 PPT（与 theory 仅 mph 文件名级关联） |
| 测试夹具 | 2 json | `tests/research/gain_ep/simulation/fixtures/field_schema_*.json`（属 simulation 包，非 theory） |

**实现规模合计：包 8,975 行 + 测试 ≈2,848 行 + 研究脚本 4,155 行（另有 COMSOL 产线脚本 9,116 行）≈ 16,000 行核心面**；
数据 280 文件 10.3 MB + mph 模板 13 MB。

---

## 2. 组织方式

### 2.1 架构位置与交互面

```
旧架构（pySciWS）：研究线 consume 技能域
research/gain_ep（研究线：管槽超表面增益 EP）
├─ theory/          ← 本管线（CMT 前向核 + EP 研究 + 逆设计 + COMSOL 验证）
├─ experiment/      ← 硬件测量（nidaqmx/步进电机/标定，与 theory 零代码耦合）⭐
├─ simulation/      ← 包内 COMSOL 场工具（farfield/field_schema/semicircle，偏实验扫描几何）⭐
├─ article/figures/ ← 论文图（ep_eigval_3d/ep_phase_wind/ep_riemann_3d，scipy.optimize.root）⭐
└─ plotting.py      ← setup_chinese_fonts（研究线字体配置）

theory 消费的技能域/基础设施（全部为"调用面"耦合）：
├─ ④ theoretical_computation.tools：visualize.setup_style/export_exploration（main() 探索图）、
│    session.ensure_session（产物落 results/plots/notes.md）
├─ ② comsol_simulation.tools：export.export_data（扫描线导出）、scan_window（仿真脚本层）
├─ ③ scientific_plotting.tools：raster/runner/style（仅仿真脚本层出图）
├─ pysci.paths：research_asset_dir/research_theory_dir/DATA_ROOT/assert_within_data（越界护栏）
└─ 第三方：numpy / scipy(block_diag×1) / matplotlib / mph(COMSOL) / pyvista / kmapper / ripser
```

- **研究线而非能力域**：无 CLI 命令入口（无 `pysci-research xxx`），运行方式 = `PYTHONPATH=src uv run --no-sync python scripts/research/gain_ep/theory/<study>.py --phase ...`（研究脚本 argparse 分段）。
- ④域已为它写过调用面文档 `cmt-pipeline-api.md`（83 行，"何时用哪个/关键语义/一行示例/分工"）——即**旧架构里它已被视为半技能化的表层**，物理实现仍钉在研究线。
- 与 gain_ep 实验数据**代码级零耦合**（theory 不 import experiment 任何模块；sim.py 特意绕开 experiment 的 nidaqmx 依赖链）；**数据级弱耦合**：sim.py/theory.py 消费 `gainEP_10in16out*.mph` 等仿真资产（在 data 的 experiment|simulation 树下），transfer_function_analysis 内嵌探针读数。

### 2.2 工作流（输入 → 理论计算 → 仿真验证 → 输出）

```
输入：MetagratingParams(f0, c0, θi, J, h[]/w[]/x[] 或 d[], c_complex[]（虚部=损耗/增益）,
      n_orders, k_modes, period)   —— 或 make_ep_params()/make_dp_params()（文献 EP/DP 构型）
  │
  ▼ ① CMT 前向求解（cmt_reflection_s_matrix，亚毫秒级，无 COMSOL）
  │    模态匹配线性方程组 [(-P2,P3);(-V2,V3)]·(A⁻;A)=(P1;V1)·A0
  │    → 反射系数 A⁻ → |M|×|M| S 矩阵（M=传播 Floquet 阶集合，端口按 kx 降序）
  │    快核：Schur 消元降维 + 装配量 LRU 缓存 + 镜像复用；批量 s_matrix_many/sweep_medium
  │
  ├─▶ ② EP 研究（ep_*，纯数值）：EP 判据定位 → 9D 流形采样/延拓/游走
  │    → λ:M→S² 映射/环绕数 → τ 配对珍珠 → σ 对称搜索 → Mapper/TDA 形态学
  │    （编排在 scripts/.../theory/*.py，按 --phase 分段，中间缓存落 pod bench/）
  │
  ├─▶ ③ 逆向设计（inverse_design）：目标 S 规格 → DE 多进程优化（checkpoint 每代落盘）
  │    → InverseResult(params/S/history) + summary.json
  │
  ▼ ④ 仿真验证对账（cmt_comsol_validation，分钟级，需 COMSOL license）
       同一 MetagratingParams 注入 mph 模板（J 路由→参数注入→几何/网格重建
       →模板态自检→网格兜底→逐端口求解→全传播阶 Fourier 提取）→ 同语义 S 矩阵
       → compare_s_matrix 幅度对账（max_abs / max_rel / dominant_rel / rms）
       → 判据：主导元绝对 |Δ|≤0.02（冻结）+ A/B 相对判据 + 互易元/功率和读数

输出（一律落 data/research/1_gain_ep/theory/<slug>/，由 ensure_session/研究脚本管理）：
  results/*.npz|json（S 矩阵、流形点云、延拓路径、对账读数）
  plots/*.png（探索图）、figures/fig*/（论文图：STYLE.yaml+ANNOTATIONS.yaml ③域契约）
  notes.md（计算日志）、report.md（研究报告：判据口径/精度现状/事故注记）
```

### 2.3 与旧 ④理论计算域、旧 COMSOL 仿真的关系

| 邻域 | 关系 | 证据 |
|---|---|---|
| ④ cas/eigen/numerical/topology | **几乎不复用**：CMT/EP 全链纯数值（numpy 线性代数），无 sympy；④域的符号判别式（cas.discriminant_2x2）只出现在 recipe 文档，未被本管线调用 | theory 包 import 清单 |
| ④ visualize/session | **薄复用**：探索图导出 + 会话目录约定（results/plots/notes.md） | `ensure_session("gain_ep","cmt_reflection_s_matrix")` |
| ② COMSOL 技能 | **薄复用**：export_data 扫描线导出；模板细节自管（16 号合并模板、域号断言、网格兜底均为研究线私有代码） | cmt_comsol_validation._export_scan |
| 旧 COMSOL 仿真（mph 模型） | **强依赖资产**：验证=专用 mph 模板（13MB 本地资产）；sim.py/theory.py=独立的 gainEP mph 自动化（不走 ②域） | data/.../管槽结构CMT验证/ |
| MATLAB 参考代码 | **前身**：CMT 计算与仿真参考.md 配套 .m 文件（N1jifen/fenbujifen/M22jifen 积分核逐个对应 Python 私有函数） | 参考目录 |

---

## 3. 实现方式（逐模块）

### 3.1 cmt_reflection_s_matrix.py（1,721 行）——CMT 建模/求解 ☆

| 段 | 职责 | 关键技术 |
|---|---|---|
| 解析积分核 | 槽内波导模重叠积分 ∫e^{tx}cos(s(x−a))dx 等三件 | **解析闭式**（三分支：t=0/共振/一般式），向量化 `_overlap_harmonic`（指数按槽预计算，N·J 次而非 N·J·K 次）；非 sympy，纯手推公式落码 |
| 装配 | P1/P2/V1/V2/V3 积分核 + Floquet 横向波数 kx^m=k sinθi+mG | 与介质无关量按 (f0,θi,几何) **LRU 缓存**（`_cached_assembly`）；介质变化只重算 β/U 与一次求解 |
| 求解 | 模态匹配线性方程组 → 反射系数 | **Schur 消元**把 (N+J·K) 主系统降到 N（槽内幅值块对角消元）；U→−1 共振模留补充块（`_SCHUR_TOL` 防 1/(1+U) 爆炸）；求解用 numpy 复数稠密解（scipy 仅 block_diag×1） |
| 端口/散射矩阵 | `Port`/`ScatteringMatrix`：传播阶集合 M=|kx^m|<k，S[p,q] 语义、`describe()` 逐元通道注释 | 声锥判据留 `_CONE_TOL` 排掠射奇异模；2×2 特例与旧接口逐元一致 |
| 互易性 | Lorentz 互易恒等式 r₀(θ)≡r₀(−θ)（含损耗/增益成立） | 测试钉到机器精度；非对角元无转置对称（文档明确写边界） |
| 快核/批量 | `CMTForwardKernel`、s_matrix_many、sweep_medium、solve_s_matrix_batch、sweep_s_matrix | **固定几何扫介质/多参数集**复用装配——"前向核复用"形态 |
| EP/DP 构型 | `make_ep_params`(cr=1.0081, ci=0.0745)、`make_dp_params` | 文献损耗 EP/DP 构型硬编码 ⭐ |
| main() | 收敛性检查（截断阶对比）→ (cr,ci) 网格定位 EP → 参数扫描 → 黎曼面 3D 图 | 走 ④域 session/visualize |

### 3.2 EP 物理判据实现（ep_manifold.py 等）☆ 判据 / ⭐ 坐标

与 ④域 recipe 的关系（用户问点）：

| | ④域 recipe（`ep_bic_hamiltonian.md`） | 本管线 |
|---|---|---|
| 对象 | 2×2 非厄米哈密顿量 H | 任意维散射矩阵 S（\|M\|×\|M\|）+ 9D 参数流形 |
| 判据 | 判别式 Λ=0（cas.discriminant_2x2，符号） | **谱版判别式** F̃=(λ₁−λ₂)²/(1+\|λ₁λ₂\|²+\|λ₁+λ₂\|²)=0（数值、归一化、一阶稳定）+ 带边行列式铅笔式 `bordered_coefficients`（slogdet 缩放）对拍 |
| 排假 | 无（2×2 下 EP=判别式过零） | **∧ `jordan_structure(S).defective`**（代数重数>几何重数才真 EP，剔除可对角化的 diabolic 点）——`jordan_structure` 用本征值聚类+秩诊断 |
| 扩展 | — | Puiseux 系数（EP 强度）、projective gap、τ 对合（ci 全反号 → S(z)≅S(τz)†⁻¹、λ(τz)=1/λ(z)*）、手征判据 {S,Γ}=0（Beenakker 1997）、响应指标 log max\|S\| 替代发散的 Petermann K |

→ **可泛化结论**："EP 判据 = 判别式过零 ∧ defective（几何<代数重数）"是 ④recipe 2×2 判据的 n×n 推广，值得写回 ④域 recipe；9D 坐标系/文献参数/τ 物理是研究线专用 ⭐。

### 3.3 仿真验证环节（cmt_comsol_validation.py，484 行）——mph（COMSOL）+ 幅度对账 ☆

- **技术**：COMSOL 官方 Java API 经 **mph**（社区 Python 客户端）；不裸写 mph 细节——"同入同出"契约：`validate_s_matrix(params)` 与 `compute_s_matrix(params)` 同参数、同端口语义、同形状。
- **流程**：`_active_grooves` 路由（w,h>1µm 前导槽计数 J∈{2..6}）→ `_enforce_model_family`（±45° 双通道+LCM，COMSOL 之前先拒）→ **只读快照** `_stable_snapshot`（size+mtime 稳定采样+复制窗口复核，防并行改模板时读半成品——2026-10-28 事故注记）→ 参数注入（6 槽写满，空槽复位 sliver）→ 几何重建 → **模板态自检** `_assert_template_state`（PML 顶带/BPF/Floquet 3 边对/扫描线存在性断言，域号不写死因 COMSOL 自动重映射）→ **网格兜底** `_run_mesh_fallback`（映射网格失败逐级降档：关 dis1→关 map1 加密→关 id1-4）→ 逐端口求解（bpf1/bpf2 切左右入射）→ 扫描线场导出（②域 export_data）→ Fourier 提取 S[p,q]=∫p_s e^{+ikx^p x}dx/(窗宽)。
- **对账**：`compare_s_matrix` 纯函数——**幅度口径**（COMSOL 相位参考与 CMT 差固定相位，验收只比幅度）：max_abs_diff / max_rel_diff（\|CMT\|>1e-3）/ dominant_rel_diff（>0.2）/ rms_diff + 逐元幅度表。**判据**：主导元绝对 |Δ|≤0.02（冻结口径，A/B gap2mm 相对判据 PASS 兜底）；已知系统偏差锚点 0.0388（金标准 0.0223）、互易元 0.9830、功率和 0.968——精度现状（D1 未闭合）如实记录在 docstring。
- **限制**：模板仅 J=2..6；c0 须 343 m/s；端点离散化 ~0.39% 抖动未修。

### 3.4 inverse_design.py（977 行）☆ 形态 / ⭐ 约束

- 目标函数协议 `term.evaluate(S, ports) -> float`：MatrixMatch（幅/相位加权，相位用 1−cosΔφ 无 2π 跳变）/ZeroElements/JordanBlock/CombinedObjective。
- `DesignSpace`：基因型 [0,1]^(5J−1)，**修复式解码**（槽宽压缩/逐槽夹紧保证间隙含环回 ≥min_gap=2mm ⭐ 用户裁决绝对物理量）；非法个体罚分忽略。
- 优化器：**自研 DE/rand/1/bin** + ProcessPoolExecutor（Windows spawn 安全）+ **每代 checkpoint**（种群/适应度/RNG 状态，`resume=True` 精确续算）；`jordan_structure` 诊断复核解。
- 测试锁行为：并行=串行、checkpoint 续算=直算、吞吐 ≥2000 评值/秒。

### 3.5 其余模块速记

| 模块 | 要点 |
|---|---|
| ep_continuation | 伪弧长延拓（切空间预测→Newton 最小范数回投，残差阈 1e-9，步长自适应 ×1.5/×0.5）+ ci2 延拓/骨架/homotopy bridge（跨无损面） |
| ep_lambda_map | 环绕数（主值角增量取整）、环路 4 类判据（口径冻结写死）、黎曼球等面积分箱、纤维维数（临界秩检测）、双叶切片分支点追踪 |
| ep_morphology | 切向游走采样云（冻结 Jacobian 的 chord-Newton 投影）、多进程分片采样、坐标极值（投影梯度上升）、kmapper/ripser 封装（懒加载社区件，含噪声标定阈值、cocycle 圈回溯） |
| ep_pearls/ep_sigma | τ 配对 Newton 求根（F̃=0 + 指标匹配）、7 类 σ 候选机械测试、频率对极 |
| field_period | 波场周期 λ_x=λ/\|sinθ\| 与结构周期 D 的 **LCM=ρ 有理化判据**（ρ=a/b、LCM=a·D≤3D 否则"周期化不可行/过复杂"）——周期化 FEM 的通用前置检查 |
| sim.py / theory.py ⭐ | gainEP mph 自动化：SimScanner 类比（传递函数/增益系数标定 pkl 缓存、ground truth 迭代求解、反馈环复平面 PNG）、cr/ci 扫描 S 提取、pyvista 3D；存储走 research_asset_dir 平行资产树 |
| transfer_function_analysis ⭐ | 32 探针复声压内嵌数据 → 传递关系探索（matplotlib Agg + ④域 setup_style），"迁移自 scripts/.../传递函数分析.py"的一次性研究件 |

### 3.6 第三方依赖清单

| 依赖 | 用途 | 备注 |
|---|---|---|
| numpy | 前向核/EP/逆设计全链 | 唯一硬依赖（core） |
| scipy | block_diag×1；脚本层 curve_fit/root | 可替换为 np.block |
| matplotlib | 探索图/研究图 | main()/transfer 分析 |
| **mph** | COMSOL Java API 桥 | cmt_comsol_validation/sim/theory.py；需 COMSOL license |
| pyvista | theory.py 3D | ⭐ |
| kmapper + ripser | Mapper/持续同调 | 懒加载，ep_morphology 封装 |
| pysci.paths / ④tools.visualize+session / ②tools.export / ③tools | 基础设施 | 见 §4 |

---

## 4. 耦合评估

### 4.1 纯逻辑（可平移，自包含）☆

| 模块段 | 行数量级 | 依赖 |
|---|---|---|
| CMT 前向核（积分核/装配/Schur/批量/镜像/铅笔式） | ~1,400 | numpy（scipy×1 可换） |
| EP 判据与谱诊断（F̃、jordan_structure、bordered_coefficients、duality_mirror、compare_s_matrix） | ~400 | numpy |
| ep_continuation/lambda_map/pearls/sigma（游走/延拓/环绕数/τ/σ） | ~1,700 | numpy |
| ep_morphology（含 kmapper/ripser 封装） | ~650 | numpy + 社区件 |
| inverse_design 目标函数/DE/checkpoint | ~900 | numpy + stdlib（进程池） |
| field_period LCM | ~180 | numpy + fractions |

合计 ≈5,200 行纯数值逻辑可整段平移；**无 sympy**（符号推导在参考文档层，代码是数值闭式）。

### 4.2 与旧基础设施耦合（迁移需重接/替换）

| 耦合点 | 涉及 | 迁移动作 |
|---|---|---|
| `pysci.paths`（research_asset_dir/research_theory_dir/assert_within_data） | cmt_comsol_validation、inverse_design、sim、transfer、全部研究脚本 | 换新路径层（薄） |
| ④域 `visualize/session` | cmt main()、transfer 分析 | 新④域等价物（setup_style/export/会话约定），薄 |
| ②域 `comsol_simulation.tools.export/scan_window` | _export_scan、仿真脚本层 | **恰是②域薄 Python API 的用武之地**（domain-02 已定形态） |
| mph 模板资产（13MB 本地、gitignored）+ 模板态/网格兜底私有逻辑 | cmt_comsol_validation | 资产随研究线保留；模板管理经验（合并模板/域号断言/快照互斥）文档化 |
| ③域 scientific_plotting tools | 仅仿真脚本层出图 | 归③域，不属本管线 |
| CLIs/运行方式 | 无正式 CLI，uv run 直跑脚本 | 不需迁移 |

### 4.3 与 gain_ep 实验数据耦合度

**代码级：零**（theory 不 import experiment；sim.py 显式避开 nidaqmx 依赖链）。
**数据级：弱**——sim.py/theory.py 消费 gainEP_10in16out*.mph 仿真资产（⭐）、transfer 分析内嵌探针读数（⭐）；标定 pkl/plots 属 experiment 树。
**物理级：强相关但接口干净**——理论 S 矩阵 ↔ 实验增益/反馈研究共用"gainEP 管槽超表面"物理对象，但仅通过数据/文档（CMT计算与仿真参考.md）相连。

### 4.4 复杂度热点（事故注记/测试钉死）

1. **模板写读互斥**（`_stable_snapshot`）：并行改造模板时 FEM 读数漂 3 倍致假红（2026-10-28）→ 稳定性采样+只读快照；
2. **同会话 J 回退六槽复位**：sliver 槽残留使含损用例 rms 0.0118→0.052（测试专钉）；
3. **Schur 消元的 U→−1 共振**：1/(1+U) 数值爆炸→补充块保留；
4. **COMSOL 相位参考差**：固定相位→验收只比幅度（口径冻结进 docstring"勿改"）；
5. **端点离散化 ~0.39% 抖动**（D3 未修）、**精度 D1 未闭合**（锚点 0.0388>0.02）——如实写档不粉饰。

**总体判断**：这是一条"**数值闭式核 + 冻结判据口径 + 对账闸门**"的研究线管线。真正值钱的不是管槽物理本身，
而是四件可泛化资产：①CMT 前向核工程形态（Schur 快核/装配缓存/批量复用）；②EP 判据升级（判别式∧defective，
n×n 版 recipe）；③理论↔仿真对账契约（同入同出+幅度口径+绝对判据+模板态自检+快照互斥）；④逆设计目标协议
与断点续算形态。管槽物理/9D 坐标/gainEP mph/研究数据是研究线本体。

---

## 5. 吸收启示（按"技能文档+社区工具+薄胶水"拆层，点到为止）

| 层 | 旧实现 | 吸收方向 | 衔接 |
|---|---|---|---|
| **④域 recipe（纯文档，☆）** | CMT 模态匹配/Schur 快核/装配缓存/批量核复用；EP 判据（F̃=0∧defective）；逆设计目标协议+DE+checkpoint；LCM 周期化判据；判据口径冻结纪律 | 写入 `data/skills/theoretical_computation/recipes/`：升级现有 `ep_bic_hamiltonian.md`（2×2→n×n 判据）+ 新增 cmt-s-matrix/verification-recipe；旧 `cmt-pipeline-api.md` 83 行调用面即雏形 | **已建成的④理论计算技能**直接挂 recipe；判据代码模式 ≤50 行/篇 |
| **②域薄胶水（仿真验证对账，☆）** | validate/compare"同入同出"契约、幅度对账四读数、模板态自检、网格兜底、快照互斥 | ②COMSOL 技能暴露薄 Python API（函数签名级），把"对账契约+模板管理经验"写成②域技能文档/薄胶水；实现归实施期/研究树 | **正在实施的②COMSOL 技能**（domain-02 已定"④域调用②域薄 API，实现归实施期/研究树"，条目 15）——本解剖即该形态的实证 |
| **研究线专用（⭐，不入仓）** | MetagratingParams 管槽物理、make_ep/DP_params、9D EP 流形全家的坐标/物理设定、DesignSpace 2mm 约束、sim.py/theory.py/transfer 分析、gainEP mph 模板、研究数据 280 文件 | 按蓝图留在本地研究树；吸收的是其**方法论注记**（事故注记/口径纪律/精度现状写法） | 与"研究线不入仓库"一致；与旧 domain-04 结论"#20 CMT 管线不保留"的关系交用户裁决 |
| **社区工具替代空间（点到为止）** | 自研 DE/rand/1/bin + ProcessPoolExecutor；kmapper/ripser/mph 已是社区件 | DE 可看 scipy.optimize.differential_evolution（④域已收 SciPy，可能直接删自研）；TDA/COMSOL 桥维持社区件 | 不新增 ④域核心依赖；唯 mph 归②域外部工具形态 |

**一句话结论**：旧件是研究线 gain_ep 下"CMT 数值闭式前向核（8,975 行包，纯 numpy 级）+ EP 流形/逆设计 + COMSOL
同入同出对账"的完整管线（含 2,848 行测试、4,155 行研究脚本、10MB 产物与完整口径/事故注记文档）；纯数值逻辑
≈5,200 行可平移，但按蓝图应**拆三层吸收**：判据/方法论→④域 recipe（升级 EP 判据为 n×n∧defective）、
验证对账契约→②域薄胶水/API 形态、管槽物理与研究数据→研究线本地保留；其中自研 DE 在吸收时优先对照
scipy.optimize 消减。

## 用户裁决记录（2026-11-17）
- **吸收边界=⭐/☆ 拆分方案**：⭐研究线专用（管槽物理/9D 坐标/gainEP mph 自动化/研究数据）留本地不入仓；☆可泛化三层吸收。
- **自研 DE 不迁移**，换 scipy.optimize.differential_evolution + ~30 行 callback 实现每代 checkpoint。
- **三层落位认可**：EP n×n 判别式→④域 recipe；对账契约（同入同出+compare_s_matrix+主导元 |Δ|≤0.02 冻结判据）→②域薄胶水（待 O2 实施完落地）；逆设计协议→④域。
