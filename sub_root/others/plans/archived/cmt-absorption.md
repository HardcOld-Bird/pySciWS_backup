# CMT 管线吸收实施计划（状态：草案，待用户确认）

> 三段式第二段产物。输入：`community_research/2026-11-17/legacy-02-cmt-gain-ep.md`（解剖+用户裁决记录）。
> 完成后归档至 `archived/`。**实施前置：②域 O2（comsol_simulation 薄胶水）收工**，避免同包冲突。

## 1. 范围

**做**（☆可泛化三层）：EP n×n 判据与 CMT/逆设计方法论 → ④域 recipe；仿真验证对账契约 → ②域薄胶水；
逆设计协议 → ④域薄胶水（scipy DE 替自研 DE）。
**不做**（⭐研究线专用，留本地不入仓）：管槽物理/9D 坐标/gainEP mph 自动化/研究数据/研究脚本 4,155 行
与 COMSOL 产线脚本 9,116 行。旧 cmt-pipeline-api.md 的路径漂移（example→gain_ep）随重写修正。

## 2. 选型定稿（用户裁决）

| 层 | 去向 | 内容 |
|---|---|---|
| EP n×n 判别式 | ④域 recipe 文档 | F̃=(λ1−λ2)²/…=0 ∧ jordan_structure.defective（排除 diabolic）；2×2 判别式的 n×n 推广 |
| 仿真对账契约 | ②域薄胶水 ≈200-300 行 | "同入同出"契约 + compare_s_matrix（max_abs/rel/rms + 主导元绝对 \|Δ\|≤0.02 冻结判据）+ 模板态自检衔接 |
| 逆设计协议 | ④域薄胶水 ≈80-120 行 | 目标函数协议 + scipy.optimize.differential_evolution + callback 每代 checkpoint（自研 DE 不迁移） |

## 3. 目录规划（新增文件）

| 内容 | 位置 |
|---|---|
| ④ recipe 文档 | `.agents\skills\theory-calculation\references\recipe-cmt-ep-ndim.md`（n×n EP 判据）、`recipe-inverse-design.md`（逆设计协议+scipy DE checkpoint 用法） |
| ④ 薄胶水 | `sub_root\src\pysci\skills\theory_calculation\inverse_design.py`（目标函数协议 + scipy DE + checkpoint callback） |
| ② 薄胶水 | `sub_root\src\pysci\skills\comsol_simulation\reconcile.py`（compare_s_matrix + 同入同出校验；待 O2 收工后按其实际模块布局落位，或并入 validate.py——以 O2 结果为准） |
| 测试 | `tests\skills\theory_calculation\test_inverse_design.py`、`tests\skills\comsol_simulation\test_reconcile.py`（固定样例恒跑） |
| 运行产物 | `sub_root\data\research\<题目>\`（研究线，不入仓） |

## 4. 实施步骤

1. ④recipe 两篇（判据推导摘要+数值鲁棒性注记：defective 判定浮点敏感，jordan_structure 输入前需容差处理）；
2. ④ inverse_design.py（scipy DE + checkpoint callback + 目标函数协议；原地 ADR 记 scipy 替代自研 DE 的裁决）；
3. ② reconcile.py（对账契约；**等 O2 收工**，读其模块布局协调落位）；
4. 测试：逆设计 checkpoint 往返、compare_s_matrix 三指标+冻结判据固定样例；
5. 冒烟：合成 S 矩阵对账全链（真实研究数据不入仓，合成样例+冻结判据实证）。

## 5. 风险与约束

- ⭐研究线代码/数据**严禁入仓**（蓝图原则）；吸收物全部按新 API 重写，不平移旧代码；
- EP 判据浮点鲁棒性（defective 判定）需容差注记；
- 对账胶水与 O2 的 comsol_simulation 模块边界以 O2 实际布局为准（计划留协调余量）；
- scipy DE 与自研 DE 的行为差异（约束/收敛判据）在 recipe 中注记。

## 6. 验收标准

recipe 两篇齐备；inverse_design checkpoint 测试过；compare_s_matrix 冻结判据测试过；合成 S 矩阵
对账冒烟全通；`tests.cmd all` 全绿；⭐部分零入仓核查；归档。

## 7. 实施结果（2026-11-17 收尾记录）

- 验收标准**全部达成**：recipe 两篇（n×n EP 判别式+6 条浮点鲁棒注记；逆设计协议+scipy DE 裁决表）；
  inverse_design.py 202 行（checkpoint/续跑测试过）；reconcile.py（同入同出契约+compare_s_matrix
  冻结判据 |Δ|≤0.02 含等号，44 项测试含贴边双向钉死）；合成 S 矩阵双扰动冒烟 PASS/FAIL 实证与
  解析值吻合。
- ⭐零入仓核查通过（MetagratingParams/9D/管槽/gain_ep 等研究线标记在 src/tests/.agents 零命中）。
- 测试合计 469 passed 全绿（基线 425 + 对账 44）。遗留：冻结判据恰值 0.02 浮点不可达属口径事实，
  改动须用户裁决（当前冻结纪律不动）。
