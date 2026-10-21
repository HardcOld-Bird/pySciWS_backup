# 任务书：理论组员 harnessing 集成（cmt-pipeline-20261021 / G3.t6，devops）

## 背景（自包含）

- **用户指示（2026-10-21）**：CMT 管槽超表面工具的**表层接口加入理论计算组员的 harnessing**，
  供其直接在理论计算/设计/分析中使用，而**无需深入了解 mph 相关细节**。
- **三表层 API（已交付，勿改物理实现）**，均在 `src/pysci/research/gain_ep/theory/`：
  1. `cmt_reflection_s_matrix.py`：`compute_s_matrix(params)`（CMT 前向，亚毫秒）、
     `solve_s_matrix`（带 `Port` 通道标注）、`CMTForwardKernel`/`sweep_medium`/`solve_s_matrix_batch`（批量）、
     `MetagratingParams`（J/逐槽 w/h/x/c_complex、D/θᵢ/f₀ 自由）；
  2. `inverse_design.py`：`inverse_design(...)`（约当块/置零/矩阵匹配目标、DE 多进程、断点续算）；
  3. `cmt_comsol_validation.py`：`validate_s_matrix(params)`/`validate_s_matrix_batch`（COMSOL 验证，
     同入同出、J2–J6 模板智能路由）。
- 用法细节以三模块 docstring 为准（签名勿凭记忆写，须对照源码核验）。
- 理论组员 harnessing = `orchestration/pods/theory/` 下 charter/部署技能（≤8KB 预算）；
  编排体系文档见 `orchestration/README.md`。

## 目标

1. **理论组员 harnessing 集成**：charter 或部署技能加入三 API 速查——何时用哪个、
   一行调用示例、参数语义（S 为 |M|×|M|、传播阶次集合随 (f₀,θᵢ,D) 变化）、
   validate_s_matrix 需 COMSOL 环境（无则报错/skip）；**不暴露 mph 细节**。
   按 harness 预算（basic.md §1）：charter ≤8KB，超出细节移 references/。
2. **目录权属新政机械护栏**（用户裁决 2026-10-21）：pod-guard/白名单机制落实
   `data/用户的inbox/` 全员只读——**唯一例外**：各「大任务」任务目录下的 `outbox/`
   子目录为常设写授权。写权限面之外一律拒绝（含组长派单白名单越界时机械拦截或告警，
   按现有 pod-guard 架构最小改动实现）。

## 验收

- charter/技能与源码签名逐一核验（不得有记忆性错漏）；harness 预算合规（delivery-gate/lint 过）；
- pod-guard 护栏：对 `data/用户的inbox/` 的写入被拒、对其 outbox 子目录写入放行的实测证据
  （测试或手工复现命令＋输出）；
- 既有理论组员工作流不受破坏（charter 其余内容语义不变）。

## 白名单（仅可写这些目录）

- `orchestration`

## 备注

- 目录权属裁决原文见 `.qoder/rules/leader-only.md` §1（目录权属二分）；
- 拿不准先 blocked 问组长。
