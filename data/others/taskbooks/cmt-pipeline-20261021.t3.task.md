# 任务书：CMT 逆向设计管线（cmt-pipeline-20261021 / G2.G2a.t3，theory）

## 背景（自包含）

- **上游已交付（直接消费，勿重推）**：
  - G1：`src/pysci/research/gain_ep/theory/cmt_reflection_s_matrix.py` 通用化正问题——
    `MetagratingParams`（period/x 自由、校验 `MetagratingParameterError`）、
    `solve_s_matrix()`→`ScatteringMatrix`（S 为 |M|×|M|、`Port` 通道标注）、`sweep_s_matrix`。
  - G2a.t2：同模块快核——`CMTForwardKernel(params)`（`s_matrix`/`solve`/`s_matrix_many`→(B,P,P)）、
    `sweep_medium`（固定几何扫介质）、`solve_s_matrix_batch`（多参数集）、缓存层。
    实测基准（J=3/N=21/K=31，单核）：冷 0.83 ms、缓存命中 0.13 ms、批量 ~7000 次/s。
  - 交付留档：`orchestration/state/deliveries/theory-20261021-184701-*.md`（G1）、
    `theory-20261021-191326-*.md`（t2）。
- **用户裁决（2026-10-21）**：①极致性能——10+ 维高维参数空间探索不得成为瓶颈，速度越快
  越好（numba 等依赖授权，但 t2 实测纯 numpy 已达亚毫秒，为用而用不必）；②S 矩阵不恒
  2×2——目标 S 规格须支持一般通道维度。
- 物理基准文档：`data/research/1_gain_ep/theory/参考-透反射管槽结构的S矩阵理论计算/CMT计算与仿真参考.md`
  （B.2 节明示：以 S 矩阵的**约当块形式**为目标函数优化几何参数）。

## 目标

1. **目标 S 规格**（支持 |M|×|M| 一般维度）：
   - 约当块判据（本征值合并＋几何重数缺失，如 |λ₁−λ₂|、|tr²−4det| 等价形式——形式你定，
     与文档「约当块形式」语义一致并说明）；
   - 单向隐身/指定元置零、幅相加权最小二乘、任意目标矩阵匹配；
   - 约定清晰的目标函数接口（可组合、可加权）。
2. **优化器外壳**：差分进化/GA（scipy 或自研），**种群级多进程并行**；断点续算；
   结果导出（参数、目标值、S 矩阵、收敛史，落 `data/research/1_gain_ep/theory/`）。
3. **几何约束**：管槽总宽≤D、最小特征尺寸、槽不重叠（复用 `MetagratingParameterError` 校验
   语义，非法个体罚分或修复策略明确）。
4. **表层 API**：`inverse_design(...)`（三表层 API 之一，docstring 完整、易用）。

## 验收

- **回收测试**：从扰动初值/随机初值反演回收已知 EP 与 DP 构型（G1 默认参数），S 匹配容差内；
- **高维可行性演示**：≥10 维参数空间（如 J=3 时 h×3+w×3+x×2+cr/ci×若干）实际优化运行，
  分钟级出结果；
- **吞吐**：评估 ≥2000 次/s/核（基于 t2 快核缓存路径），并行再乘核数；
- `PYTHONPATH=src uv run --no-sync pytest tests/research/gain_ep -q` 全绿（新增用例含回收测试）；
  ruff format/check 过。

## 白名单（仅可写这些目录）

- `src/pysci/research/gain_ep`
- `tests/research/gain_ep`
- `scripts/research/gain_ep`
- `data/research/1_gain_ep/theory`（结果导出）

## 备注

- 符号与文献一致；用法写 docstring，不新建 README。
- 性能敏感：目标函数评估路径避免重复装配（用 t2 缓存/批量接口）。
- 拿不准先 blocked 问组长。
