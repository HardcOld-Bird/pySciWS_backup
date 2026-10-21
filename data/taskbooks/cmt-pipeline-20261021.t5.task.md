# 任务书：validate_s_matrix COMSOL 验证封装（cmt-pipeline-20261021 / G2.G2b.t5，theory）

## 背景（自包含）

- **上游已交付（直接消费，勿重做）**：
  - **t4（sim）mph 模板集**：`data/research/1_gain_ep/simulation/12 反射管槽_J{2,3,4,5,6}模板.mph`
    （D/θ/f、逐槽 w/h/x/c_r/c_i 全参数化；J3 为 18 域精细模板，J2/J4/J5/J6 为 30 域
    同拓扑 sliver 切换模板）＋构建脚本 `scripts/research/gain_ep/simulation/build_cmt_j3_template.py`、
    `build_cmt_J2toJ6_templates.py`、批扫/全阶次 Fourier 提取 `batch_cmt_orders.py`；
    **必读** `data/research/1_gain_ep/simulation/12 反射管槽模板集_交付说明.md`（参数表、
    路由判据、探针→S 映射、锚点对照、已知限制）；实测数据 `cmt_batch.json`。
  - **G1/t2/t3（theory）CMT 侧**：`src/pysci/research/gain_ep/theory/cmt_reflection_s_matrix.py`
    （`MetagratingParams`/`solve_s_matrix`/`compute_s_matrix`/`CMTForwardKernel`）与
    `inverse_design.py`（`inverse_design`）。S 语义：S 为 |M|×|M|、`Port` 通道标注、
    传播阶次集合由 (f₀, θᵢ, D) 相图判定。
- **用户需求（2026-10-21，原文语义）**：封装一个 python 函数，**输入参数形式与单组 CMT
  相同、输出格式也相同**，仅内部改用专用 mph 仿真计算；支持不同管槽数、不同周期长度、
  不同等效声速；单 mph 做不到的用多 mph 智能路由（t4 已备 J2–J6 模板）。
- 既有标定口径：`scripts/research/gain_ep/simulation/CMT-COMSOL单点验证.py` 文档头
  （探针列映射、ci 符号同号、勿切周期 active、无损锚点 4 位有效数字）。
- COMSOL license 纪律：批扫合并进单 JVM 会话。

## 目标

1. **`validate_s_matrix(params, ...)`**（三表层 API 之三）：
   - 输入：与 `compute_s_matrix` **完全相同**的 `MetagratingParams`（含 J、D、θᵢ、逐槽参数）；
   - 输出：与 `compute_s_matrix` **完全相同**的 S 语义（|M|×|M|、通道标注一致）；
   - 内部：按 J 智能路由 J2–J6 模板（J 超出模板集时给出明确错误或按需扩模板——策略写明），
     参数注入＋批扫共用一次 JVM 会话，全传播阶次 Fourier 提取（复用 `batch_cmt_orders.py` 逻辑）。
2. 批量接口（参数列表一次会话求解），供高价值结果的批量验证。
3. 用法写 docstring（何时用 compute_s_matrix / inverse_design / validate_s_matrix），
   不暴露 mph 细节。

## 验收

- 无损锚点对照 CMT（沿用 4 位有效数字口径，复现 `cmt_batch.json` 水平）；
- **至少一个 J≠3 用例**（如 J=4 或 5）做 CMT×COMSOL 交叉对照（闭合 t4 申报的「J>3 未对照」
  缺口）；至少一个非平凡（含损/复声速）用例对照，差异如实申报（FEM 与 CMT 本就有离散化偏差）；
- S 维度语义与 CMT 一致的用例（如 3 端口 D=1.5λ/θ=30° 复用 t4 用例）；
- `tests/research/gain_ep/` 新增用例：**无 COMSOL 环境时自动 skip**（CI 兼容），
  `PYTHONPATH=src uv run --no-sync pytest tests/research/gain_ep -q` 全绿；ruff 过。

## 白名单（仅可写这些目录）

- `src/pysci/research/gain_ep`
- `tests/research/gain_ep`
- `scripts/research/gain_ep`
- `data/research/1_gain_ep/simulation`（对照数据落盘）

## 备注

- 已知限制继承 t4 交付说明（3 端口 L 侧反射弱、小量 |S11| ~2% 偏差），对照口径如实；
- 拿不准先 blocked 问组长。
