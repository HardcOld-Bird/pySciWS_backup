# CMT 管槽超表面管线 API（gain_ep 表层）

三模块均在 `pysci.research.gain_ep.theory`。本册只写**调用面**；物理实现与边界条件以源码
docstring 为准（签名不得凭记忆写）。不涉及 mph/COMSOL 建模细节。

## 何时用哪个

| 需求 | 入口 | 代价 / 前提 |
|---|---|---|
| 快速反射 S 矩阵（前向） | `compute_s_matrix(p)` / `solve_s_matrix(p)` | 亚毫秒级纯 Python；无需 COMSOL |
| 固定几何只扫介质 | `sweep_medium(p, c_complex_batch)` | 复用前向核，返回堆叠数组 |
| 多参数集 / 多频 | `solve_s_matrix_batch(ps)` / `CMTForwardKernel` | 同几何复用核；端口数可不同 |
| 逆向设计 | `inverse_design(space, objective, ...)` | DE 多进程；可 checkpoint/resume |
| COMSOL 验证 | `validate_s_matrix(p)` / `validate_s_matrix_batch(ps)` | **需 COMSOL/mph 环境**；无则报错或 skip |

## 关键语义

- `MetagratingParams(f0, c0, theta_i, J, h, w, d|x, c_complex, n_orders, k_modes, period)`：
  几何数组要么全部省略（EP 名义参数，J=3），要么全部给出；`c_complex` 虚部＝损耗/增益；
  `d` 为硬壁间距、`x` 为槽起点（二选一）；θi≈0 时必须显式给 `period`。
- `ScatteringMatrix`：`S[p,q]`＝端口 `ports[q]` 入射、`ports[p]` 出射；端口按 kx 降序；
  **S 为 |M|×|M|，M 随 (f0, θi, D) 变化，不恒 2×2**；`describe()` 打印逐元通道语义。
- `DesignSpace(J, f0, theta_i, period, h/w/cr/ci 上下界, min_feature, x0)`：基因型
  `[0,1]^(5J−1)=h[J]|w[J]|x[1:J]|cr[J]|ci[J]`，解码自动修复槽宽/不重叠/尾壁约束。
- 目标：`MatrixMatch(target, complex/amplitude/phase_weight)`、`ZeroElements(elements)`、
  `JordanBlock(...)` → `CombinedObjective((...))`；签名 `objective(S, ports) -> float`。
- `InverseResult`：`params/genome/objective_value/S/ports/history/generations/evaluations`，
  `summary()` 给 JSON 友好读数；`workers=0` 用满核，`checkpoint_path`+`resume=True` 续算。
- `validate_s_matrix`：与 `compute_s_matrix` **同入同出**（同端口顺序/S 语义）；要求
  c0=343 m/s、有效槽 J∈{2..6}（J2–J6 模板自动路由）；无 COMSOL 环境时抛错，调用方按
  任务约定 skip 或上报。

## 一行示例

```python
from pysci.research.gain_ep.theory.cmt_reflection_s_matrix import (
    MetagratingParams,
    solve_s_matrix,
    sweep_medium,
)

p = MetagratingParams()  # 省略几何 = EP 名义参数（J=3）
sm = solve_s_matrix(p)  # ScatteringMatrix（含 Port 语义）
print(sm.n_ports, sm.port_orders)
print(sm.describe())

import numpy as np

stack = sweep_medium(p, np.array([p.c_complex, p.c_complex * (1 - 0.01j)]))
print(stack.shape)  # (2, |M|, |M|)
```

```python
from pysci.research.gain_ep.theory.inverse_design import (
    DesignSpace,
    MatrixMatch,
    CombinedObjective,
    inverse_design,
)

import numpy as np  # noqa: E402 - 示例段独立

space = DesignSpace()  # J=3 / f0 / theta_i / period 默认
target = np.eye(2, dtype=complex)  # 默认 EP 参数 2x2；按目标 S 形状替换
objective = CombinedObjective((MatrixMatch(target=target, weight=1.0),))
res = inverse_design(space, objective, popsize=32, max_generations=50, workers=0)
print(res.summary())
```

```python
from pysci.research.gain_ep.theory.cmt_comsol_validation import validate_s_matrix

try:
    S = validate_s_matrix(p)  # 需 COMSOL/mph；无环境按任务约定 skip
except Exception as exc:  # noqa: BLE001 - 无 COMSOL 时如实上报
    print(f"COMSOL 验证不可用：{type(exc).__name__}: {exc}")
```

## 与其他技能的分工

- 前向/逆向在理论侧直接跑，**不需要**打开 COMSOL；只有 `validate_s_matrix*` 需要。
- 与 `scientific_plotting`：把 `InverseResult`/S 矩阵存 npz 后按图管线出图。
- 与 `comsol_simulation`：验证只比对**同入同出**的 S 读数，不在本层暴露 mph 节点细节。
