# Recipe: 逆设计协议（目标函数协议 + scipy DE + 每代 checkpoint）

> ④理论计算域方法配方（新 API 形态：scipy 直调 + 本域薄胶水
> `pysci.skills.theory_calculation.inverse_design`）。
> 语义提取自旧 pySciWS 研究线 `pysci.research.gain_ep.theory.inverse_design.py`
> （目标项协议/断点续算形态；⭐几何约束研究线不入仓），按新 API 重写。
> **用户裁决（2026-11-17）：自研 DE 不迁移**，统一换
> `scipy.optimize.differential_evolution` + 薄胶水每代 checkpoint；行为差异见 §4。

## 1. 目标函数协议（参数向量 → 标量）

薄胶水只约定两个可调用协议（duck typing，`typing.Protocol` 声明）：

| 协议 | 签名 | 语义 |
|---|---|---|
| `Objective` | `objective(x: NDArray) -> float` | 参数向量（一维）→ 标量目标，**越小越好** |
| `Constraint` | `constraint(x: NDArray) -> float` | 参数向量 → 标量残差，**≤0 可行**（映射为 scipy `NonlinearConstraint`） |

约定：
- 不可行点由约束表达，或在目标里返回 **`+inf` 罚分**（勿抛异常，优化器要能跳过）；
  `NaN` 视为编程错误，薄胶水直接 `ValueError` 拦截。
- 复矩阵目标（如 S 矩阵匹配）在调用方折叠成标量。旧研究线目标项的方法论注记可参考：
  矩阵匹配用加权最小二乘；**相位项用 `1−cos Δφ`**（无 2π 跳变，且 |目标|=0 的元素
  自动跳过相位项）；置零项直接 |S_pq|²；约当块目标用本征值最小两两间隙² 作平滑代理，
  最终解用 [recipe-cmt-ep-ndim.md](recipe-cmt-ep-ndim.md) 的 `jordan_structure` 复核。

## 2. scipy.optimize.differential_evolution 用法

```python
import numpy as np
from pysci.skills.theory_calculation.inverse_design import run_inverse_design

def objective(x: np.ndarray) -> float:
    return float((x[0] - 0.5) ** 2 + (x[1] + 0.3) ** 2)

res = run_inverse_design(
    objective, bounds=[(-1.0, 1.0), (-1.0, 1.0)],
    seed=42, popsize=15, maxiter=100, tol=1e-8,
    checkpoint_path="data/research/1_demo/theory/inv/results/de.npz",
)
print(res.x, res.fun)      # scipy OptimizeResult：x/fun/nit/population/…
```

要点：
- **bounds 必填**且 `lo < hi`（scipy 不校验 lo>hi，薄胶水先行拦截）；
- 总种群 = `popsize × 维数`；固定 `seed` 全流程确定性（可断言）；
- 收敛判据是**种群收敛度** `tol`（`res.convergence` = tol/种群散布度），与旧自研
  "代数+停滞史"判据不同（§4）；`polish=True` 末尾做 L-BFGS-B 精修（约束下走
  trust-constr）；
- 大目标函数用 `workers=-1`（进程池并行）时注意 Windows spawn：目标必须是可 pickle
  的顶层函数/闭包仅捕获可序列化对象。

## 3. callback 每代 checkpoint（npz/json 落盘、可恢复续跑）

scipy ≥1.15 的 DE callback **按参数名路由**：参数名恰为 `intermediate_result` 时收到
含 `population`/`population_energies`/`x`/`fun`/`nit`/`nfev`/`convergence` 的
`OptimizeResult`（否则退化为旧 `(xk, convergence)` 签名，拿不到种群）。薄胶水的
`CheckpointCallback` 即按此签名封装，**每代**落盘：

- `<path>.npz`：`generation`/`best_x`/`best_fun`/`population`/`population_energies`/
  `nfev`/`convergence`（恢复续跑的真源）；
- `<path>.json`：同内容摘要 + 时间戳（人读/巡检）。

续跑语义（`resume_from=<path>.npz`）：`init=checkpoint.population` **种群精确接续**
（`maxiter=0` 时返回值逐位等于 checkpoint 状态，可断言），`generation` 累计。
checkpoint 是运行产物，写入限 `data/` 内（`pysci.paths.assert_within_data` 护栏）。

## 4. 注记：scipy DE 替代自研 DE 的裁决与行为差异（2026-11-17）

| 维度 | 旧自研 DE/rand/1/bin（不迁移） | scipy `differential_evolution`（现行） |
|---|---|---|
| 约束处理 | **修复式解码**（repair：压缩/夹紧保证可行）+ 非法个体罚分 | `NonlinearConstraint` **可行性规则**（约束违反度参与排序）；无 repair——物理可行域要么映射进 bounds，要么写成 `Constraint` 残差 |
| 收敛判据 | 代数上限 + 停滞史（自定） | 种群收敛度 `tol`（+ `maxiter`/`maxfun` 终止）；`res.convergence` 是 tol/散布度口径 |
| 断点续算 | 种群 + **RNG 状态**精确续算（逐位一致） | 种群精确接续 + **RNG 流重启**（seed 重新播种）：续跑轨迹与"不中断直跑"不逐位一致，分布语义一致 |
| 每代 checkpoint | 自研 callback 存种群/适应度/RNG | `intermediate_result` 回调存种群/适应度/收敛度（RNG 状态拿不到，见上行） |
| 并行 | 自研 `ProcessPoolExecutor` | `workers=` 原生进程池（Windows spawn 同样要求可 pickle） |
| 精修 | 无 | `polish` 可选 L-BFGS-B/trust-constr |

## 5. 组合约束的调用形态

```python
from pysci.skills.theory_calculation.inverse_design import run_inverse_design

def gap_ok(x):        # 约束协议：≤0 可行
    return 0.1 - x[0]          # 要求 x[0] ≥ 0.1

res = run_inverse_design(objective, bounds=[(0.0, 1.0), (0.0, 1.0)],
                         constraints=[gap_ok], seed=0)
```

## 来源

目标项协议形态（`term.evaluate(S, ports) -> float` 的可组合加权、相位 `1−cos Δφ`
技巧）与断点续算纪律提取自旧 pySciWS `src/pysci/research/gain_ep/theory/
inverse_design.py`（2026-11-17 解剖 `others/community_research/2026-11-17/
legacy-02-cmt-gain-ep.md` §3.4，含用户裁决记录）；DE 实现整体换 scipy，薄胶水
`sub_root/src/pysci/skills/theory_calculation/inverse_design.py` 为按新 API 重写，
未复制旧代码。⭐研究线的管槽几何修复式解码（min_gap=2mm 等绝对物理约束）不入仓。
