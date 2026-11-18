# SciPy 边界注记（scipy.integrate 能力边界）

> 用户裁决（2026-11-17）认可：Julia DifferentialEquations.jl 拒绝后，在此写明超纲处理。
> 结论：**scipy.integrate 覆盖常规面；刚性/DAE/多尺度极限超纲时临时外跑 Julia 或
> sundials 绑定，不入主栈。**

## scipy.integrate 的常规面（够用）

| 场景 | 入口 | 备注 |
|---|---|---|
| 非刚性 ODE | `solve_ivp(..., method="RK45")` | 默认即可 |
| 刚性 ODE | `solve_ivp(..., method="BDF" / "Radau")` | 自动步长+雅可比可选 |
| 单变量积分 | `quad`（奇异点用 `points=`） | 收敛判定见 divergence-safety.md |
| 振荡积分 | `quad(weight="sin"/"cos")` | 指定权重专用核 |
| 双重/多重积分 | `dblquad` / `nquad` | 维数高时改蒙特卡洛 |
| 延迟/积分方程 | 无原生支持 | 用状态扩维或外跑 |

## 超纲边界（满足其一 → 临时外跑，不入主栈）

1. **DAE（微分代数方程）**：scipy 无原生求解器；`solve_ivp` 只能靠消元/加罚硬凑，
   索引-2 以上基本不可行 → 外跑 Julia `DifferentialEquations.jl`（Sundials IDA）。
2. **极端刚性 / 多尺度**：雅可比谱比 > 10⁶ 且 BDF/Radau 步长崩溃、或跨尺度耦合需
   IMEX/多速率积分 → 外跑 Julia（或直接调 sundials C 库的 Python 绑定）。
3. **大规模 PDE**：`solve_ivp` 全量存储+显式循环不占优；首选 `scipy.sparse` 有限差分
   自组刚性系统，仍不收敛 → 外跑。

## 外跑形态（临时，不入依赖）

- Julia：WSL/独立安装 `julia + DifferentialEquations.jl`，脚本式调用，结果 `.csv`/`.npz`
  回填 Python 侧（`session.save_results` 落盘）；**不进 pyproject、不进 venv**。
- sundials 绑定：按需临时装、用完即弃，不 pin 进项目。
- 与④COMSOL 的"外部工具"定位同形：技能文档只记调用约定，不携运行时。

## 检查清单（报数前）

- 收敛性：步长/容差收紧一档重跑，结果漂移 > 目标精度 → 不可信（见 divergence-safety.md）。
- 守恒量（能量/粒子数）沿轨迹漂移 → 方法/容差不匹配，先修再报。
- 刚性误判：非刚性方法爆掉时先换 BDF/Radau，再考虑超纲外跑。
