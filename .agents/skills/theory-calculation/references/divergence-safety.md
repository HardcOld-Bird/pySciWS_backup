# 发散安全使用须知

> 薄胶水 `pysci.skills.theory_calculation.divergence_safety` 的使用手册 + 高层模式配方。
> 三原语行为合同以旧项目 7 项测试期望为准绳（不触奇点、邻域加密、升序去重、退化区间
> 报错、窗外/NaN/Inf→NaN、复数取实部、绘图返回 Figure）。

## 1. 核心原则：nan 遮断优于 clip 饱和

发散量绘图**不要** clip 到轴限——饱和段会画成贴轴假平台，被误读为物理平台。
正确做法：窗外置 NaN，让折线自然断开（`mask_divergent` 的语义）。

```python
from pysci.skills.theory_calculation import divergence_safety as ds

x = ds.adaptive_sample_1d((0.0, 1.0), singularities=[0.5], base_resolution=200)
y = 1.0 / (x - 0.5)                       # 奇点发散
fig = ds.quick_plot_divergent(x, y, ylim=(-20.0, 20.0), title="divergent")
```

- `adaptive_sample_1d`：均匀基网格 + 奇点两侧几何趋近点；**奇点本身绝不入网格**；
  输出严格升序去重；`singularities` 区间外忽略；退化/反向区间抛 `ValueError`。
- `mask_divergent(y, window)`：窗外、NaN、±Inf 一律 NaN；复数先取实部再判窗
  （复数物理量请先自行选投影——实部/虚部/模长）。
- `quick_plot_divergent`：`ylim` 给定=遮断窗兼锁定轴范围；省略=按有限值范围自动取窗
  （仅遮 NaN/Inf）。返回 matplotlib Figure（探索图；出版图走③域）。

## 2. 积分发散检测（配方）

数值积分前先做**截断对比**：有限区间积分若随上限快速增长即疑似发散。

```python
import scipy.integrate as si

def integral_growing(f, a, b_list, rtol=0.05):
    """对一系列上限 b 截断积分；相邻结果仍在增长（>rtol）→ 疑似发散，别报数值。"""
    vals, prev = [], None
    for b in b_list:
        v, _ = si.quad(f, a, b, limit=200)
        vals.append(v)
        if prev is not None and abs(v - prev) > rtol * max(abs(v), 1.0):
            return vals, True            # 发散/未收敛迹象
        prev = v
    return vals, False
```

奇点在积分区间内时用 `quad(..., points=[s])` 显式告知，或拆区间后分别 `quad`；
`quad` 报 `IntegrationWarning` 一律视为结果不可信，不得静默吞掉。

## 3. 级数发散检测与收敛半径（配方）

- **部分和检验**：级数 `Σaₙ` 收敛 ⟺ 部分和趋于稳定。对部分和序列做差分，
  末段差分不趋于 0（或振荡放大）→ 发散，禁止截断报数。
- **收敛半径**（幂级数 `Σaₙxⁿ`）：`R = 1/limsup |aₙ|^(1/n)`，有限项用根检近似
  `R ≈ 1/|a_N|^(1/N)`（取尾段多点比对）；有理函数可直接由**极点到展开点的最小距离**
  给出精确半径（`sp.singularities` / `sp.solve(denom)` 辅助定位）。
- 展开点到最近奇点距离即收敛半径——奇点定位与采样复用 `adaptive_sample_1d`。

## 4. 围道检查（配方）

环绕数/回路积分**先验证围道安全**：回路上函数零/极点会使相位积分退化。
（见 [ep-topology-recipes.md](ep-topology-recipes.md) 的 `winding_number`：回路取样点上
`|f|` 近零即报错改道；采样密度保证相邻相位差 < π。）

## 5. `+0j` 分支切割技巧（lambdify NaN 排查）

SymPy 表达式含 `sqrt`/`log` 等复分支时，`lambdify` 到 NumPy 走实数分支会给出 NaN
（如 `np.sqrt(负数)`）。修法：数值化时把输入/常量投到复分支——`np.sqrt(x + 0j)`，
或 lambdify 前 `expr.subs(..., ...)` 保持复数语义。见
[troubleshooting.md](troubleshooting.md)。

## 6. 边界

积分/级数发散检测、收敛半径/围道检查**只提供模式不提供封装**（用户裁决：recipe 文档
形态）；需要更强的求和/求积（加速、解析延拓）→ mpmath `nsum`/`quad` 直调。
scipy.integrate 的能力边界见 [scipy-boundaries.md](scipy-boundaries.md)。
