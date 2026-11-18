# Recipe: TBIC 能带结构计算与分支排序

> ④理论计算域物理配方（新 API 形态：SymPy/NumPy/matplotlib 直调；不含旧代码）。
> 分支排序/追踪细节见 [ep-topology-recipes.md](ep-topology-recipes.md)。

## 物理背景

拓扑 BIC（TBIC）系统：基于 Σ 模式与 Π 模式紧束缚哈密顿量，计算本征频率在
`(k, C)` 参数空间的分布，展示能带交叉与拓扑保护。核心模型：
`H = [[f, v + u·e^{-ik}], [v + u·e^{ik}, f]]`，参数与腔体高度 `C` 线性相关。

## 计算模式

1. 符号建模：含 Bloch 相位 `e^{ik}` 的 2×2 哈密顿量；
2. 参数代入：`f/v/u` 与 `C` 的线性拟合关系；
3. 符号本征值：`H.eigenvals()` 求色散 `λ(k, C)`；
4. 数值网格：`lambdify` + `np.meshgrid` 求值；
5. 分支排序：简并点附近防交叉伪影（取 min/max 或相位连续性）；
6. 可视化：2D 等高线/热图探索（3D 能带曲面渲染归③/⑥域）。

## 关键代码模式（新 API）

```python
import numpy as np
import sympy as sp

from pysci.skills.theory_calculation import latex_export

# 1-2. 哈密顿量（Bloch 形式 + 线性拟合参数）
k, C = sp.symbols("k C", real=True)
H = sp.Matrix([[3890 - 10.29 * C, (1050 - 12.17 * C) + (440 + 3.65 * C) * sp.exp(-sp.I * k)],
               [(1050 - 12.17 * C) + (440 + 3.65 * C) * sp.exp(sp.I * k), 3890 - 10.29 * C]])

# 3. 符号本征值（2x2 闭式即色散关系）
eigvals = list(H.eigenvals().keys())
print(latex_export.expr_to_latex(eigvals[0]))

# 4. 数值网格求值
f1 = sp.lambdify((k, C), eigvals[0], "numpy")
f2 = sp.lambdify((k, C), eigvals[1], "numpy")
k_vals = np.linspace(0.0, 2 * np.pi, 100)
C_vals = np.linspace(0.0, 100.0, 100)
K, CC = np.meshgrid(k_vals, C_vals, indexing="ij")
Z1, Z2 = np.real(f1(K, CC)), np.real(f2(K, CC))

# 5. 分支排序（本征值实数时按大小分上下带；复数/多支时用相位连续性，
#    见 ep-topology-recipes.md「分支追踪」）
Z_lower, Z_upper = np.minimum(Z1, Z2), np.maximum(Z1, Z2)

# 6. 2D 探索（等高线；3D 能带曲面渲染归③/⑥域）
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
cs = ax.contourf(K, CC, Z_lower, levels=20, cmap="viridis")
fig.colorbar(cs, ax=ax, label="Frequency (Hz)")
ax.set_xlabel("k (rad)"); ax.set_ylabel("C (mm)")
fig.savefig("tbic_band.png", dpi=150)
```

## 要点

- `H.eigenvals()` 的键即本征值（重数为值）；2×2 时闭式即色散，不必 `numpy.linalg.eig`；
- 网格求值后 `np.real` 前先确认谱为实（非厄米时用复平面轨迹配方）；
- 分支交叉附近 `np.minimum/np.maximum` 只适用实二支；一般情形见分支追踪配方。

## 来源

旧 pySciWS `data/skills/theoretical_computation/recipes/tbic_band_structure.md`
（物理背景保留，代码按新 API 重写；原始脚本 `scripts/common/topo_bic/1-TBIC-band-structure.py`）。
