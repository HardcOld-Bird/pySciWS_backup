# Recipe: CPA-Laser 传递矩阵与零极点探索

> ④理论计算域物理配方（新 API 形态：SymPy/NumPy/matplotlib 直调；不含旧代码）。
> 传递矩阵链是物理模式（链式矩阵乘法即 `sp.Matrix` 直乘，零胶水）。

## 物理背景

CPA（Coherent Perfect Absorption）-Laser 条件：用传递矩阵方法分析含增益/损耗谐振器的
一维声学系统，在复频率平面寻找散射矩阵的极点（laser）与零点（CPA）。

## 计算模式

1. 符号定义：阻抗 `Z`、复角频率 `ω`、波数 `k`、传输线相移 `kd`；
2. 传递矩阵链：`M_loss · M_TL · M_gain`（链式乘积）；
3. 零极点条件：由总矩阵 A/B/C/D 构造极点函数与零点函数；
4. 参数空间探索：`(d, ω_r, ω_i)` 中求复变方程零点集；
5. 可视化：Re=0 与 Im=0 等值线相交（2D 截面；3D 曲线渲染归③/⑥域）。

## 关键代码模式（新 API）

```python
import numpy as np
import sympy as sp

from pysci.skills.theory_calculation import divergence_safety, latex_export

# 1. 符号定义
Zr, Zi, wr, wi, d = sp.symbols("Z_r Z_i omega_r omega_i d", real=True)
c0, rho0 = 343.0, 1.21
Z0 = rho0 * c0

Z1 = Zr + sp.I * Zi
Z2 = Zr + sp.I * Zi            # 增益臂（对称破缺时换写）
omega = wr + sp.I * wi
kd = omega / c0 * d

# 2. 传递矩阵链（sp.Matrix 直乘）
Y1, Y2 = 1 / Z1, 1 / Z2
M_loss = sp.Matrix([[1, 0], [Y1, 1]])
M_TL = sp.Matrix([[sp.cos(kd), sp.I * Z0 * sp.sin(kd)],
                  [sp.I * sp.sin(kd) / Z0, sp.cos(kd)]])
M_gain = sp.Matrix([[1, 0], [Y2, 1]])
M_T = M_loss * M_TL * M_gain          # 链式乘积即传递矩阵

# 3. 零极点函数（散射量的分子/分母）
A, B, C, D = M_T[0, 0], M_T[0, 1], M_T[1, 0], M_T[1, 1]
pole_func = sp.together(A + B / Z0 + C * Z0 + D)
print(latex_export.expr_to_latex(pole_func))
print(latex_export.matrix_to_latex(M_T, env="bmatrix"))

# 4. 数值化 + 二维截面探索（d 固定，扫复频率平面）
pole_num = sp.lambdify((wr, wi), pole_func.subs({Zr: 1, Zi: 1, d: 0.3}), "numpy")
wr_g, wi_g = np.meshgrid(np.linspace(-1000, 1000, 240),
                         np.linspace(-1000, 1000, 240))
F = np.asarray(pole_num(wr_g, wi_g), dtype=complex)  # +0j 分支问题见须知

# 5. 零点集 = Re=0 与 Im=0 等值线相交（2D；3D 零点集渲染归③/⑥域）
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.contour(wr_g, wi_g, F.real, levels=[0.0], colors="C0")
ax.contour(wr_g, wi_g, F.imag, levels=[0.0], colors="C1")
ax.set_xlabel(r"Re $\omega$"); ax.set_ylabel(r"Im $\omega$")
fig.savefig("cpa_zero_set.png", dpi=150)
```

## 要点

- 大表达式先 `sp.together`/`sp.cancel` 再 `lambdify`，比裸代入快且少 NaN；
- 复参数求值统一 `+0j` 投影到复分支（见 [divergence-safety.md](divergence-safety.md)）；
- 三维参数空间 `(d, ω_r, ω_i)` 的零点集曲面渲染是③/⑥域能力，本域只到网格求值为止。

## 来源

旧 pySciWS `data/skills/theoretical_computation/recipes/cpa_transfer_matrix.md`
（物理背景保留，代码按新 API 重写；原始脚本 `scripts/common/cpa/1-CPA-laser.py`）。
