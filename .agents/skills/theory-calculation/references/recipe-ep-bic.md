# Recipe: EP-BIC 哈密顿量本征分析与复平面轨迹

> ④理论计算域物理配方（新 API 形态：SymPy/NumPy/matplotlib 直调 + 本域薄胶水；
> 不依赖任何旧项目代码）。EP 判据细节见 [ep-topology-recipes.md](ep-topology-recipes.md)。

## 物理背景

EP-BIC（Exceptional Point - Bound State in the Continuum）：二阶非厄米哈密顿量中，
通过引入本征损耗使两个 S-BIC 模式在辐射损耗为零的条件下形成例外点。

核心模型：`H = [[ω̃₁, κ̃], [κ̃, ω̃₂]]`，其中 `ω̃ = ω - i(γʳ + γⁱⁿᵗ)`，
`κ̃ = κ - i√(γ₁ʳγ₂ʳ)`。

## 计算模式

1. 符号建模：复本征频率/耦合系数 → 2×2 哈密顿量；
2. 色散关系：`ω̃± = (ω̃₁+ω̃₂)/2 ± Λ`，`Λ = √[(ω̃₁-ω̃₂)²/4 + κ̃²]`；
3. EP 条件：`Λ = 0` ⟺ 判别式为零；
4. 参数替换：分情况代入（无损耗 diabolic point / 有损耗 EP-BIC）；
5. 数值扫描：本征值随耦合强度 κ 演化；
6. 可视化：复平面轨迹图（探索图；出版图走③域）。

## 关键代码模式（新 API）

```python
import numpy as np
import sympy as sp

from pysci.skills.theory_calculation import latex_export

# 1. 符号定义（SymPy 直调）
w1, w2 = sp.symbols("omega_1 omega_2", real=True)
g1r, g2r = sp.symbols("gamma_1_r gamma_2_r", real=True, positive=True)
g1int, g2int = sp.symbols("gamma_1_int gamma_2_int", real=True, positive=True)
kappa = sp.symbols("kappa", real=True)

# 2. 哈密顿量
w1_t = w1 - sp.I * (g1r + g1int)
w2_t = w2 - sp.I * (g2r + g2int)
kappa_t = kappa - sp.I * sp.sqrt(g1r * g2r)
H = sp.Matrix([[w1_t, kappa_t], [kappa_t, w2_t]])

# 3. EP 条件：2×2 判别式 = charpoly 二次项系数平方 - 4*一次项*常数项
lam = sp.symbols("lambda")
a2, a1, a0 = sp.Poly(H.charpoly(lam).as_expr(), lam).all_coeffs()  # a2 == 1
discriminant = sp.expand(a1**2 - 4 * a2 * a0)
print(latex_export.expr_to_latex(discriminant))  # 设为 0 即 EP 条件

# 4. 特殊情况：EP-BIC（γʳ=0，一个谐振器有本征损耗）
w0, gint = sp.symbols("omega_0 gamma_int", real=True, positive=True)
H_epbic = H.subs({w1: w0, w2: w0, g1r: 0, g2r: 0, g1int: gint, g2int: 0})

# 5. 数值扫描（NumPy 直调；sqrt 内加 +0j 保证复分支，见 divergence-safety.md）
kappa_vals = np.linspace(0.0, 0.2, 200)
w0_val, gint_val = 1.0, 0.1
root = np.sqrt(-(gint_val**2) + 4 * kappa_vals**2 + 0j) / 2
center = w0_val - 1j * gint_val / 2
eig_plus, eig_minus = center + root, center - root

# 6. 复平面轨迹（matplotlib 直调；经本征值合并点=EP，见 ep-topology-recipes.md）
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(eig_plus.real, eig_plus.imag, label="+")
ax.plot(eig_minus.real, eig_minus.imag, label="-")
ax.set_xlabel(r"Re $\tilde{\omega}$"); ax.set_ylabel(r"Im $\tilde{\omega}$")
ax.legend(); fig.savefig("ep_bic_complex_plane.png", dpi=150)
```

## 要点

- 判别式不必手推：`sp.Poly(H.charpoly(lam).as_expr(), lam).all_coeffs()` 取系数即可；
  数值判据（判别式过零 + 本征向量夹角）见 [ep-topology-recipes.md](ep-topology-recipes.md)。
- 结果导 LaTeX 一律走 `latex_export`（确定性输出，可直接贴⑤域 amsmath）。

## 来源

旧 pySciWS `data/skills/theoretical_computation/recipes/ep_bic_hamiltonian.md`
（物理背景保留，代码按新 API 重写；原始脚本 `scripts/common/ep_bic/1-EP-BIC-Hamiltonian.py`）。
