# Recipe: EP-BIC 哈密顿量本征分析与复平面轨迹

## 物理背景

EP-BIC（Exceptional Point - Bound State in the Continuum）融合：在二阶非厄米哈密顿量中，
通过引入本征损耗使两个 S-BIC 模式在辐射损耗为零的条件下形成例外点。

核心模型：H = [[ω̃₁, κ̃], [κ̃, ω̃₂]]，其中 ω̃ = ω - i(γʳ + γⁱⁿᵗ)，κ̃ = κ - i√(γ₁ʳγ₂ʳ)

## 计算模式

1. **符号建模**：定义复数本征频率和耦合系数，构建 2x2 哈密顿量
2. **色散关系**：ω̃± = (ω̃₁+ω̃₂)/2 ± Λ，Λ = √[(ω̃₁-ω̃₂)²/4 + κ̃²]
3. **EP 条件**：Λ = 0 → 判别式为零
4. **参数替换**：分情况代入（无损耗 diabolic point / 有损耗 EP-BIC）
5. **数值扫描**：本征值随耦合强度 κ 的演化
6. **可视化**：复平面轨迹图 + 实部/虚部分裂图

## 关键代码模式（新 API）

```python
import numpy as np
import sympy as sp
from sympy import I, Matrix, sqrt, symbols

from pysci.skills.theoretical_computation.tools import cas, eigen, visualize

# 1. 符号定义
w1, w2 = symbols("omega_1 omega_2", real=True)
g1r, g2r = symbols("gamma_1_r gamma_2_r", real=True, positive=True)
g1int, g2int = symbols("gamma_1_int gamma_2_int", real=True, positive=True)
kappa = symbols("kappa", real=True)

# 2. 哈密顿量
w1_tilde = w1 - I*(g1r + g1int)
w2_tilde = w2 - I*(g2r + g2int)
kappa_tilde = kappa - I*sqrt(g1r*g2r)
H = cas.symbolic_matrix([[w1_tilde, kappa_tilde], [kappa_tilde, w2_tilde]])

# 3. EP 条件（判别式 = 0）
ep_cond = cas.discriminant_2x2(H)
cas.pretty_print(ep_cond, "EP condition (set = 0)")

# 4. 特殊情况：EP-BIC（γʳ=0, 一个谐振器有本征损耗）
w0, gint = symbols("omega_0 gamma_int", real=True, positive=True)
H_epbic = H.subs({w1: w0, w2: w0, g1r: 0, g2r: 0, g1int: gint, g2int: 0})

# 5. 数值扫描
kappa_vals = np.linspace(0, 0.2, 200)
w0_val, gint_val = 1.0, 0.1

def H_numeric(k):
    Lam = np.sqrt(-(gint_val**2) + 4*k**2 + 0j) / 2
    return np.array([[w0_val - 1j*gint_val/2 + Lam, 0], [0, w0_val - 1j*gint_val/2 - Lam]])

# 直接计算本征值
eig_plus = w0_val - 1j*gint_val/2 + np.sqrt(-(gint_val**2) + 4*kappa_vals**2 + 0j)/2
eig_minus = w0_val - 1j*gint_val/2 - np.sqrt(-(gint_val**2) + 4*kappa_vals**2 + 0j)/2
eigenvalues = np.column_stack([eig_plus, eig_minus])

# 6. 复平面轨迹图
fig = visualize.quick_plot_complex_plane(
    eigenvalues, param_values=kappa_vals,
    title="EP-BIC: eigenvalue trajectories",
    mark_ep=True,
    out_path="plots/ep_bic_complex_plane.png",
)
```

## 来源

原始脚本：`scripts/common/ep_bic/1-EP-BIC-Hamiltonian.py`
