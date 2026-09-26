# Recipe: TBIC 能带结构计算与 3D 曲面可视化

## 物理背景

拓扑 BIC（TBIC）系统的能带结构：基于 Σ 模式和 Π 模式的紧束缚哈密顿量，
计算本征频率在 (k, C) 参数空间中的分布，展示能带交叉与拓扑保护。

核心模型：H = [[f, v + u*exp(-ik)], [v + u*exp(ik), f]]，参数与腔体高度 C 线性相关。

## 计算模式

1. **符号建模**：构建含 Bloch 相位 exp(ik) 的 2x2 哈密顿量
2. **参数代入**：将 f/v/u 与 C 的线性拟合关系代入
3. **符号本征值**：求解色散关系 λ(k, C)
4. **数值网格**：在 (k, C) 二维参数空间上 lambdify 并求值
5. **分支排序**：对简并点附近的本征值排序（避免交叉伪影）
6. **3D 可视化**：pyvista StructuredGrid 曲面展示能带

## 关键代码模式（新 API）

```python
import numpy as np
import sympy as sp
from sympy import I, Matrix, exp, symbols

from pysci.skills.theoretical_computation.tools import cas, eigen, numerical, visualize

# 1. 符号定义
k, C = symbols("k C", real=True)
fS, vS, uS = symbols("f_Sigma v_Sigma u_Sigma", real=True)

# 2. 哈密顿量（Bloch 形式）
H_Sigma = cas.symbolic_matrix([
    [fS, vS + uS*exp(-I*k)],
    [vS + uS*exp(I*k), fS],
])

# 3. 代入线性拟合参数
H_Sigma_C = H_Sigma.subs({
    fS: 3890 - 10.29*C,
    vS: 1050 - 12.17*C,
    uS: 440 + 3.65*C,
})

# 4. 符号本征值
eig_sys = eigen.eigensystem_symbolic(H_Sigma_C)
lam1, lam2 = eig_sys.eigenvalues[0], eig_sys.eigenvalues[1]

# 5. 数值化 + 网格求值
func1 = sp.lambdify([k, C], lam1, modules="numpy")
func2 = sp.lambdify([k, C], lam2, modules="numpy")

k_vals = np.linspace(0, 2*np.pi, 100)
C_vals = np.linspace(0, 100, 100)
K_mesh, C_mesh = np.meshgrid(k_vals, C_vals, indexing="ij")

Z1 = np.real(func1(K_mesh, C_mesh))
Z2 = np.real(func2(K_mesh, C_mesh))
# 分支排序
Z_lower = np.minimum(Z1, Z2)
Z_upper = np.maximum(Z1, Z2)

# 6. 3D 能带曲面
plotter = visualize.quick_plot_band_structure_3d(
    K_mesh, C_mesh, [Z_lower, Z_upper],
    labels=["Sigma-1", "Sigma-2"],
    colors=["blue", "cyan"],
    axis_titles=("k (rad)", "C (mm)", "Frequency (Hz)"),
    title="TBIC Band Structure",
    out_path="plots/tbic_band_3d.png",
)
```

## 来源

原始脚本：`scripts/common/topo_bic/1-TBIC-band-structure.py`
