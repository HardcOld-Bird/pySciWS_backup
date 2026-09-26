# Recipe: CPA-Laser 传递矩阵与复变零点集可视化

## 物理背景

CPA（Coherent Perfect Absorption）-Laser 条件：通过传递矩阵方法分析含增益/损耗谐振器的
一维声学系统，寻找零极点（scattering matrix 的极点/零点）在复频率平面中的分布。

## 计算模式

1. **符号定义**：定义阻抗 Z、角频率 ω（复数）、波数 k、传输线相移 kd
2. **传递矩阵构建**：M_loss * M_TL * M_gain（链式乘积）
3. **零极点条件**：从总传递矩阵元素 A/B/C/D 构建极点函数和零点函数
4. **参数空间探索**：在 3D 参数空间 (d, ω_r, ω_i) 中寻找复变方程的零点集
5. **可视化**：pyvista 3D 空间曲线（Re=0 与 Im=0 等值面的交集）

## 关键代码模式（新 API）

```python
import sympy as sp
from sympy import I, cos, sin, symbols

from pysci.skills.theoretical_computation.tools import cas, numerical, topology, visualize

# 1. 符号定义
Zr, Zi, wr, wi, d, switch = symbols("Z_r Z_i omega_r omega_i d switch", real=True)
c0, rho0 = 343, 1.21
Z0 = rho0 * c0

Z1 = Zr + I * Zi
Z2 = switch * Zr + I * Zi
omega = wr + I * wi
k = omega / c0
kd = k * d

# 2. 传递矩阵
Y1, Y2 = 1/Z1, 1/Z2
M_loss = sp.Matrix([[1, 0], [Y1, 1]])
M_TL = sp.Matrix([[cos(kd), I*Z0*sin(kd)], [I*sin(kd)/Z0, cos(kd)]])
M_gain = sp.Matrix([[1, 0], [Y2, 1]])
M_T = cas.transfer_matrix_chain(M_loss, M_TL, M_gain)

# 3. 极点函数
A, B, C, D = M_T[0,0], M_T[0,1], M_T[1,0], M_T[1,1]
pole_func = A + B/Z0 + C*Z0 + D

# 4. 参数空间 + 数值化
space = numerical.ParamSpace(
    axes=[
        numerical.ParamAxis(d, (0, 1), 50),
        numerical.ParamAxis(wr, (-1000, 1000), 50),
        numerical.ParamAxis(wi, (-1000, 1000), 50),
    ],
    fixed={Zr: 1, Zi: 1, switch: 1},
)
func = numerical.lambdify_expr(pole_func, space)

# 5. 零点集提取 + 可视化
result = topology.find_zero_set(func, space)
plotter = visualize.quick_plot_zero_set_3d(
    result.curve, result.real_face, result.imag_face,
    param_labels=("d", "Re(omega)", "Im(omega)"),
    out_path="plots/cpa_zero_set.png",
)
```

## 来源

原始脚本：`scripts/common/cpa/1-CPA-laser.py`
