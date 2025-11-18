"""
使用sympy验证CPA-laser条件。

该脚本通过sympy进行符号计算，验证CPA-laser条件文档中的理论计算过程。
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import sympy as sp
from sympy import I, cos, simplify, sin, symbols

# 将项目根目录添加到Python路径
project_root = Path(__name__).parent.parent
sys.path.insert(0, str(project_root))

from src.my_plots import plot_zero_contours

# %%
# ============================================================================
# 1. 定义符号和参数
# ============================================================================

# 常量（声速、密度、特性阻抗）
c_0 = 343
rho_0 = 1.21
Z_0 = rho_0 * c_0

# 实数参数
Z_r, Z_i, omega_r, omega_i, d, switch_factor = symbols(
    "Z_r Z_i omega_r omega_i d switch_factor", real=True
)

# 定义阻抗Z1和Z2
# 当switch_factor = 1时：Z1 = Z2
# 当switch_factor = -1时：Z1 = Z2的复共轭
Z_1 = Z_r + I * Z_i
Z_2 = switch_factor * Z_r + I * Z_i

# 定义角频率
omega = omega_r + I * omega_i

# 定义波数k
k = omega / c_0

# 定义传输线相移
kd = k * d

# 定义导纳
Y_1 = 1 / Z_1
Y_2 = 1 / Z_2

# %%
# ============================================================================
# 2. 构建传递矩阵
# ============================================================================

# M_loss：损耗谐振器（左侧）
M_loss = sp.Matrix([[1, 0], [Y_1, 1]])

# M_TL：传输线
M_TL = sp.Matrix([[cos(kd), I * Z_0 * sin(kd)], [I * sin(kd) / Z_0, cos(kd)]])

# M_gain：增益谐振器（右侧）
M_gain = sp.Matrix([[1, 0], [Y_2, 1]])

# %%
# ============================================================================
# 3. 计算总传递矩阵
# ============================================================================

# M_Tl = M_loss * M_TL * M_gain（左侧入射）
M_T = M_loss * M_TL * M_gain
M_T = simplify(M_T)

# 提取矩阵元素
A, B = M_T[0, 0], M_T[0, 1]
C, D = M_T[1, 0], M_T[1, 1]

# %%
# ============================================================================
# 4. 计算零极点
# ============================================================================

# 极点函数（为0时即为极点）（公分母）
pole_func = A + B / Z_0 + C * Z_0 + D

# 零点函数（为0时即为零点）
zero_func = (B / Z_0 - C * Z_0) ** 2 - (A - D) ** 2 - 4

# %%
# ============================================================================
# 5. 数值计算和可视化
# ============================================================================

# 示例：绘制pole_func和zero_func在不同参数空间中的零等值线
# 示例1：在 (omega_r, omega_i) 平面上绘制极点函数
fixed_params_example1 = {
    Z_r: 400,  # 阻抗实部
    Z_i: -50,  # 阻抗虚部
    d: 0.5,  # 传输线长度
    switch_factor: 1,  # 开关因子（1表示Z1=Z2）
}

fig1, ax1 = plot_zero_contours(
    func=pole_func,
    param_x=omega_r,
    param_y=omega_i,
    x_range=(0, 2000),
    y_range=(-100, 100),
    fixed_params=fixed_params_example1,
    resolution=500,
    title="极点函数零等值线 (omega_r, omega_i 平面)",
    colors="red",
)
plt.show()

# %%
# 示例2：在 (Z_r, Z_i) 平面上绘制零点函数
fixed_params_example2 = {
    omega_r: 1000,  # 角频率实部
    omega_i: 10,  # 角频率虚部
    d: 0.5,  # 传输线长度
    switch_factor: 1,  # 开关因子
}

fig2, ax2 = plot_zero_contours(
    func=zero_func,
    param_x=Z_r,
    param_y=Z_i,
    x_range=(200, 600),
    y_range=(-200, 200),
    fixed_params=fixed_params_example2,
    resolution=500,
    title="零点函数零等值线 (Z_r, Z_i 平面)",
    colors="blue",
)
plt.show()

# %%
# 示例3：在 (d, omega_r) 平面上绘制极点函数
fixed_params_example3 = {
    Z_r: 400,
    Z_i: -50,
    omega_i: 0,
    switch_factor: 1,
}

fig3, ax3 = plot_zero_contours(
    func=pole_func,
    param_x=d,
    param_y=omega_r,
    x_range=(0, 2),
    y_range=(0, 3000),
    fixed_params=fixed_params_example3,
    resolution=500,
    title="极点函数零等值线 (d, omega_r 平面)",
    colors="green",
)
plt.show()
