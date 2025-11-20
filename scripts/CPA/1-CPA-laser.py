"""
使用sympy验证CPA-laser条件。

该脚本通过sympy进行符号计算，验证CPA-laser条件文档中的理论计算过程。
"""

import sympy as sp
from sympy import I, cos, sin, symbols

from src.my_dtypes import ParamSpace3D
from src.numerical import get_numpy_func
from src.space_curve import vis_complex_equation

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

M_T = M_loss * M_TL * M_gain

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

param_space = ParamSpace3D(
    x=d,
    x_range=(0, 1),
    y=omega_r,
    y_range=(-1000, 1000),
    z=omega_i,
    z_range=(-1000, 1000),
    fixed_param={Z_r: 1, Z_i: 1, switch_factor: 1},
)
pole_np_func = get_numpy_func(pole_func, param_space)

# %%
# print(pole_np_func(1, 5, 0.6))
_, _, _ = vis_complex_equation(pole_np_func, param_space, plot=True)
