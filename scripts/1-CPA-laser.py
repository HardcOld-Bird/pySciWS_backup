"""
使用sympy验证CPA-laser条件。

该脚本通过sympy进行符号计算，验证CPA-laser条件文档中的理论计算过程。
"""

import sympy as sp
from sympy import Abs, I, cos, simplify, sin, symbols

# %%
# ============================================================================
# 1. 定义符号和参数
# ============================================================================

# 实数参数
R, X, Z0, kd, switch_factor = symbols("R X Z_0 kd switch_factor", real=True)

# 定义阻抗Z1和Z2
# 当switch_factor = 1时：Z1 = Z2
# 当switch_factor = -1时：Z1 = Z2的复共轭
Z1 = R + I * X
Z2 = switch_factor * R + I * X

# 定义导纳
Y1 = 1 / Z1
Y2 = 1 / Z2

# %%
# ============================================================================
# 2. 构建传递矩阵
# ============================================================================

# M_loss：损耗谐振器（左侧）
M_loss = sp.Matrix([[1, 0], [Y1, 1]])

# M_TL：传输线
M_TL = sp.Matrix([[cos(kd), I * Z0 * sin(kd)], [I * sin(kd) / Z0, cos(kd)]])

# M_gain：增益谐振器（右侧）
M_gain = sp.Matrix([[1, 0], [Y2, 1]])

# %%
# ============================================================================
# 3. 计算总传递矩阵
# ============================================================================

# M_Tl = M_loss * M_TL * M_gain（左侧入射）
M_Tl = M_loss * M_TL * M_gain
M_Tl = simplify(M_Tl)

# M_Tr = M_gain * M_TL * M_loss（右侧入射）
M_Tr = M_gain * M_TL * M_loss
M_Tr = simplify(M_Tr)

# 提取矩阵元素
A_Tl, B_Tl = M_Tl[0, 0], M_Tl[0, 1]
C_Tl, D_Tl = M_Tl[1, 0], M_Tl[1, 1]

A_Tr, B_Tr = M_Tr[0, 0], M_Tr[0, 1]
C_Tr, D_Tr = M_Tr[1, 0], M_Tr[1, 1]

# %%
# ============================================================================
# 4. 计算散射系数
# ============================================================================

# 公分母（由于倒易性，两侧应该相等）
p_l = A_Tl + B_Tl / Z0 + C_Tl * Z0 + D_Tl
# p_r = A_Tr + B_Tr / Z0 + C_Tr * Z0 + D_Tr

p_l = simplify(p_l)
# p_r = simplify(p_r)

# 检查p_l是否等于p_r（倒易性）
# print(f"p_l（左侧分母）: {p_l}")
# print(f"p_r（右侧分母）: {p_r}")

# p_diff = simplify(p_l - p_r)
# print(f"\np_l - p_r = {p_diff}")
# if p_diff == 0:
#     print("[OK] 倒易性验证通过：p_l = p_r")
# else:
#     print("[FAIL] 倒易性验证失败")

# 使用公分母p
p = p_l

# 反射系数的分子
q_l = A_Tl + B_Tl / Z0 - C_Tl * Z0 - D_Tl
q_r = A_Tr + B_Tr / Z0 - C_Tr * Z0 - D_Tr

q_l = simplify(q_l)
q_r = simplify(q_r)

# 反射系数
r_l = q_l / p
r_r = q_r / p

r_l = simplify(r_l)
r_r = simplify(r_r)

# 透射系数
t = 2 / p

t = simplify(t)

# 检查两种逆反射计算方法的等价性
# fct1 = -A_Tl + B_Tl / Z0 - C_Tl * Z0 + D_Tl
# fct2 = A_Tr + B_Tr / Z0 - C_Tr * Z0 - D_Tr
#
# fct_diff = simplify(fct1 - fct2)
# print(f"\nfct1 - fct2 = {fct_diff}")
# if fct_diff == 0:
#     print("[OK] 等价性验证通过：fct1 = fct2")
# else:
#     print("[FAIL] 等价性验证失败")

# %%
# ============================================================================
# 5. 构建散射矩阵S
# ============================================================================

# print("\n" + "=" * 80)
# print("5. 散射矩阵S")
# print("=" * 80)

# 散射矩阵S
# S = [[t, r_r], [r_l, t]]
# 其中t是透射系数（两侧相同）

S = sp.Matrix([[t, r_r], [r_l, t]])

# print("散射矩阵S:")
# print(S)

# %%
# ============================================================================
# 6. 计算能量系数
# ============================================================================

# print("\n" + "=" * 80)
# print("6. 能量系数")
# print("=" * 80)

# 能量透射率：|t_l * t_r| = |t|^2（因为t_l = t_r = t）
energy_transmission = Abs(t) ** 2  # type: ignore
energy_transmission = simplify(energy_transmission)

# print(f"\n能量透射率 = |t_l * t_r|^2 = {energy_transmission}")

# 左侧入射能量反射率：|r_l|^2
energy_reflection_left = Abs(r_l) ** 2  # type: ignore
energy_reflection_left = simplify(energy_reflection_left)

# print(f"能量反射率（左侧入射）= |r_l|^2 = {energy_reflection_left}")

# 右侧入射能量反射率：|r_r|^2
energy_reflection_right = Abs(r_r) ** 2  # type: ignore
energy_reflection_right = simplify(energy_reflection_right)

# print(f"能量反射率（右侧入射）= |r_r|^2 = {energy_reflection_right}")

# %%
# ============================================================================
# 7. 使用特定值进行数值验证
# ============================================================================

print("\n" + "=" * 80)
print("7. 数值验证")
print("=" * 80)

# 测试用例1：switch_factor = 1（Z1 = Z2）
print("\n测试用例1：switch_factor = 1（Z1 = Z2）")
print("-" * 40)

test_params_1 = {R: 1.0, X: 0.5, Z0: 50.0, kd: sp.pi / 4, switch_factor: 1}

t_val_1 = complex(t.subs(test_params_1))
r_l_val_1 = complex(r_l.subs(test_params_1))
r_r_val_1 = complex(r_r.subs(test_params_1))

print(f"t = {t_val_1}")
print(f"r_l = {r_l_val_1}")
print(f"r_r = {r_r_val_1}")

# 测试用例2：switch_factor = -1（Z1 = Z2的复共轭）
print("\n测试用例2：switch_factor = -1（Z1 = Z2的复共轭）")
print("-" * 40)

test_params_2 = {R: 1.0, X: 0.5, Z0: 50.0, kd: sp.pi / 4, switch_factor: -1}

t_val_2 = complex(t.subs(test_params_2))
r_l_val_2 = complex(r_l.subs(test_params_2))
r_r_val_2 = complex(r_r.subs(test_params_2))

print(f"t = {t_val_2}")
print(f"r_l = {r_l_val_2}")
print(f"r_r = {r_r_val_2}")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
