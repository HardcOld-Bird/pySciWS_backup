"""
TBIC系统能带结构的符号和数值计算。

该脚本基于"参考资料/Topo & BIC/解析理论.md"中的核心理论模型，
构建Σ模式和Π模式的哈密顿量，计算本征值（本征频率），
并在k-C参数空间中绘制三维能带图。
"""

import numpy as np
import pyvista as pv
import sympy as sp
from sympy import I, Matrix, exp, symbols

# %%
# ============================================================================
# 1. 定义符号和参数
# ============================================================================

# 定义符号变量
k, C = symbols("k C", real=True)  # 波数k和腔体高度C
fΣ, vΣ, uΣ = symbols("f_Sigma v_Sigma u_Sigma", real=True)  # Σ模式参数
fΠ, vΠ, uΠ = symbols("f_Pi v_Pi u_Pi", real=True)  # Π模式参数

print("=" * 70)
print("TBIC系统能带结构计算")
print("=" * 70)
print("\n定义的符号变量:")
print(f"  波数: k")
print(f"  腔体高度: C")
print(f"  Σ模式参数: f_Σ, v_Σ, u_Σ")
print(f"  Π模式参数: f_Π, v_Π, u_Π")
print()

# %%
# ============================================================================
# 2. 构建哈密顿量矩阵
# ============================================================================

print("=" * 70)
print("构建哈密顿量矩阵")
print("=" * 70)

# Σ模式的哈密顿量 (方程S4)
HΣ = Matrix([
    [fΣ, vΣ + uΣ * exp(-I * k)],
    [vΣ + uΣ * exp(I * k), fΣ]
])

print("\nΣ模式的哈密顿量 H_Σ:")
sp.pprint(HΣ)

# Π模式的哈密顿量 (方程S6)
HΠ = Matrix([
    [fΠ, vΠ + uΠ * exp(-I * k)],
    [vΠ + uΠ * exp(I * k), fΠ]
])

print("\nΠ模式的哈密顿量 H_Π:")
sp.pprint(HΠ)
print()

# %%
# ============================================================================
# 3. 定义参数与C的线性关系
# ============================================================================

print("=" * 70)
print("参数与腔体高度C的线性关系")
print("=" * 70)

# Σ模式的线性关系 (方程S5)
fΣ_expr = 3890 - 10.29 * C
vΣ_expr = 1050 - 12.17 * C
uΣ_expr = 440 + 3.65 * C

print("\nΣ模式 (方程S5):")
print(f"  f_Σ = 3890 - 10.29C")
print(f"  v_Σ = 1050 - 12.17C")
print(f"  u_Σ = 440 + 3.65C")

# Π模式的线性关系 (方程S7)
fΠ_expr = 5850 - 41.20 * C
vΠ_expr = 30 - 0.35 * C
uΠ_expr = 290 - 2.89 * C

print("\nΠ模式 (方程S7):")
print(f"  f_Π = 5850 - 41.20C")
print(f"  v_Π = 30 - 0.35C")
print(f"  u_Π = 290 - 2.89C")
print()

# 将线性关系代入哈密顿量
HΣ_C = HΣ.subs({fΣ: fΣ_expr, vΣ: vΣ_expr, uΣ: uΣ_expr})
HΠ_C = HΠ.subs({fΠ: fΠ_expr, vΠ: vΠ_expr, uΠ: uΠ_expr})

print("代入线性关系后的哈密顿量已更新")
print()

# %%
# ============================================================================
# 4. 计算本征值（符号形式）
# ============================================================================

print("=" * 70)
print("计算本征值（符号形式）")
print("=" * 70)

# 计算Σ模式的本征值
print("\n正在计算Σ模式的本征值...")
eigenvals_Σ = HΣ_C.eigenvals()
eigenvals_Σ_list = list(eigenvals_Σ.keys())

print("Σ模式的本征值（符号形式）:")
for i, ev in enumerate(eigenvals_Σ_list):
    print(f"\n  λ_Σ{i+1} =")
    sp.pprint(sp.simplify(ev))

# 计算Π模式的本征值
print("\n正在计算Π模式的本征值...")
eigenvals_Π = HΠ_C.eigenvals()
eigenvals_Π_list = list(eigenvals_Π.keys())

print("\nΠ模式的本征值（符号形式）:")
for i, ev in enumerate(eigenvals_Π_list):
    print(f"\n  λ_Π{i+1} =")
    sp.pprint(sp.simplify(ev))
print()

# %%
# ============================================================================
# 5. 数值化本征值函数
# ============================================================================

print("=" * 70)
print("数值化本征值函数")
print("=" * 70)

# 将符号表达式转换为numpy函数
eigenval_Σ1_func = sp.lambdify([k, C], eigenvals_Σ_list[0], modules="numpy")
eigenval_Σ2_func = sp.lambdify([k, C], eigenvals_Σ_list[1], modules="numpy")
eigenval_Π1_func = sp.lambdify([k, C], eigenvals_Π_list[0], modules="numpy")
eigenval_Π2_func = sp.lambdify([k, C], eigenvals_Π_list[1], modules="numpy")

print("\n本征值函数已转换为numpy函数")
print()



# %%
# ============================================================================
# 6. 创建参数网格并计算本征值
# ============================================================================

print("=" * 70)
print("创建参数网格并计算本征值")
print("=" * 70)

# 定义参数范围
k_range = (0, 2 * np.pi)  # k的范围：0到2π（一个周期）
C_range = (0, 100)  # C的范围：0到100 mm
k_resolution = 100  # k方向的分辨率
C_resolution = 100  # C方向的分辨率

print(f"\n参数范围:")
print(f"  k: {k_range[0]:.2f} ~ {k_range[1]:.2f} (一个周期)")
print(f"  C: {C_range[0]:.2f} ~ {C_range[1]:.2f} mm")
print(f"  分辨率: {k_resolution} × {C_resolution}")

# 创建网格
k_vals = np.linspace(k_range[0], k_range[1], k_resolution)
C_vals = np.linspace(C_range[0], C_range[1], C_resolution)
K_mesh, C_mesh = np.meshgrid(k_vals, C_vals, indexing="ij")

print("\n正在计算本征值...")

# 计算Σ模式的本征值（原始顺序）
eigenval_Σ1_mesh_raw = np.real(eigenval_Σ1_func(K_mesh, C_mesh))
eigenval_Σ2_mesh_raw = np.real(eigenval_Σ2_func(K_mesh, C_mesh))

# 计算Π模式的本征值（原始顺序）
eigenval_Π1_mesh_raw = np.real(eigenval_Π1_func(K_mesh, C_mesh))
eigenval_Π2_mesh_raw = np.real(eigenval_Π2_func(K_mesh, C_mesh))

print("本征值计算完成")

# 对本征值进行排序，确保顺序一致（从小到大）
print("\n正在对本征值进行排序，避免能带交叉...")

# 对于每个网格点，确保 eigenval_1 <= eigenval_2
# 使用 np.minimum 和 np.maximum 进行逐元素排序
eigenval_Σ1_mesh = np.minimum(eigenval_Σ1_mesh_raw, eigenval_Σ2_mesh_raw)
eigenval_Σ2_mesh = np.maximum(eigenval_Σ1_mesh_raw, eigenval_Σ2_mesh_raw)

eigenval_Π1_mesh = np.minimum(eigenval_Π1_mesh_raw, eigenval_Π2_mesh_raw)
eigenval_Π2_mesh = np.maximum(eigenval_Π1_mesh_raw, eigenval_Π2_mesh_raw)

print("本征值排序完成（λ1 ≤ λ2）")
print()

# %%
# ============================================================================
# 7. 使用PyVista绘制三维能带图
# ============================================================================

print("=" * 70)
print("使用PyVista绘制三维能带图")
print("=" * 70)

# 计算归一化缩放因子，使三个轴的显示范围相近
k_range_len = k_range[1] - k_range[0]  # ~6.28
C_range_len = C_range[1] - C_range[0]  # 100
eigenval_range_len = max(
    eigenval_Σ1_mesh.max() - eigenval_Σ1_mesh.min(),
    eigenval_Σ2_mesh.max() - eigenval_Σ2_mesh.min(),
    eigenval_Π1_mesh.max() - eigenval_Π1_mesh.min(),
    eigenval_Π2_mesh.max() - eigenval_Π2_mesh.min(),
)  # ~几千

# 选择一个参考长度（使用最大的范围）
ref_length = max(k_range_len, C_range_len, eigenval_range_len)

# 计算缩放因子，使各轴显示比例接近1:1:1
k_scale = ref_length / k_range_len
C_scale = ref_length / C_range_len
eigenval_scale = ref_length / eigenval_range_len

print(f"\n自动计算的缩放因子:")
print(f"  k轴缩放: {k_scale:.2f}")
print(f"  C轴缩放: {C_scale:.2f}")
print(f"  本征值轴缩放: {eigenval_scale:.2f}")

# 创建缩放后的网格坐标
K_mesh_scaled = K_mesh * k_scale
C_mesh_scaled = C_mesh * C_scale
eigenval_Σ1_mesh_scaled = eigenval_Σ1_mesh * eigenval_scale
eigenval_Σ2_mesh_scaled = eigenval_Σ2_mesh * eigenval_scale
eigenval_Π1_mesh_scaled = eigenval_Π1_mesh * eigenval_scale
eigenval_Π2_mesh_scaled = eigenval_Π2_mesh * eigenval_scale

# 创建PyVista绘图器
plotter = pv.Plotter()

# 为每个本征值创建结构化网格（使用缩放后的坐标）
# Σ模式 - 本征值1
grid_Σ1 = pv.StructuredGrid(K_mesh_scaled, C_mesh_scaled, eigenval_Σ1_mesh_scaled)
plotter.add_mesh(
    grid_Σ1,
    color="blue",
    opacity=0.7,
    label="Sigma mode - lambda1",
    show_edges=False,
    smooth_shading=True,
)

# Σ模式 - 本征值2
grid_Σ2 = pv.StructuredGrid(K_mesh_scaled, C_mesh_scaled, eigenval_Σ2_mesh_scaled)
plotter.add_mesh(
    grid_Σ2,
    color="cyan",
    opacity=0.7,
    label="Sigma mode - lambda2",
    show_edges=False,
    smooth_shading=True,
)

# Π模式 - 本征值1
grid_Π1 = pv.StructuredGrid(K_mesh_scaled, C_mesh_scaled, eigenval_Π1_mesh_scaled)
plotter.add_mesh(
    grid_Π1,
    color="red",
    opacity=0.7,
    label="Pi mode - lambda1",
    show_edges=False,
    smooth_shading=True,
)

# Π模式 - 本征值2
grid_Π2 = pv.StructuredGrid(K_mesh_scaled, C_mesh_scaled, eigenval_Π2_mesh_scaled)
plotter.add_mesh(
    grid_Π2,
    color="orange",
    opacity=0.7,
    label="Pi mode - lambda2",
    show_edges=False,
    smooth_shading=True,
)

# 添加自定义坐标轴（带真实刻度值）
plotter.show_bounds(
    grid="back",
    location="outer",
    ticks="both",
    xtitle="k (rad)",
    ytitle="C (mm)",
    ztitle="Frequency (Hz)",
    font_size=10,
    color="black",
    # 使用原始范围作为边界
    bounds=(
        k_range[0] * k_scale,
        k_range[1] * k_scale,
        C_range[0] * C_scale,
        C_range[1] * C_scale,
        eigenval_Σ1_mesh_scaled.min(),
        eigenval_Π1_mesh_scaled.max(),
    ),
)

# 添加图例（使用更大的尺寸和更好的位置）
plotter.add_legend(
    size=(0.25, 0.15),
    face="rectangle",
    loc="upper right",
)

# 添加标题
plotter.add_text(
    "TBIC Band Structure\nSigma mode (blue/cyan) & Pi mode (red/orange)",
    position="upper_left",
    font_size=11,
    color="black",
)

# 设置背景颜色为白色，提高可读性
plotter.background_color = "white"

# 设置相机视角
plotter.camera_position = "iso"

print("\n正在显示三维能带图...")
print("提示：")
print("  - 蓝色/青色曲面：Σ模式的两个本征值")
print("  - 红色/橙色曲面：Π模式的两个本征值")
print("  - k轴：波数（0到2π）")
print("  - C轴：腔体高度（0到100 mm）")
print("  - z轴：本征频率（Hz）")
print("  - 坐标已自动缩放以优化显示比例")
print()

# 显示图形
plotter.show()

print("可视化完成！")
print()

# %%
# ============================================================================
# 8. 验证：C=59.4mm时的能带图
# ============================================================================

print("=" * 70)
print("验证：C=59.4mm时的能带图")
print("=" * 70)

C_test = 59.4  # 文献中使用的典型值

# 计算该C值下的本征值（原始顺序）
k_vals_test = np.linspace(0, 2 * np.pi, 200)
eigenval_Σ1_test_raw = np.real(eigenval_Σ1_func(k_vals_test, C_test))
eigenval_Σ2_test_raw = np.real(eigenval_Σ2_func(k_vals_test, C_test))
eigenval_Π1_test_raw = np.real(eigenval_Π1_func(k_vals_test, C_test))
eigenval_Π2_test_raw = np.real(eigenval_Π2_func(k_vals_test, C_test))

# 对本征值进行排序，确保顺序一致
eigenval_Σ1_test = np.minimum(eigenval_Σ1_test_raw, eigenval_Σ2_test_raw)
eigenval_Σ2_test = np.maximum(eigenval_Σ1_test_raw, eigenval_Σ2_test_raw)
eigenval_Π1_test = np.minimum(eigenval_Π1_test_raw, eigenval_Π2_test_raw)
eigenval_Π2_test = np.maximum(eigenval_Π1_test_raw, eigenval_Π2_test_raw)

print(f"\nC = {C_test} mm时的能带范围:")
print(f"  Σ模式: {eigenval_Σ1_test.min():.2f} ~ {eigenval_Σ2_test.max():.2f} Hz")
print(f"  Π模式: {eigenval_Π1_test.min():.2f} ~ {eigenval_Π2_test.max():.2f} Hz")
print()

# 使用matplotlib绘制二维能带图（对应图S3）
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.matplotlib_config import setup_chinese_fonts

setup_chinese_fonts()

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制Σ模式
ax.plot(
    k_vals_test / np.pi,
    eigenval_Σ1_test,
    "b-",
    linewidth=2,
    label="Σ模式 - λ1",
)
ax.plot(
    k_vals_test / np.pi,
    eigenval_Σ2_test,
    "b--",
    linewidth=2,
    label="Σ模式 - λ2",
)

# 绘制Π模式
ax.plot(
    k_vals_test / np.pi,
    eigenval_Π1_test,
    "r-",
    linewidth=2,
    label="Π模式 - λ1",
)
ax.plot(
    k_vals_test / np.pi,
    eigenval_Π2_test,
    "r--",
    linewidth=2,
    label="Π模式 - λ2",
)

ax.set_xlabel("k (单位: π)", fontsize=12)
ax.set_ylabel("频率 (Hz)", fontsize=12)
ax.set_title(f"能带结构 (C = {C_test} mm)", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("二维能带图绘制完成！")
print("该图对应参考资料中的图S3")
print()
