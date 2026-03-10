"""
EP-BIC核心理论模型的符号和数值计算。

该脚本基于文章中的二阶哈密顿量（方程1），探索BIC与EP的融合条件。
核心模型：
    H = [[ω̃₁, κ̃], [κ̃, ω̃₂]]
其中：
    ω̃₁ = ω₁ - i(γ₁ʳ + γ₁ⁱⁿᵗ)
    ω̃₂ = ω₂ - i(γ₂ʳ + γ₂ⁱⁿᵗ)
    κ̃ = κ - i√(γ₁ʳγ₂ʳ)
"""

# import sys
# from pathlib import Path

# 添加项目根目录到路径
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy import I, Matrix, sqrt, symbols

# 导入matplotlib中文配置
from config.matplotlib_config import setup_chinese_fonts

setup_chinese_fonts()

# %%
# ============================================================================
# 1. 定义符号和参数
# ============================================================================

# 定义实数符号
ω1, ω2 = symbols("omega_1 omega_2", real=True)  # 谐振频率
γ1ʳ, γ2ʳ = symbols("gamma_1_r gamma_2_r", real=True, positive=True)  # 辐射损耗
γ1ⁱⁿᵗ, γ2ⁱⁿᵗ = symbols("gamma_1_int gamma_2_int", real=True, positive=True)  # 本征损耗
κ = symbols("kappa", real=True)  # 近场耦合系数

# 定义复数本征频率
ω̃1 = ω1 - I * (γ1ʳ + γ1ⁱⁿᵗ)
ω̃2 = ω2 - I * (γ2ʳ + γ2ⁱⁿᵗ)

# 定义复数耦合系数
κ̃ = κ - I * sqrt(γ1ʳ * γ2ʳ)

# %%
# ============================================================================
# 2. 构建哈密顿量矩阵
# ============================================================================

# 构建2x2哈密顿量矩阵（方程1）
H = Matrix([[ω̃1, κ̃], [κ̃, ω̃2]])

print("哈密顿量矩阵 H:")
sp.pprint(H)
print()

# %%
# ============================================================================
# 3. 计算特征值（色散关系）
# ============================================================================

# 计算特征值（方程2）
eigenvalues = H.eigenvals()
print("特征值（未简化）:")
for eigval, multiplicity in eigenvalues.items():
    print(f"  特征值: {eigval}, 重数: {multiplicity}")
print()

# 手动构建色散关系（方程2）
# ω̃± = (ω̃₁ + ω̃₂)/2 ± Λ
# 其中 Λ = √[(ω̃₁ - ω̃₂)²/4 + κ̃²]

Λ = sqrt((ω̃1 - ω̃2) ** 2 / 4 + κ̃**2)
ω̃十 = (ω̃1 + ω̃2) / 2 + Λ
ω̃一 = (ω̃1 + ω̃2) / 2 - Λ

print("色散关系（方程2）:")
print("ω̃₊ = (ω̃₁ + ω̃₂)/2 + Λ")
print("ω̃₋ = (ω̃₁ + ω̃₂)/2 - Λ")
print("\nΛ = ")
sp.pprint(Λ)
print()

# %%
# ============================================================================
# 4. 分析EP和BIC条件
# ============================================================================

print("=" * 70)
print("EP和BIC条件分析")
print("=" * 70)

# EP条件：Λ = 0
print("\n1. EP条件（Λ = 0）:")
print("   当 (ω̃₁ - ω̃₂)²/4 + κ̃² = 0 时，两个模式在EP处合并")

# BIC条件：辐射损耗为零
print("\n2. S-BIC条件（对称保护BIC）:")
print("   γ₁ʳ = γ₂ʳ = 0 → 哈密顿量变为厄米矩阵")
print("   此时只能形成diabolic point，不能形成EP")

# EP-BIC条件：引入本征损耗
print("\n3. EP-BIC条件（文章核心发现）:")
print("   设 γ₁ʳ = γ₂ʳ = 0（两个S-BIC）")
print("   引入本征损耗 γ₁ⁱⁿᵗ ≠ 0 或 γ₂ⁱⁿᵗ ≠ 0")
print("   EP条件变为: 2κ = γⁱⁿᵗ（假设只有一个谐振器有本征损耗）")
print()

# %%
# ============================================================================
# 5. 特殊情况：两个相同的S-BIC（图1a场景）
# ============================================================================

print("=" * 70)
print("场景1: 两个相同的S-BIC（无本征损耗）")
print("=" * 70)

# 设置条件：ω₁ = ω₂ = ω₀, γ₁ʳ = γ₂ʳ = 0, γ₁ⁱⁿᵗ = γ₂ⁱⁿᵗ = 0
ω0 = symbols("omega_0", real=True)
H_case1 = H.subs(
    {
        # ω1: ω0,
        # ω2: ω0,
        γ1ʳ: 0,
        γ2ʳ: 0,
        γ1ⁱⁿᵗ: 0,
        γ2ⁱⁿᵗ: 0,
    }
)

print("\n简化后的哈密顿量:")
sp.pprint(H_case1)

eigenvals_case1 = H_case1.eigenvals()
print("\n特征值:")
for eigval, multiplicity in eigenvals_case1.items():
    eigval_simplified = sp.simplify(eigval)
    print(f"  ω = {eigval_simplified}")

print("\n结论: 形成diabolic point，特征值线性分裂 ∝ |κ|")
print()

# %%
# ============================================================================
# 6. 特殊情况：引入本征损耗的EP-BIC（图1b场景）
# ============================================================================

print("=" * 70)
print("场景2: 引入本征损耗的EP-BIC")
print("=" * 70)

# 设置条件：ω₁ = ω₂ = ω₀, γ₁ʳ = γ₂ʳ = 0, γ₁ⁱⁿᵗ = γⁱⁿᵗ, γ₂ⁱⁿᵗ = 0
γⁱⁿᵗ = symbols("gamma_int", real=True, positive=True)
H_case2 = H.subs(
    {
        ω1: ω0,
        ω2: ω0,
        γ1ʳ: 0,
        γ2ʳ: 0,
        γ1ⁱⁿᵗ: γⁱⁿᵗ,
        γ2ⁱⁿᵗ: 0,
    }
)

print("\n简化后的哈密顿量:")
sp.pprint(H_case2)

# 计算判别式Λ
Λ_case2 = sqrt((ω0 - I * γⁱⁿᵗ - ω0) ** 2 / 4 + κ**2)
Λ_case2_simplified = sp.simplify(Λ_case2)

print("\n判别式 Λ:")
sp.pprint(Λ_case2_simplified)

# EP条件
print("\nEP条件: Λ = 0")
ep_condition = sp.Eq((-(γⁱⁿᵗ**2) / 4 + κ**2), 0)
print("  即:")
sp.pprint(ep_condition)
ep_kappa = sp.solve(ep_condition, κ)
print("\n  解得: κ = ±γⁱⁿᵗ/2")
print("  (取正值) κ_EP = γⁱⁿᵗ/2")
print()

# %%
# ============================================================================
# 7. 数值计算：特征值随耦合强度的变化
# ============================================================================

print("=" * 70)
print("数值计算：特征值随耦合强度κ的变化")
print("=" * 70)

# 设置数值参数
ω0_val = 1.0  # 归一化频率
γⁱⁿᵗ_val = 0.1  # 本征损耗

# 创建κ的数值范围
κ_vals = np.linspace(0, 0.2, 200)

# 计算两种情况的特征值

# 情况1：无本征损耗（diabolic point）
# 先替换参数，再简化
ω̃十_case1_expr = ω̃十.subs(
    {
        ω1: ω0,
        ω2: ω0,
        γ1ʳ: 0,
        γ2ʳ: 0,
        γ1ⁱⁿᵗ: 0,
        γ2ⁱⁿᵗ: 0,
    }
)
ω̃十_case1_expr = sp.simplify(ω̃十_case1_expr)

ω̃一_case1_expr = ω̃一.subs(
    {
        ω1: ω0,
        ω2: ω0,
        γ1ʳ: 0,
        γ2ʳ: 0,
        γ1ⁱⁿᵗ: 0,
        γ2ⁱⁿᵗ: 0,
    }
)
ω̃一_case1_expr = sp.simplify(ω̃一_case1_expr)

# 使用numpy模块进行lambdify
ω̃十_case1_func = sp.lambdify([κ, ω0], ω̃十_case1_expr, modules="numpy")

ω̃一_case1_func = sp.lambdify([κ, ω0], ω̃一_case1_expr, modules="numpy")

# 情况2：有本征损耗（EP-BIC）
# 先替换参数，再简化，然后lambdify
ω̃十_case2_expr = ω̃十.subs(
    {
        ω1: ω0,
        ω2: ω0,
        γ1ʳ: 0,
        γ2ʳ: 0,
        γ1ⁱⁿᵗ: γⁱⁿᵗ,
        γ2ⁱⁿᵗ: 0,
    }
)
ω̃十_case2_expr = sp.simplify(ω̃十_case2_expr)

ω̃一_case2_expr = ω̃一.subs(
    {
        ω1: ω0,
        ω2: ω0,
        γ1ʳ: 0,
        γ2ʳ: 0,
        γ1ⁱⁿᵗ: γⁱⁿᵗ,
        γ2ⁱⁿᵗ: 0,
    }
)
ω̃一_case2_expr = sp.simplify(ω̃一_case2_expr)

# 使用numpy模块进行lambdify
# 注意：当 κ < γⁱⁿᵗ/2 时，Λ = √(-γⁱⁿᵗ²/4 + κ²) 是纯虚数
# 我们需要手动处理复数平方根，避免numpy的sqrt对负数返回nan
ω̃十_case2_func_raw = sp.lambdify([κ, ω0, γⁱⁿᵗ], ω̃十_case2_expr, modules="numpy")
ω̃一_case2_func_raw = sp.lambdify([κ, ω0, γⁱⁿᵗ], ω̃一_case2_expr, modules="numpy")


# 包装函数以处理复数
def ω̃十_case2_func(κ_val, ω0_val, γⁱⁿᵗ_val):
    # 手动计算，确保使用复数平方根
    Λ_val = np.sqrt((-(γⁱⁿᵗ_val**2) + 4 * κ_val**2) + 0j) / 2
    return ω0_val - 1j * γⁱⁿᵗ_val / 2 + Λ_val


def ω̃一_case2_func(κ_val, ω0_val, γⁱⁿᵗ_val):
    # 手动计算，确保使用复数平方根
    Λ_val = np.sqrt((-(γⁱⁿᵗ_val**2) + 4 * κ_val**2) + 0j) / 2
    return ω0_val - 1j * γⁱⁿᵗ_val / 2 - Λ_val


# 计算数值结果
ω̃十_case1_vals = ω̃十_case1_func(κ_vals, ω0_val)
ω̃一_case1_vals = ω̃一_case1_func(κ_vals, ω0_val)

ω̃十_case2_vals = ω̃十_case2_func(κ_vals, ω0_val, γⁱⁿᵗ_val)
ω̃一_case2_vals = ω̃一_case2_func(κ_vals, ω0_val, γⁱⁿᵗ_val)

# 替换符号值
ω0_num = ω0_val
γⁱⁿᵗ_num = γⁱⁿᵗ_val

# 计算EP位置
κ_EP = γⁱⁿᵗ_num / 2

print("\n参数设置:")
print(f"  ω₀ = {ω0_num}")
print(f"  γⁱⁿᵗ = {γⁱⁿᵗ_num}")
print(f"  κ_EP = {κ_EP}")
print()

# %%
# ============================================================================
# 8. 可视化：复平面上的特征值演化
# ============================================================================

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 子图1：无本征损耗（diabolic point）
ax1 = axes[0]
ax1.plot(
    np.real(ω̃十_case1_vals),
    np.imag(ω̃十_case1_vals),
    "b-",
    linewidth=2,
    label="ω₊",
)
ax1.plot(
    np.real(ω̃一_case1_vals),
    np.imag(ω̃一_case1_vals),
    "r-",
    linewidth=2,
    label="ω₋",
)
ax1.scatter(
    [ω0_num],
    [0],
    color="black",
    s=100,
    zorder=5,
    marker="o",
    label="Diabolic Point",
)
ax1.set_xlabel("Re(ω)", fontsize=12)
ax1.set_ylabel("Im(ω)", fontsize=12)
ax1.set_title("情况1: 两个S-BIC (γⁱⁿᵗ=0)\nDiabolic Point", fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color="k", linewidth=0.5)
ax1.axvline(x=ω0_num, color="k", linewidth=0.5)

# 子图2：有本征损耗（EP-BIC）
ax2 = axes[1]
ax2.plot(
    np.real(ω̃十_case2_vals),
    np.imag(ω̃十_case2_vals),
    "b-",
    linewidth=2,
    label="ω₊",
)
ax2.plot(
    np.real(ω̃一_case2_vals),
    np.imag(ω̃一_case2_vals),
    "r-",
    linewidth=2,
    label="ω₋",
)

# 标记EP点
ω_EP = ω0_num - 1j * γⁱⁿᵗ_num / 2
ax2.scatter(
    [np.real(ω_EP)],
    [np.imag(ω_EP)],
    color="green",
    s=150,
    zorder=5,
    marker="*",
    label=f"EP-BIC (κ={κ_EP:.3f})",
)

ax2.set_xlabel("Re(ω)", fontsize=12)
ax2.set_ylabel("Im(ω)", fontsize=12)
ax2.set_title(f"情况2: EP-BIC (γⁱⁿᵗ={γⁱⁿᵗ_num})\n平方根色散", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color="k", linewidth=0.5)
ax2.axvline(x=ω0_num, color="k", linewidth=0.5)

plt.tight_layout()
plt.show()

print("可视化完成！")
print("\n观察:")
print("  - 左图: diabolic point处特征值线性分裂")
print("  - 右图: EP-BIC处特征值呈现平方根色散（bulk Fermi arc）")
print()

# %%
# ============================================================================
# 9. 可视化：特征值实部和虚部随κ的变化（对应文章图1）
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ===== 情况1：无本征损耗 =====
# 实部
ax1 = axes[0, 0]
ax1.plot(κ_vals, np.real(ω̃十_case1_vals), "b-", linewidth=2, label="Re(ω+)")
ax1.plot(κ_vals, np.real(ω̃一_case1_vals), "r-", linewidth=2, label="Re(ω-)")
ax1.set_xlabel("κ", fontsize=12)
ax1.set_ylabel("Re(ω)", fontsize=12)
ax1.set_title("情况1: S-BIC - 谐振频率", fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 虚部
ax2 = axes[1, 0]
ax2.plot(κ_vals, -np.imag(ω̃十_case1_vals), "b-", linewidth=2, label="-Im(ω+) = γ+")
ax2.plot(
    κ_vals,
    -np.imag(ω̃一_case1_vals),
    "r-",
    linewidth=2,
    label="-Im(ω-) = γ-",
)
ax2.set_xlabel("κ", fontsize=12)
ax2.set_ylabel("损耗 γ", fontsize=12)
ax2.set_title("情况1: S-BIC - 损耗", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([-0.01, 0.05])

# ===== 情况2：有本征损耗（EP-BIC） =====
# 实部
ax3 = axes[0, 1]
ax3.plot(κ_vals, np.real(ω̃十_case2_vals), "b-", linewidth=2, label="Re(ω+)")
ax3.plot(κ_vals, np.real(ω̃一_case2_vals), "r-", linewidth=2, label="Re(ω-)")
ax3.axvline(
    x=κ_EP,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"EP (κ={κ_EP:.3f})",
)
ax3.set_xlabel("κ", fontsize=12)
ax3.set_ylabel("Re(ω)", fontsize=12)
ax3.set_title("情况2: EP-BIC - 谐振频率", fontsize=13)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, 0.2])  # 设置x轴范围与左侧子图一致

# 虚部
ax4 = axes[1, 1]
ax4.plot(κ_vals, -np.imag(ω̃十_case2_vals), "b-", linewidth=2, label="-Im(ω+) = γ+")
ax4.plot(
    κ_vals,
    -np.imag(ω̃一_case2_vals),
    "r-",
    linewidth=2,
    label="-Im(ω-) = γ-",
)
ax4.axvline(
    x=κ_EP,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"EP (κ={κ_EP:.3f})",
)
ax4.set_xlabel("κ", fontsize=12)
ax4.set_ylabel("损耗 γ", fontsize=12)
ax4.set_title("情况2: EP-BIC - 损耗", fontsize=13)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n分析（对应文章图1）:")
print("  情况1 (左列):")
print("    - 实部: 线性分裂，间隙 ∝ |κ|")
print("    - 虚部: 始终为0（无辐射损耗）")
print("  情况2 (右列):")
print("    - 实部: κ < κ_EP时准简并（bulk Fermi arc）")
print("    -       κ > κ_EP时平方根分裂")
print("    - 虚部: κ < κ_EP时平方根分裂")
print("    -       κ > κ_EP时准简并")
print("    - 在EP处: 两个特征值完全合并")
print()

# %%
# ============================================================================
# 10. 计算特征向量（验证EP处的合并）
# ============================================================================

print("=" * 70)
print("特征向量分析")
print("=" * 70)

# 在EP点附近计算特征向量
κ_test_vals = [κ_EP - 0.01, κ_EP, κ_EP + 0.01]

for κ_test in κ_test_vals:
    H_numeric = H_case2.subs({ω0: ω0_num, γⁱⁿᵗ: γⁱⁿᵗ_num, κ: κ_test})

    # 转换为复数矩阵
    H_numeric_complex = np.array(H_numeric.tolist(), dtype=complex)

    # 计算特征值和特征向量
    eigvals, eigvecs = np.linalg.eig(H_numeric_complex)

    print(f"\nκ = {κ_test:.4f}:")
    print(f"  特征值: {eigvals[0]:.6f}, {eigvals[1]:.6f}")
    print(f"  特征向量1: [{eigvecs[0, 0]:.4f}, {eigvecs[1, 0]:.4f}]")
    print(f"  特征向量2: [{eigvecs[0, 1]:.4f}, {eigvecs[1, 1]:.4f}]")

    # 计算特征向量的内积（归一化后）
    v1 = eigvecs[:, 0] / np.linalg.norm(eigvecs[:, 0])
    v2 = eigvecs[:, 1] / np.linalg.norm(eigvecs[:, 1])
    inner_product = np.abs(np.vdot(v1, v2))
    print(f"  特征向量内积: {inner_product:.6f} (EP处应接近1)")

print("\n结论: 在EP处，特征值和特征向量同时合并，形成EP-BIC")
print("=" * 70)
