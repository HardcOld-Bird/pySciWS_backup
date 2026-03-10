"""分析COMSOL仿真得到的声压数据，探究输入输出之间的传递关系。

该脚本分析32个探针的复数声压数据，其中：
- 探针1-8与探针9-16一一对应（第一组输入输出对）
- 探针17-24与探针25-32一一对应（第二组输入输出对）
探究输出和输入之间是否存在简单的数量关系（如常数传递函数、恒定幅值比、恒定相位差等）。
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

matplotlib.use("Agg")  # 非交互式后端，避免GUI阻塞

from config.matplotlib_config import setup_chinese_fonts

# 导入项目通用的matplotlib配置
setup_chinese_fonts()

# %%
# ============================================================================
# 1. 数据准备
# ============================================================================

# 原始复数声压数据（探针1-32）
pressure_data = np.array([
    -51.51401285499149+14.357969450837878j,   # 探针1
    80.18518666945063-51.26119200751582j,      # 探针2
    -130.7528931453355+80.11119579315654j,     # 探针3
    171.65241027044797-125.44341691677063j,    # 探针4
    -185.38583737782537+141.73726059849398j,   # 探针5
    151.78780444800913-134.9469348138768j,     # 探针6
    -89.7722728426346+107.47782397089708j,     # 探针7
    26.599129288774286-62.018161270595215j,    # 探针8
    -0.25647009202352483+0.48410898048154766j, # 探针9
    -0.9175481285182243+1.9258536446089962j,   # 探针10
    7.1622930309961115+0.965932861573212j,     # 探针11
    -4.883665545649132-8.9464252170797j,       # 探针12
    3.418331815833865+7.571629625285171j,      # 探针13
    -4.271156286807189-4.176866139392936j,     # 探针14
    4.096371951628152+0.5390426541358828j,     # 探针15
    -2.1870921076283794+2.149334786479364j,    # 探针16
    -1.208750433913774+4.89990927326868j,      # 探针17
    2.371595493163145-6.551342258303415j,      # 探针18
    -2.8345868687673743+5.460104542653479j,    # 探针19
    -0.9078508988272401-1.908221652963229j,    # 探针20
    6.5060938692189865-5.330177476490223j,     # 探针21
    -6.5559791570954244+16.346216065397567j,   # 探针22
    -4.179711265828257-15.8569503696368j,      # 探针23
    4.590951747462963+6.311300769321415j,      # 探针24
    0.46005228148589294-0.7187572727791289j,   # 探针25
    -0.3714631502817595-0.3405727243717863j,   # 探针26
    0.5095213852510975+0.6336037064965777j,    # 探针27
    -0.46909142573951595-0.5280998177489071j,  # 探针28
    0.8665877530871038-0.3407579880525709j,    # 探针29
    0.47907700017113025+2.840278418785463j,    # 探针30
    -3.977232611668165+1.2675757069684197j,    # 探针31
    -0.9330164259566063-1.9031653827900932j    # 探针32
])

# 分组：输出和输入
# 第一组：探针1-8为输出，探针9-16为输入
output_1 = pressure_data[0:8]
input_1 = pressure_data[8:16]

# 第二组：探针17-24为输出，探针25-32为输入
output_2 = pressure_data[16:24]
input_2 = pressure_data[24:32]

# %%
# ============================================================================
# 2. 计算传递函数（复数比值）
# ============================================================================

# 传递函数 H = 输出 / 输入
H_1 = output_1 / input_1
H_2 = output_2 / input_2

print("=" * 70)
print("传递函数分析（复数形式）")
print("=" * 70)
print("\n第一组传递函数 H_1 (探针1-8 / 探针9-16):")
for i, h in enumerate(H_1, 1):
    print(f"  H_{i} = {h.real:+.4f} {h.imag:+.4f}j")

print("\n第二组传递函数 H_2 (探针17-24 / 探针25-32):")
for i, h in enumerate(H_2, 1):
    print(f"  H_{i} = {h.real:+.4f} {h.imag:+.4f}j")

# 统计特性
print(f"\n第一组传递函数统计:")
print(f"  均值: {np.mean(H_1).real:+.4f} {np.mean(H_1).imag:+.4f}j")
print(f"  标准差: {np.std(H_1):.4f}")

print(f"\n第二组传递函数统计:")
print(f"  均值: {np.mean(H_2).real:+.4f} {np.mean(H_2).imag:+.4f}j")
print(f"  标准差: {np.std(H_2):.4f}")

# %%
# ============================================================================
# 3. 计算幅值比和相位差
# ============================================================================

# 幅值
amp_output_1 = np.abs(output_1)
amp_input_1 = np.abs(input_1)
amp_ratio_1 = amp_output_1 / amp_input_1

amp_output_2 = np.abs(output_2)
amp_input_2 = np.abs(input_2)
amp_ratio_2 = amp_output_2 / amp_input_2

# 相位（弧度）
phase_output_1 = np.angle(output_1)
phase_input_1 = np.angle(input_1)
phase_diff_1 = phase_output_1 - phase_input_1

phase_output_2 = np.angle(output_2)
phase_input_2 = np.angle(input_2)
phase_diff_2 = phase_output_2 - phase_input_2

print("\n" + "=" * 70)
print("幅值比和相位差分析")
print("=" * 70)
print("\n第一组:")
for i in range(8):
    print(f"  对{i+1}: 幅值比 = {amp_ratio_1[i]:.4f}, 相位差 = {np.degrees(phase_diff_1[i]):+.2f}°")

print(f"\n第一组统计:")
print(f"  幅值比均值: {np.mean(amp_ratio_1):.4f}, 标准差: {np.std(amp_ratio_1):.4f}")
print(f"  相位差均值: {np.degrees(np.mean(phase_diff_1)):+.2f}°, 标准差: {np.degrees(np.std(phase_diff_1)):.2f}°")

print("\n第二组:")
for i in range(8):
    print(f"  对{i+1}: 幅值比 = {amp_ratio_2[i]:.4f}, 相位差 = {np.degrees(phase_diff_2[i]):+.2f}°")

print(f"\n第二组统计:")
print(f"  幅值比均值: {np.mean(amp_ratio_2):.4f}, 标准差: {np.std(amp_ratio_2):.4f}")
print(f"  相位差均值: {np.degrees(np.mean(phase_diff_2)):+.2f}°, 标准差: {np.degrees(np.std(phase_diff_2)):.2f}°")



# %%
# ============================================================================
# 4. 可视化
# ============================================================================

pair_labels = [f"对{i+1}" for i in range(8)]
x = np.arange(8)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("声压传递关系分析（f = 3430 Hz）", fontsize=14, fontweight='bold')

# --- 第一组 ---
ax = axes[0, 0]
ax.bar(x, amp_ratio_1, color='steelblue', alpha=0.8)
ax.axhline(np.mean(amp_ratio_1), color='red', linestyle='--', label=f'均值={np.mean(amp_ratio_1):.2f}')
ax.set_title("第一组：幅值比（输出/输入）")
ax.set_xticks(x); ax.set_xticklabels(pair_labels)
ax.set_ylabel("幅值比"); ax.legend()

ax = axes[0, 1]
ax.bar(x, np.degrees(phase_diff_1), color='darkorange', alpha=0.8)
ax.axhline(np.degrees(np.mean(phase_diff_1)), color='red', linestyle='--',
           label=f'均值={np.degrees(np.mean(phase_diff_1)):.1f}°')
ax.set_title("第一组：相位差（输出-输入）")
ax.set_xticks(x); ax.set_xticklabels(pair_labels)
ax.set_ylabel("相位差 (°)"); ax.legend()

ax = axes[0, 2]
ax.scatter(H_1.real, H_1.imag, c=range(8), cmap='viridis', s=100, zorder=5)
for i, (r, im) in enumerate(zip(H_1.real, H_1.imag)):
    ax.annotate(f"对{i+1}", (r, im), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.axhline(0, color='gray', linewidth=0.5); ax.axvline(0, color='gray', linewidth=0.5)
ax.set_title("第一组：传递函数复平面分布")
ax.set_xlabel("实部"); ax.set_ylabel("虚部")

# --- 第二组 ---
ax = axes[1, 0]
ax.bar(x, amp_ratio_2, color='steelblue', alpha=0.8)
ax.axhline(np.mean(amp_ratio_2), color='red', linestyle='--', label=f'均值={np.mean(amp_ratio_2):.2f}')
ax.set_title("第二组：幅值比（输出/输入）")
ax.set_xticks(x); ax.set_xticklabels(pair_labels)
ax.set_ylabel("幅值比"); ax.legend()

ax = axes[1, 1]
ax.bar(x, np.degrees(phase_diff_2), color='darkorange', alpha=0.8)
ax.axhline(np.degrees(np.mean(phase_diff_2)), color='red', linestyle='--',
           label=f'均值={np.degrees(np.mean(phase_diff_2)):.1f}°')
ax.set_title("第二组：相位差（输出-输入）")
ax.set_xticks(x); ax.set_xticklabels(pair_labels)
ax.set_ylabel("相位差 (°)"); ax.legend()

ax = axes[1, 2]
ax.scatter(H_2.real, H_2.imag, c=range(8), cmap='viridis', s=100, zorder=5)
for i, (r, im) in enumerate(zip(H_2.real, H_2.imag)):
    ax.annotate(f"对{i+1}", (r, im), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.axhline(0, color='gray', linewidth=0.5); ax.axvline(0, color='gray', linewidth=0.5)
ax.set_title("第二组：传递函数复平面分布")
ax.set_xlabel("实部"); ax.set_ylabel("虚部")

plt.tight_layout()

import os
save_dir = Path(__file__).parent.parent.parent / "storage" / "管槽增益特性探究" / "plots"
os.makedirs(save_dir, exist_ok=True)
save_path = save_dir / "传递函数分析.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n图表已保存至 {save_path}")



# %%
# ============================================================================
# 5. 差值传递函数分析：(输出 - 输入) / 输入
# ============================================================================

# 计算差值及其与输入的比值
diff_1 = output_1 - input_1           # 第一组：输出 - 输入
diff_2 = output_2 - input_2           # 第二组：输出 - 输入

H_diff_1 = diff_1 / input_1           # 差值传递函数
H_diff_2 = diff_2 / input_2

# 幅值比和相位差
diff_amp_ratio_1 = np.abs(diff_1) / np.abs(input_1)
diff_amp_ratio_2 = np.abs(diff_2) / np.abs(input_2)
diff_phase_1 = np.angle(diff_1) - np.angle(input_1)
diff_phase_2 = np.angle(diff_2) - np.angle(input_2)

print("\n" + "=" * 70)
print("差值传递函数分析：(输出 - 输入) / 输入")
print("=" * 70)

print("\n第一组 H_diff = (输出 - 输入) / 输入:")
for i, h in enumerate(H_diff_1, 1):
    print(f"  H_diff_{i} = {h.real:+.4f} {h.imag:+.4f}j  |H_diff|={abs(h):.4f}")
print(f"  复数均值: {np.mean(H_diff_1).real:+.4f} {np.mean(H_diff_1).imag:+.4f}j")
print(f"  幅值比均值: {np.mean(diff_amp_ratio_1):.4f}, 标准差: {np.std(diff_amp_ratio_1):.4f}")
print(f"  相位差均值: {np.degrees(np.mean(diff_phase_1)):+.2f}°, 标准差: {np.degrees(np.std(diff_phase_1)):.2f}°")

print("\n第二组 H_diff = (输出 - 输入) / 输入:")
for i, h in enumerate(H_diff_2, 1):
    print(f"  H_diff_{i} = {h.real:+.4f} {h.imag:+.4f}j  |H_diff|={abs(h):.4f}")
print(f"  复数均值: {np.mean(H_diff_2).real:+.4f} {np.mean(H_diff_2).imag:+.4f}j")
print(f"  幅值比均值: {np.mean(diff_amp_ratio_2):.4f}, 标准差: {np.std(diff_amp_ratio_2):.4f}")
print(f"  相位差均值: {np.degrees(np.mean(diff_phase_2)):+.2f}°, 标准差: {np.degrees(np.std(diff_phase_2)):.2f}°")

# %%
# ============================================================================
# 6. 差值传递函数可视化
# ============================================================================

fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9))
fig2.suptitle("差值传递函数分析：(输出 - 输入) / 输入（f = 3430 Hz）", fontsize=14, fontweight='bold')

# --- 第一组 ---
ax = axes2[0, 0]
ax.bar(x, diff_amp_ratio_1, color='steelblue', alpha=0.8)
ax.axhline(np.mean(diff_amp_ratio_1), color='red', linestyle='--',
           label=f'均值={np.mean(diff_amp_ratio_1):.2f}')
ax.set_title("第一组：|差值| / |输入|")
ax.set_xticks(x); ax.set_xticklabels(pair_labels)
ax.set_ylabel("幅值比"); ax.legend()

ax = axes2[0, 1]
ax.bar(x, np.degrees(diff_phase_1), color='darkorange', alpha=0.8)
ax.axhline(np.degrees(np.mean(diff_phase_1)), color='red', linestyle='--',
           label=f'均值={np.degrees(np.mean(diff_phase_1)):.1f}°')
ax.set_title("第一组：∠差值 - ∠输入")
ax.set_xticks(x); ax.set_xticklabels(pair_labels)
ax.set_ylabel("相位差 (°)"); ax.legend()

ax = axes2[0, 2]
ax.scatter(H_diff_1.real, H_diff_1.imag, c=range(8), cmap='viridis', s=100, zorder=5)
for i, (r, im) in enumerate(zip(H_diff_1.real, H_diff_1.imag)):
    ax.annotate(f"对{i+1}", (r, im), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.axhline(0, color='gray', linewidth=0.5); ax.axvline(0, color='gray', linewidth=0.5)
ax.set_title("第一组：H_diff 复平面分布")
ax.set_xlabel("实部"); ax.set_ylabel("虚部")

# --- 第二组 ---
ax = axes2[1, 0]
ax.bar(x, diff_amp_ratio_2, color='steelblue', alpha=0.8)
ax.axhline(np.mean(diff_amp_ratio_2), color='red', linestyle='--',
           label=f'均值={np.mean(diff_amp_ratio_2):.2f}')
ax.set_title("第二组：|差值| / |输入|")
ax.set_xticks(x); ax.set_xticklabels(pair_labels)
ax.set_ylabel("幅值比"); ax.legend()

ax = axes2[1, 1]
ax.bar(x, np.degrees(diff_phase_2), color='darkorange', alpha=0.8)
ax.axhline(np.degrees(np.mean(diff_phase_2)), color='red', linestyle='--',
           label=f'均值={np.degrees(np.mean(diff_phase_2)):.1f}°')
ax.set_title("第二组：∠差值 - ∠输入")
ax.set_xticks(x); ax.set_xticklabels(pair_labels)
ax.set_ylabel("相位差 (°)"); ax.legend()

ax = axes2[1, 2]
ax.scatter(H_diff_2.real, H_diff_2.imag, c=range(8), cmap='viridis', s=100, zorder=5)
for i, (r, im) in enumerate(zip(H_diff_2.real, H_diff_2.imag)):
    ax.annotate(f"对{i+1}", (r, im), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.axhline(0, color='gray', linewidth=0.5); ax.axvline(0, color='gray', linewidth=0.5)
ax.set_title("第二组：H_diff 复平面分布")
ax.set_xlabel("实部"); ax.set_ylabel("虚部")

plt.tight_layout()

save_path2 = save_dir / "差值传递函数分析.png"
plt.savefig(save_path2, dpi=150, bbox_inches='tight')
print(f"\n图表已保存至 {save_path2}")
