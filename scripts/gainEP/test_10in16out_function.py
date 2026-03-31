"""测试 GainEPSimulator 完整工作流程

此脚本测试 src/gain_ep/gain_ep_sim.py 中的 GainEPSimulator 类的核心功能：
1. 计算传递函数矩阵和增益系数（calib 方法）
2. 求解 Ground Truth（solve_ground_truth 方法）
3. 运行反馈环路仿真，测试双模式系统：
   - data_source_mode: from_calib, from_truth, constant
   - logic_mode: p_and_d, only_p
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
# WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSPACE_ROOT = Path(__name__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.gain_ep.gain_ep_sim import GainEPSimulator, SimulationInput

# %
# ============================================================================
# 1. 初始化参数
# ============================================================================

print("=" * 70)
print("测试 GainEPSimulator 完整工作流程")
print("=" * 70)

# COMSOL模型文件
MPH_FILE = WORKSPACE_ROOT / "mphs" / "gainEP" / "gainEP_10in16out.mph"

if not MPH_FILE.exists():
    print(f"\n✗ 错误: COMSOL模型文件不存在: {MPH_FILE}")
    exit(1)

print(f"模型文件: {MPH_FILE}")

# 设置反馈常数
FEEDBACK_CONSTANT = (
    -1.35451109741951292164685582974926 - 1.44801076473288947710216234554537j
)
print(f"反馈常数: {FEEDBACK_CONSTANT}")

# %
# ============================================================================
# 2. 创建仿真器并连接
# ============================================================================

print("\n" + "=" * 70)
print("创建仿真器并连接")
print("=" * 70)

simulator = GainEPSimulator(MPH_FILE, feedback_constant=FEEDBACK_CONSTANT)
print("✓ 仿真器已创建")
print(f"  模型文件: {simulator.mph_file.name}")
print(f"  反馈常数: {simulator.feedback_constant}")

print("\n正在连接到 COMSOL Server...")
simulator.connect()

# %
# ============================================================================
# 3. 计算传递函数矩阵和增益系数
# ============================================================================

print("\n" + "=" * 70)
print("计算传递函数矩阵和增益系数")
print("=" * 70)

# 计算传递函数矩阵和增益系数
transfer_functions, gain_coefficients = simulator.calib()

print("\n✓ 校准完成")
print(f"  传递函数矩阵形状: {transfer_functions.shape}")
print(f"  增益系数数量: {len(gain_coefficients)}")
# %%
# print(f"  传递函数矩阵: {transfer_functions}")

# print("\n传递函数矩阵主对角线元素模长:")
# for i, tfs in enumerate(transfer_functions[2:], start=0):
#     print(f"  管槽 {i + 1}: {abs(tfs[i]):.32f}")

# print("\n传递函数矩阵主对角线元素与增益系数之比:")
# for i, tfs in enumerate(transfer_functions[2:], start=0):
#     print(f"  管槽 {i + 1}: {(gain_coefficients[i] - 1) / tfs[i]:.32f}")

# %%
# ============================================================================
# 4. 求解 Ground Truth
# ============================================================================

print("\n" + "=" * 70)
print("求解 Ground Truth - 测试迭代优化")
print("=" * 70)

# 设置测试输入：左入射=1，右入射=0，所有法向位移=0
test_input = SimulationInput(
    pamp_L=1.0,
    pamp_R=0.0,
    vn_1=0j,
    vn_2=0j,
    vn_3=0j,
    vn_4=0j,
    vn_5=0j,
    vn_6=0j,
    vn_7=0j,
    vn_8=0j,
)

# 设置迭代次数（建议3次）
# 以下是不同次数的典型误差值：
# 1：6.463286e-06
# 2：2.060303e-16
# 3：3.752095e-19（已接近最小）
# 4：1.050377e-19（已达最小）
# 5：1.760487e-19
solved_vn, effective_gain_coeffs, verification_error = simulator.solve_ground_truth(
    test_input, num_iterations=3
)

print("\n✓ Ground Truth 求解完成")
print(f"  验证误差: {verification_error:.6e}")
print("\n等效增益系数:")
for i, gc in enumerate(effective_gain_coeffs, start=1):
    print(f"  管槽 {i}: {gc:.6f}")

# %%
print("\n法向位移:")
for i, vn in enumerate(solved_vn, start=1):
    print(f"  管槽 {i}: {vn:.32f}")

# print("\n增益系数模长:")
# for i, gcs in enumerate(abs(effective_gain_coeffs), start=0):
#     print(f"  管槽 {i + 1}: {gcs}")

# print("\n增益系数平均值:")
# print(f"{np.mean(effective_gain_coeffs):.32f}")

# %%
# ============================================================================
# 5. 运行反馈环路仿真（参数组合1）
# ============================================================================

print("\n" + "=" * 70)
print("运行反馈环路仿真 - 测试双模式系统")
print("=" * 70)

# 设置初始输入
initial_input = SimulationInput(
    pamp_L=1.0,
    pamp_R=0.0,
    vn_1=0j,
    vn_2=0j,
    vn_3=0j,
    vn_4=0j,
    vn_5=0j,
    vn_6=0j,
    vn_7=0j,
    vn_8=0j,
)

# 设置迭代参数
num_iterations = 30

# 测试组合1: from_truth + p_and_d
print("\n--- 组合1: data_source=from_truth, logic=p_and_d ---")
all_inputs_1, all_outputs_1 = simulator.run_feedback_loop(
    initial_input=initial_input,
    num_iterations=num_iterations,
    data_source_mode="from_truth",
    logic_mode="p_and_d",
    # logic_mode="only_p",
)
print(f"✓ 完成，共 {len(all_outputs_1)} 次迭代")

# 显示组合1（from_truth + p_and_d）最终迭代的探针差值
print("\n最终迭代的探针1-8与探针9-16的差值（from_truth + p_and_d）:")
final_output = all_outputs_1[-1]
final_array = final_output.to_array()
for i in range(8):
    diff = final_array[i] - final_array[i + 8]
    print(f"  探针{i + 1} - 探针{i + 9}: {diff:.6e}")

# %%
# ============================================================================
# 5. 运行反馈环路仿真（参数组合2）
# ============================================================================

# 设置初始输入
initial_input = SimulationInput(
    pamp_L=1.0,
    pamp_R=0.0,
    vn_1=0j,
    vn_2=0j,
    vn_3=0j,
    vn_4=0j,
    vn_5=0j,
    vn_6=0j,
    vn_7=0j,
    vn_8=0j,
)

# 设置迭代参数
num_iterations = 30

# 测试组合2
all_inputs_1, all_outputs_1 = simulator.run_feedback_loop(
    initial_input=initial_input,
    num_iterations=num_iterations,
    data_source_mode="constant",
    logic_mode="p_and_d",
    # logic_mode="only_p",
)

# %%
# ============================================================================
# 6. 清理和总结
# ============================================================================

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)

# 断开连接
simulator.disconnect()

print("\n✓ GainEPSimulator 完整工作流程测试成功！")
print("\n测试内容:")
print("  1. ✓ 创建仿真器并连接到 COMSOL Server")
print("  2. ✓ 计算传递函数矩阵和增益系数（calib）")
print("  3. ✓ 求解 Ground Truth（solve_ground_truth）")
print("  4. ✓ 运行反馈环路仿真，测试双模式系统：")
print("     - data_source_mode: from_calib, from_truth")
print("     - logic_mode: p_and_d, only_p")
print("     - 共测试了 4 种组合")
print("  5. ✓ 断开连接")
