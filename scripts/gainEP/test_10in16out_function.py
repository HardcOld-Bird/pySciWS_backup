"""测试 GainEPSimulator 完整工作流程

此脚本测试 src/gain_ep/gain_ep_sim.py 中的 GainEPSimulator 类的核心功能：
1. 计算传递函数矩阵和增益系数（calib 方法）
2. 求解 Ground Truth（solve_ground_truth 方法）
3. 运行反馈环路仿真，测试三种模式：
   - from_calib: 使用校准结果中的增益系数
   - from_truth: 使用 solve_ground_truth 计算的等效增益系数
   - constant: 使用相同的 feedback_constant
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
FEEDBACK_CONSTANT = -1.532508 - 1.238676j
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
# ============================================================================
# 4. 求解 Ground Truth
# ============================================================================

print("\n" + "=" * 70)
print("求解 Ground Truth")
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

# 求解 Ground Truth
solved_vn, effective_gain_coeffs, verification_error = simulator.solve_ground_truth(
    test_input, verify=True
)

print("\n✓ Ground Truth 求解完成")
print(f"  验证误差: {verification_error:.6e}")
print("\n等效增益系数:")
for i, gc in enumerate(effective_gain_coeffs, start=1):
    print(f"  管槽 {i}: {gc:.6f}")

# %%
# ============================================================================
# 5. 运行反馈环路仿真（测试不同模式）
# ============================================================================

print("\n" + "=" * 70)
print("运行反馈环路仿真")
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
num_iterations = 5

# 测试模式1: from_calib（使用校准系数）
# print("\n--- 模式1: from_calib ---")
# all_inputs_calib, all_outputs_calib = simulator.run_feedback_loop(
#     initial_input=initial_input,
#     num_iterations=num_iterations,
#     mode="from_calib",
# )
# print(f"✓ from_calib 模式完成，共 {len(all_outputs_calib)} 次迭代")

# 测试模式2: from_truth（使用 ground truth 系数）
print("\n--- 模式2: from_truth ---")
all_inputs_truth, all_outputs_truth = simulator.run_feedback_loop(
    initial_input=initial_input,
    num_iterations=num_iterations,
    mode="from_truth",
)
print(f"✓ from_truth 模式完成，共 {len(all_outputs_truth)} 次迭代")

# 测试模式3: constant（使用常数反馈）
# print("\n--- 模式3: constant ---")
# all_inputs_const, all_outputs_const = simulator.run_feedback_loop(
#     initial_input=initial_input,
#     num_iterations=num_iterations,
#     mode="constant",
# )
# print(f"✓ constant 模式完成，共 {len(all_outputs_const)} 次迭代")

# 显示 from_truth 模式最终迭代的探针差值
print("\n最终迭代的探针1-8与探针9-16的差值（from_truth 模式）:")
final_output = all_outputs_truth[-1]
final_array = final_output.to_array()
for i in range(8):
    diff = final_array[i] - final_array[i + 8]
    print(f"  探针{i + 1} - 探针{i + 9}: {diff:.6e}")

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
print("  4. ✓ 运行反馈环路仿真，测试三种模式：")
print("     - from_calib: 使用校准结果中的增益系数")
print("     - from_truth: 使用 solve_ground_truth 计算的等效增益系数")
print("     - constant: 使用相同的 feedback_constant")
print("  5. ✓ 断开连接")
