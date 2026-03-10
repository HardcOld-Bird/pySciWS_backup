"""
测试gainEP仿真和绘图函数

此脚本测试src/gain_ep.py中的函数，包括：
1. run_gain_ep_simulation - 运行参数扫描仿真并保存数据到硬盘
2. plot_scattering_matrix_2d - 从硬盘读取数据并绘制散射矩阵幅值的二维参数空间图
3. plot_eigenvalues_3d - 从硬盘读取数据并绘制S矩阵特征值的三维曲面图
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
WORKSPACE_ROOT = Path(__name__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))

# 导入matplotlib中文配置
from config.matplotlib_config import setup_chinese_fonts
from src.gain_ep.gain_ep_theory import (
    plot_eigenvalues_3d,
    plot_scattering_matrix_2d,
    run_gain_ep_simulation,
)

setup_chinese_fonts()

# ============================================================================
# 1. 初始化参数
# ============================================================================

# 初始化client为None
client = None

# COMSOL模型文件
MPH_FILE = WORKSPACE_ROOT / "mphs" / "gainEP" / "gainEP_basic.mph"

# 数据存储目录
DATA_DIR = WORKSPACE_ROOT / "storage" / "gainEP" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 图像输出目录
OUTPUT_DIR = WORKSPACE_ROOT / "storage" / "gainEP" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 仿真数据保存路径
DATA_SAVE_PATH = DATA_DIR / "simulation_result.npz"
# 仿真数据读取路径
DATA_LOAD_PATH = DATA_DIR / "big_data1.npz"

# %%
# ============================================================================
# 2. 运行仿真并保存数据
# ============================================================================

print("\n" + "=" * 70)
print("步骤1: 运行COMSOL仿真并保存数据")
print("=" * 70)

# 参数范围
CR_RANGE = (1.003, 0.00001, 1.005)  # (start, step, end)
CI_RANGE = (-0.074, 0.00001, -0.072)
FREQ = 3430.0  # Hz

try:
    print("\n正在连接COMSOL Server并运行仿真...")
    print("注意: 这可能需要几分钟时间...")

    # 运行仿真并保存数据到硬盘
    client = run_gain_ep_simulation(
        mph_file=MPH_FILE,
        cr_range=CR_RANGE,
        ci_range=CI_RANGE,
        save_path=DATA_SAVE_PATH,
        freq=FREQ,
        client=client,
    )

    print("\n✓ 仿真完成！数据已保存")

except Exception as e:
    print(f"\n✗ 仿真失败: {e}")
    import traceback

    traceback.print_exc()

    # 确保在出错时也断开连接
    if client is not None:
        try:
            client.disconnect()
            print("\n已断开COMSOL连接")
        except:
            pass
    exit(1)

# %%
# ============================================================================
# 3. 绘制散射矩阵二维参数空间图
# ============================================================================

print("\n" + "=" * 70)
print("步骤2: 绘制散射矩阵幅值分布")
print("=" * 70)

try:
    print("\n正在从硬盘读取数据并生成二维参数空间图...")

    save_path_2d = OUTPUT_DIR / "scattering_matrix_2d.png"
    fig = plot_scattering_matrix_2d(
        data_path=DATA_LOAD_PATH,
        save_path=save_path_2d,
        dpi=150,
    )

    print(f"✓ 图像已保存到: {save_path_2d}")

    # 显示图像
    plt.show()

except Exception as e:
    print(f"\n✗ 绘图失败: {e}")
    import traceback

    traceback.print_exc()

# %%
# ============================================================================
# 4. 绘制特征值三维曲面图（实部）
# ============================================================================

print("\n" + "=" * 70)
print("步骤3: 绘制S矩阵特征值三维曲面（实部）")
print("=" * 70)

try:
    print("\n正在从硬盘读取数据并生成三维曲面图（保存模式）...")

    save_path_3d = OUTPUT_DIR / "eigenvalues_3d.png"
    plotter = plot_eigenvalues_3d(
        data_path=DATA_LOAD_PATH,
        mode="Re",
        save_path=save_path_3d,
    )

    print(f"✓ 图像已保存到: {save_path_3d}")

    # 创建交互式版本（不保存）
    print("\n正在打开交互式3D窗口（实部）...")
    print("（关闭窗口以继续）")

    plotter_interactive = plot_eigenvalues_3d(
        data_path=DATA_LOAD_PATH,
        mode="Re",
        save_path=None,  # 不保存，只显示
    )
    plotter_interactive.show()

except Exception as e:
    print(f"\n✗ 绘图失败: {e}")
    import traceback

    traceback.print_exc()

# %%
# ============================================================================
# 5. 总结和清理
# ============================================================================

print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)
print(f"\n数据文件: {DATA_SAVE_PATH}")
print(f"\n所有输出文件已保存到: {OUTPUT_DIR}")
print("\n生成的文件:")
print(f"  1. {save_path_2d.name} - 散射矩阵幅值二维图")
print(f"  2. {save_path_3d.name} - 特征值三维曲面图")

# 断开COMSOL连接
if client is not None:
    try:
        print("\n正在断开COMSOL连接...")
        client.disconnect()
        print("✓ 已断开连接")
    except Exception as e:
        print(f"⚠ 断开连接时出现警告: {e}")
