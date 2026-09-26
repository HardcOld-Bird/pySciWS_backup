# pyright: basic
"""
# 完整有源超表面实验脚本

该脚本用于按照标准流程执行完整的演化测量和扫场实验。
"""

import numpy as np

from pysci.research.gain_ep.experiment.analyze import (
    get_sine,
    plot_waveform,
)
from pysci.research.gain_ep.experiment.config.exp_config import (
    ao_channels_static_L,
    ao_channels_static_r,
    best_frequency,
    grid,
    root_folder,
    sampling_info,
    sweep_ai_channel,
)
from pysci.research.gain_ep.experiment.use import (
    SweeperCore,
)

# 声源输出波形复振幅数组（与 ao_channels_static_L 等长）
static_cca: np.ndarray = np.full(
    len(ao_channels_static_L),
    0.01 + 0j,
    dtype=np.complex128,
)
# 创建双侧输出波形
static_output_waveform_L = get_sine(
    sampling_info=sampling_info,
    frequency=best_frequency,
    channel_names=ao_channels_static_L,
    channel_complex_amplitudes=static_cca,
    full_cycle=True,
)
static_output_waveform_r = get_sine(
    sampling_info=sampling_info,
    frequency=best_frequency,
    channel_names=ao_channels_static_r,
    channel_complex_amplitudes=static_cca,
    full_cycle=True,
)

# %%
# ============================================================================
# 1. 扫场传声器归位
# ============================================================================
# - 连接步进电机
# - 确保扫场传声器行程范围内无障碍物

# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    ao_channels_static=("default_ao",),
)
# 传声器归位
swp.move_to(311.0, 311.0)
# 删除对象，避免干扰后续实验流程
del swp

# %%
# ============================================================================
# 2. 测量样件M+左入射时的逆反射场
# ============================================================================
# - 正向摆放样件至M+状态
# - 打开扫场窗口（移除附近吸声棉），确保传声器行程范围内无障碍物
# - 开启步进电机
# - 开启NI功放机箱的声源通道

# 定义结果文件夹
result_folder = root_folder + "\\5_M+_L_1st_sweep"
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_static,
    static_output_waveform=static_output_waveform,
    point_list=grid,
)
# 扫场传声器归位
swp.move_to(1.0, 1.0)
# 步进电机校准
swp.calib()
# 开始扫场测量（阻塞式）
swp.sweep_blocking(
    result_folder=result_folder,
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
# 清理资源
del swp

# %%
# ============================================================================
# 3. 测量样件M-左入射时的逆反射场
# ============================================================================
# - 将样件翻面至M-状态

# 定义结果文件夹
result_folder = root_folder + "\\14_M-_l_1st_sweep"
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_static,
    static_output_waveform=static_output_waveform,
    point_list=grid,
)
# 扫场传声器归位
swp.move_to(1.0, 1.0)
# 步进电机校准
swp.calib()
# 开始扫场测量（阻塞式）
swp.sweep_blocking(
    result_folder=result_folder,
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
# 清理资源
del swp

# %%
# ============================================================================
# 4. 测量样件M-右入射时的镜面反射场
# ============================================================================
# - 将声源更换至右入射位置r

# 定义结果文件夹
result_folder = root_folder + "\\12_M-_R_1st_sweep"
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_static,
    static_output_waveform=static_output_waveform,
    point_list=grid,
)
# 扫场传声器归位
swp.move_to(1.0, 1.0)
# 步进电机校准
swp.calib()
# 开始扫场测量（阻塞式）
swp.sweep_blocking(
    result_folder=result_folder,
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
# 清理资源
del swp

# %%
# ============================================================================
# 5. 测量样件M+右入射时的镜面反射场
# ============================================================================
# - 将样件翻面至M+状态
# - 打开扫场窗口（移除附近吸声棉），确保传声器行程范围内无障碍物
# - 开启步进电机
# - 开启NI功放机箱的所有通道

# 定义结果文件夹
result_folder = root_folder + "\\10_M+_r_1st_sweep"
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_static,
    static_output_waveform=static_output_waveform,
    point_list=grid,
)
# 扫场传声器归位
swp.move_to(1.0, 1.0)
# 步进电机校准
swp.calib()
# 开始扫场测量（阻塞式）
swp.sweep_blocking(
    result_folder=result_folder,
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
# 清理资源
del swp

# %%
# ============================================================================
# 11. 测量背景场（仅声源）
# ============================================================================
# - 移除超表面样件，并尽可能填充吸声棉，减少反射声
# - 开启NI功放机箱的声源通道（可关闭所有反馈通道）

# ==========【左入射的逆反射】==========
# 定义结果文件夹
result_folder = root_folder + "\\9_X_L_bg_sweep"
# 绘制波形
_ = plot_waveform(
    static_output_waveform_L,
    save_path=result_folder + "\\evo_result_L_bg_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    static_output_waveform_L,
    save_path=result_folder + "\\evo_result_L_bg_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    ao_channels_static=ao_channels_static_L,
    static_output_waveform=static_output_waveform_L,
    point_list=grid,
)
# 扫场传声器归位
swp.move_to(1.0, 1.0)
# 步进电机校准
swp.calib()
# 开始扫场测量（阻塞式）
swp.sweep_blocking(
    result_folder=result_folder,
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
# 清理资源
del swp

# ==========【右入射的镜面反射】==========
# 定义结果文件夹
result_folder = root_folder + "\\10_X_R_bg_sweep"
# 绘制波形
_ = plot_waveform(
    static_output_waveform_r,
    save_path=result_folder + "\\evo_result_R_bg_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    static_output_waveform_r,
    save_path=result_folder + "\\evo_result_R_bg_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    ao_channels_static=ao_channels_static_r,
    static_output_waveform=static_output_waveform_r,
    point_list=grid,
)
# 扫场传声器归位
swp.move_to(1.0, 1.0)
# 开始扫场测量（阻塞式）
swp.sweep_blocking(
    result_folder=result_folder,
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
# 清理资源
del swp
