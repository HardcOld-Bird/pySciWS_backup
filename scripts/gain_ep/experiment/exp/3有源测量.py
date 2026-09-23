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
from pysci.research.gain_ep.experiment.calib import CaliberFishNet, FrequencyOptimizer
from pysci.research.gain_ep.experiment.config.exp_config import (
    ai_channels,
    ao_channels,
    ao_channels_feedback,
    ao_channels_static_L,
    ao_channels_static_r,
    best_frequency,
    grid,
    root_folder,
    sampling_info,
    sweep_ai_channel,
)
from pysci.research.gain_ep.experiment.sim import SimScanner
from pysci.research.gain_ep.experiment.use import (
    Evolver,
    SweeperCore,
    load_evolved_waveform,
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
# swp.move_to(311.0, 311.0)
swp.move_to(1.0, 1.0)
# 删除对象，避免干扰后续实验流程
del swp

# %%
# ============================================================================
# 2. 进行频率校准
# ============================================================================
# - 在此之前，可选完成Anemone校准（8传声器）
# - 在此之前，建议完成Octopus校准（8扬声器）
# - 使用正向样件M+（通道数左小右大）
# - 封闭扫场窗口，充分布置吸声棉
# - 连接NI主机箱
# - 连接NI功放机箱，并开启声源通道

# 创建频率优化器
fo = FrequencyOptimizer(
    ai_channels=ai_channels,
    ao_channel_max=ao_channels_static_L[0],
    ao_channel_min=ao_channels_static_r[0],
    amplitude=1.0,
)
# 执行校准，结果存储在项目storage目录下
fo.optimize(
    max_iterations=20,
    tolerance=0.001,
)

# %%
# ============================================================================
# 3. 测量正向样件M+的 FishNet_TFData
# ============================================================================
# - 开启NI功放机箱的所有通道

# 创建渔网校准器
clb = CaliberFishNet(
    ai_channels=ai_channels,
    ao_channels=ao_channels,
    sampling_info=sampling_info,
    frequency=best_frequency,
    amplitude=5.0,  # 4.0及以下是线性的，5.0时TF则会有所衰减
)
# 定义结果文件夹
result_folder = root_folder + "\\1_FishNet_TFData"
# 执行校准
clb.calibrate(
    starts_num=1,
    chunks_per_start=1,
    result_folder=result_folder,
)

# %%
# ============================================================================
# 4. 基于 FishNet_TFData 进行参数扫描仿真
# ============================================================================
# - 开启COMSOL

# Floquet 增益系数中心值（复平面）
cr_center: float = 1.005
ci_center: float = -0.075
# 参数扫描半范围
half_scale: float = 0.03

# 指定TFData路径
tf_data_path = root_folder + "\\1_FishNet_TFData\\tf_data.pkl"
# 定义结果文件夹
result_folder = root_folder + "\\2_Sim_Scan"

# 创建仿真器
sim = SimScanner()
# 连接Server
sim.connect()
# 执行仿真
_ = sim.run_scan(
    cr_min=cr_center - half_scale,
    cr_max=cr_center + half_scale,
    ci_min=ci_center - half_scale,
    ci_max=ci_center + half_scale,
    res=50,
    # swap_static_ao=True,
    fishnet_tf_data_path=tf_data_path,
    result_folder=result_folder,
)

# %%
# ============================================================================
# 5. 对正向样件M+左入射（大响应）时的响应进行演化模拟
# ============================================================================

# 声源输出波形复振幅数组（与 ao_channels_static 等长）
static_cca: np.ndarray = np.full(
    len(ao_channels_static_L),
    1.0 + 0j,
    dtype=np.complex128,
)
# 创建输出波形
static_output_waveform = get_sine(
    sampling_info=sampling_info,
    frequency=best_frequency,
    channel_names=ao_channels_static_L,
    channel_complex_amplitudes=static_cca,
    full_cycle=True,
)
# 指定TFData路径
tf_data_path = root_folder + "\\1_FishNet_TFData\\tf_data.pkl"
# 指定仿真结果路径
sim_result_path = root_folder + "\\2_Sim_Scan\\scan_result.npz"
# 创建演化器
evo = Evolver(
    ai_channels=ai_channels,
    ao_channels_static=ao_channels_static_L,
    ao_channels_feedback=ao_channels_feedback,
    static_output_waveform=static_output_waveform,
    fishnet_tf_data_path=tf_data_path,
    sim_result_scan_path=sim_result_path,
)
# 定义结果文件夹
result_folder = root_folder + "\\3_evo_L"
# 计算理论结果
_ = evo.simulate(
    cr=1.01418,
    ci=-0.05969,
    mode="floquet_probes",
    ao_amplitude_limit=100,
    result_folder=result_folder,
)

# %%
# ============================================================================
# 6. 对正向样件M+左入射（大响应）时的响应进行演化测量
# ============================================================================

# 进行实际演化
_ = evo.evolve(
    cycles_num=25,
    ao_amplitude_limit=4.0,
    result_folder=result_folder,
)
# 读取演化后的波形
evo_result_L = load_evolved_waveform(
    file_path=result_folder + "\\evolved_waveform.pkl",
    segments=2,
)
# 绘制波形
_ = plot_waveform(
    evo_result_L,
    save_path=result_folder + "\\evolved_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_L,
    save_path=result_folder + "\\evolved_waveform_detail.png",
    zoom_factor=200,
)

# %%
# ============================================================================
# 7. 对正向样件M+右入射（小响应）时的响应进行演化模拟
# ============================================================================

# 声源输出波形复振幅数组（与 ao_channels_static 等长）
static_cca: np.ndarray = np.full(
    len(ao_channels_static_r),
    1.0 + 0j,
    dtype=np.complex128,
)
# 创建输出波形
static_output_waveform = get_sine(
    sampling_info=sampling_info,
    frequency=best_frequency,
    channel_names=ao_channels_static_r,
    channel_complex_amplitudes=static_cca,
    full_cycle=True,
)
# 指定TFData路径
tf_data_path = root_folder + "\\1_FishNet_TFData\\tf_data.pkl"
# 指定仿真结果路径
sim_result_path = root_folder + "\\2_Sim_Scan\\scan_result.npz"
# 创建演化器
evo = Evolver(
    ai_channels=ai_channels,
    ao_channels_static=ao_channels_static_r,
    ao_channels_feedback=ao_channels_feedback,
    static_output_waveform=static_output_waveform,
    fishnet_tf_data_path=tf_data_path,
    sim_result_scan_path=sim_result_path,
)
# 定义结果文件夹
result_folder = root_folder + "\\4_evo_r"
# 计算理论结果
_ = evo.simulate(
    cr=1.01418,
    ci=-0.05969,
    mode="floquet_probes",
    ao_amplitude_limit=100,
    result_folder=result_folder,
)

# %%
# ============================================================================
# 8. 对正向样件M+右入射（小响应）时的响应进行演化测量
# ============================================================================

# 进行实际演化
_ = evo.evolve(
    cycles_num=25,
    ao_amplitude_limit=4.0,
    result_folder=result_folder,
)
# 读取演化后的波形
evo_result_r = load_evolved_waveform(
    file_path=result_folder + "\\evolved_waveform.pkl",
    segments=2,
)
# 绘制波形
_ = plot_waveform(
    evo_result_r,
    save_path=result_folder + "\\evolved_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_r,
    save_path=result_folder + "\\evolved_waveform_detail.png",
    zoom_factor=200,
)

# %%
# ============================================================================
# 9. 测量样件M+的反射场（反馈阵列+声源）
# ============================================================================
# - 打开扫场窗口（移除附近吸声棉），确保传声器行程范围内无障碍物
# - 开启步进电机
# - 开启全部功放通道

# ==========【左入射（大响应）的逆反射】==========
# 定义结果文件夹
result_folder = root_folder + "\\5_M+_L_sweep"
# 创建扫场波形
evo_result_L = load_evolved_waveform(
    file_path=root_folder + "\\3_evo_L\\evolved_waveform.pkl",
    segments=2,
)
# 绘制波形
_ = plot_waveform(
    evo_result_L,
    save_path=result_folder + "\\evo_result_L_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_L,
    save_path=result_folder + "\\evo_result_L_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_feedback + ao_channels_static_L,
    static_output_waveform=evo_result_L,
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

# ==========【右入射（小响应）的镜面反射】==========
# 定义结果文件夹
result_folder = root_folder + "\\6_M+_r_sweep"
# 创建扫场波形
evo_result_r = load_evolved_waveform(
    file_path=root_folder + "\\4_evo_r\\evolved_waveform.pkl",
    segments=2,
)
# 绘制波形
_ = plot_waveform(
    evo_result_r,
    save_path=result_folder + "\\evo_result_r_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_r,
    save_path=result_folder + "\\evo_result_r_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_feedback + ao_channels_static_r,
    static_output_waveform=evo_result_r,
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

# %%
# ============================================================================
# 10. 测量样件M-的反射场（反馈阵列+声源）
# ============================================================================
# - 使用反向样件M-（通道数右小左大）

# ==========【右入射（大响应）的镜面反射】==========
# 定义结果文件夹
result_folder = root_folder + "\\7_M-_R_sweep"
# 创建扫场波形
evo_result_R = load_evolved_waveform(
    file_path=root_folder + "\\3_evo_L\\evolved_waveform.pkl",
    segments=2,
    rename_channel_dict={"PXI1Slot2/ao0": "PXI1Slot2/ao1"},
)
# 绘制波形
_ = plot_waveform(
    evo_result_R,
    save_path=result_folder + "\\evo_result_R_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_R,
    save_path=result_folder + "\\evo_result_R_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_feedback + ao_channels_static_r,
    static_output_waveform=evo_result_R,
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

# ==========【左入射（小响应）的逆反射】==========
# 定义结果文件夹
result_folder = root_folder + "\\8_M-_l_sweep"
# 创建扫场波形
evo_result_l = load_evolved_waveform(
    file_path=root_folder + "\\4_evo_r\\evolved_waveform.pkl",
    segments=2,
    rename_channel_dict={"PXI1Slot2/ao1": "PXI1Slot2/ao0"},
)
# 绘制波形
_ = plot_waveform(
    evo_result_l,
    save_path=result_folder + "\\evo_result_l_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_l,
    save_path=result_folder + "\\evo_result_l_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_feedback + ao_channels_static_L,
    static_output_waveform=evo_result_l,
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

# %%
# ============================================================================
# 11. 测量背景场（仅声源）
# ============================================================================
# - 移除超表面样件，并尽可能填充吸声棉，减少反射声
# - 开启功放的声源通道（可关闭所有反馈通道）

# ==========【左入射的逆反射】==========
# 定义结果文件夹
result_folder = root_folder + "\\9_X_L_bg_sweep"
# 创建仅包含声源通道的波形
evo_result_L_bg = load_evolved_waveform(
    file_path=root_folder + "\\3_evo_L\\evolved_waveform.pkl",
    segments=2,
    picked_channels=ao_channels_static_L,
)
# 绘制波形
_ = plot_waveform(
    evo_result_L_bg,
    save_path=result_folder + "\\evo_result_L_bg_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_L_bg,
    save_path=result_folder + "\\evo_result_L_bg_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    ao_channels_static=ao_channels_static_L,
    static_output_waveform=evo_result_L_bg,
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
# 创建仅包含声源通道的波形
evo_result_R_bg = load_evolved_waveform(
    file_path=root_folder + "\\4_evo_r\\evolved_waveform.pkl",
    segments=2,
    picked_channels=ao_channels_static_r,
)
# 绘制波形
_ = plot_waveform(
    evo_result_R_bg,
    save_path=result_folder + "\\evo_result_R_bg_waveform.png",
    zoom_factor=1,
)
_ = plot_waveform(
    evo_result_R_bg,
    save_path=result_folder + "\\evo_result_R_bg_detail.png",
    zoom_factor=200,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    ao_channels_static=ao_channels_static_r,
    static_output_waveform=evo_result_R_bg,
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
