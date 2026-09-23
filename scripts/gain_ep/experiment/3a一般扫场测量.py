# pyright: basic

import numpy as np

from pysci.research.gain_ep.experiment.analyze import (
    get_sine,
)
from pysci.research.gain_ep.experiment.config.exp_config import (
    ao_channels_static_L,
    ao_channels_static_r,
    best_frequency,
    grid,
    sampling_info,
    sweep_ai_channel,
)
from pysci.research.gain_ep.experiment.use import (
    SweeperCore,
)

ao_channels_static = ao_channels_static_L + ao_channels_static_r

# 声源输出波形复振幅数组（与 ao_channels_static 等长）
static_cca: np.ndarray = np.full(
    len(ao_channels_static_L),
    0.02 + 0j,
    dtype=np.complex128,
)
# 创建输出波形
static_output_waveform = get_sine(
    sampling_info=sampling_info,
    frequency=best_frequency,
    channel_names=ao_channels_static_L,
    # channel_names=ao_channels_static_r,
    channel_complex_amplitudes=static_cca,
    full_cycle=True,
)
# 创建扫场器
swp = SweeperCore(
    ai_channels=(sweep_ai_channel,),
    sweep_ai_channel=sweep_ai_channel,
    ao_channels_static=ao_channels_static_L,
    # ao_channels_static=ao_channels_static_r,
    static_output_waveform=static_output_waveform,
    point_list=grid,
)
# 定义结果文件夹
result_folder = "D:\\EveryoneDownloaded\\L_INPUT_sweep"
# result_folder = "D:\\EveryoneDownloaded\\r_INPUT_sweep"

# %% 步进电机校准
swp.calib()

# %% 检查位置
swp.where()

# %% 移动位置1
swp.move_to(1.0, 1.0)

# %% 移动位置2
swp.move_to(1.0, 155.0)

# %% 移动位置3
swp.move_to(1.0, 311.0)

# %% 移动位置4
swp.move_to(311.0, 1.0)

# %% 移动位置5
swp.move_to(311.0, 311.0)

# %% 开始扫场测量（非阻塞式）
swp.sweep(
    result_folder=result_folder,
    lowcut=best_frequency / 2,
    highcut=best_frequency * 2,
)
# %% 查看进度
swp.get_progress()

# %% 中止扫场测量
swp.stop()

# %% 销毁扫场控制器
del swp
