"""实验测量配置模块

集中定义实验中使用的所有配置参数，包括硬件通道、采样参数、
扫场点阵、仿真参数等，方便在多个脚本和模块间共用。
"""

from pysci.research.gain_ep.experiment.analyze import init_sampling_info
from pysci.research.gain_ep.experiment.use import get_square_grid

# ============================================================================
# 硬件通道配置
# ============================================================================

# 8通道反馈阵列模拟输入（用于采集传声器信号）
ai_channels: tuple[str, ...] = (
    "PXI1Slot3/ai0",
    "PXI1Slot3/ai1",
    "PXI1Slot4/ai0",
    "PXI1Slot4/ai1",
    "PXI1Slot5/ai0",
    "PXI1Slot5/ai1",
    "PXI1Slot6/ai0",
    "PXI1Slot6/ai1",
)

# 扫场传声器模拟输入通道
sweep_ai_channel: str = "PXI1Slot2/ai0"

# 8通道反馈阵列模拟输出（用于驱动反馈扬声器）
ao_channels_feedback: tuple[str, ...] = (
    "PXI1Slot3/ao0",
    "PXI1Slot3/ao1",
    "PXI1Slot4/ao0",
    "PXI1Slot4/ao1",
    "PXI1Slot5/ao0",
    "PXI1Slot5/ao1",
    "PXI1Slot6/ao0",
    "PXI1Slot6/ao1",
)

# 声源模拟输出通道（静态/直通）
ao_channels_static_L: tuple[str, ...] = ("PXI1Slot2/ao0",)
ao_channels_static_r: tuple[str, ...] = ("PXI1Slot2/ao1",)

# 所有模拟输出通道（反馈 + 声源）
ao_channels: tuple[str, ...] = (
    ao_channels_feedback + ao_channels_static_L + ao_channels_static_r
)

# ============================================================================
# 采样参数配置
# ============================================================================

# 最佳工作频率 (Hz)，由频率校准获得
best_frequency: float = 3460.00

# 采样率 (Hz) 和每次采集的采样点数
# 采样率建议为目标频率倍数，且尽可能接近又不大于200kHz
_sampling_rate: float = best_frequency * 50
# 对于扫场，单段采样数≥8575（0.05s）以上时可正常运行
# 然而，总停顿时间过短时，步进电机将无法稳定工作
# 因此，建议至少停顿0.5s（例如0.2s×3）
# 而对演化测量，建议设置为0.4s
_samples_per_chunk: int = 68600  # 约 0.4 秒
# _samples_per_chunk: int = 34300  # 约 0.2 秒

# 采样信息对象
sampling_info = init_sampling_info(_sampling_rate, _samples_per_chunk)

# ============================================================================
# 扫场点阵配置
# ============================================================================

# 正方形网格：x/y 起止位置 (mm) 和步长 (mm)
_grid_x_start: float = 1.0
_grid_x_end: float = 311.0
_grid_y_start: float = 1.0
_grid_y_end: float = 311.0

# 扫场点阵对象
grid = get_square_grid(_grid_x_start, _grid_x_end, _grid_y_start, _grid_y_end)

# ============================================================================
# 实验结果存储路径
# ============================================================================

# 总结果文件夹根路径
root_folder: str = r"D:\EveryoneDownloaded\exp0723"
