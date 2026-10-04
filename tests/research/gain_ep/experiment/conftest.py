"""experiment（原 sweeper400）测试收集守卫。

experiment 平台在模块级 ``import nidaqmx``，且部分用例需要 NI 数据采集硬件与步进电机。
``nidaqmx`` 是 ``pyproject.toml`` 的**主依赖**（``uv sync`` 即安装，**无 extra**）；但它仅在
Windows + NI-DAQmx 驱动下真正可用，其他平台上安装可能失败。当它不可导入时，
忽略本目录下所有测试收集，避免 ``uv run pytest`` 因 ImportError 直接失败。

硬件相关用例用 ``-m "not hardware"`` 过滤（``addopts`` 已默认如此）。
"""

import importlib.util

if importlib.util.find_spec("nidaqmx") is None:
    collect_ignore_glob = ["*"]
