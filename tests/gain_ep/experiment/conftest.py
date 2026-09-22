"""experiment（原 sweeper400）测试收集守卫。

experiment 平台在模块级 ``import nidaqmx``，且部分用例需要 NI 数据采集硬件与步进电机。
nidaqmx-python 是可选依赖（``uv sync --extra experiment`` 才安装）。当它不可用时，
忽略本目录下所有测试收集，避免 ``uv run pytest`` 因 ImportError 直接失败。

需要运行这些测试时：先 ``uv sync --extra experiment``，硬件相关用例用 ``-m "not hardware"`` 过滤。
"""

import importlib.util

if importlib.util.find_spec("nidaqmx") is None:
    collect_ignore_glob = ["*"]
