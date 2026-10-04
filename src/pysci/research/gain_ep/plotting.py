"""gain_ep 研究线的共享绘图配置。

放在研究线根（而非 ``experiment/config/``）是为了让 ``theory`` 与 ``experiment`` 两侧都能
导入中文字体配置，而**不触发** ``experiment/__init__.py`` 的硬件依赖链
（``.use`` → ``measure.cont_sync_io`` 顶层 ``import nidaqmx``，实测会连带加载 59 个 ``nid*`` 模块）。

本模块是 ``setup_chinese_fonts`` 的**唯一实现**；``experiment/config/matplotlib_config.py``
仅做再导出，以保持 ``from ..config import setup_chinese_fonts`` 的既有调用面不变。
"""

import matplotlib.pyplot as plt

#: 中文字体回退链（按优先级），末位为跨平台兜底。
FONT_SANS_SERIF_CHAIN: list[str] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Microsoft JhengHei",
    "DejaVu Sans",
]


def setup_chinese_fonts() -> None:
    """配置 matplotlib 以正确显示中文与负号。

    设置 ``font.sans-serif`` 回退链，并关闭 ``axes.unicode_minus``
    （否则负号会用 Unicode 减号渲染，在缺失字形的字体下显示为方框）。
    直接改写全局 ``rcParams``，无返回值；重复调用是幂等的。
    """
    plt.rcParams["font.sans-serif"] = FONT_SANS_SERIF_CHAIN
    plt.rcParams["axes.unicode_minus"] = False
