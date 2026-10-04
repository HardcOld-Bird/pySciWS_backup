"""matplotlib 中文配置模块（再导出层）。

``setup_chinese_fonts`` 的**唯一实现**在 :mod:`pysci.research.gain_ep.plotting` —— 上提到研究线根
是为了让 ``theory`` 侧也能导入，而不触发 ``experiment/__init__.py`` 的 nidaqmx 硬件依赖链。
本模块仅做再导出，以保持 ``from ..config import setup_chinese_fonts`` 的既有调用面不变。
"""

from ...plotting import FONT_SANS_SERIF_CHAIN, setup_chinese_fonts

__all__ = [
    "FONT_SANS_SERIF_CHAIN",
    "setup_chinese_fonts",
]
