"""gain_ep 仿真→数据的导出与显示区后处理工具。

- :mod:`.farfield`：远场雷达 1°（361 点）×三组批量导出（CSV/MAT + schema），
  亦可 ``python -m pysci.research.gain_ep.simulation.farfield`` 直接跑；
- :mod:`.semicircle`：半圆显示区 ROI 裁剪与直径边以下遮盖（参数化中心/半径）。

按子模块导入（不在包级重导出）：``from pysci.research.gain_ep.simulation import
farfield, semicircle``。
"""
