"""②COMSOL 仿真域薄胶水（差异化保留部分；通用驱动走 mph + COMSOL-MCP）。

模块地图（SKILL.md 工作流对应）：

- :mod:`export` —— 出版级出图（像素级参数拼装、品牌剥离 ``logo*d=off``、空白自检）；
- :mod:`sidecar` —— 栅格 sidecar 生产端（data→pixel 变换元数据 JSON）+ 消费端
  （matplotlib/ultraplot 数据坐标叠加，衔接③域 truthful colorbar）；
- :mod:`features` —— feature-types 试建探测 + node set 提取/批量写属性；
- :mod:`plotgroup` —— 求解后绘图组数据集自动重绑（空白 PNG 防复发）；
- :mod:`validate` —— 验证函数族（能量守恒/收敛阶/参考比对，纯函数）；
- :mod:`session` —— 会话/许可证探测 + domain-check 机械检查。

全部 mph 触碰点为**鸭子类型驱动 + 懒导入**：mock mph 的机械断言恒跑，真实 COMSOL
用例标 ``-m hardware``（tests/skills/comsol_simulation/）。
"""

from __future__ import annotations

from pysci.skills.comsol_simulation import (
    export,
    features,
    plotgroup,
    session,
    sidecar,
    validate,
)

__all__ = [
    "export",
    "features",
    "plotgroup",
    "session",
    "sidecar",
    "validate",
]
