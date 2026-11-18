"""③科研绘图薄胶水（出版链 ultraplot + matplotlib 底座 + SciencePlots 样式资产）。

选型定稿与用户裁决见
``sub_root/others/community_research/2026-11-17/domain-03-plotting-selection.md``；
实施计划见 ``sub_root/others/plans/domains-03-04-plotting-theory.md``。

模块分层（全部懒导入 matplotlib/pyvista，纯逻辑不依赖绘图栈，机械可测）：

- :mod:`~pysci.skills.scientific_plotting.styles` — 期刊尺寸契约、色盲安全循环、
  样式资产 ``styles/*.mplstyle`` 的解析与应用；
- :mod:`~pysci.skills.scientific_plotting.export` — 导出封装（尺寸契约 / fonttype=42 /
  EPS+PDF+SVG+PNG 多格式 / ``_preview.png`` 预览闭环）；
- :mod:`~pysci.skills.scientific_plotting.labels` — 多子图面板标号 (a)(b)(c)；
- :mod:`~pysci.skills.scientific_plotting.audit` — 轻量出版自检（宽度 / 字号 / 字体嵌入 /
  面板号计数 / 色循环）；
- :mod:`~pysci.skills.scientific_plotting.field_panel` — 数据驱动场面板 truthful colorbar
  （端标取整 v2 :func:`ceil_label`）；
- :mod:`~pysci.skills.scientific_plotting.cli` — ``figures`` 薄路由 CLI。

ADR（2026-11-17）：样式是**内容资产**（SciencePlots 审美层基础上的自定义项固化为具体键值），
放包内 ``styles/*.mplstyle``，用 ``importlib.resources`` 定位（不写 ``parents[N]``）；
运行时**不依赖** scienceplots 包（其差②豁免 pin 只为审美基底可选叠加）。
"""
