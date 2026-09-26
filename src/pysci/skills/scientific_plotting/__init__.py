"""pySciWS scientific_plotting — AI 驱动的科研论文插图生产技能包。

面向"发表级插图"的出版规范层 + 生产管线骨架：以 matplotlib/scienceplots/ultraplot/pyvista
为绘图后端，固化期刊风格预设（宽度、字体、字号、正斜体、色盲友好配色）、多格式导出
（EPS/PDF/SVG + PNG 预览）与规范自检，形成"设计→绘制→导出→自检→视觉校验"的闭环。

与 literature_research / document_writing / comsol_simulation 架构一致：代码在
``src/pysci/skills/scientific_plotting/``，技能级资产在 ``data/skills/scientific_plotting/``，
统一 CLI 入口 ``figures.py``（console script: ``pysci-figures``）。

设计要点：技能包只承载"怎么画才合规、怎么导出、怎么自检、怎么让 Agent 看到"的可复用约定；
每幅论文插图的具体生产管线（``build_figure``）落在各研究资产目录
``data/research/<n>_<name>/article/figures/<figN>/`` 下，便于按图迭代。
"""
