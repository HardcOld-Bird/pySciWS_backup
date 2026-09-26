"""pySciWS theoretical_computation — AI 驱动的理论计算技能包。

面向"符号推理 + 数值计算 + 探索性可视化"的完整闭环：以 sympy/numpy/scipy 为计算后端，
pyvista/matplotlib 为可视化后端，提供 CAS 符号推导、N 维参数空间网格采样、本征分析、
拓扑特征检测（零点集/等值面/奇点/临界点）以及轻量级探索性绑图。

与 literature_research / document_writing / comsol_simulation / scientific_plotting 架构一致：
代码在 ``src/pysci/skills/theoretical_computation/``，技能级资产在
``data/skills/theoretical_computation/``，统一 CLI 入口 ``theory.py``
（console script: ``pysci-theory``）。

设计要点：
- 技能包承载跨研究线复用的计算工具与约定（CAS 引擎、数值化管线、拓扑探索、可视化）。
- 研究专属的理论计算脚本落在 ``src/pysci/research/<name>/theory/``（代码层）。
- 计算产物（数值结果、探索图、日志）落在 ``data/research/<n>_<name>/theory/<slug>/``（数据层）。
- 探索性可视化不追求出版规范；需要论文级插图时交接给 scientific_plotting。
"""
