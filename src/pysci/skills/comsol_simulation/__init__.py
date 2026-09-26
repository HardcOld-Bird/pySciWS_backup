"""pySciWS comsol_simulation — AI 驱动的 COMSOL 无头仿真技能包。

以 mph（JPype 直连 COMSOL Java API）为执行引擎，配合 MinerU 全量转换的本地文档索引
与 pyvista 离屏渲染，形成"查手册→建模→渲染校验→网格→求解→导出→后处理→评估"的
全自动闭环。架构与 literature_research / document_writing 一致：代码在 src/pysci/skills/，
数据在 data/skills/comsol_simulation/，统一 CLI 入口 simulation.py。
"""
