"""⑤文档写作能力域薄胶水（技能文档：``.agents/skills/document-writing/``）。

选型定稿（2026-11-17，见 ``sub_root/others/community_research/2026-11-17/``）：
DSH office 七技能（创作/编辑/渲染/导 PDF）+ pandoc 3.14（互转/引用）+ LaTeX 单轨
（latexmk+chktex，期刊投稿线）。本包只放接缝处的薄胶水，不重复实现上述能力。

模块一览：
- ``toolpaths``     —— pandoc / TeX 工具链路径常量（未入 toolchain.json 前的引用点）。
- ``pandoc_wrapper``—— pandoc.exe subprocess 包装（参数拼装 + 调用）。
- ``refs_bridge``   —— zotero-mcp 导出 JSON → refs.bib 归一化。
- ``pdf_to_png``    —— PDF 逐页渲染 PNG（视觉校对）。
- ``formula_png``   —— 公式 → 透明底 RGBA PNG（ink 裁剪）。
- ``latex_workflow``—— latexmk 调用 + 日志解析（file:line 定位）+ chktex。
- ``cli``           —— 统一 CLI 门面 ``pysci-doc``（任务路由 + 分发）。
"""

from __future__ import annotations
