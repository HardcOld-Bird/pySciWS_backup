"""pySciWS scientific_plotting tools package.

包含以下模块：
- config: 统一配置加载（读取项目根 .env）+ 路径锚点 + 技能数据目录自动创建
- style: 出版规范层——期刊风格预设(aps/nature)、宽度常量、字体链、mathtext 正斜体约定、
  fonttype=42（EPS/PDF 文字保持可编辑）、rcParams 应用与字体可用性发现
- palette: 色盲友好调色板（Okabe-Ito / Tol bright / Tol vibrant）与颜色循环
- layout: 多子图/嵌套 GridSpec 封装、(a)(b)(c) 面板标号、共享轴工具
- export: 一键导出 EPS/PDF/SVG + PNG 预览（字体嵌入、确定性元数据）
- runner: 发现并运行某研究资产目录下的图生产管线（build_figure）
- audit: 规范自检——宽度/最小字号/字体嵌入/面板标号/色盲可读性
- scaffold: 从模板新建一幅图的生产管线目录
- figures: 瘦 CLI 门面（doctor/styles/new/build/preview/audit）

设计原则（与 literature_research / document_writing / comsol_simulation 一致）：
1. 可选路径/默认值从项目根 .env 加载，代码中绝不硬编码。
2. 技能包只提供约定与工具；具体图内容在各研究代码目录的管线脚本里。
3. 管线代码位于 ``src/pysci/research/<name>/article/figures/<slug>.py``（代码层）；
   导出产物位于 ``data/research/<n>_<name>/article/figures/<slug>/out/``（数据层）。
4. 预览缓存等可再生产物写入 data/skills/scientific_plotting/（git-ignored）。
"""

__version__ = "0.1.0"
