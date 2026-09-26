"""pySciWS theoretical_computation tools package.

包含以下模块：
- config: 统一配置加载（读取项目根 .env）+ 路径锚点 + 技能数据目录自动创建
- cas: CAS 核心引擎——符号表达式构建/化简/替换/级数展开/LaTeX 输出/方程组求解
- numerical: 数值化管线——N 维参数空间定义、lambdify 封装、网格采样与求值、自适应采样
- eigen: 本征分析——符号/数值本征值-本征向量、EP 检测、简并追踪、复平面轨迹
- topology: 拓扑探索——零点集、等值面、发散点/奇点、临界点检测、环绕数
- visualize: 探索性可视化——matplotlib 快速图 + pyvista 3D 离屏渲染 + PNG 导出
- session: 计算会话管理——结果持久化、日志、产物导出到 data/
- theory: 瘦 CLI 门面（doctor/new/run/plot/list）

设计原则（与 literature_research / document_writing / comsol_simulation / scientific_plotting 一致）：
1. 可选路径/默认值从项目根 .env 加载，代码中绝不硬编码。
2. 技能包只提供跨研究线复用的计算工具与约定；研究专属计算脚本在 src/pysci/research/<name>/theory/。
3. 计算产物（结果、探索图、日志）写入 data/research/<n>_<name>/theory/<slug>/；
   技能级缓存/模板写入 data/skills/theoretical_computation/（git-ignored）。
"""

__version__ = "0.1.0"
