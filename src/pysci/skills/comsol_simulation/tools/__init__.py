"""pySciWS comsol_simulation tools package.

包含以下模块：
- config: 统一配置加载（读取项目根 .env）+ COMSOL 安装发现 + 资源护栏 + 路径锚点
- session: mph 客户端生命周期（standalone / 持久 server / 上下文管理器）
- inspect: 模型树 dump、参数与节点清单、JPype 运行时自省（活体 javadoc）、.java 摘要
- build: recipe 引擎（几何/材料/物理场/网格/研究原语）+ 外部几何导入
- run: 求解、参数扫描、comsolbatch、日志与异常链结构化捕获
- export: 图像(PNG)/网格(VTK,STL,PLY)/数据(CSV,TXT) 导出
- postprocess: pyvista 场分析与离屏渲染、网格质量、收敛性、与理论解比对、验证器
- docs: MinerU 批量转换 + SQLite FTS5 索引构建/检索/按节读取
- simulation: 瘦 CLI 门面（session/inspect/build/run/export/render/postprocess/docs）

设计原则（与 literature_research / document_writing 一致）：
1. 凭据与可选路径从项目根 .env 加载，代码中绝不硬编码。
2. 只做编排与本地文件/进程操作；求解重活交给 COMSOL（经 mph），文档转换交给 MinerU。
3. 构建产物、文档缓存、运行日志写入 data/skills/comsol_simulation/（git-ignored）。
4. 所有需要 live COMSOL 的路径都可被 hardware 标记的测试守卫跳过。
"""

__version__ = "0.1.0"
