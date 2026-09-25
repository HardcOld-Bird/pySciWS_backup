"""pySciWS document_writing tools package.

包含以下模块：
- config: 统一配置加载（读取项目根 .env）+ 外部工具链探测（TeX / LibreOffice / Python 库）
- latex_build: latexmk 编译驱动 + .log 报错解析（错误/警告定位到行号）
- pdf_render: PDF → PNG（pymupdf），供 LLM「看图」校对版式
- refs_bridge: 复用 literature_research.zotero_bridge，把 Zotero 库导出为 refs.bib
- pptx_io: python-pptx 结构化读取（逐页文本/备注/表格/图片清单）
- extract: markitdown 统一提取（pptx/docx/pdf → Markdown）
- compose: 瘦 CLI 门面（doctor / tex / read / slides / verify）

设计原则（与 literature_research 一致）：
1. 凭据与可选路径从项目根 .env 加载，代码中绝不硬编码。
2. 只做编排与本地文件/进程操作；重活交给成熟外部工具（latexmk / markitdown）。
3. 构建产物与提取缓存写入 data/skills/document_writing/cache/（git-ignored）。
"""

__version__ = "0.1.0"
