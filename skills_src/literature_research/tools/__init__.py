"""pySciWS literature_research tools package.

包含以下模块：
- config: 统一配置加载（读取项目根 .env）
- openalex_client: OpenAlex 学术元数据检索（主源）
- arxiv_client: arXiv 预印本检索
- semantic_scholar_client: Semantic Scholar（TLDR 等；校园网常不可达，best-effort）
- wos_client: Web of Science Starter API（官方 JIF/JCR 分区/ESI）
- zotero_bridge: Zotero 本地/Web API 桥接
- browser_fetch: 付费墙论文抓取（Playwright）：HTML 全文 / 正文 PDF / 补充材料
- pdf_extract: PDF → Markdown（云端 MinerU 优先，公式→LaTeX）
- research: 统一 CLI 门面（doctor/search/read/get/add/library/index）

设计原则：
1. 所有凭据从项目根 .env 加载，代码中绝不硬编码
2. 所有网络响应缓存到 skills_src/literature_research/cache/api_responses/，避免重复请求
3. 所有客户端提供统一的 dict 返回结构，便于上层组装 markdown 笔记
"""

__version__ = "0.1.0"
