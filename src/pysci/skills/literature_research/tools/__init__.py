"""pySciWS literature_research tools package.

包含以下模块：
- config: 统一配置加载（读取项目根 .env）
- openalex_client: OpenAlex 学术元数据检索（主源）
- arxiv_client: arXiv 预印本检索
- wos_client: Web of Science Starter API（Times Cited + 收录号 wos_id + JCR 链接）
- citation_verify: 引用完整性门（OpenAlex+Crossref+arXiv 三源交叉核验，冲突则标红阻写）
- zotero_cli: Zotero 库读写（委托社区 zotero-mcp 的 `zotero-cli --json`）
- browser_fetch: 付费墙论文抓取（Playwright 抓取 + trafilatura 抽正文）：HTML 全文 / 正文 PDF / 补充材料
- pdf_extract: PDF → Markdown（mineru-open-sdk 云端优先，公式→LaTeX；pymupdf4llm 兜底）
- local_ingest: 批量归档本地 PDF 文件夹（复制 → 抽取 → manifest 台账）
- rag: PaperQA2 语义检索本地文献库全文（硅基流动 embedding 索引+检索为主；可选 LLM 综述，失败降级）
- cache_manager: 两层缓存治理（stats/clean/prune + LRU）
- research: 统一 CLI 门面（doctor/search/read/get/add/citecheck/library/rag/index/ingest/cache）

设计原则：
1. 所有凭据从项目根 .env 加载，代码中绝不硬编码
2. 所有网络响应缓存到 data/skills/literature_research/cache/api_responses/，避免重复请求
3. 所有客户端提供统一的 dict 返回结构，便于上层组装 markdown 笔记
"""

__version__ = "0.1.0"
