"""基础设施层：LLM skill 后端。

当前包含 ``literature_research``（学术文献检索/精读/评估/知识库管理）。
其数据区（papers/shortlists/reviews/cache/templates/INDEX.md）位于项目根 ``literature/``，
与包内代码分离，避免把 git-ignore 的缓存与下载的 PDF 放进 ``src/``。
"""
