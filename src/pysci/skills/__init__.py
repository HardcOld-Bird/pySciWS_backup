"""基础设施层：LLM skill 后端。

当前包含：
- ``literature_research``：学术文献检索/精读/评估/知识库管理。
  数据区（papers/shortlists/reviews/cache/templates/INDEX.md）位于 ``data/skills/literature_research/``。
- ``document_writing``：LaTeX/PPTX/DOCX 文档写作（论文/幻灯片/报告的读写、编译、提取）。
  数据区（templates/projects/assets/cache）位于 ``data/skills/document_writing/``。

各 skill 的数据区与包内代码分离（镜像于 ``data/skills/``），避免把 git-ignore 的
缓存、构建产物与下载的文档放进 ``src/``。
"""
