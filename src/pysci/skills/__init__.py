"""基础设施层：LLM skill 后端。

当前包含：
- ``literature_research``：学术文献检索/精读/评估/知识库管理。
  数据区（papers/shortlists/reviews/cache/templates/INDEX.md）位于 ``data/skills/literature_research/``。
- ``document_writing``：LaTeX/PPTX/DOCX 文档写作（论文/幻灯片/报告的读写、编译、提取）。
  数据区（templates/projects/assets/cache）位于 ``data/skills/document_writing/``。
- ``comsol_simulation``：COMSOL Multiphysics 仿真自动化（建模/求解/导出/后处理）。
  数据区（docs/cache/recipes/templates/knowledge/runs）位于 ``data/skills/comsol_simulation/``。
- ``scientific_plotting``：出版级科研论文插图生产（风格预设/导出/规范自检）。
  数据区（templates/cache/recipes）位于 ``data/skills/scientific_plotting/``。
- ``theoretical_computation``：理论计算（CAS 符号推理/数值计算/参数空间探索/拓扑特征检测）。
  数据区（templates/cache/recipes）位于 ``data/skills/theoretical_computation/``。

各 skill 的数据区与包内代码分离（镜像于 ``data/skills/``），避免把 git-ignore 的
缓存、构建产物与下载的文档放进 ``src/``。
"""
