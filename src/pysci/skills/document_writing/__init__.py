"""document_writing —— 文档写作 skill 后端（LaTeX / PPTX / DOCX）。

Phase 1 能力：
- LaTeX 论文工作流：脚手架 → 撰写 → latexmk 编译 → 解析报错 → 渲染 PDF 供 LLM 看图校对。
- PPTX 阅读/提取：把旧汇报 pptx 转成 Markdown / 结构化文本（含演讲者备注与表格）。

数据区（templates/projects/assets/cache）位于 ``data/skills/document_writing/``，
与包内代码分离。唯一入口是 ``tools.compose`` CLI 门面。
"""
