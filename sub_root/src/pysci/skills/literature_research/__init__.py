"""文献调研能力域①的薄胶水后端（其余能力由社区工具 + 技能文档承载）。

模块（自研面合计约 400 行）：

- :mod:`.tex_clean` — arXiv e-print LaTeX 源清洗（去注释/preamble/thebibliography → 轻量 MD）。
- :mod:`.citecheck` — 引用核验门的机械对账（文内引用 × 文献列表，PASS/WARN/FAIL/NOT_FOUND）。
- :mod:`.frontmatter` — 笔记 frontmatter 归一化与 INDEX.md 生成。
- :mod:`.mineru_cloud` — MinerU 云端 PDF→MD API 的薄 HTTP 封装（``.env`` 的 ``MINERU_TOKEN``）。

ADR（自研理由，2026-11-17）：四件在社区均无对应薄实现——tex 源清洗是 arXiv e-print 特有细节，
citecheck/INDEX 是笔记工作流契约，MinerU 官方只提供裸 REST；全部保持纯机械、可单测
（``tests/skills/literature_research/``）。工作流与规范见
``.agents/skills/literature-research/SKILL.md``。
"""
