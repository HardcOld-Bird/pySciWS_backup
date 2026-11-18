---
name: document-writing
description: ⑤文档写作能力域操作面：任务路由（DSH office 七技能 / pandoc / LaTeX 单轨）、MD→docx/pptx 流水线、LaTeX 排版工作流、引用注入（zotero-mcp→refs.bib）、公式渲染约定与视觉验收。需要写/转/排 Office 或学术文档、生成公式图、跑 LaTeX 编译时使用。
---

# Document Writing（⑤文档写作）

文档写作能力域的操作面：**路由到社区工具与 DSH 预装技能，只在接缝处用薄胶水**。
选型依据见 `sub_root/others/community_research/2026-11-17/domain-05-docwriting-selection.md`；
本技能只放"怎么用"。薄胶水在 `sub_root/src/pysci/skills/document_writing/`（CLI：`pysci-doc`）。

## 能力分层（谁负责什么）

1. **DSH 预装 `dsh-skill-office` 七技能**（office 路由 / office-docx / office-pptx / office-xlsx /
   office-render / office-recalculate / office-export-pdf）：Office 原生格式的创作、编辑、追加、
   重开验证与 present 交付；创作层锚定 python-docx / python-pptx / openpyxl+pandas。
   **本技能不重复实现其能力，只路由**。
2. **pandoc 3.14**（`sub_root/deps_management/deps/tools/pandoc/pandoc-3.14/pandoc.exe`）：
   格式互转枢纽、docx/xlsx/pptx→MD 提取、`--citeproc` 引用排版、`--reference-doc` 家风样式。
3. **LaTeX 单轨排版**（TeX Live 2026：latexmk + chktex + 日志解析）：期刊投稿（revtex4-2 等）
   与需要 PDF 精排的场景。**Typst 不引入**（用户裁决）。
4. **薄胶水** `pysci-doc`：统一 CLI 路由、pandoc 包装、refs_bridge、PDF→PNG 校对、
   公式透明 PNG（ink 裁剪）、LaTeX 工作流（latexmk 调用+日志解析+chktex）。
5. **操作层**：LibreOffice Kit（渲染/公式重算/导 PDF）随 DSH office-render / office-recalculate /
   office-export-pdf 使用，不自写。

## 任务路由表

| 任务 | 走哪条路 | 入口 |
|---|---|---|
| 创建/编辑/追加 docx、pptx、xlsx；公式重算 | DSH office 技能 | `skill(office)` → office-docx / office-pptx / office-xlsx / office-recalculate |
| Office 文件渲染成图（视觉检查） | DSH office-render | `skill(office-render)`，产物用 read_image 验收 |
| Office 文件导出 PDF | DSH office-export-pdf | `skill(office-export-pdf)` |
| MD→docx / MD→pptx / 任意互转 / 提取→MD | pandoc | `pysci-doc convert` |
| 引文注入（--citeproc + CSL） | pandoc | `pysci-doc convert --citeproc --bib refs.bib --csl <style>.csl` |
| Zotero 文献库 → refs.bib | 薄胶水 refs_bridge | `pysci-doc refs`（数据源 zotero-mcp 导出） |
| LaTeX 论文/报告/Beamer → PDF | LaTeX 工作流 | `pysci-doc latex build`（latexmk→日志解析→chktex） |
| LaTeX 静态检查 | LaTeX 工作流 | `pysci-doc latex lint`（chktex） |
| 公式 → 透明底 PNG（插 pptx/docx） | 薄胶水公式管线 | `pysci-doc formula` |
| PDF 逐页 → PNG 视觉校对 | 薄胶水 | `pysci-doc pdf2png` |
| 记不准路由时 | 机械路由查询 | `pysci-doc route <任务关键词>` |

**原则**：Office 原生格式的创作/编辑一律交 DSH office 技能（含"重开文件验证 + present 交付"），
pandoc 只做互转/提取/引用（它不能增量追加既有 Office 文件），排版进 LaTeX 只为 PDF 精排/期刊投稿。

## MD→docx/pptx 流水线

细节见 [references/pipelines.md](references/pipelines.md)。骨架：

1. MD 源（pandoc markdown，公式用 `$...$` / `$$...$$` LaTeX 语法）；
2. `pysci-doc convert src.md -t docx --reference-doc 家风.docx`（pptx 加 `--slide-level 2`，
   演讲备注写在 `::: notes` 块内）；
3. 引文：`--citeproc --bib refs.bib --csl <style>.csl`；
4. 产物交 DSH office-render 渲染成图 → read_image 视觉验收 → present 交付。

## LaTeX 排版工作流

细节见 [references/latex-workflow.md](references/latex-workflow.md)。骨架：
`pysci-doc latex build paper.tex`（可 `--engine xelatex`）→ latexmk 编译 →
日志解析定位 `file:line` 错误 → chktex lint → 产出 PDF。

## 引用链

zotero-mcp（MCP 工具读 Zotero 库）→ 导出 JSON → `pysci-doc refs` 归一化 → `refs.bib` →
pandoc `--citeproc` + CSL styles 消费。注入约定见 [references/pipelines.md](references/pipelines.md)。

## 公式渲染约定

- **进 docx 的公式**：MD 里写 LaTeX，pandoc 转 docx 时自动变 OMML（原生可编辑），不产图。
- **进 pptx 的公式**：pptx 无逐字公式，走图片——`pysci-doc formula "E = mc^2" -o f.png`
  产**透明底 RGBA PNG**（article+amsmath → PDF → 600dpi 渲染 → ink 裁剪），再由 office-pptx 置入。
- 公式源串自检：渲染结果为空白（ink bbox 为空）时视为编译失败，报错而不是交付空图。

## 验收

1. 机械层：`sub_root/tests/tests.cmd all`（胶水单测 + pin 版本断言）。
2. 视觉层：Office 产物一律 DSH office-render 渲染成图后 read_image 检查（页数/版式/公式/表格）；
   PDF 用 `pysci-doc pdf2png` 逐页转 PNG 后同样目检。
3. LaTeX 产物检查日志无 error（warning 逐条裁决），chktex warning 逐条裁决后交付。

## 暂缓能力（用户裁决，不建）

学院 deck 三件套（spec 审计 / 焦点动画 / 几何体检）、deck_digest 图文工作区、提取乱码门——
暂缓不建；若将来恢复工作流，回到选型文档重走三段式。

## 依赖边界（拒绝清单提醒）

不引入：markitdown、pypandoc、typst-py、XlsxWriter、docxtpl、Quarto。
Office 创作库（python-docx/pptx/openpyxl）为 DSH 技能锚定的豁免 pin 件，只经 DSH 技能使用。
