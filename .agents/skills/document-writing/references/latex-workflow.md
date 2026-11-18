# LaTeX 排版工作流细节

> `document-writing` 技能的渐进加载分册。排版**单轨 LaTeX**（Typst 不引入，用户裁决）；
> 期刊投稿（revtex4-2 等）只能 LaTeX，是本线存在的理由。

## 工具链位置

- TeX Live 2026：`sub_root/deps_management/deps/tex/base/texlive/2026/bin/windows/`
  （注意 `deps/tex` 目前是 junction → `D:\XiGPrograms\tex`，物理迁移待用户关闭占用进程后执行）。
- 用到：latexmk（编译驱动）、pdflatex/xelatex/lualatex（引擎）、chktex（静态检查）。
- 代码里经 `pysci.skills.document_writing.toolpaths` 取绝对路径，不手写 `parents[N]`。

## 工作流（`pysci-doc latex build paper.tex`）

1. **latexmk 编译**：`latexmk -pdf -interaction=nonstopmode -file-line-error paper.tex`；
   中文文档用 `--engine xelatex`（映射 `-xelatex`）。latexmk 自动跑 bibtex/biber 与多轮交叉引用。
2. **日志解析**：读 `paper.log`，解析两类错误定位并统一成 `file:line` 报告——
   - file-line-error 风格：`./chapters/intro.tex:12: Undefined control sequence.`；
   - 经典风格：`! LaTeX Error: ...` 配 `l.<n>` 指针行；
   另收 `LaTeX Warning: ... on input line <n>`（引用未定义/版面溢出）作 warning。
   文件归属用括号文件栈追踪（TeX 日志 `(./file.tex` 开、`)` 闭）。
3. **chktex 检查**：`pysci-doc latex lint paper.tex`（`chktex -q -f "%f:%l:%c: %m" paper.tex`），
   输出解析成 warning 列表。warning 不阻断编译，但交付前逐条裁决（改或显式忽略理由）。
4. **PDF 产出**：`paper.pdf` 落在源文件旁；失败时报告 error 列表（file:line+message），
   不交付半成品 PDF。

## 交付裁决口径

- error > 0：不得交付，先修（报告里的 file:line 直接跳转）；
- warning 逐条裁决：改掉，或在交付说明里写忽略理由；
- 产物 PDF 用 `pysci-doc pdf2png` 逐页转 PNG 目检后再交付。

## 常用模板位

- revtex4-2（期刊投稿）、article+amsmath（报告/公式）、beamer（讲义）——
  脚手架模板放 `data/skills/document_writing/templates/latex/`（内容资产，非自研代码）。
