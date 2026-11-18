# 流水线细节：MD→Office、引用注入、公式与验收

> `document-writing` 技能的渐进加载分册。SKILL.md 是路由总纲，本册放流水线末端细节。

## 1. MD→docx 流水线

1. **源**：pandoc markdown（`smart` 默认开）；标题层级即文档结构；表格用 pipe table；
   图片用相对路径（转换前把图放在源文件旁或 `--resource-path` 覆盖）。
2. **转换**：
   ```powershell
   pysci-doc convert paper.md -t docx -o paper.docx --reference-doc data/skills/document_writing/templates/house.docx
   ```
   - `--reference-doc`：家风样式模板（styles.xml 里改好标题/正文/图表题注的 docx）。
     模板不存在时先用 DSH office-docx 建一份基准样式文档入库，再作 reference-doc。
   - 长文档加 `--toc --number-sections`。
3. **增量/局部编辑**：pandoc 只能生成全新文档；转换后要改内容 → 交 DSH office-docx
   （python-docx），不要重跑 pandoc 覆盖手工修改。
4. **验收**：DSH office-render 渲染成图 → read_image 逐页目检（标题层级、表格、图、公式）→
   `present` 交付。

## 2. MD→pptx 流水线

```powershell
 pysci-doc convert slides.md -t pptx -o slides.pptx --slide-level 2 --reference-doc house.pptx
```

- `--slide-level 2`：`##` 分页、`###` 作页内标题（学术 deck 惯例）；
- 演讲备注写在 `::: notes` 围栏块内，pandoc 会落到 speaker notes；
- 公式按"公式渲染约定"（SKILL.md）先出透明 PNG 再以图片引用；
- pandoc 不能向既有 pptx 追加页——追加/改页走 DSH office-pptx（python-pptx）；
- 引文：pptx 线先 `--citeproc`（`[@key]` 排成引文与参考文献页）；不加则引用块扁平化、`[@key]`
  原样残留（2026-11-17 冒烟实测），需由 office-pptx 补排；
- **验收**：DSH office-render 逐页渲染，页数=幻灯片数、无空白页、备注在（office-pptx 重开验证）。

## 3. 引用注入（zotero-mcp → refs.bib → citeproc）

1. **取数**：MCP 工具（zotero-mcp）读 Zotero 库，把目标条目导出为 JSON 数组存盘
   （Zotero JSON 或 CSL JSON 皆可，refs_bridge 自动识别）。
2. **归一化**：`pysci-doc refs items.json -o refs.bib`——确定性转换为 BibTeX
   （类型映射、author/editor 拼装、date→year、key 生成去重、TeX 特殊字符转义）。
   每次**全量重写** refs.bib（单调可复现，不做增量合并）。
3. **注入**：MD 正文用 `[@smith2021deep]` 引用键；转换加
   `--citeproc --bib refs.bib --csl <style>.csl`（CSL styles 仓库取 style 文件入
   `data/skills/document_writing/templates/csl/`）。
4. LaTeX 线不用 citeproc：refs.bib 直接给 latexmk+bibtex/biber，正文用 `\cite{}`。

## 4. 公式渲染约定

| 落点 | 路线 | 产物 |
|---|---|---|
| docx 正文 | MD 写 `$...$`，pandoc→OMML | 原生可编辑公式，不产图 |
| pptx / 图内公式 | `pysci-doc formula "..." -o f.png` | 透明底 RGBA PNG（600dpi，ink 裁剪） |
| LaTeX 正文 | 原生 amsmath | 不产图 |

- 管线：`article + amsmath` standalone → latexmk/pdflatex → pymupdf 600dpi 渲染 →
  ink bbox 裁剪（留 padding）→ 原子落盘。
- 源串自检：裁剪后 ink bbox 为空 → 报错（编译失败/空输出），不交付空图。

## 5. PDF→PNG 视觉校对

`pysci-doc pdf2png paper.pdf -o out_dir --dpi 200`：pymupdf 逐页渲染 PNG。
用于 LaTeX PDF 与 office-export-pdf 产物的逐页目检（DSH office-render 不渲染 PDF）。

## 6. 验收清单（主观层）

- [ ] 页数/张数与源一致，无空白页；
- [ ] 标题层级、编号、目录正确；
- [ ] 表格不越界、图片不拉伸、题注在位；
- [ ] 公式完整（docx 里 OMML 可编辑、pptx 里 PNG 清晰无黑底）；
- [ ] 引文键全解析（无 `[?]`）、参考文献列表成形；
- [ ] 中文字体不缺字（渲染图目检）。

## 7. 暂缓能力

学院 deck 三件套（spec 审计/焦点动画/几何体检）、deck_digest、提取乱码门——用户裁决暂缓不建。
