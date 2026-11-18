# ⑤办公文档写作能力域 — 选型细化调研报告

- 日期：2026-11-17
- 阶段：新需求三段式第 1 段「选型」的调研部分
- 上轮已定结论（不重复论证）：无一站式方案；pandoc / Typst 为头部候选；LibreOffice MCP≈0★、Overleaf MCP 22★ 不引入；Office/LaTeX 桥接走薄胶水。
- 用户三口径：① star≥100；② 2026-05~2026-11 每月均有提交；③ 减少自研代码。差②但**无可替代**→豁免（pin 版本+记风险）；**有替代余地→拒绝**。
- 本轮用户追加裁决（本报告按此定稿）：
  1. **python-docx/pptx 豁免引入（pin 版本）**：DSH 预装 @deepseek-ai/dsh-skill-office 创作层锚定这两库，生态位无可替代，拒绝会让预装技能空转、自研面反升。
  2. 建议组合三层分工：**DSH office 技能（Office 原生格式）+ pandoc（互转/引用）+ Typst（学术排版，按需）**；操作层用 DSH 自带 LibreOffice Kit，自研≈0。

## 一、旧能力清单（pySciWS，仅提炼能力、未复制代码）

来源：`orchestration/skills/document-writing/SKILL.md`、`src/pysci/skills/document_writing/tools/*`（18 个模块）、`tests/skills/document_writing/*`（22 个测试）、`data/skills/document_writing/{templates,cache,projects}`。

| # | 能力项 | 旧实现方式 | pandoc / Typst / DSH 路线如何覆盖 | 覆盖不了→薄胶水预算 |
|---|---|---|---|---|
| 1 | LaTeX 论文工作流（revtex4-2/article/beamer 脚手架、latexmk 编译、file:line 日志解析、chktex lint） | 自研 `latex_build.py` + 系统 TeX Live/latexmk/chktex | **无可替代部分**：期刊投稿（revtex）只能 LaTeX；pandoc 可作 md→tex 前端。Typst 覆盖轻量中文报告/论文 PDF（无需 TeX 发行版） | 保留 LaTeX 才需要：latexmk 调用+日志解析+chktex 包装 ≈150 行 |
| 2 | PDF 视觉校对（PDF→PNG 逐页渲染） | `pdf_render.py`（pymupdf） | pymupdf/PyMuPDF 薄封装；DSH office-render 只渲染 Office 不渲染 PDF | ≈50 行薄封装 |
| 3 | Zotero→refs.bib 引文刷新 | `refs_bridge.py`（复用 zotero-mcp 的 `zotero-cli --json`，转 BibTeX） | zotero-mcp（社区，三口径全过）+ pandoc citeproc 消费 CSL/BibTeX | Zotero条目→BibTeX 确定性转换 ≈100–200 行；若 zotero-mcp 直接导出 CSL JSON/BibTeX 则归零 |
| 4 | 文档提取→Markdown（pptx/docx/xlsx/html/csv；PDF→MD；旧 .doc 走 antiword/catdoc；乱码门+缓存） | `extract.py`（markitdown + pymupdf4llm） | **pandoc 3.14 已带 docx/xlsx/pptx reader**（见 §三）；PDF→MD 走 pymupdf4llm（三口径全过）；DSH office 技能可读 docx/pptx | 乱码门+提取缓存（事故后铁律）≈100 行；旧 .doc 需 antiword/catdoc 小胶水（待裁决是否保留） |
| 5 | pptx 结构化读取（逐页文本/项目符号/表格/图片/演讲者备注） | `pptx_io.py`（python-pptx） | DSH office-pptx 技能 + python-pptx（**已豁免**）；pandoc pptx reader 兜底 | 0（库内） |
| 6 | docx 结构化读写（读→MD；写标题/正文/列表/表格；run 级标注 marks） | `docx_io.py`（python-docx） | DSH office-docx 技能 + python-docx（**已豁免**）；pandoc md↔docx | 0（库内） |
| 7 | pptx 生成与增量追加（`slides new/add`；pandoc 不能追加） | `pptx_io.py`（python-pptx） | DSH office-pptx 技能（python-pptx 已豁免） | 0（库内） |
| 8 | Markdown→docx/pptx 转换（`--reference-doc` 家风样式、`--slide-level`、`::: notes` 演讲备注） | `pandoc_convert.py`（pandoc，pypandoc-binary 自带） | **pandoc CLI**（3.14，三口径全过）；不引 pypandoc 包（差②） | ≈30 行 subprocess 包装 |
| 9 | Office→PDF 转换（WPS wpscli 优先、LibreOffice headless 兜底） | `office_convert.py` | **DSH office-export-pdf（捆绑 LibreOffice Kit CLI）**，自研≈0；系统级 LibreOffice 仅兜底 | 0 |
| 10 | 公式渲染为透明底 RGBA PNG（article+amsmath→PDF→ink 裁剪 600dpi） | `formula_render.py`（TeX 管线） | Typst `--format png` 或 LaTeX standalone 产图；ink 裁剪仍需自己做 | ink 裁剪/原子落盘/源串自检 ≈100 行（或改 pymupdf 裁剪） |
| 11 | 大型图文 deck 翻译工作区（图文互证、sha1 去重、断点账本、分批视觉解读） | `deck_digest.py`（47KB，纯自研 + python-pptx） | **无社区对应**（DSH/pandoc/Typst 均不做） | 若保留：保留原设计重写 ≈数 KB 级；否则砍能力 → **待裁决** |
| 12 | pptx 逐页渲染与视觉验收（pptx→pdf→PNG、页数/非空校验） | `slides_visual.py`（LibreOffice+pymupdf） | 生产端 → **DSH office-render**；验收闸（页数=幻灯片数、非空页） | 机械验收 ≈80 行 |
| 13 | deck-spec 机械审计/字体全槽/焦点块对齐/引用框/图槽（含 geometry 遮挡越界体检） | `slides_audit.py`+`slides_fonts/layout/geometry`（python-pptx+lxml） | **无社区对应**（DSH 技能只管创作+渲染，无 spec 对账） | 若保留学院预设工作流：审计+几何 ≈中等（原实现 44KB+23KB 测试） → **待裁决** |
| 14 | 焦点块动画 timing XML 注入（学院正式汇报预设） | `slides_anim.py`（lxml 直写 OOXML） | 无社区对应（python-pptx 本身也不支持动画，旧码本就绕开它直写 XML） | 若保留 ≈17KB → **待裁决** |
| 15 | 图像置入前裁剪（去刊头/图注，px/比例） | `image_crop.py`（PIL） | PIL 薄封装（与旧实现同量级） | ≈80 行 |
| 16 | 组会 PPT 预设机械检查（标题形态/框宽公式） | `slides_visual/layout` + 大量测试 | 无社区对应 | 并入 13（若保留） |
| 17 | 工具链自检 `doctor`（TeX/LO/Pandoc/库探测） | `config.py` | 各技能自带自检；DSH office 技能加载时校验解释器/CLI 路径 | ≈50 行或并入胶水 CLI |
| 18 | xlsx **读取**（markitdown 拍平） | `extract.py` | pandoc xlsx reader（读）；**生成/编辑/重算新增覆盖**：DSH office-xlsx + openpyxl（已豁免）+ office-recalculate | 0 |
| 19 | 提取产物缓存 + 乱码门（NUL/替换字符占比>5% 拒收） | `extract.py`（事故后铁律） | DSH 技能有"重开文件验证"规范，但**不含**提取产物乱码门 | ≈100 行（若保留） → **待裁决** |
| 20 | 统一 CLI 门面 `compose`（52KB 中文子命令） | `compose.py` | 三层组合只需**薄**路由（调 DSH skill + pandoc/typst CLI） | 薄 CLI ≈数百行（远小于旧 compose） |

结论：旧清单 20 项中，**14 项被 DSH 技能/pandoc/Typst 直接或近零自研覆盖**；残余集中在 13/14/11（学院 deck 预设三件套 + digest，无社区对应）与若干 ≈50–200 行的薄胶水。

## 二、DSH 内置办公插件（本轮新增候选，预装、引入成本≈0）

本地核实：`sub_root\deps_management\deps\node\modules\node_modules\@deepseek-ai\dsh-skill-office\`（package.json version **0.2.2-rc.1，license MIT**，依赖 `@deepseek-ai/libreoffice-kit ^0.1.5`；README 已读）。

- 7 技能：`office`（路由）/ `office-docx` / `office-pptx` / `office-xlsx`（创作编辑）/ `office-render`（渲染成图）/ `office-recalculate`（公式重算）/ `office-export-pdf`（导 PDF）。
- 创作层锚定 python-docx / python-pptx / openpyxl+pandas，须经 `load_workspace_dependencies` 用工作区依赖执行；含"重开文件验证 + present 交付"流程规范。
- 操作层走 Node + 捆绑 LibreOffice Kit CLI；README 已知限制：LO 渲染与原生 Office 有差异；旧格式/加密/宏需独立工作流；**提供者不安装创作库**（由项目依赖层提供）。
- 技能允许 AGENTS.md/用户显式指定"另一库"覆盖默认库。
- 另有独立包 `@deepseek-ai/dsh-office-to-pdf`。

### 能力覆盖对比（DSH office 技能 vs pandoc/Typst）

| 能力 | DSH office 技能 | pandoc 3.14 | Typst | 备注 |
|---|---|---|---|---|
| docx 创建/编辑/追加/验证/交付 | ✅ office-docx | ⚠️ 只能生成全新文档，不能追加 | ❌ | DSH 为主 |
| pptx 创建/编辑/追加/备注 | ✅ office-pptx | ⚠️ 同上（新建可，追加不可） | ❌ | DSH 为主 |
| xlsx 创建/编辑/公式重算 | ✅ office-xlsx + office-recalculate | ⚠️ 只有 reader（读） | ❌ | DSH 为主 |
| 渲染成图（视觉检查） | ✅ office-render（LO Kit） | ❌ | ✅ 自身可导 PNG | Office 用 DSH |
| 导出 PDF | ✅ office-export-pdf | ⚠️ md→pdf 需 TeX/Typst 引擎 | ✅ 原生 PDF | Office 用 DSH |
| 格式互转（md↔docx/pptx/xlsx/odt/epub/html…） | ❌ | ✅ **枢纽** | ❌ | pandoc |
| 提取/读取 → Markdown | ⚠️ 按技能读（python-docx/pptx） | ✅ docx/xlsx/pptx reader | ❌ | 双轨 |
| 引用 citeproc + CSL | ❌ | ✅ 内置 | ⚠️ hayro/biblatex 生态较弱 | pandoc |
| 学术排版（中文论文/公式/交叉引用） | ❌ | ⚠️ md→tex 层 | ✅ 主场 | Typst 按需 |
| LaTeX 期刊投稿（revtex4-2） | ❌ | ⚠️ 作 md→tex 前端 | ❌ 期刊不收 Typst | 无可替代，工具链 |
| docx/pptx 模板套用 | ✅（python-docx/pptx 手法） | ✅ `--reference-doc` 样式模板 | ✅ typ 模板 | 双轨 |
| deck-spec 审计/焦点动画/图文 digest | ❌ | ❌ | ❌ | 薄胶水（待裁决） |

### 关键张力的客观评估（python-docx/pptx 豁免）

- (a) 若拒绝：DSH office-docx/office-pptx 技能空转，等价于自写替代技能，自研面↑，违反③；
- (b) "停更"实为成熟稳定：python-docx v1.2.0（5.8k★，last commit 2025-06）、python-pptx v1.0.2（3.6k★，last commit 2024-08）、openpyxl 3.1.5（PyPI 2024-06-28，主仓 Heptapod 反爬）——纯 Python、API 十年量级不变；
- (c) DSH 技能预装、维护面归 DSH 上游，PySci 只 pin 依赖版本；
- **裁决（用户 2026-11-17）：豁免引入，pin 版本，风险记原地 ADR**（记录：②不满足、上游低活跃、若 DSH 升级锚定 API 变化时的升级路径）。

## 三、候选指标表（三口径 + Py3.13/Win + agent 驱动 + 许可 + 边界）

核验方法：① 用 `img.shields.io/github/stars/<o>/<r>.json`；② 用 `commits.ecosyste.ms` 逐月探测 `?since=&until=&per_page=1`（2026-05…11 每月一次），辅以 shields `last-commit`；api.github.com 匿名限流（403）、GitHub HTML commits 页截断，均已绕开。ecosyste.ms 对超大仓（LibreOffice/core）索引缺失，阴性结果标注"核验受限"。

| 候选 | ① star | ② 2026-05~11 逐月 | ③ 减自研 | Py3.13/Win | agent 驱动 | 许可 | 能力边界/备注 | 判定 |
|---|---|---|---|---|---|---|---|---|
| **@deepseek-ai/dsh-skill-office**（预装 0.2.2-rc.1） | N/A（DSH 上游，GitHub 深seek-harness 仓库非公开星标） | N/A（随 DSH 版本维护） | ✅ 技能即用 | ✅ Node+LO Kit 捆绑 | `skill` 按需加载 + `load_workspace_dependencies` + shell | MIT | 不做互转/学术排版/引用；不装创作库 | **引入（首选，成本≈0）** |
| **python-docx** 1.2.0 | 5.8k ✅ | ❌（last commit 2025-06） | ✅ DSH 锚定 | ✅ 纯 Python | DSH 技能内 Python | MIT | 仅 OOXML docx，无旧 .doc | **豁免引入，pin 1.2.0（用户裁决）** |
| **python-pptx** 1.0.2 | 3.6k ✅ | ❌（last commit 2024-08） | ✅ DSH 锚定 | ✅（lxml wheel） | DSH 技能内 Python | MIT | 不支持动画（动画旧码本就 lxml 直写） | **豁免引入，pin 1.0.2（用户裁决）** |
| **openpyxl** 3.1.5 | N/A（Heptapod 主仓无星标） | ❌（PyPI 3.1.5 发布 2024-06-28；近 6 个月无发版；主仓反爬无法逐月核验） | ✅ DSH 锚定 | ✅ 纯 Python（py≥3.8） | DSH office-xlsx 内 Python | MIT | pandas（工作区既定依赖）同层 | **豁免引入，pin 3.1.5（用户裁决，同层1）** |
| **pandoc** 3.14（2026-11-11 发布） | 47k ✅ | ✅ 7/7 月（5/6/7/8/9/10/11 均有提交） | ✅ CLI 即可，无需包装库 | ✅ Windows exe/winget，与 Python 无关 | subprocess CLI | GPL-2.0 | 不做：增量追加既有 Office 文件、审计 OOXML、渲染验证；3.14 新增 **pptx/xlsx reader**（提交 2026-10-30 起，见来源） | **引入（首选）** |
| **typst** CLI（typst/typst） | 57k ✅ | ✅ 7/7 月 | ✅ CLI | ✅ Windows exe，独立于 Python | subprocess CLI | Apache-2.0 | 不做：docx/pptx/xlsx 原生格式；期刊投稿排版 | **按需引入（次选/备选，见取舍点 A）** |
| typst-py（PyPI `typst` 0.15.2，2026-11-05 发） | 367 ✅ | ❌（ecosyste.ms 8 月空；shields last=11 月） | ✅ | ✅ cp39-abi3/win_amd64 wheel | Python API `typst.compile` | Apache-2.0 | 进程内 API 便利 | **拒绝（差②且 CLI 可替代）**；想要进程内 API 须豁免 |
| pypandoc / pypandoc-binary 1.17（2026-03-14 发） | 1.2k ✅ | ❌（5/7/8/10/11 空，last 2026-09） | ✅ | ✅ 含 py3.13 classifier | Python 包装 CLI | MIT | 只是 pandoc 的 pip 壳 | **拒绝（差②、直接调 pandoc.exe 即可）** |
| markitdown（0.1.7） | 196k ✅ | ❌（6 月空；5/7/8/9/10/11 有） | ✅ | ✅ | Python/CLI | MIT | 拍平提取；已被 pandoc readers+DSH 技能+pymupdf4llm 分流 | **拒绝（差②、有替代）**；pandoc 3.14 pptx reader 太新（<1 个月），保真度需实测，见取舍点 F |
| XlsxWriter 3.3.1 | 4k ✅ | ❌（6/9/11 月空） | ✅ | ✅ | Python | BSD-3 | 只写不读；openpyxl 已覆盖 | **拒绝（差②、有替代）** |
| docxtpl 0.20.2（2025-11-13 发） | 2.7k ✅ | ❌（last commit 2026-07 → 8~11 月空） | ✅ | ✅ | Python（jinja2+python-docx） | LGPL-2.1 | docx 占位符模板填充 | **拒绝（差②；DSH office-docx + python-docx 薄胶水可替代）** |
| **zotero-mcp**（54yyyu） | 5.9k ✅ | ✅ 7/7 月 | ✅（上轮已定） | ✅ 独立 MCP 进程，不占工作区依赖 | MCP stdio 工具 | MIT | 只管文献库读写，不产 .bib | **引入（引用链，复核通过）** |
| **CSL styles**（citation-style-language/styles） | 3.9k ✅ | ✅ 7/7 月 | ✅（纯数据） | ✅（与平台无关） | 文件：`pandoc --csl` | CC-BY-SA 4.0 | 纯样式数据 | **引入（引用样式）** |
| **pymupdf4llm** 2.1 | 2.3k ✅ | ✅ 7/7 月（5~10 月逐月+shields 11 月） | ✅ | ✅ wheel（底层 pymupdf 同源） | Python API | **AGPL-3.0** | PDF→Markdown；PDF→PNG 校对用 pymupdf | **引入（PDF 提取），AGPL 许可见取舍点 D** |
| **DSH LibreOffice Kit**（@deepseek-ai/libreoffice-kit） | 预装 | N/A | ✅ 自研≈0 | ✅ win32-x64 捆绑 | shell CLI | 随 DSH(MIT) | LO 与原生 Office 渲染有差异 | **引入（操作层主路线）** |
| LibreOffice/core（系统级兜底） | 4.7k ✅ | ⚠️ 核验受限（ecosyste.ms 该仓索引缺失；shields last=today，月度发布节奏活跃） | ✅ | ✅ exe | subprocess soffice | MPL-2.0 | 仅在需系统级转换兜底时装 | **按需（兜底），②核验受限记风险** |
| Quarto（quarto-cli 1.12） | 6.2k ✅ | ✅ 7/7 月 | ⚠️ 引入会加一层编排 | ✅ 独立发行版 | CLI | MIT | 发布/网站编排，旧清单无此需求，pandoc 是其底层 | **暂不引入（三口径过了但无需求位）** |
| pandoc/lua-filters（示例库） | 677 ✅ | ❌（last commit 2025-12） | ✅ | ✅ | pandoc `--lua-filter` | MIT | 只是示例集合 | **不作依赖；按需自写 lua filter（薄胶水 ≈数十行）** |
| typst/templates（官方模板） | 434 ✅ | ❌（last commit 2026-03 → 4 月起空） | ✅ | ✅ | 文件 | Apache-2.0 | 模板是标记内容非代码 | **不引入；自写/摘抄 typ 模板（内容资产，非自研代码）** |
| TeX Live + latexmk + chktex（系统工具链） | N/A（CTAN/系统件） | N/A（随发行版更新） | ✅ | ✅ Windows（TUNA 镜像） | subprocess | GPL-2.0 | **revtex 期刊投稿无可替代**；若裁掉 LaTeX 路线可不装 | **按需安装（取舍点 A）** |
| antiword/catdoc（旧 .doc） | 低星 | 未核 | ✅ | ✅ exe | CLI | GPL | markitdown 不支持旧 .doc（旧事故教训） | 按需（取舍点 B 附带） |

## 四、豁免与拒绝清单（三口径判定汇总）

**豁免引入（差②但无可替代，pin 版本 + 原地 ADR 记风险）**
1. python-docx 1.2.0、python-pptx 1.0.2、openpyxl 3.1.5（+pandas 工作区既定）——**用户 2026-11-17 裁决**：DSH 技能生态位无可替代，拒绝则预装技能空转、自研面反升。

**全三口径通过 → 直接引入**
pandoc 3.14、zotero-mcp、CSL styles、pymupdf4llm（+pymupdf）、DSH office 七技能与 LibreOffice Kit（预装，指标 N/A 但成本≈0）、typst CLI（指标全过，引入与否见取舍点 A）。

**拒绝（差②且有替代余地）**
- markitdown（6 月缺）→ pandoc readers + DSH 技能 + pymupdf4llm 分流
- pypandoc/pypandoc_binary（多月缺）→ 直接调 pandoc.exe
- typst-py（8 月缺）→ typst CLI
- XlsxWriter（6/9/11 月缺）→ openpyxl（DSH 层）
- docxtpl（8~11 月缺）→ DSH office-docx + python-docx 薄胶水
- pandoc/lua-filters（2025-12 后无提交）→ 自写 lua filter
- typst/templates（2026-03 后无提交）→ 自备模板标记文件
- LibreOffice MCP、Overleaf MCP（上轮结论：≈0★/22★，不引入）

**暂不引入（三口径过但无需求位）**：Quarto。

## 五、建议组合（定稿）

**第一层：DSH office 七技能 + python-docx/pptx/openpyxl+pandas（pin 版本，豁免）**
- 覆盖：Office 原生格式创作/编辑/追加/重开验证/present 交付；xlsx 生成与公式重算；渲染成图、导 PDF（捆绑 LibreOffice Kit，操作层自研≈0）。

**第二层：pandoc 3.14 CLI（格式互转 / 提取 / 引用 citeproc）**
- md↔docx/pptx/xlsx/odt/epub/html/LaTeX 互转；`--reference-doc` 家风模板；docx/xlsx/pptx→md 提取；citeproc+CSL 引文（配 zotero-mcp 出的 .bib/CSL JSON）。
- 某子能力若 pandoc 已覆盖（如 md→pdf 经 LaTeX、公式排版进 docx/tex），**不另引**对应库。

**第三层：Typst（学术排版，按需引入）**
- 仅在确认存在"pandoc+DSH 覆盖不了的排版需求"（轻量中文论文/讲义 PDF、不想装 TeX 发行版）时引入 typst CLI；否则降为备选。
- LaTeX 工具链（TeX Live+latexmk）仅 revtex 期刊投稿线保留——该子能力**无可替代**（期刊只收 LaTeX）。

**引用链**：zotero-mcp（拉库）→ 薄转换 → pandoc `--citeproc` + CSL styles。

## 六、薄胶水预算（残余自研）

| 项 | 内容 | 估算 | 条件 |
|---|---|---|---|
| 1 | 薄 CLI/路由门面（调 DSH 技能 + pandoc/typst/LO Kit，doctor 自检） | ≈300–500 行（旧 compose 52KB 大幅缩水） | 必做 |
| 2 | pandoc 调用包装（reference-doc、slide-level、citeproc 参数） | ≈50 行 | 必做 |
| 3 | refs_bridge：Zotero 条目→BibTeX/CSL JSON 转换 | ≈100–200 行；若 zotero-mcp 原生导出则≈0 | 必做（可压缩） |
| 4 | PDF→PNG 视觉校对（pymupdf 薄封装） | ≈50 行 | 必做（DSH render 不管 PDF） |
| 5 | 公式→透明 PNG 的 ink 裁剪（Typst/LaTeX 出图后处理） | ≈100 行 | 需要 deck 公式图时 |
| 6 | deck-spec 机械审计 + 字体全槽 + 几何体检（旧 #13/#16） | 中等（原 44KB+23KB 量级） | **待裁决 C**：保留学院预设工作流才做 |
| 7 | 焦点块动画 timing XML 注入（旧 #14） | ≈17KB | **待裁决 C** |
| 8 | deck_digest 图文互证工作区（旧 #11） | 数 KB 级重写 | **待裁决 B** |
| 9 | 乱码门 + 提取缓存（旧 #19 事故铁律） | ≈100 行 | **待裁决 B** |
| 10 | image_crop（去刊头/图注） | ≈80 行 | 按需 |
| 11 | latexmk 日志 file:line 解析 + chktex 包装 | ≈150 行 | **待裁决 A**（保留 LaTeX 线才做） |
| 12 | 旧 .doc 提取（antiword/catdoc 路由） | ≈50 行 | 按需 |

**量级判断**：必做项合计 ≈500–900 行（旧域自研以 18 模块 300KB+ 为主）；若 C（学院 deck 三件套）与 B（digest/乱码门）被裁掉，自研面收敛到 **<500 行纯胶水**。

## 七、待用户确认的取舍点

- **A. 排版双轨还是单轨**：revtex 期刊投稿无可替代→若保留投稿能力需装 TeX Live+latexmk+胶水#11；Typst 是否作为轻量日常排版线引入（指标全过，可先不装、按需启用）。
- **B. 旧域两项重资产是否随⑤域重建**：deck_digest 图文翻译工作区、提取乱码门+缓存——无社区对应，保留=自研，砍=能力降级。
- **C. 学院 deck 预设三件套（spec 审计/焦点动画/几何体检）是否保留**——唯一"大头"薄胶水，取决于用户是否继续该工作流。
- **D. pymupdf/pymupdf4llm 的 AGPL-3.0**：个人科研内部使用无碍，若将来分发产物/服务需评估；不接受则 PDF 提取回退 pandoc `pdf` reader（质量较低）。
- **E. 拒绝清单是否认可**：markitdown / pypandoc / typst-py / XlsxWriter / docxtpl 全部因差②且有替代而拒绝；如需其中任一（尤其 typst-py 进程内 API、markitdown 提取兜底）须按"无可替代"给豁免。
- **F. pandoc 3.14 新 pptx reader 保真度**：2026-10-30 起才合入（演讲者备注/表格/图表支持仍在快速迭代），上线前需用真实 deck 实测；不达标再回头豁免 markitdown。

## 八、来源 URL

- 星标/最近提交：`https://img.shields.io/github/stars/<owner>/<repo>.json`、`https://img.shields.io/github/last-commit/<owner>/<repo>.json`（pandoc、typst、XlsxWriter、quarto-cli、LibreOffice/core、zotero-mcp、CSL styles、markitdown、pymupdf4llm、lua-filters、pypandoc、typst/templates、typst-py、python-docx、python-pptx、docxtpl 等逐仓查询）
- 逐月提交核验：`https://commits.ecosyste.ms/api/v1/hosts/GitHub/repositories/<owner>%2F<repo>/commits/?since=YYYY-MM-DD&until=YYYY-MM-DD&per_page=1`（每月一次；GitHub API 匿名 403 限流，HTML 页截断）
- PyPI 版本/发版日期：`https://pypi.org/pypi/pypandoc-binary/json`（1.17，2026-03-14）、`https://pypi.org/pypi/typst/0.15.2/json`（2026-11-05，win_amd64 abi3）、`https://pypi.org/pypi/openpyxl/3.1.5/json`（2024-06-28，源 foss.heptapod.net/openpyxl/openpyxl）、`https://pypi.org/pypi/docxtpl/0.20.2/json`（2025-11-13，LGPL-2.1，依赖 python-docx）
- pandoc 3.14 与 pptx/xlsx reader 证据提交：
  - https://github.com/jgm/pandoc/commit/befc07b447d46e2c54a505925a7d37516a156215（Bump to 3.14，2026-11-11）
  - https://github.com/jgm/pandoc/commit/8d3137e5f4cea5067f5155804575903a75de3189（Pptx reader: read charts as tables）
  - https://github.com/jgm/pandoc/commit/1744ea89a95e01ec145e11b2bf085a8bf4e64abf（Pptx reader: bold/italic/hyperlinks）
  - https://github.com/jgm/pandoc/commit/41d9ef8eafb343728722bf703b45028e53ee2fc0（Xlsx reader: read charts）
- 许可证原文：`https://raw.githubusercontent.com/54yyyu/zotero-mcp/main/LICENSE`（MIT）、`https://raw.githubusercontent.com/pymupdf/pymupdf4llm/main/LICENSE`（AGPL-3.0）
- DSH 内置插件（本地核实）：`D:\XXXIIIGGG\projects\pySci\pySciWS2\sub_root\deps_management\deps\node\modules\node_modules\@deepseek-ai\dsh-skill-office\{package.json,README.md}`，依赖 `@deepseek-ai/libreoffice-kit`
- 旧能力清单来源：`D:\XXXIIIGGG\projects\pySci\pySciWS\orchestration\skills\document-writing\SKILL.md`、`src\pysci\skills\document_writing\tools\*.py`（18 模块）、`tests\skills\document_writing\*.py`（22 测试）、`data\skills\document_writing\{templates,cache,projects}`
