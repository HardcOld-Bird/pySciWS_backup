# legacy-01：旧项目「中英对照生产线」解剖（只读分析）

> 2026-11-17 · 分析对象 `D:\XXXIIIGGG\projects\pySci\pySciWS`（只读，未复制任何代码）
> 目的：为 pySciWS2 是否吸收该能力提供组织/实现/耦合事实底座。本文不含社区选型结论（另轮）。

---

## 1. 资产定位总览

旧项目中"中英对照"能力的正式名是 **bilingual 中英对照文档生产线**，挂在
**literature-research（文献研究）技能域**下，入口是 `pysci-research bilingual <子命令>`。
全部资产清单（行数为精确计数）：

| 类别 | 路径（相对 pySciWS） | 行数 | 角色 |
|---|---|---|---|
| 核心实现① | `src/pysci/skills/literature_research/tools/bilingual.py` | 1274 | 环节 1：切分/工作区/对账/出稿 CLI 管线 |
| 核心实现② | `src/pysci/skills/literature_research/tools/bilingual_render.py` | 383 | 环节 2/3：bilingual.md → HTML → PDF 渲染 |
| 核心实现③ | `src/pysci/skills/literature_research/tools/reorder_suggest.py` | 540 | 旁路：源 md 文序订正建议（bilingual-orderfix） |
| CLI 壳 | `src/pysci/skills/literature_research/tools/research.py`（cmd_bilingual ≈3550 行起） | ~30 | 参数分发到上述模块（`bilingual prepare/status/submit/render/reorder-suggest`） |
| 编辑守卫 | `orchestration/guards/pod-guard.mjs`（work.md 硬校验段） | ~50 | PreToolUse 钩子：work.md 行数/行首 ID 不得改动 |
| 技能文档 | `orchestration/skills/literature-research/references/bilingual.md` | 97 | lit 操作分册（命令面/契约/故障手册） |
| 模板 | `data/skills/literature_research/templates/bilingual.css` | 96 | PDF 版式真源（双栏 A4、0.6cm 页边距、图注分级……） |
| 模板 | `data/skills/literature_research/templates/katex/`（KaTeX 0.19.0 本地 bundle + woff2 字体 + SOURCE.txt） | 约 600KB | 公式排版资产（MIT，注明选型理由：同步排版、bundle 小） |
| 测试 | `tests/skills/literature_research/test_bilingual_split.py` | 220 | 句切分/结构解析回归 |
| 测试 | `tests/skills/literature_research/test_bilingual_pipeline.py` | 731 | prepare/status/submit 管线回归 |
| 测试 | `tests/skills/literature_research/test_bilingual_render.py` | 483 | 渲染层回归（版式/宽元素/图注粒度） |
| 测试 | `tests/skills/literature_research/test_bilingual_guard.py` | 226 | pod-guard work.md 钩子回归（真 node 直跑） |
| 测试 | `tests/skills/literature_research/test_reorder_suggest.py` | 140 | 文序建议回归 |
| 任务书/计划 | `data/others/taskbooks/bilingual-pipeline.*.md`、`bilingual-orderfix.*.md`、`orchestration/state/plans/bilingual-*` | 约 10 件 | 开发过程任务书（cli/render/verify/orderfix 五段） |
| 生产脚本 | `orchestration/pods/lit/bench/`：audit_layout.py(185)、merge_parts{,2,3}.py(88/65/65)、probe_slice/probe_pysbd/probe_split_drift.py(45/36/47)、recover_translations.py(73)、report_metagrating/metamirror.py(154/132)、build_review_md.py(209)、build_metagrating_main.py(102) 等 | 约 15 份 | 实产中的一次性排版抽查/断点续译/交付报告脚本（非工具链） |
| 数据 | `data/skills/literature_research/bilingual/<stem>/`（6 个 stem 实产工作区）+ `data/用户的inbox/.../bilingual/` 交付件 | 6 篇论文 | work.md / manifest.json / images/ / bilingual.{md,pdf,html} / reorder-* |

**实现规模合计：实现 2197 行 + 测试 1800 行 ≈ 4000 行**（不含 bench 脚本与 vendored KaTeX）。

---

## 2. 组织方式

### 2.1 架构位置与交互面

```
literature-research 技能域（旧架构 src/pysci/skills/literature_research/）
├─ 既有上游（bilingual 的输入来源，不重复实现）
│   ├─ pdf_extract/read → MinerU 风格 markdown → cache/extracted/<stem>.md
│   └─ cache/pdfs/（源 PDF，按 slug/DOI/arXiv id 命名）
├─ tools/bilingual.py        ← 环节1 prepare/status/submit（CLI 子命令）
├─ tools/bilingual_render.py ← 环节2 render（submit 自动串联）
├─ tools/reorder_suggest.py  ← 旁路 reorder-suggest（源 md 文序坏了先修）
└─ 交互的其他模块
    ├─ cache_manager.find_cached_pdf / title_consistency_note（PDF 反解+题名一致性）
    ├─ openalex_client.get_work（DOI 元数据题名对照，离线降级）
    ├─ browser_fetch._launch（playwright 浏览器启动链：chrome→edge→bundled chromium）
    ├─ pysci.paths（DATA_ROOT/assert_within_data 越界护栏）、config.settings（缓存路径/mineru_token）
    └─ orchestration/guards/pod-guard.mjs（work.md 编辑硬校验，DSH PreToolUse 钩子）
```

- **技能域归属**：literature-research；技能文档在 `orchestration/skills/literature-research/references/bilingual.md`（SKILL.md 索引"中英对照需求时读"）。
- **与文档写作域无耦合**：产物是 md/pdf/html，不经 docx/pandoc-office 链路，版式自成一套。
- **与 PDF 提取产物强相邻**：输入就是 MinerU 抽取的 markdown + 源 PDF；图片回补直接调 MinerU 云端重抽，PyMuPDF 按页序配对兜底。

### 2.2 工作流（输入→阶段→输出）

```
输入：MinerU 风格源 md（cache/extracted/<stem>.md）+ 源 PDF
  │
  ├─ [可选修复] bilingual reorder-suggest
  │     源 md 文序错乱（图注残段散落/公式被图隔开）时：基于源 PDF 文本块坐标
  │     推断真实阅读序 → reorder-suggested.md + reorder-report.md（只出建议，人工落盘）
  │
  ▼ 阶段1  bilingual prepare
  │  parse_markdown 切单元（H 标题/S 句/C 图注/T 表注）+ 透传块（公式/表格/图片行/参考文献）
  │  → verify_slices 切片自检（拼回源文须无损，不过即拒）→ mark_abstract 摘要识别
  │  → 产 work.md（每行 "ID<TAB>英文原文"）+ manifest.json（单元/透传块/元信息）
  │  → 图片回补 images/（MinerU 云端按名落盘 → PyMuPDF 顺序配对兜底，引用名闭合才放行）
  │  → stdout 翻译指引；--reuse 按英文原文等值回填旧译文；含译文工作区默认拒覆写
  │
  ▼ 翻译循环（唯一的人工/LLM 环节）
  │  译者（lit 组员=LLM 代理，或人）逐句把 work.md 的英文替换为中文：
  │  保留行首 ID、不增删行（pod-guard 钩子硬校验）、行内 $公式$/引文照抄、术语先读源文档统一
  │  bilingual status 随时对账（已译/未译/存疑/占位假译计数）
  │
  ▼ 阶段2  bilingual submit
  │  全部 ID 有真译文（CJK 判据 + 占位假译拒绝）才出稿 → bilingual.md
  │  **英文侧从 manifest 程序化重嵌入（逐字一致，不取 work.md）**，中文侧取 work.md
  │  → 自动串联渲染 bilingual.pdf（失败仅警告，md 照常交付）
  │  → 复制到交付目录（--out-dir 或 data/用户的inbox/bilingual/<stem>/，按中文标题命名，md/pdf 成对）
  │
  ▼ 阶段3  bilingual render（独立重试入口）
     bilingual.md → pandoc(--mathjax) HTML → 自包含 HTML（内联 CSS/KaTeX/字体 data:URI）
     → playwright 系统 Chrome headless 打印 A4 双栏 PDF → PyMuPDF 叠加奇偶页码
```

**输出格式**：工作区 `bilingual.md` + `bilingual.pdf` + `bilingual.html`（调试图）；交付件
md/pdf 成对（按文档中文标题命名）。没有 docx。bilingual.md 形态是**渲染层输入契约**：
标题 `# 中文`+`*English heading*`、句对=中文行+斜体英文行、段界 `---`、图原位图注紧随、
公式/表格/参考文献原样透传、摘要/正文中文独行小标题（无英文行，2026-11-03 裁决）。

---

## 3. 实现方式（逐模块）

### 3.1 bilingual.py（1274 行）——管线与契约

| 段 | 职责 | 关键技术 |
|---|---|---|
| 句切分 `split_sentences` | 段内软换行合并 → **pysbd** 切句 | 28 个学术缩写 + 省略号换哨兵防误切；句末只认 `. ? !`；**游标法按原文取碎片间分隔**（保证逐字拼回，防 `Fig. 3(a)` 回并漂移） |
| 结构解析 `parse_markdown` | 源 md → Unit(H/S/C/T) + PassBlock(formula/table/image/reference) | 正则状态机；图注/表注按 `Fig./Table/图/表` 前缀分类（无前缀一律当正文，宁漏勿错）；图与其图注同 para；ID=`类型字母+4位序号` |
| 切片自检 `verify_slices` | 各块按 order 拼回源文须无损 | 归一化空白后逐块 `find` 存在性核对，错位即拒 |
| 摘要识别 `mark_abstract` | 元数据/作者行（Received/DOI/角标）之后首段全组标 `role=abstract` | 启发式 + manifest 可手改 |
| 图片回补 `backfill_images` | MinerU 云端重抽按名落盘；不闭合→PyMuPDF 按页序顺序配对（告警） | mineru SDK（`model=vlm, ocr, formula, table`）+ fitz；`images_closed` 为 true 才许出稿 |
| PDF 反解 `_find_source_pdf` | stem → md 头部 DOI/arXiv id → 标题 slug 三级反解 | cache_manager.find_cached_pdf；题名一致性三面对照（PDF 内容题名/抽取件题名/OpenAlex DOI 题名）只告警 |
| 对账 `classify_translation/reconcile` | 已译/未译/存疑三桶 | 判据：缺 ID 或==英文=未译；含 CJK=已译；其余=存疑。**占位假译**（`假译/旧译-` 前缀族或"仅差前缀"等值）记未译、拒出稿、--reuse 不回填（2026-10-23 事故防复发） |
| 出稿 `render_bilingual_md` | manifest+work.md → bilingual.md | **英文侧从 manifest 重嵌入**（构造保证逐字一致）；摘要/正文标注、段界 `---`、句对空行全在此层 |
| 译文复用 `load_reuse_map` | --reuse 旧译回填 | 归一化空白后**英文原文等值**匹配（无 TM 模糊匹配） |
| 防覆写闸 `guard_translated_workspace` | 含译文工作区再 prepare 默认拒；--force 自动备份 `*.bak-<ts>` | 数据丢失口的机械预防 |
| 命名 `output_basename` | 交付件按文档中文标题命名 | 非法字符清洗/60 字截断/空回退 `bilingual` |

### 3.2 bilingual_render.py（383 行）——排版层

- **管道**：契约 md → `to_pandoc_md` 预处理（`---`→`***` 防 pandoc YAML 吞行；`*英文行*`→`<em>` 原样内联；文本行 markdown 转义而 `$..$` 数学保留）→ **pypandoc**（`--mathjax` 出 `\(...\)`/`\[...\]` 语义定界）→ 自包含 HTML（内联 bilingual.css + KaTeX bundle + woff2 字体 data:URI，零外部请求）→ **playwright** 系统 Chrome headless `page.pdf`（A4、四边 0.6cm、printBackground）→ **PyMuPDF** 逐页叠加页码（奇数页右下、偶数页左下，落页边距带）。
- **版式控制**：样式真源 = `bilingual.css`（全文双栏、栏间细线、中文正文衬线 11pt、英文斜体灰小、图注小字分级 `.caption`/`.caption-cont`）。
- **宽元素策略**（复杂度热点）：页内 JS 测元素 **min-content 宽**（临时 `width:min-content` 重排读 rect，公式加 `nowrap`），与打印栏宽（由 A4/页边距/栏距算出，注入 `__COL_W_PX__`）比较，超宽 `zoom` 等比缩放进栏——**只缩不放、禁跨栏**（2026-11-03 用户二轮裁决删跨栏特判）。文档内留有事故注记：早先用 `scrollWidth` 量宽导致 13/13 公式误判跨栏（"公式切断双栏"事故根因）。
- **图注粒度**（另一热点）：图片段落后的句对组，**首对与图同包 `.figure` 原子**（防栏/页断裂），余对 `.caption-cont` 自由流布——三轮修复后定案（长图注整组原子会跳栏留洞）。
- **KaTeX 选型注记**（SOURCE.txt 内原地 ADR）：弃 MathJax 选 KaTeX——同步排版无 typeset 时序坑、bundle 小、auto-render 即开即用。

### 3.3 reorder_suggest.py（540 行）——文序订正建议（唯一带"对齐"色彩的模块）

- 输入源 md + 源 PDF → PyMuPDF 取全部文本块 bbox → **排版带聚类**（同页 y 交叠并簇；文本带"左栏自上而下→右栏自上而下"，图表带按 (y,x)）→ md 块与 PDF 块**文本相似度映射**（difflib SequenceMatcher + 词元包含度 + 首锚三路取大，阈值 0.45；公式走 `\tag{N}`↔PDF `(N)` 尾锚、面板标签走全等）→ 同一 PDF 块的 md 块归组同位（图注残段自动缝回）、图片挂最近图注、标签挂相邻图组 → 输出 `reorder-suggested.md`（块原文逐字，便于 diff）+ `reorder-report.md`（原位→建议位 + PDF 页/带/栏/y 依据 + 类型）。
- **设计立场**：只出建议不硬排（无 `--apply`），映射不上单列"未映射"交人工——人工校对是设计的一部分。

### 3.4 双语来源 / 对齐 / 质控——三个关键问题的直接回答

| 问题 | 旧实现答案 |
|---|---|
| 双语从哪来？ | **无翻译 API、无机器翻译引擎**。英文=源文档切片；中文=翻译组员（LLM 代理/人）逐句替换 work.md 行尾。设计原则"**除翻译本身外一切琐事程序化**"。术语一致性靠纪律（"先读源文档统一"）+ 实产中手工维护的 per-job 术语表（bench/glossary_*.md 与交付报告"s1 术语对照"节），**无术语库/记忆库机制**；仅有 `--reuse` 英文原文等值回填（防整篇重译）。 |
| 对齐机制？ | **不做双文本对齐**——对齐由构造保证：单元 ID 寻址（`H0001/S0001/...`），英文侧出稿时从 manifest 程序化重嵌入（误改 work.md 英文行不伤交付物），句对/段界/图注位置全由 manifest 的 order/para 字段确定性还原。粒度=句级（标题/图注/表注为单元）+ 段落边界标记。真正的"对齐算法"只有 reorder_suggest 的 md↔PDF 块映射（修源文序，不是双语对齐）。 |
| 质量控制？ | 全部机械闸门：①切片自检（拼回源文无损）；②未译/存疑/占位假译三桶对账，未译或含假译**拒绝出稿**；③图片引用名闭合校验；④题名三面一致性告警（OpenAlex）；⑤pod-guard 行结构硬校验；⑥PDF 版式抽查脚本（audit_layout：页码奇偶/页边距/栏宽实测渲染 PNG 目检）。**无回译校验、无 LLM 质量评估、无术语一致性自动核对**。 |

### 3.5 第三方依赖清单

| 依赖 | 用途 | 备注 |
|---|---|---|
| pysbd | 英文句切分 | 唯一纯算法依赖 |
| pypandoc（+pandoc 二进制） | md→HTML | 环境要求 |
| playwright + 系统 Chrome/Edge | HTML→PDF 打印 | 复用 browser_fetch._launch 启动链 |
| PyMuPDF（fitz） | 图片抽取兜底/页码叠加/PDF 文本块坐标 | 三处复用 |
| mineru SDK（云端） | 图片回补重抽 | 网络/配额依赖，失败可降级 |
| KaTeX 0.19.0 | 公式排版 | vendored bundle（MIT，SOURCE.txt 记来源与选型） |
| stdlib difflib/re/json/dataclasses | 相似度/解析/契约 | — |

### 3.6 测试面（1800 行，5 文件）

离线确定性设计（MinerU/渲染打桩或临时 stem）；render 测试锁死版式契约（CSS 断言/宽元素缩放/图注粒度/空 alt 图片规则）；guard 测试用 subprocess 真跑 `node pod-guard.mjs`（不 mock）；pipeline 测试覆盖 --reuse/命名/摘要标注/占位拒绝/防覆写。测试即契约文档——bilingual.md 形态"已由 pipeline 测试锁死"。

---

## 4. 复杂度与耦合评估

### 4.1 可迁移的"纯逻辑"（自包含、与旧基础设施无耦合或弱耦合）

| 模块段 | 行数量级 | 说明 |
|---|---|---|
| 句切分 + 缩写保护 + 游标逐字保证 | ~90 | 仅依赖 pysbd |
| parse_markdown / verify_slices / Unit/PassBlock 契约 | ~240 | re + dataclasses |
| work.md/manifest 契约、对账三桶、占位判定、命名 | ~250 | 纯函数 |
| render_bilingual_md（bilingual.md 形态生成） | ~50 | 纯函数，但形态契约与渲染层成对 |
| reorder_suggest 全模块（排版带聚类/相似度映射/建议报告） | ~540 | 仅 difflib + fitz，**是最可整搬的一块** |
| to_pandoc_md / build_html / add_page_numbers | ~250 | 依赖 pypandoc/fitz + 模板目录（模板可随迁） |
| bilingual.css + KaTeX bundle | 96 行 + 600KB | 数据资产，可原样随迁（LICENSE 齐备） |

### 4.2 与旧基础设施深度耦合、迁移需重写/替换的部分

| 耦合点 | 涉及 | 迁移动作 |
|---|---|---|
| `pysci.paths`（DATA_ROOT/assert_within_data） | bilingual.py 全部路径 | 换新项目路径层（薄） |
| `config.settings`（module_dir/cache_extracted/cache_pdfs/mineru_token） | 源解析、PDF 反解、模板定位 | 换新配置面（薄） |
| `cache_manager.find_cached_pdf / title_consistency_note` | PDF 三级反解、题名一致性 | 逻辑归 cache 基础设施，需重接或重写 |
| `openalex_client.get_work` | 题名对照（可选降级） | 换社区 OpenAlex 客户端或删 |
| `browser_fetch._launch` | playwright 浏览器启动链 | 换独立 launch（薄） |
| mineru SDK 云端 | 图片回补 | 与新项目 MinerU 云提取统一（literature-research 域已有同款依赖） |
| `research.py` CLI 壳 | 命令注册 | 重接新 CLI（薄） |
| `orchestration/guards/pod-guard.mjs` work.md 段 | 编辑期行结构硬校验 | **旧编排基础设施件**；新项目若无 PreToolUse 钩子体系需另找等价护栏（或退化为纯 CLI 校验） |
| INBOX 交付约定（data/用户的inbox、只增不删） | submit 末步 | 换新项目交付约定（薄） |

### 4.3 复杂度热点（重写时最易踩坑，均有事故注记/测试钉死）

1. `split_sentences` 游标逐字拼回（"切片自检失败"类 bug 的根源，pysbd 会在面板号前误断）；
2. 渲染层宽元素量宽口径（scrollWidth 事故 → min-content + 打印栏宽双口径）；
3. 图注粒度（图+首对原子、余对自由流布的三轮裁决）；
4. 占位假译判定（夹具混入交付物事故 → 前缀族 + "仅差前缀"等值双判据）；
5. reorder 的公式 `\tag{N}` 尾锚与 PDF 伪字形（`ð/Þ`）适配。

**总体判断**：这是一条"**契约驱动的流水线**"——真正值钱的不是某个算法，而是三份契约
（work.md 行契约、manifest 切片契约、bilingual.md 渲染契约）+ 一排机械闸门。
实现本体约 2200 行，纯逻辑占比高（估计 70%+ 可整段平移），耦合集中在路径/缓存/浏览器
启动/编辑守卫四个接缝，全部是"薄"接缝，唯 pod-guard 依赖旧编排体系需另想等价物。

---

## 5. 吸收启示（按"技能文档+社区工具+薄胶水"拆层，点到为止）

| 层 | 旧实现 | 吸收方向（社区候选概念，不展开） |
|---|---|---|
| **翻译引擎层** | LLM 代理/人逐句改 work.md（无 API） | 社区机器翻译/LLM 翻译库（deep-translator、Argos Translate、transformers MT 类）可作引擎候选；术语库/记忆库可看 TMX/术语管理类工具——旧实现此层最薄，恰是社区化空间最大处 |
| **对齐/结构层** | pysbd 句切 + manifest ID 寻址（构造式对齐）+ reorder 的 md↔PDF 块映射 | pysbd 本身即社区件可留；PDF 结构/阅读序可看 PDFMiner.six、Docling、MinerU 全管线等文档解析社区件；双语对齐如需真对齐可看 sentence-aligner/GALE 类（旧实现不需要） |
| **排版层** | pandoc + 自包含 HTML + Chrome headless 打印 + PyMuPDF 页码 + KaTeX | 全链本就是社区件组合；候选方向：Typst（原生双栏/中文排版）、WeasyPrint、pandoc 模板体系——bilingual.css/KaTeX bundle 可作为样式资产保留 |
| **质量层** | 纯机械闸门（对账/占位/闭合/自检/版式抽查） | 闸门逻辑是纯胶水可平移；可补社区候选：回译/QE 类（COMET/sacrebleu 系）、术语一致性核对、LLM 评审——旧实现空白处 |
| **薄胶水（保留自研）** | 三份契约 + CLI 状态机 + pod-guard 等价护栏 + 交付命名 | 建议原样吸收为薄层；pod-guard 的 work.md 行结构校验可改造成独立 CLI 校验/编辑前检查，脱离旧编排 |

**一句话结论**：旧件是一条契约驱动、机械闸门齐全、翻译环节刻意留给"人/LLM 代理"的
句级中英对照生产线（约 2200 行实现 + 1800 行测试，6 篇实产验证）；可迁移纯逻辑占比高，
值得吸收——建议按上表五层拆解，其中排版层与翻译引擎层优先走社区，三契约+闸门作薄胶水保留。
