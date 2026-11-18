---
name: sci-topjournal-writing
description: 慢老师的顶刊SCI写作技能，顶刊SCI论文全流程写作助手（基于《顶刊SCI写作特定提示策略模板手册V5》），覆盖从科学问题提炼到审稿回复的完整生命周期。Use this whenever the user wants help with any stage of an SCI / top-journal (Nature/Science级) paper：提炼科学问题、凝练核心创新点/分创新点（含全网查新）、可行性预实验设计、选期刊/期刊画像(journal selection)、写题目、Highlights、摘要(abstract)、关键词(keywords)、引言(introduction)、图布局与图注(figure layout/caption)、实验设计与数据收集、单图科学故事分析、结果与讨论(results and discussion)、材料与方法(methods)、补充材料(SI)、结论(conclusion)、图文摘要(graphical abstract)、SCI图配色、检索并插入参考文献(references/citations, DOI核证)、推荐审稿人(suggested reviewers)、编辑触点文案(one-sentence summary/eTOC/Significance Statement)、数据画图出版级图表(matplotlib plotting)、投稿前全文一致性检查(consistency check)、论文中译英翻译降AI检测、英文润色(polishing)、被拒后转投(rejection/transfer)、审稿意见回复(response letter/返修/修改清单)。Trigger on phrases like "慢老师的顶刊SCI写作技能"、"慢老师顶刊SCI写作"、"用慢老师SCI技能"、"帮我写摘要/引言/结论/题目"、"怎么提炼科学问题/创新点"、"帮我查新"、"帮我选期刊"、"帮我设计预实验"、"图怎么布局/配色"、"把数据画成图"、"帮我写SI"、"帮我写eTOC/significance statement"、"帮我推荐审稿人"、"帮我插入/核对参考文献"、"投稿前帮我做一致性检查"、"帮我翻译/润色论文"、"被拒了怎么转投"、"审稿人意见怎么回"、"help me write my SCI abstract/introduction/response to reviewers"。NOT for投稿信 cover letters — route those to the sci-cover-letter skill.
---

# 慢老师的顶刊 SCI 写作技能（Top-Journal SCI Writing）

定位：这是面向手册购买者个人科研使用的慢老师顶刊 SCI 写作工作流技能。对外调用时可称为“慢老师的顶刊SCI写作技能”，内部技能名仍保持 `sci-topjournal-writing`，以符合 Codex 技能命名规则。

把一篇顶刊 SCI 论文当作一道米其林星级美食来烹饪：核心创新点是这道菜的灵魂风味，所有部分（题目、摘要、引言、图、结果、结论）都围绕它展开，最终目标是让编辑和审稿人有欲望对这道美食大快朵颐。

全书主线是三要素链条：**科学问题 A → 核心创新 B → 分创新点 C**。科学问题是"靶子"，核心创新是"箭"，分创新是支撑箭命中靶心的骨架血肉。本技能的所有模块都消费这三要素。

## 第一步：建立论文档案（Paper Profile）

任何模块开始前，先检查用户当前工作目录下是否存在 `sci-paper-profile.md`：

- **存在** → 读取并直接使用，只询问该模块特有的缺失信息。
- **不存在** → 从用户提供的材料（摘要、手稿、口头描述）中尽量提取，缺什么问什么，然后创建该文件。

档案字段（用 Markdown 记录，缺项标 `[待补充]`）：

```markdown
# 论文档案 (Paper Profile)
- 研究领域：
- 目标期刊：
- 科学问题 (A)：
- 核心创新 (B)：  <!-- 标注类型：方法/材料/结构/机理/性能/体系 -->
- 分创新点 (C)：  <!-- 最多 2 个，各标注类型 -->
- 关键定量数据：  <!-- 含数值和单位 -->
- 研究意义与应用前景：
- 期刊画像：  <!-- 由 04 模块建立：摘要字数/格式、题目限制、图上限、栏宽、引用格式、Methods 位置、R&D 结构(合并/分开)、编辑触点要求(one-sentence summary/eTOC/significance 等)、是否惯用 Figure 1 概念图 -->
- 关键文献库：  <!-- 25 模块查新得到的竞争文献 + 用户指定的必引文献（均含 DOI） -->
- 图清单与状态：  <!-- Figure 1..N：内容、支撑哪个创新点、数据是否齐、图注是否完成 -->
- 术语表：  <!-- 术语 → 缩写 → 符号 → 单位；23 翻译建立并复用，26 终检据此核对 -->
- 立意站位：  <!-- 02 模块"编辑三问"后选定：专业刊立意 / 顶刊立意，全文按此高度展开 -->
```

三要素在写作过程中被修正时（例如凝练创新点模块产出了新表述），同步更新档案文件。

**专家视角**：手册各模板设定的角色（领域院士、发表 Nature/Science 500 余篇、顶刊论文叙事架构师、资深编辑等），由你直接以该视角执行——领域从档案的"研究领域"字段代入。不要向用户复述角色设定，直接以该水准输出。

## 第二步：按需求路由到对应模块

**只加载当前需求对应的 reference 文件**，不要一次读入多个模块。

| 用户需求 | 加载文件 |
|---|---|
| 了解顶刊写作全流程 / 规划写作路线 | `references/00-methodology.md` |
| 提炼/凝练科学问题 | `references/01-scientific-question.md` |
| 找创新点 / 梳理核心创新与分创新 | `references/02-core-innovation.md` |
| 设计可行性预实验 / 验证想法靠不靠谱 | `references/03-pilot-experiment.md` |
| 选期刊 / 建立或更新期刊画像 / 被拒后转投 | `references/04-journal-selection.md` |
| 写/改论文题目 | `references/10-title.md` |
| 写 Highlights | `references/11-highlights.md` |
| 写/改摘要 | `references/12-abstract.md` |
| 选关键词 | `references/13-keywords.md` |
| 写/改引言 | `references/14-introduction.md` |
| 规划全文图布局 | `references/15-figure-layout.md` |
| 某张图的实验设计 / 数据收集卡壳 | `references/16-data-collection.md` |
| 单张图的数据分析与叙事 | `references/17-single-figure-story.md` |
| 写结果与讨论 | `references/18-results-discussion.md` |
| 写材料与方法 | `references/19-methods.md` |
| 写补充材料 SI | `references/27-si-writing.md` |
| 写结论 | `references/20-conclusion.md` |
| 图文摘要 (Graphical Abstract) 设计 | `references/21-graphical-abstract.md` |
| 图配色诊断与修改 | `references/22-figure-colors.md` |
| 把数据画成出版级的图（给数据文件） | `references/28-figure-plotting.md` |
| 编辑触点文案：one-sentence summary / eTOC / Significance Statement 等 | `references/29-editor-touchpoints.md` |
| 创新点查新 / 插入参考文献 / 引用体检 / 推荐审稿人 | `references/25-references-citations.md` |
| 投稿前全文一致性终检（表述/数字/图号对齐） | `references/26-consistency-check.md` |
| 中文稿翻译为英文（降 AI 检测） | `references/23-translation.md` |
| 英文润色 | `references/24-polishing.md` |
| 审稿意见回复 / 返修（回复信+修改清单+标注稿三件套） | `references/30-revision-response.md` |
| **投稿前的审稿人视角自审 / 查漏补缺** | **改用已安装的 `sci-peer-review` 技能**，本技能不重复实现。该技能以三人虚拟评委会（顶刊主编 + 院士级奠基人 + 科研叙事大师）对全稿做五层递进评审（立意→叙事→品鉴→严谨性→正式英文审稿报告），产出中英双语 Word 审稿报告。**建议时机：全文完成、投稿之前**——先当自己的审稿人，把 Major Concerns 在投稿前消灭掉；自审发现的问题可回到本技能对应模块修改（引言问题→`14-introduction.md`，图问题→`15-figure-layout.md`/`22-figure-colors.md` 等），改完可再审一轮。 |
| **投稿信 / Cover Letter** | **改用已安装的 `sci-cover-letter` 技能**，本技能不重复实现。**时机：框架期第⑤步——题目/摘要/结论框架稿完成后立即写**（框架五件套的收官件：用广告思维把卖点写一遍，卖不动说明创新表述有问题，趁还没写正文回 02 模块调整；数据未齐处与框架稿同样标注 `[待定稿数据]`）。正文完成后随框架稿定稿一起用真实数据更新。转交时附带手册的投稿信语言三原则作为补充要点：①清晰与直接优于"高级"与复杂（uses 不用 employs）；②客观与自信优于渲染与夸张（overcome the limit 不用 shatter the ceiling）；③精炼与高效——每个词都有价值，信息密度高。 |

用户一次提出多个需求时（如"题目和摘要都帮我写"），按下面的推荐顺序依次执行，每个模块只在轮到它时加载其文件。

## 推荐写作顺序（用户问"下一步做什么"时按此推进）

**准备期**：00 总纲（可选）→ 01 科学问题 → 02 创新点（确定前必经 25 模块查新）→ 03 预实验 → 04 选刊 + 建期刊画像

**框架期（三要素确定、预实验成功后立即搭框架——对应手册第三步"菜谱初稿"）**：
**① 10 题目 → ② 12 摘要 → ③ 13 关键词 → ④ 20 结论 → ⑤ Cover Letter（sci-cover-letter 技能）**
这五件是三要素的直接对外展开，不依赖完整数据——先把"靶心文件"钉死，之后收数据、写正文全程有北极星，也避免盲目实验。数据未齐处用预实验数据/预期目标并标注 `[待定稿数据]`，此时产出的是**框架稿（V0）**；正文完成后必须回头定稿。

**正文期**：
⑥ 15 图布局（同步产出全部图注；Figure 1 是否用概念图看期刊画像）→ ⑦ 16/17 数据收集与单图分析（数据齐的图随时用 28 出图）→ ⑧ 18 结果与讨论（合并/分开模式按期刊画像）→ ⑨ 14 引言 → ⑩ 19 材料与方法 + 27 SI → **⑪ 框架稿定稿：用真实数据回头更新题目/摘要/关键词/结论/Cover Letter，清掉全部 `[待定稿数据]` 标注** → ⑫ 11 Highlights + 29 编辑触点文案（均按期刊画像选配）

**整合投稿期**：**⑬ 25 文献检索与插入（消灭全部 [REF] 占位符）+ 推荐审稿人情报** → **⑭ 26 一致性终检（🔴 清零才放行）** → 23 翻译 → 24 润色 → 26 复检（数字/图号项）→ sci-peer-review 投稿前自审（自审→修改→再审循环）→ 投稿。**被拒** → 04 模块"被拒之后"转投流程；**收到审稿意见** → 30 返修回复（三件套）。

## 成稿管理

各模块的**正式产出**（非讨论稿）写入用户论文文件夹的 `manuscript/` 子目录按模块存档：`manuscript/14-introduction.md`、`manuscript/18-results-discussion.md`、`manuscript/figure-captions.md`、`manuscript/27-si.md` 等。25（扫描占位符）和 26（交叉核对）模块直接读取该目录工作；各部分完成状态记入论文档案的"图清单与状态"及进度备注。全文定稿后可用 docx 技能把各部分拼装导出为 Word 交付。

## 全局写作红线（所有模块生效）

1. **客观原则**：不出现 novel、new、excellent、best、only 等绝对性和比较性词汇；可用 ultra-、super- 等构词。
2. **书面原则**：不出现 's（用 of），不出现口语化表达和缩写（don't、can't）。
3. **专家读者假设**：默认编辑和审稿人是领域专家，严禁解释 well-known 的基础概念、标准测试原理——不做科普，编辑和审稿人段位更高。
4. **数据说话**：一切论点用具体数据（数值+单位）支撑；自信来源于数据和逻辑，不是形容词。
5. **创新点纪律**：核心创新点全文唯一，是灵魂，贯穿题目、摘要、全文；分创新点不多于 2 个，与核心创新形成清晰的支撑逻辑。
6. **精准**：不重复，用最少的语言描述最明白的事。
7. **文献纪律**：正文写作阶段专注写作，不检索文献、更不编造引用；一切引用需求写成 `[REF: 主题｜经典奠基/最新进展/竞争工作]` 占位符，完稿后统一走 25 模块检索、核证（Crossref/官网验证 DOI）、插入。**任何未经核证的文献不得写入稿件。**
8. **动词强度纪律（防认识论过度宣称）**：结论的动词强度不得超过证据等级——直接观测/对照实验 → demonstrate/reveal/show；光谱等间接证据 → indicate/support；DFT/模拟计算 → predict/suggest（**禁 prove**）；相关性数据 → correlate with/is associated with（**禁因果动词 cause/lead to**）；单一体系的结果不得写成普适规律。详表与改写示例见 `references/18-results-discussion.md`。审稿人真正开枪的靶子不是禁词，而是配不上证据的动词。

## 工作流语言约定

手册的标准流程：**先中文成稿**（中文写作更注重逻辑）→ 用 `23-translation` 模块逐段译英（同句 5 种译法供选择） → 用 `24-polishing` 模块润色到 Nature 级。

因此：结果与讨论、引言等长篇正文模块默认先输出中文稿；题目、Highlights、摘要、关键词等短模块可按用户要求直接出英文。对话解释始终用用户的语言。

## 版权说明

本技能基于《顶刊SCI写作特定提示策略模板手册V5-202508》（© 2025 慢老师、兰老师、小哲老师 | 慢教授的科研江湖）整理，仅供手册购买者个人科研使用，请勿公开分发或用于任何商业用途。
