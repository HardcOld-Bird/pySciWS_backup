# 中英对照生产线吸收实施计划（状态：草案，待用户确认）

> 三段式第二段产物。输入：`community_research/2026-11-17/legacy-01-bilingual-parallel.md`（旧件解剖）、
> `bilingual-absorption-survey.md`（五层调研 + 用户裁决记录）。归属 **①文献调研域**（用户裁决）。
> 完成后归档至 `archived/`。

## 1. 范围与总账

吸收旧 bilingual 生产线（旧 2197 行自研 + 1800 行测试），按五层重构为"社区工具+薄胶水"：
**自研 ≈1850 行（含新增 TM/回译/术语能力）/ ≈1650 行（等价口径），测试 1500-1600 行平移护航**。
不做：web MT 聚合器类（全停更）、COMET/OpenKiwi（重/死）。

## 2. 选型定稿（用户裁决）

| 层 | 引入 | 口径/豁免 |
|---|---|---|
| 翻译主通道 | 译者=pod（LLM/人逐句改 work.md，不变） | 旧契约平移 |
| 自动填充 | deepl-python（MIT 全过）；argostranslate **可选件豁免 pin**（离线无可替代） | 豁免 |
| TM/术语 | translate-toolkit（GPL-3 **已接受**，TMX/TBX+tmdb 模糊匹配，升级旧 --reuse） | 全过 |
| 句切 | **wtpsplit 豁免 pin**（小模型 CPU 推理，SaT 模型 vendor 入项目内）；备选 spaCy | 豁免（②仅 9 月提交） |
| MD 解析 | markdown-it-py（1.4k★ 全过）替正则状态机 | 全过 |
| **排版** | **(a) HTML 双栏链**：pandoc→自包含 HTML（KaTeX）→Chrome 打印→PyMuPDF 叠页码；平移 bilingual.css + 模板 | 按"自研量小"双指标胜出 |
| 质量层 | 机械闸门平移 + 回译校验/术语一致性核对（~80 行）；sacrebleu 可选件豁免 | — |
| 薄胶水 | 三契约 + CLI 状态机 + check-work 护栏 + 交付约定 ~570 行 | 无可替代（自研合理区） |

## 3. 实施步骤

1. **依赖引入**：deepl-python、translate-toolkit、markdown-it-py、wtpsplit（pin）+ argostranslate/sacrebleu
   （可选件 pin）；wtpsplit SaT 小模型 vendor 至项目内（CPU 推理）；浏览器打印链依赖 playwright + Chrome
   （⑤域已有 LibreOffice/pandoc 链，Chrome 若缺则 `plotly_get_chrome` 同款渠道或用户装机，实施时核对）。
2. **技能文档**：`.agents/skills/literature-research/` 下增 `references/bilingual.md`（或独立子技能）：
   prepare→翻译（pod/TM 回填）→submit→render→reorder-suggest 工作流、契约说明、质量闸门。
3. **逻辑平移与重接**（自研 ≈1850 行）：
   - 平移（~980 行）：切分/manifest 契约/对账三桶/防覆写、reorder_suggest 540 行、渲染器 ~310 行、bilingual.css；
   - 新写（~870 行）：CLI 状态机与护栏 ~570、TM 集成（translate-toolkit 回填）~150、回译/术语核对 ~80、
     启动链重写/接缝 ~60、wtpsplit/markdown-it-py 替换接缝；
   - 移除旧耦合：旧 paths/config/cache/旧 MinerU 封装/pod-guard.mjs（新护栏用新机制实现）。
4. **测试平移**：1500-1600 行测试适配新路径（483 行渲染测试随迁护航；旧期望值保留）。
5. **冒烟**：真实小文（短 PDF）全链 prepare→翻译→submit→render 产双栏 A4 PDF 目检（视策略 2.4 先 VLM 查图）。

## 4. 风险与约束

- **硬件约束**（用户口述）：仅 CPU/可用内存 ~5GB——SaT 小模型 CPU 推理、argos 仅按需（慢速接受）、
  浏览器打印避免并行多实例；
- wtpsplit/argostranslate 差②豁免 pin + ADR；
- translate-toolkit GPL-3（个人不分发，已接受）；
- playwright+Chrome 浏览器依赖（打印链）实施时核对可用性；
- 旧契约（manifest/逐字一致）是核心资产，平移时以旧测试期望为准绳不改语义。

## 5. 验收标准

真实短文全链产双栏 A4 中英对照 PDF（逐句对齐、页码、图注原子、宽元素缩进正确）；TM 回填生效；
质量闸门（占位假译拒出稿等）全过；`tests.cmd all` 全绿；归档。

## 6. 实施结果（2026-11-17 收尾记录）

- 验收标准**全部达成**（真实短文全链冒烟）：22 单元真译→submit 拒占位假稿实证→英文侧逐字重嵌入
  →双栏 A4 PDF（逐句对齐/页码奇右偶左/图注原子/KaTeX 公式，逐页目检全过）→ reorder 报告 →
  **薄 TM 22/22 精确回填实测**。交付副本在 `data\用户的inbox\bilingual\_smoke_biling_nemnes\`。
- 实施三路+TM 重写完成：核心 1754 行 + 渲染 448 + reorder 499 + 薄 TM（tmdb→自研 SQLite+RapidFuzz
  110 行，用户裁决方案 B，translate-toolkit 3.20.6）+ 测试 ~2300 行平移/新增。
- 冒烟抓修 2 个真实输入缺陷（MinerU 图+图注同段落剥离、"i. e."缩写哨兵），各带回归用例。
- 最终 `tests.cmd all`：**306 passed / 12 hardware** 全绿。
- 遗留：argos 实翻仅差句切分模型下载（网络超时，语言包已就位）；摘要作者行 `<sup>` HTML 识别；
  reorder 跨页摘录噪声；TMX/TBX 导入导出预留接口。均为非阻塞改进项。
