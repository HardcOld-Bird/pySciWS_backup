# bilingual-absorption-survey：中英对照生产线五层社区调研

> 2026-11-17 · 目标：pySciWS2 ①文献调研域吸收旧 pySciWS「bilingual 中英对照生产线」
> （旧件解剖见 `legacy-01-bilingual-parallel.md`，实现 2197 行 + 测试 1800 行）。
> 本文为吸收路径的社区调研底座，含五层候选表、排版层自研量量化对比、建议组合与待裁决点。

---

## 0. 口径与方法

**三口径**（用户裁决④）：① star ≥ 100；② 2026-05~11 每月有提交；③ 减自研（能替换旧自研/少写新自研）。
**豁免规则**：差②但**无可替代** → 豁免引入 + pin + ADR；**有替代** → 拒绝。

**取证方法**（2026-11-17，均为当日实测）：
- 仓库元数据：`api.github.com/repos/<o>/<r>`（stars / pushed_at / license / archived）；
- 逐月提交：`stats/participation`（52 周逐周提交计数，全窗口精确）+ GitHub commits atom feed
  （`commits/<branch>.atom`，回溯抽样）+ ecosyste.ms repos API（pySBD 交叉复核，含 OSSF scorecard）；
- PyPI：`pypi.org/pypi/<pkg>/json`（版本 / requires_python / 许可证 / wheel 形态）；
- CTAN：`ctan.org/json/pkg/<id>`（LaTeX 宏包版本/日期/许可证）；
- 徽章兜底：`img.shields.io/github/stars|last-commit/<o>/<r>`。
- 局限：`api.github.com` 未认证限流导致个别 participation 未能取到，已用 atom/pushed_at 交叉近似，
  表中逐月列标注证据强度：**[全窗口]** participation 覆盖全年 / **[抽样]** atom 覆盖数月 / **[近似]** 仅 pushed_at。

**Py3.13/Windows 基线**：新项目 `sub_root/pyproject.toml` 锁 `requires-python ==3.13.15`（Windows），
故一切原生扩展需有 win_amd64 cp313 wheel；纯 Python 包（py3-none-any）判 ✓。

---

## 1. 翻译引擎层（社区空间最大）

旧实现此层为**空**：无翻译 API、无 MT 引擎、无 TM/术语库——"译者=pod"（LLM 代理或人逐句改 work.md）
是唯一翻译通道，仅 `--reuse` 英文原文等值回填。因此本层调研分四簇：MT 引擎库、离线 NMT、
TM/术语库（TBX/TMX）、质量估计（QE，详见第 4 层）。

### 1.1 候选表（MT 引擎/聚合器）

| 候选 | 仓库 | ★ | ②2026-05~11 | 许可证 | Py3.13/Win | agent 可编程 | 判定 |
|---|---|---|---|---|---|---|---|
| **deepl-python**（官方 DeepL SDK） | DeepL/deepl-python | 1341 | ✓ [抽样] 10-30~11-06 密集提交、1.33.0 发版 | MIT | ✓ 纯 Python | ✓ API 类 | **引入（可选适配器）** |
| **argostranslate**（离线 NMT） | argosopentech/argos-translate | 6665 | ✗ [抽样] 窗口内仅 6/8/11 月有提交 | MIT | ~ ctranslate2 4.8.0 win cp313 需装机验证；sentencepiece ✓ | ✓ | **豁免 pin（可选件）**：离线 MT 无可替代；ADR 记 ② 不达标 |
| deep-translator（多引擎聚合） | nidhaloff/deep-translator | 2054 | ✗ 末次提交 2024-07 | Apache-2.0 | ✓ 纯 Python | ✓ | **拒绝**（② 死 + 有替代） |
| translatepy | Animenosekai/translate | 558 | ✗ 末次提交 2024-07 | **AGPL-3.0** | ✓ | ✓ | **拒绝**（② 死 + AGPL） |
| translators（多引擎聚合） | UlionTse/translators | 2694 | ✗ 末次提交 2026-01 | GPL-3.0 | ✓ | ✓ | **拒绝**（② 不达标 + 有替代） |
| LLM 直调（pod 本体） | —（agent 自身） | — | — | — | — | ✓✓ | **保留为主通道**（0 依赖，即旧设计） |

**聚合器类整体结论**：web MT 聚合器（deep-translator/translatepy/translators）全部停更或半停更，
**该子类按口径②整类拒绝**；仍活跃的是官方 API SDK（deepl-python）与离线 NMT（argos）。

### 1.2 TM / 术语库（TBX/TMX）候选

| 候选 | 仓库 | ★ | ② | 许可证 | 判定 |
|---|---|---|---|---|---|
| **translate-toolkit**（TTK：TMX/TBX/PO 读写 + tmdb 模糊 TM + 质检过滤器 pofilter） | translate/translate | 987 | ✓ [抽样] 11-13~16 每日提交（Weblate 社区驱动） | **GPL-3.0** | **引入**（TM/TBX 后端）；GPL 按 pymupdf4llm 先例接受（个人不分发） |
| OmegaT | —（Java 桌面） | — | — | GPL | 拒绝（非 agent 可编程，桌面件） |
| 薄 JSON 术语表 + 自研核对 | — | — | — | — | 豁免兜底（~60 行，若用户拒绝 GPL/TTK 过重时启用） |

### 1.3 "译者=pod"模式适配结论

work.md 行契约（`ID<TAB>译文`）下，任何引擎都只是**可替换的"自动填充译者"**，人机皆可：
- **主通道不变**：LLM/人逐句改 work.md（0 依赖、契约即插槽）；
- **MT 自动填充（可选）**：deepl-python（在线、需 key、有术语表/翻译记忆 API）与 argostranslate
  （离线、无 key、MIT）各出一个 ~30 行适配器写回 work.md；argos 新版支持 `ARGOS_CHUNK_TYPE=NONE`
  关闭其内置句切，正合"我们自己已切好句、逐句灌入"的模式；
- **TM/术语**：translate-toolkit tmdb/TBX 承接旧件缺失的"翻译记忆+术语库"，`--reuse` 从
  "英文原文等值"升级为"模糊 TM 匹配"（仍走同一回填口，契约不变）。

---

## 2. 对齐/结构层

### 2.1 pysbd 复核（旧件核心依赖）

| 项 | 证据 | 判定 |
|---|---|---|
| ★944 ✓；MIT ✓；纯 Python ✓ | GitHub API | 口径①过 |
| **52 周提交全 0**（participation [全窗口]）；master 末次实质提交 2021-02，pushed_at 2024-08 仅 tag；OSSF scorecard "Maintained: 0 commits in 90 days" | participation + ecosyste.ms | **口径②死亡** |
| 有替代（wtpsplit/spaCy） | 本节 | **按规则：拒绝，不再引入** |

### 2.2 句切替代

| 候选 | 仓库 | ★ | ② | 许可证 | Py3.13/Win | 判定 |
|---|---|---|---|---|---|---|
| **wtpsplit**（SaT，ML 句切 + ONNX 离线推理） | segment-any-text/wtpsplit | 1382 | ✗ [全窗口] 主干窗口内仅 2026-09-25 一次提交（2026-02 v2.2.1、CI 已加 py3.13） | MIT | ✓ wheel py3-none-any；onnxruntime/tokenizers cp313 win ✓；**注意首用需下 SaT 模型（建议 vendor 进项目自包含）** | **豁免 pin + ADR**（唯一活跃现代 SBD；替代件 spaCy 重且学术缩写句切弱）→ **待裁决 A** |
| spaCy sentencizer/parser | explosion/spaCy | ~30k | ✓ 极活跃 | MIT | ✓ 有 win wheel | 备选：口径全过但依赖重（模型数百 MB）、标点句切对 `Fig. 3(a).` 类仍需哨兵保护 |
| syntok / blingfire / NLTK punkt | fnl/syntok 等 | <100 或停更 | ✗ | — | — | 拒绝（①/②） |

无论换哪个句切器，旧件的**游标法逐字拼回 + 缩写哨兵 + verify_slices 切片自检**（~90 行）都保留：
切片自检是"拼回源文须无损"的机械闸门，与句切器无关，且正好兜住 wtpsplit 的行为漂移。

### 2.3 MinerU 产物 MD 解析

| 候选 | 仓库 | ★ | ② | 许可证 | 判定 |
|---|---|---|---|---|---|
| **markdown-it-py**（CommonMark token 流） | executablebooks/markdown-it-py | 1.4k | ✓ [抽样] 11-09/10 密集提交、v4.3.0（2026-11-10）；全窗口细节未逐一核 | MIT | **引入**：替换旧 `parse_markdown` 正则状态机的切词底座（图注前缀分类等启发式保留），净减自研 ~70-80 行 |
| mistune | lepture/mistune | 3.1k | ~ [抽样] 10 月有提交 | BSD-3 | 备选（renderer 驱动，token 流不如 markdown-it-py 适合结构抽取） |
| 旧件正则状态机 | — | — | — | — | 判定：MinerU MD（公式 `$$`/HTML 表格/图片行）杂、正则脆弱是事故温床，token 流是正解 |

对齐机制不变：仍是**构造式对齐**（manifest ID 寻址 + 英文侧出稿程序化重嵌入），旧件本就不做
双文本对齐；reorder_suggest 的 md↔PDF 块映射（540 行）是唯一"对齐算法"，**整段平移**（仅 difflib+fitz）。

---

## 3. 排版层（重点：两路线自研量量化）

### 3.1 路线 (a)：旧 HTML 链平移

管道：契约 md → `to_pandoc_md` → pandoc(--mathjax) → 自包含 HTML（内联 CSS/KaTeX/字体 data:URI）
→ playwright 系统 Chrome headless 打印 A4 双栏 → PyMuPDF 叠页码。

| 构成 | 旧行数 | 平移/新写 | 新维护面 | 说明 |
|---|---|---|---|---|
| to_pandoc_md 预处理 | ~80 | 直接平移 | 80 | 纯函数 |
| build_html 自包含装配 | ~90 | 直接平移 | 90 | KaTeX bundle + bilingual.css 作资产随迁（0 新行，LICENSE 齐备） |
| 宽元素 JS（min-content 量宽 + 栏宽比较 + zoom） | ~60 | 直接平移 | 60 | **两起事故修复已固化**（scrollWidth 误判 13/13 公式、禁跨栏裁决），测试钉死 |
| 图注粒度（图+首对原子 / 余对自由流） | ~40 | 直接平移 | 40 | 三轮裁决产物 |
| PyMuPDF 页码 | ~40 | 直接平移 | 40 | |
| playwright 启动 + page.pdf | ~30 | **重写**（browser_fetch._launch → 直连 launch） | 25 | 薄 |
| 路径/配置/CLI 接缝 | — | **新写** | 35 | pysci.paths/config 换新项目路径层 |
| **合计** | 383 + 96(css) | 平移 ~310 / 新写 ~60 | **~370**（其中代码 ~270+新写 60，css 96 随迁） | 483 行渲染回归测试**可平移护航** |

**风险**：低。事故热点均有注记+测试；环境代价 = pandoc + playwright + 系统 Chrome（pypandoc 可
换 subprocess 直调 pandoc 再省 1 依赖）。

### 3.2 路线 (b)：LaTeX 链（paracol / reledpar 宏包）

管道：契约 md →（pandoc Lua filter 或 Python 转换器）→ .tex（句对宏）→ xelatex（ctex/xeCJK）→ PDF。

**宏包成熟度**（CTAN 实测，均内置 TeXLive 2026——新项目 `deps/tex/base/texlive/2026` 已 vendored）：

| 宏包 | 版本/日期 | 许可证 | 角色 | 判定 |
|---|---|---|---|---|
| **paracol** | 1.37 / 2025-07-14（repo codeberg mgkurtz，活跃） | LPPL-1.3c | 双栏流/同步/逐栏浮动，官方定位即 "bilingual documents" | **(b) 首选** |
| reledpar + reledmac | 2.25.10 / 2025-05-13；2.44.4 / 2026-03-06 | LPPL-1.3 | 行级同步平行文本（校勘本向） | 备选（行同步对我们非必需，栈更重） |
| gb4e | CTAN 常青 | LPPL | 语言学例句/词对齐注解 | 不适用（词级 gloss，非句对双栏） |
| verse | CTAN 常青 | LPPL | 诗歌行对齐 | 不适用 |

注：句对形态是"中文行+斜体英文行"**同栏纵向堆叠**（旧件 CSS 双栏流布，非左右页同步），
故 paracol 的同步能力用不上，仅用其逐栏浮动 + 双栏流；LaTeX 处理宽公式可 `\resizebox` 但
**旧件"只缩不放、禁跨栏"的量宽口径要在 TeX 里重新实现/调试**（旧事故面重踩）。

| 构成 | 新写行数 | 说明 |
|---|---|---|
| md→LaTeX 转换（pandoc Lua filter：句对宏、图+首对原子 minipage、公式/表/参考文献透传、摘要标注） | 150–200 | 全新 |
| template.tex（ctex + paracol + 栏间线 + 中文衬线 11pt/英文斜体灰小 + 图注分级 caption/caption-cont） | 80–120 | 全新 |
| 宽元素处理（adjustbox/resizebox 宏 + display 公式策略） | 30–50 | 全新 + **调试不可预估** |
| 图注粒度（figure 原子 vs 自由流布） | 30–50 | 全新（旧三轮裁决的 TeX 版重解） |
| xelatex 编译胶水 + 页码 | 25 | 页码 fancyhdr 原生（省旧 PyMuPDF 40 行） |
| **合计** | **~310–450（中位 ~380，全新）** | 483 行渲染测试全部作废（重写 ~400 行） |

### 3.3 自研量对比表（用户裁决②量化）

| 维度 | (a) HTML 链平移 | (b) LaTeX 链 | 胜者 |
|---|---|---|---|
| **需新写** | **~60 行** | **~310–450 行** | (a) |
| **总自研维护面** | **~370 行**（含 css 96 资产） | **~310–450 行**（中位 ~380） | (a) 略胜（且 (a) 为平移件、(b) 为全新件） |
| 测试资产 | 483 行回归测试平移护航 | 全部重写 | (a) |
| 事故风险 | 已踩平（min-content/图注粒度有注记） | 宽公式/图注粒度在 TeX 重踩，调试量未知 | (a) |
| 依赖面 | pandoc + playwright + Chrome + KaTeX bundle | xelatex + paracol（TeXLive 2026 已 vendored，更干净） | (b) |
| 中文/公式 | KaTeX vendored（已验证） | ctex/amsmath 原生 | (b) |

**推荐：(a) HTML 链平移**——按"哪条自研代码量小走哪条"两口径（新写量 60 vs ~380；维护面 370 vs ~380）
均胜，且测试可平移、风险低。LaTeX 链留档备选：若未来⑤写作域全面 LaTeX 单轨化、要消灭浏览器依赖
时再切换（届时 paracol 1.37 为首选宏包）。WeasyPrint 已评估否决（无 JS 量宽，宽元素逻辑失效、多栏支持弱）。

---

## 4. 质量层

旧件全机械闸门（切片自检/三桶对账/占位假译拒绝/图片闭合/题名三面一致/pod-guard/audit_layout 抽查），
**平移保留**（纯胶水）。社区补强候选：

| 候选 | 仓库 | ★ | ② | 许可证 | 判定 |
|---|---|---|---|---|---|
| **回译校验（back-translation）** | 用第 1 层引擎适配器反向翻译对照 | — | — | — | **补**（~40 行胶水，无新依赖；中→英→中 漂移检测） |
| **术语一致性核对** | glossary JSON/TBX + TTK 或薄核对 | — | — | — | **补**（~40 行；旧件纯纪律，实产已养 bench/glossary_*.md 习惯，机械化它） |
| sacrebleu（BLEU/chrF 参考实现） | mjpost/sacrebleu | 1265 | ✗ [全窗口] participation：窗口内仅 7/8 月有提交 | Apache-2.0 | **可选豁免 pin**（QE 指标事实标准、无等价替代；默认不引，需要量化指标时加）→ 待裁决 D |
| COMET / COMETKiwi（"KiwiEstimate" 系 QE） | Unbabel/COMET | 790 | ✗ 末次提交 2026-04-21 | Apache-2.0 | **拒绝**（② 不达标 + torch/大模型权重重载，与逐句工作流收益不匹配；记文献佐证） |
| OpenKiwi | Unbabel/OpenKiwi | ~500 | ✗ 末次提交 **2020-11** | Apache-2.0 | **拒绝**（死库） |
| LLM 评审（agent 自评/互评） | — | — | — | — | 可选（0 依赖，技能文档约定抽查比例） |

**建议补强最小集 = 回译 + 术语一致性**（两者共 ~80 行胶水、0 新依赖）；神经 QE 全家不进主链。

---

## 5. 薄胶水保留量（契约/CLI 状态机/护栏，无可替代）

| 件 | 保留量 | 说明 |
|---|---|---|
| 三契约本体（work.md 行契约 / manifest 切片契约 / bilingual.md 渲染契约） | ~350 行 | 真正值钱的部分，纯函数 |
| CLI 状态机（prepare/status/submit/render/reorder-suggest + 防覆写/自动备份） | ~120 行 | 无可替代 |
| pod-guard 等价护栏（work.md 行结构硬校验） | ~60 行 | 旧 PreToolUse 钩子 → 独立 CLI 校验（`pysci-lit bilingual check-work`）+ 可选 DSH 钩子适配 |
| 交付命名/目录约定（中文标题命名、md/pdf 成对） | ~40 行 | 薄 |
| **合计** | **~570 行** | 分布于各模块，是"技能文档+社区工具+薄胶水"中的胶水本体 |

---

## 6. 建议组合（五层）与自研面总账

| 层 | 建议组合 | 社区件 | 新增/保留自研 |
|---|---|---|---|
| ① 翻译引擎 | 译者=pod 主通道 + deepl-python（可选在线）+ argostranslate（豁免 pin，可选离线）+ translate-toolkit（TM/TBX） | 3 | ~100 行适配器/TM 回填 |
| ② 对齐/结构 | wtpsplit（豁免 pin，弃 pysbd）+ markdown-it-py（替正则切词）+ reorder_suggest 平移 | 2 | ~770 行（游标句切 90 + 结构契约 160 + 出稿 50 + reorder 520 保留） |
| ③ 排版 | **(a) HTML 链平移**：pandoc + playwright Chrome + PyMuPDF + KaTeX bundle | 4 | ~370 行（新写仅 ~60） |
| ④ 质量 | 旧机械闸门平移 + 回译 + 术语一致性；sacrebleu 可选豁免 | 0-1 | ~230 行（闸门 150 + 新补 80） |
| ⑤ 薄胶水 | 三契约 + CLI 状态机 + check-work 护栏 + 交付约定 | 0 | ~570 行（与上互为视图，不重复计） |

**自研总账（vs 旧件 2197 行）**：

| 块 | 旧 | 新 | 手段 |
|---|---|---|---|
| bilingual.py 管线 | 1274 | ~910 | markdown-it-py 减正则、mineru_cloud/缓存设施复用①域、路径层薄化 |
| bilingual_render.py | 383 | ~330 | 平移 270 + 新写 60 |
| reorder_suggest.py | 540 | ~520 | 整搬（fitz 已有） |
| CLI 壳 + 护栏 | ~80 | ~90 | pod-guard → check-work |
| 翻译引擎/TM/术语胶水 | 0 | ~100 | 新增（能力面扩大） |
| 质量层回译/术语核对 | 0 | ~80 | 新增（能力面扩大） |
| **合计** | **2197** | **≈ 1850（含新增能力）** / **≈ 1650（等价能力口径）** | **-16% ~ -25%** |

测试面：1800 行中渲染 483 + 管线/切分 951 + guard 226 + reorder 140；
(a) 路线下 ~90% 可平移，预计 1500–1600 行护航，另补 wtpsplit/引擎适配器小测。

---

## 7. 三口径判定汇总

| 候选 | ①★≥100 | ②每月提交 | ③减自研 | 裁决 |
|---|---|---|---|---|
| deepl-python | ✓ 1341 | ✓ [抽样] | ✓ | 引入（可选） |
| argostranslate | ✓ 6665 | ✗ 6/8/11 月 | ✓（离线无可替代） | **豁免 pin + ADR**（可选件） |
| translate-toolkit | ✓ 987 | ✓ [抽样] | ✓ | 引入（GPL-3 记录） |
| markdown-it-py | ✓ 1.4k | ✓ [抽样/近似] | ✓ | 引入 |
| wtpsplit | ✓ 1382 | ✗ 仅 9 月 | ✓（pysbd 已死） | **豁免 pin + ADR**（备选 spaCy） |
| sacrebleu | ✓ 1265 | ✗ 7/8 月 | 可选 | 可选豁免（默认不引） |
| pysbd | ✓ 944 | ✗ 全 0 | ✓ | **拒绝**（有替代） |
| deep-translator | ✓ 2054 | ✗ 2024-07 | ✓ | 拒绝 |
| translatepy | ✓ 558 | ✗ 2024-07 | ✓ | 拒绝（+AGPL） |
| translators | ✓ 2694 | ✗ 2026-01 | ✓ | 拒绝 |
| COMET/COMETKiwi | ✓ 790 | ✗ 2026-04 | 可选 | 拒绝（重） |
| OpenKiwi | ✓~500 | ✗ 2020-11 | — | 拒绝（死） |
| mistune | ✓ 3.1k | ~ | 备选 | 不引（记录） |
| paracol/reledmac/reledpar | n/a（CTAN 年发版，逐月口径不适用） | — | 仅 (b) 用 | 记录（(a) 采纳后不引） |

---

## 8. 待用户裁决点

- **A. 句切替换**：pysbd 按规则拒绝后，豁免引 wtpsplit（推荐，MIT/活跃/离线 ONNX，需 vendor 模型）
  还是引 spaCy（口径全过但重）？或维持 pysbd 豁免（需推翻"有替代→拒绝"前提）？
- **B. argostranslate 取舍**：豁免 pin 为离线可选件（推荐），还是离线 MT 无真实需求、干脆不引（省依赖）？
- **C. translate-toolkit 的 GPL-3.0**：接受（先例 pymupdf4llm AGPL）引 TM/TBX 后端，
  还是退化为 ~60 行薄 JSON 术语表+difflib 薄 TM（多 ~80 行自研）？
- **D. sacrebleu**：QE 指标可选件豁免 pin，还是不引（回译+术语一致性已够）？
- **E. 排版路线确认**：按裁决口径推荐 (a) HTML 链平移（新写 ~60 行 vs LaTeX 链 ~380 行全新）；
  若用户更看重"依赖面干净/统一 LaTeX 单轨"则选 (b)（自研量更大）。
- **F. wtpsplit 模型自包含**：SaT 模型文件 vendor 进 `sub_root`（自包含原则，~百 MB 级）还是允许 HF 下载？

---

## 附录：主要证据源（2026-11-17 取证）

- https://api.github.com/repos/{nipunsadvilkar/pySBD, segment-any-text/wtpsplit, Animenosekai/translate,
  nidhaloff/deep-translator, argosopentech/argos-translate, translate/translate, DeepL/deepl-python,
  UlionTse/translators, mjpost/sacrebleu, Unbabel/COMET}（元数据）
- https://api.github.com/repos/{nipunsadvilkar/pySBD, mjpost/sacrebleu}/stats/participation（逐周提交）
- https://github.com/{translate/translate, argosopentech/argos-translate, segment-any-text/wtpsplit,
  DeepL/deepl-python, executablebooks/markdown-it-py}/commits/<branch>.atom（提交抽样）
- https://repos.ecosyste.ms/api/v1/hosts/GitHub/repositories/nipunsadvilkar%2FpySBD（OSSF scorecard 复核）
- https://pypi.org/pypi/deep-translator/json；https://ctan.org/json/pkg/{paracol,reledpar,reledmac}
- https://img.shields.io/github/{stars,last-commit}/{executablebooks/markdown-it-py, lepture/mistune,
  Unbabel/OpenKiwi, nipunsadvilkar/pySBD}
- 旧件解剖：`legacy-01-bilingual-parallel.md`（行数/耦合/事故注记的事实底座）

## 用户裁决记录（2026-11-17）
- 句切：**豁免 wtpsplit**（pin+ADR），备选 spaCy；pysbd 维持拒绝。
- TM/术语：**接受 translate-toolkit（GPL-3，个人不分发同 pymupdf 先例）**；argostranslate 离线翻译**可选件豁免 pin**。
- 排版层：**(a) HTML 双栏链**（按'自研量小'裁决标准双指标全胜）。
- sacrebleu 可选件豁免 pin；wtpsplit SaT 模型 **vendor 进项目内**（自包含）。
- **全局硬件约束（用户口述）**：本机仅 CPU、无独显/显存、16GB 内存（日常可用约 5GB）——本地模型推理等重量级计算需谨慎；SaT 用小模型 CPU 推理、argos 仅按需且接受慢速；影响后续域选型（⑦AI 生图须云 API 优先）。
