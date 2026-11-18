# 文本链实施计划：①文献调研 + ⑤文档写作（状态：✅ 已完成 2026-11-17）

> 三段式第二段产物。选型输入：`sub_root/others/community_research/2026-11-17/`（00-synthesis.md 裁决记录
> + domain-01/05 两份选型报告）。完成后本文件归档至 `archived/`。

## 1. 范围

**做**：①文献域、⑤写作域的"技能文档 + 社区依赖 + 薄胶水"落地；旧能力清单对照裁决执行。
**不做**（用户裁决暂缓/不保留）：学院 deck 三件套、deck_digest、提取乱码门、中英对照生产线、
SciMago 索引、WoS 自研客户端、缓存/doctor 自检、Typst、marker、付费墙浏览器抓取（降级 agent web 工具）。

## 2. 选型定稿（含豁免记录）

### ①文献域
| 角色 | 引入 | 版本策略 | 口径/豁免 |
|---|---|---|---|
| 检索/获取聚合 | paper-search-mcp（3.1k★ MIT，22 源+OA 回退+引文关系，自带 DSH bundle） | **pin** | 差②，**豁免：唯一聚合通道**，风险记原地 ADR |
| PDF→MD（云） | MinerU 云端 API（MINERU_TOKEN 旧 .env 已有）；本地不装重推理 | pin API 版本 | ②全过；本机硬件不足，裁决走云端 |
| 文献库+RAG | Zotero（外部软件）+ zotero-mcp（PyPI `zotero-mcp-server` 5.9k★ MIT） | pin | ②全过；语义检索承接旧 RAG |
| RAG 可选件 | paper-qa（9.6k★） | pin，可选件不进主链 | 差②，**豁免 pin 为可选件**（用户裁决） |
| 兜底解析 | pymupdf4llm（2.3k★ AGPL） | pin | ②全过；AGPL 用户已接受（个人不分发） |
| 方法层 | 笔记 rubric / citecheck / 综述脚手架 / 取证规范 | 技能文档 + 薄胶水 ≤400 行 | 自研（无社区对应） |

### ⑤写作域
| 角色 | 引入 | 版本策略 | 口径/豁免 |
|---|---|---|---|
| Office 创作 | DSH 预装 `dsh-skill-office` 七技能 + python-docx 1.2.0 / python-pptx 1.0.2 / openpyxl 3.1.5 + pandas | **pin** | 差②，**豁免：DSH 技能锚定无可替代**（ADR） |
| 格式互转/引用 | pandoc 3.14+（GPL-2.0，exe）+ citeproc/CSL styles | pin minor | ②全过 |
| 排版 | **LaTeX 单轨**：latexmk + chktex + 日志解析胶水 | TeX 发行版系统装 | 用户裁决；Typst 不引入 |
| 操作层 | DSH 自带 LibreOffice Kit（渲染/公式重算/导 PDF） | 随 DSH | 预装零成本 |
| 引用链 | zotero-mcp + CSL styles（与①共享） | — | — |

**拒绝清单**（差②有替代，用户已认可）：markitdown、pypandoc(-binary)、typst-py、XlsxWriter、docxtpl、
pandoc/lua-filters（社区）、typst/templates、Quarto（无需求位）、GROBID、arxiv.py、arxiv-mcp-server、
pyalex、semanticscholar、scholarly、habanero、pdfplumber、Better BibTeX。

## 3. 实施步骤

1. **配置迁移**：旧 `pySciWS\.env` → 新项目根 `.env`（已在 gitignore）；仅取所需 key：
   ①域 MINERU_TOKEN、WOS_API_*、OPENALEX_*、ZOTERO_*、ELSEVIER_API_KEY、ARXIV_USER_AGENT 及
   HTTP_* 通用项；其余 key 留待各域选型时按需迁移。Zotero 数据目录路径按新机现状核对。
2. **依赖引入**：`sub_root/pyproject.toml` + uv sync（pymupdf4llm、zotero-mcp-server、openpyxl、
   pandas、python-docx、python-pptx）；全部 pin 精确版本。
3. **外部工具链（用户裁决：全量项目内自包含，一律置于 `sub_root/deps_management/deps/`）**：
   - pandoc：官方 exe 放 `deps/tools/pandoc/`（自包含）；
   - TeX：目标 `deps/tex/`。**实施记录（2026-11-17）**：直接迁移被拒（`tex`/`tex\base` 改名 Access
     denied，但写删与叶子改名正常 → 有句柄持有者，疑似 VS Code 文件监看或终端 CWD 驻留 base 子树）；
     现以目录 junction `deps/tex` → `D:\XiGPrograms\tex` 顶位（pdflatex/latexmk/chktex 经其验证可用）。
     物理迁移待用户关闭持有进程后执行：先 `Remove-Item deps/tex`（仅删链接），再 move，重验三工具。
   - Zotero 桌面端为外部 GUI 软件（API 走 zotero-mcp，无需装 BBT）。
4. **MCP 接入**：paper-search-mcp 按其 DSH profile bundle 方式接入并 pin；zotero-mcp-server 以
   stdio MCP 挂 DSH 配置；两者写入项目 DSH profile（`dsh_home` 的 cordis.patch.yml）。
5. **技能文档重建**（`.agents/skills/`，形态借鉴 gpt-researcher/hyperresearch 的 skill 设计）：
   - `literature-research/SKILL.md`：检索→获取→MinerU 云提取→笔记 rubric→citecheck→综述脚手架；
   - `document-writing/SKILL.md`：路由 DSH office 技能 + pandoc 互转 + LaTeX 排版工作流。
6. **薄胶水**（`sub_root/src/pysci/`，预计 ①≤400 行 + ⑤≈600-900 行）：
   - ①：arXiv LaTeX 清洗 ~150、citecheck 比对 ~100、frontmatter 归一化 ~150；
   - ⑤：薄 CLI 路由 ~300-500、pandoc 包装 ~50、refs_bridge（zotero-mcp→refs.bib）~100-200、
     PDF→PNG 校对 ~50、公式 ink 裁剪 ~100、LaTeX 工作流（latexmk 调用+日志解析+chktex）~150-300。
7. **QA 同步**（`sub_root/tests/`）：机械断言层（胶水单测、pin 版本断言）、基准层（PDF→MD/MD→docx
   转换保真样例；**pandoc 3.14 pptx reader 真实 deck 实测**，不达标再议）、主观层（渲染检查清单）；
   全量测试通过后收尾。

## 4. 风险与 ADR 记录点

- pin 版本漂移：paper-search-mcp（10-02 后无提交）、python-docx/pptx（停更）——ADR 记录豁免理由与
  替代观察名单；
- MinerU 云 API：网络/配额依赖（本机无本地提取能力是既定裁决）；
- pandoc 3.14 pptx/xlsx reader 过新（2026-10-30 起）：步骤 7 实测保真度为验收门；
- WoS Starter API 覆盖面有限（引文网络深挖靠 OpenAlex）。

## 5. 验收标准

①：给定主题 → 多源检索→去重→OA 下载→MinerU 云提取→结构化笔记+citecheck 全链可跑通；
⑤：MD → docx/pptx 生成、docx 编辑、公式渲染、LaTeX 排版→PDF 全链可跑通，DSH office 技能渲染验收；
QA 全量测试通过；plans 归档。

## 6. 实施结果（2026-11-17 收尾记录）

- 验收标准**全部达成**：①链路真实端到端（4 源检索→OA 下载→MinerU 云 49K 字提取→五段笔记→
  citecheck PASS→tex_clean 真实 e-print）；⑤链路真实产物（MD→docx/pptx、LaTeX→PDF 中文目检完美、
  公式透明 PNG、pdf2png）。
- pandoc 3.14 pptx reader **保真门达标**（关键要素可恢复，markitdown 维持拒绝）。
- 额外修复：MinerU 云协议 5 处偏差（实测锁定主形态，mock 同步）；formula_png 参数回归；
  chktex 格式串；frontmatter 提取稿口径；presets 片段随 MCP 条目再生成。
- 最终 `tests.cmd all`：ruff All checks passed + pytest 109 passed / 3 hardware（TeX 实跑 3/3 过）。
- 遗留（用户/后续）：TeX junction 物理迁移；Zotero 10+ 本地写授权；WOS_* 待 paper-search-mcp
  发版连接器；PAPER_SEARCH_MCP_UNPAYWALL_EMAIL 可补；paper-qa 可选件未启用；eval 基准层
  **用户裁决暂缓**（2026-11-17：七域齐后统一建，含 pptx 保真门与文献链基准）。
