# 能力域①文献调研 · 选型细化调研报告（2026-11-17）

> 三段式第一段「选型」的调研部分。旧 pySciWS 仅作能力清单对照，**未复制任何代码**（本报告只含文字提炼）。
> 核实口径（用户裁决）：① star≥100；② 2026-05~2026-11 **每月均有提交**；③ 减少自研代码。
> 豁免规则：差口径②若"无可替代（唯一通道）"→豁免引入（pin 版本+记风险）；有替代余地→拒绝。
> gpt-researcher 只借鉴技能/插件形态，不整机引入（用户裁决③）。

## 0. 结论速览

- **首选四件套**：paper-search-mcp（检索/下载/引文，豁免②）+ MinerU（PDF→MD，三口径全过）+ Zotero+zotero-mcp（文献库+语义检索，三口径全过）+ 技能文档/薄胶水（笔记 rubric、citecheck、综述脚手架）。
- **次选**：marker（PDF→MD 备选/对照，三口径全过）、paper-qa（科研 RAG 深度最佳但 bursty，②不达标→待用户裁决豁免与否）、PyMuPDF/pymupdf4llm（轻量兜底）、llama_index（仅语义检索不敷时）。
- **拒绝**：arxiv.py、arxiv-mcp-server、pyalex、semanticscholar、habanero、scholarly、pdfplumber、GROBID、Better BibTeX（暂缓）——理由见 §2.3。
- **薄胶水估计**：新自研 ≤400 行（arXiv LaTeX 源清洗、citecheck 比对、frontmatter 归一化/索引）+ 纯技能文档；对比旧项目 21 个自研模块。

---

## 1. 任务一：旧能力清单（仅对照，严禁复制代码）

来源：`pySciWS/orchestration/skills/literature-research/`（SKILL.md + 33 个 reference 分册）、`src/pysci/skills/literature_research/tools/`（21 个模块）、`tests/skills/literature_research/`（40+ 测试）、`data/skills/literature_research/`（数据布局）。

| # | 能力项 | 旧实现方式（自研模块/工具） | 新方案是否仍需要 | 备注 |
|---|---|---|---|---|
| 1 | 多源文献检索 | `research search`：OpenAlex+arXiv 双源合并去重、年份/引文/OA 过滤、shortlist 快照；主路 paper-search-mcp（22 源） | **需要** | 旧方案已社区主路化，方向延续 |
| 2 | 引文网络滚雪球 | `citegraph`：OpenAlex 双向（references/citers） | **需要** | 新增可用 paper-search-mcp OpenAlex 关系工具 |
| 3 | 期刊/venue 质量 | `journal`：自建 SciMago 分区索引（build-scimago）+ OpenAlex JIF 估计 + journal_tier | 需要（可简化） | 自建 SciMago 索引=旧项目特有，可不保留（agent 查分区表即可） |
| 4 | 全文获取（OA→付费墙） | `read`：arXiv 直下→OA oa_url HTTP 直取（%PDF 魔数校验、原子写）→Playwright 无头浏览器抓付费墙/Cloudflare（机构 IP、--headed）；fetch order + 缓存复用 | **需要** | 付费墙浏览器抓取是差异化能力；新方案降级为薄胶水/agent web 工具 |
| 5 | PDF→Markdown（公式→LaTeX） | `pdf_extract`：MinerU 云（公式→LaTeX、表格→HTML）+ pymupdf4llm 本地回退 | **需要** | 社区首选即 MinerU（同血统） |
| 6 | arXiv LaTeX 源直转 | `pdf_extract._tex_to_markdown`：e-print 源→去注释/去 preamble/去 thebibliography→MD | 需要（薄胶水） | 旧项目特有细节清洗，~100-150 行胶水可重写 |
| 7 | OA 回退链 | `oa_fallback`：OpenAIRE/CORE/PMC/Unpaywall | **需要** | paper-search-mcp `download_with_fallback` 内置同款 |
| 8 | 结构化文献笔记 + 评估 rubric | `notes`：note 骨架 + TLDR/Key Claims/Novelty/Rigor/Relevance + note-merge 合同 + frontmatter 归一化 | **需要** | 技能文档承载（零代码），rubric 直接沿用口径 |
| 9 | 引用完整性门（citecheck） | `citation_verify`：写入前 OpenAlex+Crossref+arXiv 三方交叉核验，硬冲突拦截；审计 ✓/△/✗/? | **需要** | 薄胶水：调 paper-search-mcp DOI 工具比对 frontmatter |
| 10 | Zotero 集成 | `zotero_cli`：委托 zotero-cli --json（zotero-mcp）；frontmatter→BibTeX、zotero_key/uri 回写、INDEX.md 重建 | **需要** | zotero-mcp 三口径全过，委托面不变 |
| 11 | 本地语料 RAG/问答 | `rag`：PaperQA2 索引 + SiliconFlow 嵌入 + rag search/ask（带引文+路径） | **需要** | 换社区实现（zotero-mcp 语义检索 或 paper-qa，见 §3） |
| 12 | 批量本地 PDF 摄取 | `local_ingest`：manifest 台账、可续跑、无 DOI PDF 库 | 需要（可简化） | zotero-mcp 批量导入/paper-qa index 覆盖大半 |
| 13 | 缓存治理 | `cache_manager`：stats/clean/prune LRU、命中复用 | 可选 | 旧项目特有自检，可不保留（目录约定替代） |
| 14 | 中英对照文档生产线 | `bilingual`/`bilingual_render`：句级切分+图片回补+双栏 A4 PDF+译文复用+reorder-suggest | **旧项目特有，建议不保留/后置** | 纯自研大件（数百行+渲染模板），与"减自研"冲突；如需可作为独立后续需求 |
| 15 | 综述/调研报告脚手架 | `review new/status/sync`：survey 骨架、链接与计数审计 | **需要** | 形态借鉴 gpt-researcher/STORM（多阶段编排），技能文档承载 |
| 16 | 非学术网页取证规范 | `fetch-evidence`：[直取]/[二手互证] 双标签、403/地域封锁决策树 | **需要** | 纯技能文档（零代码） |
| 17 | WoS/Scopus 富化 | `wos_client`：WoS Starter API（wos_id + Times Cited） | 可选 | paper-search-mcp 有 WoS/Scopus 显式连接器（需 key）→自研可不保留 |
| 18 | 链接校验/doctor 自检 | `links_verify` + `doctor` | 可选 | 旧项目特有自检，可简化或不保留 |
| 19 | 评估/筛选工作流 | 9 步工作流（search→screen→add→citegraph→read→note→index→review→citecheck） | **需要** | 技能文档主体，直接作为新技能文档骨架 |

测试反映的期望行为（`tests/skills/literature_research/`）：PDF 后端可切换、引文核验分级（PASS/WARN/FAIL/NOT_FOUND）、note merge 契约、缓存命中复用、bilingual 防伪译、bib 一致性、引用路径可解析——这些行为期望应在新技能文档中以验收标准形式保留，代码不迁移。

---

## 2. 任务二：社区候选细化核实

### 2.1 核实方法与数据源

- 星级/最后提交/许可证：`img.shields.io/github/stars|last-commit|license/<owner>/<repo>`（SVG 文本）。
- **逐月提交（2026-05~2026-11）**：commits.ecosyste.ms 镜像 `commits?per_page=1&until=<月末>` 探针法（返回 ≤该日期的最后一条提交时间戳；落月内=当月有提交）。GitHub API 匿名配额被同 IP 会话占满，此通道不计配额且与 shields last-commit 交叉验证一致。
- Py3.13/Windows：PyPI 版本级 JSON（`pypi.org/pypi/<pkg>/<ver>/json`：requires_python、classifiers、wheel 列表）+ README 环境表。
- 许可证以 shields/PyPI/仓库 LICENSE 为准；未核实处显式标注。
- 弱点声明：月末探针证明"当月有提交"，但不计提交数；bot 提交（dependabot/CLA）也算提交，个别条目已注明。

### 2.2 候选指标总表

| 候选（仓库） | 能力域 | ★ | ①≥100 | ②逐月 5–11 | ③减自研 | Py3.13/Win | 驱动方式 | 许可证 | 能力边界/备注 |
|---|---|---|---|---|---|---|---|---|---|
| **openags/paper-search-mcp** | 检索/OA下载/引文 | 3.1k | ✓ | **5✓6✓7✓8✓9✓10✓ 11✗** | 强（22源聚合+去重+OA回退+WoS/Scopus连接器） | ✓ Py3.10+ 纯 wheel，README 含 Windows | MCP(stdio/sse/http)+CLI `paper-search`+Claude 技能+**DSH profile bundle** | MIT | Google Scholar 易封；OpenAlex 需免费 key 提额；引文关系为 OpenAlex 有预算查询 |
| **54yyyu/zotero-mcp** | 文献库 | 5.9k | ✓ | **全✓**（05-26/06-23/07-29/08-30/09-26/10-30/11-15） | 强（Zotero 全操作+BibTeX+语义索引） | ✓（旧项目 Py3.13+Win 实证） | MCP + `zotero-cli --json` | MIT | 需 Zotero 桌面端+本地 API 或 Web API key；内置 **ChromaDB 语义检索**、Scite 撤稿警报、WebDAV/linked-file |
| **opendatalab/MinerU** | PDF→MD | 84k | ✓ | **全✓**（05-29/06-29/07-30/08-30/09-29/10-30/11-13） | 强（公式→LaTeX/表格→HTML/OCR） | △ requires `<3.14,>=3.10`（3.13 收录）但**官方脚注：Windows 因 ray 仅 3.10–3.12** | CLI `mineru`+REST API+MCP server+WebUI | Apache-2.0 **+附加条款**（MAU>1亿或月收>$20M 需商业许可；对外在线服务须署名） | PDF/DOCX/PPTX/XLSX/图→MD/JSON；pipeline 后端纯 CPU 可跑；VLM/hybrid 后端需 GPU 8GB+ |
| **datalab-to/marker**（原 VikParuchuri/marker 迁移） | PDF→MD | 41k | ✓ | **全✓**（05-05/06-29/07-20/08-27/09-13/10-29/11-14） | 强 | ✓ Py3.10+ 纯 wheel，torch≥2.7（cp313 win wheel 在） | CLI `marker(_single)`+Python API+`marker_server` HTTP | 代码 Apache-2.0；**模型权重 OpenRAIL-M**（科研/个人/<$5M 免费） | 2.1.0（10-29 发版）；balanced 需 surya VLM 推理服务（vLLM Docker GPU 或 llama.cpp CPU）；fast 纯 CPU（公式弱：arXiv math 23.4 vs balanced 83.9） |
| **pymupdf/PyMuPDF**（+pymupdf4llm） | 轻量 PDF 抽取 | 11k | ✓ | ✓（抽检 05-28/08-27/11-13，持续高频） | 中（兜底抽取） | ✓ cp313 win wheels | Python API/CLI | **AGPL-3.0**（双许可商业） | 文本/表格/图片抽取，公式丢失；个人科研使用无碍 |
| **Future-House/paper-qa** | 语料 RAG/问答 | 9.6k | ✓ | **5✗6✓7✗8✓9✗10✓11✗**（bursty） | 强（科研 RAG+引文+元数据+撤稿检查） | ✓ Py≥3.11（3.13 分类器）纯 wheel | CLI `pqa`+Python API（LiteLLM 任意后端） | Apache-2.0 | CalVer（2025-12 起，无兼容承诺→须 pin）；上次发版 2026-03-18；含 Zotero/office/docling readers、本地嵌入 |
| blazickjp/arxiv-mcp-server | 深 arXiv/引文图 | 3.3k | ✓ | 5✓**6✗**7✓8✓**9✗**10✓11✓ | 中 | ✓ Py3.10+ 纯 | MCP | Apache-2.0 | get_paper_latex/read_paper_section/watch_topic/citation_graph（S2 单跳） |
| lukasschwab/arxiv.py | arXiv API | 1.6k | ✓ | 5✓**6✗**7✓**8✗9✗**10✓11✓（多为 dependabot） | 中 | ✓（2026-11 起 Py3.11–3.15） | Python API | MIT | 薄客户端；arXiv REST 直调亦可 |
| J535D165/pyalex | OpenAlex | 433 | ✓ | **全✗**（最后提交 2026-02-23） | 中 | ✓ 纯 | Python API | MIT | OpenAlex 2026-02-13 起 **API key 必需**（免费 100k credits/日） |
| danielnsilva/semanticscholar | S2 API | 486 | ✓ | ✗（最后提交 2026-07-03，其后静默） | 中 | ✓ 纯 | Python API | —（未核实） | S2 匿名限速，key 提额 |
| sckott/habanero | Crossref | 254 | ✓ | ✗（10-25 仅 dependabot，未逐月核） | 弱 | ✓ 纯 | Python API | —（未核实） | Crossref REST 直调即可 |
| scholarly-python-package/scholarly | Google Scholar | 1.9k | ✓ | ✗（最后提交 2026-03-24） | 中 | ✓ 纯 | Python API | —（未核实） | Google Scholar 反爬/CAPTCHA，不稳定 |
| jsvine/pdfplumber | PDF 抽取 | 11k | ✓ | ✗（最后提交 2026-06-15） | 中 | ✓ 纯 | Python API | MIT | 表格/文本抽取；被 PyMuPDF 覆盖 |
| kermitt2/grobid | PDF→TEI/全文 | 5.2k | ✓ | 未逐月核（ecosyste.ms 未收录该库；shields 显示活跃） | 中 | △ **Java 服务**（Docker/JVM 部署） | REST/grobid-client-python | Apache-2.0 | 引文/头部解析强，但部署形态与 Windows+Python 自包含冲突 |
| retorquere/zotero-better-bibtex | Zotero BibTeX | 7.3k | ✓ | ✗（最后提交 2026-07-13，8–11 空） | 中 | n/a（Zotero 插件 XPI） | Zotero 插件 | MIT | 成熟稳定；zotero-mcp 自带 BibTeX 导出 |
| run-llama/llama_index | RAG 框架 | 53k | ✓ | ✓（抽检 11-14，高频活跃） | △ 框架重、胶水反而多 | ✓ | Python API | MIT | 仅当语义检索需自定义管线时引入 core |
| assafelovic/gpt-researcher | 报告生成 | 31k | ✓ | ✓（抽检 11-09） | — | ✓ | Python | Apache-2.0 | **仅形态借鉴**（用户裁决）：多阶段调研编排/子任务拆分 |
| stanford-oval/storm | 综述生成 | 32k | ✓ | ✗（最后提交 2025-09-30，停更>1年） | — | ✓ | Python | MIT | **仅形态参考**（多视角提问→大纲→写作） |
| （补充）paper-mcp / academix-mcp 等 | 多源检索替代品 | <100 或形态不符 | ✗/— | — | — | — | — | — | 无可比的本地 22 源聚合 MCP+CLI→支撑 paper-search-mcp"唯一通道"判断 |

### 2.3 三口径/豁免判定

**通过（三口径全过或实质通过）**
- **zotero-mcp**：①5.9k✓ ②全✓ ③强✓ → **引入（首选）**。
- **MinerU**：①84k✓ ②全✓ ③强✓ → **引入（首选 PDF→MD）**。风险注记：MinerU 开源许可含附加条款（个人科研无碍；若未来对外在线服务须署名）；Windows+Py3.13 官方仅 3.10–3.12（ray 依赖）→ 用 uv tool 隔离 py3.12 环境安装 mineru，或走 mineru-api/云 HTTP 客户端（旧方案即 MinerU 云）。
- **marker**：①41k✓ ②全✓（注意仓库已迁移 datalab-to/marker）③强✓ → **引入（次选/与 MinerU 对照）**。风险注记：balanced 模式依赖 surya VLM 推理服务（llama.cpp CPU/vLLM GPU）；模型权重 OpenRAIL-M（科研免费）。
- **PyMuPDF/pymupdf4llm**：①②✓ ③中 → **引入（轻量兜底）**。注记 AGPL-3.0。
- **llama_index**：②✓ 但③弱（框架重）→ **次选，暂不引入**（仅当 zotero-mcp 语义检索不敷）。

**差口径② → 豁免评估**
- **paper-search-mcp**（11 月无提交，最后 10-02）：检索聚合"唯一通道"判断成立（同类 paper-mcp/academix-mcp/pvliesdonk-scholar-mcp 在星级、覆盖、形态上均不可比；自研 22 源胶水违反口径③）→ **建议豁免引入：pin 版本 + 记风险**（近 6 个月中 6 个月有提交、10 月初仍在发版，维护连续性尚可）。**待用户确认**。
- **paper-qa**（5/7/9/11 月空，bursty；上次发版 2026-03）：本地语料 RAG **有替代余地**（zotero-mcp 内置 ChromaDB 语义检索随主库②全过；llama_index ②✓）→ 按规则应**拒绝**；但其"科研引文问答+元数据+撤稿检查"契合度最高、替代需多写胶水（违反③）。**建议：豁免 pin 为可选组件或拒绝，请用户裁决**。

**差口径② → 拒绝（有替代）**
- arxiv-mcp-server（6/9 空）：深 arXiv 工具可由 paper-search-mcp arXiv 连接器 + e-print LaTeX 薄胶水覆盖。
- arxiv.py（6/8/9 空）：arXiv REST 直调/连接器覆盖。
- pyalex（2 月起停更）：OpenAlex 由 paper-search-mcp 连接器覆盖（注意需免费 API key）。
- semanticscholar（7 月起静默）、habanero（仅 dependabot）、scholarly（3 月起静默+反爬）：被聚合连接器覆盖/不稳定。
- pdfplumber（6 月起静默）：PyMuPDF 覆盖。
- Better BibTeX（7 月起静默）：zotero-mcp 自带 BibTeX；如需 pin citekey 行为可后置安装（成熟稳定，风险低）。
- GROBID：部署形态（Java 服务）与 Windows+Python 自包含原则冲突，且公式场景 MinerU/marker 更优 → 拒绝。

**形态借鉴（不引入代码）**
- gpt-researcher（活跃）：借鉴"研究任务拆分→子代理并行→证据聚合→报告渲染"的技能编排形态。
- STORM（停更）：借鉴"多视角提问→大纲→逐节写作+引文"的综述管线形态。

---

## 3. 建议组合与薄胶水估计

```
检索/下载/引文 ── paper-search-mcp（MCP 主路 + CLI 回退；pin 版本，豁免②）
                  └ OpenAlex references/citing 有预算查询（citegraph 两跳内）
PDF→MD        ── MinerU（主；uv tool 隔离 py3.12 或云 API）+ marker（次/对照）
                  └ PyMuPDF/pymupdf4llm（轻量兜底，AGPL 注记）
文献库/管理   ── Zotero + zotero-mcp（MCP+zotero-cli；含 ChromaDB 语义检索→承接旧 rag）
语料 RAG/问答 ── zotero-mcp 语义检索（主）± paper-qa（可选，豁免 pin 待裁决）
笔记/rubric   ── 技能文档（TLDR/Key Claims/Novelty/Rigor/Relevance + merge 契约）
citecheck     ── 薄胶水（paper-search-mcp DOI 核验工具 × frontmatter 比对）
综述/报告     ── 技能文档（借鉴 gpt-researcher/STORM 形态），agent 自己写
取证/规范     ── 技能文档（[直取]/[二手互证]）
```

**薄胶水估计（新自研面，合计 ≤400 行）**
1. arXiv e-print LaTeX 源获取 + tex→md 清洗（去 %注释/preamble/thebibliography）≈ 100–150 行；
2. citecheck 薄层（调 paper-search-mcp `get_crossref_paper_by_doi`/OpenAlex 比对 + 分级输出）≈ 100 行；
3. 笔记 frontmatter 归一化 + INDEX.md 生成 ≈ 150 行；
4. 其余全部为技能文档（工作流 9 步、评估 rubric、取证规范、故障手册）。

对比旧项目：21 个自研模块 → 3 个薄胶水模块；维护面大幅收缩。bilingual 生产线（旧最大自研件）默认不保留。

---

## 4. 待用户确认的取舍点

1. **paper-search-mcp 豁免**：口径②差 11 月（最后提交 10-02）。"无可替代（唯一 22 源聚合通道）"判断是否认可 → 豁免引入（pin+风险）？
2. **paper-qa 去留**：bursty（5/7/9/11 月空）。豁免 pin 为可选 RAG 组件，还是按"有替代→拒绝"规则拒绝、RAG 全靠 zotero-mcp 语义检索？
3. **PDF→MD 主力**：MinerU（②全过、公式强、但 Win+Py3.13 需隔离 3.12 环境或云 API）vs marker（②全过、Apache-2.0、balanced 需 VLM 推理服务）。二者都收（主+备）还是只收其一？
4. **中英对照生产线**：旧项目特有大自研件，默认不保留/后置，是否同意？
5. **WoS/Scopus**：是否申请 API key 使用 paper-search-mcp 显式连接器？不配则以 OpenAlex 引文数为准。
6. **付费墙浏览器抓取**（机构 IP Playwright）：保留为薄胶水，还是降级为 agent web 工具按需处理？

---

## 5. 来源 URL 清单

- 星级/最后提交/许可证徽章：`https://img.shields.io/github/{stars,last-commit,license}/<owner>/<repo>`（各候选逐一拉取，2026-11-17）
- 逐月提交探针：`https://commits.ecosyste.ms/api/v1/hosts/GitHub/repositories/<owner>%2F<repo>/commits?per_page=1&until=<date>`（2026-11-17 核验；镜像数据与 shields last-commit 交叉一致）
- paper-search-mcp README（能力矩阵/安装/DSH）：https://raw.githubusercontent.com/openags/paper-search-mcp/main/README.md
- MinerU README + LICENSE.md（环境表脚注 4、许可附加条款）：https://raw.githubusercontent.com/opendatalab/MinerU/master/README.md · https://raw.githubusercontent.com/opendatalab/MinerU/master/LICENSE.md
- PyPI 版本 JSON（requires_python/wheel）：https://pypi.org/pypi/pyalex/json · https://pypi.org/pypi/mineru/3.4.5/json · https://pypi.org/pypi/marker-pdf/2.1.0/json · https://pypi.org/pypi/paper-qa/2026.3.18/json
- zotero-mcp 语义检索文档：https://github.com/54yyyu/zotero-mcp/blob/master/docs/semantic-search.md
- 迁移确认（marker）：https://github.com/datalab-to/marker（与 VikParuchuri/marker 共 SHA 历史，41k★）
- 多源检索替代品检索（支撑"唯一通道"）：https://relatedrepos.com/gh/openags/paper-search-mcp ；paper-mcp（pypi）；kyeshmz/academix-mcp（Cloudflare Workers 形态不符）
- 旧项目能力清单来源：`pySciWS/orchestration/skills/literature-research/`（SKILL.md、references/*.md）、`src/pysci/skills/literature_research/tools/*.py`（21 模块清单及 import 面）、`tests/skills/literature_research/`（40+ 测试名）

## 附：逐月提交证据（探针时间戳，2026）

| 仓库 | 5月 | 6月 | 7月 | 8月 | 9月 | 10月 | 11月(至17) | 判定 |
|---|---|---|---|---|---|---|---|---|
| openags/paper-search-mcp | 05-28 | 06-23 | 07-07 | 08-18 | 09-28 | 10-02 | ✗ | 差 1 月→豁免评估 |
| 54yyyu/zotero-mcp | 05-26 | 06-23 | 07-29 | 08-30 | 09-26 | 10-30 | 11-15 | 全过 |
| opendatalab/MinerU | 05-29 | 06-29 | 07-30 | 08-30 | 09-29 | 10-30 | 11-13 | 全过 |
| datalab-to/marker | 05-05 | 06-29 | 07-20 | 08-27 | 09-13 | 10-29 | 11-14 | 全过 |
| Future-House/paper-qa | ✗ | 06-05 | ✗ | 08-12 | ✗ | 10-26 | ✗ | bursty→豁免/拒绝裁决 |
| blazickjp/arxiv-mcp-server | 05-18 | ✗ | 07-29 | 08-26 | ✗ | 10-08 | 11-14 | 拒绝 |
| lukasschwab/arxiv.py | 05-22 | ✗ | 07-07 | ✗ | ✗ | 10-27 | 11-13 | 拒绝 |
| pymupdf/PyMuPDF | 05-28 | — | — | 08-27 | — | — | 11-13 | 抽检过（高频） |
| 其余（pyalex/semanticscholar/scholarly/pdfplumber/BBT/storm） | — | — | — | — | — | — | 最后提交分别为 02-23/07-03/03-24/06-15/07-13/2025-09-30 | 拒绝/暂缓 |
