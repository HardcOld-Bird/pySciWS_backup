# 通用 Agent 框架 / 工作流自动化平台的科研覆盖面调研

- 调研日期：2026-11-17
- 调研角度：社区优先调研之"通用 Agent 框架 / 工作流自动化平台"角度（对应 PySci 7 能力域：①文献调研 ②有限元仿真(COMSOL) ③科研绘图 ④理论计算 ⑤办公文档写作 ⑥3D建模 ⑦AI生图）
- 核心问题：OpenHands、AutoGen/AG2、CrewAI、LangGraph/LangChain、smolagents、Dify、n8n、Flowise、Open WebUI、MetaGPT 是否构成"社区已有的**一站式科研方案**？"

---

## 0. 结论速览（TL;DR）

**它们不是"一站式科研方案"，而是宿主 / 胶水基座。** 逐条核实后的三点判断：

1. **科研模板高度集中在①文献调研**（n8n 有 7+ 个官方研究类模板、Dify 有 2 个高使用量科研 DSL 模板 + arXiv/Zotero 插件、LangGraph/smolagents 有 deep-research 参考实现、AG2 有 research 群聊示例）。**⑦AI生图** 有平台级插件（Dify ComfyUI/Midjourney、Open WebUI 内置图像生成）。**②COMSOL、③科研绘图、④理论计算、⑥3D建模在所有 10 个框架中均无现成模板**；**⑤办公文档** 只有零散第三方节点（n8n 的 Carbone/TurboDocx 等）。
2. **"框架 = 科研能力"的预期不成立**：CrewAI 官方示例全是营销/招聘/写书类，官方 examples 仓库已归档；LangGraph 官方模板只有 5 个通用骨架；其旗舰 open_deep_research 与 CrewAI 的 crewAI-examples 均已 **archived**。接入框架后，科研能力的绝大部分仍要自己写——"自研负担"只是部分换成了"组装负担"。
3. **真正让"薄胶水更薄"的是 MCP/工具层而非框架层**：Blender MCP（34.5k★）、arXiv MCP（3.3k★）、COMSOL MCP（944★）已达可用规模；而 **LibreOffice MCP（0★）、Overleaf/LaTeX MCP（22★/2★）、Zotero MCP（162★ 且 8 月后停更）均未过 PySci 阈值**，这三块胶水仍需自担。

**PySci 引入口径（star≥100；2026-05~11 每月均有提交；能减少自研代码量）逐条核验结果**：
- ✅ 通过：n8n、Dify、Open WebUI、OpenHands、LangGraph（LangChain 未逐月核，见 §5）、CrewAI、AG2、smolagents
- ❌ 不通过：**microsoft/autogen**（2026-05~09 零提交，2026-10 起文档明示 "archived, no longer maintained"）、**MetaGPT**（最后提交 2026-01-21，2026-05~11 零提交）、**Flowise**（2026-08-13 后零提交，9~11 月为空）

---

## 1. 三类宿主的区分（对 PySci 意义完全不同）

| 类别 | 代表 | 它提供什么 | 对 PySci 的意义 |
|---|---|---|---|
| **通用 agent 运行时** | OpenHands | 一个能在容器里读改代码、执行命令、装依赖的"数字工人"（含 Skills/Plugins/Plugin Marketplace/MCP） | 可作为**执行器工具接入点**：替代自研"跑脚本的 agent"；但它不带科研语义，①-⑦能力仍要给它写 skill |
| **agent 开发框架** | LangGraph/LangChain、CrewAI、AG2/AutoGen、smolagents、MetaGPT | 编排原语（图/群聊/角色/CodeAgent）+ 少量参考模板 | **胶水基座**：把 PySci 技能串成流程的骨架；不是能力来源，科研逻辑要自写 |
| **自动化 / 低代码平台** | n8n、Dify、Flowise、Open WebUI | 可视化工作流、插件市场、可导入 DSL/JSON 模板、自带 UI 与调度 | **模板可直接复用的唯一一层**（主要是①⑦⑤）；代价是引入 Node/Docker 常驻服务，与 PySci"自包含、库优先"有张力 |

---

## 2. 框架对比总表（数据核实于 2026-11-17）

数据源：[ecosyste.ms repos API](https://repos.ecosyste.ms/)（GitHub 元数据镜像，last_synced 2026-11-11~16）+ [commits.ecosyste.ms API](https://commits.ecosyste.ms/)（逐月提交核验，方法见 §5）。GitHub HTML 页在本环境被导航截断、shields.io/api.github.com 与本机 PowerShell 出网均不可用，故统一改用上述开放镜像。

| 框架 | 类别 | Stars | 最近提交(pushed_at) | 2026-05~10 逐月提交 | 许可证 | 部署形态 | Windows | PySci 口径判定 |
|---|---|---:|---|---|---|---|---|---|
| [n8n-io/n8n](https://github.com/n8n-io/n8n) | 自动化平台 | 210,201 | 2026-11-11 | 5✓6✓7✓8✓9✓10✓(11/16 亦有) | fair-code（Sustainable Use + n8n License，**非 OSI**；自托管可用） | npm(npx)/Docker/官方云 | 原生支持 npm 安装（文档层面，未实机核实） | ✅ 通过 |
| [langgenius/dify](https://github.com/langgenius/dify) | 自动化平台 | 161,152 | 2026-11-14 | 5✓6✓7✓8✓9✓10✓ | 修改版 Apache-2.0（禁多租户、禁去 LOGO） | Docker Compose 自托管 / SaaS | 需 Docker Desktop + WSL2（未实机核实） | ✅ 通过 |
| [open-webui/open-webui](https://github.com/open-webui/open-webui) | 自动化平台(UI) | 158,341 | 2026-11-11 | 5✓6✓7✓8✓9✓10✓ | 自定义 Open WebUI License（BSD-3 + **品牌保留条款**：>50 用户不得移除品牌，非 OSI） | pip / Docker | pip 方式可用（未实机核实） | ✅ 通过（许可证需留意） |
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | 开发框架 | 149,684 | 2026-11-16 | **未逐月核实**（仅核 star/pushed_at） | MIT | Python 库（pip/uv） | 纯 Python，可用 | ✅ 通过（活跃度按 pushed_at 推断） |
| [FoundationAgents/MetaGPT](https://github.com/FoundationAgents/MetaGPT) | 开发框架 | 71,288 | **2026-01-21** | 5~11 月 **0 提交**（API 查询返回空） | MIT | pip / Docker | 一般可用 | ❌ 活跃度不通过 |
| [microsoft/autogen](https://github.com/microsoft/autogen) | 开发框架 | 61,822 | 2026-11-12（仅文档） | **5✗6✗7✗8✗9✗** 10✓(仅 docs)11✓(仅 docs) | 根 LICENSE 检出为 CC-BY-4.0 | Python 库 | 可用 | ❌ 不通过：维护模式→归档 |
| [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | 开发框架 | 60,902 | 2026-11-13 | 5✓6✓7✓8✓9✓10✓ | MIT | Python 库 | 可用 | ✅ 通过 |
| [FlowiseAI/Flowise](https://github.com/FlowiseAI/Flowise) | 自动化平台 | 55,570 | **2026-08-13** | 5✓6✓7✓8✓(8/13 止) 9✗10✗ | Apache-2.0 主体 + `enterprise/` 商业许可 | npm / Docker / 云 | 支持 | ❌ 活跃度不通过 |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | 开发框架 | 44,848 | 2026-11-12 | 5✓6✓7✓8✓9✓10✓ | MIT | Python 库 + LangGraph CLI/Server | CLI 可用（本地 server 建议 WSL2，未实机核实） | ✅ 通过 |
| [huggingface/smolagents](https://github.com/huggingface/smolagents) | 开发框架 | 30,305 | 2026-11-10 | 5✓6✓7✓8✓9✓10✓（多为 docs/CI/安全类，功能开发放缓） | Apache-2.0 | Python 库 | 可用 | ✅ 通过（注意强度下降） |
| [ag2ai/ag2](https://github.com/ag2ai/ag2)（AutoGen 社区延续） | 开发框架 | 5,060 | 2026-11-12 | 5✓6✓7✓8✓9✓10✓ | Apache-2.0 | Python 库 | 可用 | ✅ 通过 |
| [OpenHands/OpenHands](https://github.com/OpenHands/OpenHands)（原 All-Hands-AI） | agent 运行时 | 93,606 | 2026-11-16 | 5✓6✓7✓8✓9✓10✓ | MIT | pip/uv CLI、Docker、Web/企业版、SaaS | 官方以 Linux/Docker 为主，Windows 需 WSL2/Docker（未实机核实） | ✅ 通过 |

### 关键负面证据（逐条核实）

- **AutoGen 已死**：GitHub atom 提交流显示 2025-09 后几乎无代码提交；2026-04-06 "Update maintenance mode banner … redirect new users to Microsoft Agent Framework"；2026-10-16 "docs: clarify AutoGen is archived and no longer maintained"；2026-10-28 "docs: add archive caution to landing page"。ecosyste.ms 逐月查询 2026-05~09 返回 `[]`。
- **MetaGPT 停更**：pushed_at = 2026-01-21；2026-05-01~11-17 提交查询返回 `[]`（archived 标志仍为 false，但实质停更）。
- **Flowise 半停更**：最后提交 2026-08-13；2026-09-01~11-17 查询返回 `[]`；2026-08-03 提交里还出现 SECURITY.md "Sunset notice"。
- **两个旗舰科研模板被归档**：[langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research)（12,678★，MIT，`archived: true`，最后提交 2026-08-10）、[crewAIInc/crewAI-examples](https://github.com/crewAIInc/crewAI-examples)（6,162★，`archived: true`，最后提交 2026-04-20）。

---

## 3. 科研模板 / 示例清单（按 7 个能力域）

图例：✅ 有现成官方或高星社区模板 ｜ △ 有零件/可拼但无成品 ｜ ✗ 未找到

| 能力域 | n8n | Dify | LangGraph/LangChain | smolagents | AG2 | CrewAI | OpenHands | Open WebUI | Flowise | MetaGPT |
|---|---|---|---|---|---|---|---|---|---|---|
| ①文献调研 | ✅ | ✅ | ✅ | ✅ | ✅ | △ | △ | △ | ✗ | ✗(停更) |
| ②COMSOL 仿真 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| ③科研绘图 | △ | △ | △ | △ | ✗ | ✗ | △ | ✗ | ✗ | ✗ |
| ④理论计算 | △ | △ | △ | △ | ✗ | ✗ | △ | ✗ | ✗ | ✗ |
| ⑤办公文档 | △ | △ | △ | ✗ | ✗ | ✗ | △ | ✗ | ✗ | ✗ |
| ⑥3D 建模 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| ⑦AI 生图 | △ | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ | ✅ | ✗ | ✗ |

> △ 的含义：平台提供了通用代码节点 / 通用节点（Dify Code Node、n8n Code/Python 节点、LangGraph 的 Python REPL 工具、OpenHands 直接写脚本），能"自己拼出来"，但**没有科研成品模板**——即这部分负担仍在 PySci 侧。

### ① 文献调研（唯一被充分覆盖的域）

| 模板/示例 | 平台 | 性质与质量 | 覆盖点 |
|---|---|---|---|
| [ResearchOps MegaAgent 科研工程化选题与文献情报平台](https://marketplace.dify.ai/template/shorpan/701bf356-d7ac-4bb2-a883-87d233cbfd5c) | Dify 模板市场 | 个人发布者 shorpan，v0.6.0，**usage_count=1035**（2026-06-15 发布）；DSL 可直接导入；质量中上（arXiv/OpenAlex/Crossref/SemanticScholar/PubMed/EuropePMC 六源检索→去重打分→文献矩阵/研究空白/方法建议报告）；**弱点：Semantic Scholar 走第三方代理需自备 key** | ① |
| [Hugging Face Daily Paper Digest](https://marketplace.dify.ai/template/langgenius/Hugging%20Face%20Daily%20Paper%20Digest) | Dify 模板市场 | **官方组织 langgenius 发布**，v0.5.0，**usage_count=2557**（2026-03 发布、2026-07 更新）；定时触发+邮件推送，工程质量较好 | ① |
| [arXiv 论文摘要工作流 #2904](https://n8n.io/workflows/2904-arxiv-paper-summarization-with-chatgpt/)、[系统性文献综述 #12827](https://n8n.io/workflows/12827-automate-systematic-literature-reviews-with-google-drive-gpt-4-gemini-qdrant-and-airtable/)、[论文元数据抽取 #9994](https://n8n.io/workflows/9994-automated-academic-paper-metadata-and-variable-extraction-with-gemini-to-google-sheets/)、[从标题摘要生成期刊论文 #10314](https://n8n.io/workflows/10314-qwen-max-journal-paper-generation-from-titleabstract/) | n8n 官方模板库（社区上传、官方收录，库总量 **14,381** 个工作流） | 单点工作流，质量参差（多为 demo 级），需自行接 LLM key；但**结构可直接 JSON 导入** | ①（+⑤ 抽论文要素） |
| [Open deep research #2883](https://n8n.io/workflows/2883-open-deep-research-ai-powered-autonomous-research-workflow/)、[自建 deep research #2878](https://n8n.io/workflows/2878-host-your-own-ai-deep-research-agent-with-n8n-apify-and-openai-o3/)、[Deep research + Notion #7160](https://n8n.io/workflows/7160-deep-research-agent-automated-research-and-notion-report-builder/) | n8n 官方模板库 | 深度研究编排，成熟度中 | ① |
| [open_deep_research](https://github.com/langchain-ai/open_deep_research) | LangChain 官方 | 12.7k★ LangGraph 深度研究参考实现，**已 archived（2026-08）**，作参考代码仍极有价值 | ① |
| LangGraph 官方模板：[ReAct / Memory / Retrieval / Data-Enrichment 等 5 个](https://docs.langchain.com/oss/python/langgraph/) | LangChain 官方（langgraph repo `docs/concepts/template_applications.md`） | 通用骨架，**无科研专属**；`langgraph new` 可脚手架 | △① |
| [examples/open_deep_research](https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research) | Hugging Face 官方示例 | CodeAgent 深度研究（含 HF space demo），代码质量高、可直接改 | ① |
| [agentchat_groupchat_research](https://docs.ag2.ai/0.9.3/docs/use-cases/notebooks/notebooks/agentchat_groupchat_research/)、agentchat_web_info | AG2/AutoGen 官方 notebook | 教程式（多 agent 群聊做 research + 网络信息工具），非成品 | △① |
| CrewAI：官方 [Examples 页](https://docs.crewai.com/v1.15.41/en/examples/example.md) 无科研类；`crewAI-examples` 已归档；社区有 [crewai-AutoResearch-Synthesis](https://github.com/notsointresting/crewai-AutoResearch-Synthesis) | CrewAI | 官方科研模板缺位，社区项目 star 未过阈值 | △① |
| Open WebUI：[Web Search + Save to Knowledge](https://openwebui.com/posts/web_search_09105af9) | Open WebUI | 搜索→RAG 入库的通用能力，非科研模板 | △① |
| OpenHands：[Skills and Plugins](https://docs.openhands.dev/enterprise/skills-and-plugins) + [Plugin Marketplace](https://docs.openhands.dev/enterprise/plugin-marketplace) | OpenHands 官方 | 有技能/插件机制与市场（企业版文档），**未见官方科研技能**；靠"写码能力"覆盖 | △① |

### ⑦ AI 生图

| 来源 | 说明 |
|---|---|
| [Dify 官方插件：图像生成与媒体工具集](https://github.com/langgenius/dify-official-plugins)（含 ComfyUI、OpenAI Images 等）+ [ComfyUI 插件](https://marketplace.dify.ai/plugin/langgenius/comfyui) + [Midjourney 插件 (AceDataCloud)](https://marketplace.dify.ai/plugin/acedatacloud/midjourney) | 平台级插件，可直接装；Dify 模板市场亦有生图应用 |
| [Open WebUI 内置图像生成/编辑](https://docs.openwebui.cn/features/image-generation-and-editing/usage/)（OpenAI Images、ComfyUI、A1111 后端可配） | ✅ 开箱即用 |
| n8n 社区工作流（如 [n8n-ai-art-content-machine](https://github.com/justhassann/n8n-ai-art-content-machine)） | 成品级社媒向，科研向需改 | 

### ⑤ 办公文档（仅零散零件）

- n8n：第三方节点 [n8n-nodes-carbone](https://github.com/carboneio/n8n-nodes-carbone)（DOCX/PPTX/ODT 模板渲染→PDF）、[TurboDocx×n8n 集成](https://www.turbodocx.com/integrations/n8n/document-generation)；官方有 Google Docs/Drive 类模板。
- Dify：以文档**解析**（RAG 摄入）为主，生成 Office 文档无官方插件。
- LaTeX：见 §4（Overleaf MCP 生态 <100★）。

### ②③④⑥（COMSOL / 绘图 / 理论计算 / 3D）

- **10 个框架的模板/市场中均未检索到任何科研有限元、符号/数值数学、科研绘图（matplotlib/seaborn/Origin 级）、3D 建模成品模板**。
- ③④ 可用通用 code node（Dify Code Node、n8n Code node、LangGraph Python REPL 工具、OpenHands 写脚本）承载，但模板层为空。
- ⑥ 的唯一成熟能力来自**平台之外**的 Blender MCP（§4）。

---

## 4. 外部重型工具集成生态（"薄胶水能否更薄"的关键）

| 工具 | 现成集成 | Stars | 许可证 | 最近提交 | 2026-05~11 活跃 | PySci 口径 | 结论 |
|---|---|---:|---|---|---|---|---|
| **Blender（⑥）** | [ahujasid/mcp-for-blender](https://github.com/ahujasid/mcp-for-blender)（原名 blender-mcp） | **34,494** | MIT | 2026-11-15 | 持续（pushed_at 近今） | ✅ 双指标通过 | **胶水可显著变薄**：⑥ 只需写技能文档 + MCP 挂接 |
| **arXiv/文献（①）** | [blazickjp/arxiv-mcp-server](https://github.com/blazickjp/arxiv-mcp-server) | **3,271** | Apache-2.0 | 2026-10-08 | 活跃（10 月有提交；11 月未见） | ⚠️ star✓，11月未核到提交 | 可作为①检索底座 |
| **COMSOL（②）** | [wjc9011/COMSOL_Multiphysics_MCP](https://github.com/wjc9011/COMSOL_Multiphysics_MCP) | **944** | MIT | 2026-10-31 | 5✓6✓7✓8✓9✓10✓，**11/1~17 无提交** | ⚠️ star✓；11 月缺（17 天窗口） | **重要发现：②已有 MCP 级胶水**（含声学/PDE/网格序列工具、论文引用指引），但项目年轻（2026-02 创建），需评估成熟度 |
| **Zotero（①）** | [kujenga/zotero-mcp](https://github.com/kujenga/zotero-mcp) | 162 | MIT | **2026-08-07** | 8/7 后无提交（9~11 空） | ❌ 活跃度不通过 | 星数刚过线但已停滞；另有 richardjlyon/zotero-mcp(1★)、PyPI `zotero-mcp-server` — 生态小且散 |
| **Zotero（平台内）** | Dify 插件 [jay0130-a/zotero](https://marketplace.dify.ai/plugin/jay0130-a/zotero)；n8n 社区模板（[ayn8n](https://www.ayn8n.com/workflows/automated-retrieval-of-bibliographic-data-from-zotero-library)、[n8nflow](https://www.n8nflow.net/workflows/2444-get-bibliographic-data-from-your-zotero-library)、[lapage](https://lapage.vn/utilities/n8n-templates/zotero-collection-items-fetcher)） | — | — | — | — | ⚠️ 星数不可核（平台内无 star） | 可当参考实现，不构成"社区已有" |
| **LaTeX/Overleaf（⑤）** | [ghostiee-11/overleaf-mcp](https://github.com/ghostiee-11/overleaf-mcp)（24 tools）22★；[nomanfoundhere/overleaf-mcp](https://github.com/nomanfoundhere/overleaf-mcp) 2★ | <100 | MIT | 2026-07 / 2026-09 | — | ❌ star 不达标 | 但 LaTeX 本身是 CLI 工作流（latexmk/pandoc），**自研胶水天然薄**，可不依赖 MCP |
| **LibreOffice（⑤）** | [sandraschi/libreoffice-mcp](https://github.com/sandraschi/libreoffice-mcp) 4★；[patrup/mcp-libre](https://github.com/patrup/mcp-libre) 0★；[jwingnut/mcp-libre](https://github.com/jwingnut/mcp-libre) 0★；npm `@passerbyflutter/libreoffice-mcp-tools` | ≈0 | MIT | 2025-06 ~ 2026-10 | 无一活跃+过星 | ❌ 双指标均不通过 | **⑤LibreOffice 桥接必须自研薄胶水**（headless/UNO，工作量可控） |
| **Office 文档生成（⑤）** | [carboneio/n8n-nodes-carbone](https://github.com/carboneio/n8n-nodes-carbone)（仅 n8n 内） | 未核 | — | — | — | ⚠️ | 思路可借鉴（模板渲染引擎） |

**要点**：MCP 生态呈现"**两端热、中间冷**"——3D（Blender 34.5k★）和文献（arXiv 3.3k★）两端成熟；办公文档（LibreOffice/LaTeX/Zotero）中间薄弱。因此 PySci 的 7 域里，⑥ 和 ① 的胶水最薄，⑤ 的胶水最厚，② 处于"有 944★ MCP 但需验证"的中间态。

---

## 5. 核实方法与口径执行记录

- **star / pushed_at / license**：`https://repos.ecosyste.ms/api/v1/hosts/GitHub/repositories/{owner}/{repo}`（ecosyste.ms 是 GitHub 元数据开放镜像，本调研时 last_synced 2026-11-11~16）。
- **逐月提交**：`https://commits.ecosyste.ms/api/v1/hosts/GitHub/repositories/{owner}/{repo}/commits?since=<月初>&until=<下月1日>&per_page=1`，对 2026-05~10 每月各查一次（返回 `[]` 即该月无提交）；11 月以 pushed_at（或单独查 11/1~11/17）补充。n8n 的 5 月、Flowise 的 6~8 月等由单次窗口查询交叉验证。
- **归档/维护状态**：GitHub `commits/main.atom` 提交流（AutoGen 案例）+ ecosyste.ms `archived` 字段（open_deep_research、crewAI-examples 案例）。
- **模板质量**：Dify 用 `https://marketplace.dify.ai/api/v1/templates/{id}` 取 usage_count/版本/发布日期；n8n 用模板页 URL 与库规模（14,381）；CrewAI 用官方 docs 的 examples 页。
- **未逐条核实项（诚实声明）**：
  1. LangChain 主仓未做逐月提交查询（仅 star + pushed_at=2026-11-16）；
  2. Windows 支持均为**文档/部署形态层面的判断**，未在本机实测（PySci 是 Windows 11 + Python 3.13 环境，Dify/n8n/OpenHands 依赖 Docker/WSL2，smolagents/CrewAI/LangGraph/AG2 为纯 pip 库）；
  3. Dify/n8n/Flowise/Open WebUI 平台内模板市场的**完整科研类目清单**未能全量枚举（市场页为 JS 应用，抓取被截断），本报告列出的是检索可确认的条目，可能低估；
  4. `ahujasid/mcp-for-blender`、`zotero-mcp` 等仅核了 star/pushed_at，未做逐月提交全查（Blender MCP pushed_at=2026-11-15 足够判定活跃）；
  5. 本次会话中 shields.io、api.github.com、libraries.io 及本机 PowerShell 出网均失败（TLS/网络限制），全部数据改走 ecosyste.ms 与 raw.githubusercontent/atom feed。

---

## 6. 对 PySci 的建议（工具接入点，而非能力来源）

1. **不要把任何框架当"科研能力来源"**：②③④⑥ 在 10 个框架中模板全空，① 的成品也只是"检索+摘要+报告"流水线，与 PySci 的技能文档模型不匹配。框架的价值是**执行宿主与流程骨架**。
2. **值得作为"工具接入点"的（按 PySci 三口径全过）**：
   - **n8n**（210k★）：唯一同时拥有"可导入的科研工作流模板 + 文档生成节点 + 调度"的自动化层；适合做**跨域编排/定时任务**（文献日报、批量出图流水线）。代价：Node 常驻服务 + fair-code 许可（自托管 OK，不可做竞品 SaaS）。
   - **Dify**（161k★）：科研 DSL 模板（ResearchOps、HF Paper Digest）可直接导入复用，插件市场补齐 arXiv/Zotero/生图；适合做**面向人的科研 UI**。代价：Docker 部署 + 修改版 Apache（多租户/LOGO 限制）。
   - **Blender MCP（34.5k★）** 与 **arXiv MCP（3.3k★）**：作为 ⑥① 的**现成胶水**直接接入，PySci 侧只写技能文档。
   - **OpenHands（93.6k★）**：作为"能执行科研脚本的通用 worker"的候选宿主；若与 PySci 现有 worker 体系重叠，则只取其 Skills/MCP 约定，不引入运行时。
   - **LangGraph（44.8k★, MIT）/ smolagents（30.3k★, Apache-2.0）**：作为**技能编排库**（in-process 库形态，最贴合 PySci"自包含、库优先"），参考 open_deep_research 的实现写①的技能。
3. **需警惕/排除**：AutoGen（已归档）、MetaGPT（停更 10 个月）、Flowise（停更 3 个月 + enterprise 双许可）；Open WebUI 许可证带品牌保留条款（>50 用户）；AG2/CrewAI 虽过口径但官方科研模板缺位。
4. **胶水厚度预判（⑤最厚）**：LibreOffice/Overleaf/Zotero 三个 MCP 均未过阈值——⑤ 的桥接代码须计入自研预算；可优先走 LibreOffice headless CLI + 模板渲染的朴素路线，绕开 MCP 生态的不成熟。
5. **一句话定性**：这批框架 = "**宿主 + 胶水基座**"；它们把"写 agent 运行时"的自研负担省掉了，但没有把"科研能力"交付给你——一站式科研方案在 2026-11 的社区里**仍然不存在**。

---

## 7. 来源 URL 汇总

**元数据/活跃度**
- https://repos.ecosyste.ms/ （及 `/api/v1/hosts/GitHub/repositories/{owner}/{repo}`）
- https://commits.ecosyste.ms/ （及 `/api/v1/hosts/GitHub/repositories/{owner}/{repo}/commits?since=&until=`）
- https://github.com/microsoft/autogen/commits/main.atom （AutoGen 归档证据）
- https://raw.githubusercontent.com/langgenius/dify/main/LICENSE ｜ https://raw.githubusercontent.com/open-webui/open-webui/main/LICENSE ｜ https://raw.githubusercontent.com/FlowiseAI/Flowise/main/LICENSE.md

**模板/示例**
- https://marketplace.dify.ai/template/shorpan/701bf356-d7ac-4bb2-a883-87d233cbfd5c （ResearchOps MegaAgent）
- https://marketplace.dify.ai/api/v1/templates/253d7ffa-442c-4744-99b6-510b3d6f53bb （HF Daily Paper Digest，usage 2557）
- https://marketplace.dify.ai/plugin/langgenius/arxiv ｜ /plugin/jay0130-a/zotero ｜ /plugin/langgenius/comfyui ｜ /plugin/acedatacloud/midjourney
- https://n8n.io/workflows/2904-arxiv-paper-summarization-with-chatgpt/ ｜ /12827-... ｜ /9994-... ｜ /10314-... ｜ /2883-open-deep-research-... ｜ /2878-... ｜ /7160-...
- https://github.com/langchain-ai/open_deep_research （archived）
- https://raw.githubusercontent.com/langchain-ai/langgraph/f8c1a323cc50a6da14836ebc4c53c79db0f6508c/docs/docs/concepts/template_applications.md （LangGraph 官方 5 模板）
- https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research
- https://docs.crewai.com/v1.15.41/en/examples/example.md （CrewAI 官方示例清单，无科研）｜ https://github.com/crewAIInc/crewAI-examples （archived）
- https://docs.ag2.ai/0.9.3/docs/use-cases/notebooks/notebooks/agentchat_groupchat_research/
- https://docs.openhands.dev/enterprise/skills-and-plugins ｜ /enterprise/plugin-marketplace
- https://openwebui.com/posts/web_search_09105af9 ｜ https://docs.openwebui.cn/features/image-generation-and-editing/usage/
- https://github.com/carboneio/n8n-nodes-carbone ｜ https://www.turbodocx.com/integrations/n8n/document-generation

**重型工具集成**
- https://github.com/ahujasid/mcp-for-blender ｜ https://github.com/wjc9011/COMSOL_Multiphysics_MCP ｜ https://github.com/blazickjp/arxiv-mcp-server
- https://github.com/kujenga/zotero-mcp ｜ https://github.com/richardjlyon/zotero-mcp ｜ https://github.com/sandraschi/libreoffice-mcp ｜ https://github.com/patrup/mcp-libre ｜ https://github.com/jwingnut/mcp-libre
- https://github.com/ghostiee-11/overleaf-mcp ｜ https://github.com/nomanfoundhere/overleaf-mcp
