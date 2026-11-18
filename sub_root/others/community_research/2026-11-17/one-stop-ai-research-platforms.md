# 一站式科研智能体 / 自动化研究平台 —— 社区调研报告

- 调研日期：2026-11-17
- 调研角度：社区里有没有项目把科研工作流（文献调研→理论/计算→实验代码/绘图→论文写作，甚至仿真、3D、AI 生图）打包成"一站式"解决方案？还是都只覆盖个别环节？
- 能力域编号（PySci 7 域）：①文献调研 ②有限元仿真（COMSOL）③科研绘图 ④理论计算（符号/数值）⑤办公文档写作（Word/PPT/Excel/LaTeX/论文）⑥3D 建模 ⑦AI 生图（扩散模型）
- PySci 引入判定口径：1) star ≥ 100；2) 2026-05～2026-11 每月均有提交；3) 能减少自研代码量。

---

## 0. 核心结论（TL;DR）

1. **没有任何开源项目覆盖 PySci 的 ①-⑦ 全组合**。②有限元仿真（COMSOL）、⑥3D 建模（Blender/CAD）、⑦本地扩散模型生图这三个小众项，在所有调研到的"科研智能体"项目中**无一覆盖**（个别项目有"AI 配图"但均为调用托管 API，如 Gemini 图像模型 / MiniMax，不是本地扩散工作流）。
2. 社区的"一站式"实际是 **①文献/网络调研 → ⑤报告/论文写作** 这一条主轴（部分附带 ④轻量 Python 代码执行、③由代码间接绘图）。真正宣称"全自动科研闭环"的项目（idea→实验代码→论文）是 **SakanaAI/AI-Scientist(-v2)** 与 **SamuelSchmidgall/AgentLaboratory**，但两者分别在 2025-12 与 2025-08 后**停止提交**，不满足活跃度口径。
3. **活跃且最接近"一站式"的两个**：
   - **bytedance/deer-flow**（85,344★，MIT）：2.0 起重写为"SuperAgent harness"（子代理+沙箱+持久记忆+skills/tools/MCP/全栈插件平台），是唯一活跃的"通用科研+创造编排底座"，理论上可挂 COMSOL/Blender 等自定义工具——但它是一个平台级产品，不是薄胶水。
   - **assafelovic/gpt-researcher**（30,690★，Apache-2.0）：①+⑤ 做得最深（含 paper UI、python-pptx 导出、AI 配图、arXiv/PubMed/Semantic Scholar 检索源），扩展机制齐全（MCP client/server、entry-point 检索插件、`.claude/skills/` 技能包），三条引入判定全部通过。
4. ①文献环节若要"库级"嵌入，**Future-House/paper-qa**（9,583★，Apache-2.0，PyPI 活跃发版）是最成熟的 Python 库；要"应用级"自托管则 **LearningCircuit/local-deep-research**（9,567★，MIT，当前极活跃，v2.0.x）。
5. 形态上值得注意的趋势：**"agent skill" 形态**正在兴起（hyperresearch 4,769★、DeepPaperNote、claude-deep-research-skill 等，把研究流程做成 Claude Code/Codex 技能包），与 PySci "技能文档 + 社区工具/薄胶水"的重建路线同构，可直接借鉴其技能组织方式。

---

## 1. 调研方法说明（数据来源与限制）

### 1.1 环境限制
- 本会话 pwsh 无法外联（TLS 握手失败，直连与 `http://127.0.0.1:12450` 代理均失败），故未用 curl/API 批量抓取。
- `api.github.com`、`img.shields.io`、`r.jina.ai`、`libraries.io`、`repos.ecosyste.ms/api/v1/*`（JSON API）经 web_fetch 均不可达（fetch failed / 401 / 超时）。
- `github.com` 仓库 HTML 主页可达但正文被截断，star 数位于截断线之后，无法直接读取。

### 1.2 实际采用的数据源（均经 web_fetch 核实）
| 数据源 | 用途 | 可靠性 |
|---|---|---|
| `repos.ecosyste.ms/hosts/GitHub/repositories/<owner>%2F<repo>` | star 数、fork、许可证、语言、pushed_at、tag 列表 | 高（同步时间为调研前 2-8 天，页面标注 "Last synced at"） |
| `commits.ecosyste.ms/hosts/GitHub/repositories/<owner>%2F<repo>/commits` | 逐条提交及相对日期（每页 100 条）、"过去一年提交数/贡献者" | 高 |
| `github.com/<owner>/<repo>/commits/<branch>.atom` | 最近 20 条提交的**精确 UTC 日期**（含 `<updated>` 最后提交时间） | 高 |
| `repos.ecosyste.ms/.../releases` | release/tag 及相对日期、release notes（功能细节） | 高 |
| `raw.githubusercontent.com`（README） | 定位、功能、扩展机制 | 高 |
| web_search（含知乎/钛媒体/腾讯云开发者/公众号检索） | 中文社区讨论、候选补充、awesome 清单 | 中（摘要级） |
| `AaronAust1n/awesome-deep-research` README（raw） | 候选枚举、star 量级、归档状态 | 中（清单自报数据，标"清单数据"者未独立核实） |

### 1.3 核实口径
- **star 数**：以 ecosyste.ms 为准（调研日快照）；awesome 清单/搜索摘要数据仅作量级参考并标注。
- **"近 6 个月每月均有提交"（2026-05～2026-11）**：优先用 commits.ecosyste.ms 逐条提交日期 + release 日期逐月对照；仅能给出"最后提交时间"的标注"逐月未能核实"。
- **未能核实项**：`princeton-nlp/OpenScholar`、`osunlp/MDAgents`、独立 `AgentRxiv` 仓库在 ecosyste.ms 均 404（未收录或已删除/转移），star 数与活跃度未能核实，另行标注。

---

## 2. 总览对比表

覆盖列：✓=覆盖；(~)=间接/部分（如通过代码执行绘图、LaTeX 论文、API 配图）；空=不覆盖。

| 项目 | star | 形态 | 活跃度（2026-05~11） | 许可证 | ① | ② | ③ | ④ | ⑤ | ⑥ | ⑦ | 判定①②③ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bytedance/deer-flow | 85,344 | 自托管 SuperAgent 平台(Python) | ✅ 每月（v2.0.0≈6月/v2.1.0≈9月/v2.2.0≈10-11月，push 2天前） | MIT | ✓ | ✗ | (~) | (~) | ✓ | ✗ | (~)API生图技能 | ✅✅✅ |
| assafelovic/gpt-researcher | 30,690 | Python库+自托管Web+MCP server | ✅ 每月（100 commits/21天；v3.5.0≈5月→v3.7.0≈9月） | Apache-2.0 | ✓ | ✗ | ✗ | ✗ | ✓(含pptx/paper UI) | ✗ | (~)Gemini配图 | ✅✅✅ |
| stanford-oval/storm | 31,879 | Python库+自托管服务 | ❌ 停止（最后提交2025-09-30） | MIT | ✓ | ✗ | ✗ | ✗ | ✓(长报告) | ✗ | ✗ | ✅❌✅ |
| huggingface/smolagents | 30,305 | Python agent 框架 | ✅（push 5天前，v1.26.0） | Apache-2.0 | (~)例程 | ✗ | ✗ | (~) | (~) | ✗ | ✗ | ✅✅✅(框架) |
| Alibaba-NLP/DeepResearch(Tongyi) | 20,139 | 深研agent+开源模型 | ❌（push≈2026-02） | Apache-2.0 | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✅❌✅ |
| langchain-ai/open_deep_research | 12,678 | Python(工作流/多agent) | ❌（已归档*，push≈2026-08） | MIT | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✅❌✅ |
| SakanaAI/AI-Scientist | 14,990 | 研究流水线(模板式) | ❌（push≈2025-12） | other(自定义) | ✓ | ✗ | ✓ | ✓ | ✓(LaTeX) | ✗ | ✗ | ✅❌✅ |
| SakanaAI/AI-Scientist-v2 | 7,478 | 研究流水线(树搜索) | ❌（最后提交2025-12-19） | other(自定义) | ✓ | ✗ | ✓ | ✓ | ✓(LaTeX) | ✗ | ✗ | ✅❌✅ |
| Future-House/paper-qa | 9,583 | Python 库（RAG） | ⚠️ 有月度空档（7月、9-10月中无提交；10月下旬/11月恢复） | Apache-2.0 | ✓(深) | ✗ | ✗ | ✗ | (~) | ✗ | ✗ | ✅⚠️✅ |
| LearningCircuit/local-deep-research | 9,567 | 自托管Web应用 | ✅ 极活跃（100 commits/8天，v2.0.1≈本周） | MIT | ✓ | ✗ | ✗ | ✗ | ✓(多格式导出) | ✗ | ✗ | ✅✅✅ |
| SamuelSchmidgall/AgentLaboratory | 5,931 | 研究流水线(端到端) | ❌（最后提交2025-08-20） | MIT | ✓ | ✗ | ✓ | ✓ | ✓(LaTeX) | ✗ | ✗ | ✅❌✅ |
| HKUDS/AI-Researcher | 5,884 | 自主创新研究系统 | ❌（push≈2025-11） | ⚠️无 | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✅❌✅ |
| jordan-gibbs/hyperresearch | 4,769 | **agent skill**+MCP | ✅（创建于2026-04，push 4天前，v0.13.0） | MIT | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✅✅✅ |
| nickscamara/open-deep-research | 6,298 | Next.js Web 应用 | ❌（push>1年） | other | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✅❌✅ |
| lamm-mit/SciAgentsDiscovery | 638 | 多agent(本体图) | ❌（push>1年） | Apache-2.0 | ✓ | ✗ | ✗ | (~) | ✗ | ✗ | ✗ | ✅❌✅ |
| multimodal-art-projection/AutoKaggle | 319 | 多agent数据科学 | ❌（push≈2024-12） | Apache-2.0 | ✗ | ✗ | (~) | ✓ | ✗ | ✗ | ✗ | ✅❌✅ |
| scicode-bench/SciCode | 246 | **基准测试**（非平台） | ✅（push≈2026-11-03） | Apache-2.0 | ✗ | ✗ | ✗ | (~) | ✗ | ✗ | ✗ | ✅✅— |
| OpenScholar / MDAgents / AgentRxiv | 未能核实 | — | — | — | — | — | — | — | — | — | — | 见 §4 |

\* langchain-ai/open_deep_research 的"已归档"状态据 awesome-deep-research 清单（标注 status-archived），GitHub 页截断未能直接核实归档横幅；ecosyste.ms 显示最后 push 约 2026-08。

---

## 3. 逐候选详细评估

### 3.1 全流程自动化研究类

#### 3.1.1 SakanaAI/AI-Scientist（v1）
| 指标 | 值 | 来源 |
|---|---|---|
| star / fork | 14,990 / 2,089 | ecosyste.ms（同步于调研前 4 天） |
| 许可证 | other（自定义；ecosyste.ms 检测为 "other"） | ecosyste.ms |
| 最后 push | 约 2025-12（"Pushed at 11 months ago"） | ecosyste.ms |
| 近 6 个月提交 | 无 | ecosyste.ms pushed_at |

- **定位**：全流程自动化研究——自动产生 idea→改代码做实验（ML 模板：nanoGPT 等）→绘图→LaTeX 论文→自动审稿→迭代。是"端到端科研闭环"的标杆。
- **覆盖**：①(相关工作调研) ③(实验图) ④(代码实验) ⑤(LaTeX 论文)。**实验域仅限机器学习**，无 ②⑥⑦。
- **形态**：研究流水线脚本集合（Python + Jupyter），LLM 依赖强（OpenAI/Anthropic/DeepSeek 等 API）；非服务、非库。
- **扩展性**：模板机制（新增研究模板可跑不同实验），无 tool/MCP/plugin 机制。
- **判定**：①✅（≥100）②❌（2025-12 后无提交）③⚠️（模板改动量大，反而增加维护面）→ **不建议引入，仅作流水线设计参考**。
- 来源：https://github.com/SakanaAI/AI-Scientist ；https://repos.ecosyste.ms/hosts/GitHub/repositories/SakanaAI%2FAI-Scientist

#### 3.1.2 SakanaAI/AI-Scientist-v2
| 指标 | 值 | 来源 |
|---|---|---|
| star / fork | 7,478 / 1,015 | ecosyste.ms |
| 许可证 | other（2025-12-19 刚更新过 LICENSE，自定义+responsible use 条款） | ecosyste.ms + commits atom |
| 最后提交 | **2025-12-19**（精确，"Update LICENSE" merge） | github.com/SakanaAI/AI-Scientist-v2/commits/main.atom |
| 2026-05~11 提交 | 无（atom feed 显示 2026 年零提交；2025-10 为外部贡献的 Ollama 支持后被 revert） | atom feed |

- **定位**：agentic tree search 的 workshop 级自动科研（idea→实验→写作→评审），比 v1 更通用（不限于 ML 模板，可跑通用代码实验）。
- **覆盖**：①③④⑤（同 v1，④的自由度更高）；无 ②⑥⑦。
- **形态**：Python 流水线；LLM 依赖强（OpenAI/o3、Claude、DeepSeek 等；社区贡献 Ollama 支持）。
- **扩展性**：无 tool/MCP 机制；实验通过"通用 Python 脚本"执行，理论可包一层 COMSOL/Blender 调用，但需自研。
- **判定**：②❌ → **不建议作为依赖引入；其"实验即通用代码"的设计思想值得在 PySci 技能文档中借鉴**。
- 来源：https://github.com/SakanaAI/AI-Scientist-v2 ；https://github.com/SakanaAI/AI-Scientist-v2/commits/main.atom

#### 3.1.3 SamuelSchmidgall/AgentLaboratory
| 指标 | 值 | 来源 |
|---|---|---|
| star / fork | 5,931 / 806 | ecosyste.ms |
| 许可证 | MIT | ecosyste.ms |
| 最后提交 | **2025-08-20**（精确，README 更新） | commits/main.atom |
| 2026-05~11 提交 | 无 | atom feed |

- **定位**："end-to-end autonomous research workflow"——文献综述→实验规划→代码实验（mlesolver）→论文写作（papersolver）→加入 **AgentRxiv**（agent 预印本库，供 agent 协作共享成果）。是与 AI-Scientist 齐名的全流程项目。
- **覆盖**：①③④⑤；无 ②⑥⑦。
- **形态**：Python 脚本 + YAML 配置；LLM 依赖（OpenAI o3-mini/Gemini 等）。
- **扩展性**：代码执行环节是通用 Python（"improvements to code execution"），无 plugin/MCP 机制。
- **判定**：②❌（15 个月无提交）→ **不引入；AgentRxiv 无独立活跃仓库（在本仓库内，"added agentrxiv support" 提交可证）**。
- 来源：https://github.com/SamuelSchmidgall/AgentLaboratory ；https://github.com/SamuelSchmidgall/AgentLaboratory/commits/main.atom

#### 3.1.4 HKUDS/AI-Researcher（NeurIPS 2025）
| 指标 | 值 | 来源 |
|---|---|---|
| star / fork | 5,884 / 729 | ecosyste.ms |
| 许可证 | **None（无 LICENSE 文件，法律上不可自由使用）** | ecosyste.ms |
| 最后 push | 约 2025-11（"about 1 year ago"） | ecosyste.ms |

- **定位**："Autonomous Scientific Innovation"——idea→文献→代码实现→实验验证→写作的自主研究系统，另有商业版 novix.science。
- **覆盖**：①③④⑤（偏 ML/CV 研究）；无 ②⑥⑦。
- **判定**：①✅ ②❌ ③❌（无许可证）→ **排除**。
- 来源：https://github.com/HKUDS/AI-Researcher

#### 3.1.5 bytedance/deer-flow ★重点
| 指标 | 值 | 来源 |
|---|---|---|
| star / fork | **85,344 / 12,026** | ecosyste.ms |
| 许可证 | MIT | ecosyste.ms |
| 最后 push | 2 天前（≈2026-11-15） | ecosyste.ms |
| 发版 | v2.0.0（"5 months ago"≈2026-06，里程碑 182 PR）、v2.1.0（"about 2 months ago"≈2026-09，772 PR/178 贡献者）、v2.2.0（最近，1068 PR） | repos.ecosyste.ms/.../releases |
| 2026-05~11 逐月 | **每月有发版/大量提交**（5 月有 v2.0-m1-rc 系列 tag，6 月 v2.0.0，9 月 v2.1.0，10-11 月 v2.2.0 + push 2 天前） | ecosyste.ms tags + releases |

- **定位**：2025-05 以"开源 Deep Research"（langmanus 系）出道，2.0（2026）**推倒重写为 long-horizon SuperAgent harness**："researches, codes, and creates"——深度调研+代码沙箱执行+报告/播客/视频等内容创作，多子代理、持久记忆、定时任务、IM 渠道（Slack/Telegram/飞书/钉钉/微信/QQ）。是最接近"通用科研工作台"的活跃开源项目。
- **覆盖**：①✓（deep-research+知识库检索 LightRAG/RAGFlow/WeKnora）；③(~)（沙箱 Python 绘图）；④(~)（沙箱代码执行）；⑤✓（结构化研究报告、文档、播客/视频脚本）；⑦(~)（MiniMax image/video 生成技能、音乐生成技能——**托管 API，非本地扩散**）；②⑥✗（但可作为工具挂入）。
- **形态**：自托管平台（Gateway+Web+沙箱 Docker/K8s+Postgres/Redis），Python(LangGraph) + TS 前端；LLM 多供应商（含本地 Ollama/vLLM）。**重量级**：部署面大（Helm/compose/多实例）。
- **扩展性（最强）**：
  - MCP（client/server、OAuth、连接池、Capability Center 统一管理）；
  - **全栈插件平台**（浏览器页面+会话动作+模型工具统一命名空间、versioned manifest、插件可调用宿主模型）；
  - **skills 系统**（SKILL.md 式技能，SkillScan 安全扫描、沙箱内投影）；
  - Custom Agents（可导出/导入为可移植包）、子代理批处理、工具授权分层（Layer-1）、消息网关。
  - → **理论上可把 COMSOL LiveLink for MATLAB/Java、Blender bpy 脚本封装为 tool/MCP 挂入**，实现②⑥的自动化；⑦可挂扩散模型推理服务。
- **判定**：①✅ ②✅ ③✅ → **值得深入评估**（作为"编排底座"候选）。风险：平台复杂度与 PySci"薄胶水"哲学相悖；与 DSH 工作区职能重叠；宜评估"只用其 skills/MCP 设计"还是"整机部署"。
- 来源：https://github.com/bytedance/deer-flow ；https://repos.ecosyste.ms/hosts/GitHub/repositories/bytedance%2Fdeer-flow ；https://repos.ecosyste.ms/hosts/GitHub/repositories/bytedance%2Fdeer-flow/releases
- 中文社区：钛媒体《57k Star霸榜GitHub！字节这款超级智能体》（https://m.tmtpost.com/7939926.html ）、腾讯云开发者《字节跳动DeerFlow 2.0：7万Star的超级智能体底座》（https://cloud.tencent.com.cn/developer/article/2699825 ）。

#### 3.1.6 Google "AI co-scientist"
- 基于 Gemini 的多智能体科研假设生成系统（2025-02 发布）；**无开源仓库**（仅论文与博客），中文社区有讨论（知乎"全自动科研或许在不久的将来会成为现实…"）。不满足开源引入条件，仅作背景。

### 3.2 深度调研 / 报告生成类（①→⑤ 主轴）

#### 3.2.1 assafelovic/gpt-researcher ★重点
| 指标 | 值 | 来源 |
|---|---|---|
| star / fork | **30,690 / 4,222** | ecosyste.ms |
| 许可证 | **Apache-2.0** | ecosyste.ms |
| 最后 push | 4 天前（≈2026-11-13）；commits 页显示**近 21 天 100+ 提交**（共 2 页） | ecosyste.ms + commits.ecosyste.ms |
| 发版 | v3.7.0（"about 2 months ago"≈2026-09）、v3.5.0（"6 months ago"≈2026-05）、v3.4.x（2026-01~04 逐月）→ **2026-05~11 逐月有发版/提交** | repos.ecosyste.ms/.../releases |
| Python | **3.12+**（v3.7.0 起；兼容 PySci 的 3.13） | release notes |

- **定位**：自主深度研究 agent——任意主题/任意数据源→多轮检索、抓取、推理→带引用的长报告。功能面：deep research、multi-agent（LangGraph/deepagents）、本地文档 RAG、报告对话、**paper UI（学术论文式报告，v3.6.1 "New paper UI"）**、多格式导出（依赖 python-pptx 等，含 PDF/DOCX/PPTX 路径）。
- **覆盖**：①✓（**学术检索源**：arXiv、PubMed/PMC、Semantic Scholar、OpenAlex、scholarly 等 20+ 检索器）；⑤✓（报告/论文/pptx 导出）；③✗（不做科研绘图）；⑦(~)（**v3.4.0 起内置 AI 配图**：用 Gemini `gemini-2.5-flash-image`（nano banana）在报告中生成插图，`IMAGE_GENERATION_*` 配置；是 API 生图非本地扩散）；②④⑥✗。
- **形态**：Python 库（pip 包 gpt-researcher）+ 自托管 Web（FastAPI+Next.js，Docker）+ **MCP server**（独立仓库 assafelovic/gptr-mcp，378★）+ `.claude/skills/`（v3.4.0 内置 1500+ 行 SKILL.md，可被 Claude Code 直接作为技能使用）。LLM 依赖：任意供应商（OpenAI/Anthropic/本地均可）。
- **扩展性（强）**：
  - **MCP**：作为 client 挂 MCP 工具（MCPRetriever、MCP 连接头配置、max_tools 管控），作为 server 对外提供深研能力（gptr-mcp）；
  - **检索插件**：v3.7.0 起 `gpt_researcher.retrievers` entry point——第三方检索源可独立发包挂入（"community retriever plugins"）；
  - **agent skill 形态**：`.claude/skills/` 目录（含 SKILL.md/REFERENCE.md、8-step feature pattern），与 PySci 技能文档路线直接同构；
  - 自定义报告提示词族、LangSmith/Langfuse 可观测性、质量评估（quality_eval）。
- **判定**：①✅ ②✅ ③✅ → **最值得深入评估**（作为 ①文献调研+⑤报告写作 的社区底座；②⑥⑦ 需在 PySci 侧以 MCP/插件补 COMSOL、Blender、扩散模型工具）。
- 来源：https://github.com/assafelovic/gpt-researcher ；https://repos.ecosyste.ms/hosts/GitHub/repositories/assafelovic%2Fgpt-researcher ；https://repos.ecosyste.ms/hosts/GitHub/repositories/assafelovic%2Fgpt-researcher/releases ；https://commits.ecosyste.ms/hosts/GitHub/repositories/assafelovic%2Fgpt-researcher/commits ；https://github.com/assafelovic/gptr-mcp

#### 3.2.2 stanford-oval/storm
| 指标 | 值 | 来源 |
|---|---|---|
| star / fork | 31,879 / 3,011 | ecosyste.ms |
| 许可证 | MIT | ecosyste.ms |
| 最后提交 | **2025-09-30**（精确；atom feed `<updated>`） | github.com/stanford-oval/storm/commits/main.atom |
| 2026-05~11 提交 | 无（feed 内 20 条提交止于 2025-09；此前为 2025-05、2025-01） | atom feed |

- **定位**：LLM 知识策展系统——调研主题→多视角提问→生成带引用的维基式长报告（Co-STORM 人机协作模式）。①✓⑤✓（报告），无 ②③④⑥⑦。
- **形态**：Python 库 `knowledge-storm`（PyPI 有 v1.1.0）+ 自托管服务；LLM 依赖（litellm 集成）。
- **扩展性**：可自定义 RM/检索模块与 LM 接口；无 MCP/plugin 机制。
- **判定**：②❌（13 个月无提交）→ **不建议引入**；其"多视角提纲+引用核实"的报告方法论值得在技能文档中借鉴。
- 来源：https://github.com/stanford-oval/storm ；https://github.com/stanford-oval/storm/commits/main.atom

#### 3.2.3 LearningCircuit/local-deep-research ★重点
| 指标 | 值 | 来源 |
|---|---|---|
| star / fork | 9,567 / 916（open issues 1,455） | ecosyste.ms |
| 许可证 | MIT | ecosyste.ms |
| 最后 push | 4 天前；**近 8 天 100 条提交**（commits 页），PR 编号已至 #8700+ | ecosyste.ms + commits.ecosyste.ms |
| 发版 | v1.10.x 系列持续 → **v2.0.0（≈1 周前）→ v2.0.1（2 天前准备）** | tags + commits |
| 2026-05~11 逐月 | **每月大量提交**（近 8 天即 100 条，无月度空档迹象） | commits.ecosyste.ms |

- **定位**：本地/自托管深研助手——支持本地 LLM（llama.cpp/Ollama）与云端，10+ 检索引擎（arXiv、PubMed、私有文档、Federal Register 等），RAG 私有库、Zotero 同步、报告生成（PDF/ODT/RIS 导出）、加密与隐私。口号 "~95% on SimpleQA"。①✓⑤✓。
- **形态**：自托管 Web 应单（Docker Compose 一键）；LLM 依赖任意（本地优先）。
- **扩展性**：自定义检索引擎/下载器、MCP（依赖树含 mcp-runtime，"mcp-runtime-minor-and-patch" 组）、Elasticsearch/SearXNG 可插拔；作为**应用**扩展需改其代码（贡献型扩展）。
- **判定**：①✅ ②✅ ③✅ → **值得深入评估**（若 PySci 需要"一键自托管文献调研站"选它；若要库级嵌入选 paper-qa）。注意：issue 量极大（1,455 open）、近月开发呈 AI 辅助高频提交特征，需评估质量与 API 稳定性。
- 来源：https://github.com/LearningCircuit/local-deep-research ；https://commits.ecosyste.ms/hosts/GitHub/repositories/LearningCircuit%2Flocal-deep-research/commits

#### 3.2.4 huggingface/smolagents（含 open_deep_research 例程）
| 指标 | 值 | 来源 |
|---|---|---|
| star / fork | 30,305 / 3,208 | ecosyste.ms |
| 许可证 | Apache-2.0 | ecosyste.ms |
| 最后 push | 5 天前；tags 至 v1.26.0 | ecosyste.ms |

- **定位**："barebones library for agents that think in code"（代码即动作的 agent 框架）；`examples/open_deep_research` 是其深研例程（非独立产品）。框架级覆盖 ①(~)⑤(~)。
- **扩展性**：任意 Python 工具即工具、MCP、模型无关（本地/云端）。可作为**薄胶水底座**自建 PySci 编排（用它挂 COMSOL/Blender/扩散模型脚本），但等于自研编排层。
- **判定**：①✅ ②✅（框架级持续维护）③⚠️（作为框架可减少自研，但科研工作流仍需自建）→ **列为"框架备选"，与 DeerFlow/gpt-researcher 的成品形态对比后再定**。
- 来源：https://github.com/huggingface/smolagents ；https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research

#### 3.2.5 langchain-ai/open_deep_research
- 12,678★，MIT，Python（workflow 与 multi-agent 两种实现）。**awesome 清单标注 status-archived（不再维护）**；ecosyste.ms 显示最后 push 约 2026-08（"3 months ago"）。仅 18 个提交（ecosyste.ms commit stats）。→ **排除**（LangChain 官方转向其商业产品 LangSmith/Agent Starter Pack）。来源：https://github.com/langchain-ai/open_deep_research ；awesome 清单。

#### 3.2.6 nickscamara/open-deep-research
- 6,298★，other（非 OSI 许可证），TypeScript/Next.js，基于 Firecrawl 的"Deep Research 克隆"。最后 push >1 年（≈2025-11 前）。awesome 清单亦标 Inactive。→ **排除**。来源：https://github.com/nickscamara/open-deep-research

#### 3.2.7 Alibaba-NLP/DeepResearch（Tongyi DeepResearch）
- 20,139★，Apache-2.0，Python；30B-A3B MoE 开源模型+agentic 训练范式，MCP 工具调用，BrowseComp/GAIA/HLE SOTA（清单数据）。最后 push 约 2026-02（"9 months ago"）→ **近 6 个月无提交，不满足判定②**。价值在"开源深研模型权重"而非平台。来源：https://github.com/Alibaba-NLP/DeepResearch

#### 3.2.8 其他 ≥1000★ 深研项目（awesome-deep-research 清单数据，未独立核实）
| 项目 | star(清单) | 备注 |
|---|---|---|
| bytedance/deer-flow | 83.2k+ | 已在 §3.1.5 独立核实 |
| 666ghj/BettaFish | 42.3k+ | 中文舆情分析，非科研工作流 |
| ItzCrazyKns/Vane(Perplexica) | 36.9k+ | 搜索/问答引擎（①的检索层备选） |
| dzhng/deep-research | 19.7k+ | TS 迭代深研，MIT，报告生成 |
| google-gemini/gemini-fullstack-langgraph-quickstart | 18.3k+ | 全栈深研样板 |
| Alibaba-NLP/DeepResearch | 20.0k+ | 已核实 |
| MiroMindAI/MiroThinker + MiroFlow | 8.4k+ / 3.1k+ | 开源 ODR 全栈（模型+框架+数据） |
| zilliztech/deep-searcher | 8.3k+ | 私有数据深搜 SDK |
| langchain-ai/local-deep-researcher | 9.4k+ | 本地深研（LangChain） |
| SkyworkAI/DeepResearchAgent | 3.5k+ | 分层多 agent，含文献检索+浏览器自动化 |
| u14app/deep-research | 4.7k+ | PWA/SaaS 形态，含知识图谱，MIT |
| ruc-datalab/DeepAnalyze | 4.7k+ | 自主数据科学→报告 |
| TIGER-AI-Lab/OpenResearcher | 1.3k+ | 深研轨迹合成+30B 模型（2026-02 开源） |
| IAAR-Shanghai/SurveyX | 990+ | **综述论文自动生成（TeX 输出）**——⑤的学术写作专用工具，可关注 |
| mshumer/OpenDeepResearcher | 2.8k+ | 已 inactive |
| btahir/open-deep-research | 2.1k+ | 已 archived |
| HKUDS/Auto-Deep-Research | 1.7k+ | 深研个人助理 |
| jina-ai/node-DeepResearch | 5.2k+ | 迭代搜索问答 |
| miurla/morphic | 9.1k+ | 生成式搜索 UI |

来源：https://raw.githubusercontent.com/AaronAust1n/awesome-deep-research/main/README.md （另见 DavidZWZ/Awesome-Deep-Research 清单）

### 3.3 学术专用 agent 类

#### 3.3.1 Future-House/paper-qa（paper-qa2）
| 指标 | 值 | 来源 |
|---|---|---|
| star / fork | 9,583 / 954 | ecosyste.ms |
| 许可证 | Apache-2.0 | ecosyste.ms |
| 最后 push | 7 天前（≈2026-11-10）；最近提交 22 天前 | ecosyste.ms + commits.ecosyste.ms |
| 发版 | CalVer：**v2026.08.12（最新 tag）**、v2026.03.18、v2026.03.12、v2026.03.03、v2026.02.27、v2026.02.16、v2026.01.05 … | ecosyste.ms tags |
| 逐月提交（2026-05~11） | **不满足"每月均有"**：commits 页显示 22 天前(≈10-26)←3 个月前(≈8 月)←5 个月前(≈6 月)←6 个月前(≈5 月)——**7 月与 9 月~10 月中为提交空档**；近一年 96 commits/6 位贡献者 | commits.ecosyste.ms |

- **定位**：面向科学文献的高精度 RAG（"High accuracy RAG for answering questions from scientific documents with citations"）——PDF/Office 解析（docling/PyMuPDF/nemotron-parse）、公式/图表理解、引用质量分级（journal quality）、LitQA/PaSa 评测。是①的"文献问答/证据综合"最强库。
- **覆盖**：①✓（深）；⑤(~)（回答/综述片段，非整篇论文写作）；其余✗。
- **形态**：**Python 库（PyPI: paper-qa，含 paper-qa-nemotron 等包）**，Trusted Publishing 发版；LLM 依赖任意（litellm：OpenAI/Anthropic/本地）；无服务端。
- **扩展性**：库级 API/Settings 可扩展（自定义 reader/parser、embedding、检索）；无 MCP/tool 机制（是"被编排的库"而非"编排器"）。
- **判定**：①✅ ②⚠️（月度空档，"持续但不连续"）③✅（库级直接减少①自研）→ **推荐引入**（作为①文献环节的嵌入式库，注意活跃度为"bursty 维护"）。
- 来源：https://github.com/Future-House/paper-qa ；https://commits.ecosyste.ms/hosts/GitHub/repositories/Future-House%2Fpaper-qa/commits ；https://repos.ecosyste.ms/hosts/GitHub/repositories/Future-House%2Fpaper-qa

#### 3.3.2 jordan-gibbs/hyperresearch（agent skill 形态代表）
| 指标 | 值 | 来源 |
|---|---|---|
| star / fork | 4,769 / 494 | ecosyste.ms |
| 许可证 | MIT | ecosyste.ms |
| 创建/最后 push | 创建约 2026-04（"7 months ago"）；push 4 天前；tags 至 v0.13.0 | ecosyste.ms |

- **定位**："Convert Claude Code or Codex into the most intelligent Deep Research Agent"——16 步深研流水线技能：检索→引用核查→持久化可搜索知识库（wiki），另有托管 API + MCP server（hyperresearch.ai）。①✓⑤✓。
- **形态**：**Claude Code/Codex skill + MCP**（技能包形态，与 PySci 路线同构）。
- **判定**：①✅ ②✅ ③✅ → **建议作为"技能组织方式"的参考模板**（其 SKILL.md 结构、引用核查、source vault 设计可直接借鉴到 PySci ①技能文档）。
- 同类：DeepPaperNote（1.1k★，论文深读→Obsidian 笔记 skill）、claude-deep-research-skill（1.2k★，8 阶段流水线 skill）、Deep-Research-skills（2.3k★，中英双语 skill）。来源：awesome-deep-research 清单。

#### 3.3.3 lamm-mit/SciAgentsDiscovery（SciAgents）
- 638★，Apache-2.0，Python；本体图+多 agent 的材料科学假设生成（MIT Buehler 组）。最后 push >1 年（"over 1 year ago"）。①(~)④(~)；无②（不做 FEM）⑥⑦。判定②❌→排除。来源：https://github.com/lamm-mit/SciAgentsDiscovery

#### 3.3.4 multimodal-art-projection/AutoKaggle
- 319★，Apache-2.0；Kaggle 数据科学多 agent（reader/planner/coder/reviewer）。最后 push "almost 2 years ago"（≈2024-12）。④(~)数据科学。判定②❌→排除。来源：https://github.com/multimodal-art-projection/AutoKaggle

#### 3.3.5 scicode-bench/SciCode
- 246★，Apache-2.0，push≈2026-11-03。**是科研编码基准测试（benchmark）**，不是 agent 平台；对 PySci 的价值是④理论计算能力的评测集。来源：https://github.com/scicode-bench/SciCode

#### 3.3.6 OpenScholar / MDAgents / AgentRxiv（未能核实）
- **OpenScholar**（Princeton，科学文献合成 RAG）：`princeton-nlp/OpenScholar` 与检索到的 `zhangchenhaobest/OpenScholar` 在 ecosyste.ms 均 404；star/活跃度**未能核实**。搜索确认存在官方实现仓库（"official implementation of OpenScholar"）但镜像/转移情况不明。
- **MDAgents**（OSU NLP，医疗决策多 agent）：`osunlp/MDAgents` 在 ecosyste.ms 404，**未能核实**；且其域为医疗决策，与 PySci 7 域错位。
- **AgentRxiv**：无独立活跃仓库；已随 AgentLaboratory 合入（commit "added agentrxiv support"，2025-03）。

### 3.4 中文社区观察（知乎/公众号/媒体）
1. **对比评测**：《GPT Researcher、Local Deep Research、STORM 与 DeerFlow 对比》（https://menuagentic.com/zh/blogs/gpt-researcher-vs-local-deep-research-vs-storm-vs-deerflow ）——中文社区公认的"深研四件套"，均为主轴①→⑤，无一提仿真/3D/生图。
2. **DeerFlow 热度**：钛媒体、腾讯云开发者等大量中文报道（"7 万 Star 超级智能体底座"），社区视其为"通用 agent 底座"而非科研专用。
3. **国内科研平台**：公众号"小图微课丨从假设到验证，AI 跑通整条闭环——书生·端砚科研发现平台"（上海 AI Lab 系，http://mp.weixin.qq.com/s?__biz=MzA4OTM4MzAzMQ==&mid=2650858411&idx=1&sn=6f2ff006d63cdf469adc60733d50fc46 ）——国内宣称"科研发现闭环"的平台，但为机构平台/网页服务，非可自托管开源，开源度未能核实。另有"光谷科研专属'小龙虾'"（微信一键调用千款科研工具的聚合入口，https://www.ctdsb.net/c1734_202603/2697074.html ）——工具聚合而非工作流引擎。
4. **AI Scientist 讨论**：知乎"全自动科研或许在不久的将来会成为现实…"等讨论多为展望/担忧，无新工程实现。
5. 结论：中文社区同样**没有出现覆盖覆盖②⑥⑦的"一站式科研"开源项目**；讨论重心在深研报告（Deep Research）与通用 agent 底座（DeerFlow）。

---

## 4. 核心问题回答

**"有没有项目在把科研工作流打包成一站式？"**
- 有"打包"意图并公开可用的：**AI-Scientist-v2、AgentLaboratory、AI-Researcher**（idea→实验→论文闭环），但**全部停止维护**（2025-08～2025-12 后无提交），且实验域偏 ML、无仿真/3D/生图。
- 活跃项目全部只覆盖**①文献/网络调研 → ⑤报告/论文写作**（+④轻量代码执行）：gpt-researcher、local-deep-research、paper-qa、Tongyi DeepResearch、DeerFlow、STORM 等。
- **最接近 PySci 能力组合的**：
  1. **活跃度上**：DeerFlow（①③~④~⑤⑦~，且 skills/MCP/插件机制足以挂载②⑥⑦的社区工具）——但它本身不含 COMSOL/Blender/扩散模型；
  2. **契合度上**：gpt-researcher（①⑤最深 + MCP/插件/skill 三重扩展 + AI 配图⑦~）；
  3. **闭环思想上**（仅参考不引入）：AI-Scientist-v2。
- **②仿真、⑥3D、⑦本地扩散生图三个小众项，社区"科研智能体"里零覆盖**——这三块仍须按"各能力域单独社区调研"的路线自选工具（COMSOL LiveLink、Blender/bpy、ComfyUI/diffusers），再通过 MCP/tool 挂到上述底座。

## 5. PySci 引入判定汇总

| 项目 | ①star≥100 | ②每月提交(26-05~11) | ③减自研 | 结论 |
|---|---|---|---|---|
| gpt-researcher | ✅ 30,690 | ✅ 逐月发版+近3周100+提交 | ✅ ①⑤直接复用 | **深入评估（首选）** |
| bytedance/deer-flow | ✅ 85,344 | ✅ v2.0/2.1/2.2 逐月 | ✅ 编排底座 | **深入评估（底座候选）** |
| paper-qa | ✅ 9,583 | ⚠️ 有月度空档（7月、9-10月中） | ✅ ①库级 | **深入评估（①库级）** |
| local-deep-research | ✅ 9,567 | ✅ 极活跃 | ✅ ①⑤应用级 | 备选（自托管站点） |
| smolagents | ✅ 30,305 | ✅ | ⚠️ 框架需自建流程 | 框架备选 |
| hyperresearch 等 skill | ✅ | ✅ | ✅ 组织方式参考 | 借鉴 SKILL.md 结构 |
| AI-Scientist(-v2)/AgentLaboratory/AI-Researcher/STORM/Tongyi/nickscamara/langchain-odr/AutoKaggle/SciAgents | ✅/⚠️ | ❌ | — | 不引入 |

## 6. 建议下一步
1. 对 **gpt-researcher** 做 PoC：Python 3.13 + Windows 兼容性、paper UI/pptx 导出质量、MCP 挂自定义工具（COMSOL LiveLink 脚本、bpy 脚本）的可行性；确认其 Apache-2.0 与 PySci 自包含依赖策略兼容。
2. 对 **DeerFlow** 做"轻评估"：只取其 skills/MCP/插件规范作为 PySci ②⑥⑦ 技能挂接设计参考；除非用户想要一个独立的自托管科研工作站，否则不整机引入（违背薄胶水原则的量级）。
3. 对 **paper-qa** 做①环节嵌入评估（PyPI 安装、本地模型路径、与 Zotero/私有 PDF 库对接）；对 local-deep-research 视用户是否需要"一键自托管调研站"决定。
4. ②⑥⑦ 三个小众项继续按各自能力域单独做社区调研（COMSOL 自动化、Blender/bpy、ComfyUI/diffusers），本调研确认它们不会被任何"一站式科研平台"顺带解决。

## 7. 来源 URL 汇总
- ecosyste.ms 仓库页（star/许可证/pushed_at）：
  - https://repos.ecosyste.ms/hosts/GitHub/repositories/SakanaAI%2FAI-Scientist
  - https://repos.ecosyste.ms/hosts/GitHub/repositories/SakanaAI%2FAI-Scientist-v2
  - https://repos.ecosyste.ms/hosts/GitHub/repositories/assafelovic%2Fgpt-researcher （+ /releases）
  - https://repos.ecosyste.ms/hosts/GitHub/repositories/stanford-oval%2Fstorm
  - https://repos.ecosyste.ms/hosts/GitHub/repositories/SamuelSchmidgall%2FAgentLaboratory
  - https://repos.ecosyste.ms/hosts/GitHub/repositories/Future-House%2Fpaper-qa
  - https://repos.ecosyste.ms/hosts/GitHub/repositories/LearningCircuit%2Flocal-deep-research
  - https://repos.ecosyste.ms/hosts/GitHub/repositories/nickscamara%2Fopen-deep-research
  - https://repos.ecosyste.ms/hosts/GitHub/repositories/langchain-ai%2Fopen_deep_research
  - https://repos.ecosyste.ms/hosts/GitHub/repositories/huggingface%2Fsmolagents
  - https://repos.ecosyste.ms/hosts/GitHub/repositories/HKUDS%2FAI-Researcher
  - https://repos.ecosyste.ms/hosts/GitHub/repositories/scicode-bench%2FSciCode
  - https://repos.ecosyste.ms/hosts/GitHub/repositories/multimodal-art-projection%2FAutoKaggle
  - https://repos.ecosyste.ms/hosts/GitHub/repositories/lamm-mit%2FSciAgentsDiscovery
  - https://repos.ecosyste.ms/hosts/GitHub/repositories/bytedance%2Fdeer-flow （+ /releases）
  - https://repos.ecosyste.ms/hosts/GitHub/repositories/Alibaba-NLP%2FDeepResearch
  - https://repos.ecosyste.ms/hosts/GitHub/repositories/jordan-gibbs%2Fhyperresearch
- commits.ecosyste.ms（逐条提交日期）：
  - https://commits.ecosyste.ms/hosts/GitHub/repositories/assafelovic%2Fgpt-researcher/commits
  - https://commits.ecosyste.ms/hosts/GitHub/repositories/LearningCircuit%2Flocal-deep-research/commits
  - https://commits.ecosyste.ms/hosts/GitHub/repositories/Future-House%2Fpaper-qa/commits （+ /commits 汇总页"过去一年 96 commits"）
- GitHub atom（精确最后提交日期）：
  - https://github.com/SakanaAI/AI-Scientist-v2/commits/main.atom （2025-12-19）
  - https://github.com/SamuelSchmidgall/AgentLaboratory/commits/main.atom （2025-08-20）
  - https://github.com/stanford-oval/storm/commits/main.atom （2025-09-30）
- 清单/社区：
  - https://raw.githubusercontent.com/AaronAust1n/awesome-deep-research/main/README.md
  - https://github.com/DavidZWZ/Awesome-Deep-Research
  - https://menuagentic.com/zh/blogs/gpt-researcher-vs-local-deep-research-vs-storm-vs-deerflow
  - https://m.tmtpost.com/7939926.html
  - https://cloud.tencent.com.cn/developer/article/2699825
  - https://zhuanlan.zhihu.com/p/2078193534117720082 （AI Deep Research 工具排行榜 Top10）
  - http://mp.weixin.qq.com/s?__biz=MzA4OTM4MzAzMQ==&mid=2650858411&idx=1&sn=6f2ff006d63cdf469adc60733d50fc46 （书生·端砚）
  - https://www.ctdsb.net/c1734_202603/2697074.html （"小龙虾"工具聚合）
- 其他仓库页：https://github.com/assafelovic/gptr-mcp ；https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research

## 8. 未能核实项（透明度）
1. OpenScholar、MDAgents、独立 AgentRxiv 的 star/活跃度（ecosyste.ms 404，GitHub HTML 截断无法读 star）。
2. awesome-deep-research 清单中各项目 star 数（清单自报快照，如 DeerFlow 83.2k 与实测 85,344 存在快照差）。
3. langchain-ai/open_deep_research 的"已归档"状态仅由清单佐证，GitHub 归档横幅未能直接核实。
4. gpt-researcher、local-deep-research 的"每月均有提交"以发版日期+高密度提交页近似核实（近端逐日可见，2026-05~07 靠 release 时间点佐证），未做 GitHub commit search 逐月计数（api.github.com 不可达）。
