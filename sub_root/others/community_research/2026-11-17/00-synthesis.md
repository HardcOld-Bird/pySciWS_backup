# 2026-11-17 社区调研汇总：科研能力组合有无"一站式"社区方案

> 状态：待用户审核。本汇总由工作 Agent 汇合 4 路子代理调研 + 主会话亲手核验写成。
> 明细来源（本目录）：`one-stop-ai-research-platforms.md`（A）、`general-agent-frameworks.md`（B）、
> `science-engineering-suites.md`（C）、`per-domain-flagships.md`（D）。
> 判定口径（蓝图 2.1）：① star ≥ 100；② 近 6 个月（2026-05 ~ 2026-11）每月均有提交；③ 引入可减少自研代码量。

## 0. 调研问题与结论

**问题**：对 PySci 的能力组合（文献调研、COMSOL 仿真、科研绘图、理论计算、办公文档写作、3D 建模、AI 生图），
社区有没有有影响力的一站式解决方案，让我们不必逐域自组？

**结论：没有。三条独立证据线一致**——

1. **A 线（科研智能体平台）**：社区"一站式科研"实为 ①文献调研→⑤报告/论文写作主轴。宣称全流程闭环的
   （AI-Scientist/-v2、AgentLaboratory、STORM、AI-Researcher）全部已停止维护；②COMSOL、⑥3D、⑦本地生图
   在所有科研智能体中**零覆盖**（个别"AI 配图"均为托管 API）。
2. **B 线（通用 Agent 框架/工作流平台）**：n8n/Dify/LangGraph 等只是宿主/胶水基座；科研模板集中在
   ①+⑦插件，②③④⑥在 10 个框架的模板/市场中**全空**。接入框架后科研能力仍要自写。
3. **C 线（科学计算/工程套件）**：开源界无套件吃掉 ②仿真+⑥3D+③绘图+⑤文档；真正"套件级一站式"
   只有商业软件（COMSOL/Ansys 本身就是）。

**但格局清晰，"社区组合"路线完全可行**：7 个能力域全部存在过口径的社区主力，且存在一个公共增效层——
**MCP 生态**（Blender MCP 35k★、arXiv MCP 3.3k★、COMSOL-MCP 977★、zotero-mcp 5.9k★），
让多数域的自研面收敛到"配置 + 薄胶水"。社区形态还出现与 PySci 同构的"agent skill"路线
（如 hyperresearch 4.8k★），可作各域技能文档的形态参考。

**对蓝图路线的验证**：不照搬旧 pySciWS、逐域三段式、社区优先组合——此路线被调研证实是唯一现实解，
且比预想更乐观（每域都有 ≥1 个高星活跃候选）。

## 1. 覆盖矩阵（能力域 × 社区现状）

| 域 | 一站式？ | 社区主力候选（★/判定） | MCP/接口层 | 真实缺口 |
|---|---|---|---|---|
| ① 文献调研 | 智能体覆盖最深 | marker 41k★✅、zotero-mcp 5.9k★✅、GROBID 5.2k★✅；paper-qa 9.6k★⚠️bursty | arXiv MCP 3.3k★✅、zotero-mcp | 检索/下载源 API 选型待逐域定 |
| ② COMSOL 仿真 | 无（仅商业） | mph 611★⚠️发版驱动（唯一通道）、COMSOL-MCP 977★⚠️临界 | COMSOL-MCP（基于 mph，100+ 工具） | 官方无 Python API；二者均差口径② |
| ③ 科研绘图 | 无 | matplotlib、plotly 18.9k★✅；pyqtgraph 备选 | — | seaborn/plotnine 差口径② |
| ④ 理论计算 | 无 | SymPy 15.1k★✅、SciPy✅、JAX✅；SageMath 2.6k★✅(非pip,GPL) | — | 无 |
| ⑤ 文档写作 | 无 | pandoc 46.9k★✅（CLI 枢纽）、Typst 57.2k★✅(wheel 直装) | LibreOffice MCP≈0★ | python-docx/pptx 停更⚠️；Office/Zotero 桥计自研预算 |
| ⑥ 3D 建模 | 无（FreeCAD 近似） | build123d 3.7k★✅、CadQuery 6.1k★✅、FreeCAD 35.4k★✅、trimesh/gmsh✅ | Blender MCP 35k★✅ | gmsh star 贴线(135★) |
| ⑦ AI 生图 | 智能体零覆盖 | diffusers 34.9k★✅、ComfyUI 147.2k★✅ | ComfyUI 自带 HTTP API | InvokeAI 不支持 Py3.13；A1111/Fooocus 停更 |

（✅=三口径全过；⚠️=差口径②等，需用户裁决；star 为 2026-11-17 采样）

## 2. 头部"准一站式"候选的正确定位

| 项目 | ★ | 定位建议 |
|---|---|---|
| gpt-researcher（Apache-2.0，活跃） | 31k | ①⑤最深的社区主轴；MCP+检索插件+skills 三重扩展与 PySci 同构。候选：整机引入吃掉①⑤，或仅借鉴其技能形态 |
| bytedance/DeerFlow（MIT，当天有提交） | 85k | 唯一活跃的通用科研 harness；平台级体量与薄胶水原则冲突。候选：仅作 ②⑥⑦ 工具挂接设计参考，不整机引入 |
| AI-Scientist / STORM / AgentLaboratory | 15k/32k/6k | 全部停更，不引入；其"全流程闭环"路线暂不成熟 |

## 3. 数据核验记录

- 主会话亲手核验（shields.io，2026-11-17）：COMSOL-MCP 977★/oct ✅、MPh 611★/aug ✅、
  deer-flow 85k/today ✅、gpt-researcher 31k/nov ✅、blender-mcp 35k/上周 ✅、arxiv-mcp 3.3k/上周 ✅、
  zotero-mcp 5.9k/上周 ✅、marker 41k/上周 ✅——与子代理报告全部一致。
- B/D 两会话内 web_fetch 受限，数据经 ecosyste.ms 镜像 + Node fetch + GitHub HTML 逐月探测交叉取得；
  本汇总已对其中头部数据点二次实测复核。
- 已解决的数据分歧：B 报"Zotero MCP 162★ 停更"查的是同名衍生仓；主仓为 `54yyyu/zotero-mcp`
  （5,916★，MIT，2026-11 仍活跃），D 路为准。
- 未能核实项：openpyxl 主仓 Heptapod 被反爬拦截；gmsh star（官网分发为主）；LangChain 逐月提交。

## 4. "差口径②"集中清单（发版驱动/成熟稳定型）

mph、COMSOL-MCP（11 月暂缺，17 日观察中）、paper-qa、seaborn、plotnine、python-docx、python-pptx、
OpenSCAD、A1111、Fooocus。此类项目功能成熟、发版驱动或维护转弱，但仍在广泛使用。
**需用户定通用处理原则**（见待裁决问题 2）。

## 5. 用户裁决记录（2026-11-17）

1. **总方向**：认可"无一站式、走分域组合"结论；商业 SaaS/云端方案不入自包含工作区，不再补充调研。
2. **口径②豁免原则**：**按无可替代性分级豁免**——先定"无可替代"判定，无可替代者（如 mph）豁免自动
   生效（pin 版本 + 原地 ADR 记录维护风险）；有替代余地者（如 python-docx，可由 pandoc/Typst 替代）
   不豁免、仍拒绝。
   - **2026-11-17 修正**（用户确认）：DSH 预装 `@deepseek-ai/dsh-skill-office` 七技能的创作层锚定
     python-docx / python-pptx / openpyxl，在该生态位上实际无可替代（拒绝 = 预装技能空转 + 自写
     替代技能，自研面反升）。故 **python-docx/pptx 豁免引入（pin 版本）**，⑤域直接复用 DSH 七技能。
3. **gpt-researcher 定位**：**借鉴形态，不整机引入**——仅参考其技能/插件设计，①⑤域自选零件组合，
   不背整个上游维护面。
4. **启动顺序**：**文本链先行**——①文献→⑤写作，之后再推 ③绘图→④计算→②仿真→⑥3D→⑦生图。
5. **⑤域路线（含细化裁决）**：DSH office 七技能 + python-docx/pptx/openpyxl（Office 原生格式创作）
   ｜pandoc（格式互转/引用）｜**LaTeX 单轨**（latexmk+日志解析+chktex 薄胶水重建；**Typst 不引入**）；
   渲染/公式重算/导 PDF 用 DSH 自带 LibreOffice Kit。细化裁决：旧域特有重资产（学院 deck 三件套、
   deck_digest、提取乱码门）**全部暂缓**、日后再立项；pymupdf4llm（AGPL-3.0）**接受**（个人科研
   不分发）；⑤域候选**拒绝清单认可**。
6. **①域裁决（2026-11-17）**：paper-search-mcp **豁免引入**（pin 版本 + 原地 ADR 记风险，唯一聚合
   通道）；paper-qa **豁免 pin 为可选件**（不进主链，RAG 主链由 zotero-mcp 语义检索承担）；PDF→MD
   **只收 MinerU 且一律走云端 API**（本机硬件不足以本地提取；MinerU 云 API key 在旧项目 `.env`，
   实施时迁入项目内配置；marker 不收）；WoS/Scopus 连接器**按需启用**（WoS Starter API key 同在旧
   项目 `.env`）。旧特有能力（中英对照生产线、SciMago 自建索引、WoS 自研客户端、缓存/doctor 自检）
   **不保留**；付费墙浏览器抓取降级为 agent web 工具（零自研）。
   ⑦生图 ComfyUI+diffusers 组合待 ⑦域选型轮再确认。

## 6. 下一步

按裁决启动 **①文献域"新需求三段式·选型"**：对照旧 pySciWS 文献能力清单（仅对照能力，不复制代码）→
细化候选（检索/获取/PDF 解析/文献库/问答/综述）→ 按三口径 + 无可替代性豁免判定 → 交用户确认候选
与取舍后，在 `sub_root\others\plans` 立计划实施。⑤写作域随后同流程。本目录调研材料即各域选型输入。

## 7. 全局环境约束（2026-11-17 用户口述）
本机**仅 CPU、无独显/显存、16GB 内存（日常可用约 5GB）**。本地模型推理等重量级计算需谨慎考虑：MinerU 已裁决走云端 API；中英对照的 SaT 句切用小模型 CPU 推理、argostranslate 仅按需；**⑦AI 生图域选型须云 API 优先**（.env 已有 ARK_API_KEY/SILICONFLOW_API_KEY），本地扩散仅轻量兜底。
