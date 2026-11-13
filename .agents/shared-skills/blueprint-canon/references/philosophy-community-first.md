> **所属**：PySci 蓝图
> **上位**：[SKILL.md](../SKILL.md)
> **效力**：与实现、旧计划、旧技能文档冲突时，一律以蓝图为准。
> **基准**：2026-11-13；DSH 0.2.1-rc.5；Python 3.13.15。

# 社区优先与薄胶水（详版）

## 2. 社区优先与薄胶水原则

本条关于新功能开发中对待第三方依赖的策略。

### 2.1 社区引入策略

本项目对第三方依赖**非常欢迎**。自研代码力求只做“薄胶水层”，非必要不自研，更不保留两套实现。对于无分发需求的个人项目，体积不是负担，**维护面**才是。

对于本项目的**各个抽象层面的各种需求**，优先调查社区基础设施（包括但不限于 agent skill、DSH 插件、Python 包、科研工具/MCP、命令行工具等），并积极引入。“积极”在这里详细定义为，**同时满足以下三个条件**时可直接引入，无需用户审批（但仍建议通知用户）：

1. 社区仓库（GitHub/GitLab 等）star 100 以上；
2. 近 6 个月中，每个月均有提交（项目历史短于 6 个月的，从项目创建始记）；
3. 引入可减少预期自研/自维护代码量（即便实际只使用了该项目的一小部分功能）。

上述 3 个条件中，前两条不全满足的，则与用户讨论后决定是否引入。对于单个需求有多个社区仓库均满足上述需求的，按 star 数排列可选方案，与用户讨论后决定选型。存在优质社区项目，但其功能实现与原有需求有出入的，与用户讨论是否应迁就社区实现，修改原有功能预期。

**技能重建的附加规则**：7 个能力域默认不复制旧实现，每个重走三段式，社区调研逐轮交用户确认；允许一个能力域由多个社区依赖组合替代。

---

## 全生态扩展点地图

**这张表回答什么**：一个新需求可能落在哪些生态与形态上，以及每个形态的接入成本。它不回答具体实现——那属于源码与原地文档。

**生态总览**：A. 通用 Agent / B. DSH / C. 科学 Python / D. 科研工具与 MCP / E. Node。

**执行顺序**：定位生态（A 通用 agent / B DSH / C 科学 Python / D 科研工具与 MCP / E Node）→ 查该形态社区现成物 → 按 2.1 判定（三条全满足直接引入；否则与用户讨论；多候选按 star 排序）→ 无现成物才自研，并在原地注释记录理由。成本图例：`配置` < `文件` < `脚本` < `原生` < `引擎`；只表达维护面。

### A. 通用 Agent 生态

**范围**：核心内容在 DSH、Claude Code、Codex、Kilo Code、Open Code 等 agent 之间可移植；更换宿主只需改发现路径或客户端配置。
**推荐发现方式**：MCP 查官方 Registry（若可用）、Smithery、mcp.so、Glama、`awesome-mcp-servers`；skill 查 `anthropics/skills`、GitHub topic `agent-skills` / `claude-skills` 与各 agent 技能市场；再用源仓库、`SKILL.md`、许可证与隔离试用核验。目录负责广度，源仓库负责真实性。

| 扩展点 | 载体 / 入口 | 接入方式 | 成本 | 现状（基准日） |
|---|---|---|---|---|
| skill pack | `SKILL.md` + 资源目录；`.agents/*-skills`、`$DSH_HOME/skills` | 放目录；或作为 skill provider / bundle 分发 | 文件 | 在用：`blueprint-canon`、`pysci-infra`；7 个科研技能待重建 |
| MCP server / gateway | 外部进程；stdio / streamable-http；tools + resources | 在客户端配置 server | 配置 + 外部进程 | 未挂；按技能选型评估 |

### B. DSH 生态

**范围**：依赖 DSH 的 profile/Loader、服务、UI 或市场格式；离开 DSH 即无效。
**推荐发现方式**：官方 GitHub topic `dsh-plugin`、Discussions/Discord 与 npm scope 定位；用 dshbase、dsharness、dsh-market 扩广度；回源核验 `dsh.bundle` / `dsh.client`、兼容范围、近 6 个月提交与 license，在一次性 profile 试装。market index 是安装/消费标准，不是发现入口。
**社区渠道**：dshbase（非官方）、DSH 官方 GitHub topic `dsh-plugin`、npm、DSH Discussions/Discord。

| 扩展点 | 载体 / 入口 | 接入方式 | 成本 | 现状（基准日） |
|---|---|---|---|---|
| plugin | Cordis 模块（`name`/`inject`/`apply`） | 装包或本地挂载 | 脚本 | 无自研；内置若干 |
| bundle / profile bundle | `dsh.bundle.patch` + `dependencies`；含官方 experimental bundles | `dsh plugin add` 或 Web Plugins 页 | 配置（纯组合）/ 脚本（带胶水） | 未装第三方 |
| profile / patch layer（config-only） | `$DSH_HOME/profiles/<name>/cordis.patch.yml`（本项目 `DSH_HOME` 在 `sub_root/deps_management/deps/node/modules/dsh_home`） | 覆盖、禁用、插入条目 | 配置 | 在用：模型、provider、spill、三 preset |
| preset pack | `dsh-agent-preset` 的 `config.id` + `plugins` | profile 声明；会话/子代理选择 | 配置 | 自研 `pysci-worker` / `pysci-leader` / `pysci-meta` |
| provider pack（model / web / subagent） | `dsh-llm-*`、`dsh-web-*`、`dsh-subagent-*` | 替换或新增 provider 行 | 配置 | deepseek-account + deepseek-official + xiaomi(pi-ai)；spawn/fork |

> 能力类别（orchestrator / agent team、memory / context、security / policy、office / docs、notifications）通常作为上述形态的内部实现，不单列为“形态”。
> 内置能力 `schedule`、`goal`、`workflow` 不是扩展接缝，但可直接用于编排；不占本表维护面。

### C. 科学 Python 生态

**范围**：项目科研能力可直接引入的 PyPI 包与工具；运行在项目 `.venv`（Python 3.13.15）。
**推荐发现方式**：PyPI 分类器、[awesome-python](https://github.com/vinta/awesome-python)、GitHub topic `python-library` / `scientific-computing`；用 libraries.io / pypistats 看使用度；核对 `requires-python`、Windows wheel、许可证与最近发布。
**社区渠道**：PyPI、GitHub、awesome-python、libraries.io、各领域 awesome 列表。

| 扩展点 | 载体 / 入口 | 接入方式 | 成本 | 现状（基准日） |
|---|---|---|---|---|
| Python packages | `pyproject.toml` + uv | 加依赖 + `uv sync`；技能重建时逐项添加 | 配置 + 脚本 | 最小集；7 个技能依赖待选型 |
| CLI tools | 包内 console script 或 `uv tool` | 装项目内 uv tool / venv | 配置 | 按技能选型 |

> 旧 pySciWS 的 7 个技能后端是**参考实现**，不是默认基线；选型时可对照能力清单，但不得直接复制。

### D. 科研工具与 MCP

**范围**：无法用纯 Python 包覆盖的科研软件与外部服务：COMSOL、Blender、Zotero、文献检索/下载、LaTeX/LibreOffice、云 PDF 抽取、AI 图像生成等。
**推荐发现方式**：先查对应软件的官方 API/CLI 与社区 MCP server（GitHub topic `mcp-server`、MCP Registry、Smithery）；核对与本地版本兼容性、许可证、GUI/驱动/许可证需求；项目内隔离试装。
**社区渠道**：官方文档/API、MCP 注册表、GitHub topic `mcp-server`、各软件社区。

| 扩展点 | 载体 / 入口 | 接入方式 | 成本 | 现状（基准日） |
|---|---|---|---|---|
| COMSOL | `mph` Python 库 / COMSOL Server / 社区 MCP | 项目内 Python 包 + 外部 license | 配置 + 外部软件 | 未接；技能重建时评估 |
| Blender | headless CLI / blender-mcp | 外部 Blender + 项目内 MCP/脚本 | 配置 + 外部软件 | 未接；3D 技能重建时评估 |
| Zotero / 文献检索 | zotero-mcp、paper-search-mcp、OpenAlex/arXiv/WoS API | 项目内 uv tool 或直接 HTTP | 配置 + API key | 未接；文献技能重建时评估 |
| LaTeX / Office | TeX Live、LibreOffice、Pandoc | 系统 PATH 外部工具 + Python 薄胶水 | 配置 + 外部软件 | 未接；写作技能重建时评估 |

### E. Node 生态

Node 只用于承载 DSH（`@deepseek-ai/dsh` 及其依赖）；科研能力不引入 Node。发现渠道：npm registry、GitHub topic `nodejs`；先确认是否真的需要 Node，再看活跃度与许可证。
