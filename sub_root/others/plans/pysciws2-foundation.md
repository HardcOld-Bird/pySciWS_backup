# pySciWS2 奠基实施计划（待审核）

> **状态**：v2.0（2026-11-13，用户已批准；含技能重建政策增补）。开始实施。
> **执行人**：当前会话（项目专属 DSH 正常运转前的“助产士”）。
> **范围**：pySciWS2 的顶层设计、项目专属 DSH、三 preset、蓝图、依赖管理、分层测试与框架迁移（不含研究内容迁移）。
> **非目标**：不在本阶段搬运 `src/pysci/research/`、`data/research/`、`scripts/research/`、`tests/research/` 中的研究内容；不立即替换远端仓库。

---

## 0. 结论先行

1. **pySciWS2 = pySci 的框架 + 项目专属 DSH + 三 preset + 蓝图 + 两层项目支撑系统**。研究内容按用户决定“稍后手动搬”，本阶段只保证框架可运转、可验证、可交接。
2. **worker 是默认（主力）模式**：吸收旧 pySciWS 全部组员技能，直接执行科研生产任务；使用 `AGENTS.md` 与 `.agents/skills`。
3. **leader 是可选模式**：继承旧“组长中心制”，只做规划、派发、验收与对用户沟通；不加载生产技能，使用 `AGENTS.leader.md` 与 `.agents/leader-skills`；子代理只能是 worker。
4. **meta 只改 harness**：继承 DSH 内置 cordis/创意模式，只对用户与 eval 结果负责；使用 `AGENTS.meta.md` 与 `.agents/meta-skills`；子代理可使用全部三个 preset。
5. **`orch` 体系整体废弃**：`pysci-orch`、pods、taskbooks、watchdog、夜航、`.qoder`、Qoder hooks 全部移除；其能力分别由 DSH `subagent`、leader 技能、meta/evals 承接。
6. **依赖全项目内**：node/npm/uv/Python/git/DSH 全部在 `sub_root/deps_management/` 内；项目外只保留 `common\bin` 路由 shim。
7. **测试三层**：`sub_root/tests`（可机械断言）、`sub_root/evals`（可基准测试）、meta（主观层）；不设 `sub_root/qa` 包裹层，比 mySimsWS2 更扁平。
8. **远端替换是最后一步**：在项目 DSH 正常运转、用户明确授权后，用 pySciWS2 的 git 历史强制替换 pySciWS 远端；不保留旧快照。

### 0.1 v2.0 修订（2026-11-13，用户批准后增补）

用户批准本计划，同时增补一条**技能政策**，覆盖本计划中“直接迁移技能”的旧表述：

- **7 个生产技能不直接搬迁**：不复制旧 `src/pysci/skills/*` 后端、旧 `.agents/skills` 文档、旧 `data/skills` 资产、旧技能测试作为新项目实现。
- **每个能力域各自重走“新需求三段式”**：选型（重点重做社区调研，最大化减少自研代码）→ 计划与实施（按选型写薄胶水）→ QA 同步（更新分层测试并跑全量）。
- **允许一换多**：一个旧技能可以由多个社区依赖/工具组合替代；不要求保持旧 CLI 形状，只要求能力覆盖与可验证。
- **用户每轮在位**：每次社区调研的候选、取舍与“自研 vs 社区”结论都必须交用户审核或补充，再进入实施。
- **执行者**：技能重建在项目专属 DSH 正常运转后，由项目 DSH 的 worker/leader 执行；本会话只负责奠基，不预先替项目 DSH 做技能选型。
- **旧仓库的角色**：旧代码、旧技能文档、旧测试仅作参考与能力清单来源；是否复用某段旧代码，由该技能的三段式选型结果决定，默认不复用。
- **奠基范围因此收窄**：P0–P6 只交付“核心框架 + 项目 DSH + 三 preset + 蓝图 + 依赖管理 + 分层测试骨架 + `pysci-infra`”，不交付 7 个生产技能；7 个技能由项目 DSH 逐项三段式重建。
- **交接标准相应调整**：项目 DSH 能正常运转、能读蓝图、能跑测试、能发起三段式技能重建，即满足交接；不以“7 个技能已重建”为前提。

---

## 1. 现状盘点

### 1.1 pySciWS2（当前目录）

| 项 | 现状 | 处理 |
|---|---|---|
| `AGENTS.md` | 3,731 B，逐字复制自 mySimsWS2，标题仍为“MySims 项目宪法” | 整体重写为 PySci 宪法 |
| `AGENTS.leader.md` | 0 B | 新写 |
| `AGENTS.meta.md` | 0 B | 新写 |
| `.agents/skills/blueprint-canon/` | 仅有 2,201 B 的 SKILL.md 骨架，项目概述与设计摘要为空 | 移至共享蓝图目录并重写 |
| `.agents/leader-skills/`、`.agents/meta-skills/` | 空目录 | 新建技能 |
| `sub_root/deps_management/{deps,deps_manager}` | 空目录 | 建依赖管理系统 |
| `sub_root/tests/`、`sub_root/evals/` | 空目录 | 建分层测试系统 |
| `sub_root/others/plans/` | 空目录 | 本计划落点 |
| `src/`、`scripts/`、`data/`、`pyproject.toml` | 不存在 | P4 从 pySciWS 迁移框架 |
| `.git` | 不存在 | P0 初始化 |

### 1.2 mySimsWS2（参考范式）

- 分层蓝图：`AGENTS.md` + `.agents/skills/blueprint-canon/{SKILL.md,references/*}`；哲学详版、社区优先分册、设计摘要分册。
- 三套技能面：`.agents/skills`（work）、`.agents/meta-skills`（meta）；对应 `AGENTS.md` / `AGENTS.meta.md`。
- 自定义 preset：`mysims-work`、`mysims-work-blender`、`mysims-work-godot`、`mysims-meta`；profile patch 在 `dsh_home/profiles/web/cordis.patch.yml`。
- 依赖管理：`sub_root/deps_management/{toolchain.json,deps_manager,deps,cache}`；init/update/status/rollback/log；`common\bin` shim。
- 分层测试：`sub_root/qa/tests`、`sub_root/qa/evals`；黄金任务、A/B、预审表、改进台账。
- 可复用的机制结论：preset 用 `cordis:include` + `patches` 增量组合；子代理白名单 `tool-subagent.allowedAgentPresets`；`includeDefaultRoots: false` + `customSkillDirs` 做技能隔离；preset 改动须在无会话时小步进行。

### 1.3 pySciWS（旧范式，迁移源）

- 项目性质：个人科研工作区；文献调研、理论计算、COMSOL 仿真、科研绘图、论文写作、AI 绘图、3D 建模；人保留真实实验。
- 代码：`src/pysci`（src-layout 单包）+ `scripts/` + `data/`（平行资产树）+ `tests/`；uv + `.venv`；Python 3.13.15。
- 7 个生产技能后端：`literature_research`、`comsol_simulation`、`document_writing`、`scientific_plotting`、`theoretical_computation`、`ai_drawing`、`modeling3d`；统一 CLI `pysci-*`。
- 旧编排：`.qoder/`（rules/hooks/skills）+ `orchestration/`（pods、state、bench、guards、13 个成员）；`pysci-orch` 为组长唯一通道；成员通过 headless pod 派发。
- 技能文档真源：`orchestration/skills/<name>/SKILL.md` + `references/`（质量高，可直接迁移改造）；`pods/` 是各成员副本，迁移时只取 `orchestration/skills/` 真源。
- Git：分支 `main`；远端 `https://github.com/HardcOld-Bird/pySciWS_backup.git`；工作区有 11 项未提交改动（多为 orchestration/pods/settings，随废弃而丢弃）。

### 1.4 当前 DSH 环境

- 本会话运行在全局桌面 profile：`C:\Users\antar\.dsh\profiles\desktop`。
- 项目专属 DSH 将使用项目内 `DSH_HOME = sub_root/deps_management/deps/node/modules/dsh_home`，由项目 `dsh.cmd` shim 注入；与全局 profile 完全隔离。
- `common\bin` shim 层按“当前目录向上找 `sub_root\deps_management\toolchain.json`”路由；mySimsWS2 与 pySciWS2 可共用同一层，前提是相对布局一致、shim 文本保持通用。

---

## 2. 目标形态

### 2.1 顶层结构

```text
pySciWS2/
├── AGENTS.md                     # worker 工作说明（宪法 + 必读规则）
├── AGENTS.leader.md              # leader 工作说明
├── AGENTS.meta.md                # meta 工作说明
├── .agents/
│   ├── shared-skills/            # 新增：worker/leader 共享的只读蓝图
│   │   └── blueprint-canon/{SKILL.md,references/*}
│   ├── skills/                   # worker 技能：pysci-infra + 后续逐项重建的生产技能
│   ├── leader-skills/            # leader 专属技能（无生产技能）
│   └── meta-skills/              # meta 专属技能
├── src/pysci/
│   ├── paths.py                  # 资产与支撑目录的权威索引
│   ├── toolchain.py              # 项目内工具链绝对路径
│   ├── skills/                   # 初始仅 pysci_infra；7 个生产技能由项目 DSH 重建
│   ├── support/evals/            # 新增：eval runner（Python）
│   └── research/<line>/          # 研究线（本阶段不迁）
├── scripts/
│   ├── research/<line>/          # 研究脚本（本阶段不迁）
│   └── ...                       # MCP 安装脚本按技能选型再建
├── data/
│   ├── skills/                   # 初始为空；按技能选型再建资产
│   ├── research/<n>_<line>/      # 研究资产（本阶段不迁）
│   └── 用户的inbox/               # 用户专属配置区
├── sub_root/
│   ├── deps_management/
│   │   ├── toolchain.json        # 版本/URL/SHA256 唯一真源
│   │   ├── deps_manager/         # init/update/status/rollback/log + 子更新器 + shims
│   │   ├── deps/                 # 工具链安装物（git/node/uv/python）
│   │   └── cache/                # 下载与包缓存
│   ├── tests/                    # 第 1 层：可机械断言
│   ├── evals/                    # 第 2 层：可基准测试
│   └── others/{plans,archived}/  # 过程产物
├── pyproject.toml
├── .python-version
├── .gitignore / .gitattributes
└── .venv/                        # uv 项目环境（gitignore）
```

**根目录策略**：根目录只允许版本控制、agent 元数据与 `pyproject.toml` / `.python-version` 这类不可收纳项；`.venv/` 作为 uv 的固定例外留在根目录（项目内，非全局）。其余一切进 `sub_root/`。

### 2.2 三 preset 一览

| 维度 | `pysci-worker`（默认） | `pysci-leader`（可选） | `pysci-meta` |
|---|---|---|---|
| 基础 | DSH 内置 standard | DSH 内置 standard | DSH 内置 cordis（创意模式） |
| 工作说明 | `AGENTS.md` | `AGENTS.leader.md` | `AGENTS.meta.md` |
| 技能根 | `.agents/skills` + `.agents/shared-skills` | `.agents/leader-skills` + `.agents/shared-skills` | Creator 技能 + `.agents/meta-skills` |
| 生产技能 | 交接时仅 `pysci-infra` + shared 蓝图；7 个生产技能由项目 DSH 逐项三段式重建 | **无** | 无 |
| 直接任务 | 是（默认主力） | 否，只规划/派发/验收 | 否，只改 harness |
| 用户沟通 | 可直接沟通 | 用户只与组长沟通 | 只对用户与 eval 结果负责 |
| 子代理 preset | 仅 `pysci-worker` | 仅 `pysci-worker` | 全部三个 |
| fork 子代理 | 继承 worker（允许） | **禁用**（会继承 leader） | 继承 meta（允许） |
| workflow 子代理 | 继承 worker（允许） | **禁用**（会继承 leader） | 继承 meta（允许） |
| 默认会话 | 是（`selectedDefault`） | 否 | 否 |

> leader 保留旧“组长中心制”的唯一例外：用户明确要求组长亲做时，leader 可以执行；默认一律派发。该例外须在 `AGENTS.leader.md` 写明并留痕。

### 2.3 为什么增加 `.agents/shared-skills`

蓝图 `blueprint-canon` 必须同时对 worker 与 leader 可见，但 leader 又不能看到 `.agents/skills` 里的生产技能。DSH 的技能发现是“按根目录扫描直接子目录”，无法只从一个目录里摘一个技能；复制蓝图会造成两份真源。因此：

- 蓝图正典放 `.agents/shared-skills/blueprint-canon/`（人类维护面，唯一真源）。
- worker 的 `customSkillDirs = [.agents/skills, .agents/shared-skills]`。
- leader 的 `customSkillDirs = [.agents/leader-skills, .agents/shared-skills]`。
- meta 不挂共享目录，使用 `.agents/meta-skills/blueprint-canon/`（meta 视角裁剪版，只导航、只读，正文指向共享正典）。

这是对用户提出的“三个 `.agents` 子目录”的最小增补；若不接受，可退化为“leader-skills 里放一个薄指针技能”，但会多一份易漂移的入口。

---

## 3. 蓝图制度

### 3.1 构成

| 文件 | 作用 | 维护者 |
|---|---|---|
| `AGENTS.md` | 设计哲学·简版 + 必读工作规则；worker 常驻注入 | 人类 |
| `AGENTS.leader.md` | leader 身份、权限、派发与验收规程 | 人类 |
| `AGENTS.meta.md` | meta 身份、三不碰、harness 改进与 eval 规程 | 人类 |
| `.agents/shared-skills/blueprint-canon/SKILL.md` | 蓝图正典入口：项目概述、法则、分册目录、协议 | 人类 |
| `references/philosophy-detailed.md` | 设计哲学详版 | 人类 |
| `references/philosophy-community-first.md` | 社区优先与薄胶水；全生态扩展点地图 | 人类 |
| `references/abstract-core.md` | 设计摘要总纲：项目支撑系统 + 科研生产能力 + 研究线 | 人类与 Agent 协商 |
| `references/abstract-harness.md` | 三 preset、会话模型、子代理矩阵、orch 废弃理由 | 人类与 Agent 协商 |
| `references/abstract-support-systems.md` | 依赖管理 + 分层测试（三层） | 人类与 Agent 协商 |
| `references/abstract-workspace.md` | `src/scripts/data/sub_root` 的布局、镜像关系、本地/入库边界 | 人类与 Agent 协商 |

设计摘要**不共用 mySimsWS2 正文**；只在“依赖管理系统”“分层测试系统”两个子节参考其结构，按 PySci 的 Python/科研场景重写。游戏、Godot、资产管线等内容不进入 PySci 蓝图。

### 3.2 哲学微调点（相对 mySimsWS2）

1. **蓝图原则**：不变。
2. **社区优先与薄胶水**：不变；扩展点地图的 C 区从 Godot 换成“科学 Python / 科研工具生态”（PyPI、Zotero/paper-search/COMSOL/Blender MCP、LaTeX/LibreOffice 等外部工具）。
3. **自包含与勤分类**：不变；根目录策略改为 PySci 的 `src/`、`scripts/`、`data/`、`sub_root/` 布局；依赖管理边界与 mySimsWS2 同构。
4. **原地文档**：不变；`paths.py` 仍是资产布局权威索引，蓝图只写设计目标。
5. **极简原则**：不变；Qoder/orch 的废弃路径直接删除，不做兼容层。

### 3.3 预算与机械校验

- 全量注入档（`AGENTS*.md`）：**每文件 ≤ 8 KB**。
- `SKILL.md`：**每文件 ≤ 8 KB**；超出细节进同目录 `references/*.md`。
- 按需档 `references/*.md`：**单文件 ≤ 8 KB**；超出则拆分为分册索引 + 单册。
- 机械校验进 `sub_root/tests/harness/`：
  - 所有 `AGENTS*.md` 字节数；
  - 所有 `SKILL.md` frontmatter 合法（name 为 kebab-case、description 非空）；
  - 蓝图内部链接存在；
  - 技能名无重复；
  - preset 声明的技能根、白名单与蓝图矩阵一致；
  - 上游 standard/cordis 片段与 `--check` 一致。

### 3.4 `AGENTS.md` 与技能的分层

旧 pySci 的 `basic.md` 约 8 KB，若与哲学简版一起塞进 `AGENTS.md` 会超预算。新分工：

- **`AGENTS.md`（≤8 KB）**：哲学简版 + 不超过约 10 条“不读就会做错”的硬规则——路径与资产落盘、CLI 调用、测试入口、网络代理、禁止 Qoder/orch、数据区边界。
- **`.agents/skills/pysci-infra/`**：环境与运维细节——PowerShell 7 陷阱、`uv`/venv 同步、依赖更新、worktree、MCP 安装、常见故障。
- **生产技能**：各自领域知识（绘图、仿真、写作、文献、理论、作画、3D）。
- 原则：常驻注入只放“不读会做错”的内容；能按需读的细节一律下沉到技能 `references/`。

---

## 4. 项目专属 DSH 与 preset 接线

### 4.1 项目内 DSH 安装

- **Node/DSH**：`sub_root/deps_management/deps/node/`
  - `runtime/<version>/`：vendored Node；
  - `modules/package.json` + `package-lock.json`：`@deepseek-ai/dsh` 以精确版本钉住；
  - `modules/node_modules/`：npm 安装物；
  - `modules/dsh_home/`：项目 `DSH_HOME`（sessions、profiles、credentials、cache、spill）。
- **Python**：uv 管理，Python 3.13.15；`.venv` 在项目根（uv 约定，项目内）。
- **git/uv**：vendored 于 `deps/`；shim 路由。
- **shim**：部署到 `D:\XXXIIIGGG\common\bin`（与 mySimsWS2 共用），按 cwd 找 `toolchain.json` 路由；`dsh.cmd` 注入 `DSH_HOME`、`DSH_AGENTS_HOME`、项目 Node 运行时与临时目录。
- **凭据**：`init-deps` 从 `%USERPROFILE%\.dsh\.credentials.yaml` 复制到项目 `dsh_home`（需用户同意；绝不入库）；模型 provider 行从当前桌面 profile 摘取，去掉 mySims 专属项。

### 4.2 profile 与片段

```text
dsh_home/profiles/web/
├── cordis.yml              # []
├── package.json            # bundles: dsh-base, dsh-web-app, experimental-session-search
├── pnpm-workspace.yaml
├── cordis.patch.yml        # 模型/provider/UI/spill + 三个 preset + registry
└── pysci/
    ├── standard.plugins.yml  # 由上游 standard.patch.yml 生成，勿手改
    └── cordis.plugins.yml    # 由上游 cordis.patch.yml 生成，勿手改
```

- `standard.plugins.yml` / `cordis.plugins.yml` 由 `deps_manager/scripts/extract_presets.mjs` 生成；`--check` 模式进测试。
- 上游漂移只进同步脚本；定制只进 include 的 `patches`；两者永不混写。
- `cordis.patch.yml` 是 preset 真源；改动前备份，且只在无运行会话时小步改（mySimsWS2 2026-11-05 事故教训）。

### 4.3 preset 声明草图

```yaml
- insert:
    - id: preset-pysci-worker
      name: "@deepseek-ai/dsh-agent-preset"
      config:
        id: pysci-worker
        name: PySci Worker
        order: 10
        description: PySci 默认工作模式：科研生产 + 项目支撑系统
        plugins:
          - id: inc-standard
            name: cordis:include
            config:
              path: pysci/standard.plugins.yml
              patches:
                - id: agent-instructions
                  name: "@deepseek-ai/dsh-agent-instructions"
                  config:
                    maxBytes: 65536
                    instructionFileCandidates: [AGENTS.md]
                    localInstructionFileCandidates: [AGENTS.local.md]
                - id: persona
                  name: "@deepseek-ai/dsh-persona"
                  config:
                    prefix: |
                      你是 PySci 科研工作区的工作 Agent（worker），默认主力模式。
                      你直接执行科研生产任务，并对产物质量负责。
                    complete: true
                - id: skill-filesystem
                  name: "@deepseek-ai/dsh-skill-filesystem"
                  config:
                    includeDefaultRoots: false
                    customSkillDirs:
                      - .agents/skills
                      - .agents/shared-skills
                - id: tool-subagent
                  name: "@deepseek-ai/dsh-tool-subagent"
                  config:
                    provider: spawn
                    toolName: subagent
                    modelSelectionSettings: true
                    agentPreset: pysci-worker
                    allowedAgentPresets: [pysci-worker]

    - id: preset-pysci-leader
      name: "@deepseek-ai/dsh-agent-preset"
      config:
        id: pysci-leader
        name: PySci Leader
        order: 11
        description: 可选组长模式：只规划、派发与验收，不直接生产
        plugins:
          - id: inc-standard
            name: cordis:include
            config:
              path: pysci/standard.plugins.yml
              patches:
                - id: agent-instructions
                  name: "@deepseek-ai/dsh-agent-instructions"
                  config:
                    maxBytes: 65536
                    instructionFileCandidates: [AGENTS.leader.md]
                    localInstructionFileCandidates: [AGENTS.leader.local.md]
                - id: persona
                  name: "@deepseek-ai/dsh-persona"
                  config:
                    prefix: |
                      你是 PySci 项目的组长（leader），只对用户负责。
                      你不直接生产；你把任务拆解、派发给 worker 子代理，并验收结果。
                    complete: true
                - id: skill-filesystem
                  name: "@deepseek-ai/dsh-skill-filesystem"
                  config:
                    includeDefaultRoots: false
                    customSkillDirs:
                      - .agents/leader-skills
                      - .agents/shared-skills
                - id: tool-subagent
                  name: "@deepseek-ai/dsh-tool-subagent"
                  config:
                    provider: spawn
                    toolName: subagent
                    modelSelectionSettings: true
                    agentPreset: pysci-worker
                    allowedAgentPresets: [pysci-worker]
                - id: tool-subagent-fork
                  name: "@deepseek-ai/dsh-tool-subagent"
                  disabled: true
                - id: tool-workflow
                  name: "@deepseek-ai/dsh-tool-workflow"
                  disabled: true

    - id: preset-pysci-meta
      name: "@deepseek-ai/dsh-agent-preset"
      config:
        id: pysci-meta
        name: PySci Meta
        order: 12
        description: 元 Agent：只改进 worker/leader 的 harness，只对用户与 eval 结果负责
        plugins:
          - id: inc-cordis
            name: cordis:include
            config:
              path: pysci/cordis.plugins.yml
              patches:
                - id: agent-instructions
                  name: "@deepseek-ai/dsh-agent-instructions"
                  config:
                    maxBytes: 65536
                    instructionFileCandidates: [AGENTS.meta.md]
                    localInstructionFileCandidates: [AGENTS.meta.local.md]
                - id: persona
                  name: "@deepseek-ai/dsh-persona"
                  config:
                    prefix: |
                      你是 PySci 项目的元 Agent（harness 改进工程师），职责单一：
                      改进 worker/leader 的 harness（preset、技能、工作说明、规程），
                      只对用户与基准测试结果负责。
                    complete: true
                    includeRuntimeContext: false
                - id: skill-filesystem
                  name: "@deepseek-ai/dsh-skill-filesystem"
                  config:
                    includeDefaultRoots: false
                    customSkillDirs:
                      # 与 mySimsWS2 meta 同形；上游 cordis 片段自带此表达式，patch 时逐字保留
                      - !!js process.getBuiltinModule('node:path').join(process.getBuiltinModule('node:path').dirname(process.getBuiltinModule('node:module').createRequire(new URL('index.js', baseUrl)).resolve('@deepseek-ai/dsh-agent-preset/package.json')), 'skills')
                      - .agents/meta-skills
                - id: tool-subagent
                  name: "@deepseek-ai/dsh-tool-subagent"
                  config:
                    provider: spawn
                    toolName: subagent
                    modelSelectionSettings: true
                    agentPreset: pysci-worker
                    allowedAgentPresets: [pysci-worker, pysci-leader, pysci-meta]

- id: agent-preset-registry
  name: "@deepseek-ai/dsh-agent-preset-registry"
  config:
    default: standard
    selectedDefault: pysci-worker
```

### 4.4 关键机制注记

- **技能隔离**：三个 preset 均 `includeDefaultRoots: false`，只挂各自显式目录；避免用户级/全局技能泄漏，也避免 `.agents/skills` 被 leader 看到。
- **子代理白名单**：`allowedAgentPresets` 是模型可见的枚举；`agentPreset` 是默认值；两者必须同时包含默认 id。
- **fork**：fork 子代理继承父 preset，无法在创建时选 preset。因此 leader 禁用 `tool-subagent-fork`，否则会得到“leader 子代理”，违反“leader 子代理只能是 worker”。该继承行为以 P3 冒烟为准。
- **workflow**：workflow 的 `agent()` 子代理同样继承父 preset，`workflow-ptc` 无 preset 选择项。因此 leader 禁用 `tool-workflow`；leader 用同一回合的多个 `subagent` 调用做并行派发。worker 与 meta 保留（其子代理继承自身 preset，仍在允许范围内）。该继承行为以 P3 冒烟为准。
- **深度**：默认 `maxDepth = 1`。leader 必须把任务拆到叶子再派给 worker；worker 子代理不能再向下派发。这是有意的“扁平化”，避免层级失控。
- **技能目录**：`customSkillDirs` 的相对路径按 `process.cwd()` 解析；项目 `dsh.cmd` 会 `pushd` 到项目根，保证稳定。
- **preset 生效**：新建会话按当前 revision 组合；运行中会话保留旧代际。改 `cordis.patch.yml` 后重启 DSH，不做热改。
- **leader 的“全套工具”例外**：这里指读/写/命令/问询/待办/调度等生产与协作工具；`subagent_fork` 与 `workflow` 是仅有的两个例外，因为它们的子代理无法在创建时选 preset，会继承 leader，直接违反“leader 子代理只能是 worker”。若您坚持保留这两个工具，只能靠 `AGENTS.leader.md` 纪律约束，无法机械保证，不建议。
- **bundle 可用性**：`dsh-experimental-session-search` 等 experimental bundle 是否随当前 DSH 版本提供，以 P1 实测为准；不可用则退化为普通会话检索，不阻塞主线。

---

## 5. 项目支撑系统

### 5.1 依赖管理系统

**目标**：换机 = 复制目录 + 一次 `init-deps`；日常 = 一条 `update-deps`；项目外只有路由 shim。

**目录**：

```text
sub_root/deps_management/
├── toolchain.json
├── deps_manager/
│   ├── init-deps.cmd / init-deps.ps1
│   ├── update-deps.cmd / update-deps.ps1
│   ├── lib/_toolchain.ps1
│   ├── shims/{dsh,git,node,npm,npx,uv,uvx,ruff}.cmd
│   ├── sub_updaters/{git,node,npm,uv,python,dsh}.ps1
│   └── scripts/extract_presets.mjs
├── deps/{git,node,uv,python}/...
└── cache/
```

**依赖分类**（按“是否有可回退版本记录”）：

| 类别 | 成员 | 管理方式 |
|---|---|---|
| managed | DSH（npm）、Python 包（uv） | 锁文件 + git 跟踪；自动更新/回退 |
| vendored | git、node/npm、uv、Python 解释器 | `toolchain.json` 钉版本；init 安装/修复；current 指针 |
| detected | Blender、COMSOL、TeX Live、LibreOffice、NI-DAQmx 驱动 | 只比对版本并提示，不自动装 |
| reference | 社区 MCP 的独立工具环境（zotero-mcp、paper-search-mcp、comsol-mcp） | 项目内 uv tool，单独更新 |

**生命周期**：

- `init-deps`：审计 → 安装 vendored 工具 → 部署 shim → 前置 PATH（管理员一次）→ 创建 `.venv` → `npm install` DSH → 生成 `dsh_home` profile → 复制凭据 → 冒烟。
- `update-deps`：预检（git 仓库、工作区干净、项目工具就绪、DSH 未运行）→ 按顺序调用子更新器 → 汇总 → 尝试 push；提供 `status` / `rollback` / `log`。
- 硬规则：没有 git 不更新；工作区脏不更新；DSH 运行中不更新；非项目内工具不更新。

**从 mySimsWS2 裁剪**：删除 Godot/Android/WSL/Blender/py4godot/gdUnit4/asset_pipeline 子更新器与清单；保留 git/node/npm/uv/python/DSH 框架；`_toolchain.ps1` 去掉 WSL/Godot 分支，保留 shim/PATH/current 指针逻辑。

**与 pySciWS 旧依赖管理的差异**：旧版只有 init/update 两个脚本、无 shim、无 DSH；新版采用 mySimsWS2 的统一更新器骨架，但按 Python-only 裁剪。

### 5.2 分层测试系统

**定位**：“测试层级 = 抽象层级”。三层首尾相接：

```text
主观层（meta）——产出经第 2 层检验
   ↑
第 2 层：sub_root/evals —— 黄金基准裁决 worker/leader 的 harness 改进
   ↑
第 1 层：sub_root/tests —— 机械断言项目代码、技能契约、harness 配置
```

#### 第 1 层：`sub_root/tests`

- **一个入口**：`sub_root/tests/tests.cmd`（薄壳）→ `tests.ps1`：
  - `all`：ruff check + ruff format --check + pytest（默认排除 hardware） + harness 契约测试；
  - `fast`：pytest `-m "not slow and not hardware"`；
  - `hardware`：pytest `-m hardware`（NI/COMSOL，按需）；
  - `cov`：pytest --cov；
  - `harness`：只跑 harness 契约测试。
- **布局**：
  ```text
  sub_root/tests/
  ├── conftest.py
  ├── test_paths.py / test_toolchain.py
  ├── skills/<skill>/...          # 从旧 tests/skills 迁移
  ├── support/...                 # eval runner 单测
  ├── harness/                    # 新增：蓝图/preset/技能契约
  └── research/<line>/...         # 本阶段不迁，后续本地保留
  ```
- **harness 契约测试**（新增，重点）：
  - `AGENTS*.md` 预算；
  - `SKILL.md` frontmatter 与命名；
  - 蓝图链接完整性；
  - preset YAML 解析、id 唯一、include 路径存在；
  - 三 preset 的技能根/工作说明/子代理白名单与蓝图矩阵逐项一致；
  - leader 技能目录不含生产技能名；
  - meta 技能目录不含 worker/leader 技能名；
  - `extract_presets.mjs --check`。
- **确定性**：固定 seed；网络/LLM/COMSOL/Zotero 用 mock/cassette；`hardware`、`network`、`slow` 标记；默认不联网。
- **产物**：`sub_root/tests/artifacts/`（gitignore）。

#### 第 2 层：`sub_root/evals`

- **被测物**：worker/leader 的 harness 版本（preset 组合 + 技能 + 工作说明 + 规程），以 patch 文件标识。
- **黄金任务**：直接照搬真实摩擦（例如“新增一个研究线脚手架”“修复路径迁移后的测试失败”“给某 CLI 加子命令”）；任务包含 `prompt.md`、`task.yaml`（快照、模型路由、预算、判据）、`checks/`（机械判据）。
- **运行器**：`src/pysci/support/evals/`（Python 包）+ `pysci-eval` CLI：
  - 优先用项目 `dsh --profile headless "<任务>"` 跑一次全新持久会话（DSH 官方入口，读最终答案 + 会话日志）；需要细粒度事件/token 归因时回退到 DeepSeekHarness SDK。
  - 复制隔离工作区 → 私有 `DSH_HOME` → 经项目 `dsh.cmd` 跑任务；
  - 记录 events、token、墙钟、轮次；
  - 机械判据在隔离工作区执行；行为类问题用行为/结果契约；最终由人类终审。
- **A/B**：同一任务、同一项目快照、同一模型路由，对比 base / delta harness；先跑聚焦任务，默认再跑全部有效黄金任务做不劣化回归。
- **产物**：`sub_root/evals/{tasks,runs,reports,overlays}`；
  - `reports/improvement-ledger.md`（改进台账）；
  - 每次对比产出预审表（初判 + 证据引用）交人类复核。
- **MVP**：1–2 个黄金任务 + 单臂/双臂跑通即可；不追求一开始就全覆盖。

#### 第 3 层：meta（主观层）

- 不设目录；产物沉淀在 `sub_root/evals` 的台账与报告。
- 由 `pysci-meta` preset + `.agents/meta-skills` 执行；冷启动，不携带既往会话上下文。
- **三不碰**：不改蓝图（人类维护面）；不改 `sub_root/tests`、`sub_root/evals` 的判定逻辑与项目代码（归 worker）；不碰研究内容。发现漏测/误拦记入台账，由用户转交 worker。
- 只对用户与 eval 结果负责；每次改进走“问题来源 → 改动 → 基准结果 → 是否实装”的台账。

---

## 6. 框架迁移、去 Qoder 化与技能重建

### 6.1 能力域与重建归属

| 旧成员/能力 | 新形态 | 旧资产处理 |
|---|---|---|
| sim | worker 技能（能力域：有限元仿真），名字待选型后定 | 旧后端/文档仅作能力参考，不迁 |
| figure | worker 技能（科研绘图） | 同上 |
| writing | worker 技能（文档写作 + 期刊写作） | 同上 |
| theory | worker 技能（理论计算） | 同上 |
| lit | worker 技能（文献研究） | 同上 |
| drawing | worker 技能（AI 绘图） | 同上 |
| model3d | worker 技能（3D 建模） | 同上 |
| devops | `pysci-infra`（本次奠基交付）+ meta 技能 `harness-dev` | 旧 devops 的依赖/测试经验作参考，不迁代码 |
| reviewer | leader 技能 `review-acceptance` + worker 自检清单 | 不单设 reviewer preset |
| bencher | meta 技能 `eval-runner` + `sub_root/evals` | 新写 |
| deputy | 并入 leader 的决策支持 | 删除独立角色 |
| twin | 删除 | 夜航/taste 机制不再需要 |
| inspector | 删除 | 旧项目已停用 |
| orchestration（组长通道） | leader preset + `large-task-orchestration` 技能 | `pysci-orch` 删除 |

> 7 个生产技能是**能力域**，不是要照搬的包名；每个能力域在项目 DSH 接手后各自重走三段式，选型结果可能是一个社区工具、多个社区依赖的组合，或少量薄胶水。

### 6.2 本次奠基的代码与目录动作

**进入新项目（核心框架，裁剪/重写后）**：

- `src/pysci/{__init__.py,paths.py,toolchain.py}`
- `pyproject.toml`：只保留构建/工具配置与最小依赖（ruff/pytest/pre-commit/commitizen 等）；生产依赖随技能选型逐项添加
- `.python-version`、`.env`（本地复制，不入库）
- `tests/{conftest.py,test_paths.py,test_toolchain.py}` → `sub_root/tests/`
- `.pre-commit-config.yaml`（移除 `harness-budget` 本地钩子）
- `data/用户的inbox/pysci_settings.toml`（本地复制，不入库）

**不进入新项目（旧仓库保留作参考）**：

- `src/pysci/skills/*`（7 个生产后端）
- `orchestration/skills/*`（旧技能文档）
- `data/skills/*`（技能资产；如某技能选型后确有复用价值，再按需取单个文件）
- `scripts/*`（MCP 安装脚本、cold_call、edit_progress）
- `tests/skills/*`、`tests/research/*`
- `.qoder/`、`orchestration/` 全目录
- 旧 `pyproject.toml` 的生产依赖清单

**新项目初始化**：

- `.agents/skills/pysci-infra/`（依赖/测试/evals 运维）
- `.agents/leader-skills/{large-task-orchestration,review-acceptance}/`
- `.agents/meta-skills/{blueprint-canon,harness-dev,eval-runner,repo-recon}/`
- `src/pysci/skills/__init__.py`（空命名空间，等技能重建）
- `src/pysci/support/evals/`
- `sub_root/{deps_management,tests,evals}/`
- `sub_root/others/plans/`

**关于“删除”**：旧仓库原地保留，本次不删；新项目从不复制 `.qoder`/`orchestration`，因此新仓库里没有它们。只有在按技能选型复用某段旧代码时，才需要处理该代码的旧 import（见附录 A）。

### 6.3 技能重建三段式（项目 DSH 接手后执行）

对 7 个能力域逐个执行，不得跳步：

1. **选型**：盘点旧技能能力（只读参考）→ 按 `philosophy-community-first` 的全生态扩展点地图做社区调研（PyPI、GitHub、MCP registry、awesome lists；核对 star / 近 6 个月活跃度 / 许可证 / Python 3.13 / Windows 可用性）→ 提出候选组合（允许一换多）→ 与用户讨论确认。每轮社区调研用户在位。
2. **计划与实施**：在 `sub_root/others/plans/` 写该技能计划；定义 CLI/接口、数据契约、与 `paths.py` 的落盘约定；只写社区工具表达不了的薄胶水；完成后归档计划。
3. **QA 同步**：在 `sub_root/tests/skills/<skill>/` 补机械断言；跑 `tests.cmd all`；需要时更新 `abstract-*` 设计摘要（只改设计目标，不写实现细节）。

规则：

- 默认不保留旧实现；若某段旧代码确有社区无法替代的价值，须在选型记录中写明“为什么社区不可行”，再以薄胶水形式保留。
- 一个技能可以由多个社区依赖组合；不为“单包”而迁就功能缺口。
- 技能对外 CLI 名可改；不要求兼容旧 `pysci-*` 入口；确需兼容时显式标注“已弃用”（极简原则）。
- 技能文档只写领域决策与调用面，细节进 `references/`；每个 `SKILL.md` ≤ 8 KB。

---

## 7. 分阶段实施

> 阶段可顺序执行；P2 与 P4 可在 P1 完成后部分并行。每阶段结束必须有可复核的验收证据。

### P0 · 方案冻结与安全网

- [x] 用户审核并批准本计划（2026-11-13，含 `.agents/shared-skills` 增补、TUI/orch 删除、`.env` 本地复制）；同日增补技能重建政策（见 §0.1）。
- [ ] 记录旧仓库 HEAD、远端、未提交改动清单；在项目外生成 `git bundle` 备份（只作安全网，不作为最终快照保留；P7 完成后删除或移出项目）。
- [ ] `pySciWS2` 初始化 `.git`（`main`），提交当前骨架（含本计划）。
- [ ] 写入 `.gitignore` / `.gitattributes` 基线（deps/cache/dsh_home/.venv/凭据/研究树/产物）。
- **注意**：旧 pySciWS 的 `.gitignore` 用 `/sub_root/` 整目录忽略；新项目必须改为**精细规则**——忽略 `sub_root/deps_management/{deps,cache}`、`dsh_home`、`tests/artifacts`、`evals/runs` 等，但**跟踪** `toolchain.json`、`deps_manager/`、`tests/`、`evals/tasks`、`others/plans/`。否则依赖管理器与分层测试代码进不了仓库。
- **验收**：`git status` 干净；备份可 `git bundle verify`；旧仓库未被修改。

### P1 · 项目内工具链与 DSH 启动

- [ ] `toolchain.json`：git/node/uv/Python 3.13.15/DSH 精确版本；detected 项清单。
- [ ] `deps_manager` 骨架：`_toolchain.ps1`、`init-deps`、`update-deps`、`status/rollback/log`、`sub_updaters`、`shims`。
- [ ] `npm install --save-exact @deepseek-ai/dsh@<version>` 到 `deps/node/modules`；提交 `package.json` + `package-lock.json`。
- [ ] 创建 `dsh_home/profiles/web`：`cordis.yml`、`package.json`、`pnpm-workspace.yaml`、`cordis.patch.yml`；从当前桌面 profile 摘取模型/provider/UI 配置。
  - 优先让 `dsh web` 首次启动**自动初始化** `profiles/web`（DSH 官方模板），再在其 `cordis.patch.yml` 上叠加；手写 profile 只作备选，避免 bundle 版本漂移。
- [ ] 按需创建 `dsh_home/profiles/headless`（同 preset、无 UI），供 eval 与自动化冒烟使用；`dsh --profile headless "<任务>"` 能读最终答案。
- [ ] 生成 `pysci/standard.plugins.yml` 与 `pysci/cordis.plugins.yml`；`extract_presets.mjs --check` 可跑。
- [ ] 复制凭据到项目 `dsh_home`（本地、gitignore）。
- [ ] 部署 shim；`dsh --dsh-router-info` 指向项目；启动 Web GUI。
- **验收**：从 pySciWS2 根目录运行 `dsh web` 打开项目专属 Web GUI；`DSH_HOME` 为项目内；模型可对话；`dsh --profile headless "只回答 OK"` 能退出并打印结果；从 mySimsWS2 根目录运行 `dsh` 仍路由到 mySimsWS2（共享 shim 不串台）。

### P2 · 蓝图 v1

- [ ] 重写 `AGENTS.md`（PySci 宪法 + worker 必读规则）。
- [ ] 写 `AGENTS.leader.md`、`AGENTS.meta.md`。
- [ ] 建 `.agents/shared-skills/blueprint-canon/`：SKILL.md + `philosophy-detailed.md` + `philosophy-community-first.md` + `abstract-core.md` + `abstract-harness.md` + `abstract-support-systems.md` + `abstract-workspace.md`。
- [ ] 建 `.agents/meta-skills/blueprint-canon/SKILL.md`（meta 视角，只读导航）。
- [ ] 所有文件过 8 KB 预算与链接检查。
- **验收**：用户复核内容；无 MySims 游戏内容残留；meta 视角不复制正文；链接全通。

### P3 · 三 preset 接线

- [ ] 在 `cordis.patch.yml` 写入 `pysci-worker`、`pysci-leader`、`pysci-meta` 与 registry。
- [ ] 建三个技能目录与占位技能（先验证可见性）。
- [ ] 子代理白名单、fork/workflow 禁用、persona、agent-instructions、skill roots 按 §4 落位。
- [ ] 先用 `dsh --profile web --dump-config` 检查组合树与 preset 行，再启动会话。
- [ ] 冒烟矩阵（新建会话）：
  - worker：见 `AGENTS.md`、`.agents/skills` + shared；技能目录含 `pysci-infra`（7 个生产技能待项目 DSH 重建）；subagent 枚举只有 worker；fork/workflow 可用。
  - leader：见 `AGENTS.leader.md`、leader-skills + shared；**不见**生产技能；subagent 枚举只有 worker；fork/workflow 不可用。
  - meta：见 `AGENTS.meta.md`、meta-skills + Creator 技能；不见 worker/leader 技能；subagent 枚举三个全有。
  - registry：新建普通会话默认落 worker。
- **验收**：上述矩阵逐项通过；preset 切换不导致工具表坍缩；改动前后有备份。

### P4 · 核心框架迁移（技能不迁）

- [ ] 建最小 `src/pysci` 包：`__init__.py`、`paths.py`、`toolchain.py`；`pyproject.toml` 最小依赖；`.python-version`；`uv sync`。
- [ ] 迁 `tests/{conftest.py,test_paths.py,test_toolchain.py}` → `sub_root/tests/`；改 `testpaths`、conftest、markers。
- [ ] 建 `.agents/skills/pysci-infra/`（依赖/测试/evals 运维文档）。
- [ ] `paths.py` 增 `TESTS_ROOT`/`EVALS_ROOT`；`toolchain.py` 核对 `DEPS_ROOT` 与 `common\bin` 解析。
- [ ] `.pre-commit-config.yaml` 移除 `harness-budget` 本地钩子；预算/技能契约改到 `sub_root/tests/harness/`。
- [ ] 全仓确认无 `.qoder`/`orchestration`/`pysci-orch` 引用（新项目本来就没有）。
- **验收**：`uv sync` 成功；`pytest` 全绿（hardware 默认排除）；`pysci-infra` 技能可见；新仓库不含 7 个生产技能后端。

### P5 · 分层测试与 eval 骨架

- [ ] 第 1 层：`tests.cmd` 入口、`sub_root/tests/harness/` 契约测试（蓝图预算、preset、技能目录、上游片段 `--check`）。
- [ ] 第 2 层：`src/pysci/support/evals/` + `pysci-eval` CLI；`sub_root/evals/{tasks,runs,reports,overlays}` 骨架；1 个 smoke 黄金任务（例如“按三段式新增一个技能选型记录”）。
- [ ] 第 3 层：`AGENTS.meta.md`、meta 技能（`harness-dev`、`eval-runner`、`repo-recon`）；改进台账与预审表模板。
- [ ] 跑一次单任务 eval（base vs 无操作 delta）验证管线。
- **验收**：`tests.cmd harness` 全绿；eval 能完成一次隔离运行并产出报告；meta 能读取报告与台账。

### P6 · 交接

- [ ] 启动项目专属 DSH；按 §8 交接矩阵逐项验收。
- [ ] 在计划文件中记录交接状态与遗留项（原地文档，不另建交接文档）。
- [ ] 当前会话停止项目工作；后续由项目 DSH 接手。
- **验收**：项目 DSH 的 worker 会话能读蓝图、跑测试、见 `pysci-infra`；leader/meta 冒烟通过；用户确认。

### P7 · 远端仓库替换（单独授权，最后执行）

- [ ] 预检：旧仓库工作区干净、备份可验证、远端分支/标签清单确认、用户明确下达“替换”指令。
- [ ] 在 pySciWS2 添加 `origin` = `https://github.com/HardcOld-Bird/pySciWS_backup.git`。
- [ ] `git push --force origin main`；删除远端旧分支/标签，只保留新 `main`。
- [ ] 更新 `pyproject.toml` 的 Homepage/Repository/Issues URL；验证 `git ls-remote` 与干净 clone。
- [ ] 旧本地目录在用户确认后归档/删除（不保留旧快照）；**研究内容未迁完前不得删除旧目录**。
- **验收**：远端只有新历史；clone 可用；旧提交不可达；用户确认。

---

## 8. 项目专属 DSH 交接验收矩阵

| # | 场景 | 期望 |
|---|---|---|
| 1 | 从 pySciWS2 根目录运行 `dsh web` | 打开项目 Web GUI；`DSH_HOME` 在项目内；模型可对话 |
| 2 | 新建普通会话 | 默认落 `pysci-worker` |
| 3 | worker 会话 | 见 `AGENTS.md`；技能目录含 `pysci-infra` + shared 蓝图（7 个生产技能待项目 DSH 重建）；subagent 枚举只有 `pysci-worker` |
| 4 | worker 派发子代理并指定 `pysci-leader` | 被拒绝（不在枚举） |
| 5 | leader 会话 | 见 `AGENTS.leader.md`；技能只有 leader-skills + shared；不见生产技能；subagent 枚举只有 `pysci-worker`；`subagent_fork`/`workflow` 不可用 |
| 6 | leader 派发 worker 做一个小任务 | 成功；worker 能执行基建任务（如跑测试）；结果回 leader；用户只与 leader 对话 |
| 7 | meta 会话 | 见 `AGENTS.meta.md`；技能含 meta-skills + Creator 技能；不见 worker/leader 技能；`tool-cordis` 可用；subagent 枚举三个全有 |
| 8 | meta 派发 worker/leader/meta | 三者均可用 |
| 9 | 从 mySimsWS2 根目录运行 `dsh web` | 仍路由到 mySimsWS2 的 `DSH_HOME`（共享 shim 不串台） |
| 10 | `extract_presets.mjs --check` | PASS（片段与上游一致） |
| 11 | `sub_root\tests\tests.cmd harness` | PASS（蓝图/preset/技能契约） |
| 12 | `uv run --no-sync pytest` | 全绿（hardware 默认排除） |

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| preset 热改导致工具表坍缩 | 会话工具面丢失，需回滚 | 只在无运行会话时改；每步备份；小步 + 重启 + 冒烟；保留 `cordis.patch.yml.bak-*` |
| 共享 `common\bin` shim 与 mySimsWS2 串台 | 打开错误项目的 DSH | shim 保持通用、按 cwd 找 `toolchain.json`；两个项目都做路由冒烟；相对布局保持一致 |
| 上游 DSH 升级导致 preset 片段漂移 | preset 挂载失败或工具面变化 | 版本钉死；`extract_presets.mjs --check` 进测试；升级走单独 update 步骤 |
| 技能名冲突/非法 frontmatter | 技能被静默跳过 | harness 契约测试：kebab-case、唯一性、description 非空 |
| leader 通过 fork/workflow 得到 leader 子代理 | 违反“子代理只能 worker” | leader 禁用 `tool-subagent-fork` 与 `tool-workflow`；冒烟验证 |
| `includeDefaultRoots: false` 隐藏用户级技能 | 某些全局技能不可用 | 有意隔离；需要时显式加入对应 preset 的 `customSkillDirs` |
| 凭据复制/模型配置错误 | DSH 无法对话 | init 冒烟；凭据只在本地；提供 `dsh --dsh-router-info` 与模型自检 |
| 框架迁移遗漏 import | 核心包/测试断裂 | 只迁核心文件 + `ruff` + `pytest` + 全仓 grep；技能代码不迁，避免旧 import 面 |
| 研究内容未迁导致新 DSH“空转” | 交接后无法立刻做研究 | 用户已确认“稍后手动搬”；交接标准是框架可运转，研究内容由用户搬入 |
| 强制推送不可逆 | 旧历史丢失 | P7 单独授权；项目外 bundle 安全网；预检远端；用户明确确认 |

---

## 10. 已确认的决策（2026-11-13）

1. **`.agents/shared-skills` 增补**：接受（理由见 §2.3）。
2. **裸 `pysci` TUI**：删除；用户交互统一走 DSH Web GUI；不保留旧 7 个生产 CLI，技能重建后按选型结果定义新入口。
3. **`reviewer`**：不单设 preset，拆为 leader 的 `review-acceptance` 技能 + worker 自检清单。
4. **`.env`**：从旧项目复制到新项目（本地、gitignore）。
5. **评估 runner 归属**：`src/pysci/support/evals/`（同一 venv、可单测），数据在 `sub_root/evals/`。
6. **P7 时机**：项目 DSH 稳定运行至少一个完整任务周期后再执行；不早于 P6。
7. **技能政策（v2.0 增补）**：7 个生产技能不直接搬迁；每个能力域各自重走三段式，重点重做社区调研，允许一换多；每轮社区调研用户在位；由项目 DSH 接手执行。

---

## 11. 本阶段不做什么

- 不迁移 `src/pysci/research/`、`data/research/`、`scripts/research/`、`tests/research/`。
- 不删除 pySciWS 旧仓库/旧目录。
- 不修改 pySciWS 的远端。
- 不实现全部 eval 任务库；只做 MVP 骨架。
- 不在奠基阶段引入 MCP；MCP 作为各技能选型的候选之一，按三段式评估。
- 不迁移/不重建 7 个生产技能；由项目 DSH 逐项三段式重建。本阶段只交付 `pysci-infra` 与框架。

---

## 12. 审核后第一步

本计划已获批准（2026-11-13）。从 **P0** 开始：记录旧仓库状态、生成项目外 bundle、初始化 pySciWS2 git、写 `.gitignore`/`.gitattributes`，然后进入 P1 工具链与 DSH 启动。P1 的第一次 `dsh web` 启动即是“项目专属 DSH 正常运转”的起点；后续按阶段推进，直到 P6 交接。技能重建由项目 DSH 接手后逐项执行（§6.3）。

---

## 附录 A：旧仓库参考审计（默认不迁移）

> 7 个生产技能默认全部重建，不复制旧代码/资产/测试。以下审计仅在某个技能选型后决定“复用某段旧代码”时参考；不要把它当作迁移清单。

### A.1 体量与范围

| 项 | 审计结果 | 处理（默认） |
|---|---|---|
| `data/skills` git 跟踪面 | 60 文件 / 1.96 MB | 不迁；若某技能选型后需要某个模板，再单文件取用 |
| `tests` git 跟踪面 | 344 文件 | 不迁；新技能重建后另写测试 |
| `src/pysci/skills` git 跟踪面 | 173 文件 | 不迁；旧仓库保留作能力参考 |
| `tests/skills` 生产测试 | ai_drawing 12、comsol 18、document 22、literature 34、plotting 20、theory 1 | 不迁；新技能重建后另写测试 |
| `modeling3d` 测试 | 无独立 `tests/skills/modeling3d/` | 新技能重建时补 smoke；旧 `tests/research/smoke_model3d` 属研究内容，暂不迁 |

### A.2 已知 import 触点（仅当复用旧代码时处理）

> 默认不迁移任何旧技能代码；只有某技能选型决定复用旧实现时，才按本表处理其触点。

| 文件:行 | 现状 | 处理 |
|---|---|---|
| `src/pysci/skills/literature_research/tools/research.py:1173` | `from pysci.skills.orchestration.tools import procs` | 把 `procs.py` 迁到 `src/pysci/support/procs.py`，改 import |
| `scripts/cold_call.py:16` | `from pysci.skills.devops.tools.coldcall import main` | 删除脚本；如仍需要冷启动直调，另建 `pysci-infra` 薄工具 |
| `scripts/edit_progress.py:17` | `from pysci.skills.orchestration.tools.editprogress import main` | 删除脚本（Qoder taskbook 专用，不再需要） |
| `tests/skills/ai_drawing/test_tier0_unavailable.py:21` | `from pysci.skills.devops.tools import budget as budget_mod` | `budget.py` 迁 `src/pysci/support/budget.py`，改 import |
| `tests/skills/literature_research/test_api_workarounds_doc.py:12` | 同上 | 同上 |
| `tests/skills/literature_research/test_fetch_evidence_doc.py:19` | 同上 | 同上 |
| `tests/skills/literature_research/test_read_fetch_order.py:33` | `from pysci.skills.orchestration.tools import procs` | 改 `pysci.support.procs` |

> 先改 `literature_research` 的 `procs` 依赖，再删 `orchestration`；否则 `tests` 会在中途大面积 ImportError。

### A.3 `paths.py` 与 `toolchain.py`

- `paths.py` 删除：`ORCHESTRATION_ROOT`、`ORCH_STATE_ROOT`、`PODS_ROOT`、`WORKTREES_ROOT`，以及相关注释。
- `paths.py` 保留：`PROJECT_ROOT`、`ASSET_ROOT`、`DATA_ROOT`、六个技能 `*_ROOT`、`research_*_dir()`、`assert_within_data()`。
- `paths.py` 新增：`TESTS_ROOT = PROJECT_ROOT / "sub_root" / "tests"`、`EVALS_ROOT = PROJECT_ROOT / "sub_root" / "evals"`、`SUPPORT_ROOT = PROJECT_ROOT / "src" / "pysci" / "support"`。
- `toolchain.py` 已与 mySimsWS2 布局同相对路径（`DEPS_ROOT`、`UV_TOOL_DIR`、`UV_TOOL_BIN_DIR`、`UV_CACHE_DIR`），基本原样复用；只核对 `PROJECT_ROOT` 解析与 `common\bin` 注释。

### A.4 `pyproject.toml`

- 删除 `[project.scripts]`：`pysci-orch`、`pysci-dev`、`pysci-bench`、`pysci-inspect`、裸 `pysci`。
- 新增 `pysci-eval = "pysci.support.evals.cli:main"`（名称可再定）。
- 删除仅被 orchestration/TUI 使用的依赖：`textual`、`chinesecalendar`（已核对，生产技能无引用）。
- 保留：`ruff`、`pytest`、`pytest-cov`、`pre-commit`、`commitizen`、`scipy-stubs`；生产依赖不动。
- `[tool.pytest.ini_options] testpaths = ["sub_root/tests"]`；markers 保留 `hardware`、`slow`，新增 `network`、`integration`（按需）。
- `[tool.coverage.run] source = ["src/pysci"]`；`omit` 增加 `*/support/evals/tests/*`（如需要）。

### A.5 `.pre-commit-config.yaml` / `.gitattributes`

- 删除 local `harness-budget` 钩子（引用 `orchestration/guards/budget_check.py` 与 `.qoder/rules`）。
- 保留 ruff-check / ruff-format / commitizen / 官方 hooks；预算与技能契约改由 `sub_root/tests/harness/` 执行。
- `.gitattributes` 可原样复用（`*.sh eol=lf`、`*.cmd`/`*.ps1` `eol=crlf`）；新增技能资源如含二进制，按需补规则。

### A.6 如复用旧代码，可考虑抢救的最小件

| 来源 | 去向 | 理由 |
|---|---|---|
| `orchestration/tools/procs.py` | `src/pysci/support/procs.py` | 通用进程工具；`literature_research` 依赖 |
| `devops/tools/budget.py` | `src/pysci/support/budget.py` | 3 个测试依赖；预算检查仍有用 |
| `devops/tools/skilltax.py` | 暂不迁 | 如 harness 测试需要技能分类，再评估 |
| 其余 orchestration/devops 工具 | 删除 | 与 Qoder/orch/pods 强绑定 |

### A.7 若复用旧代码的参考顺序

> 默认跳过本节；核心框架的裁剪动作见 §6.2 与 §7 P4。只有某技能选型决定复用旧实现时，才按以下顺序处理：

1. 先建 `pysci/support/{procs,budget}.py` 等被复用的最小件。
2. 改该旧代码的 import 触点（A.2）。
3. 再移除它依赖的旧 orchestration/devops 模块；不要反向先删。
4. 复用部分另写测试，不搬旧测试。

