# PySci 项目宪法

## 一、设计哲学（简版）

本项目遵循以下核心原则。每条原则旨在引导你识别相关场景，并查阅 `blueprint-canon` 技能的详版文档以获取完整内涵。

1. **蓝图原则**
   分层蓝图是本项目的最高准则和唯一人类维护面；一切决策以蓝图为准，其他产物（如实施计划）仅为可漂移的中间过程。
   *（详见：详版 1.1, 1.2, 1.3）*

2. **社区优先与薄胶水原则**
   优先引入符合活跃度标准的社区基础设施，自研代码仅限于必要的“薄胶水层”，以维护面最小化为最高优先级。接到新需求时，先按[全生态扩展点地图](.agents/shared-skills/blueprint-canon/references/philosophy-community-first.md)定位接缝并调查社区，再决定采用、改造或自研。
   *（详见：详版 2.1 与上述地图）*

3. **自包含与勤分类原则**
   依赖与配置力求留在项目目录内（全局自包含），子目录间保持松耦合且语义清晰，目录结构推崇深层级分类而非平铺。
   *（详见：详版 3.1, 3.2）*

4. **原地文档原则**
   不维护独立文档，一切末端细节以源代码为唯一真源；通过详尽注释、docstring 及“原地 ADR”实现自文档化，消除同步地狱。
   *（详见：详版 4.1）*

5. **极简原则**
   每个功能只保留唯一的最新正确实现，默认不向后兼容、不设回退，废弃路径直接报错；确需兼容或回退时必须先请示用户，并显式标注“已弃用”。
   *（详见：详版 5.1）*

## 二、PySci 特有约束

### 2.1 技能重建，而非搬迁

pySci 的 7 个科研能力域（文献研究、有限元仿真、文档写作、科研绘图、理论计算、AI 绘图、3D 建模）**不直接复制旧 pySciWS 的实现**。每个能力域各自重走“新需求三段式”，重点重做社区调研，最大化减少自研代码；允许一个能力域由多个社区依赖组合替代。每次社区调研的候选与取舍必须交用户审核或补充，用户在每轮调研在位。技能重建由项目专属 DSH 接手后逐项执行。

### 2.2 研究内容边界

仓库只含可重建的框架、技能文档、支撑系统与最小资产骨架。研究线代码/数据/脚本/测试（`src/pysci/research/`、`data/research/`、`scripts/research/`、`tests/research/`）只在本地保留，永不上传；具体研究内容由用户自行管理。`data/用户的inbox/` 是用户专属区，基建不得依赖其中文件。

### 2.3 项目专属 DSH

本项目使用项目内自包含的 DSH（`sub_root/deps_management/deps/node/modules`），`DSH_HOME` 在项目内，不依赖全局 DSH。交互面为 Web GUI（`dsh web`）；自动化/eval 可用 `dsh --profile headless`。三个 preset：`pysci-worker`（默认）、`pysci-leader`（可选）、`pysci-meta`（仅 harness）。不保留 Qoder、`pysci-orch`、pods、taskbooks、watchdog 或夜航机制。

## 三、新需求开发三段式

面对任何新需求（需要开发新功能）时，依次完成以下三段，不得跳步。

1. **选型**：遵守“社区优先与薄胶水原则”，查阅[全生态扩展点地图](.agents/shared-skills/blueprint-canon/references/philosophy-community-first.md)，调研社区基础设施，并与用户讨论、确认选型。技能重建必须逐项执行本段，用户参加每轮调研。
2. **计划与实施**：根据确定的选型，在 `sub_root/others/plans` 创建计划并实施；全部完成后归档。计划不是蓝图，允许漂移。
3. **QA 同步**：若新功能涉及分层测试（`sub_root/tests`、`sub_root/evals`），或应被纳入测试覆盖（拿不准时咨询用户），则同步更新测试并执行全量测试，确保新功能与原有代码良好协作。

## 四、工作方式

- 收到用户消息时，积极使用 `create_goal` / `todo_write` / `subagent`；简单任务可显式声明省略。
- 默认由 worker 直接执行；大型任务可由用户选择 leader 模式，由 leader 拆解并派发给 worker 子代理。用户只与 leader 沟通，不直接指挥其 worker。
- meta 只改 harness，不碰项目代码、测试判据与研究内容；详见 `AGENTS.meta.md`。
- **视觉策略**：交付图片等视觉内容前，必须先使用 VLM 能力查看该图片，检查内容正确性；涉及视觉的工作应更积极地查看图片，不要因成本而回避。
- **会话检索**：读取指定会话时使用完整 ID（根会话带 `session-` 前缀，子代理会话为裸 uuid）；`session_search` 是字面词段匹配，中文查询取标点/空白之间的完整词段。

## 五、环境与常用命令

- 首次/换机：`sub_root\deps_management\deps_manager\init-deps.cmd`
- 依赖状态：`sub_root\deps_management\deps_manager\update-deps.cmd status`
- 依赖更新：`sub_root\deps_management\deps_manager\update-deps.cmd update`（版本真源为 `toolchain.json`）
- 测试入口：`sub_root\tests\tests.cmd all`（P5 交付后可用）
- 模型服务：项目 DSH 使用 `DEEPSEEK_API_KEY` / 账户凭据；密钥只放 `.env` 或 DSH 凭据库，不入库。
- GitHub 直连不可达时，可尝试本机代理 `http://127.0.0.1:12450`（代理不一定开启；失败就回退直连或询问用户）。
