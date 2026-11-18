# 项目支撑系统

> **所属**：PySci 蓝图 · 设计摘要
> **上位**：[设计摘要（总纲）](abstract-core.md)
> **效力**：与实现、旧计划、旧提示词冲突时，以蓝图为准。

项目支撑系统由两部分组成：**依赖管理系统**负责“如何取得、固定、更新、回退”；**分层测试系统**负责三层验证——机械断言层让代码与技能契约反复可验，基准测试层以黄金基准裁决 harness 改进，主观层由 meta 执行改进、人类评审。依赖管理让项目目录成为完整、可复制、可重建的环境（除一层只做路由的 shim）；分层测试让代码、技能契约与 Agent 行为都能被检查。

## 依赖管理系统

**定位**。以项目目录为边界的依赖生命周期管理：工具链、DSH、Python 环境与第三方依赖收进项目，使“换一台机器”退化为“复制目录 + 一次初始化”。

**设计目标**。

- **唯一外部前提**：Windows + PowerShell；除把路由层加入系统 PATH 这类一次性操作外，不要求机器级安装。**已知例外**：以注册表发现安装位置的软件（COMSOL `COMSOLROOT`、Zotero App Paths）自包含迁移后需一次管理员级注册表同步——迁移动作，非持续依赖。
- **项目内自足**：工具本体、解释器、DSH、npm 包根、Python 包与环境、下载缓存、模型资产、上游源仓全部位于项目内；项目外只留一层只做路由的 shim；大型第三方软件按需迁入 `deps/`（不入库、不随仓库分发）。
- **shim 只路由**：按“当前所在项目”转发命令给项目内实现，不承载工具本体、不复制逻辑；项目命令不得回退到系统同名工具。
- **两段式生命周期**：init 负责首次、换机与迁移（补齐工具、部署/核验 shim、创建环境）；update 负责日常（依赖未就绪则明确拒绝并给修复方式，绝不“半套成功”）。
- **可复现**：版本由 `toolchain.json`（版本、地址、校验和）与锁文件固定；默认镜像下载；同一工具只保留一份实现。
- **单一真源与分层**：自动管理工具以 `toolchain.json` 为唯一真源；**手动迁移的大型软件以项目内目录 + 原地 ADR 为真源**（路径由代码内工具路径常量统一解析，清单不登记无消费者的条目）。工具、缓存、上游源码、模型资产、构建产物分层存放。
- **凭据**：API key 等密钥集中在项目根 `.env`（不入库、不进缓存），按域按需迁移；管线从环境读取，不硬编码。
- **可观测与可回退**：变更留记录；有锁文件的依赖可回退；失败能定位到“工具、版本、步骤”。

**依赖分类**（按“有没有可回退的版本记录”）：

- **managed**：有锁文件、被 git 跟踪 → 自动更新、自动回退（DSH 的 npm 包、Python 包）。
- **vendored**：项目内二进制、有版本清单与 current 指针 → 修复/对齐到清单版本（git、node/npm、uv、Python）。
- **手动 vendor**：项目内二进制或资产、**无清单指针、无自动对齐** → 只做存在性/版本探测；升级即重走迁移（大型软件、模型权重、上游源仓浅克隆）。
- **detected**：机器或项目内已有、无锁文件、无自动降级 → 只比对版本并提示手动处理（未迁入的软件与驱动）。
- **tool**：项目内 uv tool；亦可为“venv 包 + DSH profile 挂载（link/stdio 条目）”的混合形态（MCP server）。

**运行方式**。总更新器不实现各生态细节，只做预检、编排、汇总与回退。

- **init**：审计 → 安装 vendored 工具 → 核验 shim → 建 `.venv` → 装 DSH → 建/迁 `dsh_home` → 验证。
- **update**：先预检（git 仓库、工作区干净、工具就绪、DSH 未运行），按序调子更新器+汇总+推送已提交变更；另供 status/rollback/log。
- **硬规则**：没有 git 不更新；工作区脏不更新；DSH 运行中不更新；非项目内工具不更新。

**目前纳入项目依赖面的条目**：

| 依赖 | 类别 | 管理方式 |
|---|---|---|
| git / node / uv / Python | vendored | 按清单下载校验；shim 路由 |
| DSH | managed | npm 依赖（`package-lock` 锁定）；可升级/回退 |
| Python 包 | managed | uv 依 `pyproject.toml` 解析到 `.venv` |
| COMSOL / TeX Live / Blender / Zotero | 手动 vendor | 目录复制进 `deps/`；升级=人工迁移+一次注册表同步 |
| pandoc、模型权重、缓存 | 手动 vendor | 落 `deps/tools/models/cache`；路径由代码常量解析，不入库 |
| MCP 上游源仓 | 手动 vendor | `deps/repos` 浅克隆 pin commit |
| MCP server（paper-search/zotero/comsol/blender） | tool | venv 包或源仓 link 挂 DSH profile（insert 条目+env），stdio 冒烟 |
| LibreOffice / NI-DAQmx | detected | 只比对提示；LibreOffice 转换用 DSH 捆绑 Kit |

**验收标准**：

- **新机器**：PowerShell → 复制完整项目目录（含本地 `deps/`）→ 一次 init → 能启动项目 DSH；唯一附加动作=一次管理员级注册表同步（仅注册表发现类软件）。
- **日常**：一条 update 检查并更新全部依赖；任一项未就绪时明确拒绝并给出修复方式。
- **诊断与回退**：失败可定位到工具、版本与步骤；managed 依赖可回退；变更史可查。

## 分层测试系统

**定位**。“测试层级 = 抽象层级”：可判定性自下而上三层——**可机械断言层**（`sub_root/tests`）、**可基准测试层**（`sub_root/evals`）、**主观层**（meta）。首尾相接：主观层不可测试，但其产出（harness 改进）经可基准层检验；可基准层产出经机械层检验；缺口由人类在最终产物中发现并回补。人类注意力收敛为各层审核门与主观层专项改进；真正资产是“观测面 + 判据 + 确定性 + 基线”。

### 第 1 层：可机械断言层（`sub_root/tests`）

- **一个入口**：`tests.cmd all`（ruff + pytest 默认排除 hardware + harness 契约）；`fast`/`hardware`/`cov` 变体。
- **技能契约**：`SKILL.md` frontmatter 合法（kebab-case、description 非空）、名称唯一、引用文件存在、预算达标；**各域薄胶水在 `tests/skills/<domain>/` 有配套机械断言**（与技能文档同源维护），外部重型依赖（COMSOL/Blender/浏览器打印等）标 `hardware`。
- **harness 契约**：`AGENTS*.md` 预算、preset 可解析且与蓝图矩阵一致、技能目录隔离、上游片段 `--check` 通过。
- **确定性**：固定 seed；网络/LLM/COMSOL/Zotero 用 mock/cassette；默认不联网。

### 第 2 层：可基准测试层（`sub_root/evals`）

> **当前状态**（2026-11-18 裁决）：暂缓建设——机械层已覆盖契约与胶水；基准任务待需要时统一补建（素材已留存）。

- **被测物与任务**：worker/leader 的 harness 版本（preset+技能+工作说明+规程，以 patch 标识）；黄金任务照搬真实摩擦，含 `prompt.md`、`task.yaml`（快照/路由/预算/判据）、`checks/`。
- **运行与对比**：`src/pysci/support/evals/` + `pysci-eval`；优先 `dsh --profile headless`，需细粒度事件回退 SDK；同任务/快照/路由对比改进前后，先聚焦后全量回归；记录 token、墙钟、轮次。
- **产物**：`evals/{tasks,runs,reports,overlays}`；台账 + 预审表（初判+证据）交人类复核。

### 第 3 层：主观层（meta）

- 无目录；产物沉淀在 `sub_root/evals` 台账与报告。
- 由 `pysci-meta` preset + `.agents/meta-skills` 执行；冷启动，不带既往会话上下文。
- **三不碰**：不改蓝图；不改 `tests`/`evals` 判定逻辑与项目代码；不碰研究内容。
- 只对用户与 eval 结果负责；走“问题来源 → 改动 → 基准结果 → 是否实装”台账。
- 隔离不求绝对：小幅泄露可接受，维护面优先；meta 可查看 worker/leader 环境复现问题。
