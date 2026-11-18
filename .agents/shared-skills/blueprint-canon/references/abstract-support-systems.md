# 项目支撑系统

> **所属**：PySci 蓝图 · 设计摘要
> **上位**：[设计摘要（总纲）](abstract-core.md)
> **效力**：与实现、旧计划、旧提示词冲突时，以蓝图为准。

项目支撑系统由两部分组成：**依赖管理系统**负责“如何取得、固定、更新、回退”，其项目依赖清单是“取得之后放在项目里的东西”；**分层测试系统**负责三层验证——可机械断言层让代码与技能契约以可执行判据反复验证，可基准测试层以黄金基准裁决工作 Agent 的 harness 改进，主观层由 meta 执行 harness 改进并由人类评审。依赖管理让项目目录本身成为一个完整、可复制、可重建的环境（除一层只做路由的 shim 外）；分层测试让代码、技能契约与工作 Agent 的行为都能被检查。

## 依赖管理系统

**定位**。一套以项目目录为边界的依赖生命周期管理：把工具链、DSH、Python 环境与第三方依赖收进项目，使“换一台机器”退化为“复制目录 + 运行一次初始化”。

**设计目标**。

- **唯一外部前提**：Windows + PowerShell；除把路由层加入系统 PATH 这类一次性显式操作外，不要求机器级安装；不依赖预装的 git、node、uv 或 Python。**已知例外**：少数以系统注册表发现安装位置的软件（COMSOL 经 `COMSOLROOT`、Zotero 经 App Paths）在项目内自包含迁移后，需**一次管理员级注册表同步**（记录于各自迁移动机与坑点册）；这是迁移动作而非持续依赖。
- **项目内自足**：工具本体、解释器、DSH、npm 包根、Python 包与环境、下载缓存全部位于项目目录内；项目外只保留一层只做路由的 shim。大型第三方软件按需迁入 `deps/`（体量不入库、不随仓库分发）。
- **shim 只路由**：全局 shim 目录按“当前所在项目”把命令转发给项目内实现，不承载工具本体、不复制逻辑；项目命令不得回退到系统同名工具。
- **两段式生命周期**：init 负责首次、换机与迁移（补齐工具、部署/核验 shim、创建环境）；update 负责日常（只在依赖已就绪时工作，未就绪则明确拒绝并指出修复方式，绝不“半套成功”）。
- **可复现**：版本由 `toolchain.json`（版本、下载地址、校验和）与锁文件固定；下载默认使用可用镜像；同一工具只保留一份实现。
- **单一真源与分层**：自动管理工具的版本与下载地址以 `toolchain.json` 为唯一真源；**手动迁移的大型软件以项目内目录 + 原地 ADR 为真源**，由代码内工具路径常量统一解析（工具链清单的消费者只覆盖其建模的那些工具，不强行登记无消费者的条目）。工具、缓存、上游源码、模型资产、构建产物分层存放。
- **凭据**：第三方 API key 等本地密钥集中在项目根 `.env`（gitignore 排除，不入库、不进缓存），按能力域按需迁移；管线从环境读取，不硬编码。
- **可观测与可回退**：依赖变更留下记录；有锁文件的依赖可回退；失败能定位到“哪个工具、哪个版本、哪一步”。

**依赖分类**（按“有没有可回退的版本记录”，而非按生态划分）：

- **managed**：有锁文件、被 git 跟踪 → 自动更新、自动回退（DSH 的 npm 包、Python 包）。
- **vendored**：项目内二进制、有版本清单与 current 指针 → 修复/对齐到清单版本（git、node/npm、uv、Python 解释器）。
- **手动 vendor**：项目内二进制或资产、**无清单指针、无自动对齐** → 只做存在性/版本探测；升级即重走迁移（大型第三方软件、模型权重、上游源码浅克隆）。
- **detected**：机器或项目内已有、无锁文件、无自动降级 → 只比对版本并提示手动处理（未迁入项目的大型软件与驱动）。
- **tool**：项目内 uv tool（如后续引入的 MCP server），单独钉版本、单独更新；亦可为“项目 venv 包 + DSH profile 挂载（link 或 stdio 条目）”的混合形态。

**运行方式**。总更新器不实现各生态的更新细节，只做预检、编排、汇总与回退。

- **init**：审计 → 安装 vendored 工具 → 核验 shim → 创建项目 `.venv` → 安装 DSH → 创建/迁移 `dsh_home` → 验证。
- **update**：统一入口，先预检（git 仓库、工作区干净、项目内工具就绪、DSH 未运行），再按顺序调用子更新器并汇总，最后尝试推送已提交的变更；另提供 status、rollback、log。
- **硬规则**：没有 git 不更新；工作区脏不更新；DSH 运行中不更新；非项目内工具不更新。

**目前纳入项目依赖面的条目**（自动化管理 + 手动 vendor）：

| 依赖 | 类别 | 管理方式 |
|---|---|---|
| git | vendored | 按清单下载、SHA256 校验、解压；项目级 git 配置；shim 路由 |
| node / npm / npx | vendored | 按清单安装；npm 随 Node，不单独 pin；npm cache/prefix 在项目内 |
| uv / uvx | vendored | 按清单安装；shim 路由 |
| Python 解释器 | vendored | 由项目 uv 安装 standalone CPython 3.13.15 到 `deps/python/` |
| DSH | managed | 作为 npm 依赖装入 `deps/node/modules`，`package-lock` 锁定；shim 路由；可升级/回退 |
| Python 包 | managed | uv 依 `pyproject.toml` 解析到项目 `.venv`；技能重建时逐项添加；网络受限工具按域以可选件/豁免 pin 引入 |
| **COMSOL / TeX Live / Blender / Zotero** | **手动 vendor**（2026-11 起迁入 `deps/`） | 整体目录复制进项目；升级=人工迁移 + 一次注册表同步（COMSOLROOT / App Paths）；许可证与数据目录随行 |
| **pandoc、wtpsplit 等模型权重、rembg 缓存、MCP 上游源仓** | **手动 vendor** | 分别落 `deps/tools`、`deps/models`、`deps/cache`、`deps/repos`（浅克隆 pin commit）；路径由代码常量解析，不入库 |
| **MCP server（paper-search / zotero / comsol / blender）** | **tool（混合形态）** | 项目 venv 包或源仓 link 挂 DSH profile（`cordis.patch.yml` 以 insert 条目 + env 注入），stdio 冒烟校验 |
| LibreOffice / NI-DAQmx | detected | 只比对版本并提示；不自动安装（LibreOffice 转换另有随 DSH 捆绑的 Kit 可用） |

**验收标准**：

- **新机器**：安装 PowerShell → 复制完整项目目录（含本地 `deps/`）→ 运行一次 init → 能启动项目 DSH；手动 vendor 的大型软件随目录复制即可用，唯一附加动作为一次管理员级注册表同步（仅限以注册表发现的软件）。
- **日常**：一条 update 检查并更新全部依赖；任一项未就绪时明确拒绝，并给出修复方式。
- **诊断与回退**：失败可定位到工具、版本与步骤；managed 依赖可回退；依赖变更史可查。

## 分层测试系统

**定位**。“测试层级 = 抽象层级”：项目的可判定性自下而上分为三层——**可机械断言层**（`sub_root/tests`）、**可基准测试层**（`sub_root/evals`）、**主观层**（meta）。三层首尾相接：主观层本身不可测试，但其产出（worker/leader 的 harness 改进）经可基准测试层的黄金基准检验；可基准测试层的产出经可机械断言层的测试检验；测试的缺口由人类在最终产物中发现并回补。人类的注意力收敛为各层的审核门与主观层的专项改进；测试套件只是执行器，真正的资产是“观测面 + 判据 + 确定性 + 基线”。

### 第 1 层：可机械断言层（`sub_root/tests`）

- **一个入口**：`sub_root/tests/tests.cmd all` 依次跑 ruff、pytest（默认排除 hardware）与 harness 契约测试；`fast` 只跑快速子集，`hardware` 只跑外部设备用例，`cov` 附带覆盖率。
- **技能契约**：每个技能的 `SKILL.md` 有合法 frontmatter（kebab-case、description 非空）、名称唯一、引用文件存在、预算达标。
- **技能胶水断言**：每个能力域的薄胶水在 `tests/skills/<domain>/` 有配套机械断言（与技能文档同源维护）；外部重型依赖（COMSOL/Blender/浏览器打印等）标 `hardware` 分离。
- **harness 契约**：`AGENTS*.md` 预算、preset YAML 可解析、三 preset 的技能根/工作说明/子代理白名单与蓝图矩阵一致、leader 技能目录不含生产技能名、meta 技能目录与 worker/leader 隔离、上游 `standard/cordis` 片段 `--check` 通过。
- **确定性**：固定 seed；网络/LLM/COMSOL/Zotero 用 mock/cassette；`hardware`、`network`、`slow` 标记；默认不联网。

### 第 2 层：可基准测试层（`sub_root/evals`）

> **当前状态**（2026-11-18 用户裁决）：七域重建期间**暂缓建设**——第 1 层机械断言已覆盖技能契约与胶水，基准任务待需要时统一补建（素材与冒烟脚本已留存可复用）。本节为设计目标，非当前待办。

- **被测物**：worker/leader 的 harness 版本（preset 组合 + 技能 + 工作说明 + 规程），以 patch 文件标识。
- **黄金任务**：直接照搬真实摩擦（例如“按三段式新增一个技能选型记录”“修复路径迁移后的测试失败”）；任务包含 `prompt.md`、`task.yaml`（快照、模型路由、预算、判据）、`checks/`（机械判据）。
- **运行器**：`sub_root/src/pysci/support/evals/` + `pysci-eval` CLI；优先用项目 `dsh --profile headless` 跑一次隔离会话，需要细粒度事件时回退到 SDK；记录 token、墙钟、轮次。
- **前后对比**：同一任务、同一项目快照、同一模型路由，对比“改进前/后”；先跑聚焦任务，默认再跑全部有效黄金任务做不劣化回归。
- **产物**：`sub_root/evals/{tasks,runs,reports,overlays}`；`reports/improvement-ledger.md` 记录改进台账；每次对比产出预审表（初判 + 证据引用）交人类复核。

### 第 3 层：主观层（meta）

- 不设目录；产物沉淀在 `sub_root/evals` 的台账与报告。
- 由 `pysci-meta` preset + `.agents/meta-skills` 执行；冷启动，不携带既往会话上下文。
- **三不碰**：不改蓝图；不改 `sub_root/tests`/`sub_root/evals` 的判定逻辑与项目代码；不碰研究内容。
- 只对用户与 eval 结果负责；每次改进走“问题来源 → 改动 → 基准结果 → 是否实装”的台账。
- 隔离不求绝对：小幅泄露可接受，维护面优先；meta 可主动查看 worker/leader 环境以复现问题。
