# pySciWS2 临时交接文档

> **临时文件**：项目专属 DSH 完成接手验收后删除；不进入长期文档体系。
> **日期**：2026-11-13
> **当前阶段**：P0–P5 已完成；**P6 交接待项目 DSH 验收**；P7 远端替换未开始。

## 1. 交接结论

pySciWS2 的奠基工作已推进到“项目专属 DSH 可用”：

- 项目内自包含工具链与 DSH（0.2.1-rc.5）已安装，`DSH_HOME` 在项目内。
- 三个 preset 已接线：`pysci-worker`（默认）、`pysci-leader`（可选）、`pysci-meta`（仅 harness）。
- 蓝图 v1、`pysci-infra`、核心框架（`paths.py` / `toolchain.py` / 最小 `pyproject.toml`）、分层测试入口与 eval 骨架已就位。
- **7 个科研技能尚未迁移**：按用户新政策，由项目 DSH 接手后逐个能力域重走三段式（社区调研优先）。

## 2. 项目 DSH 启动与端口

### 2.1 启动

在项目根目录执行：

```powershell
dsh web
```

`common\bin` 的 `dsh.cmd` 会按当前目录找到 `sub_root\deps_management\toolchain.json`，路由到本项目 DSH，并注入项目内 `DSH_HOME`。

当前实例已运行在 **3081**；启动时会打印带 token 的 URL（每次启动 token 不同），以打印结果为准。

### 2.2 端口规则（已固化）

- 本项目默认端口已设为 **3081**：`dsh web` 不带 `--port` 也使用 3081。
- **3080 属于隔壁项目的 DSH，不要占用**。
- 如果 3081 被占用：用 `dsh web --port 0`（让操作系统挑一个空闲端口），或显式 `dsh web --port 3082`。
- DSH 没有“3081 占用就自动 3082”的内置逻辑；`--port 0` 是唯一自动挑空闲端口的选项。固定 3081 + 必要时 `--port 0` 是当前推荐用法。
- 固化位置：`sub_root/deps_management/deps/node/modules/dsh_home/profiles/web/cordis.patch.yml` 的 `webserver` 行（`port: !!js ctx.webStartup.port ?? 3081`）。该行是整段 config 覆盖，升级 DSH 后若键变化需同步。

### 2.3 其他 profile

- 无头运行：`dsh --profile headless "<任务>"`；需要 `DEEPSEEK_API_KEY`（项目 `.env` 或环境变量）。
- 注意：headless profile 不使用 preset，默认组合是 standard；它适合 smoke/eval 基线，不适合验证 preset 隔离。

## 3. P6 交接验收（请在项目 DSH 的 meta 会话中执行）

### 检查 A：worker（默认）

新建一个会话（默认应落 `pysci-worker`），询问：

> 列出你当前可用的技能名称，并说明 `subagent` 工具允许哪些 `agent_preset`。

预期：

- 技能含 `pysci-infra` 与共享蓝图 `blueprint-canon`；
- `subagent` 枚举只有 `pysci-worker`；
- 工作说明是 `AGENTS.md`。

### 检查 B：leader

新建 leader 会话，询问：

> 你的工作说明文件是哪一份？列出可用技能；`subagent` 允许哪些 preset？`subagent_fork` 和 `workflow` 是否可用？

预期：

- 工作说明是 `AGENTS.leader.md`；
- 技能只有 leader-skills（`large-task-orchestration`、`review-acceptance`）+ 共享蓝图；
- **看不到** `pysci-infra`；
- `subagent` 仅 `pysci-worker`；
- `subagent_fork` / `workflow` 不可用（leader 禁用这两个，避免子代理继承 leader）。

### 检查 C：meta

新建 meta 会话，询问：

> 你的工作说明文件是哪一份？列出可用技能；`subagent` 允许哪些 preset？

预期：

- 工作说明是 `AGENTS.meta.md`；
- 技能含 `.agents/meta-skills`（`blueprint-canon`、`harness-dev`、`eval-runner`、`repo-recon`）+ Creator 技能；
- **看不到** worker/leader 技能；
- `subagent` 可选 `pysci-worker` / `pysci-leader` / `pysci-meta`。

### 静态与机械复核（可在 meta 会话直接跑）

```powershell
dsh --profile web --dump-config
sub_root\tests\tests.cmd all
sub_root\tests\tests.cmd harness
```

预期：

- `--dump-config` 无 patch 警告，含 `pysci-worker` / `pysci-leader` / `pysci-meta`、`selectedDefault: pysci-worker`、`port: ... 3081`；
- `tests.cmd all` 全绿（当前 27 个测试）；
- `tests.cmd harness` 全绿。

### 验收失败时

- **不要**在运行中热改 preset；先记录命令、输出、`dsh --profile web --dump-config` 的相关段落。
- 交接文档本身不修 harness；把问题交给用户/当前会话，或按 `harness-dev` 技能的备份 + 小步流程处理。

## 4. 接手后的第一件事：7 个技能逐个三段式重建

用户已明确：**技能不直接搬迁**。7 个能力域（文献研究、有限元仿真、文档写作、科研绘图、理论计算、AI 绘图、3D 建模）各自重走：

1. **选型**：盘点旧能力（只读参考）→ 按 `philosophy-community-first` 的扩展点地图做社区调研（PyPI、GitHub、MCP registry、awesome lists；核对 star、近 6 个月活跃度、许可证、Python 3.13/Windows 可用性）→ 提出候选组合 → **与用户讨论确认**。每轮社区调研用户在场。
2. **计划与实施**：在 `sub_root/others/plans` 写该技能计划；定义 CLI/接口、数据契约、落盘约定；只写社区工具表达不了的薄胶水；完成后归档。
3. **QA 同步**：在 `sub_root/tests/skills/<skill>/` 补机械断言；跑 `tests.cmd all`；需要时更新设计摘要。

规则：

- 一个能力域可以由**多个社区依赖组合**替代；不要求保持旧 CLI 形状。
- 默认不保留旧实现；若复用某段旧代码，须在选型记录中写明“为什么社区不可行”。
- 旧仓库 `D:\XXXIIIGGG\projects\pySci\pySciWS` 仅作能力参考，不是迁移基线。
- 建议顺序（可调整）：文献研究 → 科研绘图 → 文档写作 → 理论计算 → 有限元仿真 → AI 绘图 → 3D 建模。

## 5. 已完成部分的证据

- Git：`main` 分支，提交 `35ffdc8`（骨架）、`1d8bfda`（工具链/蓝图/框架）、`ffb97e8`（测试/eval）、`44b5a9d`（进度记录）。
- 旧仓库安全网：`D:\XXXIIIGGG\backups\pySciWS-20261113.bundle`（已验证完整历史）。
- 工具链：`sub_root/deps_management/toolchain.json`；git 2.56.0.3 / node 26.12.1 / uv 0.13.5 / Python 3.13.15 / DSH 0.2.1-rc.5。
- 项目 DSH：`sub_root/deps_management/deps/node/modules/dsh_home`。
- 蓝图：`.agents/shared-skills/blueprint-canon/`（7 个文件，全部 ≤8 KB）。
- 预设：`dsh_home/profiles/web/cordis.patch.yml`；生成片段 `pysci/{standard,cordis}.plugins.yml`。
- 测试：`sub_root/tests/tests.cmd all` 当前 27 passed。
- Eval smoke：`sub_root/evals/runs/example-smoke-20261113T094104Z`（returncode 0，8.3s）。

## 6. 已知边界与待办

- **7 个生产技能为空**：`src/pysci/skills/` 只有空命名空间；`data/skills/` 为空；待逐个三段式重建。
- **研究内容未迁**：`src/pysci/research/`、`data/research/`、`scripts/research/`、`tests/research/` 仍只在旧项目；按用户决定“稍后手动搬”。
- **GUI preset 冒烟**：静态接线与 `--dump-config` 已通过；三个顶层会话的可见性检查由本次交接验收完成。
- **eval 仍是 MVP**：`pysci-eval` 当前用 headless 默认组合跑单任务；preset 感知的 A/B、预审表、改进台账由项目 DSH 继续完善。
- **P7 未开始**：远端仓库替换是最后一步，需用户单独授权；研究内容未迁完前不得删除旧目录。

## 7. 临时性

本文件只服务本次交接。项目 DSH 完成 P6 验收并在自己的会话中记录后，可删除本文件；不要把它变成长期文档（蓝图与源码才是真源）。
