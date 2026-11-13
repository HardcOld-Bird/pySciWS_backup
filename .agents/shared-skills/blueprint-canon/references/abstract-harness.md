# 三 preset 与 DSH harness

> **所属**：PySci 蓝图 · 设计摘要
> **上位**：[设计摘要（总纲）](abstract-core.md)
> **效力**：与实现、旧计划、旧提示词冲突时，以蓝图为准。

## 定位

PySci 使用项目内自包含的 DSH，以三个 preset 区分工作模式。harness 的目标不是“让 Agent 更忙”，而是让用户只面对必要的决策点：默认 worker 直做；大型任务交给 leader 组织；harness 自身的问题交给 meta。

## 三个 preset

| 维度 | `pysci-worker`（默认） | `pysci-leader`（可选） | `pysci-meta` |
|---|---|---|---|
| 基础 | DSH 内置 standard | DSH 内置 standard | DSH 内置 cordis（创意模式） |
| 工作说明 | `AGENTS.md` | `AGENTS.leader.md` | `AGENTS.meta.md` |
| 技能根 | `.agents/skills` + `.agents/shared-skills` | `.agents/leader-skills` + `.agents/shared-skills` | Creator 技能 + `.agents/meta-skills` |
| 生产技能 | 有（随技能重建逐步增加） | **无** | 无 |
| 直接任务 | 是（默认主力） | 否；只规划、派发、验收 | 否；只改 harness |
| 用户沟通 | 可直接沟通 | 用户只与组长沟通 | 只对用户与 eval 结果负责 |
| 子代理 preset | 仅 `pysci-worker` | 仅 `pysci-worker` | 全部三个 |
| fork 子代理 | 继承 worker（允许） | 禁用（会继承 leader） | 继承 meta（允许） |
| workflow 子代理 | 继承 worker（允许） | 禁用（会继承 leader） | 继承 meta（允许） |
| 默认会话 | 是（`selectedDefault`） | 否 | 否 |

## 会话模型

- **worker 直做**：用户与 worker 直接对话；worker 可自行派发 worker 子代理做并行子任务（深度默认 1，子代理不能再向下派发）。
- **leader 组织**：用户只与 leader 对话；leader 把任务拆到叶子，派发给 worker 子代理，验收后向用户汇报。用户不直接指挥 worker。leader 保留“用户明确要求亲做”的唯一例外，但默认一律派发。
- **meta 改进**：meta 冷启动，读会话历史与 eval 结果，改 harness；不参与项目生产。meta 可派发任意 preset 以复现问题或验证改进。

## 子代理规则（硬约束）

- worker 的 `tool-subagent.allowedAgentPresets = [pysci-worker]`，默认 `agentPreset = pysci-worker`。
- leader 的同上：只允许 worker；默认 worker。
- meta 的允许 `[pysci-worker, pysci-leader, pysci-meta]`。
- leader 禁用 `subagent_fork` 与 `workflow`：这两者的子代理无法在创建时选 preset，会继承 leader，违反“leader 子代理只能是 worker”。leader 用同一回合的多个 `subagent` 调用做并行派发。
- 深度默认 1：leader 必须把任务拆到叶子；worker 子代理不能再向下派发。

## DSH 接线

- 项目 DSH 安装于 `sub_root/deps_management/deps/node/modules`；`DSH_HOME` 在 `.../dsh_home`。
- preset 声明在 `dsh_home/profiles/web/cordis.patch.yml`；基础片段 `pysci/standard.plugins.yml` 与 `pysci/cordis.plugins.yml` 由上游同步脚本生成，不手改。
- 交互面是 `dsh web`；自动化/eval 用 `dsh --profile headless`。
- 技能隔离用 `includeDefaultRoots: false` + 显式 `customSkillDirs`；不用用户级/全局技能根，避免跨项目泄漏。
- preset 改动只在无运行会话时小步进行，改前备份、改后冒烟；运行中会话保留旧代际。

## 废弃项

- Qoder（`.qoder`、rules、hooks、worktrees）全部废弃。
- `pysci-orch`、pods、taskbooks、watchdog、夜航、jury、twin 等旧编排机制全部废弃；其能力分别由 DSH `subagent`、leader 技能、meta/evals 承接。
- 不为旧 `pysci-*` CLI 保留兼容入口；技能重建后按选型定义新入口。

## 预算

- `AGENTS*.md` 每文件 ≤ 8 KB。
- 每个 `SKILL.md` ≤ 8 KB；超出细节进 `references/`。
- 常驻注入只放“不读会做错”的内容；能按需读的细节下沉到技能。
