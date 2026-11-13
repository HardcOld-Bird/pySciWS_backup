---
name: repo-recon
description: meta 用仓库结构元数据定位 harness 配置、技能目录、preset、eval 产物与工作说明，再决定是否搜内容。需要在改动前确认“配置/技能/任务在哪”、或排查路径漂移时使用。
---

# 仓库侦察（meta 视角）

目标：用**当次**仓库状态回答“在哪”，不维护目录地图。

一句话：`git ls-tree` 看骨架 → 读边界文件 → 再搜内容。

## 1. 先拿骨架

```powershell
git ls-tree -d -r --name-only HEAD |
  Where-Object { ($_ -split '/').Count -le 4 }
```

- 只列已跟踪目录，天然排除 `node_modules/`、`dsh_home/`、`.venv/` 与产物。
- 未跟踪目录（如 `dsh_home/`）不在树里；需要时用 `git status --porcelain` 补一眼，但不要把忽略目录当项目结构。

## 2. 边界文件

出现以下文件时，以它们为模块边界，不跨目录猜：

- `AGENTS.md` / `AGENTS.leader.md` / `AGENTS.meta.md`：三 preset 的工作说明。
- `.agents/shared-skills/blueprint-canon/`：共享蓝图（只读）。
- `.agents/skills/`、`.agents/leader-skills/`、`.agents/meta-skills/`：三套技能面。
- `sub_root/deps_management/toolchain.json`：依赖版本真源。
- `sub_root/deps_management/deps/node/modules/dsh_home/profiles/web/cordis.patch.yml`：preset 真源。
- `sub_root/deps_management/deps/node/modules/dsh_home/profiles/web/pysci/*.plugins.yml`：上游片段（生成物）。
- `sub_root/tests/`、`sub_root/evals/`：分层测试与基准。
- `sub_root/others/plans/`：计划（可漂移，不是蓝图）。

## 3. 最后才搜内容

- 结构仍不足时再用 `grep` / `glob`；先报候选路径 + 依据，再读文件。
- 需要设计 Why 时配合 `blueprint-canon`（meta 视角）；它是蓝图导航，不是目录索引。

## 高频入口

| 问题 | 先看 |
|---|---|
| meta 工作说明/技能 | `AGENTS.meta.md`、`.agents/meta-skills/` |
| worker/leader 工作说明/技能（只读） | `AGENTS*.md`、`.agents/skills|leader-skills` |
| preset/harness 真源 | `.../profiles/web/cordis.patch.yml`、`.../profiles/web/pysci/` |
| DSH 安装与状态 | `.../deps/node/modules/`、`.../dsh_home/` |
| 测试/基准 | `sub_root/tests/`、`sub_root/evals/` |
| 计划 | `sub_root/others/plans/` |

> 这些只是“从哪开始”；路径若变，以第 1 步的树为准。
