---
name: repo-recon
description: 用仓库自身的结构元数据定位源码、资产、依赖、计划与技能文档，再决定是否搜内容；适合“某能力域/依赖/计划/技能/数据在哪”或需要先了解项目结构时。
whenToUse: 用户提到“源码/数据/依赖/计划/技能/某能力域在哪”，或需要先了解项目整体结构时。
---

# 仓库侦察（repo-recon）

目标：用**当次**仓库状态回答“在哪”，不维护目录地图。命令默认在工作区根（git 根）执行；Python 工具与测试命令在 `sub_root/` 内执行。路径以 `git ls-tree` 为准。

一句话：`git ls-tree` 看骨架 → 边界文件定模块（注意两个根）→ `toolchain.json` 核依赖 → `SOURCE.md` 查出处 → 再读文件。

## 1. 先拿骨架（不要先全文搜索）

```powershell
git ls-tree -d -r --name-only HEAD |
  Where-Object { ($_ -split '/').Count -le 4 }
```

- 只列**已跟踪**目录，天然排除 `node_modules/`、`dsh_home/`、`.venv/`、依赖安装物和产物。
- 未跟踪/忽略目录（`sub_root/deps_management/deps/` 下的安装物、`data/` 白名单外的本地内容、`用户的inbox/` 本地文件）不在树里；需要时用 `git status --porcelain` 补一眼，但不要把忽略目录当项目结构。
- 非 ASCII 路径会被 git 转义显示（`sub_root/\347\224\250...` = `sub_root/用户的inbox`）；`git -c core.quotepath=false ls-tree ...` 显示原字。
- bash 等价：`... | awk -F/ 'NF<=4'`。

## 2. 依赖与参考资料的真源（无子模块）

- 本仓库**没有子模块**（无 `.gitmodules`、无 `160000` 行）；第三方材料不走 submodule。
- 依赖版本真源是 `sub_root/deps_management/toolchain.json`（git/node/uv/python/dsh 的版本 + URL + sha256）；安装物在 `sub_root/deps_management/deps/`，多数被 `.gitignore` 忽略、不入树。
- npm/DSH 依赖声明：`sub_root/deps_management/deps/node/modules/package.json`（+ `package-lock.json`）；安装/更新入口：`sub_root/deps_management/deps_manager/`（`init-deps.*` / `update-deps.*`）。
- vendored 第三方资产随 `SOURCE.md` / `SOURCE.txt` 记出处（例：`sub_root/data/skills/comsol_simulation/manual/SOURCE.md`、`.../bilingual/assets/katex/SOURCE.txt`）；问“这是什么、从哪来”先读同目录 SOURCE 文件。
- 约定：`deps_management/deps/` 通常表示第三方安装物，不是自研目录；自研代码在 `sub_root/src/pysci/`。社区调研记录（选型证据）在 `sub_root/others/community_research/<date>/`。

## 3. 标出边界清单

出现以下文件时，以它们为模块边界，不要跨目录猜：

- **两个根**：工作区根（git 根，`AGENTS*.md` 与 `.agents/` 所在）与 Python 项目根 `sub_root/`（`pyproject.toml` 所在）。代码里对应 `pysci.paths.WORKSPACE_ROOT` / `PROJECT_ROOT`；路径问题先看这两个常量（`sub_root/src/pysci/paths.py`），别数 `parents[N]`。
- `AGENTS.md` / `AGENTS.leader.md` / `AGENTS.meta.md`：三 preset 的工作说明（`.local.md` 为本机覆盖，不入仓）。
- `.agents/shared-skills/`（共享技能/蓝图）、`.agents/skills/`、`.agents/leader-skills/`、`.agents/meta-skills/`：各 preset 技能面。
- `sub_root/pyproject.toml`：Python 项目根（uv/pytest/ruff/hatch 配置都在这里）；配套 `sub_root/.python-version`、`sub_root/uv.lock`。
- `sub_root/src/pysci/` ↔ `sub_root/data/`：源码树与资产树（镜像关系）。
- `sub_root/deps_management/toolchain.json`：工具链版本真源；`sub_root/subagent-model-policy.yaml`：子代理模型策略真源。
- `sub_root/tests/`（`harness/`、`skills/<域>/`）与 `sub_root/evals/`（`tasks/`、`overlays/`、`reports/`）：分层测试与基准。
- `sub_root/others/plans/`：计划（可漂移，不是蓝图；`archived/` 是归档）。
- `sub_root/用户的inbox/`：用户自管区（仅 `pysci_settings.toml` 入库，其余本地保留；Agent 原则只读）。

## 4. 版本要交叉核对

- `toolchain.json` 是**声明**版本，实际以正在使用的运行时/安装物为准（DSH 运行时在 `deps_management/deps/node/`）；存疑先报告差异，再动手。
- `package.json` / `uv.lock` / `.python-version` 可能与 `toolchain.json` 不同步；出现不一致先报告，不要自行“修正”。
- `data/` 与 `src/` 是镜像关系但不是同步关系：新增能力域目录时两侧都要建骨架（`data/skills/<域>/` 只入骨架面，其余本地保留）。

## 5. 最后才搜内容

- 结构仍不足时再用 `grep` / `glob`；先报候选路径 + 依据，再读文件。
- 需要“整体理解/设计 Why”时配合 `blueprint-canon`（蓝图，不是目录索引）。

## 高频入口（起点，不是地图）

| 问题 | 先看 |
|---|---|
| 某能力域源码 | `sub_root/src/pysci/skills/<域>/`（与 `sub_root/data/skills/<域>/` 镜像） |
| 某能力域技能文档 | `.agents/skills/<域>/SKILL.md` |
| Python 依赖/版本 | `sub_root/pyproject.toml`、`sub_root/uv.lock`、`sub_root/deps_management/toolchain.json` |
| 依赖安装/更新 | `sub_root/deps_management/deps_manager/`（安装物在 `deps/`，多不入库） |
| 第三方材料出处 | 同目录 `SOURCE.md`/`SOURCE.txt`；选型证据在 `sub_root/others/community_research/` |
| 测试/基准 | `sub_root/tests/`、`sub_root/evals/` |
| 计划（非蓝图） | `sub_root/others/plans/`（`archived/` 归档） |
| 蓝图/设计 Why | `AGENTS.md`、`.agents/shared-skills/blueprint-canon/` |
| 路径常量 | `sub_root/src/pysci/paths.py`（`WORKSPACE_ROOT` / `PROJECT_ROOT`） |
| 用户自管区 | `sub_root/用户的inbox/`（Agent 原则只读） |

> 这些只是“从哪开始”；路径若变，以第 1 步的树为准。
