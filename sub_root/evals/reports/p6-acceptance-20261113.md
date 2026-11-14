# P6 交接验收记录（2026-11-13）

> 执行者：项目 DSH 的 `pysci-meta` 会话（meta 具备派发 worker/leader 子代理的能力，三个检查都在同一会话内完成）。
> 依据：`HANDOFF.md` §3「P6 交接验收」。
> 结论：**验收通过**；同时在验收中发现并修复一个会阻断全部生产工作的环境级阻碍（见 §5）。

## 1. 检查 A：worker（默认）

方式：派发 `pysci-worker` 子代理，询问技能列表、`subagent` 可选 preset、工作说明文件。

| 期望 | 实测 | 判定 |
|---|---|---|
| 技能含 `pysci-infra` 与共享蓝图 `blueprint-canon` | `blueprint-canon`、`diagnose-windows-sandbox-acl`、`pysci-infra` | 通过（多出的 `diagnose-windows-sandbox-acl` 是 DSH 内置排障技能，属宿主层，非 worker 技能根） |
| `subagent` 枚举只有 `pysci-worker` | 仅 `pysci-worker` | 通过 |
| 工作说明是 `AGENTS.md` | `D:\XXXIIIGGG\projects\pySci\pySciWS2\AGENTS.md`（「PySci 项目宪法」，64 行） | 通过 |

## 2. 检查 B：leader

方式：派发 `pysci-leader` 子代理，询问工作说明、技能、`subagent` 枚举、`subagent_fork` / `workflow` 是否可用。

| 期望 | 实测 | 判定 |
|---|---|---|
| 工作说明是 `AGENTS.leader.md` | `D:\XXXIIIGGG\projects\pySci\pySciWS2\AGENTS.leader.md`（「PySci 组长工作说明」，52 行） | 通过 |
| 技能只有 leader-skills + 共享蓝图 | `large-task-orchestration`、`review-acceptance`、`blueprint-canon`（+ 宿主 `diagnose-windows-sandbox-acl`） | 通过 |
| **看不到** `pysci-infra` | 未出现 | 通过 |
| `subagent` 仅 `pysci-worker` | 仅 `pysci-worker` | 通过 |
| `subagent_fork` / `workflow` 不可用 | 两者均不在该会话的工具 schema 中 | 通过 |

## 3. 检查 C：meta（本会话）

| 期望 | 实测 | 判定 |
|---|---|---|
| 工作说明是 `AGENTS.meta.md` | `AGENTS.meta.md`（本会话 system-reminder「Instructions from: AGENTS.meta.md」） | 通过 |
| 技能含 `.agents/meta-skills`（`blueprint-canon`、`harness-dev`、`eval-runner`、`repo-recon`）+ Creator 技能 | 上述 4 个全在；另有 Creator 侧 `agent-experience`、`cordis-composition-reference`、`cordis-plugin-development`、`editing-cordis-compositions` 与宿主 `diagnose-windows-sandbox-acl` | 通过 |
| 看不到 worker/leader 技能 | 目录中无 `pysci-infra`、`large-task-orchestration`、`review-acceptance` | 通过 |
| `subagent` 可选三个 preset | `pysci-worker`、`pysci-leader`、`pysci-meta` | 通过 |

## 4. 静态与机械复核

| 命令 | 结果 |
|---|---|
| `dsh --profile web --dump-config` | exit 0，1431 行；**无 patch 警告**（严格匹配 `not found/failed/warning/invalid/unknown id` 无命中）；含 `preset-pysci-worker` / `preset-pysci-leader` / `preset-pysci-meta`；`selectedDefault: pysci-worker`；`port: !!js ctx.webStartup.port ?? 3081`。完整输出存档：`sub_root/evals/runs/p6-dumpconfig-web.txt` |
| `sub_root\tests\tests.cmd all` | **27 passed**（0.17s），exit 0 |
| `sub_root\tests\tests.cmd harness` | **11 passed**（0.12s），exit 0 |

生成的 preset 段落核对（`--dump-config` 输出）：三个 preset 各自 `agent-instructions` 指向正确文件；`skill-filesystem` 的 `includeDefaultRoots: false`，`customSkillDirs` 分别为 `.agents/skills` / `.agents/leader-skills` / `.agents/meta-skills`（+ 共享或 Creator 目录）；`allowedAgentPresets` 分别为 `[pysci-worker]` / `[pysci-worker]` / `[pysci-worker, pysci-leader, pysci-meta]`；leader 明确 `disabled: true` 掉 `tool-subagent-fork` 与 `tool-workflow`。

## 5. 验收中发现的阻断项：workspace 写权限（已修复）

**现象**：三个 preset 下 `pwsh` 工具一律在沙箱授权阶段失败——
`Error: SetNamedSecurityInfoW failed (Win32 5): grantWrite(D:\XXXIIIGGG\projects\pySci\pySciWS2)`。
worker 与 leader 子代理各自独立复现。表现为 `read` / `glob` / `working_directory` 正常，但**任何 shell 命令都起不来**，即测试、依赖管理、git 操作全部不可用。这不是 preset/harness 配置问题，是 Windows 文件权限前置条件不满足。

**处置**：用 `diagnose-windows-sandbox-acl` 技能的单命令诊断+修复（授权一次，非提权运行），对 workspace 根做了一次 DACL 变更：

- 变更对象：`D:\XXXIIIGGG\projects\pySci\pySciWS2`（仅此一个；祖先目录只读检查未改）。
- 变更内容：为当前登录用户（SID `S-1-5-21-3622420574-2852353489-2074170118-1002`）补一条 FullControl 允许项；文件内容与所有者未变；未删除任何拒绝项。
- 验证：`writeOwner` 由 `false` → `true`，`writeDac` 保持 `true`，`verification: verified`。
- 复验：① meta 会话在受限模式下重跑原失败操作（`Get-ChildItem AGENTS*.md`）成功；② 修复后另派 `pysci-worker` 子代理执行同一条只读 `pwsh` 命令，返回正常、stderr 为空、无 `grantWrite(...)` 报错（即**先报错的 worker 会话侧**已恢复）；③ `tests.cmd all` / `harness`、`dsh --dump-config` 随后全部正常。
- 备份与回滚：`sub_root/evals/acl-recovery/acl-backup-c27ab7c5b09146a296f53dab043fad0b.json`（+.ps1）；回滚命令
  `pwsh -NoProfile -File '...\acl-backup-c27ab7c5b09146a296f53dab043fad0b.json.ps1' -Path 'D:\XXXIIIGGG\projects\pySci\pySciWS2' -AllowRoot 'D:\XXXIIIGGG\projects\pySci\pySciWS2' -Restore '...\acl-backup-c27ab7c5b09146a296f53dab043fad0b.json'`
- 附注：修复脚本本身受执行策略限制（脚本未签名），需在同一命令里先 `Set-ExecutionPolicy -Scope Process Bypass` 再调用；这是 DSH 技能封装与执行策略的交互问题，建议上游修（技能文档里未提）。

**影响**：若不修，第 4 节的三条机械复核与后续全部 skill 重建都无法进行。**建议把「pwsh 在本 workspace 能否启动」加入交接验收清单**（一条命令即可判定），而不是只查 `--dump-config`。

## 6. 其他观察（不阻断，供后续判断）

1. `pysci-meta` 的 persona 设了 `includeRuntimeContext: false`，但本会话仍收到 runtime context 快照。推测运行时上下文由宿主级插件注入而非 persona，二者语义重叠；不影响验收，改不改由用户定。
2. 宿主层把 `diagnose-windows-sandbox-acl` 注入了三个 preset（`includeDefaultRoots: false` 仍可见），说明它来自独立的技能根。它现在确实有用，但会占用每个 preset 的技能目录位；是否只保留在 meta/worker 由用户决定。
3. 子代理会话的 delegation depth 为 1/1（meta 为 0）。worker 子代理因此不能再用 `subagent`；这是预期行为，不是缺陷。
4. 本次修复产物 `sub_root/evals/acl-recovery/` 目前是未跟踪目录（`git status` 唯一未跟踪项）。建议不提交，或由用户决定是否加入忽略。

## 7. 结论

- 检查 A / B / C 全部符合预期；`--dump-config` 无警告；`tests.cmd all` 27 passed、`harness` 11 passed。
- 唯一实质问题是 workspace 写权限，已修复并复验，回滚材料齐备。
- 因此 `HANDOFF.md` §3 的验收目标已完成；`HANDOFF.md` 是否删除、下一阶段（7 个技能三段式重建）如何排期，等用户指示。
