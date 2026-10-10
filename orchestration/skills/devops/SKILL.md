---
name: devops
description: devops 组员的作业规程：worktree 隔离实施、PYTHONPATH 覆盖测试策略、技能部署执行、pysci-dev 工具面、git 纪律与巡检清单。devops pod 会话中被派发基础设施任务时使用。
---

# devops 作业规程

角色边界与红线见 charter（`.qoder/rules/charter.md`）；本文放**操作知识**。
专属 CLI：`pysci-dev`（doctor / sync / worktree）。你不使用 pysci-orch。

## worktree 作业规程（一切代码改动）

```bash
# 1. 开 worktree（从最新 main）
#    ← drain/backlog 派发时，<任务名> 必须用**任务书 id**（如 20261010-xxx），便于孤儿回收清理
pysci-dev worktree add <任务名>        # = git worktree add -b wt-<任务名> .qoder/worktrees/<任务名>
WT="$(git rev-parse --show-toplevel)/.qoder/worktrees/<任务名>"

# 2. 在 worktree 内改代码（Edit/Write 用 $WT 下路径）

# 3. 测试 —— PYTHONPATH 覆盖策略（实测定型 2026-10-09）：
#    复用主 .venv 全部依赖，worktree 代码优先于 editable 安装的主树 src；
#    收集 1452 项测试约 7.5s，无需在 worktree 内新建环境。
PYTHONPATH="$WT/src" uv run --no-sync pytest "$WT/tests" -x -q
PYTHONPATH="$WT/src" uv run --no-sync python -c "import <改动模块>; ..."   # 定向验证

# 3b. 仅当 pyproject 依赖本身变更时，才在 worktree 内建独立环境（重、分钟级）：
#     uv sync --project "$WT"   然后用 "$WT/.venv" 运行

# 4. 合并回 main（在主树操作）：
git merge --no-ff wt-<任务名>      # 或 cherry-pick；冲突按项目纪律解决，不丢弃改动

# 5. commit（cz 规范；pre-commit 必须全过，禁 --no-verify）
#    合并/提交后**尝试一次限时 push**（best-effort，用户裁决 2026-10-10）：
timeout 25 git push origin main   # 忽略一切失败（网络/认证/超时）；不重试、不设代理
#    ↑ 超时是硬要求：直连失败/凭据缺失会挂起并阻塞 drain。push 按分支增量，
#      首次成功即带上此前所有积压提交——故失败当预期、不着急。
# 6. 清理
pysci-dev worktree remove <任务名>
```

注意：worktree 检出**不含未跟踪重数据**（data/research 的 simulation/storage 等被
gitignore）；需要真实数据验证时以绝对路径只读引用主树。

**worktree 命名约定**：drain/backlog 派发时用**条目 id** 作 `<任务名>`（take_first 已把
`worktree=id` 记入条目）。若 devops 会话中途死亡留下孤儿条目，下次 drain 孤儿回收会据
`wt-<id>` 名 `git worktree remove --force` + `branch -D` 清理残留（否则重派 `worktree add`
撞名失败）。手动/组长临时派发可用任意名——不属孤儿、不会被自动清理（防误删并发作业）。

**批量消化任务书**（drain 组包，backlog 20261010-batch-digest）：任务书可能含**种子**（必做）
+ 同提请者**组包菜单**。你**自主选取 ≥1 项**（必含种子；可只吃种子），把选中项合并到
**同一 worktree**（名 = 种子 id）实施，可按需重写为合并方案。交付时在 `<result>` 以
`backlog id=<id1>,<id2>,…` 列出**本次完成的全部 id**（必含种子），orch 逐 id 销账；未完成的
**不要**列入（留 pending 下轮）。种子受阻照常交付 `<blocked>`（组包不连坐）。

## 技能部署（真本 → 副本）

- 真本：`orchestration/skills/<name>/`（git 跟踪）；manifest：同目录 `manifest.toml`；
- 改完真本执行 `pysci-dev sync`（与 pysci-orch sync 同引擎）；`sync --check` 查漂移；
- `tree_hash` **行尾无关**（哈希前 CRLF→LF，2026-10-10 修 backlog 20261010-003127-devops）：
  worktree 内 sync 提交的 `skills-deployed.json` 对 main 即正确，无需合并后重跑；doctor
  「部署台账巡检（sync --check）」哨兵记录哈希漂移；
- 红线：只碰 manifest 声明的部署名；**不删除成员自建技能**；不手改部署副本。
- **真本 references 超 8KB 的处置**（basic.md §1 按需档）：按 `##` 主题拆成
  `<stem>-0N-<slug>.md` 分册（逐字搬运、拼回原文须逐字节相等），原名留作**分册索引**
  （列出各册所含小节与尺寸）；所属 SKILL.md 的索引行标 ``*`` 并在节首加一行
  「（`*`=分册索引，按需读单册）」——SKILL.md 自身也须 ≤8192B（超限就压缩措辞，
  细节本就在分册里）。改完跑 `pysci-dev doctor` 的预算审计复测。

## pysci-dev 命令面

```
pysci-dev doctor [--pod <id>]   # pod 健康巡检：hooks 接线/leader 规则排除/charter/AGENTS.md 体量/积压
                                 # + 部署台账巡检（sync --check 漂移）+ harness 预算三档审计（≤8192B/文件）
pysci-dev sync [--check]        # 技能部署
pysci-dev worktree add|remove|list [<name>]
```

## 新 pod 脚手架（只读层入库面）

每 pod 仅 5 个文件入 git：`.gitignore`、`.qoder/settings.json`、`.qoder/mcp.json`、
`.qoder/rules/charter.md`、`AGENTS.md`；`bench/`·`outbox/`·`.qoder/skills/` 被 pod
`.gitignore` 排除，`inbox/` 空目录不跟踪——后三者须**合并后在 main 上** `mkdir` +
`sync` 再生。settings.json 模板必带两件事：① 两个 hook 接线（Stop=delivery-gate、
PreToolUse=pod-guard，均 `node ../../guards/*.mjs`）；②
`"agentsMdExcludes": ["**/leader-only.md"]` 排除组长专属根规则（缺则组员注入
`leader-only.md`；glob 写成 `**/*.md` 会连 `basic.md` 公约数一起吞掉，doctor 两项都判 ✗）。

## guards 面（delivery-gate 预算硬闸）

`orchestration/guards/delivery-gate.mjs`（Stop hook）除交付格式外，还按 basic.md §1
执法组员**自维护层**四条：`AGENTS.md` ≤8192B；自建 rules 禁 `trigger: always_on`
（charter 豁免）；自建 rules 与自建 skills 的 `description` 行合计各 ≤8192B；
每个自建 `SKILL.md` ≤8192B。违规 exit 2 + stderr 给出路径/实测字节/上限/整改动作，
格式与预算两类问题**合并成一轮**报（否则格式退回吃掉唯一一次拦截）。口径为落盘字节
（CRLF 计税），与 `wc -c` / `find -size +8192c` 一致。

- **部署副本豁免**：`PYSCI_DEPLOYED_SKILLS`（runner 注入，`;` 分隔）里的技能目录不算
  自建；该变量缺失时自建技能那一支整体不执法——分不清自建与副本时卡组员不如不卡。
- **`stop_hook_active` 优先**：循环防护为真时一切放行（拦一次即止，兜底靠 doctor 与
  pre-commit 真本侧）。
- **fail-open 会吞掉 guard 自己的 bug**：预算段包在 try 里，异常即静默放行。改 guard
  必须先跑 `tests/skills/orchestration/test_delivery_gate_budget.py`（node 跑真 guard）；
  ESM 顶层 `const` 有 TDZ，`run()` 只能放文件末尾。

## git 纪律

- commit：cz 规范（feat/fix/docs/refactor/test/chore + scope）；一笔提交一个主题；
- pre-commit 钩子全过（ruff 自动修复后需重新 add 再 commit——被钩子改写不算失败）；
- **push**：每次 commit 后尝试一次 `timeout 25 git push origin main`（best-effort，
  用户裁决 2026-10-10：校园网偶尔可直连 GitHub，不开常驻代理）——忽略一切失败
  （网络/认证/超时），不重试、不设代理；**超时是硬要求**（挂起会阻塞 drain）；
- 禁止 --no-verify；禁止 reset --hard/force 类破坏性操作，除非任务书明确授权；
- 合并前先 `git status` 检查工作区；发现非本任务的未提交改动，保留并在交付中报告。

## 巡检清单（doctor 之外的人工项）

- 台账：`orchestration/state/ledger/deliveries.jsonl` 中 ctx_ratio 持续 >60% 的成员
  → 提请组长归档轮换；
- transcript 抽查：pod 键目录（`~/.qoder-cn/projects/<pod键>/`）会话 jsonl 中检索
  pod-guard 拦截记录，核对其后有无 Bash 绕行迹象（诚实条款审计端）；
- AGENTS.md 体量：>8KB 的成员提请其自行精简（蒸馏纪律）。
