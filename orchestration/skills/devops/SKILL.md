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

注意：worktree 检出不含被 gitignore 的未跟踪重数据（data/research 的 simulation/storage）；
需真实数据验证时用绝对路径只读引用主树。

**worktree 命名约定**：drain/backlog 派发用**条目 id** 作 `<任务名>`（take_first 记
`worktree=id`）。会话中途死亡的孤儿条目，下次 drain 回收据 `wt-<id>` 名
`git worktree remove --force` + `branch -D` 清理（否则重派 `worktree add` 撞名）。手动/
临时派发可用任意名——不属孤儿、不自动清理（防误删并发作业）。

**批量消化任务书**（drain 组包，backlog 20261010-batch-digest）：任务书可能含**种子**（必做）
+ **组包菜单**。**自主选取 ≥1 项**（必含种子，可只吃种子），合并到**同一 worktree**（名
= 种子 id）实施。交付时 `<result>` 以 `backlog id=<id1>,<id2>,…` 列**本次完成全部 id**
（必含种子），orch 逐 id 销账；未完成的**不要**列入（留 pending 下轮）。种子受阻照常
`<blocked>`（组包不连坐）。

## 技能部署（真本 → 副本）

- 真本：`orchestration/skills/<name>/`（git 跟踪）；manifest：同目录 `manifest.toml`；
- 改完真本执行 `pysci-dev sync`（与 pysci-orch sync 同引擎）；`sync --check` 查漂移；
- `tree_hash` **行尾无关**（哈希前 CRLF→LF，2026-10-10 修 backlog 20261010-003127-devops）：
  worktree 内 sync 提交的 `skills-deployed.json` 对 main 即正确，无需合并后重跑；doctor
  「部署台账巡检（sync --check）」哨兵记录哈希漂移；
- 红线：只碰 manifest 声明的部署名；**不删除成员自建技能**；不手改部署副本。
- **真本 references 超 8KB 的处置**：按 `##` 主题拆 `<stem>-0N-<slug>.md` 分册（逐字
  搬运、原名留分册索引），所属 SKILL.md 索引行标 ``*`` + 节首加图例行。机械拆分与验证
  次序（含 pre-commit 会先改写文件的坑）见
  [references/budget-layers.md](references/budget-layers.md) §4。
- **机械生成/改写 markdown 的 EOF 窗口**：`pysci-dev mdgen-check <paths…>` 提前把
  pre-commit（trailing-ws + end-of-file-fixer）改写**做完**并打印 pre/post sha256；
  内容等价性断言以 **post_sha256**（钩子终态）为基准，非生成前字节。首提必过。

## pysci-dev 命令面

```
pysci-dev doctor [--pod <id>]   # pod 健康巡检：hooks 接线/leader 规则排除/charter/AGENTS.md 体量/积压
                                 # + 部署台账巡检（sync --check 漂移）+ harness 预算三档审计（≤8192B/文件）
pysci-dev sync [--check]        # 技能部署
pysci-dev worktree add|remove|list [<name>]
pysci-dev mdgen-check <paths…>  # 提前做完 pre-commit 对 md 的改写、打印终态哈希
```

## 新 pod 脚手架（只读层入库面）

每 pod 仅 5 个文件入 git：`.gitignore`、`.qoder/settings.json`、`.qoder/mcp.json`、
`.qoder/rules/charter.md`、`AGENTS.md`；`bench/`·`outbox/`·`.qoder/skills/` 被 pod
`.gitignore` 排除，`inbox/` 空目录不跟踪——后三者须**合并后在 main 上** `mkdir` +
`sync` 再生。settings.json 模板必带两件事：① 两个 hook 接线（Stop=delivery-gate、
PreToolUse=pod-guard，均 `node ../../guards/*.mjs`）；②
`"agentsMdExcludes": ["**/leader-only.md"]` 排除组长专属根规则（缺则组员注入
`leader-only.md`；glob 写成 `**/*.md` 会连 `basic.md` 公约数一起吞掉，doctor 两项都判 ✗）。

## harness 预算三层执法（basic.md §1）

三档上限（每文件/每类合计 ≤8192B）由三层兜住，口径统一为**落盘字节**（CRLF 计税，
同 `wc -c` / `find -size +8192c`）：

| 层 | 位置 | 管谁 |
|---|---|---|
| 组员侧硬闸 | `guards/delivery-gate.mjs`（Stop） | 自维护层四条：`AGENTS.md` 尺寸、自建 rules 禁 `always_on`、rules·skills 的 `description` 合计、自建 `SKILL.md` 尺寸 |
| 真本侧硬闸 | pre-commit `harness-budget`（`guards/budget_check.py`） | 真本 `SKILL.md`/`references/*.md` 与根 `basic.md`·`leader-only.md` 每文件 |
| 观测 | `pysci-dev doctor` 三档审计 | 全量扫描 + 台账，唯一能看见「会话中途死掉留下的膨胀」的一层 |

改任一层、加新 guard、或处置超限 references 前，先读
[budget-layers.md](references/budget-layers.md)（豁免/TDZ 教训/Stop 探针/分册次序）与
[guards-in-force.md](references/guards-in-force.md)（fail-open 留痕+活体探针+元哨兵）。

## git 纪律

- commit：cz 规范（feat/fix/docs/refactor/test/chore + scope）；一笔提交一个主题；
- pre-commit 钩子全过（ruff 自动修复后需重新 add 再 commit——被钩子改写不算失败）；
- **push**：每次 commit 后尝试一次 `timeout 25 git push origin main`（best-effort，用户
  裁决 2026-10-10 校园网偶尔可直连 GitHub）——忽略一切失败（网络/认证/超时），不重试、
  不设代理；**超时是硬要求**（挂起会阻塞 drain）；
- 禁止 --no-verify；禁止 reset --hard/force 类破坏性操作，除非任务书明确授权；
- 合并前先 `git status` 检查工作区；发现非本任务的未提交改动，保留并在交付中报告。
- **状态分储**：registry.json=低频配置（git）/ sessions-runtime.json=会话池运行态
  （心跳每跳必变，gitignore）。语义见 references/registry-runtime.md。

## 巡检清单（doctor 之外的人工项）

- 台账：`orchestration/state/ledger/deliveries.jsonl` 中 ctx_ratio 持续 >60% 的成员
  → 提请组长归档轮换；
- transcript 抽查：pod 键目录（`~/.qoder-cn/projects/<pod键>/`）会话 jsonl 中检索
  pod-guard 拦截记录，核对其后有无 Bash 绕行迹象（诚实条款审计端）；
- AGENTS.md 体量：>8KB 的成员提请其自行精简（蒸馏纪律）。
