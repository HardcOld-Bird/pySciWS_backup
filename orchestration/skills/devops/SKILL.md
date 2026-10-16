---
name: devops
description: devops 组员的作业规程：worktree 隔离实施、PYTHONPATH 覆盖测试策略、技能部署执行、pysci-dev 工具面、git 纪律与巡检清单。devops pod 会话中被派发基础设施任务时使用。
---

# devops 作业规程

角色边界与红线见 charter（`.qoder/rules/charter.md`）；本文放**操作知识**。
专属 CLI：`pysci-dev`（见下命令面）。
不使用 pysci-orch（组长专属）。

## worktree 作业规程（一切代码改动）

```bash
# 1. 开 worktree（从最新 main）；<任务名> 用法见下「worktree 命名约定」
pysci-dev worktree add <任务名>        # = git worktree add -b wt-<任务名> .qoder/worktrees/<任务名>
ROOT="$(git rev-parse --show-toplevel)"   # 在 main/pod 侧执行 → 主树根
WT="$ROOT/.qoder/worktrees/<任务名>"

# 2. 在 worktree 内改代码（Edit/Write 用 $WT 下路径；**不要 cd 进 worktree**）

# 3. 测试 —— PYTHONPATH 覆盖策略（复用主 .venv；worktree 代码优先于 editable 主树）：
#    **前提=cwd 在 worktree 之外**；已 cd 进去时 uv 就地建空 <wt>/.venv 缺依赖即崩，
#    那时改用主解释器 "$ROOT/.venv/Scripts/python.exe" -m pytest。细节见 references/worktree-merge.md
PYTHONPATH="$WT/src" uv run --no-sync pytest "$WT/tests" -x -q

# 3b. pyproject 依赖变更才 uv sync --project "$WT"（分钟级），之后用 "$WT/.venv"

# 4. 合并回 main（先在 worktree 内 commit；cz 规范、pre-commit 全过、禁 --no-verify）：
pysci-dev worktree merge <任务名>   # 交集检查 → cp 运行态 → merge --no-ff → 会话数自检
                                    #   → **成功自动摘面**（失败/blocked 保留；--keep 留检视）
                                    #   交集非空 rc=2 列文件（merge-base 起算）；--no-cp-state 不搬运行态
# 5. 推送（跨平台超时 + 成败留痕；best-effort）
pysci-dev push
# 6. 清理：merge 已自动做；遗留面 `pysci-dev worktree clean [--apply]`（未合入拒绝；脏面/未注册 --discard）
```

**主树同步入口/依赖一律 `pysci-dev venv-sync`**（裸 `uv sync`/`uv run` 遇运行中 exe 会先删光
其余入口，半坏态静默；见 `references/uv-venv-sync.md`）。

worktree 不含 gitignored 未跟踪重数据；需真数据时以绝对路径只读引用主树。

**worktree 命名约定**：drain/backlog 派发用**条目 id**（或异名重建后的 `<id>-r<N>`）作
`<任务名>`（孤儿回收五面含**进程**/**进度**，据此定位；细节见 `references/worktree-merge.md` §3）。

**批量消化任务书**（drain 组包）：任务书可能含**种子**（必做）
+ **组包菜单**。**自主选取 ≥1 项**（必含种子，可只吃种子），合并到**同一 worktree**（名
= 种子 id）实施。交付时 `<result>` 以 `backlog id=<id1>,<id2>,…` 列**本次完成全部 id**
（必含种子），orch 逐 id 销账；未完成的**不要**列入（留 pending 下轮）。种子受阻照常
`<blocked>`（组包不连坐）。

## 技能部署（真本 → 副本）

真本在 `orchestration/skills/<name>/`（git 跟踪；manifest 同目录 `manifest.toml`）；
改完 `pysci-dev sync` 部署、`sync --check` 查漂移。**红线**：只碰 manifest 声明的部署名、
不删除成员自建技能、不手改任何 pod 内的部署副本。分册搬运/EOF 窗口的细则与逐项语义见
`references/dev-cli.md`。

## pysci-dev 命令面

```
pysci-dev doctor | sync [--check] | worktree add|merge|remove|list|clean
pysci-dev push [--timeout S] | mdgen-check <paths…> | doc-faces <paths…|--changed>
pysci-dev venv-sync | probe | skilltax
```

逐项语义与旗标见 `references/dev-cli.md`。

## 新 pod 脚手架（只读层入库面）

每 pod 仅 5 个文件入库（`.gitignore`/`.qoder/settings.json`/`.qoder/mcp.json`/
`.qoder/rules/charter.md`/`AGENTS.md`）；`bench/`·`outbox/`·`.qoder/skills/`·`inbox/`
须在**合并后于 main 上** `mkdir` + `sync` 再生。settings 模板必带 hook 接线 +
`agentsMdExcludes`（恰好排除 leader-only）。逐项、零技能 pod 裁定与注册面见
`references/pod-scaffold.md`。

## harness 预算三层执法（basic.md §1）

三档上限（每文件/每类合计 ≤8192B）由三层兜住，口径统一为**落盘字节**（CRLF 计税，
同 `wc -c` / `find -size +8192c`）：

| 层 | 位置 | 管谁 |
|---|---|---|
| 组员侧硬闸 | `guards/delivery-gate.mjs`（Stop） | 自维护层四条：`AGENTS.md` 尺寸、自建 rules 禁 `always_on`、rules·skills 的 `description` 合计、自建 `SKILL.md` 尺寸 |
| 真本侧硬闸 | pre-commit `harness-budget`（`guards/budget_check.py`） | 真本 `SKILL.md`/`references/*.md` 与根 `basic.md`·`leader-only.md` 每文件 |
| 观测 | `pysci-dev doctor` 三档审计 | 全量扫描 + 台账，唯一能看见「会话中途死掉留下的膨胀」的一层 |

两层硬闸与 doctor 均已给**余量读数 + >95% 预警**（未超限即提示、不拦截）：把「到顶即拒」
补成「逼近即知」，避免「加一句、删一段」靠临时腾字节。

改任一层、加新 guard、或处置超限 references 前，先读
`references/budget-layers.md`（豁免/TDZ 教训/Stop 探针/分册次序）与
`references/guards-in-force.md`（fail-open 留痕+活体探针+元哨兵）。

经验底账（事故经过与判据；AGENTS.md 只留约束）：`references/lessons-incidents.md`。

## git 纪律

- commit：cz 规范（feat/fix/docs/refactor/test/chore + scope）；一笔提交一个主题；
- pre-commit 钩子全过（ruff 自动修复后需重新 add 再 commit——被钩子改写不算失败）；
- **push**：见上 worktree 规程第 5 步（一次、限时、忽略失败、不重试、不设代理）；
- 禁止 --no-verify；禁止 reset --hard/force 类破坏性操作，除非任务书明确授权；
- 合并前的「main 脏面 ∩ 本次改动」由 `pysci-dev worktree merge` 内建（手算须按 merge-base
  起算——`main..HEAD` 在分支落后时会把别人的提交也算进来）；
- 发现与本单无关的未提交改动：保留、不替组长清理，交付中报告。

## 巡检清单（doctor 之外的人工项）

- 台账：末跳 `ctx_ratio` >60% 已机械化立案（`inspectd.check_context_pressure`）→ 组长
  `orch sessions --archive <sid> --distill` 归档轮换；
- **压测面已移交 bencher**（E4）：案卷与基线区归他，你只维护 checks 注册表与
  `orch bench eval` 抽检面；动了 harness 面后不自跑压测，交付里写明「需复测/重取基线」，
  由组长 `orch bench add` 入队（分界全文见 charter）。
- transcript 抽查（诚实条款审计端）：`~/.qoder-cn/projects/<pod键>/*.jsonl` 里检索
  pod-guard 拦截记录，核对其后有无 Bash 绕行迹象；
- AGENTS.md 体量：>8KB 的成员提请其自行精简（蒸馏纪律）。
