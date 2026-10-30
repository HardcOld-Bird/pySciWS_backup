---
name: bencher
description: bencher 组员的测量规程：pysci-bench 命令面、套件案卷写法、基线捕获与副本 A/B 机制、二元判决口径、清理与自查清单。bencher pod 会话中被派发测量任务时使用。
---

# bencher 测量规程

角色边界与权限面见 charter（`.qoder/rules/charter.md`）；本文放**操作知识**。
专属 CLI：`pysci-bench`（run / capture / ab / baseline / report / replica）。
不使用 `pysci-orch`（组长专属：队列与 worker 生命周期在他那边），不使用 `pysci-dev`。

## 命令面（全部复用 orchestration 共享库，本 CLI 只做装配与呈现）

```bash
pysci-bench run --all                     # 全套案，台账行 runner=bencher
pysci-bench run --case fig-basic --label after-<commit> --tier minimum
pysci-bench baseline [--label L]          # 回填基线快照（默认只取 bencher 通道）
pysci-bench report                        # 读数表 + 每案基线捕获状态
pysci-bench capture --case fig-basic --ref HEAD   # 基线捕获（见下）
pysci-bench ab --ref-a <旧> --ref-b <新> --cases fig-basic,lit-arxiv --label L
pysci-bench replica                       # 残留测量副本自查（清理纪律）
```

退出码是判决的一部分：`run` 0=全通过 / 1=有案失败 / 2=案卷或用法错；
`ab` 0=改善 / 1=未改善 / 2=用法错或副本残留 / 3=inconclusive。**判决写进交付正文**
（命令 + 实际读数 + 二元结论），不许只说「跑通了」。

## 案卷（`state/evals/harness/<case-id>.md`，归你所有）

头部标记行 + `## 任务书` 正文，解析器读的是标记行不是散文：

```markdown
member: figure
tier: minimum
max_turns: 16
timeout_s: 900
dirs: data/research/eval_scratch/fig-basic
产物: data/research/eval_scratch/fig-basic/eval_band check=figure-audit
断言: ^(图已生成)
说明: 案卷意图（人读，不参与判决）
## 任务书
（派发正文：可证伪的产物与断言）
```

头部是 `键: 值` 标记区，**未知键当场抛错**（拼错的判据等于没执行），键表见
`orchestration` 技能的 `references/harness-eval.md`；案 id = 文件名 stem，不写进头部。

- **`dirs:` 产物每轮清空**：残留会让组员「复核现成的」而不是「生产」，测到的是复核成本。
- 断言锚点（图路径 / arXiv id / 交付块内容）**先实测取确定性结果**再写进案内，不凭印象。
- 判据必须**可由组员正常达成**：历史 blocked 案改编成正向断言，否则测的是案卷的苛刻而非
  harness 的能力。
- `max_turns` 取「最小可达成量 ×1.4」。贴线（turns 用满）的读数不进结论——先放宽再测。

## 基线捕获与副本 A/B（E4 定案的机制）

`capture` = **commit 记录 + gitignored 面归档**，不建常驻 worktree：tracked 面 git 本来就
保得住，常驻 detached worktree 只是多占一份盘且会随 refs 移动失真；真正**不可再生**的是
各 pod 里 gitignored 的组员面（自建 rules/skills 等），所以归档只收它们
（`state/evals/baselines/<case>/manifest.json` 入库，物理归档不入库）。

`ab` = 两 ref 各 materialize 一份 detached worktree（`.qoder/worktrees/bench-<label>-<side>`）
→ 各跑案内案 → `eval.compare` 出二元判决 → **用后即弃**（`finally` 里 destroy + prune）。
副本重锚原理：子进程带 `PYTHONPATH=<副本>/src`，`pysci.paths` 就近上溯找到副本根的
`pyproject.toml`，于是 state/pods/技能真本全是副本的；`.venv` 与 `.env` 由
`PYSCI_VENV_SCRIPTS` 与复制动作补回。细节与坑（为何还原先于 sync、标记带为何存在）
见 `references/bench-ops.md`。

## 测量纪律（§0.5 无平局）

- 只有三个出口：**通过/失败**、**改善/未改善**，以及 **inconclusive**。inconclusive 不是
  平局，它意味着**测量本身不可信**，必须附证据并标 `escalate=twin`（taste 未启用则挂起留痕）。
- **窗口期污染是头号不可信来源**：`ref_a..ref_b` 之间若另有碰过 harness 面的提交，读数差
  是那几个提交的合力，任何「改善」都是错的因果。工具默认停手（`--force` 才照跑，慎用）。
- 一轮只跑一次，不并发；压测行落 `state/ledger/evals.jsonl`（gitignored）且带
  `runner=bencher`，与组长抽检通道（`runner=leader`）永远分得开。
- 交付里若发现设施缺陷 → `<infra_suggestion kind="defect">` 附复现命令与实际输出，
  **不要自己去修** src/guards/技能真本（那会造成无人审计的并行写面）。

## 收尾自查（合并/交付前）

```bash
git worktree list | grep -c 'bench-'   # 必须 0
git status --porcelain                 # 不得多出与本跳无关的脏面
pysci-bench replica                    # 残留副本读数
```
