# worktree 作业细节：测试配方与合并前交集检查

`pysci-dev worktree` 的调用面在 SKILL.md，本文放**为什么这样判**与踩过的坑。

## 1. 测试配方（2026-10-09 定型，2026-10-12 补边界）

```bash
# cwd 必须在 worktree 之外（main 或 pod 目录）
PYTHONPATH="$WT/src" uv run --no-sync pytest "$WT/tests" -x -q
```

- **`PYTHONPATH` 覆盖**让 `pysci` 从 worktree 的 `src/` 导入（editable 主树让位于路径搜索），
  主 `.venv` 的依赖照用——不建第二套环境，秒级起测。
- **前提=cwd 在 worktree 之外**：`uv` 沿 cwd 找 `.venv`，已 cd 进 worktree 时它就地新建
  空 `<wt>/.venv`（2026-10-12 实测），缺 pyyaml 直接 import 崩。曾被误记成「此法必崩」，
  实为 cwd 问题。已 cd 进去时用主解释器，不依赖 cwd：
  `PYTHONUTF8=1 PYTHONPATH="$WT/src" "$ROOT/.venv/Scripts/python.exe" -m pytest "$WT/tests" -q`
- **副作用（可防亦可用）**：`pysci.paths.PROJECT_ROOT` 就近找 `pyproject.toml`，故在 worktree
  内跑 sync/drain 等，读写的是**worktree** 的 `orchestration/state/*.json`（backlog.json 被
  git 跟踪！）。实验后 `git checkout -- orchestration/state/…` 还原，别把无关脏面带进交付。
- **跨树自检认得这个配方**（backlog 20261012-094646-devops）：护栏拦的是「脚下在 worktree、
  代码面指外层 main」的事故形状；配方是反方向（脚下主仓、代码面指其下 linked worktree），
  判据 = `PROJECT_ROOT` 在脚下树之内 **且** `PROJECT_ROOT/.git` 是**文件** → 放行，**不再需要
  `--force`**。同理 `pysci-dev mdgen-check` 的相对入参与 pre-commit 的 cwd 一律按
  `PROJECT_ROOT` 解析——按 cwd 解析会拿**另一棵树**的字节当「钩子终态基准」（静默假基准）。
- 仅当 pyproject 依赖变更才 `uv sync --project "$WT"`（分钟级），之后改用 `$WT/.venv`。
- **跑套件不点着队列**（backlog 20261012-191512-devops）：`orch status`/`approve`/`dispatch`
  三入口内嵌巡检拉起，夹具级 patch 挡不住漏挂 `iso_state` 的用例——实测两个这样的用例真起
  过 `_inspect-worker`，run_log 落在当时所在 worktree 里，那把 `inspector.lock` 随后被 merge
  的 cp 面拖进 main。故真 `Popen` 之前还有一道咽喉闸：`drain._spawn_detached` 见
  `PYTEST_CURRENT_TEST`/`PYTEST_VERSION` 即就地返回（两变量被子进程继承，测试里跑真 CLI 也
  挡得住）。闸在派生面不在判定面：测试里 `spawned=True` 是空话，要断言 argv 就 patch 本函数；
  确需真派生的用例自清这两枚变量（`test_drain_wake.py` 备有反例，防闸焊死生产面而全绿）。

## 2. 合并前交集检查（backlog 20261012-082932-devops）

任务书里的「若 main 工作区存在**会被本次合并触碰的**未提交改动 → `<blocked>`」需要一把
可执行的尺子。`pysci-dev worktree merge` 已内建：**先于任何写操作**求交，非空即列文件、
rc=2，不 cp、不 merge。

### 判据：merge-base 起算，不是 `main..HEAD`

```bash
BASE=$(git merge-base HEAD wt-<任务名>)           # 分叉点
git diff --name-only "$BASE"..wt-<任务名>          # ← 本次合并真正触碰的文件面
git -c core.quotepath=false status --porcelain    # main 工作区未提交面（含未跟踪）
```

`git diff --name-only main..HEAD` **不可用作交集**：两点 diff 是「main 尖端 vs 分支尖端」，
分支落后 main 时把**别人新提交的文件**也算成本次改动。实测两度踩：某次 `main..HEAD` 报
8 文件，其中 `orchestration/state/backlog.json`、`registry.json` 我一次都没提交（main 前进了
两个提交），与 main 的脏 backlog.json 求交得**假阳性交集**，而 `git merge-tree` 实际 rc=0；
上一轮 P0 据此交出过错误的 `<blocked>`，白耗一轮。三点评测（merge-base / 两点 / merge-tree）
用可执行输出裁决，结论就是本节第一行那条。

### 工具行为边界

- **未跟踪目录**脏项（porcelain 里以 `/` 结尾）按前缀命中其下的提交文件——那种合并 git 本身
  也会以 "untracked working tree files would be overwritten" 拒绝，工具先把清单摊开；
- 重命名取**新**路径（被覆盖的是新路径上的工作）；非 `XY path` 形态的行跳过——宁可漏报也
  不把无关输出算成冲突；
- `merge-base`/`diff` 失败（rc≠0）→ **fail-open 跳过检查**并打印告警：算不出就堵死一切合并，
  等于把一个 usability 缺陷换成另一种误报，且 `git merge` 自身仍是最后一道拒覆盖的闸；
- 检查排在 `_cp_ignored_state` 与会话数校验**之前**：被挡下时 main 一个字节都没被动过，
  所以中止是零成本的，组长清理后重跑即可。

### 与本单无关的脏项

照常合并（交集为空即放行），交付时在 `<result>` 里报告其存在——不替组长清理别人的工作区。
运行态文件（`sessions-runtime.json` 等 gitignored）不进交集：它们由 merge 的 cp 阶段专门
搬运（见 `registry-runtime.md`）；**但 `*.lock` 永不跨树**（`dev.MERGE_STATE_EXCLUDE_SUFFIXES`）。
锁不是数据，是「谁在跑」的活体读数，归属判据 = 文件里的 pid + run_log 双锚；把他树锁 cp 进
main，main 就认到一把两头都不属于自己的锁——`orch status` 误报 worker 在跑，真在跑的 drain
反被这把假锁挡住不唤醒（实测两次合并各拖进一把 `inspector.lock`，人工删掉才解封）。锁的现场性
由判活面负责（`lock_status` 认墓碑/残骸），不属于合并面。

## 3. worktree 命名与孤儿回收

- **drain/backlog 派发**：`<任务名>` = 条目 id（take_first 记 `worktree=id`）。会话中途死亡的
  孤儿条目，下次 drain 回收据 `wt-<id>` 名定位残留面；名字不合规就定位不到，重派
  `worktree add` 还会撞名；
- **回收先抢救（第五面=进度，backlog 20261012-234000-leader）**：孤儿面**不再默认销毁**——有未提交
  改动 / 未合并提交 / 读数拿不准 → 自动打 ``wip(salvage)`` 提交、保面保分支、`source=salvage`
  续作单自动插队；`remove --force` 只对被证明干净的面开放。抢救提交**不是交付、勿据此销账**。
  判据与续作指令形状见 orchestration 技能 `references/salvage-reclaim.md`；
- **回收管四面**（backlog 20261012-184000-leader）：锁、backlog 条目、worktree 名之外还有
  **进程**——worker 被杀时它派生的 headless 会话活在独立进程树里，不随父进程退出，仍往原
  worktree 写。`dispatch` 在派生瞬间把会话 pid 记进运行态（`pid` 键的存活期=该跳在途期），
  drain 回收时判活 + 核对 argv 里的 sid 后终止进程树；**判不准就不开火**（pid 会被系统复用），
  该条目改走下面的异名重建；
- **异名重建 `<id>-r<N>`**：旧面没能确认干净（杀不掉 / pid 已易主 / `remove --force` rc=0 但
  目录仍趴在原地）→ 条目 `worktree_gen` 升一代，任务书直接把新名交给 worker。**世代号读磁盘
  事实**（已有的 `<id>-r*` 目录），不是计数器自增，所以别把名字改回旧名——那个目录里可能正
  有第二个写者；
- **同名目录 ≠ 你的面**：`worktree add` 撞见同名目录时会区分「登记在册」（继续用）与
  **残骸未登记**（旧会话句柄没放，remove 只是半途成功）。后者照提示换新一代名，**别在残骸目录
  里 `rm -rf`**，旧目录留给组长处置；
- **手动/临时派发**：可用任意名——它不属孤儿集合、不会被自动清理，这正是防误删并发作业的
  理由；
- 收口后自己清：`pysci-dev worktree remove <名>` 会连分支一起删；分支还有未合并提交时
  `branch -D` 的输出会提示，别把它当成可忽略的告警。
