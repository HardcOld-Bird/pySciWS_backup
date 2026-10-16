# 任务书：加急——空终端风暴根治（窗口泄漏 × 看门狗空转刷屏）

**紧急度：用户电脑当前每分钟被 ~28 个空终端窗口夺焦点，持续 30 分钟+，优先于一切在途改进。**

## 成因（组长已用进程取证实锤，2026-10-15 17:1x）

50ms 进程采样（ctypes Toolhelp32）实录：
- `pythonw.exe pid 21344` = `orch _watchdog-loop --run-log …/watchdog-runs/20261015-153304.log --interval 60`
  （启动于 15:33:04），**每 tick 连发约 28 个 `git.exe`**（实测节奏：单波 7+ 连发、
  `git.exe<-git.exe`/`sh.exe<-git.exe` 链、`conhost.exe/OpenConsole.exe` 伴生）；
- **pythonw 是 GUI 子系统进程、无控制台** → 其派生的 console 子系统程序（git/bash）若不带
  `CREATE_NO_WINDOW`，Windows 会分配**新可见控制台**；git 秒退 → 窗口闪现夺焦点。
  用户现象「每分钟连续约 28 个空终端、没显示标题就消失」与 tick 扫描波形完全吻合；
- 同源闪窗还见于 `_queue-worker`（python，无控制台）提交链的 git 串与 pre-commit
  fixer 串（trailing-whitespace-fixer 等，同轮采样实录）。

对照现状：`runner.py:319`（组员会话派生）与 `drain._spawn_detached` 已带隐藏旗标
（2026-10-14「空终端根治」只修了这两处）；**其余 20+ 处 `subprocess.run` 裸调用全漏**：
checks.py:69、drain.py:184/968、sessions.py:222/241、statecommit.py:32、sync.py:107、
registry.py:1181/1372、replica.py:88/374、verdict.py:195、jury.py:217、eval.py:345、
procs.py:35、以及 `unmerged_worktree_alerts`/`reclaim_orphans` 内部的 git 调用（以实测为准，
grep `subprocess.run\|Popen` 全量核对）。

## 修复要求（双层）

### ① 窗口层（必须，根治闪窗）
- 统一隐藏策略：建议在 `procs.py` 提供单一 `run_hidden()`（win32 恒带
  `CREATE_NO_WINDOW`，POSIX 不变）包一层，**所有** spawn 点（含 git 调用）改走它；
  同一能力不留两份口径（runner/drain 现有旗标并入该口径）。
- 判据：无论父进程有无控制台，派生 console 程序均不产生可见窗口。

### ② 体量层（必须，消灭空转刷屏）
- 看门狗 tick 的巡检面（`unmerged_worktree_alerts`、`reclaim_orphans`、inspector tick
  内部命令）**批量化/节流**：例如 `git worktree list --porcelain` + `git for-each-ref`
  一两次调用取代 per-worktree 多命令；或结果按 (branch, head) 缓存、无变化的 tick 复用。
- 目标：**空闲 tick 的 git 派生 ≤ 2 次**；有变化时才多花命令。

## 验收预期
1. 修后空闲期用进程采样（50ms Toolhelp32 或等价）观察 ≥2 分钟：
   无「git/conhost/OpenConsole 连发波」，空终端不再闪现；
2. 功能不回退：worktree 留一个未合并提交 → 看门狗仍能检出并 NEEDS_LEADER 出声
   （去重语义不变）；队列自愈、inspector tick 照常；
3. 全量 pytest 绿；新增回归：patch 记录 spawn kwargs，断言 win32 下所有 spawn 点
   均带 CREATE_NO_WINDOW（防再漏）。

## 白名单（写权限）
- src/pysci/skills/orchestration/tools/
- tests/
- （若需动 hooks/巡检面说明文档：orchestration/skills/orchestration/references/）
