# 任务书 orch-auto-wake

（派发时间：2026-10-10T07:32:00+08:00）

# 任务书：orch 自动唤醒 devops 消化 backlog（元基础设施，插队执行——理由：让全队列自动化）

## 背景与目标

现状：`orch approve` 只把建议入 backlog，devops 需组长手动 `dispatch devops
--from-backlog` 才消费——与 README §6 设计意图（批准后自动、后台、不阻塞主线）不符。
目标：approve 后 orch 自动以**分离后台进程**唤醒 devops，FIFO 串行消化整个 backlog，
组长零阻塞、零轮询。

## 功能规格

1. **新内部命令** `pysci-orch _drain-devops`（下划线前缀=内部命令，help 中隐藏或标注）：
   - 循环：`backlog_take_first()` → 生成任务书 → `do_dispatch("devops", ...)` →
     销账（复用现有 --from-backlog 逻辑，可重构提取共用函数）→ 取下一项，直到队列空；
   - 每项完成后把关键结果（backlog id、commit hash 若交付中有、耗时）追加到
     `orchestration/state/devops-runs/<启动ts>.log`；全部完成后写 `<启动ts>.done`
     （JSON：处理项数、各项 id 与结果、总耗时）；
   - 单项 dispatch 失败（run_failed/blocked）：记录后**跳过该项**（标记 backlog 条目
     status=needs_leader，不再重试），继续下一项——一项卡住不得阻塞全队列；
   - 运行期间持有锁（见 3）。
2. **approve 自动唤醒**：`orch approve <id> --note ...` 成功入队后，默认检查锁 →
   空闲则 spawn 分离子进程运行 `_drain-devops`：
   - Windows 分离进程：`subprocess.Popen([...], creationflags=DETACHED_PROCESS |
     CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW, stdout/stderr → <run>.log,
     stdin=DEVNULL)`；命令用 `sys.executable`（或 uv run 等价形式，注意从任意 cwd 可用）；
   - approve 输出追加一行：「devops 已后台唤醒（日志 orchestration/state/devops-runs/<ts>.log）；
     结果见 orch status / 该目录 .done 文件」；
   - `--no-wake` 旗标跳过唤醒；reject 不唤醒；
   - 锁忙时输出「devops 正在处理队列，本项将由当前 drain 循环接手」，不重复 spawn。
3. **锁**：`orchestration/state/devops.lock`（JSON：pid、started、run_log）。
   - 获取：原子写入（先写 .tmp 再 os.replace + 已存在时判活）；
   - 判活（Windows）：`tasklist /FI "PID eq <pid>"` 或等价；进程死但锁在 → 视为
     stale，接管；锁龄超过 timeout 上限（如 6h）也判 stale；
   - `_drain-devops` 启动时取锁、结束（含异常）释放（try/finally）。
4. **status 增强**：显示 devops worker 状态（running: 当前项 / idle）、最近一次
   run 的 .done 摘要（若有）、needs_leader 条目提醒。
5. **合并安全**：drain 循环派发 devops 的任务书模板追加一句：「合并前 git status
   检查——若 main 工作区存在会被本次合并触碰的未提交改动，交付 <blocked> 说明，
   等待组长清理；无关的未提交改动可照常合并」。

## 文档同步

- README §6 数据流更新（自动唤醒 + needs_leader 分支 + 锁语义）；
- orchestration 技能真本（orchestration/skills/orchestration/SKILL.md）命令面与
  「改进循环」节同步；改完 `pysci-dev sync` 部署；
- backlog 条目 20261010-orch-auto-wake-devops 由组长登记（你交付后它应显示 done——
  注意：本任务书就是该条目的实施，交付中注明 backlog id=20261010-orch-auto-wake-devops
  以便销账；但该条目不在 pending 队列中，drain 不会碰它，由组长手动销账）。

## 验证要求（交付中给证据）

1. 单元级：锁的获取/stale 接管/释放路径（可用脚本直接调用锁函数模拟）；
2. 集成级（worktree 内）：构造一条假 backlog 条目（内容=「向 bench/wake-test.txt
   写 ok 并 commit 到 worktree 分支后放弃合并」之类的无害任务，或 --dry 机制），
   走 approve→自动唤醒→drain→.done 全链，展示 run 日志；
3. 防并发：锁忙时第二次 approve 不 spawn 第二 worker（展示输出）；
4. `uv run pytest tests/ -x -q`（PYTHONPATH 覆盖法）全绿 + ruff 全绿。

## 约束

worktree 隔离实施；cz commit（建议 `feat(orchestration): approve 自动唤醒 devops
后台消化 backlog`）；不 push；交付 <result> 含改动清单/验证证据/commit hash +
backlog id 注明。
