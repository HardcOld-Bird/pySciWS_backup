# 后台 worker 与唤醒通道（drain / watch / bench-*）

SKILL.md 的命令面只留一行索引；本文放**为什么**与**踩过的坑**。

## 1. 两条 worker 通道，一套锁原语

| 通道 | 队列 | 锁/日志面 | 唤醒方式 |
|---|---|---|---|
| devops drain | `state/backlog.json` | `devops.lock` / `devops-runs/` / `devops.cancel` | `approve` 自动派生 + `orch watch` 事件即时唤醒 |
| bencher（§5.5，E4） | `state/bencher/queue.json` | `bencher.lock` / `bencher-runs/` / `bencher.cancel` | **择机查看**：终态进 `orch status` 的「待查看」面，不即时打断 |
| inspector（§6，E5a） | `state/inspector/queue.json` | `inspector.lock` / `inspector-runs/` / `inspector.cancel` | **不需要唤醒**：闲时自主激活，拉起内嵌在 status/approve/dispatch |

三条通道共用 `drain.WorkerPaths`（锁判活、stale 回收、墓碑、`.done` 身份三元组、协作停止）——
单 worker 互斥由锁保证，故获锁一刻仍在 `in_progress` 的条目必是上一任被杀时的孤儿，
`reset_orphans()`/`reset_orphan_backlog()` 就地复位。**新通道不再另写事故硬化逻辑**；队列形状
同理，增删/取项/孤儿复位/裁剪都在 `workq.py` 一处（防双源漂移）。

**孤儿复位管五面**（进程面 20261012-184000-leader，抢救面 20261012-234000-leader）：锁、条目、
worktree 名、进程之外还有**进度**——回收不再默认销毁半成品：脏面（未合并/证据不足同理）→ 落
`wip(salvage)` 提交并保面保分支、队列自动接线续作；确认干净才 `remove --force` + `branch -D`。
判据与形状见 `salvage-reclaim.md`。

第四面**进程**：worker 被杀 ≠ 它派生的 headless 会话被杀（独立进程树），那个会话仍在往原
worktree 写，重派同名即双写。`dispatch` 派生瞬间把会话 pid 记进运行态会话池，drain 按任务书名
对上 pid，**判活 + 核对 argv 里的 sid** 两闸都过才 `taskkill /T /F`；判不准（pid 复用、杀完仍活、
`remove --force` rc=0 但目录还在）就**绝不对旧 pid 动手**——条目升一代 `<id>-r<N>` 换隔离面，
新名写进重派任务书。日志痕迹：`终止孤儿会话进程`/`已不是会话…不动手`/`异名重建`；
`.done` 的 `cleaned`/`regenerated`/`salvaged` 三键。`--dry` 一条都不执行。

## 2. watch 挂载：自动为主，手动兜底（机制全文 README §4.5 v2）

- **自动**：`Stop` 钩子每回合先过**唤醒权归属闸**（`session_id`：无 owner 认领、异会话不抢、
  stale 接管，README §4.5），过闸者挂 `orch watch`（命中 `exit 2`、超时/异常 `exit 0`）。**无需手动挂**。
- **手动兜底（仅钩子不可用时）**：Bash `run_in_background` 启动，且**必须是该后台调用的唯一顶层
  命令**（`&`/`&&`/管道嵌入即被父任务连坐杀死——2026-10-12 夜航停摆 6h 的成因）。
- 漏挂/断唤的账由下次 `orch status` 补账：「忘了挂」≠「事件丢了」。
- **测试/诊断性挂载前须确认无 live 挂载或先停**：`arm()` 覆写 `state/watch.json`，会抹掉
  在等那任的心跳与黑匣子；`recent`/`exit_reason` 是脱靶复盘的现场，status 读出其末帧。
- 事件类型：基线差分六类 `QUEUE-COMPLETE`/`NEEDS-LEADER`/`QUOTA-STOP`/`CHANNEL-STOP`/`CRASH`/
  `SALVAGE`（两类 STOP＝系统性故障全局停止：充值 vs 等渠道）；＋统一事件流 `state/events.jsonl`
  直发的 `JOB-DONE`/`STEP-REVIEW-DONE`/`QUEUE-EMPTY`/`TASKBOOK-NEW`（按水位线读出即唤醒）。
  醒来先 `orch events drain` 排空（幂等＝合并唤醒）→ 逐 plan `orch plan advance`。
  `BENCH-DONE <run>` 可选挂载、默认不挂（§5.1-B）。

## 3. bench-* 使用要领（组长侧）

```bash
uv run --no-sync pysci-orch bench add run --cases fig-basic,lit-arxiv --label after-<commit> --start
uv run --no-sync pysci-orch bench add ab  --ref-a dd6e37b --ref-b HEAD --cases fig-basic --label L
uv run --no-sync pysci-orch bench add capture --case fig-basic --ref HEAD
uv run --no-sync pysci-orch bench add check --brief "<核查任务书正文>" --requested-by leader
uv run --no-sync pysci-orch bench queue          # 看在途/待查看；--all 看历史
uv run --no-sync pysci-orch bench seen <id>      # 看完摘掉（否则一直挂在 status 上）
uv run --no-sync pysci-orch bench stop           # 协作停止：条目间隙干净退出，不 kill
```

- **入队即返**：`bench add` 不阻塞；`--start`（或随后 `bench start`）派生分离 worker，队列空自毁。
- 待查看面是**判决入口**：`ab` 项的 `改善/未改善/inconclusive`、`run` 项的 `通过 n/m` 都在这一行；
  细节看 `state/evals/ab/<label>.json` 与 run 日志。
- `inconclusive` 恒带 `escalate=twin`：不是平局而是「测量不可信」——先看污染提交（`pollution`），
  换 ref 对或把混杂提交并入被测面，**不要**用 `--force` 把不可信因果当结论。
- **系统性故障**（`dispatch.SYSTEMIC_KINDS`，三通道同族处置）：worker 全局停止、剩余条目保持
  pending 不烧账，`.done` 附 `stopped=<kind>`；恢复动作**不同**——`quota_exhausted`=充值/换渠道；
  `channel_error`=渠道服务面故障，等恢复即可（2026-10-12 实测：BYOK 坏时 CLI 不报 credit 文案，
  而是合成 `model="<synthetic>"` 回合说 "Custom model service error…"，故判据取**固定文案 ∧
  会话档末轮形状**合取，单特征会误伤复述该句的复盘任务）。恢复后条目照常 FIFO 续跑
  （drain 由下次 `approve` 接管，压测队列由 `bench start`）。
- **预设席位回退链**（用户 2026-10-12 链化；全文 README §3.5）：`[model] preset` 选预设，角色
  （main/pro，`--model-tier` 覆盖）取该预设**有序链**，当前席不可用即沿链一席一跳降级；**全链
  耗尽**才保留系统性 kind 全局停止。预设不可激活 → dispatch 拒发、drain **拿锁前**拒开工
  （`skipped=preset_unavailable`）。**例外：`channel_error` 先自救**（同会话 resume 大概率即愈，
  双闸 `same_seat_retry` 10 + `budget_s` 1800s 都过才换席，付费席降后手；`quota_exhausted`/启动即
  失败不自救；自救**不入**回退事件面，次数走台账 `same_seat_retries`）；非系统性失败同席重试一次。
  **两条面**：调用者面 `[!] 档位回退：… 已切换 <新席>` + 观察面 `state/channel-events.jsonl`。
  **粘滞窗**（`skip_window_s` 默认 3600s）：坏席进窗跳过、过期复探、跑通即销；窗内全链按失败
  最早一席强探不停摆。换席后实际渠道看台账 `model`。
- `approve` 时选 **M**（基线捕获 + 实施后副本 A/B）的判决包接口：E5a/E5b 未实施，
  目前由组长手动 `bench add capture`/`bench add ab` 完成同一效果（E4 只交付运行职责）。

## 4. 压测隔离面（为什么不会污染生产）

`eval.run_case` 两头夹：任务书附 `EVAL_HOP_NOTE`（禁改自维护层）+ 前后拍该 pod 跟踪脏面快照，
只还原本跳新增者（路径进台账 `reverted`）。副本 A/B 更严格：测量在 detached worktree 里跑、
用后即弃，读数经 `ROWS_BEGIN/ROWS_END` 由父进程代写主树台账（`runner=bencher`）。反复出现
`reverted`＝案内任务书该收紧，别加重事后还原。

## 5. 唤醒未达排障清单（定序，别跳步）

「队列空了却没唤醒」按序证伪：**① listener 有没听见**（`watch-arm.log` 有无「命中事件」）→
**② 消费账撞车**（`watch.json` 的 `consumed` 与当前流身份是否同型）→ **③ 水位倒挂**
（「已排空至 X」> 流 max seq＝被回卷）→ **④ 流被覆盖**（`events.jsonl.bak-*` 对照条数/序号）→
**⑤ listener 活死**（`watchdog-runs` 末条心跳 age / status watch 段）。完整读命令与「迟到补报
≠ 聋期」分档见 `wake-troubleshoot.md`；水位倒挂由 `events.status_lines()` 在 `orch status` 点名。
