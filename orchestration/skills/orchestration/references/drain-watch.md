# 后台 worker 与唤醒通道（drain / watch / bench-*）

SKILL.md 的命令面只留一行索引；本文放**为什么**与**踩过的坑**。

## 1. 两条 worker 通道，一套锁原语

| 通道 | 队列 | 锁/日志面 | 唤醒方式 |
|---|---|---|---|
| devops drain | `state/backlog.json` | `devops.lock` / `devops-runs/` / `devops.cancel` | `approve` 自动派生 + `orch watch` 事件即时唤醒 |
| bencher（§5.5，E4） | `state/bencher/queue.json` | `bencher.lock` / `bencher-runs/` / `bencher.cancel` | **择机查看**：终态进 `orch status` 的「待查看」面，不即时打断 |
| inspector（§6，E5a） | `state/inspector/queue.json` | `inspector.lock` / `inspector-runs/` / `inspector.cancel` | **不需要唤醒**：闲时自主激活，拉起内嵌在 status/approve/dispatch |

三条通道共用 `drain.WorkerPaths`（锁判活、stale 回收、released 墓碑、`.done` 身份三元组、
协作停止信号）——单 worker 互斥由锁保证，故获锁一刻仍在 `in_progress` 的条目必是上一任
被杀死时的孤儿，`reset_orphans()`/`reset_orphan_backlog()` 就地复位。**新通道不再另写一套
事故硬化逻辑**（`WorkerPaths` 存在的全部理由）；队列形状同理，增删/取项/孤儿复位/裁剪都在
`workq.py` 一处（防双源漂移）。

**孤儿复位管五面**（进程面 20261012-184000-leader，抢救面 20261012-234000-leader）：锁、条目、
worktree 名、进程之外还有**进度**——回收不再默认销毁半成品：脏面（未合并/证据不足同理）→ 落
`wip(salvage)` 提交并保面保分支、队列自动接线续作；确认干净才 `remove --force` + `branch -D`。
判据与形状见 `salvage-reclaim.md`。

第四面**进程**：worker 被杀不等于它派生的 headless 会话被杀（独立进程树），那个会话还在往原
worktree 写，重派同名就是双写。`dispatch` 派生瞬间把会话 pid 记进运行态会话池（存活期=该跳
在途期），drain 按任务书名对上 pid，**判活 + 核对 argv 里的 sid** 两闸都过才 `taskkill /T /F`。
判不准（pid 被复用、杀完仍活、`remove --force` rc=0 但目录还在）就**绝不对旧 pid 动手**，
条目升一代 `<id>-r<N>` 换隔离面，新名写进重派任务书。日志痕迹：`终止孤儿会话进程` /
`已不是会话…不动手` / `异名重建`；`.done` 的 `cleaned`/`regenerated`/`salvaged` 三键。
`--dry` 一条都不执行。

## 2. watch 挂载：自动为主，手动兜底（机制全文 README §4.5 v2）

- **自动（常态，2026-10-13 起）**：`Stop + asyncRewake` 钩子**每回合接管**挂 `orch watch`
  （抢「最新在听者」token；命中 `exit 2` 唤醒、超时/异常 `exit 0`；旧钩子自查 token 让位）。
  **无需组长手动挂**——交办/唤醒输出只回报「已自动挂载」。
- **手动兜底（仅钩子不可用时）**：以 Bash `run_in_background` 启动 `orch watch`，且**必须是该
  后台调用的唯一顶层命令**（`&`/`&&`/管道嵌入即被父任务连坐杀死——2026-10-12 夜航停摆 6h 的成因）。
- 漏挂/断唤的账由下次 `orch status` 补账，「忘了挂」≠「事件丢了」。
- **测试/诊断性挂载前须确认无 live 挂载或先停**：`arm()` 覆写 `state/watch.json`，会抹掉
  在等那任的心跳与黑匣子；`recent`/`exit_reason` 是脱靶复盘的现场，status 读出其末帧。
- 事件类型：基线差分六类 `QUEUE-COMPLETE`/`NEEDS-LEADER`/`QUOTA-STOP`/`CHANNEL-STOP`/`CRASH`/
  `SALVAGE`（两类 STOP 是系统性故障全局停止：充值 vs 等渠道）；＋统一事件流 `state/events.jsonl`
  生产者直发的 `JOB-DONE`/`STEP-REVIEW-DONE`/`QUEUE-EMPTY`/`TASKBOOK-NEW`（watch 按水位线读出即
  唤醒）。醒来先 `orch events drain` 排空（幂等＝合并唤醒）→ 逐 plan `orch plan advance`。
  `BENCH-DONE <run>` 可选挂载、默认不挂（§5.1-B）。

## 3. bench-* 使用要领（组长侧）

```bash
uv run --no-sync pysci-orch bench-add run --cases fig-basic,lit-arxiv --label after-<commit> --start
uv run --no-sync pysci-orch bench-add ab  --ref-a dd6e37b --ref-b HEAD --cases fig-basic --label L
uv run --no-sync pysci-orch bench-add capture --case fig-basic --ref HEAD
uv run --no-sync pysci-orch bench-add check --brief "<核查任务书正文>" --requested-by leader
uv run --no-sync pysci-orch bench-queue          # 看在途/待查看；--all 看历史
uv run --no-sync pysci-orch bench-seen <id>      # 看完摘掉（否则一直挂在 status 上）
uv run --no-sync pysci-orch bench-stop           # 协作停止：条目间隙干净退出，不 kill
```

- **入队即返**：`bench-add` 不阻塞；`--start`（或随后 `bench-up`）派生分离 worker，组长继续
  别的活。队列空时 worker 自毁，非常驻。
- 待查看面是**判决的入口**：`ab` 项的 `改善/未改善/inconclusive`、`run` 项的
  `通过 n/m` 都在这一行里；细节看 `state/evals/ab/<label>.json` 与 run 日志。
- `inconclusive` 恒带 `escalate=twin`：不是平局而是「测量不可信」——先看污染提交（`pollution`），
  换 ref 对或把混杂提交并入被测面，**不要**用 `--force` 把不可信因果当结论。
- **系统性故障**（`dispatch.SYSTEMIC_KINDS`，drain/bench/inspect 三通道同族处置）：worker
  全局停止、剩余条目保持 pending 不烧账，`.done` 附 `stopped=<kind>`；两类**恢复动作不同**
  ——`quota_exhausted`=充值或换渠道；`channel_error`=渠道服务面故障，等恢复即可
  （2026-10-12 实测：BYOK 坏时 CLI 不报 credit 文案，而是合成一条 `model="<synthetic>"` 的
  assistant 回合说 "Custom model service error…"，故判据取**固定文案 ∧ 会话档末轮形状**合取；
  单特征会误伤复述该句的复盘任务）。恢复后条目照常 FIFO 续跑（drain 由下一次 `approve`
  接管，压测队列由 `bench-up`）。
- **预设席位回退链**（用户 2026-10-12 链化 / 22:4x 挂预设；全文 README §3.5）：
  `[model] preset` 选预设，角色（main/pro，`--model-tier` 覆盖，旧 flash/max 归一）取该预设
  **有序链**，当前席不可用 → 沿链一席一跳降级；**全链耗尽**才保留系统性 kind 全局停止。
  预设不可激活 → dispatch 拒发、drain **拿锁前**拒绝开工（`skipped=preset_unavailable`）。
  **例外：`channel_error` 先自救**（backlog 20261013-143000-leader）——同会话 resume 大概率
  即愈、零工作损失，双闸（`model_fallback.same_seat_retry` 缺省 10 + `budget_s` 缺省 1800s）
  都过才换席（付费席降为后手）；`quota_exhausted` 与启动即失败不自救；自救**不入**回退事件面，
  次数/耗时走台账 `same_seat_retries`；非系统性失败仍同席重试一次（换席白烧）。**两条面**：
  调用者面 `[!] 档位回退：… 已切换 <新席>`（dispatch 输出 + run 日志）+ 观察面
  `state/channel-events.jsonl`（mirror / `orch status` 健康段报计数与最近几条）。**降级粘滞窗**
  （`skip_window_s` 默认 3600s）：坏席进窗即跳过、过期自动复探、跑通即销；全链在窗内按**失败
  最早**（最先到期）一席强探，不停摆。换席后实际用的渠道看台账 `model` 字段。
- `approve` 时选 **M**（基线捕获 + 实施后副本 A/B）的判决包接口：E5a/E5b 未实施，
  目前由组长手动 `bench-add capture`/`bench-add ab` 完成同一效果（E4 只交付运行职责）。

## 4. 压测隔离面（为什么不会污染生产）

`eval.run_case` 两头夹：任务书附 `EVAL_HOP_NOTE`（禁改自维护层）+ 前后拍该 pod 的
跟踪脏面快照，只还原本跳新增者（还原路径进台账 `reverted`）。副本 A/B 更严格：测量在
detached worktree 里跑，用后即弃，读数经 `ROWS_BEGIN/ROWS_END` 标记带由父进程代写主树
台账（`runner=bencher`）。反复出现 `reverted` = 案内任务书该收紧，别加重事后还原。
