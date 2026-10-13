# 后台 worker 与唤醒通道（drain / watch / bench-*）

SKILL.md 的命令面只留一行索引；本文放**为什么**与**踩过的坑**。

## 1. 两条 worker 通道，一套锁原语

| 通道 | 队列 | 锁/日志面 | 唤醒方式 |
|---|---|---|---|
| devops drain | `state/backlog.json` | `devops.lock` / `devops-runs/` / `devops.cancel` | `approve` 自动派生 + `orch watch` 事件即时唤醒 |
| bencher worker（README §5.5，E4） | `state/bencher/queue.json` | `bencher.lock` / `bencher-runs/` / `bencher.cancel` | **择机查看**：终态进 `orch status` 的「待查看」面，不即时打断 |
| inspector worker（README §6，E5a） | `state/inspector/queue.json` | `inspector.lock` / `inspector-runs/` / `inspector.cancel` | **不需要唤醒**：闲时自主激活，拉起内嵌在 status/approve/dispatch |

三条通道共用 `drain.WorkerPaths`（锁判活、stale 回收、released 墓碑、`.done` 身份三元组、
协作停止信号）——单 worker 互斥由锁保证，所以获锁一刻仍在 `in_progress` 的条目必是上一任
被杀死时留下的孤儿，`reset_orphans()`/`reset_orphan_backlog()` 就地复位。
**新通道不再另写一套事故硬化逻辑**（E4 的约束，也是 `WorkerPaths` 存在的全部理由）；
队列面同理——三套队列的增删/取项/孤儿复位/裁剪都在 `workq.py` 一处（E5a 把 benchd 的
自研队列改为委派，避免双源漂移）。

**孤儿复位管五面**（进程面 20261012-184000-leader，抢救面 20261012-234000-leader）：锁、条目、
worktree 名、进程之外还有**进度**——回收不再默认销毁半成品：脏面（未合并/证据不足同理）→ 落
`wip(salvage)` 提交并保面保分支、队列自动接线续作；确认干净才 `remove --force` + `branch -D`。
判据与形状见 `salvage-reclaim.md`。

第四面**进程**：worker 被杀不等于它派生的 headless 会话被杀（独立进程树），那个会话还在往原
worktree 写，重派同名就是双写。`dispatch` 在派生瞬间把会话 pid 记进运行态会话池（`pid` 键
的存活期=该跳在途期），drain 回收时按任务书名把孤儿条目对上 pid，**判活 + 核对
argv 里的 sid** 两道闸都过才 `taskkill /T /F`。判不准（pid 已被系统复用、杀完复查仍活着、
`remove --force` rc=0 但目录还在）就**绝不对旧 pid 动手**，条目升一代 `<id>-r<N>` 换隔离面，
新一代名直接写进重派的任务书。run 日志里的痕迹：`第四面回收：终止孤儿会话进程` /
`已不是会话…不动手` / `旧隔离面无法确认干净 → 异名重建`；`.done` 摘要对应
`worktrees_cleaned`、`worktrees_regenerated` 与 `worktrees_salvaged` 三个键。`--dry` 演练
一条都不执行。

## 2. watch 挂载纪律（2026-10-12 夜航停摆 6h 的教训）

- **`orch watch` 必须是该后台调用的唯一顶层命令**。用 `&`/`&&`/管道把它嵌进别的命令，
  父任务终结会连坐杀死等待器——事件命中也没有唤醒通道。
- 漏挂/断唤的账由下次 `orch status` 补账（fired 未确认逐条列明细），所以「忘了挂」
  不等于「事件丢了」。
- **测试/诊断性挂载前须确认无 live 挂载或先停**：`arm()` 覆写 `state/watch.json`，会抹掉
  在等那任的心跳与黑匣子；`recent`/`exit_reason` 是脱靶复盘的现场，status 读出其末帧。
- 事件类型（六类）：`QUEUE-COMPLETE` / `NEEDS-LEADER` / `QUOTA-STOP` / `CHANNEL-STOP` /
  `CRASH` / `SALVAGE`（抢救知会，不处置）。两类 STOP 都是**系统性故障全局停止**，
  恢复动作不同（充值 vs 等渠道）。
  `BENCH-DONE <run>` 属**可选挂载、默认不挂**——§5.1-B 明写测量完成不该即时打断组长。

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

- **入队即返**：`bench-add` 不阻塞；`--start`（或随后 `bench-up`）派生分离 worker，
  组长继续别的活。队列空时 worker 自毁，非常驻、不占资源。
- 待查看面是**判决的入口**：`ab` 项的 `改善/未改善/inconclusive`、`run` 项的
  `通过 n/m` 都在这一行里；细节看 `state/evals/ab/<label>.json` 与 run 日志。
- `inconclusive` 恒带 `escalate=twin`：它不是平局，是「测量不可信」。此时先看污染提交
  列表（`pollution`），要么换 ref 对，要么把混杂提交并入被测面，**不要**用 `--force`
  把不可信的因果当成结论。
- **系统性故障**（`dispatch.SYSTEMIC_KINDS`，drain/bench/inspect 三条通道同族处置）：worker
  全局停止、剩余条目保持 pending 不烧账，`.done` 附 `stopped=<kind>` 与分列说明，两类
  **恢复动作不同**——`quota_exhausted`=额度耗尽须充值或换渠道；`channel_error`=渠道服务面
  故障，等渠道恢复即可（2026-10-12 实测：BYOK 渠道坏了 CLI 不报 credit 文案，而是合成一条
  `model="<synthetic>"` 的 assistant 回合说 "Custom model service error…"，故判据取**固定
  文案 ∧ 会话档末轮形状**合取，文案单特征会误伤复述该句的复盘任务）。恢复后条目照常 FIFO
  续跑（drain 由下一次 `approve` 自动接管，压测队列由 `bench-up`）。
- **预设席位回退链**（用户裁决 2026-10-12 21:1x 链化 / 22:4x 挂预设，README §3.5）：
  `data/pysci_settings.toml [model] preset` 选预设，成员角色（main/pro，`--model-tier`
  覆盖，旧 flash/max 归一）取该预设的**有序链**，当前席不可用（额度 / 渠道 / 启动即失败）
  → 沿链一席一跳自动降级；**全链耗尽**才保留系统性 kind 并全局停止。预设不可激活（deepseek
  过渡期）→ dispatch 拒发、drain 在**拿锁前**拒绝开工（`skipped=preset_unavailable`，
  不把整队列标成失败）。非系统性失败仍是同席重试（故障不在渠道，换席白烧）。两条面都必须
  看得见：调用者面逐条
  `[!] 档位回退：<角色> 席<名>不可用 → 已切换 <新席>（原因 …）`（dispatch 输出 + run 日志），
  观察面 `state/channel-events.jsonl`（mirror 概览项「回退」/`orch status` 健康段报计数与最近
  几条）。**降级粘滞窗**（`registry.model_fallback.skip_window_s`，默认 3600s）：坏席进窗后
  窗内不再探测它（每次派发仍打一行「席 X 跳过中」），过期自动恢复探测，跑通即销窗；全链都在
  窗内时按**失败最早**（=最先到期）的一席强探一次，不停摆。换席后想知道实际用了哪个渠道：看
  台账 `model` 字段（记的是**实际所用席位**，不是配置里的首席）。
- `approve` 时选 **M**（基线捕获 + 实施后副本 A/B）的判决包接口：E5a/E5b 未实施，
  目前由组长手动 `bench-add capture`/`bench-add ab` 完成同一效果（E4 只交付运行职责）。

## 4. 压测隔离面（为什么不会污染生产）

`eval.run_case` 两头夹：任务书附 `EVAL_HOP_NOTE`（禁改自维护层）+ 前后拍该 pod 的
跟踪脏面快照，只还原本跳新增者（还原路径进台账 `reverted`）。副本 A/B 更严格：测量在
detached worktree 里跑，用后即弃，读数经 `ROWS_BEGIN/ROWS_END` 标记带由父进程代写主树
台账（`runner=bencher`）。反复出现 `reverted` = 案内任务书该收紧，别加重事后还原。
