# 后台 worker 与唤醒通道（drain / watch / bench-*）

SKILL.md 的命令面只留一行索引；本文放**为什么**与**踩过的坑**。

## 1. 两条 worker 通道，一套锁原语

| 通道 | 队列 | 锁/日志面 | 唤醒方式 |
|---|---|---|---|
| devops drain | `state/backlog.json` | `devops.lock` / `devops-runs/` / `devops.cancel` | `approve` 自动派生 + `orch watch` 事件即时唤醒 |
| bencher worker（REFACTOR §8/E4） | `state/bencher/queue.json` | `bencher.lock` / `bencher-runs/` / `bencher.cancel` | **择机查看**：终态进 `orch status` 的「待查看」面，不即时打断 |
| inspector worker（REFACTOR §7/E5a） | `state/inspector/queue.json` | `inspector.lock` / `inspector-runs/` / `inspector.cancel` | **不需要唤醒**：闲时自主激活，拉起内嵌在 status/approve/dispatch |

三条通道共用 `drain.WorkerPaths`（锁判活、stale 回收、released 墓碑、`.done` 身份三元组、
协作停止信号）——单 worker 互斥由锁保证，所以获锁一刻仍在 `in_progress` 的条目必是上一任
被杀死时留下的孤儿，`reset_orphans()`/`reset_orphan_backlog()` 就地复位。
**新通道不再另写一套事故硬化逻辑**（这是 E4 的实现约束，也是 `WorkerPaths` 存在的全部理由）；
队列面同理——三套队列的增删/取项/孤儿复位/裁剪都在 `workq.py` 一处（E5a 把 benchd 的
自研队列改为委派，避免双源漂移）。

## 2. watch 挂载纪律（2026-10-12 夜航停摆 6h 的教训）

- **`orch watch` 必须是该后台调用的唯一顶层命令**。用 `&`/`&&`/管道把它嵌进别的命令，
  父任务终结会连坐杀死等待器——事件命中也没有唤醒通道。
- 漏挂/断唤的账由下次 `orch status` 补账（fired 未确认逐条列明细），所以「忘了挂」
  不等于「事件丢了」。
- **测试/诊断性挂载前须确认无 live 挂载或先停**：`arm()` 覆写 `state/watch.json`，会抹掉
  在等那任的心跳与黑匣子；`recent`/`exit_reason` 是脱靶复盘的现场，status 读出其末帧。
- 事件类型：`QUEUE-COMPLETE` / `NEEDS-LEADER` / `QUOTA-STOP` / `CRASH`。
  `BENCH-DONE <run>` 属**可选挂载、默认不挂**——§5.1-B 明写测量完成不该即时打断组长。

## 3. bench-* 使用要领（组长侧）

```bash
uv run pysci-orch bench-add run --cases fig-basic,lit-arxiv --label after-<commit> --start
uv run pysci-orch bench-add ab  --ref-a dd6e37b --ref-b HEAD --cases fig-basic --label L
uv run pysci-orch bench-add capture --case fig-basic --ref HEAD
uv run pysci-orch bench-add check --brief "<核查任务书正文>" --requested-by leader
uv run pysci-orch bench-queue          # 看在途/待查看；--all 看历史
uv run pysci-orch bench-seen <id>      # 看完摘掉（否则一直挂在 status 上）
uv run pysci-orch bench-stop           # 协作停止：条目间隙干净退出，不 kill
```

- **入队即返**：`bench-add` 不阻塞；`--start`（或随后 `bench-up`）派生分离 worker，
  组长继续别的活。队列空时 worker 自毁，非常驻、不占资源。
- 待查看面是**判决的入口**：`ab` 项的 `改善/未改善/inconclusive`、`run` 项的
  `通过 n/m` 都在这一行里；细节看 `state/evals/ab/<label>.json` 与 run 日志。
- `inconclusive` 恒带 `escalate=twin`：它不是平局，是「测量不可信」。此时先看污染提交
  列表（`pollution`），要么换 ref 对，要么把混杂提交并入被测面，**不要**用 `--force`
  把不可信的因果当成结论。
- 额度耗尽（`quota_exhausted`）是系统性故障：worker 全局停止、剩余条目保持 pending，
  `.done` 附 `stopped="quota_exhausted"`；渠道恢复后 `bench-up` 续跑。
- `approve` 时选 **M**（基线捕获 + 实施后副本 A/B）的判决包接口：E5a/E5b 未实施，
  目前由组长手动 `bench-add capture`/`bench-add ab` 完成同一效果（E4 只交付运行职责）。

## 4. 压测隔离面（为什么不会污染生产）

`eval.run_case` 两头夹：任务书附 `EVAL_HOP_NOTE`（禁改自维护层）+ 前后拍该 pod 的
跟踪脏面快照，只还原本跳新增者（还原路径进台账 `reverted`）。副本 A/B 更严格：测量在
detached worktree 里跑，用后即弃，读数经 `ROWS_BEGIN/ROWS_END` 标记带由父进程代写主树
台账（`runner=bencher`）。反复出现 `reverted` = 案内任务书该收紧，别加重事后还原。
