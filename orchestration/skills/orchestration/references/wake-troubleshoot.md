# 唤醒未达排障清单（watch / 事件流）

「队列空了却没唤醒」的**定序**排查——2026-10-16 缺陷 `20261016-203600-devops` 排障时
「每跳都靠猜」，实测多花 30+ 分钟。路径以仓库根为基准。

## 唤醒未达排障清单（2026-10-16 实证定序；每步一条命令，先证伪最外层）

1. **listener 有没听见**：`tail -5 .qoder/hooks/watch-arm.log`（零「命中事件」＝没听见）。
2. **消费账撞车**：读 `state/watch.json` 的 `consumed` — 身份是否与当前流撞车（`JOB-DONE|#N`
   ＝旧流序号身份）。
3. **水位倒挂**：`orch events list` 首行；「已排空至 X」> 流 max seq＝流被回卷（现自愈并点名）。
4. **流被覆盖**：`ls -t state/events.jsonl.bak-*|head -1` 与现流对照条数/序号（merge cp 旁证；
   唤醒链已入同步排除名单）。
5. **listener 活死**：`ls -t state/watchdog-runs/*.log|head -1` 末条心跳 age，或 `orch status`
   watch 段。补报成因由 `watch._listener_covered_event` 分档：**事件时刻有在岗 listener** 者
   属去重/水位吞后补报，**不是聋期**（20261016-215945-devops 修）。
