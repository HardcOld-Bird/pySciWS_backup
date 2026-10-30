# 唤醒未达排障清单（watch / 事件流）

「队列空了却没唤醒」的**定序**排查——2026-10-16 缺陷 `20261016-203600-devops` 排障时
「每跳都靠猜」，实测多花 30+ 分钟。路径以仓库根为基准。

## 唤醒未达排障清单（2026-10-16 实证定序；每步一条命令，先证伪最外层）

1. **listener 有没听见**：`tail -5 .qoder/hooks/watch-arm.log`（零「命中事件」＝没听见）。
   先分档（见下「唤醒通道分档」）：末条是「命中事件 → 唤醒组长」（会话附着，真唤醒）还是
   「命中事件但无会话附着」（看门狗补位看护，**听见了但叫不醒**——要等 owner 下次挂载
   迟到唤醒，不是聋）；时间戳 `[]` / token `N-` ＝ coreutils 化石 PATH（下节）。
2. **消费账撞车**：读 `state/watch.json` 的 `consumed` — 身份是否与当前流撞车（`JOB-DONE|#N`
   ＝旧流序号身份）。
3. **水位倒挂**：`orch events list` 首行；「已排空至 X」> 流 max seq＝流被回卷（现自愈并点名）。
4. **流被覆盖**：`ls -t state/events.jsonl.bak-*|head -1` 与现流对照条数/序号（merge cp 旁证；
   唤醒链已入同步排除名单）。
5. **listener 活死**：`ls -t state/watchdog-runs/*.log|head -1` 末条心跳 age，或 `orch status`
   watch 段。补报成因由 `watch._listener_covered_event` 分档：**事件时刻有在岗 listener** 者
   属去重/水位吞后补报，**不是聋期**（20261016-215945-devops 修）。

## 唤醒通道分档（2026-10-27 wake-repair 实测定稿）

- **会话附着 watch**（Stop 钩子 spawn，`asyncRewake`）：命中「stderr + exit 2」由 harness
  注入 rewake——**唯一真唤醒通道**；`wake-claim status` 的 `owner watch` 判的就是它。
- **看门狗补位实例**（`watchdog._rearm` 重臂，**无 harness 附着**）：exit 2 **无接收方**
  （实测喊给空气→退出→再臂→再命中死循环）。现挂 `orch watch --guardian` **看护模式**：
  命中只登记（watch-arm.log「看护登记」+ `watch.json` 的 `guard_seen`）、基线不推进留待
  owner 迟到唤醒、心跳不断（看门狗不再空转重臂）。
- **手动兜底挂载**（组长 `run_in_background`）：命中即进程退出＝后台通知唤醒；无 owner
  身份不记消费账，事件留给下次挂载。
- **token 战况**：`watch-holder.token` ＝ `<pid>-<epoch>`，持有者每轮刷 epoch＝心跳；补位
  实例接管前查心跳，新鲜即「会话 watch 在岗」不顶替（防顶掉活 watch——2026-10-27 事故
  根因 2，日志「被取代→让位退出」紧跟补位「接任」即此病）。
- **coreutils 化石**：补位实例时间戳 `[]` / token `$$-` ＝ spawn PATH 缺 MSYS `usr\bin`
  （date/cat/sed 全部 command not found——事故根因 3）。watch-arm.sh 已 `$BASH` 目录自愈、
  `watchdog._rearm` 已前插 vendored 工具目录；再见 `[]` 查
  `state/watchdog-runs/*.spawn.log` 的 command not found 行。
