# 唤醒链三级防线与巡更规程（autonomy-fix a1–a3；队列感知巡更 adhoc-20261014-124521-s1）

本册＝**组长侧**就位与巡更手册（三层防线各管什么 / 巡更自动起止 / 开新机就位做什么）；
设计背景见 `orchestration/README.md` §4.5，本册只写**操作步骤与可证伪判据**。

## 1. 三级防线（谁在哪一级兜底）

| 级别 | 机制 | 生命周期 | 覆盖的失效 |
|---|---|---|---|
| 会话级（①） | `.qoder/hooks/watch-arm.sh`（`Stop` 钩子每回合接管、token 竞争、循环 `orch watch`） | 组长会话回合内 | 事件产生时**有监听**：命中即 `exit 2` 唤醒 |
| 会话级（②） | **队列感知巡更伪事件**（`PENDING-WORK`/`STALL`，随 ① 同帧判定） | 随 ① 同生命周期（**自动起止**：队列全空即零唤醒） | 没有事件但**状态该看而没人看**（队列/backlog/事件流/任务书非空、条目停滞） |
| 机器级（③） | `pysci-watchdog`（`orch watchdog start`；OS 计划任务每 5 min 幂等点火） | 独立于会话/机器重启 | listener/watchdog **被无声杀死**、会话消亡连坐 |

关系：① 主路（零轮询、事件驱动）；② 队列感知巡更（补「没事件但该知道」，**无需挂/撤
cron**）；③ 监督（不制造事件，只保证 ① 在跑：心跳 stale → 重臂）。三层互相独立：任一层
坏掉，另两层仍在。

## 2. 巡更（②：队列感知，自动起止）

**判据（单一实现 = `watch.queue_pressure`，四处消费面共用）**：

- `PENDING-WORK`（**有活却没人干**，四条任一成立；含**夜航判据**见 §5）：devops backlog 有
  `pending` **且无 devops 队列 worker 在跑**（执行归 FIFO，worker 在场即降噪）；任一成员队列
  有 `pending` 且该成员**无存活 worker**；统一事件流有**超 30 min 未排空**的事件；`data/
  taskbooks/` 有 `new` 态未处理。
- `STALL`（**条目停滞，叠加心跳判活**）：任一队列条目 `in_progress` 且距 `started_at` 超
  阈值，**且该成员无心跳新鲜的活会话**（`backpressure.live_entries` ≤120s）——
  慢而不死（重型环节正当耗时）不报，死而僵才报（2026-14 晚实测误报的根治面）。

**节流（硬要求）**：同类事件 **≤1 次/30 min**（防唤醒→看一眼→再唤醒的循环）。账落
`watch.json` 跨挂载继承；被节流时**不推进条件水位**，窗口一过照报，不丢。

**全空即静默**：队列全空 → 零唤醒（原手工 cron「全空闲即撤」语义，现由代码达成）；组长
**不手工挂/撤巡更 cron**。

**与「禁 sleep-轮询」不冲突**：禁的是拿轮询**等工作完成**；巡更只低频读一次状态面，
驱动仍靠 ① 的事件唤醒。

**阈值可配**：`registry.json` 顶层 `watch` 块（`stall_threshold_s` 3600 /
`pending_throttle_s` 1800）；逐键 fail-open（读坏按出厂默认）。

**降级注（仅兜底）**：**钩子与看门狗全灭**时才手工挂一条 harness cron 顶一阵：

```
巡更：跑 `pysci-orch status`，只核三件事——① `orch events drain` 排空积压 +
`orch queues list` 看队列与延期；② 有 in-flight 则 `orch plan advance` 推进，否则不动作；
③ 异常（CRASH/QUEUE-TIMEOUT/needs_leader/看门狗不在场）留痕。全程只读、零改动。
```

（`CronCreate` 周期 `*/30 * * * *`，7 天自动过期；恢复正常后 `CronDelete` 撤掉。）

## 3. 新会话 / 新机器就位（开机步骤）

1. **看门狗（机器级③）**：`pysci-orch watchdog status` → 不是 `running` 就
   `pysci-orch watchdog start`（幂等，锁在手即拒绝）；**注册 OS 计划任务**（重启/重登后
   自动点火，免记手工）：
   ```
   powershell -ExecutionPolicy Bypass -File orchestration\scripts\watchdog-scheduler.ps1 -Mode install
   ```
   卸载同命令 `-Mode uninstall`；进程另用 `orch watchdog stop`。该 PS1 **纯 ASCII 有意
   为之**（PS 5.1 无 BOM UTF-8 按 ANSI 解 → CJK 吞行；仓内 `.ps1` 同规）。
2. **唤醒权（自动归属，wake-owner）**：**零手工**——钩子每回合读 stdin 载荷
   `session_id` 走 `wake-claim --check`：无 owner 自动认领（首会话即主会话）、owner 续
   心跳、**异会话永不抢占**（不监听、不抢 token、绝不 `exit 2`）、owner stale（心跳
   >30 min 且其 watch 已死）自动接管、`SessionEnd` 钩子显式释放（换会话即干净迁移）。
   `pysci-orch wake-claim status` 查当前 owner 与 stale 判定；`--force [--session-id <id>]`
   是**应急强制迁移**（非常规步骤）。载荷缺失/解析失败一律 fail-open 走旧 token 语义留痕。
3. **listener（会话级①）**：无需手工——`Stop` 钩子下回合自动接管（`watch-arm.sh` 每回合
   幂等自启动看门狗）；漏挂由 `orch status` 补账（`reconcile`）。
4. **巡更（②）**：**无需动作**——队列感知自动起止。仅 §2「降级注」那种全灭态才手工挂 cron。
5. **自检**：`pysci-orch watchdog status` 四行（看门狗态/心跳年龄/listener/逻辑实例数——口径
   = distinct `--run-log`，非进程数）应为 `running`/`<300s`/`活着`/`1`；`orch status` 的
   watch 段应显示已挂载或可补账。

## 4. 判据（可证伪）

- 迟到唤醒：制造「挂载前的事件」→ 新挂载**首轮**命中且 detail 带「迟到唤醒」；
  再挂载**不重复**（水位单调）。
- **同一事件只唤醒一次**：命中后**立即重挂载**，重复检测同一事件 → 不再唤醒
  （消费账 `watch.consumed`；2026-10-14 12:44 双唤醒的回归判据）。
- 队列感知：造「成员队列 pending 无 worker」→ 一轮内出 `PENDING-WORK`；清空后连续周期
  **零唤醒**；窗口内被节流（`watch.throttle` 有账）。
- STALL：`started_at` 超阈值的 `in_progress` 条目，**且该成员无活会话心跳** → 出 `STALL`；
  心跳新鲜（慢而不死）→ 不出。
- backlog FIFO：`approve` → 计划器入队 `queues/devops.json`（面板可见）→ worker 消化销账；
  计划器锁忙**不丢件**（持久 backlog 排队）；完成唤醒=事件流 `QUEUE-EMPTY`。判据面
  `test_backlog_fifo.py`。
- 看门狗重臂：令心跳 stale > 300 s → 看门狗日志出现 `[→] 重臂监听循环`。
- 队列超时：入队 > 1800 s 未取走 → 事件层出 `QUEUE-TIMEOUT` 且 TUI 详情点名（不许静默积压）。
- 唤醒权自动归属：不同 `session_id` 的载荷喂 `wake-claim --check`——首会话 **claim** 返回
  0、owner 返回 0（续心跳）、异 id 返回 3（钩子退 `exit 0`：不监听、不抢 token、绝不
  `exit 2`）、心跳 >30 min + owner watch 已死 → 返回 0 且接管、`SessionEnd` 返回 4（释放后
  退出）、垃圾/空载荷返回 0 且 fail-open 留痕；owner 缺席期事件靠迟到唤醒补（水位单调）。

判据的自动化面见 `tests/skills/orchestration/test_queue_watch.py`（队列感知 + 双消费回归）、
`test_wake_owner.py`（唤醒权五类语义）、`test_backlog_fifo.py`（approve→入队→销账 / 锁忙
排队 / STALL 心跳双向 / 降噪）与 `test_autonomy_e2e.py`（迟到唤醒 / 看门狗 / 三线 e2e）。

## 5. 夜航模式（PSEP_2；总开关 `data/pysci_settings.toml [night] enabled`）

用户离机无人值守推进：**手动单开关**（钩子每回合现读）。**政策注入**（`night-policy.sh`：
SessionStart/PostCompact 注授权书全文；**开关变更下一轮生效**——Stop 钩子按会话差分注入，
无变化零输出）+ **判据**（`night-gate.sh` 拒 `AskUserQuestion` 转 `twin consult`；
`mirror-pending.json` 非空 → 夜航 `PENDING-WORK`）；两层 **fail-open**。**开关只归用户手动**
（LLM 永不改写），终态写总结即结束推进。保险丝层**缓行**（预留键勿设，§3.4）；twin 代裁走
`orch twin review`。全文见 README §4.5。
