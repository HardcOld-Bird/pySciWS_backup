# 唤醒链三级防线与巡更规程（autonomy-fix a1–a3；队列感知巡更 adhoc-20261014-124521-s1）

本册是**组长侧**的就位与巡更手册：三层防线各管什么、巡更怎么自动起止、新会话/新机器
开机就位做什么。设计背景见 `orchestration/README.md` §4.5；本册只写**操作步骤与可证伪判据**。

## 1. 三级防线（谁在哪一级兜底）

| 级别 | 机制 | 生命周期 | 覆盖的失效 |
|---|---|---|---|
| 会话级（①） | `.qoder/hooks/watch-arm.sh`（`Stop` 钩子每回合接管，token 竞争，循环 `orch watch`） | 组长会话回合内 | 事件产生时**有监听**：命中即 `exit 2` 唤醒 |
| 会话级（②） | **队列感知巡更伪事件**（`PENDING-WORK`/`STALL`，随 ① 同帧判定） | 随 ① 同生命周期（**自动起止**：队列全空即零唤醒） | 没有事件但**状态该看而没人看**（队列/backlog/事件流/任务书非空、条目停滞） |
| 机器级（③） | `pysci-watchdog`（`orch watchdog start`；OS 计划任务每 5 min 幂等点火） | 独立于会话/机器重启（③ 的 OS 注册在） | listener/watchdog **被无声杀死**、会话消亡连坐 |

关系：① 是主路（零轮询、事件驱动）；② 是**队列感知的自动巡更**——补「没事件但该知道」，
判据全在代码（`watch.queue_pressure`），**无需组长挂/撤任何 cron**；③ 是**监督**——它不制造
事件，只保证 ① 那一环还在跑（心跳 stale → 重臂）。三者互相独立：任何一层坏掉，另两层仍在。

## 2. 巡更（②：队列感知，自动起止）

**判据（单一实现 = `watch.queue_pressure`，四处消费面共用）**：

- `PENDING-WORK`（**有活却没人干**，四条任一成立）：devops backlog 有 `pending`；任一成员
  队列有 `pending` 且该成员**无存活 worker**；统一事件流有**超 30 min 未排空**的事件
  （刚产生的事件正是本帧要唤醒的那几条，计入会与既有事件重复）；`data/taskbooks/` 有
  `new` 态未处理。
- `STALL`（**条目停滞**）：任一队列条目 `in_progress` 且距 `started_at` 超阈值。

**节流（硬要求）**：同类事件 **≤1 次/30 min**——防「唤醒→看一眼→无新事→再唤醒」的循环。
节流账落在 `watch.json`，跨挂载继承（重启不重置、不堆积）；被节流时**不推进该条件水位**，
故窗口一过信号照报，不丢。

**全空即静默**：队列全空 → 条件不成立 → 纯 TIMEOUT 循环**零唤醒**（这正是原先手工巡更
cron「全空闲即撤」的语义，现在由代码自动达成）。组长**不手工挂/撤巡更 cron**。

**与铁律「禁止 sleep-轮询」的关系**：禁的是拿轮询**等工作完成**（把 LLM 当定时器烧 token）；
巡更是**低频读一次状态面**的兜底，驱动工作仍靠 ① 的事件唤醒——两者不冲突。

**阈值可配**：`registry.json` 顶层 `watch` 块——`stall_threshold_s`（默认 3600）、
`pending_throttle_s`（默认 1800）；逐键 fail-open（读坏就按出厂默认跑）。

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
   卸载同命令 `-Mode uninstall`（删除任务；看门狗进程另用 `orch watchdog stop`）。该 PS1
   **纯 ASCII 有意为之**（PowerShell 5.1 对无 BOM UTF-8 按 ANSI 解码，CJK 会被解成反引号
   转义下一行 → ParserError；仓库里所有 `.ps1` 都遵守）。
2. **唤醒权（主会话定向，wake-claim）**：主会话跑一次 `pysci-orch wake-claim --name main`
   （首次就位/换会话时）。此后钩子每回合 `wake-claim --check`：**异会话的钩子直接退出**
   （不监听、不抢 token、绝不 `exit 2`）——次要会话不再误唤醒；`wake-claim status` 查看
   当前 owner、`… release` 退回 legacy（未认领时仍是旧的「最新在听者胜」）。
   **局限**：两端取不到会话区分变量（`WT_SESSION` 等缺席）时认领记 legacy（认领输出会写明）。
   **同窗格续持**：`WT_SESSION` 是终端窗格 GUID——同一窗格新开的会话**自动沿用**认领
   （无需重认）；换窗格/换终端才需重跑 `wake-claim`（自动认领补全后零手工，见 backlog）。
3. **listener（会话级①）**：无需手工——`Stop` 钩子下一次回合自动接管（`watch-arm.sh` 每回合
   还**幂等自启动看门狗**）；漏挂由 `orch status` 补账（`reconcile`）。
4. **巡更（②）**：**无需动作**——队列感知自动起止。仅 §2「降级注」那种全灭态才手工挂 cron。
5. **自检**：`pysci-orch watchdog status` 三行（看门狗态 / 心跳年龄 / listener 活与不活）
   应为 `running` / `<300s` / `活着`；`orch status` 的 watch 段应显示已挂载或可补账。

## 4. 判据（可证伪）

- 迟到唤醒：制造「挂载前的事件」→ 新挂载**首轮**命中且 detail 带「迟到唤醒」；
  再挂载**不重复**（水位单调）。
- **同一事件只唤醒一次**：命中后**立即重挂载**，重复检测同一事件 → 不再唤醒
  （消费账 `watch.consumed`；2026-10-14 12:44 双唤醒的回归判据）。
- 队列感知：制造「一成员队列 pending 且无 worker」→ watch 一轮内出 `PENDING-WORK`；
  清空后连续周期**零唤醒**；窗口内重复条件被节流（`watch.throttle` 有账）。
- STALL：`started_at` 超阈值的 `in_progress` 条目 → 出 `STALL`。
- 看门狗重臂：令心跳 stale > 300 s → 看门狗日志出现 `[→] 重臂监听循环`。
- 队列超时：入队 > 1800 s 未取走 → 事件层出 `QUEUE-TIMEOUT` 且 TUI 详情点名（不许静默积压）。
- 唤醒权认领：认领后 `wake-claim --check` 在**同身份**返回 0、**异身份**返回 3（钩子据此退
  `exit 0`：不监听、不抢 token、绝不 `exit 2`）；`release` 后回 0（legacy）。owner 缺席期间
  发的事件在 owner 下次挂载被**迟到唤醒**补上（水位单调，重挂不重复消费）。

判据的自动化面见 `tests/skills/orchestration/test_queue_watch.py`（队列感知 + 双消费回归）、
`test_wake_claim.py`（唤醒权四类语义）与 `test_autonomy_e2e.py`（迟到唤醒 / 看门狗 / 三线 e2e）。
