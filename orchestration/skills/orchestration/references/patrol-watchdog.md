# 唤醒链三级防线与巡更规程（autonomy-fix a1–a3）

本册是**组长侧**的就位与巡更手册：三层防线各管什么、什么时候挂巡更 cron、新会话/新机器
开机就位做什么。设计背景见 `orchestration/README.md` §4.5（取证实录在那里）；本册只写
**操作步骤与可证伪判据**。

## 1. 三级防线（谁在哪一级兜底）

| 级别 | 机制 | 生命周期 | 覆盖的失效 |
|---|---|---|---|
| 会话级（①） | `.qoder/hooks/watch-arm.sh`（`Stop` 钩子每回合接管，token 竞争，循环 `orch watch`） | 组长会话回合内 | 事件产生时**有监听**：命中即 `exit 2` 唤醒 |
| 会话级（②） | **巡更 cron**（harness `CronCreate`，30 min 一次） | 活跃期（**全空闲即撤**） | 没有事件但**状态该看而没人看**（漏挂、wake 送达失败、队列悄悄积压） |
| 机器级（③） | `pysci-watchdog`（`orch watchdog start`；OS 计划任务每 5 min 幂等点火） | 独立于会话/机器重启（③ 的 OS 注册在） | listener/watchdog **被无声杀死**、会话消亡连坐 |

关系：① 是主路（零轮询、事件驱动）；② 是**低频人工巡更**，补的是「没事件但该知道」；
③ 是**监督**——它不制造事件，只保证 ① 那一环还在跑（心跳 stale → 重臂）。
三者互相独立：任何一层坏掉，另两层仍在。

## 2. 巡更 cron（②：活跃期挂，30 min）

**与「禁止 sleep-轮询」的关系**：铁律禁的是**用 sleep 轮询去等工作完成**（那是把 LLM 当
定时器烧 token）。巡更 cron 是**兜底巡更**——低频（30 min）读一次状态面，不是驱动工作的
机制；工作完成仍由 ①（watch 事件）唤醒。两者互不替代。

**挂载（组长活跃期）**：用 harness 的 `CronCreate`，周期 `*/30 * * * *`，prompt 用下面模板；
**7 天自动过期**（harness 侧），故长期不必回收。

**prompt 模板**（照抄；`*/30` 意为每 30 分钟触发一次）：

```
巡更：跑 `pysci status`（或 `pysci-orch status`），只核三件事——
① 队列/事件水位：`orch events drain` 排空积压、`orch queues list` 看成员队列与延期；
② 有 in-flight 工作则对照 `orch plan advance` 推进到下一环节，否则不动作；
③ 异常（CRASH/QUEUE-TIMEOUT/needs_leader/看门狗不在场）留痕到交付或 backlog。
全程零改动、只读；一切待办原样写给组长，不自行处置。
```

**全空闲自动撤销**：连续 **2 次**巡更读到「队列全空 + 无积压事件 + 无 in-flight + 看门狗在场」
即用 `CronDelete` 撤掉本条 cron（cron 本身要花 LLM 轮次，空转即浪费）。撤除后若再次需要
（挂长任务前）再按上面模板挂回。

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
2. **listener（会话级①）**：无需手工——`Stop` 钩子下一次回合自动接管；漏挂由
   `orch status` 补账（`reconcile`）。
3. **巡更 cron（②）**：按 §2 模板挂一条（活跃期才挂；长期任务期间尤其）。
4. **自检**：`pysci-orch watchdog status` 三行（看门狗态 / 心跳年龄 / listener 活与不活）
   应为 `running` / `<300s` / `活着`；`orch status` 的 watch 段应显示已挂载或可补账。

## 4. 判据（可证伪）

- 迟到唤醒：制造「挂载前的事件」→ 新挂载**首轮**命中且 detail 带「迟到唤醒」；
  再挂载**不重复**（水位单调）。
- 看门狗重臂：令心跳 stale > 300 s → 看门狗日志出现 `[→] 重臂监听循环`。
- 队列超时：入队 > 1800 s 未取走 → 事件层出 `QUEUE-TIMEOUT` 且 TUI 详情点名（不许静默积压）。
- 巡更 cron：`CronList` 里能看到挂着的 30 min 条目；全空闲 2 次后被 `CronDelete` 撤除。

判据的自动化面见 `tests/skills/orchestration/test_autonomy_e2e.py`。
