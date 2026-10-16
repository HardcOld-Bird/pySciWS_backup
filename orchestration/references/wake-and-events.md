# 等待与唤醒模型（README §4.5 全量）

> 长任务零轮询的等待/唤醒/巡更细节与实测判据；由 `orchestration/README.md §4.5` 迁入。

### 4.5 等待与唤醒模型（长任务零轮询）

问题：组员执行可能长达数小时（复杂仿真），组长（TUI 会话）必须「睡眠等待」而非
轮询（轮询 = 持续烧 token）。

**原生机制盘点（组长会话可用的等待通道）**：

| 机制 | 形态 | 等待期 token 成本 | 时长上限 | 适用 |
|---|---|---|---|---|
| 前台 Bash | 阻塞工具调用 | 零（阻塞中） | **600 s 硬上限**——直接回答：*不允许* 2 小时的同步工具调用 | 短任务 |
| **后台 Bash（run_in_background）** | 完成时**事件驱动通知**唤醒会话 | **零** | 无文档上限（**实测 130 s 与 3600 s 均存活并成功通知**） | **主路线** |
| Monitor 工具 | shell 侧监视进程，每行 stdout 成一条通知；persistent=会话生命周期 | 零（shell 轮询不耗 token，仅结果行触发） | persistent 无上限 | 辅助：监视状态文件/多任务 |
| CronCreate | 定时向会话投递提示 | 每次触发一轮 LLM | 7 天自动过期 | 兜底看门狗（低频巡检） |

**主路线设计**：长任务以**后台 Bash 任务**方式启动（`plan advance` 入队即返回 + `orch watch`
事件唤醒；计划外单件走 `plan adhoc` 的同步阻塞原语）——阻塞侧 orch 进程内部同步
等待组员 headless 子进程（进程级阻塞，非 LLM 轮询），完成后执行全部机械后处理
（交付解析、机械验收、台账、状态落盘），将「下一步指引」写入 stdout 后退出 →
Qoder 后台任务完成通知**自动唤醒组长会话** → 组长读取指引、做交接决策。等待期间：
零 token 消耗；TUI 不被阻塞（用户可继续与组长对话其他事项）。

**状态与恢复**（等待链路的故障面）：
- orch 在 `state/plans/<id>.status` 写入运行态（环节、组员进程 pid、启动时间）；
- 组长 TUI 关闭/机器重启 → 后台链全部死亡，但**计划状态机与组员会话都落盘**：
  重开会话后 `orch status` 检测 stale running（pid 已死）→ 按组长决策以
  `plan advance`（必要时 `--session new`）重派或续作（组员会话持久化与进程生命周期无关，
  §1.1 已证）；
- 兜底看门狗（可选，Phase 2 评估）：低频 cron（如每 30 min）提示组长运行
  `orch status`——防「通知机制自身故障」导致的永久沉睡；成本每次一小轮。

**组长等待纪律**（写入 orchestration 技能）：派发长任务必须走后台通道；**禁止
sleep-轮询循环**；等待期间不主动检查输出文件（通知会来）；用户插话优先响应。

**后台 drain 的等待：`orch watch` 事件等待器**（backlog 20261010-orch-watch）。
detached drain 不被任何前台进程持有，组长交办完即失去观察面——旧状只能靠「组长记得跑
`orch status`」，属纪律而非保证。硬化理念（用户裁决 2026-10-10：能代码硬保证的不靠 LLM
纪律；**事件判定全在代码 + 挂载提示在唤醒当刻由代码注入**；挂载动作自 2026-10-13 起由
Stop 钩子自动完成——见下方「自动挂载」，此前登记的「残余软点=组长照 [NEXT] 挂一条后台
命令」就此收口）：

- `pysci-orch watch [--timeout S=4h] [--interval S=60]` **阻塞轮询**（只读状态文件 +
  Python 侧判活，不起 shell 故 MSYS 路径转换问题免疫；间隔硬钳制 ≤60 s），命中即**单行**
  打印事件并退出 0，超时无事件打印 `TIMEOUT` 退出 0：
  - `QUEUE-COMPLETE <run>`——完成身份三元组 **(run 名, completed, count)** 相对基线变化
    （同 run 的 `.done` 被重写也能再报）。**收尾事实优先于锁的自称**：`.done` 由 drain 在
    `release_lock` 之后写出，摘要存在即已收尾，此刻锁仍显示 running 只可能是 unlink 失败
    残留 + 判活误真（Windows pid 回收/tasklist 超时保守判活）——旧判定让锁把事件压制后
    又无条件推进基线，导致 210557 那场 `fired=[]` 且补账同哑（backlog 20261011-220622-leader）。
    同一条「收尾事实优先」现已在**锁面**同样执法（§6：released 墓碑 + 「该 run 已有
    `.done`」判活否决），新代码不再产生谎锁帧；观察面的 `_in_progress` 抑制支与转移帧兜底
    因此退为保险——护升级前残留的旧基线，也顶住 `.done` 半写这类读不出证据的瞬态
  - `NEEDS-LEADER <id>`——backlog 中**新出现**的 needs_leader 条目
  - `QUOTA-STOP <run>`——额度类全局停止（`.done.stopped`，或 run 日志的 `[Q]` 痕迹——后者
    早一拍，不必等摘要落盘）
  - `CHANNEL-STOP <run>`——渠道级故障全局停止（同上，run 日志的 `[C]` 痕迹早一拍）：
    两类停止动作相同、恢复动作不同（充值/换渠道 vs 等渠道恢复），故分列两个事件
  - `CRASH <run>`——锁 pid 已死且该 run 无 `.done`
- **自动挂载（2026-10-13 硬化；v2 修缺陷）**：组长回合结束由 `Stop + asyncRewake` 钩子
  （`.qoder/hooks/watch-arm.sh`）自动接管等待器——**每回合的钩子都抢「最新在听者」身份
  （token）并挂 `orch watch`**，当前会话最近一任即**有效唤醒源**；旧钩子在下一周期（120s）
  自查 token 发现被取代即让位退出（防多 watcher 叠加）。命中即 `exit 2` 唤醒组长；超时/异常
  一律 `exit 0`（不卡会话）。**实测推翻早先的「结构上限」登记**：「唤醒注册只能由组长会话
  的工具发起」对 hook **分离进程**成立，但 `asyncRewake` 钩子由 harness 自身消费、**能**唤醒
  模型，且在组长 TUI 下**后台**执行（回合即刻结束、钩子在后台阻塞）。**v1 缺陷（实证教训）**：
  v1 的「已在挂载则放行」守卫会挡住**当前会话唯一能唤醒的那任**——旧会话那任 hook 跨重启仍占着
  watch，新会话各任被守卫挡成 no-op（exit 0），旧 hook 命中时 `exit 2` 又唤不动已重启的会话
  （harness 只认**最近一任**），致 2026-10-13 20:57 的 QUEUE-COMPLETE 唤醒丢失。v2 取消守卫、
  改为每回合接管。

**唤醒权自动归属（`wake-claim`；用户设计定稿 2026-10-14 晚，钩子 v6——「机制优于纪律」）**：
v2 的「最新在听者胜」在**主会话 + 次要会话并用**时会被抢——次要会话一有回合即抢走唤醒权，
drain/事件的 `exit 2` 被投到次要会话（污染其上下文，用户明确要避免；凌晨「被取代却无接任」
的谜之日志疑为同源）。第一版**认领制**（手工 `wake-claim` 认领 + env 取证身份）被判「硬保证
退化成软步骤」废弃；现行**零人工自动归属**：钩子不再丢弃 stdin 载荷——解析 `session_id` /
`hook_event_name`，真源 `state/wake-owner.json`（`session_id` + `claimed_at` +
`last_turn_at`），判定单一实现在 `wake-claim --check`：

| 情形（载荷判定） | 钩子行为 |
|---|---|
| **无 owner → 自动认领**（首会话即主会话，零人工） | 抢 token、挂 watch，照常 `exit 2` 唤醒 |
| **owner = 我 → 续心跳** | 照常抢 token、挂 watch |
| **owner ≠ 我 → 永不抢占**（次要会话自动排除） | **不监听、不抢 token、绝不 `exit 2`**（`exit 0`，留痕） |
| **owner stale → 自动接管**：心跳断供 >30 min **且** owner 自己的 watch 已死 | 新会话认领为 owner，照常监听 |
| **SessionEnd**（`SessionEnd` 钩子接线 `.qoder/hooks/watch-arm.sh`） | owner 显式释放真源后退出；非 owner 结束不误释放 |
| 空载荷（看门狗重臂）/ 解析失败 / 无 `session_id` / 判定命令本身失败 | **fail-open 走旧 token 语义**并留痕（唤醒链不塌） |

- **接管的 watch 判据带身份**：钩子把 owner 的 `session_id` export 给 `orch watch`
  （`watch.json` 的 `armed_by`）——只有 **owner 自己**的 watch 挡接管；看门狗重臂的
  无主替身（`armed_by` 空）不冒充，否则接管条款在看门狗常开的环境里永远触发不了。
- **不存在「转投其他会话」语义**（那正是本单要根治的）：owner 不在场时事件**不丢**——
  持久水位 + 迟到唤醒由 owner 下次挂载补上。`pysci-orch wake-claim status` 读当前 owner
  与 stale 判定；`wake-claim --force [--session-id <id>]` 是**应急强制迁移**（缺省=释放，
  下一钩子自动认领），**非常规步骤**——常规归属全部由钩子按载荷自动完成。
- **实测留痕**（2026-10-14 17:51–18:11，隔离 ROOT + 双 session_id 载荷喂真实钩子）：
  ```
  [17:51:36] 归属判定：claim：无 owner → 本会话自动认领（sid=sess-A）        ← 首会话零人工
  [17:52:20] 归属判定：deny：非 owner（owner=sess-A，心跳 44s…）→ 不监听不抢，退出
  [17:53:38] 空载荷（看门狗重臂/无 stdin）→ fail-open 走旧 token 语义        ← 重臂无主替身
  [17:55:38] 被取代（现持 174918-…）→ 让位退出                              ← v2 自仲裁仍成立
  [18:08:27] 归属判定：takeover：owner stale（心跳 11307s > 1800s 且其 watch
             已死）→ 自动接管（sess-A → sess-B）                            ← 替身在场也不挡接管
  [18:11:07] SessionEnd：release：owner 会话结束 → 显式释放（sid=sess-B）→ 钩子退出不监听
  [18:11:23] 归属判定：claim：无 owner → 本会话自动认领（sid=sess-C）        ← 干净迁移
  ```
  实测当场抓到并修掉一个接线缺陷：看门狗重臂实例会**继承**派生方的 `PYSCI_WAKE_OWNER`
  （替身 watch 盖上 owner 的 `armed_by` → 把接管永久挡死）——现由钩子 `unset` +
  `watchdog._rearm` 剥离双保险（各有回归测试钉住）。

**迟到唤醒 + 看门狗（2026-10-14 自动化加固 a1；用户实测发现 02:30–07:30 五小时唤醒真空）**：

- **取证：watcher 被「无声杀死」，而非自身出错**（全部可复核）。隔离 ROOT 复现
  （`QODER_PROJECT_DIR` 指向临时目录，`bash watch-arm.sh`）：
  ```
  # 起一任
  [2026-10-14 08:30:57] 接任 watch（token=2728-1791937856)
  # taskkill /F 杀掉后 —— 日志**零新增**
  [2026-10-14 08:30:57] 接任 watch（token=2728-1791937856)
  ```
  而 watcher 的**所有正常退出路径都留痕**：token 被他人改写 → `被取代（现持 someone-else）→ 让位退出`；`orch watch` 返回非 0 → `watch rc=1 → 退出`；自然超时不退出（TIMEOUT 后继续循环）。⇒「零留痕」只能来自**外部终止**（进程被杀、没机会落盘）。与 02:22–07:40 现场同形（token 迁移 + 会话 UUID 迁移 f80af65f→6140625f，疑似会话重载连带清理其孤儿钩子进程——此点无法从进程内证实，故不做断言，只以看门狗兜底）。
- **迟到唤醒（持久水位）**：`arm()` 不再把对比基线重置为「挂载瞬间快照」，而是**继承上一任持久水位**（`watch.json` 的 `base`；首次无历史才用当下快照种子）。聋期（无监听期间）产生的事件在下一次**任何**挂载的**首轮轮询**即命中唤醒，detail 标注 `〔迟到唤醒：发生于挂载前（聋期）〕`；水位单调（`advance_base` 落账），重复挂载不重复消费。实测留痕：
  ```
  挂载前基线 events_seq = 0
  聋期（无人监听）产生的事件 seq = 1
  新挂载继承持久水位（catchup=True），基线 events_seq = 0
  → 唤醒命中：QUEUE-EMPTY run-022113
     detail = devops 队列消化完：处理 3 项 〔迟到唤醒：发生于挂载前（聋期）〕
  落账后水位 events_seq = 1（单调，不重复消费）
  ```
- **看门狗（独立于会话/钩子生命周期）**：`pysci-orch watchdog start|stop|status` 拉起一个**分离进程**，每 60 s 读 `watch.json` 的 `heartbeat_at`——心跳新鲜（≤300 s；取值 **≥ watch 自身挂载宽限 240 s**，免把「刚 arm 完就被杀」误判成活着）不动，**stale 即重臂**一条 `.qoder/hooks/watch-arm.sh` 监听循环（它自抢 token 并循环 `orch watch`，**不新造第二套循环**），带 ≥300 s 冷却防堆进程。**`stop` = 广播停止纪元**（backlog 20261014-135935-devops）：写
  `{"scope":"all","requested_at":…}` 而非单 pid——**一次停止覆盖同刻全部实例**（含不持锁的
  孤儿；实测病体＝两实例互抢锁记录、旧 stop 只停一个），**锁空闲照样投递**（旧语义在此返回
  「无需停止」、孤儿永远收不到）；各实例按**自身启动时刻**消费纪元，晚于纪元启动的新实例
  不受影响（调度器/下一跳照常拉起——「不留孤儿」由此改为「停得干净、重启照常」）。
  OS 侧重启持久化（每 5 min 幂等 `orch watchdog start`，`pythonw` 无窗口）：`orchestration/scripts/watchdog-scheduler.ps1 -Mode install|uninstall|status`。实测留痕：
  ```
  看门狗 summary：rearms=1 checks=2 stopped=cooperative
    [▶] 看门狗启动（心跳stale）
    [→] 重臂监听循环（>300s 无心跳）
    [√] 看门狗退出（已释放锁）
  监听替身是否被真实拉起：是
  ```
- **三级防线（谁在哪一级兜底，a1–a3 定型）**：
  1. **会话级·watch 钩子**——`Stop` 钩子每回合接管 `watch-arm.sh`（token 竞争、循环 `orch watch`）：有监听时命中即 `exit 2` 唤醒。主路，**零轮询**。
  2. **会话级·队列感知巡更**（`watch` 的 `PENDING-WORK` / `STALL` 伪事件，与 ① **同一循环同帧**判定，走同一条 `detect → echo → exit 2` 唤醒链）——兜「**没有事件但状态该看**」：队列 / devops backlog / 事件流 / `data/taskbooks/` **有活却没人干**，或条目 `in_progress` 停滞超期。判据单一实现 `watch.queue_pressure`（四条任一即「非空」：backlog pending；成员队列 pending 且**该成员无存活 worker**；事件流有超 30 min 未排空；taskbooks 有 `new` 态）。**自动起止**：队列全空 → 纯 TIMEOUT 循环**零唤醒**，组长**不手工挂/撤 cron**（2026-10-14 用户设计，手工巡更 cron 退役）。同类事件 **≤1 次/30 min**（`registry.watch.pending_throttle_s` / `stall_threshold_s` 可配），节流账随 `watch.json` 持久化（重启不重置）。它与铁律「禁止 sleep-轮询」**不冲突**——禁的是拿轮询**等工作完成**，巡更不是驱动工作的机制。就位步骤与可证伪判据见 orchestration 技能 `references/patrol-watchdog.md`。
  3. **机器级·`pysci-watchdog`**（`orch watchdog start|stop|status`）——listener/watchdog 被无声杀死、会话消亡连坐时于阈值内重臂。**注册/卸载**（重启/重登后自动点火）：
     ```
     powershell -ExecutionPolicy Bypass -File orchestration\scripts\watchdog-scheduler.ps1 -Mode install
     powershell -ExecutionPolicy Bypass -File orchestration\scripts\watchdog-scheduler.ps1 -Mode uninstall
     ```
     该 PS1 **纯 ASCII** 有意为之（PowerShell 5.1 对无 BOM UTF-8 按 ANSI 解码，CJK 被解成反引号转义下一行 → ParserError；仓库内 `.ps1` 一律遵守）。
  三层**互相独立**：任何一层坏掉，另两层仍在。
- **日志完整化（hook v3）**：环境缺失 / token 写失败 / token 自抢分支均留痕，另加 `TERM/INT` 陷阱留痕（SIGKILL 无法拦，故由看门狗按心跳判死重臂）。
- **已知残余风险**：① SIGKILL 类外部终止由看门狗兜底，但**阈值 300 s 内**仍有真空（心跳新鲜期不会重臂）；② 看门狗本身若被同一类外部杀手干掉，只有 `watchdog-scheduler.ps1`（OS 任务计划）能在下一个 5 min 点重臂——未注册该任务时，重启后看门狗不在场；③ `schtasks` 默认当前用户交互令牌，**用户未登录时任务不跑**（此时也无组长会话可唤醒，影响面为零）。

#### 夜航模式（PSEP_2，2026-10-14 晚定型；规格 `data/others/original_designs/PSEP_2.md`）

用户离机时段需无人值守推进：**手动单开关** `data/pysci_settings.toml [night] enabled`——
**热切换**（钩子每次触发现读文件，无需重启会话）；读取失败/键缺席/非法值一律回落 `false`
并出声（`settings.night_enabled` 读侧 fail-open）。**关闭即回到现状、零残留**。三层：

- **① 政策注入层**（`.qoder/hooks/night-policy.sh`，挂 **SessionStart + PostCompact**）：
  ON 时经既有钩子体系 `hookSpecificOutput.additionalContext` 注入**政策全文＝用户授权书**
  （持续推进不等用户 / 一般决策自行执行且事后追溯责任在用户 / 高风险**咨询 twin 且其决策权
  等同用户**（leader-only.md §3 限时预授权；twin 不可达降级副组长+自行决断+留痕）/ 两终态
  「全部完成或严重阻碍」/ 禁把 LLM 当定时器 / 权限弹窗属硬阻塞）；OFF 零注入零噪声。选这两
  个触发点而非每回合 Stop：逐回合注入烧 token，而上下文压缩正是政策最易丢的时刻。判定与
  文本单点 `tools/night.py`（NIGHT_POLICY）。
- **② 判据层**：`.qoder/hooks/night-gate.sh`（`PreToolUse` matcher=AskUserQuestion）——夜航内
  **提问工具成不可能**（`exit 2` + 转 `orch consult-twin` 指令）；`watch.queue_pressure`
  第六条夜航判据——`state/mirror-pending.json` 非空 → 夜航 `PENDING-WORK`（沿既有 30 min
  节流；非夜航不加）。两层皆 **fail-open**（任何异常放行/零输出，不塌唤醒主链路）。
- **③ 保险丝层（缓行，规格先立）**：仅当「有未竟事项且组长无活动」的停滞才打回，两终态
  一律静默；PENDING-WORK 已覆盖大部分停滞面，故**只预留** `periodic_wake_s` 键、实施前勿设。
- **配套审计（唯一软依赖）**：twin 代裁全链可追溯——记录 `evals/taste/<case>.json`（案卷摘要/
  选择/理由/置信度/时间/**夜航预授权**标记）+ **单一账目源** preference-journal（`channel=twin`
  + `case_id`）+ `orch twin-review --list/--confirm/--reject`（否决入 override 账，供后续改进
  追溯）。判据自动化面：`test_night_policy.py` / `test_night_gate.py` / `test_twin_trace.py`；
  操作步骤与可证伪判据另见技能 `references/patrol-watchdog.md` §5。
- **手动用法（降级/兜底仍适用）**：组长以 Bash `run_in_background` 启动，进程退出即原生完成
  通知唤醒本会话；一次 watch 报一批事件即退。**交办/唤醒/watch 输出不再催挂**（旧 `watch.mount_lines`
  [NEXT] 段已删，backlog 20261013-222533-leader）——只回报「已自动挂载」。
  **硬警示**：`orch watch` 必须作为后台 Bash 调用的**唯一顶层命令**——
  禁止用 `&`/`&&`/管道把它嵌进别的命令，嵌入即被父任务连坐杀死、事件通知随之丢失。
- **边界**：watch 随组长会话消亡，**不影响 detached drain**；基线快照与心跳落
  `state/watch.json`（gitignore：纯运行态）。**抑制帧与瞬态空值（`.done` 半写等）不推进
  基线**——被压住/读不出的身份保持原值，下一帧继续比较，绝不把账吞死。每帧现场（ts、
  锁态、锁 run、done 身份、抑制原因）进 `recent` 环形缓冲，命中/超时写 `exit_reason`；
  被外部杀死的那任没有退出记录，下次挂载或 `orch status` 会读出这条「遗言」。漏挂/会话
  中断期间的账由下次 `orch status` 的 watch 段**一次性补账**（报完即推进基线，重复运行不
  刷屏；watch 仍在挂载时 status 不抢账，免双份唤醒）。**单一消费水位**（2026-10-14 12:44
  双唤醒实锤后加）：命中事件的**身份**记进 `watch.json` 的 `consumed` 账（读盘为准、跨实例
  跨挂载继承），唤醒前先过它——被取代的 watch 实例即便仍持自己的旧基线，也拿不到第二次
  唤醒权（`watch.filter_consumed`/`mark_consumed`）。同刻多事件按 CRASH > QUOTA-STOP >
  CHANNEL-STOP > NEEDS-LEADER > TASKBOOK-NEW > JOB-DONE > QUEUE-EMPTY > QUEUE-COMPLETE >
  **PENDING-WORK > STALL** > SALVAGE 取主事件、其余以「同时」附注——**队列感知伪事件排在
  真实事件之后、SALVAGE 之前**（与真实事件同帧时让位，独处时才是主行），否则 nudge 会把
  a1「首个唤醒=真实事件」的契约挤掉；SALVAGE 仍保持最低优先级。**弃 Monitor 工具方案**
  （会话侧注册更重，且同样要组长发起）。

---
