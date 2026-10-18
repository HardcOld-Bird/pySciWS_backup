# 新会话/新机器就位与唤醒权绑定（patrol-watchdog 分册 02）

> 由 `patrol-watchdog.md` 按 basic.md §1「超预算即拆分」移出（wake-bind-20261018 s2，
> 2026-10-18）。主册保留三级防线/巡更/判据；本册是开机清单与唤醒权/监听者语义。

## 新会话 / 新机器就位（开机步骤）

1. **看门狗（机器级③）**：`pysci-orch watchdog status` → 不是 `running` 就
   `pysci-orch watchdog start`（幂等，锁在手即拒绝）；**注册 OS 计划任务**（重启/重登后
   自动点火，免记手工）：
   ```
   powershell -ExecutionPolicy Bypass -File orchestration\scripts\watchdog-scheduler.ps1 -Mode install
   ```
   卸载同命令 `-Mode uninstall`；进程另用 `orch watchdog stop`。该 PS1 **纯 ASCII 有意
   为之**（PS 5.1 无 BOM UTF-8 按 ANSI 解 → CJK 吞行；仓内 `.ps1` 同规）。
2. **唤醒权（程序化绑定主会话，wake-owner）**：钩子自登记存活会话（`state/sessions-live.json`：
   `session_id/cwd/pid/created_at/first_seen`），走 `wake-claim --check`（判定单点）：
   - 候选集 = **存活 ∧ cwd==项目根**（pod 直聊永不入候选）；
   - **主会话 = 创建更早者**（OS `create_time` 优先、`first_seen` 回退；平局按 `session_id`）；
   - owner 记录指向者仍存活 → 保持/恢复（watchdog 重启按记录恢复）；
   - owner 缺席/死亡 → 最早存活候选**自动补位**；晚创建者恒 deny（不监听/不抢/绝不 `exit 2`）；
   - **存活性 PID 优先**：心跳过期而 PID 存活 = 存活（长回合不判死）；PID 缺失才看心跳；
   - `SessionEnd` 注销登记；owner 结束立即补位最早剩余存活候选（墓碑只留痕、不阻止补位）。
   `pysci-orch wake-claim status` 查候选集/创建时序/owner 存活性；
   `--force --session-id <id>` 仅**应急**（目标存活期间保持），`--force` 不带 id = 清除记录
   （下一钩子按创建时序自动补位）。载荷缺失/解析失败 fail-open 留痕。
   owner 缺席期间事件**不丢**（持久水位＋迟到唤醒）。
3. **listener（会话级①）**：无需手工——`Stop` 钩子下回合自动接管（`watch-arm.sh` 每回合
   幂等自启动看门狗）；漏挂由 `orch status` 补账（`reconcile`）。**只有 owner 侧挂载可消费
   事件**：看门狗重臂的无主替身/被自动补位取代的旧实例命中时**不记消费账、不推进水位**，
   事件留给 owner 下次挂载的迟到唤醒。
4. **巡更（②）**：**无需动作**——队列感知自动起止。仅主册 §2「降级注」那种全灭态才手工挂 cron。
5. **自检**：`pysci-orch watchdog status` 四行（看门狗态/心跳年龄/listener/逻辑实例数——口径
   = distinct `--run-log`，非进程数）应为 `running`/`<300s`/`活着`/`1`；`orch status` 的
   watch 段应显示已挂载或可补账。
