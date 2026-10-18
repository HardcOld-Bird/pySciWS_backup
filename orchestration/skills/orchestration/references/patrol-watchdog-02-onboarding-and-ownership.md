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
2. **唤醒权（绑定主会话，wake-owner）**：钩子每回合读 stdin 载荷 `session_id` + `cwd`，
   走 `wake-claim --check`（判定单点）：仅**无 owner 记录**（冷启动/人工释放后）自动认领
   （首会话即主会话）、owner 续心跳；**异会话与 pod 直聊永不抢占**（不监听、不抢 token、
   绝不 `exit 2`）。**自动转让已取消**（用户 2026-10-18 裁决，twin 裁 A）：
   - owner stale **不再自动接管**、`SessionEnd` **不再让任意会话自动升主**（释放写墓碑）；
   - 迁移**只走人工** `pysci-orch wake-claim --force --session-id <id>`；
   - `--force` 不带 id = 人工释放（删除记录，下一钩子自动认领）。
   `pysci-orch wake-claim status` 查当前 owner/心跳并给人工迁移指引。载荷缺失/解析失败一律
   fail-open 走旧 token 语义留痕。owner 缺席期间事件**不丢**（持久水位＋迟到唤醒）。
3. **listener（会话级①）**：无需手工——`Stop` 钩子下回合自动接管（`watch-arm.sh` 每回合
   幂等自启动看门狗）；漏挂由 `orch status` 补账（`reconcile`）。**只有 owner 侧挂载可消费
   事件**：看门狗重臂的无主替身/被迁移取代的旧实例命中时**不记消费账、不推进水位**，
   事件留给 owner 下次挂载的迟到唤醒。
4. **巡更（②）**：**无需动作**——队列感知自动起止。仅主册 §2「降级注」那种全灭态才手工挂 cron。
5. **自检**：`pysci-orch watchdog status` 四行（看门狗态/心跳年龄/listener/逻辑实例数——口径
   = distinct `--run-log`，非进程数）应为 `running`/`<300s`/`活着`/`1`；`orch status` 的
   watch 段应显示已挂载或可补账。
