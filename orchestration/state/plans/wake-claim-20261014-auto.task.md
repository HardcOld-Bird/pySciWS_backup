# wake-claim 补充件：自动认领语义（用户升级设计 2026-14 晚，取代手动认领为主的方案）

## 设计修订（用户裁决方向：机制优于纪律）

原任务书以 `wake-claim` 手动认领为主——用户指出这是「硬保证（钩子必挂）→软步骤
（组长记得）」的退步。查证 hook-config 文档：**钩子 stdin JSON 自带 `session_id`**
（现行 watch-arm.sh 用 `cat >/dev/null` 丢弃了它）。改为**零人工自动认领**：

1. **读 stdin 取 `session_id`**（watch-arm.sh 不再 `cat >/dev/null`，改为解析留用；
   解析失败 fail-open 走旧 token 语义并留痕）。
2. **owner 真源**（`orchestration/state/wake-owner.json`）：
   - 无 owner → 本会话**自动认领**（首会话即主会话）；
   - owner == 我 → 续心跳、照常持 token；
   - owner ≠ 我 → **永不抢占**（次要会话自动排除），退出留痕；
   - owner stale（心跳超阈值且其 watch 已死）→ 新会话**自动接管**；
   - 注册 `SessionEnd` 钩子：owner 会话结束显式释放（干净迁移）。
3. **`wake-claim` 命令降级**为显式强制迁移工具（`--force` 覆盖 owner），非常规步骤；
   `status` 可读当前 owner。
4. 迟到唤醒兜底不变：owner 缺席期间事件不丢，owner/接管者下次挂载即补唤醒——
   **不存在「转投其他会话」语义**。

## 验收预期（在原任务书基础上替换目标 1/2 的人工语义）

- `uv run --no-sync pytest` 全仓全绿（3143+ 基线）+ 新增：首会话自动认领/异 id
  不抢/owner stale 自动接管/SessionEnd 释放/解析失败 fail-open 五类测试；
- 实测留痕：同项目模拟两会话（不同 session_id 的 stdin 载荷喂钩子）→ 仅 owner
  唤醒；owner 消亡 → 新会话接管；
- README §4.5 / patrol-watchdog.md / 技能同步：唤醒路由=自动会话归属，无手工步骤。

白名单与纪律沿原任务书+补充件；迟到唤醒/消费账语义不许再破坏；grep 判据约定沿用。
