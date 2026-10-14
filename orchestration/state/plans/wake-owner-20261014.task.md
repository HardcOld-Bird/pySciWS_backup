# 唤醒权自动归属（用户设计定稿 2026-14 晚）：session_id 识别 + 零人工认领

（本单为自包含合并版，取代 wake-claim-20261014.task.md 手动认领方案——用户裁决
「机制优于纪律」：硬保证不许退化成软步骤。）

## 事故现场

用户习惯双会话并行（主组长会话工作 + 次要会话处理琐事）。watch 钩子 token 竞争
「最新在听者胜」→ 次要会话一有回合即抢走唤醒权，drain/事件被误投次要会话（污染其
上下文）。凌晨「被取代无接任」谜之日志疑同源。**关键查证（hook-config 文档）**：钩子
stdin JSON 自带 `session_id`（现行 watch-arm.sh 第一行 `cat >/dev/null` 把它扔了）。

## 目标（三件，缺一不验收）

1. **读 stdin 取会话身份**：watch-arm.sh 不再丢弃载荷——解析 `session_id`（及
   `hook_event_name`），解析失败 fail-open 走旧 token 语义并留痕。
2. **owner 自动归属**（真源 `orchestration/state/wake-owner.json`：session_id +
   claimed_at + last_turn_at）：
   - 无 owner → 本会话**自动认领**（首会话即主会话，零人工）；
   - owner == 我 → 续心跳、照常持 token；
   - owner ≠ 我 → **永不抢占**（次要会话自动排除），退出留痕；
   - owner stale（心跳超阈值且其 watch 已死）→ 新会话**自动接管**；
   - 注册 `SessionEnd` 钩子：owner 会话结束显式释放（干净迁移）。
3. **wake-claim 命令降级**：`status` 读当前 owner；`--force` 显式强制迁移（应急），
   **非常规步骤**。迟到唤醒兜底不变：owner 缺席期间事件不丢、下次挂载补唤醒；
   **不存在「转投其他会话」语义**。

## 验收预期（机械）

- `uv run --no-sync pytest` 全仓全绿（3143+ 基线）+ 五类新测试：首会话自动认领/
  异 id 不抢/owner stale 自动接管/SessionEnd 释放/载荷解析失败 fail-open；
- **实测留痕**：同项目双会话模拟（不同 session_id 的 stdin 载荷喂钩子）→ 仅 owner
  唤醒；owner 消亡 → 新会话接管；
- README §4.5 / patrol-watchdog.md / 编排技能同步（唤醒路由=自动会话归属，零手工）。

## 白名单（涉改文件）

`.qoder/hooks/watch-arm.sh`、`src/pysci/skills/orchestration/tools/**`（orch/watch/
watchdog）、`tests/**`、`orchestration/README.md`、`orchestration/skills/orchestration/**`、
`.gitignore`（按约定已附）。

## 纪律（违约即返工）

- 迟到唤醒/消费账语义不许再破坏（dispatch-unify 曾打挂，已返工修绿）；
- 钩子任何异常 fail-open（exit 0，不崩主链路）；
- 机械验收不过不许合并（今日纪律事故留痕）；
- grep 判据约定（20261014-075230）；不动 pyproject.toml（确需改→venv-sync）；
  编辑项目文件用文件编辑工具；CLI 首选裸 `pysci-X`；
  交付 XML（`<result>`/`<blocked>`）；建议带合法 `kind`。
