# 唤醒权认领制（用户设计方向 2026-14 晚）：主会话定向唤醒，次要会话不再被误投

## 事故现场

用户习惯：主组长会话工作时另开次要会话处理琐事。watch 钩子的 token 竞争是
「最新在听者胜」（v2 为修会话重启丢唤醒的规则）——次要会话一有回合即抢走唤醒权，
drain/事件的 exit-2 唤醒被投到次要会话（污染其上下文，正是用户要避免的）。
凌晨「被取代却无接任」的谜之日志疑为同源。

## 目标（三件，缺一不验收）

1. **主会话认领**：`pysci-orch wake-claim [status|release]`——主会话跑一次即认领
   唤醒权（真源 `orchestration/state/` 一文件，含认领时刻与会话身份）。钩子仅在
   「本钩子属于 owner 会话」时抢 token；非 owner 钩子**永不抢占**（可只读旁听，
   不得 exit 2 唤醒）。
2. **会话身份取证**：钩子运行环境里可用什么区分会话（QODER_* 环境变量面实测：
   在 watch-arm.sh 加一次性诊断日志枚举 env，取证后固化判据；若无可靠会话 id，
   退化方案：认领时把 owner 指纹（如 token 来源 pid 链/用户指定名）写入真源并说明
   局限）。
3. **安全语义（a1 迟到唤醒兜底，无需接管者）**：owner 不在场时事件**不丢**——
   持久水位+迟到唤醒保证 owner 下次挂载立即补唤醒；owner 永久迁移 = 新会话跑
   `wake-claim` 接管。文档写明：**不存在「转投其他会话」语义**（那正是本单要根治的）。

## 验收预期（机械）

- `uv run --no-sync pytest` 全仓全绿（含在途返工的基线）+ 新增：owner 钩子持权/
  非 owner 不抢/认领迁移/owner 缺席事件不丢（迟到唤醒接续）四类测试；
- 实测留痕：双会话模拟（同项目两次钩子触发）→ 仅 owner 被唤醒；
- README §4.5 / patrol-watchdog.md / 编排技能同步（唤醒路由语义 + wake-claim 用法）。

## 白名单（涉改文件）

`.qoder/hooks/watch-arm.sh`、`src/pysci/skills/orchestration/tools/**`（orch/watch/
watchdog）、`tests/**`、`orchestration/README.md`、`orchestration/skills/orchestration/**`、
`.gitignore`（按约定已附）。

## 纪律（违约即返工）

- 迟到唤醒/消费账语义不许再破坏（上一单刚打挂过，返工在先）；
- 钩子任何异常 fail-open（exit 0，不崩主链路）；
- grep 判据约定（20261014-075230）；不动 pyproject.toml（确需改→venv-sync）；
  编辑项目文件用文件编辑工具；CLI 首选裸 `pysci-X`；
  交付 XML（`<result>`/`<blocked>`）；建议带合法 `kind`。
