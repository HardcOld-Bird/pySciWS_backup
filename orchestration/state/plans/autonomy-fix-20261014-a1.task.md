# 自动化加固 a1：watch 唤醒链加固（迟到唤醒 + 看门狗自愈）

（计划 autonomy-fix-20261014 环节 a1；背景=2026-10-14 用户实测发现 02:30–07:30
五小时唤醒真空。**本环节是当前最高优先级**，直接关系「自动化有名无实」指控的根治。）

## 故障证据（2026-10-14 组长取证，全部可复核）

1. `.qoder/hooks/watch-arm.log`：`02:22:20 接任（token=3754-…）` → `02:23:02 被取代`
   （旧者让位）→ **02:23–07:40:30 零日志** → `07:40:30 接任（token=2035-…）` 且此后
   **旧者无「被取代」留痕**。watcher 所有正常退出路径都留痕（命中/被取代/rc 退出），
   零留痕 = 进程被**无声杀死**。期间会话 UUID 迁移（f80af65f→6140625f，上下文压缩/
   会话重载），疑似孤儿钩子进程随会话清理。
2. `orchestration/state/events.jsonl` 唯一一行 = `02:30:23 seq=1 QUEUE-EMPTY
   20261014-022113`（drain 完成）——**事件发射层（m5）正常**。
3. `state/watch.json`（07:40:30 挂载）的 `base` 已含 `done=20261014-022113` /
   `done_completed=02:30:23` / `events_seq=1`——**聋期事件被新基线吸收，永不触发**。
4. 钩子面：`.qoder/settings.json` `Stop` → `watch-arm.sh`（asyncRewake=true,
   timeout=15000）；脚本内部 token 竞争（`watch-holder.token`）+ `while` 循环跑
   `orch watch --timeout 120`；**每回合点火依赖组长会话活跃**——不被唤醒就没有下一次
   点火，死循环依赖。

## 目标（三件，缺一不验收）

1. **迟到唤醒（持久水位）**：watch 挂载/轮询的对比基线改为**持久水位**（如
   `events-seen.json` 序号 + done 标记集合，或等价单调解），不再用「挂载瞬间快照」
   吞事件。聋期（无监听期间）产生的事件，下次任何挂载**立即命中唤醒**（迟到唤醒），
   事件 detail 标注 late/迟发来源。水位单调，不重复消费。
2. **静默死亡取证 + 看门狗自愈**：
   - **取证**：复现 watcher 静默死亡（会话重载/进程清理/休眠恢复等候选逐一证伪），
     结论写 README §4.5 或代码注释（含复现命令与实际输出）；
   - **看门狗**：独立于会话/钩子生命周期的自愈——watch 心跳（watch.json
     heartbeat_at）stale 超阈值（建议 ≤5min）自动重臂监听。实现自选（Windows 计划
     任务 / 独立 detached 循环 / OS 调度器拉起的 watchdog 脚本），**必须实测证明**：
     kill watcher → 阈值内看门狗重臂并留痕；
   - **日志完整化**：token 每次写入、进程每次退出（含被取代）必留痕，消除「02:23 后
     零留痕」类盲区。
3. **回归测试**（钉死本场景）：① 聋期产生事件 → 新挂载在 ≤1 个轮询周期内迟到唤醒；
   ② watcher 被杀 → 看门狗阈值内重臂；③ 水位单调性（重复挂载不重复唤醒）。

## 验收预期（机械）

- `uv run --no-sync pytest` 全绿（1342+ 基线）+ 上述三类新测试；
- 实测留痕：制造聋期事件 → 迟到唤醒日志/水位读数；kill watcher → 看门狗重臂日志；
- README §4.5 更新为机制实况（迟到唤醒语义、看门狗、已知残余风险）；
- 现有 `orch watch` CLI 语义兼容（调用方不回归）。

## 白名单（涉改文件）

`src/pysci/skills/orchestration/tools/**`（watch.py / events.py / orch.py 等）、
`.qoder/hooks/**`（watch-arm.sh、看门狗脚本）、`.qoder/settings.json`（钩子注册如需）、
`tests/**`、`orchestration/README.md`（§4.5）、`.gitignore`（按约定已附）、
OS 调度器注册脚本与安装说明（落 scripts/ 或 orchestration/，注明安装/卸载方式）。

## 纪律（违约即返工）

- 事件流只追加；迟到唤醒不重复消费（水位单调）；
- **grep 判据约定（20261014-075230-devops）**：禁词一律运行期拼接/转义（防常量折叠进
  .pyc），判据限定 `*.py` 或排除 `__pycache__`；
- 看门狗属常驻机制：安装须幂等、可卸载、不留孤儿进程；实测含「卸载后不留痕」；
- 不动 pyproject.toml；确需改则改后必须 `pysci-dev venv-sync`；
- 编辑项目文件用文件编辑工具；CLI 首选裸 `pysci-X`；
- 交付按 pod 交付协议 XML（`<result>`/`<blocked>`）；blocked 带 tags 与 reason；
  改进建议带合法 `kind` 属性。
