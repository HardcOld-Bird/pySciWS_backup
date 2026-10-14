# 自动化加固 a2：inspector 自主巡逻（闲时自调度 + 队列超时告警）

（计划 autonomy-fix-20261014 环节 a2；背景=2026-10-14 用户实测：02:30–07:30
inspector 零巡检、02:21 入队的巡逻积压 5 小时——「闲时自主巡逻」名不副实。
交接物：a1 的 `watchdog.py`（`pysci-orch watchdog start` 常驻自愈循环，
60s tick）、统一事件层 `events.py`、hook v3。）

## 故障证据

- `inspector-runs/` 在 2026-10-14 02:08 与 07:42 之间**零 run**；`insp-20261014-022154-patrol`
  02:21:54 入队后 `pending=1` 积压整夜。
- 成因：inspector 是「orch 命令拉起」式（`orch inspector-up`），无自调度；唤醒链断
  （a1 已修）+ 无命令 = 永久沉睡。status 里「闲时自主巡逻」的承诺与实现不符。

## 目标（三件，缺一不验收）

1. **闲时自调度巡检**：无任何 orch 命令时也按 cooldown（3600s）自主巡检。实现路径自选：
   - 复用 a1 看门狗 tick 顺带触发（**注明耦合点，不新造第二套调度循环**），或
   - inspector worker 自调度（空闲超时自再入队/自唤醒）。
   既有纪律全保留：让路（plan-run / devops-drain）、backlog>12/bench>6 节流、cooldown、
   hops/wallclock 预算闸。
2. **队列 pending 超时告警**：入队 30min 未处理 → 接入统一事件层发事件（复用
   NEEDS-LEADER 或新类型，接 `events.py` 词表与唤醒策略），TUI「盲审与审查」/「队列与
   运行」读数可见超时项。**不许静默积压**。
3. **名实相符**：README §6/§7 与 `orch status` 文案同实现对齐（如自调度语义、超时阈值）；
   占位/承诺文案遵守 f1 已立的规矩（不写兑现不了的承诺）。

## 验收预期（机械）

- `uv run --no-sync pytest` 全绿（1352+ 基线）；新增：**模拟「组长 5h 无操作」**（假时钟、
  零 orch 命令）→ 巡检 run 数 ≥ floor(5h/cooldown) 或等价自主活动留痕；pending 超 30min
  → 事件发出且 TUI 读数可见；
- **实测留痕**：令 inspector 空闲过一个 cooldown 周期（无任何 orch 命令）→ 自调度巡检
  发生（inspector-runs 日志）；
- 既有 inspector 纪律测试全过（让路/节流/预算闸不回归）。

## 白名单（涉改文件）

`src/pysci/skills/orchestration/tools/**`（inspectd.py、events.py、watchdog.py 如取耦合
方案、tui/modules.py 读数）、`tests/**`、`orchestration/README.md`（§6/§7）、
`.gitignore`（按 20261014-012859-devops 约定已附）。

## 纪律（违约即返工）

- **不造第二套调度/唤醒循环**：优先挂进 a1 看门狗 tick 或既有拉起链，注明耦合点；
- 事件层词表单一实现（events.py TYPES/WAKE 表扩展，不另立事件通道）；
- **grep 判据约定（20261014-075230-devops）**：禁词运行期拼接/转义，判据限 `*.py`；
- 不动 pyproject.toml；确需改则改后必须 `pysci-dev venv-sync`；
- 编辑项目文件用文件编辑工具；CLI 首选裸 `pysci-X`；
  Windows 命令走 Python subprocess 传 argv（Git Bash `/` 前缀坑）；
- 交付按 pod 交付协议 XML（`<result>`/`<blocked>`）；blocked 带 tags 与 reason；
  改进建议带合法 `kind` 属性（上一环节有条建议因 kind 为空被 gate 拒收，勿再犯）。
