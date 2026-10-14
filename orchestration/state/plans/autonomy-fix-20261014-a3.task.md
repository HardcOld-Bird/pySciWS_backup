# 自动化加固 a3：e2e 防回归 + 文档实况 + 巡更规程入技能

（计划 autonomy-fix-20261014 环节 a3，收官环节。
交接物：a1 迟到唤醒/看门狗（watch.py/watchdog.py/hook v3/watchdog-scheduler.ps1）、
a2 inspector 自调度（inspectd.self_patrol_tick / 队列超时告警 / TUI 读数）。）

## 目标（三件，缺一不验收）

1. **跨组件 e2e 防回归测试**（真实联动链，不是单测拼凑；能仿真即仿真、标注真实面残余）：
   - **聋期丢唤醒场景**（2026-10-14 02:30 现场复刻）：无监听期间产生事件 → 下次挂载
     ≤1 个轮询周期内**迟到唤醒**发生且 `late` 标注正确、水位单调（重复挂载不重复消费）；
   - **watcher 被杀 → 看门狗重臂**：kill 监听 → 心跳 stale（≤300s 阈值）→ 看门狗重臂
     且留痕；
   - **inspector 自调度链**：watchdog tick → `self_patrol_tick` → 队列 → worker。
     ⚠ **用户指令（2026-10-14）：inspector 实测移夜间**——e2e 用开关隔离/monkeypatch 模拟
     开启态，**不得真实拉起巡检**（当前全局开关=off，保持不动）；
   - **队列超时告警链**：入队 30min 未处理 → 事件层发出 → TUI 读数可见。
2. **文档实况与一致性**：
   - `orchestration/README.md` §4.5/§6/§7 校对至与实现一致（a1/a2 已各改一处，本环节
     通读对齐）：三级防线描述——watch 钩子（会话级）/ 巡更 cron（会话级、自适应）/
     `pysci-watchdog` OS 计划任务（机器级，`watchdog-scheduler.ps1 -Mode install`
     注册步骤与卸载）；
   - 一致性测试更新（沿 `test_readme_refactor_closeout` 惯例，文档与实现不许漂移）。
3. **巡更规程入编排技能**（未来组长开机就位，零上下文依赖）：
   - 落 `orchestration/skills/orchestration/SKILL.md`（注意 **≤8KB 预算**，细则放
     `references/*.md` 并留一行索引）：组长**活跃期**挂 30min 巡更 cron（prompt 模板：
     查状态/对照事件水位推进/异常留痕）、**全空闲自动撤销**、新会话/新机器就位步骤
     （含 PS1 注册）；与铁律「禁止 sleep-轮询」的关系说明（cron 是兜底巡更，不是轮询
     工作机制）。

## 验收预期（机械）

- `uv run --no-sync pytest` 全绿（1357+ 基线）+ 上述四条 e2e 场景测试；
- **inspector 全局开关终检 = off**（用户指令；测试不得改变它）；
- README/技能与实现一致性测试过；技能文件 ≤8KB（delivery-gate 会机械校验）；
- 全仓无回归。

## 白名单（涉改文件）

`tests/**`、`orchestration/README.md`、`orchestration/skills/orchestration/**`
（SKILL.md/references）、`src/pysci/skills/orchestration/tools/**`（仅 e2e 所需小修）、
`.gitignore`（按约定已附）。

## 纪律（违约即返工）

- e2e 优先仿真与 monkeypatch：**不得真实拉起 inspector 巡检**（用户指令）；
- 文档只写实测过的命令与行为（§2 上报纪律）；不写兑现不了的承诺（f1 规矩）；
- **grep 判据约定（20261014-075230-devops）**：禁词运行期拼接/转义，判据限 `*.py`；
- 技能/全量注入文档守预算（SKILL.md ≤8KB）；
- 不动 pyproject.toml；确需改则改后必须 `pysci-dev venv-sync`；
- 编辑项目文件用文件编辑工具；CLI 首选裸 `pysci-X`；
- 交付按 pod 交付协议 XML（`<result>`/`<blocked>`）；blocked 带 tags 与 reason；
  改进建议带合法 `kind` 属性。
