# pySci 多 Agent 编排顶层设计（唯一权威）

**状态**：v3.0（2026-10-15 依五归档重写）。本文是编排体系**唯一权威**（架构/目录契约/协议/现行设计）。**历史沿革不搬**——阶段史、裁决全文、实现细则活在 git 历史与五归档 `data/others/original_designs/`
（`1_old_README.md` 旧权威／`2_REFACTOR`（归档） 自改进 E0–E7／`PSEP_1.md` 控制台+队列化／`PSEP_2.md`
夜航／`PSEP_3.md` superstep·返工流·裁决书+E6·路由）。细节块（同目录 `references/`）：
`plan-and-routing.md`／`wake-and-events.md`／`quality-and-improvement.md`／`guards.md`／
`decisions.md`／`history-and-risks.md`——**按节搬迁，未砍契约**。

> **设计理念与权力结构（用户裁决，语义保留）**：用户是科研工作者，关注面与责任面最小化——仅在直接
> 影响产物质量或长期影响项目的顶层决策点出面；末端工作与基建维护由 LLM Agent 自动化；组长价值在
> **宏观视野**。基建合格标准：**即使 flash 级轻量模型也能驱动整套编排**（「下一步」由工具直接给出，
> 机械环节自动完成，组长只面对真正的决策点）。项目内**组长决定权高于用户**（用户有建议/指引权、无
> 强制否决权；组长对项目内一切事项有裁定权，承接 `basic.md` §1.1 并同样适用于编排体系本身）；例外：
> **护栏级条款**（红线/审批权/角色边界/门禁策略）修改须**用户明确授权**（§16）。
>
> **优先复用原生机制（用户裁决，语义保留）**：每个 pod 是完整 Qoder 项目，直接用原生项目级功能
> （rules/AGENTS.md 加载、技能发现、会话分储/resume/自动压缩、hooks、分层 settings）；一切自研（orch
> CLI、guards、技能部署）以「原生表达不了」为存在前提。

## 1. 角色与进程模型

**星形拓扑**：组长（TUI 主会话）派发 → headless 组员；**组长严禁末端工作**（例外：用户点名亲做）。
13 席 = 7 专职（sim figure writing theory lit drawing model3d）＋ 6 特殊：**deputy**（咨询/二审，近
平级：拒绝权+异议义务）·**reviewer**（主观质量重炮，可选非硬门）·**devops**（基建维护）·**bencher**
（harness 压测与 A/B 判决）·**twin**（审美代理：判断通道+终裁，§5.3/§15）·**inspector**（闲时巡检：
发现→分类→分流，§5.4/§6）。原生机制全部复用，`orch` 只薄封装（§1.1 实测记录移档）。

## 2. 目录契约

**唯一真源 `src/pysci/paths.py`**（`PROJECT_ROOT`/`ORCHESTRATION_ROOT`/`ORCH_STATE_ROOT`/`PODS_ROOT`/
`DATA_ROOT`/`research_*_dir()`）。`orchestration/` = README（权威）+ `skills/`（真本）+ `guards/` +
`pods/<id>/`（组员，从属 orchestration/）+ `state/`（运行态）+ `bench/`；`src/pysci/` 只**定义**、
`scripts/` 只**使用**；写 `data/` 前过 `assert_within_data()`。**写权限矩阵**（pod-guard 两层执法，§9）：自维护层放行（bench/outbox/AGENTS.md/自建 rules·skills）／
只读层拦截（charter/部署技能/settings/mcp.json/inbox）／pod 外拦（白名单除外）；名单经
`PYSCI_TASK_DIRS`+`PYSCI_READONLY_PATHS` 注入。

## 3. 记忆、会话、派发与协议

- **3.1 记忆**：全部复用原生加载（rules→charter→AGENTS.md→技能→references 按需）；常驻注入档 ≤8KB/
  文件（`basic.md` §1 三层预算，§9 机械执法）。
- **3.2 会话池**：原生分储 + `orch sessions` 薄封装；默认**续最近活跃会话**，组长确认归属；膨胀看
  `ctx_ratio`（不看 est_tokens）。
- **3.3 派发**：三入口（`dispatch`/`plan`/`queues`）统一走**成员 FIFO 队列**（§8）；worker 消化后把
  分类/正文/验收/建议/时长写回条目。
- **3.4 交付协议**：组员最终回复**含且仅含一个** `<result>`/`<blocked>` XML 块（互斥）+ 可选
  `<infra_suggestion kind="defect|usability">` + `<artifact check=…>`；`<blocked tags>` 词表在
  `state/failure-tags.json`；**delivery-gate 强制**格式。
- **3.5 模型预设路由（用户裁决 2026-10-12）**：max/flash 档位退役，改**提供商预设**（toml `[model]
  preset`，峰谷自动切）；逐项 `--effort` 覆盖。

## 4. orch CLI（pysci-orch）——组长唯一工作通道

**4.1 命令面（plan-first 分层，用户裁决）**：日常面 = `status`／`plan`／`next`／`taskbooks`／`queues`／
`events`／`approve`／`reject`／`watch`／`wake-claim`／`consult[-twin]`／`review`／`jury`／`stats`／
`bench-*`／`inspector-*`／`taste`；底层（降级/调试）= `dispatch`/`sessions`/`ledger`/`sync`。**全表与
旗标见技能 `references/command-surface.md`。**

**4.2 计划驱动（plan-first）**：一切工作走 plan；`steps[]` 元素二型**递归**（普通 step ｜ **superstep**
`{id, mode: serial|parallel, children}`），叶子在 state.json 扁平化为**点路径 id**、按 id 对账。
**就绪谓词**：serial 前兄弟须 done（完成级）／parallel 前兄弟须 started（派发级）；**kind:self**＝组长
自办（`plan advance --evidence` 收尾）；`advance` 收尾全部终态 + [NEXT] **三栏工作清单**、`run` 等全部
在途。**返工流**：TUI 三段式驳回 → `plan rework` 三档骨架 → 送审 gate（verdict=A 才派单）/两次不过
中止 → `taskbooks report` 全流程报告。**跨任务路由 `next`**（优先级→动作类别→序：任务卡片 + 四机械
推送点）。**全量见 `orchestration/references/plan-and-routing.md`**（细节块统一在
`orchestration/references/` 下——技能侧 `references/` 是**另一套**，勿混）。

**4.3 设计红线**：① 每条命令输出自含「下一步」；② fail-open（orch 故障按**附录 A** 降级，state 全可读）；
③ orch 自身也是 devops 维护对象，改进走同一 backlog 循环（§6）。

**4.4 统计系统（用户裁决）**：台账每跳记 credits + est_tokens + 耗时；计划完成自动呈现本计划总量/各成员
占比 + `orch stats` 独立入口（监测成员耗用、指导 harness/预设调整）。

**4.5 等待与唤醒模型（长任务零轮询）**：`events drain` 排空积压 → 推进；`Stop` 钩子每回合自动挂
`orch watch`（事件驱动，**禁止 sleep-轮询**）；**三级防线**（会话 listener／队列感知巡更伪事件
PENDING-WORK·STALL／机器级 watchdog）；**唤醒权自动归属**（钩子读 stdin `session_id`：自动认领/续
心跳/异会话永不抢占/stale 接管/SessionEnd 释放）。**全量见 `references/wake-and-events.md`**（手册见技能 `references/patrol-watchdog.md`）。

## 5. 质量保证（两层）

机械验收（**成员声明式** `<artifact check=…>` → 确定性脚本，零 LLM；不得阻碍灵活任务、组长可抽检）／
reviewer（主观学术质量，**可选重炮**非硬门）／**判决包**（四态强制：`approve` 必带
`--jury`/`--bench`/`--override`；缺陷×当跳 blocked＝例外；J 路出**综合裁决书**〔胜方+多数理由+各席
confidence+twin 终裁；落 jury JSON/journal/decisions/TUI〕+ **E6 twin 终裁**〔平票或 inconclusive 过半
自动收敛到确定选项、不给平局；不可达则**保守回退**挂起+出声〕）／Tier A（失败标签阈值自动立案）／
Tier B（bencher harness 套件）。**全量见 `references/quality-and-improvement.md`**（判决包细则见技能
`references/verdict-package.md`）。

## 6. 基础设施改进循环

`<infra_suggestion>`/inspector 发现 → `approve`（判决包）→ **backlog**（`state/backlog.json`）→
devops FIFO 消化（组包）→ 交付→机械验收→（带 M 的）**A/B 验证**闭环（改善→销账／未改善→自动纠正
提案／不可判→挂起）。**E0–E7 自改进体系**已收官并入；全量与设计纪律见
`references/quality-and-improvement.md`。

## 7. devops 作业模式

devops 承担基建维护（orch/guards/技能部署/验收脚本注册/pod 健康巡检）：**worktree 隔离实施** → 全量
测试 → 合并；技能改动**只改真本**（`orchestration/skills/`）后 `pysci-dev sync` 部署。逐条见 devops
技能与 `pods/devops/.qoder/rules/charter.md`。

## 8. MCP 归属

组员 MCP **下放 pod**（各 pod `.qoder/mcp.json` 自配），**用户级最终清零**；自配权不下放；组长零末端
MCP（用户裁决）。

## 9. 强制层

pod-guard（PreToolUse 写入口，按 §2 矩阵两层执法；**不删除成员自建技能**）／delivery-gate（Stop
hook：§3.4 格式强制 + 自维护层预算硬闸四条）／harness-budget（pre-commit 真本侧 ≤8192B 硬拦，
`pysci-dev doctor` 三档巡检）／yolo 下 hooks 照常执行／**诚实条款**（guard 只拦写工具，Bash 可绕 → 目标是把失效从「失误」升级为
「刻意违规」，由 transcript 审计发现）／**恢复面三闸**（WIP-salvage/sid 在途闸/CHANNEL-STOP）与
**交集成因停链**（`merge_dirty_face`）。**全量见
`references/guards.md`**；细则见技能 `references/tier-a-tags.md`。

## 10. 端到端工作流

用户投递大任务（TUI）→ 组长 `taskbooks list` → 拆解建 plan（`plan new`/`plan rework`）→ 逐环节
`plan advance`（任务书延迟书写；self 环节 `--evidence`）→ 交接决策（继续/amend/drop）→ 全 plan 完成 →
组装全流程报告 → TUI 验收（accept／reject→返工流）。全程事件驱动唤醒（§4.5）+ 路由卡片收尾（§4.2）。

## 11. CLI 调用范式（全员统一）

首选**裸命令**（PATH 含 `.venv/Scripts`）：`pysci-orch`/`pysci-dev`/`pysci`；不存在时 fallback
`uv run --no-sync pysci-X`（**裸名只试一次，禁重试循环**）；工具内部程序化子进程统一 `uv run
--no-sync`；改 `pyproject.toml`（含 console script）后必先 `pysci-dev venv-sync`。细则 `basic.md` §3。

## 12. 阶段计划（历史 → 归档）

Phase 0–4、E0–E7、M1–M7 **均已完成**；逐条阶段记述**移档**
（`references/history-and-risks.md` + 五归档 + git 历史）。**阶段完成须当轮登记本文**；体系常态运行。

## 13. 裁决记录

用户/组长裁决逐条留档（护栏级，**修改须用户授权**）——**全量表移入 `references/decisions.md`**（唯一
权威仍是本文件；回灌工具 `bench/backfill_preference_journal.py` 以该文件为源）。**裁决语义一字未改**。

## 14. 已知代价与风险

要点：用户注意力负担（并行分支多时）、返工流程刚性成本、`kind:self` 依赖组长如实提交 evidence、并发
写者互踩（§9 在途闸兜底）。**API Key 永不入库**（只能用户本人提供；`pysci-dev keyscan` 哨兵）。详表与
降级手册见 `references/history-and-risks.md`。

## 15. 夜航模式（PSEP_2）

单开关 `data/pysci_settings.toml [night] enabled`（读侧 fail-open、热切换）。两层：**政策注入**
（SessionStart + PostCompact 注入政策全文；用户在场时该注入是**授权书**——事后追溯责任在用户、决策权等同
用户、限时预授权）+ **判据拦截**（`AskUserQuestion` 拒绝并转 twin 终裁；mirror-pending 非空触发夜航
PENDING-WORK）；配套 **twin 仲裁盲测追溯板块**。细则见技能 `references/patrol-watchdog.md` §5。

## 16. 维护条款

- **护栏级条款**（红线/审批权/角色边界/门禁策略；即 §2 矩阵、§4.3、§5.3、§9、§13 中标「用户裁决」
  者）修改**须用户明确授权**（组长转呈）；其余条款由组长会同 devops 修订并向用户报告。
- **阶段完成当轮登记**；**ADR 惯例**：架构性决策落 `state/` 台账并在 §13 登记裁决。
- **预算纪律**：常驻注入档/按需档/技能 SKILL.md 各 ≤8KB（`basic.md` §1，§9 机械执法）；技能改动只改真本
  后 `pysci-dev sync`。

## 附录 A：降级手动操作手册（orch 故障时）

照 `references/history-and-risks.md` 附录 A 的原生 exe 直调模板手动派发（**勿用 .cmd 包装器**），台账
手工补记，尽快 adhoc 派 devops 修复。
